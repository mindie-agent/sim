# Staged analytic GQA decode

`StagedGQADecode` is an explicit decode recipe in
`mindie_sim/prediction/attention_decode.py`.  It consumes the frontend
metadata contract below and emits a finite program on the ordinary
`Engine`, using `Stage`, `KernelProgram` and `PipelineProgram`.  It is a
practical phase/tile model, not a reconstruction of a vendor kernel,
compiler schedule or CANN dispatch record.

## Selected domain

The ordinary frontend selects this recipe through
`runtime.kernels.recipes.flash_attention_decode`. Both target `decode` and
incremental `draft` phases supply the contract below; prefill and draft cache
preparation keep their separate recipes:

```python
metadata["gqa_decode"] = {
    "requests": ((q_len, context), ...),
    "query_heads": H,
    "kv_heads": Hkv,
    "head_dim": D,
    "query_dtype": "bf16",
    "kv_dtype": "bf16",
    "causal": True,
    "window": 0,
    "cp": 1,
}
```

`q_len` and `context` are nonnegative integers, `H` is divisible by `Hkv`,
and one query/KV group has `G = H // Hkv` query heads.  Speculative decode
`q_len > 1` is allowed.  The current domain is causal, no sliding window and
no context parallel.  Other `window`, `cp`, causal, phase or head-divisibility
values raise `ModelingError` instead of silently selecting another path.
`bf16`, `fp16` and `fp32` payload dtypes are accepted; the selected frontend
contract uses bf16.

The query tile is the flattened row space
`(token, query_head_in_group)`.  A tile can therefore contain several real
tokens for a small GQA group or only part of one token's group when
`q_len * G > query_tile`.  Every tile keeps its real row tail.  The active KV
horizon also uses the real causal tail `context + token_last + 1`, never a
paged-cache allocation or a padded bucket.

## Options and partitions

`StagedGQADecode(key_tile=128, query_tile=64, kv_splits=1, slots=2)` declares:

- `key_tile`: K/V row tile.
- `query_tile`: flattened query-row tile.
- `kv_splits`: requested split count.  The effective count is
  `min(kv_splits, ceil((context + q_len) / key_tile))` per request.
- `slots`: bounded key/V pipeline window.

The real key tiles of a request are partitioned into contiguous, nonempty
ranges.  A split that has no valid key tile for a particular query tile is
not launched for that output.  Thus a requested split count cannot create an
artificial empty partial or GM round trip.  All contributed splits for one
output are the first contiguous range of real split ranges; the final output
is written exactly once.

## Owner tasks

The physical task tuple is `(request, kv_head, query_tile, kv_split)`.  Tasks
are assigned round-robin to `context.core_count` physical owners.  Each owner
receives a lazy `PipelineProgram` with one resident task at a time, so an owner
with several tasks serializes them.  All arithmetic, vector and declared copy
roles for a task are placed on that owner with `context.owned(...)`; HBM is
global.  A task pinned to one owner cannot borrow another owner's idle matrix
or vector units.

Each task executes:

1. one resident query load (`q_load`), including the Q HBM read and, when the
   hardware exposes `mte1`, the Q operand movement;
2. a bounded pipeline of real K/V tiles:
   `k_load -> qk -> softmax -> {v_load, probability_operand} -> pv -> online_update`;
3. `online_update` explicitly depends on the previous tile's
   `online_update`, preserving the running max/sum and output accumulator;
4. a partial store, or a direct final output store when only one split
   contributes.

The K/V path is owned by the AIC read role (`mte2_aic` when the hardware
declares split copy roles, otherwise `mte2`).  Cube work is placed on the
matrix owner; softmax, online reduction and epilogue work are placed on the
vector owner. Split partials are read by the AIV pipe. `mte1` includes both
Q/K/V operands and the probability operand consumed by PV, when that role is
declared.
No on-chip movement is added to HBM.

If more than one split contributes to an output, every producer stores an
FP32 partial accumulator plus FP32 max and sum scalars.  A separate merge
stream starts each merge only after a per-output split barrier observes every
partial producer.  The merge reads the partial output, max and sum, performs
the split reduction and writes the final output once.  When only one split
contributes, no partial is materialized and the producer writes the final
output directly.

## Work formulas

For one output query tile, let `pairs` be the valid causal query/key pairs
over its flattened query rows and let `row_tiles` be the sum, over those rows,
of the number of key tiles intersecting the row's causal horizon.  For a
request with one query head and no window,
`causal_pairs(q, c) = q * (c + 1) + q * (q - 1) // 2`.

- QK and PV FLOPs: `4 * D * pairs`.  Summed over all GQA rows this is the
  required `4 * causal_pairs * H * D`; KV is not multiplied by `H`.
- K/V HBM: `horizon * 2 * D * kv_dtype_bytes` per contributing query tile and
  KV head.  Each K/V tile is loaded once for that tile, regardless of `G`.
- Q HBM: `contributing_splits * rows * D * query_dtype_bytes` per query tile
  and KV head.  Q is reloaded by each contributing split, not by every query
  head row separately.
- Partial HBM, when `contributing_splits > 1`:
  `contributing_splits * rows * (4 * D + 8)` bytes written and the same amount
  read by the merge.
- Final output HBM: `rows * D * query_dtype_bytes`, written once per output.
- Vector work is explicit logical arithmetic, not a measured multiplier:
  `7 * pairs` for score scale/mask, row max, subtract, exp, sum and
  probability conversion; `(3 * D + 10) * row_tiles` for tile-local sum and
  online max/rescale updates (two numerator scalings and one add per output
  element); plus `rows * (D + 1)` for a direct final
  normalize/cast, or
  `contributing_splits * rows * (3 * D + 9) + rows * (D + 1)` for the split
  merge.  These are declared elementary operations with the same units as
  other vector work.

The OP's semantic aggregate fields remain the frontend's analytical aggregate
work.  The executable recipe declares the actual Q duplication, split partial
round trips, merge reads/writes and bounded tile work.  `executed_service`
therefore includes that extra physical traffic; it is intentionally not
rewritten into the semantic aggregate.  No source boundary is claimed as
`kernel`.

## On-chip lifetimes and limitations

When the declared hardware exposes `l1`, `l0a`, `l0b`, `l0c` and `ub`, a task
retains Q in L1/L0A, an online state in UB, score/probability tiles in
L0C/UB, K/V operand tiles in L1/L0B, and a per-tile PV accumulator in L0C.
Buffers are scoped per owner and pipeline slot, and are released by the
consuming stage or final store.  A declared buffer smaller than a required
tile raises an explicit capacity error; it is not silently dropped.  Live
task and allocation counts are bounded by the owner streams and pipeline
window, not by context length.

Deliberate limitations:

- Scalar control, loop, barrier and synchronization latency is not modeled.
- Page-table/address generation, cache replacement, bank conflicts and
  instruction issue are not reconstructed.
- The key/V pipeline moves full valid tail rows analytically; it does not
  claim a vendor's exact overlap, double-buffer assignment or MTE1/MTE2
  issue order.
- The split merge streams one partial at a time into a bounded partial buffer
  and running accumulator. It is not a measured reduction kernel.
- No latency floor, fitted duration, measured duration, hardware-name branch,
  model-name branch or native-source-equivalence claim is used.

## Launched blocks do not establish active owner count

The retained Qwen3.8 decode capture reports 32 AIC and 64 AIV launched blocks,
with 24 query heads, four KV heads, width 256 and query lengths one or four.
That block count cannot by itself establish eight KV splits.

The inspected `ops-transformer` revision
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`, in
`attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp`,
provides a concrete counterexample:

- `CheckPFAMergeSupport` (lines 137-147) permits merging GQA query rows when
  `q_len * group < 64` under the listed layout/feature conditions.
- `ComputeLoopParams` (lines 257-273) then iterates KV heads, rather than all
  query heads.
- `ComputeSplitNBSeq` (lines 550-650) assigns complete outer query tiles to
  owners; it does not partition their KV loops.
- `ApplySplitStrategy` (lines 1233-1247) enables `SplitS2` only when
  `isIFAFlag && !pfaMergeFlag && CheckFlashDecode`.
- `SetBlockDim` (lines 1487-1494) still launches the platform block geometry.

Thus the compatible merged branch has four matrix owners for a single
request, despite the launch of 32 blocks. The declaration `kv_splits=1`
represents this mechanism. The historical binary/tiling key is not retained,
so this remains a source-compatible declaration, not exact dispatch proof.
The phase model uses an explicit 128-row key tile; it does not claim to
reconstruct the source's 256-row macro tile and inner operand tiling for
query length four.

Tests in `tests/test_attention_decode.py` independently derive causal FLOPs,
ragged/zero-context HBM ledgers with split partials, the GQA KV-reuse ratio,
input-readiness ordering, bounded live state, owner pinning under contention,
split-barrier ordering and multiple query tiles.  Synthetic rates establish
structure and trends, not hardware accuracy.

## Paged tensor declarations

`runtime.attention_policy.kv_storage = "paged"` explicitly declares the TND
query/output and paged K/V interface for ordinary GQA. The frontend derives
each cache row's packed width from local KV heads times head dimension, and
its page length from `runtime.kv_block_size`. A fused batch retains its actual
request count; separate calls each declare one request. Query lengths 1, 2
and 4 remain geometry, without inferring speculative-step roles.

The declaration adds comparison metadata only. It neither prices allocated
cache capacity as active work nor selects a different kernel program. Default
`"auto"` preserves fresh-prefill contiguous declarations. CP/PCP-sharded paged
interfaces remain undeclared until their local ABI is modeled. Cache capacity,
block-table contents, active context and captured binary identity cannot be
inferred from this partial tensor contract.
