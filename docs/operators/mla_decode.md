# Latent MLA decode attention and split-KV merge

Scope: source-derived isolated operator for the **non-HIP**, **`IS_MLA=True`**,
**BF16-cache** grouped decode launch with an **explicit split count**.
Implementation id: `triton_decode_grouped_mla`.

This is a mechanism plan, not an accuracy claim and not a capture match. The
sources are the current checkout files whose SHA256 identities match
the planner's retained source identity; no historical capture image supplied
a parameter. Heldout case geometry is used only as test geometry; no timing is
read from it. No MLA decode capture row was found in the read-only
`data/verification`/`data/observations` paths, so this plan is source-derived
and makes no capture-dispatch claim.

## 1. Source identity

| File | SHA256 | Lines used |
|---|---|---|
| `/Users/maoxx241/code/vllm-v0.25.1/vllm/v1/attention/backends/mla/triton_mla.py` | `5e9aa13eca7ce718ce0648f11454bb0e0e98e9f5da40ee22bd8d0b05f4ba1d03` | 35-47 auto split (not used); 82-89 supported dtypes; 95-103 page-size contract; 166-171 logit cap rejected; 199-208 wrapper q concat and `o`/`lse` zero-init; 210-216 split selection; 224-237 scratch and cache alias; 241-256 grouped call with `is_mla=True` |
| `/Users/maoxx241/code/vllm-v0.25.1/vllm/v1/attention/ops/triton_decode_attention.py` | `b9823c65eba22d76abf64dad7d7d16a950f67471c86d6cb5b3e983aaf14381ac` | 59-65 page stride; 279-467 grouped stage 1 (329-355 Q/split bounds, 361-441 K/PE QK + latent PV, 443-467 partial/LSE stores); 470-572 MLA block dispatch; 575-639 stage 2 merge; 642-681 stage-2 launch; 719-753 grouped wrapper; 756-817 grouped dispatch |

The two source files are the current checkout, not a historical capture image.
Their byte counts and hashes are also recorded in the module's `SOURCE_FILES`
and in the campaign manifest.

## 2. Selected dispatch

`make_op` validates and retains every branch flag in
`OpSpec.metadata["dispatch"]`. The campaign's four `expected_dispatch` keys are
a subset; the remaining flags make the source block selection auditable.

| Flag | Value | Source reasoning |
|---|---|---|
| `implementation` | `triton_decode_grouped_mla` | grouped MLA branch |
| `is_mla` | `true` | `v = tl.trans(k)` at line 432; no `V_Buffer` read |
| `is_hip` | `false` | non-HIP `BLOCK=32` (line 506), `BLOCK_DMODEL=512`, `BLOCK_DPE=64` (lines 491-497) |
| `logit_cap` | `0.0` | `TritonMLAImpl` rejects a truthy soft cap (lines 166-171); tanh branch off |
| `dtype`, `cache_dtype` | `bf16` | selected BF16 query and BF16 cache; FP8 dequant branch not selected |
| `key_heads` | `1` | latent MLA cache has one 576-wide KV row per token |
| `grouped`, `kv_group_num` | `true`, `H` | `kv_group_num = q_heads // 1 > 1` selects the grouped kernel |
| `num_kv_splits` | campaign value | explicit split count; `_compute_num_kv_splits`/`_MIN_WORK_PER_SPLIT` are not used |
| `page_size` | `16` | `TritonMLABackend` supports multiples of 16 (lines 95-103); kernel page stride is lines 59-65 |
| stage 1 kernel | `_fwd_grouped_kernel_stage1` | lines 279-467 |
| stage 2 kernel | `_fwd_kernel_stage2` | lines 575-639 |
| `block_h`, `block_n` | `16`, `32` | lines 506, 513 |
| `block_dmodel`, `block_dpe`, `block_dv` | `512`, `64`, `512` for the 576-row case; `256`, `32`, `256` for 288 | lines 490-504 |
| `num_warps`, `num_stages`, `stage2_*` | `4`, `2`, `4`, `2` | stage 1 lines 521-532, 566-567; stage 2 lines 678-679 |
| `qk_parts` | `latent_plus_pe` | two `tl.dot` calls, lines 394 and 405 |
| `pv_source` | `latent_k_alias` | `v = tl.trans(k)` line 432 |
| `v_buffer_reads` | `0` | `V_Buffer` is only used in the `not IS_MLA` branch |
| `softmax` | `online_lse` | lines 434-441 and 607-635 |
| `empty_splits` | `stage1_launches_stage2_skips` | grid includes every split; stage 2 guards with `split_kv_end > split_kv_start` |
| `kernel_launches` | `("stage1_grouped_decode", "stage2_split_merge")` | two unconditional wrapper calls (lines 736-753) |
| `source_kernel_launches` | `2` | provenance flag only; it never expands execution |
| `stage1_program` / `stage2_program` | `per_split_head_cta` / `per_head_merge_cta` | source grids `(B, G, S)` / `(B, H)` |
| `tile_order` | `qk_softmax_pv` | source loop body lines 369-441 |
| `split_reduction_owner` | `batch_head` | stage-2 grid `(batch, head_num)` line 663 |

`OpSpec` shapes are semantic/local `(B, H, Lk)`, padded `(B, G*16, Lk)` (head
lane padding visible), and launch `(B, G, S)` for stage 1. The stage-2 child
uses launch shape `(B, H)`.

Unsupported neighbours raise `ValueError` and are never substituted: HIP block
dispatch, `IS_MLA=False`/separate-V, FP8 cache dequant, non-zero logit cap,
`page_size` not a multiple of 16, `key_heads != 1`, `heads < 2` (the non-grouped
MHA path), and head dimensions other than the two explicit special cases.

## 3. Launch boundary and host submissions

The operator boundary starts at `decode_attention_fwd` and covers two
unconditional native launches:

1. `_decode_grouped_att_m_fwd`: stage-1 grid `(B, G, S)`.
2. `_decode_softmax_reducev_fwd`: stage-2 grid `(B, H)`.

`kernels(op)` returns two `OpSpec` children with local ids `stage1_grouped_decode`
and `stage2_split_merge`; the merge child depends on the stage-1 child. The
coordinator binds `PlannedRecipe(module.build, module.kernels)`. Both
`operator_costs` and `Execution` expand the parent before pricing, so eager
execution submits exactly **two** host launches and graph capture submits the
captured group once.

`source_kernel_launches = 2` is metadata. A recipe created without `kernels`
still executes/prices one physical op; the test suite includes that
counterexample explicitly. `build` therefore requires an explicit
`kernel_phase` on the child and raises on the unexpanded parent.

Outside the boundary, but documented in `TritonMLAImpl.forward_mqa`:

* optional `q = torch.cat((q_nope, q_pe), -1)` (lines 199-201);
* `o` and `lse` zero-initialisation (lines 205-208);
* the shared-workspace reservation and `torch.empty` fallback (lines 218-232).

The wrapper's auto split selector (lines 210-216) is also outside the selected
policy: the case supplies `num_kv_splits` explicitly to the kernel entry point.

## 4. CTA ownership and lazy recurrence

Stage 1 launches `B * G * S` source CTAs. Each CTA is one `KernelProgram`
generated lazily by the per-physical-owner stream for
`index % core_count`; a stream is a `PipelineProgram` with one resident CTA
(`_CTA_STREAM_SLOTS = 1`). A stream visits only its own CTA indices, so an idle
unit never supplies a pinned CTA and the number of live CTA programs is bounded
by the declared physical owners. The CTA count and stream assignment are
derived from the source grid, not from a batch-size fit.

Each stage-1 CTA:

1. loads its `Q` latent/PE rows and `B_Seqlen` (`q_load`);
2. for every `BLOCK_N` key tile in its split span, in source order:
   `k_load` (page id + latent/PE K row; `k_scale`/`v_scale` on the first tile),
   `qk` (two matrix dots plus scale/mask), `softmax` (row max, online
   re-scale/exp and sum), `pv` (latent K transposed as V and the accumulator
   update). Both the softmax and PV state chains are stage tails of the same
   CTA, and the tile pipeline uses one slot so `qk(i+1)` cannot start before
   `pv(i)` completes. The state (`acc`, `e_sum`, `e_max`) never leaves the CTA;
3. after the tile loop, stores only its valid head lanes' FP32 partial output
   and FP32 LSE (`scratch_store`), and only for a non-empty split.

The stage-1 CTA pins `cube`, `vector` (mapped through the shared `unit_group`
partition) and declared `mte2`/`mte3` to one physical owner. `hbm` is global
memory and is deliberately not pinned. Stage 2 emits `B * H` CTAs; each merges
only its own batch/head's non-empty split slots and writes BF16 `o`/`lse`.
Its reduction chain is likewise a per-CTA stage tail, so no cross-head
aggregate reduction exists.

The pipeline's first tile is also the empty-split gate. An empty split keeps
the Q/sequence load but has no tile stages and no scratch store, exactly like
the source's `if split_kv_end > split_kv_start:` branch. Stage 2 iterates only
non-empty slots; the skipped loop iterations perform no K/partial load and no
merge arithmetic.

Token count is never enumerated. The key-tile loop is a `PipelineProgram`
whose slot window bounds live stages; a million-token context creates eight
stage-1 CTA generators and one bounded tile pipeline per CTA, not 31k `Stage`
objects up front.

## 5. Geometry, split, page and tail rules

For request `b` with sequence length `L_b`, explicit splits `S`, page size `P`,
block width `Bn = 32`, block height `Bh = 16`:

```
C_b = ceil(L_b / S)                 # source kv_len_per_split
N_b = ceil(L_b / C_b)               # non-empty split slots
span_s = [s*C_b, min((s+1)*C_b, L_b)) for s in 0..S-1
T_b = sum over non-empty spans of ceil((end-start)/Bn)
G   = ceil(H / min(Bh, H))          # head-block programs (key_heads = 1)
```

A tile tail loads only lanes with `offs_n < split_kv_end`, so no key or
page-id lane beyond the split end is charged. Page ids are derived from the
global token position (`offs_n // P`) and may cross page boundaries inside a
split, including a split boundary in the middle of one page. The QK dot is the
sum of a 512-column latent dot and a 64-column PE dot for the 576-row case
(256 + 32 for 288). The PV dot consumes `tl.trans(k)` of the already-loaded
latent tile; there is no second 512-column V read.

## 6. Physical work ledger per kernel

Symbols: `Lk = latent + rope`, `Dv = BLOCK_DV`, `Dm = BLOCK_DMODEL`,
`Dpe = BLOCK_DPE`. All bytes are exact element counts; no cache-line rounding.
`hbm = reads + writes`, `mte2 = reads`, `mte3 = writes` per kernel.

### Stage 1 per request

```
reads  = S*H*Lk*2                         # Q loaded by every split, even empty
       + G*S*4                            # B_Seqlen per launched program
       + G*N_b*8                          # k_scale/v_scale, non-empty branch
       + G*L_b*4                          # page-id lanes, one per valid key
       + G*L_b*(Dm+Dpe)*2                 # one 576-row K read per key
writes = N_b*H*(Dv*4 + 4)                 # FP32 partials + FP32 LSE
vector = G*(T_b*(7*Bh*Bn + Bh*Dv + 5*Bh)
            + N_b*(Bh*Dv + 2*Bh))
flops  = G*T_b*2*Bh*Bn*(Dm+Dpe+Dv)        # latent QK + PE QK + latent PV
```

The vector-per-tile terms are: scale, tail mask, row max, `p`
subtract/exp/cast, `sum(p)` = `7*Bh*Bn`; `acc *= re_scale` = `Bh*Dv`;
max/re-scale/e_sum update = `5*Bh`. The epilogue is `acc/e_sum` plus the LSE
add/log. These totals are cross-checked against the sum of the per-CTA stages
by the execution engine, not used as a single simultaneous task.

### Stage 2 per request

```
reads  = H*4                              # B_Seqlen per head program
       + H*N_b*(Dv*4 + 4)                 # only non-empty partial slots
writes = H*(Dv*2 + 2)                     # BF16 o + BF16 lse
vector = H*(N_b*(3*Dv + 7) + (2*Dv + 3))
```

The stage-2 split merge terms are: max, old-scale subtract/exp, `acc` scale,
new-scale subtract/exp, `acc` multiply/add, `e_sum` multiply/add = `3*Dv + 7`.
The final term is result divide, LSE add/log, and the `Dv + 1` FP32-to-BF16
store conversions.

K is re-read once per head-block program (`G`), and Q is re-read once per split
program. That is repeated source traffic, not a double-load of K and V inside a
program. Partial scratch is written once and read once per non-empty slot.

### Buffers

* Global scratch: `(B, H, S, Dv+1)` FP32, reserved `4*B*H*S*(Dv+1)` bytes.
  Only non-empty slots are written/read; the reservation is recorded but is
  not traffic. The wrapper's workspace manager may pre-reserve a
  max-model-length buffer; that wrapper-level reservation is outside this
  per-call operator boundary and is not charged.
* Outputs: `o` `(B, H, Dv)` BF16 and `lse` `(B, H)` BF16.
* No on-chip `Allocation` is asserted. The source's shared-memory/register
  footprint is compiler-dependent (tile operands plus Triton `num_stages`
  double buffering), and the framework's `Geometry.buffers` pools are on-chip
  capacities without a source-declared generic shared-memory role. The
  per-CTA recurrence state is program-resident and stays on the CTA's pinned
  owner; `Geometry.buffer_units`/`Allocation.units` are therefore not
  exercised by this plan. `verify_operator` consequently sees
  `peak_buffers == {}` and `remaining_buffers == {}`.

## 7. Independent oracles and counterexamples

`tests/operators/test_mla_decode.py` re-derives the ledger, split merge,
per-CTA traffic and placement independently of the planner's private formulas.
59 focused tests pass.

* Dense NumPy oracle:
  `softmax((q_nope @ k_latent.T + q_pe @ k_pe.T) * scale) @ k_latent`,
  compared with stage-1 partials and a stage-2 online-LSE merge. For 17 keys
  with 12 latent/4 PE columns and splits `1, 2, 3, 5, 8, 17`, max `o` and LSE
  errors are `4.5e-16`. At the selected 512/64 geometry with 3 keys and 7
  splits, max `o` error is `3.1e-15` and max LSE error `7.1e-15`; the four
  empty slots are exercised.
* Alias counterexample: the dense result uses `p @ k_latent` (the latent tile
  already loaded as K); an independently random `p @ V` is not close. This
  distinguishes the `IS_MLA` path from the `not IS_MLA` branch.
* Hand-computed per-kernel ledgers: for `L=1,S=1`, stage 1 is
  `1,114,112` BF16 FLOPs, `52,432` HBM bytes, `19,600` read bytes,
  `32,832` write bytes, `20,080` vector ops; stage 2 is `49,312` HBM bytes,
  `32,896` read bytes, `16,416` write bytes, `41,120` vector ops. Their sum
  reproduces the composite ledger (`101,744` HBM, `52,496` read, `49,248`
  write, `61,200` vector).
* Empty splits: `L=1,S=1` and `L=1,S=8` have identical stage-1 MMA, K/PV,
  scratch and vector work; `S=8` adds seven launched `Q` row blocks and seven
  `B_Seqlen` loads, but no `k_scale`/`v_scale`, K traffic or scratch stores.
  A trace with `H=2,S=8` shows split 0 executing `q_load, k_load, qk, softmax,
  pv, scratch_store` and the other seven CTAs executing only `q_load`.
* Tail counterexample: `L=17,S=2` restarts tiles at the split boundary, so it
  executes 2 tiles (64 MMA columns) while `ceil(17/32)=1`; the extra 32 columns
  are masked, not free.
* Per-CTA ledger oracle: at `L=17,S=2` the two CTAs read exactly
  `18,436 + 10,412 + 32,832` and `18,436 + 9,256 + 32,832` bytes, execute
  `1,114,112` BF16 FLOPs each and `11,856` tile vector ops plus `8,224`
  epilogue ops. The K term is `valid_n * 576 * 2` with no separate V term.
* Ownership counterexample: a single stage-1 CTA with 32 key tiles has
  `0.03565 ms` of matrix work at one fixed per-SM rate. Its total is
  `0.03695 ms` on a 1-owner device and `0.03576 ms` on a 32-owner device (the
  small difference is shared-HBM service, not extra matrix service); it does
  **not** become 32x faster. The pinned placement is one `cube` unit, so the 31
  idle units cannot supply service. With 32 independent CTAs on 32 owners, the
  stage-1 duration is legitimately `1/32` of the one-owner duration.
* Recurrence/ownership trace: at `L=129,S=2` on two owners, every stage-1 task
  is placed on its owner's `cube` unit and vector subblock, `q_load` precedes
  the first `k_load`, `qk(i)` ends before `softmax(i)` starts, `softmax(i)`
  ends before `pv(i)` starts, and `pv(i)` ends before `qk(i+1)` starts.
* Split-reduction trace: at `L=3,S=8,H=4`, each stage-2 CTA reads exactly three
  non-empty partial slots (`512*4+4` bytes each), merges them serially, writes
  one BF16 output row, and is pinned to one owner's vector subblock. No CTA
  reads a skipped empty slot.
* Laziness: a million-token context creates 8 stage-1 CTA generators (one per
  split) and a 3,907-tile `PipelineProgram`, not an enumerated tile list. A
  32,768-token execution completes with `live_tasks == 0` and
  `peak_live_tasks < 100`. A 4,096-split grid is represented by 64 one-owner
  streams of 64 CTAs each, not 4,096 eagerly built CTA programs.
* Launch-boundary counterexamples: eager `Execution` with
  `PlannedRecipe(build, kernels)` charges two host submissions; the same
  recipe with `replay_scope="full"` charges one captured submission. A
  `PlannedRecipe(build_without_kernels)` charges one submission and prices one
  op even though the metadata says `source_kernel_launches = 2`.
* Byte/work oracle for all five fixed cases is checked against the actual
  public isolated execution path for both unified-memory and split-copy
  controls. The executed serviced work equals each kernel's declared work
  within the shared engine's tolerance; no buffers remain live.

## 8. Executed checks

Commands (using the provisioned shared interpreter and this worktree on
`PYTHONPATH`):

```sh
PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  -m pytest tests/operators/test_mla_decode.py -q
# 59 passed

PYTHONPATH=. /Users/maoxx241/code/sim-rewrite-20261027/.venv/bin/python \
  tools/run_operator_cases.py mindie_sim.prediction.operators.mla_decode \
  tests/fixtures/operator_cases/mla_decode.json \
  -o data/verification/operators/mla_decode.json
# {"status": "passed", "cases": 10, "output": "data/verification/operators/mla_decode.json"}
```

The campaign runner output contains five cases (`mla_decode-0`..`2`
development, `mla_decode-3`..`4` heldout_geometry), each with unified-memory
and split-copy controls: 10 structural rows, all finite, work-conserving,
identity-invariant and dispatch-matching. Each row exposes two kernel children.
This is structural verification, not accuracy acceptance and not a
capture-dispatch claim.

## 9. Ledger for the fixed cases

Source-derived totals from the current plan (no measurement inputs):

| Case | Split | B | H | G | S | non-empty | tiles/request | stage1 FLOPs | stage1 HBM | stage2 HBM | total HBM | total read | total write | total vector |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | development | 1 | 16 | 1 | 1 | 1 | 1 | 1,114,112 | 52,432 | 49,312 | 101,744 | 52,496 | 49,248 | 61,200 |
| 1 | development | 1 | 16 | 1 | 2 | 2 | 5 | 5,570,560 | 251,676 | 82,144 | 333,820 | 251,740 | 82,080 | 141,536 |
| 2 | development | 2 | 32 | 2 | 4 | 4,4 | 16,4 | 44,564,480 | 2,043,464 | 591,232 | 2,634,696 | 2,043,720 | 590,976 | 1,066,560 |
| 3 | heldout_geometry | 2 | 8 | 1 | 8 | 8,8 | 8,23 | 34,537,472 | 1,039,168 | 279,136 | 1,318,304 | 1,039,232 | 279,072 | 713,056 |
| 4 | heldout_geometry | 1 | 24 | 2 | 4 | 4 | 35 | 77,987,840 | 2,677,480 | 221,712 | 2,899,192 | 2,677,576 | 221,616 | 1,068,488 |

`stage2` has zero matrix FLOPs; its reads/writes are the `H*4` sequence loads,
non-empty partial/LSE loads and BF16 output stores described above. The
`tiles/request` column is `T_b` before the stage-1 head-block multiplier `G`.
Heldout geometry entries are test cases, not calibration data.

## 10. Limitations and unresolved parameters

* The selected source is the raw kernel entry, not the wrapper's auto split
  selection. `VLLM_BATCH_INVARIANT` is not selected; the case's explicit `S` is
  modelled.
* Only `(latent, rope) = (512, 64)` and `(256, 32)` are selected. The generic
  `next_power_of_2(Lv)` fallback can overlap latent and PE columns when `Lv` is
  not a power of two and needs a separate address review; it is not silently
  substituted.
* FP8 cache dequantisation, HIP block dispatch, separate-V GQA, non-zero
  `logit_cap`, `page_size % 16 != 0`, and `key_heads != 1` are unsupported
  selections, not fallbacks.
* Vector service treats one exp/log/select/cast as one vector operation. There
  is no SFU-specific rate and no instruction-issue or memory-response resource
  in the shared engine; those rates are unsupported rather than fitted.
  Integer address arithmetic and the page-id `int64` conversion are not charged
  as vector work. The `qk += tl.dot(qpe, kpe)` and `acc += tl.dot(p, v)`
  accumulator adds are treated as part of the tensor-core GEMM accumulation
  (`2*M*N*K`), not as separate vector adds. If a compiler materializes them
  separately, the omitted vector work is `Bh*Bn` (PE) and `Bh*Dv` (PV) per
  tile; this is a compiler-lowering boundary, not a fitted factor.
* Page-id loads are charged as one 4-byte lane per valid key. Duplicate page ids
  within a page, cache-line granularity, `.ca` cross-tile reuse and cross-head
  block reuse are not credited.
* The BF16 `k_scale`/`v_scale` loads are compile-time-dead in the source branch
  but are charged conservatively at source level (8 bytes per non-empty
  program). This is a known small overcount, retained explicitly.
* `B_Seqlen` and `Req_to_tokens` entries are assumed 4-byte integers. The
  supplied kernels receive them as pointer tensors and do not declare their
  width in these files; the 4-byte assumption is stated rather than inferred
  from a measurement.
* Global scratch capacity is not checked. `Geometry` exposes only on-chip
  `buffers`; a global-memory pool would be needed for a capacity-checked
  scratch lease. The current `Allocation` API has no global-memory pool name in
  `RecipeContext.pool_names`, and the source does not declare one. Proposed
  interface: an optional global pool mapped through the same
  `context.allocate(name, pool, size, units)` path, with independent per-unit
  residency.
* On-chip Q/K/accumulator residency is not asserted because the source does
  not declare shared-memory sizes and Triton `num_stages` pipelining is
  compiler-dependent. `Allocation.units`, `Geometry.buffer_units` and
  `RecipeContext.pool_units` are therefore unused by this operator; the
  per-CTA recurrence state is carried by the program's stage tails and pinned
  through `Stage.placement`.
* Stage-1 key tiles execute with one pipeline slot so the source loop body
  stays ordered (`qk(i+1)` after `pv(i)`). Triton's software-pipelined load
  overlap (`num_stages=2`) is not separately modelled; the engine therefore
  serialises Q/K load service with the arithmetic tail. This is a conservative
  mechanism simplification, not an efficiency multiplier. A future
  `PipelineProgram` API that allows a next-tile load to prefetch while
  preventing next-tile arithmetic from overtaking the current tile's PV would
  be needed to model that overlap without borrowing idle matrix units.
* `num_stages`/`num_warps` are recorded as source block parameters; no
  additional wave, occupancy, instruction-issue or SFU factor is applied.
* `Geometry.compute_units` sets the number of CTA streams and the physical
  owner count. The explicit split count and source grid determine the declared
  work; no occupancy efficiency is applied.

## 11. Integration

The coordinator binds the module explicitly; nothing is auto-discovered:

```python
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators import mla_decode

recipes = RecipeBook(recipes={
    "triton_decode_grouped_mla": PlannedRecipe(
        mla_decode.build, mla_decode.kernels
    ),
})
```

`kernels` is mandatory for this operator: it creates the two source launches.
Without it the parent has no `kernel_phase` and `build` raises a deliberate
error. To wire the real graph, provide Q already concatenated to
`(B, H, latent+rope)` BF16, the paged cache view with last dimension
`latent+rope`, the per-request sequence lengths, page size, block table, and
the explicit split count. Keep the wrapper's q-cat and `o`/`lse` zero-init as
separate preceding materialisations unless the coordinator chooses to widen the
boundary. The coordinator owns the registry, graph wiring and acceptance.

## 12. Delivered files

SHA256 of the delivered machine-readable/code files:

| File | SHA256 |
|---|---|
| `mindie_sim/prediction/operators/mla_decode.py` | `51d6b0adc140ab1f78da4f300213f16bd82acf9e8f9a9428e1a6f37619f4a180` |
| `tests/operators/test_mla_decode.py` | `d0a424a4c1f693b24819dde7fab0fa2672c8ebf6beeb4979f1eda227c38d2ca7` |

`data/verification/operators/mla_decode.json` is the unmodified runner's
structural-control output. Its dictionary key order and last-bit service
rounding are not byte-stable between interpreter runs, so no file hash is
pinned; the status, dispatch, work-conservation and identity checks are stable.

The campaign case file and shared core are not modified. The coordinator owns
Git metadata, registry binding and integrated acceptance.
