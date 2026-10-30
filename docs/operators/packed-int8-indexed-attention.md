# Packed INT8 indexed attention: explicit analytic branch

`runtime.kernels.programs.indexed_attention = {"kind": "indexed_attention_packed_int8", "scratch_policy": "offchip"}` now runs through ordinary `target_graphs` and the common Engine. It replaces the semantic attention envelope only when explicitly selected. It does not change the BF16 source binding or hardware rates. `compact: false` retains the expanded phase control.

This is **a caller-supported layout adaptation, not an identified native `KvQuantSparseFlashAttentionVllm` kernel**. The retained source host rejects the captured no-RoPE geometry. Keeping the distinction visible is necessary to avoid turning an unsupported source match into apparent accuracy.

## Evidence and representation

The profiler receipts at `acceptance-20261030-glm-operators-67ca7673` retain:

| Capture | Q and output | INT8 K and V | Indices | Block table | Mean task time |
|---|---|---|---|---|---:|
| A3 ACL c32 | BF16 `[192,4,512]` | `[38820,128,1,528]` | INT32 `[192,1,2051]` | INT32 `[32,1025]` | 829.848 µs |
| A3 ACL c64 | BF16 `[384,4,512]` | `[38820,128,1,528]` | INT32 `[384,1,2051]` | INT32 `[64,1025]` | 1632.353 µs |

Each has two length arrays `[B]` and empty optional dequant tensors / softmax outputs. Q/B establishes six physical query rows per sequence. Array extents do not establish valid index values, per-query causal contexts, or scalar operator attributes. In particular, `2051` is observed index capacity; the scenarios retain the declared model top-k of 2048 and context 4096. They do not convert three extra positions into measured valid cache rows.

Pinned local caller: `vllm-ascend-v41-private`, revision `ae923905948f360f4b7ab06df52d0789a0c9316e`:

- `vllm_ascend/attention/utils.py:81–92`: packed row size is `latent + 2*rope + 4*(latent/128)`. Latent512 / rope0 therefore gives **512 INT8 values + four FP32 scales = 528 bytes**.
- `vllm_ascend/device/device_op.py:458–490`: `key=kv` and `value=kv` alias the same packed cache; TND query, PA_BSND cache, block1, per-tile quantization, tile128 and caller rope width are explicit.
- The local kernel is a different supported domain: `csrc/attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp:1282–1291,1372–1382` requires Q576, packed KV656 and rope64. The kernel reads scales at byte640, and pads a **656-byte GM row to 672 bytes in UB**. Its existence does not establish support for the captured 528-byte row.
- `op_kernel/kv_quant_sparse_flash_attention_service_vector_mla.h:652–851` provides the reference copy/conversion sequence; `855–940` divides selected row pairs between the paired AIVs, with a maximum32-row merge chunk. `200–220` allocates189952 bytes per AIV for preload3. The adaptation retains this allocation and source pair split; the no-RoPE dispatch remains unverified.

The evidence JSON records exact source and receipt hashes. No remote collection or raw CSV rescan was performed.

## Executed mechanism

The new module supplies a phase factory to the existing indexed-attention scheduler. It does not have another Engine or a separate matrix/attention timing formula.

1. Read each selected packed row **once**, plus the declared INT32 index / page-reference work. DataCopyPad's local 544-byte row pitch does not increase 528-byte GM payload or duplicate K/V traffic.
2. Account INT8→FP16→FP32 conversion, FP32 scale multiplication, FP32→BF16 conversion, scale broadcast and ND→NZ copy as typed vector element work. The five512-element passes and four8-lane scale broadcasts are work counts, not a fitted CPI.
3. Publish each AIV's expanded BF16 rows and its own512-byte valid-size descriptor to the explicitly placed scratch. An idle payload sibling still publishes that descriptor. Then use the existing QK → softmax → PV → partial update → output pipeline, MTE1 directional service modes and real AIC/AIV pipes.
4. Divide query ownership over physical AICs and retain each sequence's causal positions. AIV dequantization uses the source pair partition. A one- or two-row tail cannot borrow the idle sibling's vector supply; uneven tails have a declared busiest-unit supply bound. Detailed stages and compact stages remain fluid under external contention.
5. Reserve189952 bytes on each participating AIV's UB for the full call and release it at completion. Scratch has four KV rings and three score/output slots. `resident` requires independently declared L2 capacity and copy service; the supplied A3 profile has neither, so retained scenarios use `offchip`.
6. Pay one outer invocation completion constraint, not one per query, tile, or conversion phase.

The physical representation, coverage and unresolved mechanisms are exported in `metadata.program_execution`. Original caller tensor/source contracts remain unchanged. This does not assert a native ABI mapping to an unverified binary; profiler pairing must label it as a conditional analytic comparison.

## Current result and limits

Both scenarios execute one ordinary batched attention call at the first indexed layer. The four-layer construction includes the real preceding model structure, but only that attention call is costed; these numbers are **not complete-body results**.

| Scenario | Observed mean | Compact offchip | Detailed offchip | Compact vs observed |
|---|---:|---:|---:|---:|
| c32, q6, B32 | 0.829848 ms | 1.298268 ms | 1.358257 ms | +56.45% |
| c64, q6, B64 | 1.632353 ms | 2.596536 ms | 2.717950 ms | +59.07% |

The remaining residual is not solved. Offchip scratch dominates shared HBM and explains why AIV scaling can plateau. The unverified private dispatch, actual selection masks, scratch residency, scalar dependent lookups, flag response and conversion instruction issue rates remain distinct unknowns. No cache hit ratio or timing floor is fitted to these observations.

Verification: 30 focused tests; 32 actual Engine controls against the `0163abc5` BF16 implementation show identical work, serviced resources and completion times; 36 detailed/compact executions cover c32/c64 with HBM, AIC and AIV at0.5/1/2. All conserve physical service and release live tasks/programs. Single-operator link sensitivity is inapplicable: this kernel has no communication demand.

The default compact approximation is about4.4% below detailed execution at the original profile. At2× HBM it is16.2% /15.5% below detailed execution, so this scan **does not establish ≤15% approximation error for every resource balance**. Both modes are physically nonincreasing in each tested resource axis; a noncritical axis may plateau. The original BF16 paths are unchanged.

Small-tail ownership correction: the compact stage uses the union of the physical units that its phases actually use. For one or two selected rows, an independent job on sibling AIV1's vector pipe does not delay this kernel's AIV0 work, in either execution mode. Eight additional Engine controls cover this contention counterexample and the uneven per-AIV publish payload with deliberately constrained MTE3. Uneven fractions still use the declared busiest-unit bound; neither mode claims exact arbitration for arbitrary external contention.

Reproduce the maintained checks:

```sh
python -m pytest -q tests/test_indexed_quant_attention.py tests/test_indexed_attention_pipeline.py tests/test_indexed_attention_batch.py tests/test_indexed_attention_selection.py
```

Retained inputs: `examples/source-families/packed-int8-indexed-c32.json`, `packed-int8-indexed-c64.json`. Retained evidence: `data/verification/packed-int8-indexed-analysis-20261030.json`.
