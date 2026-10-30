# BF16 multi-bank GroupedMatmul: source mapping and capture boundary

This explicit program adds the retained CANN nonquant, multi-bank `SPLIT_M`
branch to the common Engine. It does not replace the default grouped recipe.
The mapping explains part of the new Kimi E16 BF16 c32 operator activity gap;
it does not establish captured binary dispatch or whole-body accuracy.

## Source and physical contract

The source is `cann/ops-transformer` revision
`a3925fd49cd256ad478d2faf983bf2961b8ae2c8`:

- `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp`,
  lines 45–56, 135–152, 327–365 and 380–403.
- `gmm/grouped_matmul/op_kernel/arch35/non_quant/grouped_matmul_kernel.h`.

The wrapper's Blaze dependency is `cann/ops-tensor` revision
`5074e676c509af531df1f83554b6e9394cb45859`, pinned in the retained
`cmake/third_party/ops-tensor.cmake`. The official dependency files are
[block scheduler](https://raw.gitcode.com/cann/ops-tensor/raw/5074e676c509af531df1f83554b6e9394cb45859/include/blaze/gemm/block/block_scheduler_grouped_matmul.h),
[block matrix](https://raw.gitcode.com/cann/ops-tensor/raw/5074e676c509af531df1f83554b6e9394cb45859/include/blaze/gemm/block/block_mmad_matmul_basic.h)
and [grouped kernel](https://raw.gitcode.com/cann/ops-tensor/raw/5074e676c509af531df1f83554b6e9394cb45859/include/blaze/gemm/kernel/kernel_grouped_matmul.h).
All five file hashes are retained in the program and receipt.

For more than one bank, this branch keeps base N=256. The single-bank
`FormulateBasicBlock` underfill retile is not used. Hardware L0 capacities
choose base K and M; A5's declared capacities give M/N/K=256/256/64.
L1 partitions and the wrapper's minimum K step give a 256-element refill.
A and B have two L1 and two L0 slots; L0C has one slot. Each source output
tile has its own finite refill/micro-compute pipeline and direct BF16
Fixpipe output to GM. AIV/MTE3 output work is not added.

The source scheduler traverses M windows of four (with its larger final
window), reversing N in alternate windows. Its starting AIC rolls across
banks rather than resetting at each bank. Zero-row banks consume group-list
metadata, but no operand tiles. Every launched AIC reads every INT64
`group_list` entry. This is scalar GM payload only; unspecified scalar/flag
latency is not fitted.

For the two Kimi banks, N=6144 gives 48 output tiles: 16 AICs own two tiles
and 16 own one. N=3584 gives 28 tiles, with four AICs idle. Repeated private
A operand copies and B bank reads are charged to their actual AIC MTE2 pipe.
This differs from the generic N128 grid even when both grids fill the device.

Valid integer expert rows determine arithmetic and payload. The source passes
`curM` to `Mmad`; its NZ frame alignment alone does not prove an M16 cycle
minimum. We retain logical FLOPs and explicitly leave physical small-M issue
cycles and ND-to-NZ transaction efficiency unresolved. Allocated activation
capacity is never substituted for valid routed rows.

Select this source branch explicitly:

```json
{"runtime":{"kernels":{"families":{
  "grouped_gemm":{"kind":"ascend_bf16_grouped","compact":true}
}}}}
```

It requires BF16 A/B/output, no quantized epilogue, at least two banks, exact
integer `matrix_group_rows`, and private AIC copy/buffer declarations. The
ordinary frontend retains separate SwiGLU and shared-expert operations.
`compact=false` retains the detailed shared `RefillPipeline` oracle. The
compact option uses the existing closed-form matrix-cycle mechanism with
identical resource work and finite capacities. External contention remains a
fluid approximation; cross-output-tile prefetch is not reconstructed.
There is one inclusive kernel invocation for all owners and banks.

## Twelve retained capture comparisons

The producer and report are
`data/verification/kimi-bf16-grouped-20261030.producer.py` and
`data/verification/kimi-bf16-grouped-20261030.json`. The producer uses the
profiler's selected q1 batch geometry for all twelve inputs (B1/2/4/8/16/32,
p4k/p16k), and retains both source and observation hashes. Context=4096 is a
conditional probe declaration; these matrix geometries do not depend on KV
context. Group-list values, exact capture binary tiling, and resolved
force-EPLB startup remain unknown. Comparisons are descriptive and unassigned.

Selected p4k single-call means, microseconds:

| Batch | Role | Observed | Generic | Explicit source |
|---:|---|---:|---:|---:|
| 1 | expert up | 21.918 | 22.059 | 29.389 |
| 1 | expert down | 16.045 | 12.286 | 12.441 |
| 32 | expert up | 46.237 | 23.889 | 30.880 |
| 32 | expert down | 22.016 | 12.931 | 13.086 |

The two families span 92 MoE layers. For B32 the isolated activity sum moves
from 3.387 ms to 4.045 ms, compared with 6.279 ms observed; for B1 it moves
from 3.160 ms to 3.848 ms, compared with 3.493 ms observed. The B1 up result
is a counterexample to treating this source mapping as an unconditional
accuracy improvement. Under the declared forced-balanced/SP scenario,
B1/2/4/8 all have local rows `(4,4)`, while observed up mean changes from
about 22.4 us at B2 to 41.2 us at B4. The absent group-list/binary-dispatch
contract prevents assigning that discontinuity to one modeled mechanism.

## PMU activity and shared-expert overlap

Both native GroupedMatmul calls report AI_CORE, BlockNum=32 and MixBlockNum=0.
Their weight banks are BF16 NCL; activations/outputs are BF16 ND. The c1/c32
activation capacities are 16/64 rows. These capacities are not measured
valid group rows. A full streaming pass retains every PMU sample mean for
5,428 and 4,232 calls per role respectively. Raw files over 50 MiB are never
loaded in full.

| Call | AIC active us | MAC us | MTE1 us | MTE2 us | Fixpipe us |
|---|---:|---:|---:|---:|---:|
| B1 up | 21.638 | 3.892 | 5.328 | 17.352 | 2.440 |
| B1 down | 15.773 | 2.893 | 3.963 | 10.893 | 1.778 |
| B32 up | 45.958 | 7.316 | 10.004 | 41.320 | 5.865 |
| B32 down | 21.744 | 3.397 | 4.647 | 18.062 | 2.982 |

These pipeline activity counters overlap and must not be summed into latency.
A bounded first-1,200-event check, using integer nanoseconds parsed with
Decimal, finds the first B32 up kernel overlaps 45.296 us of shared-expert
Slice/Cast/Muls/Tanh/Mul/Sigmoid activity on another stream; its duration is
47.351 us. The following down call has no overlapping kernel in that window.
The analogous B1 up overlap is 20.803 us out of 21.566 us. This is evidence of
concurrency, not proof that those small vector payloads explain the full MTE2
gap.

Formal cases currently leave `multistream_overlap_shared_expert=false`.
An explicit existing shared-stream arm changes a one-layer FFN graph from
0.148019 ms to 0.114178 ms with the source grouped program and unchanged work.
It is a conditional graph control, not an observed-body replay. Consequently,
the 0.657 ms increase in B32 isolated matrix activity must not be added
linearly to predicted body latency.

## Verification and remaining boundary

The focused run passed 39 tests in 0.70 s: real 93-layer frontend selection,
integer/zero banks and tails, independent ASWT coordinates, private owner
work, five rate axes, late readiness, finite buffers, compact/detailed
single-owner equality, and one inclusive outer invocation. The twelve-case
producer and two streaming capture audits completed in about six seconds
with a 148.1 MB process peak. Rates, launch parameters, routing, and observed
measurements were not fitted.

The next useful evidence is the actual binary tiling/group-list contract or
an explicit concurrent full-body replay. The unresolved ND-to-NZ transaction,
small-M issue and scalar/flag mechanisms remain visible; the report does not
claim that passing execution invariants establishes measured accuracy.
