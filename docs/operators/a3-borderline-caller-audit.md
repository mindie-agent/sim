# A3 near-threshold caller audit

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

## V4 Flash: an omitted TP output reconstruction

One complete retained target forward was checked for each c4 and c8 input. All
43 layers have the same observed sequence: full-row FP32 router, TP-local
`MoeGatingTopKHash`, TP-local BF16 `MoeDistributeDispatchV2`,
`MoeDistributeCombineV2` returning the same local rows, then `hcom_allGather_`.
The c4 body has full M=8/local M=1; c8 has full M=16/local M=2. The gather starts
after combine completes. The original Sim graphs had no finalize gathers.

The independent pinned caller is `vllm-ascend` revision
`8c30e4f491f73cba737a1dacc522894e437cdf3d`:
`ops/fused_moe/prepare_finalize.py:291-306` partitions input and logits;
`:196-228,316-325` reconstructs the output; `routed_experts.py:656-675,713-717`
places routing and finalize on the respective sides. The observed source shape
sequence supports this explicit caller contract. It does not identify the
captured HCCL algorithm or prove the same compiled source revision.

The cases have DP1/TP8/EP8/ETP1, no latent expert projection and no SP selection.
They fit this caller domain. The new selection is independent of transport:
these inputs retain generic dispatch/combine, **not arch35 V3**. Their captured
V2 protocol remains a source-program gap. Expanded dispatch buffer M48/M96 is
capacity, not an inferred actual group_list or routing histogram.

Bounded capture evidence
retains all 43 caller chains per case, source and extraction identities, and row
locators. Its producer stopped after 17,126/25,366 streamed rows respectively.
The original captures and original inputs are unchanged.

## Separate tensor-precision correction and algorithm conditional

The already-present `grouped_output_a` was declared INT8 by the all-W8A8 input,
while the retained `TransposeBatchMatMul` ABI is BF16 A/B/output. The new
`mla.o_a` child override fixes only that projection, including resident weight
bytes and its activation quantizer. This is described separately in
[grouped-output-precision.md](grouped-output-precision.md).

Six explicit inputs in
`examples/cases/high-concurrency-20261030/a3-v4-caller-conditional/` preserve the
base hardware, topology, expert quantization and routing. For each c4/c8 they
contain caller-only, caller plus BF16 output A, and the latter with AICPU Mesh
AllGather as a conditional algorithm. Generic AG is the baseline. There is no
native V2 duration, missing-role activity, fitted delay or multiplier added.

Actual public `predict(scope="body")`, q2, context4096, 43 layers, runtime
`da45ba91` gives:

| Input | Base ms | Caller + generic AG ms | + BF16 output A ms | + conditional AICPU AG ms | Retained body mean ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| c4 / B4 | 26.525128 | 27.290209 | 27.638235 | 26.992612 | 32.485961 |
| c8 / B8 | 33.555657 | 34.333318 | 34.706975 | 34.060729 | 37.763131 |

The caller+BF16 generic-AG arms have conditional errors -14.92%/-8.09%. The
AICPU algorithm lowers time and has errors -16.91%/-9.80%; it is not selected
because it improves a score. These are conditional single-rank body comparisons,
not proof of exact protocol or observed routing. Context4096 is declared, and
force balancing, acceptance or prefix-hit rates are not newly assumed.

All eight baseline/conditional graphs completed in under 3 seconds, peak RSS
under 104 MiB, with zero live tasks/programs. Both expert GEMMs in every layer
retain identical shape, FLOPs, HBM and vector work across arms (86 nodes per
case). Full router rows stay unchanged; local top-k and gather shape boundaries
change. Parent physical inputs remain unmodified. The original catalog has
3 hash-routing and 40 generic router-top-k layers while the capture labels all
43 as `MoeGatingTopKHash`; this remaining semantic/source distinction is retained,
not silently replaced using the kernel name.

Simulation receipt and producer
retain each input SHA, exact source runtime, per-layer shapes, work conservation,
and conditional algorithm results. Children have an independent 60-second
budget. The observed means come from the existing profiler
`acceptance-20261030-runtime40e4300e/decode-forward-remaining-replay.json`,
104 complete target forwards per selected geometry. That report's old prediction
is not relabeled current; the fresh runtime baseline independently reproduces it.

## Kimi A3 four-layer c32: unmatched references are not extra work

The existing semantic inventory
already has 9 AG and 8 RS calls, matching the observed attention/FFN boundaries.
The 1.375535 ms/F `reduce_scatterAicpuKernel` controller overlay is not added to
that device communication. The corrected FIA reference represents all 32
sequences (about 160 us), not one sequence's approximately 5 us. The captured
16-head core versus logical 12-head projection contract remains a distinct
source-selection question; the arch35 source buffer allocation cannot simply
be assigned to A3.

The six `DequantSituQuant` calls (0.498032 ms/F) overlap existing semantic
SiTU/quantization and matrix epilogue boundaries. Their unmatched native label
is not evidence of six absent operators. The current generic short convolution
has a possible explicit q1 source-compatible replacement (three calls with
BF16 input/output [32,4608], width4, INT32 prefix33/indices32). Its entire
observed activity is only about 0.066 ms/F, so it cannot explain the 1.365 ms body
gap by itself. No Kimi timing or mechanism was changed in this audit.
