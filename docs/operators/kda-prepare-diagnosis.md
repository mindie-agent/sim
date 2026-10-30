# KDA Prepare: source mechanism diagnosis

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The retained Kimi capture assigns **517.895295 ms across 276 closed calls**
to the source-compatible `GATE_PREPARE` position, versus **9.591555312 ms**
from the prior generic semantic projection. This diagnosis identifies missing
causal structure; it does not infer a correction factor or claim to explain
the complete residual.

The compact evidence and declared resource rates are in
`kda-prepare-diagnosis-20261029.json`.
The capture's independently closed four-launch blocks are retained in
`model-scoreboard-20261029-kimi-kda-call-blocks.json`.
The source is vLLM-Ascend revision
`ae923905948f360f4b7ab06df52d0789a0c9316e`; the JSON hashes each inspected
source file and confirms that the local files match that revision.

This comparison remains conditional on source order and compatible geometry.
The loaded capture binary is unknown. The descriptor permits 8129--8192 valid
tokens; the arithmetic below explicitly selects 8192, one sequence, 12 heads,
K/V width 128 and chunk size 64. It does not equate buffer capacity with valid
tokens. Q/K/V and raw gate are BF16, beta is FP32, and `cu_seqlens` is present.

## The first priority is gate ownership, not a global slowdown

`chunk_kda_fwd_common.h:164-200` runs gate cumsum before Prepare, then executes
a global synchronization. BF16 raw gate does not select the fused
`computeGateInPrepare` path. In
`kda_gate_cumsum_kernel.h:188`, varlen task count is `sequence_count * heads`:
only **12 AIVs** own the gate stage. Each owns 8192 sequential rows, with a
two-slot row pipeline. The FP32 bulk fast path at line 451 does not apply to
BF16 input.

After the barrier, `chunk_kda_fwd_prepare.h:4731,4807` assigns
`chunk_count * (heads / 2)` paired tasks round-robin. Here there are **768
tasks**, 24 per AIC, using all **32 AICs and 64 AIVs**. Restricting the whole
stage to 12 heads is wrong; using all 64 AIVs for its gate scan is also wrong.

The gate's BF16 input and FP32 output contain 25,165,824 and 50,331,648 bytes.
With the current case's declared `mte2_aiv=4.224 TB/s` and `mte3=4 TB/s`,
restricted to 12 of 64 owners, the corresponding service lower bounds are
**0.031775 ms** and **0.067109 ms**. These supply bounds do not price row
barriers, instruction issue, event latency or partial overlap.

An explicit Prepare binding must absorb the existing prefill `delta_gates`
work once. External beta sigmoid, Q/K normalization and the later POST_WU
stage remain distinct. A mixed prefill/decode call must preserve decode gate
work rather than removing a shared producer unconditionally.

## Precision, small blocks and scratch dependencies matter

The direct-UB score path uses two 32-row blocks. For one head/chunk, the
QK and KK products perform
`2 products * 2 FLOPs * 32 * (32 + 64) * 128 = 1,572,864` BF16 FLOPs,
or **2,415,919,104 FLOPs** per call. A full 64-by-64 square for both row
blocks would overcount this selected source branch.

The actual dispatcher at `chunk_kda_fwd_prepare.h:4009-4018` calls
`ComputeAkkMergeCubeWorkspace`, including when the similarly named optimized
`Arch35` implementation exists in the file. Each head/chunk executes four
16-cubed and two 32-cubed **IEEE FP32** GEMMs, for **251,658,240 FLOPs**
and **9,216 small GEMM calls** across the stage. `CubeGemmSolveSub` at
line 3505 performs GM-to-L1, L1-to-L0, matrix execution and output publication
with completion dependencies. HF32 is explicitly disabled for this solve.

The case declares BF16 at 432.5376 TFLOP/s and IEEE FP32 at 27.0336 TFLOP/s.
Since both modes share the Cube, their compute-only supply lower bounds sum
to **0.014895 ms**. The merge's 37,748,736 bytes of GM inputs and 18,874,368
bytes of outputs imply separate AIC input and Fixpipe supply bounds of
**0.008937 ms** and **0.002793 ms**. Those bounds can overlap with other
work and must not simply be added into a latency estimate.

GM is an address space, not proof of an HBM miss. The score path publishes
directly into UB; it must not be charged an invented HBM round trip. A
practical source recipe should retain the private-copy and scratch ledgers
separately and state its scratch residency assumption.

## The diagonal solve has a serial dependency chain

`SolveDiagonalBlocksInRows` at line 3018 calls
`ForwardSubDiag16StridedRegbase` at line 609. Each 16-by-16 block has 119
masked 16-lane multiplications and dependent additions, 14 row additions
with row barriers, and 16 scalar reads and writes. The defined pair helper
does not enter this dispatch.

Across the full call, the verified subset contains **24,870,912 effective
lane operations** and **98,304 scalar reads and writes each**. Each active
AIV encounters 12,768 dependent additions and 1,344 row barriers.
An aggregate dense solve cannot express this dependency chain. Conversely,
turning every barrier into an arbitrary fixed delay would fit an unknown
cost rather than model it.

The proposed refinement is a finite owner task pipeline using the existing
Engine: gate rows on sequence/head owners, a global edge, paired score
blocks, dependent diagonal phases, and the six FP32 merge phases with
their scratch paths. It must keep one native Prepare invocation rather than
charge a kernel minimum for each internal phase. Unsupported tail or
implementation branches should remain explicit support limits.

The known arithmetic and supply bounds are still far below the observed
**1.876432 ms per call**. They establish why the current mechanism is
incomplete, not where all of the missing time belongs. Instruction issue,
synchronization cost, residency and the loaded source branch remain
independent evidence questions. No rate or constant in this diagnosis is
derived from the measured residual.
