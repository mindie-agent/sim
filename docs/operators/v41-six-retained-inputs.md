# Six additional V4.1 input declarations

The new input set is under
`examples/cases/high-concurrency-20261030/v41-new6/`. Its manifest is
`data/verification/v41-six-inputs-20261030.json`. It reuses the profiler's
312 KiB retained-source receipt and does not scan the six kernel CSVs again.
All six inputs expose `model_coverage="partial"`; this is independent of
whether the analytic prediction executes successfully.

The A5 c128/p2k case retains the independently declared 40-layer checkpoint,
TP1/DP8/EP8, mixed FP8/FP4 representation, Engram host path and no speculative
decoding. The current benchmark supplies the same engine configuration as the
existing DP8 cases. Its client concurrency 128 does not establish a per-rank
batch of 16. B1/B16 probes are conditional workload controls until the profiler
identifies the actual forward geometry.

The five A3 c4/c8/c16/c32/c64 cases have their own contract: TP4/DP4/EP16,
W8A8, DSpark seven proposals, and Engram disabled by the actual launch's
`--hf-overrides`. Engram is absent from the normalized `LayerPlan`, rather than
merely suppressed in a report. The launch selects the installed vLLM 0.31.0
pair in the standing nightly A3 container; it is not a 0.30 release declaration.
The current captures profile 4096 input tokens and 32 output tokens. Their
retained prefix hit ratio is 96.88%, not 100%.

A3 checkpoint config and quant-description bytes were not retained. The
profiler's bounded c4 evidence now establishes two cycles with 40 target layers
and three auxiliary layers each. Target hidden rows are 2 while attention query
rows are 8 and block-table batch is 1. The auxiliary queries have seven rows,
with hidden rows changing from 2 to 4 in the second cycle; those auxiliary
shapes are not applied to the target. Other catalog dimensions remain an
explicit independent scenario until the current checkpoint is identified.
W8A8 is materialized as an explicit quantization record. Source launch and
captured `DispatchFFNCombine` operands establish INT8 expert weights but do not
establish all role-specific exceptions. These uncertainties remain visible in
the input and public output coverage.

The launch requests fused MC2, FlashComm1 and shared-expert DP. Their actual
source execution is not yet represented by these templates. The ordinary
analytical decomposition retains router, dispatch, expert matrices and combine
work; it cannot be paired against the fused `DispatchFFNCombine` event as if
those were separately observed kernels. No arch35 MC2 selector is installed on
the A3 cases. The declared DSpark target verification probe uses eight query
rows per sequence and excludes draft generation/acceptance; it is not a serving
replay and does not scale query work by acceptance.

Profiler evidence records `DispatchFFNCombine` input capacity `[64,5120]`,
route capacity `[64,6]`, and INT8 weight banks `[24,5120,4608]` /
`[24,2304,5120]`. A capacity of 64 does not establish 64 requests or valid
expert rows. HcPre local rows, SparseFlashMla query rows and block-table batch
also have different TP/FlashComm boundaries. Current q/B/context, actual group
lists, graph padding and native fused scheduling remain unresolved; client
concurrency is never silently divided by DP to fill those fields.

All six inputs build real 40-layer target graphs. The five A3 inputs have
identical physical work at the same declared conditional workload, with INT8
ordinary/expert banks and zero Engram operators. Full A3 target-body runs at
q8/context4096 complete in 32.1553 ms for B1 per DP replica and 62.0776 ms for
B16, retaining all four DP participants and leaving zero live tasks/programs.
These are execution controls for the analytic decomposition, not measured
body errors or client TPOT predictions. The B1 control also lacks the observed
FlashComm hidden-row boundary, despite matching the attention query extent.
The A5 full 40-layer B16/q1/context2048 control completes in 19.5082 ms across
all eight DP participants, also with zero live tasks/programs. Its physical
result is unchanged by explicit model serialization. The previous c64
capture's sampled rows and client metrics are retained only in a clearly named
historical reference; they are not attributed to the new c128 capture.

The files contain explicit normalized model records so their declared
quantization and coverage reach the simulator. This avoids relying on the
older HF-wrapper path that ignored outer `quant` / `quantization` / `metadata`
overrides. That parser issue is tracked separately from these input records.
