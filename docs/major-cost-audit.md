# Whole-model cost and decision-sensitivity audit

This audit uses the retained six-model phase/mapped reports and the installed
`c3da0893` wheel. It precedes the subsequent hardware-input and cohort-screening
repairs. It reads no measurements, changes no rates to fit timings, and makes no
whole-model accuracy claim. The new model probes each have a 30 s deadline; the
completed probes used about 33 s in total.

## Priorities supported by concrete counterexamples

1. **Keep a fidelity check on configuration rankings.** Serial phase aggregation
   can reverse a hardware choice even with identical physical work. It is a
   coarse screening model, not a substitute for overlap-sensitive decisions.
2. **Record the finite mapping gap for narrow output matrices.** The current
   default 128/256 tiles leave most owners idle for a major FP32 projection.
   A small-M candidate is materially faster despite doing more private copying.
   This is a later generic candidate-domain improvement. No new recipe or
   model-name exception is added in this audit.
3. **Inspect collective startup assumptions before bandwidth tuning.** More than
   half of Kimi's simulated graph span comes from declared
   dispatch/combine startup rounds. Its wire service is almost irrelevant at
   this probe size. This is an assumption to resolve, not evidence that a
   real collective uses the declared schedule.
4. **Make ordinary operator screening accept full cohorts.** The public study
   path fails on the already-supported pairwise expert cohort when it attempts
   per-replica isolation. The current patch repairs this composition gap using
   the existing joint collective machinery; the baseline evidence below
   intentionally retains the old failure.

These findings take priority over fitting unmatched grouped event residuals.
No new instruction-level simulation is proposed.

The implementation order separates actual missing work from fidelity choices.
Input-override and cohort-entry fixes are integrated in `8374ed1a`. Independent
matrix groups are integrated in `f2a2885a` with the aggregate-domain correction
in `b1f18c27`; physical-source expert communication volume is corrected in
`65433eba`. These later fixes do not rewrite the `c3da0893` baseline evidence.
Ordinary W8A8 output post-scaling remains a concrete contract gap described
below. Narrow-tile search is deferred. The phase counterexample limits how
rankings may be interpreted and does not propose another execution engine.

## Cost concentration and timing boundaries

The retained phase probe geometry is prefill 256 tokens, except Kimi/GLM at
252, and decode one query token with context 2048. GLM decode has two sequences;
the other cases have one sequence per participating DP replica. The table
contains sums of isolated programs, not request latency.

| Case | Prefill isolated ms | Main prefill costs | Decode isolated ms | Main decode costs |
| --- | ---: | --- | ---: | --- |
| Qwen3.8 | 66.979 | Dense gate/up 28.5%, down 19.3%, linear QKVZ 16.8% | 13.883 | Dense gate/up 29.5%, down 22.2%, linear QKVZ 19.0% |
| Qwen3.5 | 14.131 | Expert tensor reduce 40.6%, grouped gate/up 13.6%, all-reduce envelope 8.8% | 2.666 | Expert tensor reduce 23.3%, all-reduce envelope 22.6%, linear reduce 17.0% |
| Qwen3.6 | 14.646 | Dense gate/up 25.7%, all-reduce envelope 20.4%, linear reduce 15.3% | 6.610 | All-reduce envelope 33.9%, linear reduce 25.4%, dense gate/up 13.3% |
| Kimi K3, four DP representatives | 571.254 | Grouped gate/up 16.8%, latent projection 11.4%, dispatch/combine 21.7% | 208.390 | Dispatch/combine 54.8%, latent projection 6.8% |
| GLM5.3 | 66.543 | HC projection 30.1%, grouped gate/up 13.4%, all-reduce envelope 10.8% | 25.931 | All-reduce envelope 26.0%, linear reduce 19.7%, dispatch/combine 24.3% |
| V4, four DP representatives | 402.868 | Grouped gate/up 36.5%, HC projection 22.1%, grouped down 18.3% | 38.423 | HC projection 18.7%, grouped gate/up 17.8%, grouped output B 9.2% |

Each DP representative models its homogeneous TP/PCP shard resource plane.
Summing four DP representatives is not a single-request duration, and the sum
must not be divided by TP again. The full graph preserves shared-resource
contention and cohort dependencies:

| Complete phase graph | Isolated sum ms | Graph span ms | Wall s | Final tasks/programs |
| --- | ---: | ---: | ---: | --- |
| Kimi K3 decode, DP4/TP8/EP32 | 208.389850 | 52.097463 | 21.847 | 0 / 0 |
| V4 decode, DP4/TP1/EP4 | 38.422835 | 9.633921 | 5.368 | 0 / 0 |

Kimi executed 15,496 frontend nodes and 591,264 engine tasks; peak live tasks
were 31,808. Its isolated-cost precedence bound is 52.097463 ms, matching the
graph span at this geometry. This is a bound for the declared graph, not a
general parallel speedup. V4's sum divided by four is 9.605709 ms, slightly
below its actual graph span because the graph retains shared effects.

## A real phase-ranking reversal

The independent control is BF16 `[M,N,K]=[4,2,40]`, macro tile `[4,2,2]`,
inner tile `[2,2]`, one physical owner and two refill/consumer slots. It executes
640 matrix operations, 496 HBM bytes, 480 private refill bytes, 640 inner-copy
bytes and 16 output bytes for every candidate and fidelity.

Candidate A has 1 ms refill and 1 ms compute per panel. Candidate B makes refill
ten times faster but compute 1.5 times slower. The actual shared Engine gives:

| Candidate | Detailed overlapping pipeline ms | Serial phase model ms |
| --- | ---: | ---: |
| A: balanced load/compute | 21.0 | 40.0 |
| B: faster copy, slower matrix | 30.1 | 32.0 |

The detailed model prefers A; the phase model prefers B. This is the declared
phase abstraction's cost, not missing bytes or an engine arithmetic bug.
`tests/test_phase_ranking.py` retains the executable counterexample. A
phase-only ranking must remain explicitly conditional until the relevant
leading candidates are checked with the detailed recipe or a justified
overlap-aware abstraction.

## Narrow-N mapping sensitivity

The real GLM prefill operator `hc_projection [252,24,16384]` has 198,180,864
FP32 FLOPs. The default mapping selects `[128,128,64]` with 128x128 inner tiles.
It has only two output tiles and therefore uses only two of the 32 owners.
The full-chip matrix service is 0.007330909 ms, while its ownership-constrained
mapping bound is approximately 0.119156 ms.

An explicit analytical candidate domain with M of 16/32/64, N of 32, K of
64/128 and inner tiles 16x32 selects `[16,32,128]`, using 16 owners:

| Same operator | Default domain ms | Small-M domain ms |
| --- | ---: | ---: |
| Detailed mapped execution | 0.119589 | 0.021206 |
| Serial phase execution | 0.222460 | 0.042078 |

HBM remains 18,112,128 bytes and matrix work is unchanged. Private MTE2 and
MTE1 each increase from 19,660,800 to 41,680,896 bytes; the improvement does
not come from deleting copies. Better owner participation outweighs the extra
private weight traffic in this declared hardware model.

The GLM family occurs 90 times and contributes 30.1% of its phase prefill
sum. V4 has a similar `[256,24,20480]` family contributing 22.1%. This is a
material reason to review the generic mapping search domain. The probe does
not establish a vendor's preferred tile, native instruction padding or actual
runtime dispatch, and no full-model improvement is claimed from multiplying
this one operator's change.

## Communication assumptions on the critical path

Kimi has 92 expert dispatches and 92 combines per DP representative. Each
declares 31 rounds, with the largest route latency 5 microseconds. For each
direction:

```text
declared startup = 92 * 31 * 0.005 ms = 14.26 ms
actual simulated total along the path = 14.26494592 ms
```

Startup is 99.965% of the direction's simulated time. Dispatch plus combine
contribute 28.529892 ms, or 54.8% of the complete 52.097463 ms graph.
Increasing bandwidth alone barely changes this term. Useful next evidence
concerns the actual collective algorithm, admissible overlap and latency
domain. The ordinary catalog still selects `ResourceEnvelope`; it must not
be described as already using the explicit pairwise recipe.

All numbers in this section are predicted costs at declared geometry and
transport assumptions, not measured performance accuracy. The physical-source
volume correction is integrated, but general nonuniform return traffic remains
an unresolved contract defect: the combine byte matrix is the transpose of the
dispatch matrix, and unequal or idle sources cannot be reconstructed from one
uniform-peer scalar per source. The current nonuniform frontend can still
return `modeled` or `served` with the wrong return-traffic direction; it does
not reject that domain. Balanced equal-source controls describe the verified
scope, not an existing runtime restriction or a validation of that inverse.

The related entry-point counterexample was run on the installed `c3da0893`
baseline using the existing
`balanced-experts-control.json` and `study(..., level="operators")` with a
four-token prefill. That baseline result explicitly failed with
`operator_collective_requires_complete_cohort:use_graph_costs`. Graph execution
already supported that complete cohort. The `8374ed1a` repair now shares its
cohort-aware isolated pricing, preserving participant attribution and
communication work. The retained failure is historical evidence, not the
current entry-point behavior.

Independent review of the new entry point compared two EP cohorts with
DP4/TP2/PP2/EP4, shuffled replica order, an idle sender, two pipeline stages,
repeated collectives and local compute. All 20 occurrences, 130 ms of isolated
completion spans and the entire serviced-resource ledger matched graph
isolation. MLA and activation-quantization source controls also matched graph
isolation without expanding their kernels twice. Hardware override review
found and repaired a second-layer alias collision in study candidates:
`hbm_bw_bytes_per_s`, `rates.memory` and `rates.hbm` now describe the same later
bandwidth override. These are implementation checks, not measurement accuracy.
The same review found an accepted top-level Ascend `hbm_gib` being shadowed
by the profile; it now supplies the machine default before explicit device
and per-device overrides. The final focused review passed 25 tests in 0.28 s
with both discovered precedence defects repaired.

## Context and precision sanity checks

No order-of-magnitude matrix unit error was found. Matrix service is stored in
`peak_ms`, with `1000 * FLOPs / declared_peak`, on a resource whose rate is one.
That ledger is not a FLOP count. INT8/FP8 use their declared compute peaks.
MXFP4 routed experts explicitly retain BF16 computation: packed weight plus
group-32 scale is 0.53125 B/value, while the inner compute-width bound is
2 B/value. The 3.7647x expansion is an explicit representation assumption,
not already-modeled FP4 arithmetic throughput.

There is nevertheless a concrete ordinary W8A8 contract omission. The
installed-wheel frontend emits 270 INT8 matrix occurrences for each Qwen3.5
prefill/decode probe. Every occurrence retains an explicit activation
quantization producer, but all have zero vector work and no `matrix_epilogue`.
The ordinary matrix path counts row/column scales as K-panel operand traffic
without expressing the final INT32-to-BF16 scale application. The already
available public `MatrixEpilogue` mechanism has not been bound to this ordinary
path. This is missing semantic work; a zero residual or passing source-specific
grouped test would not resolve it.

Kimi/V4 packed MXFP4-to-BF16 operators likewise declare
`dequant_before_compute=True`, while widened inner-copy bytes do not imply
that conversion latency has executed. Their zero conversion vector work is
an explicit unresolved primitive/service domain. It must not be filled with
an inferred timing multiplier. The durable audit contains the per-family
frontend counts and relevant metadata from the independent quantization audit.

The installed-wheel context probes retain the same model, routing and sequence
counts:

| Probe change | Relevant isolated work/time behavior | Whole-cohort isolated change |
| --- | --- | ---: |
| Qwen3.5 context 2048 -> 32768 | Flash attention 0.005256 -> 0.083899 ms, approximately 16x | +2.95% |
| Kimi context 2048 -> 32768 | Latent attention 0.057277 -> 0.906624 ms, approximately 16x | +0.41% |
| GLM context 2048 -> 32768 | Index scores 0.002930 -> 0.046860 ms; selected attention retains its top-k work | +0.17% |
| V4 context 2048 -> 4095 | Index scan approximately doubles; compressed/window-selected attention remains bounded | +0.026% |

These trends do not show a new numerical scaling bug in the declared analytic
domain. Small whole-model sensitivity follows from the larger matrix and
communication terms; it does not validate native MLA/Indexer semantics.
The profiler lane owns those semantic bindings and this audit does not modify
them. V4 stays inside its declared 4096-token serving envelope.

## Evidence and next boundary

The durable concentration inputs are
`data/verification/model-convergence-20261029-phase.json`,
`model-convergence-20261029-balanced-mapped.json` and
`v4-phase-delivery-20261029.json`. New bounded probe results, their source
hashes, the compact cost concentration table and low-precision frontend
counts are retained together in
`data/verification/major-cost-audit-20261029.json`. It records the installed
wheel identity and keeps the old cohort failure separate from the subsequent
patch-review results. The phase counterexample is also an executable test.

Root owns the hardware-input and cohort-screening fixes. This lane changes no
runtime or live profiler files. The grouped, communication and input/cohort
correctness fixes are integrated; the W8A8 output contract remains open.
Resolve that missing work and assess the integrated model costs before
refining individual kernel residuals or expanding the finite mapping domain.
