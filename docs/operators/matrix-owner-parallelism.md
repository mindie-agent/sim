# Underfilled matrix owner grids

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](../verification.md).

The default finite mapping search includes M/N tiles of 16, 32 and 64 for
FP32 projections when the ordinary 128-element grid would leave compute owners
idle. BF16, FP8 and INT8 projections keep the 128-element base domain unless a
caller explicitly declares smaller `narrow_candidates`, `underfilled_candidates`
or a complete candidate list and its compute tiles. The same capacity
checks, physical copy demands and resource bounds select among all candidates.
Explicit candidate lists retain their declared domain; already full grids keep
the existing search. This is a generic mapping choice, not captured vendor
dispatch reconstruction. The original `83f720ae` applied the smaller domain to
all precisions. That was an unsupported dispatch generalization: mathematical
feasibility alone does not establish a kernel's tile choice. The correction
restricts both automatic narrow and underfilled domains, while preserving FP32
router behavior, explicit experiments and source-owned mappings. It changes no
hardware rates or work quantities. The earlier census below remains historical;
new body comparisons must isolate this correction from other source changes.

The A5 GLM-Flash router was already an FP32 **Cube** operation. Its M=128,
N=288, K=4096 geometry exposed only three owners with 128-element tiles. The
new grid exposes independent work without changing FP32 arithmetic, logical
FLOPs or globally reused HBM bytes. More owners do perform more private operand
copies. No HF32 substitution, resource rate or invocation floor changed.

Equal resource bounds now prefer fewer output tiles per owner before lower
buffer residency. Independent review found that the former residency tie-break
could switch to two serial waves when MTE1 doubled, producing a 10.34% slowdown.
The retained counterexample now selects the one-wave plan. This does not promise
global strict monotonicity: a separate same-tile shared-HBM scheduling effect of
up to 0.642% remains in both detailed and compact execution.

## Actual tensor-cohort controls

Seven retained A5 router cohorts supplied the comparison after prediction.
All seven meet the per-call tier criterion; their measured means are below
50 us. They do not establish whole-body accuracy or actual vendor tile sizes.

| M | Observed us/call | Old grid us | Expanded grid us |
| ---: | ---: | ---: | ---: |
| 16 | 9.71 | 21.99 | 4.09 |
| 32 | 12.32 | 40.16 | 6.59 |
| 64 | 13.50–13.54 | 80.00 | 10.51 |
| 128 | 19.91–19.93 | 159.68 | 19.75 |
| 256 | 29.76 | 159.68 | 31.49 |

Producer
and controls
retain original case/observation hashes, physical service and selected mappings.

## Verification

58 focused tests passed. Independent review exercised 1,000 owner enumerations,
96 geometry/work controls, 750 Engine executions and 258 pipe-bound controls.
Late input readiness, independent groups, tails, work totals and buffer release
were checked. These counts describe correctness checks, not measured accuracy.

A final 20-probe census completed across six model declarations and four source
controls, with identical semantic work in every comparison. The longest worker
took 18.14 seconds. Source-control totals can change through their generic
projection matrices; they are not all fixed source-kernel timings. The
retained comparison
contains both package identities and the full family/service deltas. It reports
prediction changes, not measured improvement. Both arms use the same declared
steady-matmul mechanism, with the old domain explicitly disabled in the control.
