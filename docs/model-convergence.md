# Cross-model convergence probes

`tools/model_convergence.py` runs fixed, bounded probes across model families.
It answers which implemented programs dominate predicted cost, which selected
source paths the ordinary frontend actually uses, and how an explicit shared
mechanism change affects the same inputs. It does not use measurements to
produce predictions or turn completed runs into accuracy acceptance.

## October 29 major-cost correction with fixed hardware

The major-cost comparison holds the old normalized hardware constant so
matrix/cohort corrections are not mixed with the hardware-input precedence
repair. `model-convergence-20261029-major-baseline.json` uses the installed
`c3da0893` package and the census driver from `18282f42`; the paired
`model-convergence-20261029-major-current.json` uses committed `8374ed1a`.
Both retain the original model/runtime declarations, forced balanced routing,
default dense `TiledMatmul` and default balanced expert mapping. Prefill is
256 tokens except Kimi/GLM at 252; decode is one query token at context 2048,
with two GLM sequences and one for the other cases.

Both sides completed all twelve probes in 103.06 and 109.93 host seconds.
All twelve pair with unchanged semantic work. Costs below are sums of
isolated device programs in milliseconds, including complete DP/EP cohorts:

| Model | Baseline prefill | Corrected prefill | Baseline decode | Corrected decode |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 | 45.657573 | 45.657573 | 10.753194 | 10.753194 |
| Qwen3.5 | 12.486662 | 12.486662 | 2.494902 | 2.494902 |
| Qwen3.6 | 11.220206 | 11.220206 | 6.151702 | 6.151702 |
| Kimi K3 | 397.041772 | 395.101855 | 190.421759 | 189.045844 |
| GLM5.3 | 49.030270 | 48.978085 | 24.451167 | 24.398061 |
| V4.1 DP4 | 233.421354 | 214.684967 | 31.285375 | 27.220845 |

V4's corrected costs are 8.03% lower for prefill and 12.99% lower for decode.
Independent head/group matrices now retain their group-major owner grid and
group-local weight payload. This changes private copies and participating
owners, not logical model work. The three Qwen controls are unchanged.
These are corrected predictions, not measured speedups or accuracy scores.

The hardware resolution report is separate:
`hardware-input-resolution-20261029.json`. Raw case files now honor their
explicit native MTE values instead of silently inheriting profile rates.
No case rate was tuned to a timing observation. Re-running the original raw
cases therefore need not reproduce the fixed-hardware table.

`major-cost-communication-20261029.json` holds the same normalized hardware
and balanced routing while comparing declared envelope versus explicit
ring/pairwise programs. Its bounded complete graphs use `phase_matmul`:

| Case | Envelope graph (ms) | Explicit graph (ms) |
| --- | ---: | ---: |
| GLM, two sequences | 25.931032 | 18.821800 |
| Qwen3.6 | 6.610140 | 6.610140 |
| V4, four DP participants | 9.633921 | 9.633921 |

GLM's 27.42% difference establishes sensitivity to declared communication
assumptions. It does not identify the deployed algorithm. The Kimi probe in
that report has different isolated transport scopes on its two sides; its
208.389850/194.265346-ms sums must not be treated as algorithm sensitivity.
All eight probes completed within the declared budget; no Kimi full graph
was added to that campaign.

The subsequent `7aa7acbb` correction changes reporting/ranking only. Public
operator studies now separate every transport call's per-replica or joint
cohort isolation, including mixed dispatch/combine choices. Full graph
rankings share their complete execution boundary. The numerical campaigns
retain their actual `8374ed1a` provenance and were not relabeled as reruns.
See `major-cost-audit.md` for remaining communication, narrow-matrix and
ordinary W8A8 output work; forced balance does not establish correct return
flows for unequal DP sources.

## Probe and runtime boundaries

The default matrix has twenty probes:

- Qwen3.8, Qwen3.5, Qwen3.6, Kimi K3, GLM5.3 and V4.1 DP4, each at prefill
  256 tokens and decode one token with 2,048 preceding context tokens.
- Ordinary GQA/RMSNorm, integer-routed grouped GEMM, dense MLA decode and
  dynamic INT8 source controls, each at prefill and decode. Their original
  small supported geometry is retained, including the grouped control's
  actual integer routing scenarios.

Model dimensions, precision and deployment declarations stay unchanged.
`target_graphs` constructs one independent DP cohort; the Kimi and V4.1 cases
contain all four EP-participating DP replicas. Tokens and sequences are per
participating replica. Other independent DP cohorts are not multiplied into
these sums. Prefill/decode probes supply target-only pure-model work, excluding
draft processing, serving admission, sampling and request constraints.

The default `operators` level runs distinct physical operators and weights
their isolated costs by actual occurrences. That sum is not the graph span:
it includes work on simultaneously active replicas. `--level graph` adds a
separate overlap-aware graph duration using the same public engine. Neither
boundary is TTFT, TPOT or client-stream latency.

Each probe runs in one child process. Defaults are 60 seconds per probe, 300
seconds total, at most 24 probes and at most 50,000 expanded operators per
probe. An exhausted budget retains `timeout` or `not_started`; it never makes
a missing cost zero. The report keeps bounded top signatures and all family
totals, with no device execution or weights. The current probe suite is small;
these limits do not establish arbitrary-model memory or runtime bounds.

`--python` selects the interpreter/package being assessed. Workers use isolated
`-I` startup, so an installed baseline wheel does not accidentally import
the active checkout. `--source-root` explicitly selects a checkout for a later
run. Each successful probe records a hash of the imported package's Python
and JSON files, its matrix module hash and import path. A report whose workers
observe different package hashes is `partial`, not a coherent before/after run.

Each input has its raw file hash and canonical input fingerprint, plus the
fully normalized case fingerprint. Comparison requires identical probe
geometry and base case. Explicit matrix recipe axes may differ:
`runtime.kernels.programs.matmul` and the two
`runtime.kernels.recipes.balanced_grouped_gate_up` / `balanced_grouped_down`
entries. The latter can be supplied together with `--balanced-matmul`.
A changed model, rate, parallel configuration, routing policy or workload is
unpaired. The report retains the declared recipes and resolved balanced
program classes, including when a default recipe supplied the implementation.
This does not rank different models against each other.
Comparison coverage includes probes present only in the reference. A completed
`--only` subset cannot erase unexecuted baseline probes from its comparison;
they remain unpaired without a numerical difference.
Work comparisons retain every numerical delta. Relative `1e-12` and absolute
`1e-9` tolerances distinguish floating-point summation order from changed
work; these are arithmetic roundoff tolerances, not performance tolerances.

Mapped programs retain selected outer and inner tiles, per-owner buffer
footprints, rejected candidates, the resource bound used for selection and
undeclared transfer roles. Identical diagnostics are deduplicated; top
signatures reference them by fingerprint and invocation counts include actual
operator repetition. Graph diagnostics have their own recorded/omitted counts,
independent of isolated analysis. A source-bound program remains identified as
such even when other operators use the generic mapping selector.

## Baseline cost concentration

The baseline uses the already installed `edf9e156` delivery wheel recorded by
`data/verification/wheel-20261029-calls-rounds.json`, with wheel SHA-256
`20831058a4221e7a3e9feedb4d025086c6d37630c59bcddf3bfbe439ace9401e`.
The compact result is
`data/verification/model-convergence-20261029-baseline.json`.

The table gives the generic `TiledMatmul` share of each probe's isolated sum,
not a measured bottleneck or graph critical-path share:

| Case | Prefill 256 | Decode 1 / context 2,048 | Other prominent predicted work |
| --- | ---: | ---: | --- |
| Qwen3.8 | 75.03% | 96.17% | Prefill native GDN phases; decode dominated by dense gate/up, down and QKVZ matrices |
| Qwen3.5 | 79.35% | 38.32% | Grouped expert matrices; decode expert-tensor reduction and allreduce |
| Qwen3.6 | 38.75% | 24.67% | Allreduce and linear reduction dominate the decode sum |
| Kimi K3 | 59.16% | 30.07% | Expert dispatch/combine account for about 59.9% of the decode sum |
| GLM5.3 | 52.86% | 17.37% | Hyper-connection projection in prefill; reductions and expert communication in decode |
| V4.1 DP4 | 97.67% | 92.57% | Grouped experts, output projection and hyper-connection projection |

These data make generic matrix mapping and communication high-impact shared
refinement targets. They do not establish which mechanism is inaccurate. The
original local and semantic shapes remain in `top_signatures`; family labels
alone are not used to infer physical implementation.

The catalog cases, except Qwen3.8's selected native prefill branches, currently
use analytical programs at their ordinary frontend boundaries. The four source
controls prove that supported GQA/norm, grouped BF16/ND, dense MLA decode and
dynamic INT8 declarations reach their source planners. They do not silently
upgrade the catalog deployments to those implementations. Source control
prefill and decode are reported separately; MLA prefill intentionally retains
its analytical path.

## Explicit mapped candidate at 76df99d

`data/verification/model-convergence-20261029-mapped.json` records the public
`{"kind":"mapped_matmul"}` candidate at core commit `76df99d`. All twenty probes
completed in 106.284 seconds, below the declared 60-second per-probe and
420-second total budgets. Every worker used package content hash
`c6b785bd2325e88cbaf80bded3899b2bc7a33d8222c66866e95013267ae70ffe`;
the report retains exact imported paths, individual matrix source hash and
driver hash. This is the matrix-mechanism comparison before subsequent GDN
structure changes, not a final total for those later changes.
The retained reports also precede the comparison helper's missing-probe
coverage repair. Their probe sets were independently checked to be identical;
original runtime and driver hashes remain unchanged.

The unchanged cases use the same logical arithmetic and traffic under both
predictions. The only numeric work delta is V4.1 prefill's `mte3_bytes`:
`3.814697265625e-06` bytes out of approximately 25.2 billion bytes, arising
from floating-point accumulation. Actual earlier removal of expert input
quantization was caught by this census and repaired before this retained run.
The regression test checks real low-precision grouped input producers and
their payload/scale traffic.

These are isolated program sums in milliseconds. Each cell compares the
installed baseline with the explicit mapped candidate for the same probe:

| Case | Prefill baseline -> mapped | Decode baseline -> mapped |
| --- | ---: | ---: |
| Qwen3.8 | 45.567822 -> 35.885977 (-21.247%) | 10.753194 -> 10.757934 (+0.044%) |
| Qwen3.5 | 43.399142 -> 43.348472 (-0.117%) | 3.109166 -> 3.118781 (+0.309%) |
| Qwen3.6 | 11.220206 -> 10.676194 (-4.849%) | 6.151702 -> 6.165536 (+0.225%) |
| Kimi K3 | 618.000488 -> 598.296998 (-3.188%) | 190.962753 -> 190.001486 (-0.503%) |
| GLM5.3 | 51.043402 -> 50.992620 (-0.099%) | 24.088517 -> 24.074947 (-0.056%) |
| V4.1 DP4 | 1054.982717 -> 1027.699667 (-2.586%) | 33.765647 -> 29.768140 (-11.839%) |

Mapping screens eight declared combinations: outer M/N dimensions of 128 or
256, and K panels of 64 or 128; inner M/N consumption tiles are 128. The
selected programs run through the same finite-resource engine. GLM5.3 and
V4.1 FP32 hyper-connection projections select K64 where double-buffered K128
operands exceed the declared L0 capacity. Kimi, GLM and V4.1 preserve independent
head/group matrices rather than treating their total weight payload as one B
matrix resident in L0. Selection is a resource-bound screen, not a prediction
of vendor dispatch or a claim that the chosen candidate is optimal.

The six GQA/norm, source grouped and INT8 control probes retain exactly the
same isolated sums. MLA control prefill changes from 1.566664 to 1.021640 ms
and decode from 2.789500 to 2.642108 ms because its surrounding generic batched
projections now retain their independent heads. Its explicitly bound source
attention programs remain unchanged. A separate sixteen-head MLA control
executes the complete graph with declared 64 KiB L0 operand capacities and
checks unchanged semantic work and source attention selection.

The matched graph reports are
`model-convergence-20261029-graph-baseline.json` and
`model-convergence-20261029-graph-mapped.json` in `data/verification/`:

| Complete graph probe | Baseline span (ms) | Mapped span (ms) |
| --- | ---: | ---: |
| Qwen3.5 decode / context 2,048 | 3.109166 | 3.118781 |
| GQA/norm source control prefill | 0.452600 | 0.452600 |
| MLA source control decode | 2.789500 | 2.642108 |

All three mapped graphs complete with zero live tasks/programs and no omitted
mapping invocations. These particular graphs serialize their reported work,
so span equals the isolated sum within arithmetic roundoff; the test suite
also contains a real two-replica expert cohort where those values differ.
Neither successful graph execution nor a lower predicted cost is measured
accuracy evidence.

The remaining high-impact work is now visible across models:

- Aggregate expert programs retain their existing analytical `TiledMatmul`
  family binding, with `matrix_mapping_domain=aggregate_experts`. They account
  for 76.40% of Qwen3.5 prefill, 89.78% of V4.1 prefill and 35.98% of Kimi
  prefill in this mapped run. Probability moments do not identify individual
  expert row counts or tile residency. Explicit source grouped kernels remain
  a separate supported domain; the new dense/batched selector does not claim
  to resolve these aggregate expert costs.
- Kimi decode dispatch/combine and GLM/Qwen3.6 decode reductions remain major
  predicted costs. A generic matrix change cannot settle their collective
  algorithms, placement or critical-path impact.
- Mappings still use declared rates, candidate sets, resource ownership and
  cache-reuse assumptions. The retained tensor-contract medians identify
  refinement priorities; they do not establish this candidate's kernel
  dispatch, shape-general timing accuracy or heldout model accuracy.

The next grouped mechanism should first accept explicit integer histograms.
With M128 tiles, `(128,128)` and `(1,255)` both contain 256 rows and two active
experts, but require two and three expert-local tiles respectively. A shared
ragged program must preserve those domains, one kernel submission, per-owner
resources and operand lifetimes. It cannot turn probability moments into an
invented routing realization. The existing source planners keep their
selection domains and priority.

For communication, complete cross-DP cohorts currently rendezvous before
their participant programs but cannot select the local-only ring recipe.
Joint per-peer round dependencies, integer sender/receiver payloads and the
composition of finite credits with routed hops remain separate shared work.
The current uniform transport or aggregate approximations do not claim those
mechanisms. These are refinement targets, not changes included in the reports.

## Retained large-prefill impact

The optional prior-report inputs do not rerun or train the predictor:

- `data/verification/gdn-impact-baseline-edf9e156-prefill16k.json` retains the installed baseline's
  16K Qwen3.8 isolated predictions, total 2,368.328367 ms.
- `data/verification/qwen38-operator-prefill.json` retains tensor-contract
  capture medians and profiler source lines.

`retained_gemm_impact` requires a unique predicted variant and captured
signature for each recipe/shape/dtype contract before projecting a median
across modeled occurrences. Ambiguous variants remain unpaired. Input hashes,
source references, shapes and signed/absolute residuals are preserved.

The four largest GEMM projections contribute **+780.104968 ms** signed
residual: gate/up +329.918913 ms, QKVZ +210.582334 ms, down +160.883982 ms and
linear output +78.719739 ms. This identifies a much larger shared cost issue
than inspecting only native GDN errors. It is conditional on tensor matching:
the compiled GEMM schedule and capture dispatch are not identified by that
match, and the observed medians multiplied by modeled repetitions are not
a measured model duration. The model sum is a stated priority denominator,
not an error bound or a claim of whole-model accuracy.

## Reproduction and next comparisons

Use the Python executable in the retained wheel manifest's `environment`:

```sh
.venv/bin/python tools/model_convergence.py \
  --python /absolute/path/to/retained-wheel-env/bin/python \
  --runtime-label edf9e156-wheel-20831058 \
  --prior-operators data/verification/gdn-impact-baseline-edf9e156-prefill16k.json \
  --prior-comparison data/verification/qwen38-operator-prefill.json \
  -o data/verification/model-convergence-20261029-baseline.json
```

For the same probes against the active checkout:

```sh
.venv/bin/python tools/model_convergence.py \
  --source-root /Users/maoxx241/code/sim-rewrite-20261027 \
  --runtime-label mapped-matmul-current \
  --matmul '{"kind":"mapped_matmul"}' \
  --reference data/verification/model-convergence-20261029-baseline.json \
  --probe-seconds 60 --total-seconds 420 \
  -o /tmp/model-convergence-candidate.json
```

`--matmul` accepts one JSON recipe declaration and changes only that public
configuration axis. `--only qwen38 v41-dp4` restricts costlier exploratory
runs without modifying the preset. Final baseline and candidate runs should
use the same set of probes. Program-cost and physical-service changes remain
separate from unchanged semantic FLOPs/bytes.

The tests exercise actual public source-family calls, repeated layers,
cost/occurrence conservation, explicit mapping changes, unequal-input rejection
and complete cross-DP cohorts. Mapping diagnostics are checked against the
actual executed shape and occurrence count while source-bound programs remain
distinct. The tests also establish that duplicate tensor
variants cannot be counted twice against a single captured median. These tests
verify the census, not the timing accuracy of the modeled programs.

## Forced-balance mechanism comparison

`--routing '{"policy":"balanced"}'` is an explicit synthetic routing scenario.
It counts each logical token once per complete EP cohort, preserves integer
expert rows and empty banks, and selects equal-group execution only when the
nonzero expert geometry and rank workloads are equal. Uniform peer traffic
remains a declared transport approximation; it is not an inferred token
permutation. This scenario is distinct from the catalog's probability moments.

Both sides of a forced-balance comparison must use that same routing scenario.
The aggregate-tile control explicitly binds the two balanced recipes to
`TiledMatmul(128,128,128)`. The candidate uses their default `MappedMatmul`,
which retains separate expert matrices in one grouped program. Dense matrices
use the same explicit `mapped_matmul` configuration on both sides. The control deliberately
retains the earlier aggregate-tile approximation; its timings are neither a
vendor baseline nor evidence that one implementation is faster on hardware.

Perfect balance needs the number of logical top-k assignments to divide by
the number of experts. For these deployments, prefill256 is valid for
Qwen3.5 and V4.1; Kimi K3 and GLM5.3 use prefill252. These are new paired
workloads and must not be compared numerically with the retained prefill256
reports. `--model-prefill-tokens` expresses this geometry without editing the
original case files.

Decode remains one query token per request with context2048. GLM5.3 uses two
simultaneous requests through `--model-decode-sequences '{"glm53":2}'`: one
request supplies eight assignments to sixteen EP ranks, which does not meet
the equal-rank execution domain. The single-request refusal remains a separate
scope control and is not converted to a zero-cost successful probe.

The paired commands are:

```sh
.venv/bin/python tools/model_convergence.py \
  --source-root /Users/maoxx241/code/sim-rewrite-20261027 \
  --only qwen38 qwen35 qwen36 kimi-k3 glm53 v41-dp4 \
  --routing '{"policy":"balanced"}' \
  --matmul '{"kind":"mapped_matmul"}' \
  --model-prefill-tokens '{"kimi-k3":252,"glm53":252}' \
  --model-decode-sequences '{"glm53":2}' \
  --balanced-matmul '{"kind":"tiled_matmul","tile_m":128,"tile_n":128,"tile_k":128}' \
  --runtime-label balanced-aggregate-tile-control \
  --probe-seconds 60 --total-seconds 300 \
  -o data/verification/model-convergence-20261029-balanced-control.json

.venv/bin/python tools/model_convergence.py \
  --source-root /Users/maoxx241/code/sim-rewrite-20261027 \
  --only qwen38 qwen35 qwen36 kimi-k3 glm53 v41-dp4 \
  --routing '{"policy":"balanced"}' \
  --matmul '{"kind":"mapped_matmul"}' \
  --model-prefill-tokens '{"kimi-k3":252,"glm53":252}' \
  --model-decode-sequences '{"glm53":2}' \
  --runtime-label balanced-equal-group-mapping \
  --reference data/verification/model-convergence-20261029-balanced-control.json \
  --probe-seconds 60 --total-seconds 300 \
  -o data/verification/model-convergence-20261029-balanced-mapped.json
```

The retained October 29 execution used committed runtime `475a0b71` from
`/tmp/sim-balanced-final-20261029-3tulxnka` for both `--root` and
`--source-root`, with the current census driver. This kept active, unrelated
native-contract edits outside the comparison. Both reports have package hash
`937cde53caf60ade4bc24f789092a94750af8224ef1f4797b112c2b8a1a23b8b`
and driver hash
`0668bf67607c8a8d2e1b6b58b812b56d29bd2466f97c69023d76c5336effcce6`.

Both sides completed all twelve probes, and all twelve paired with unchanged
semantic work and operator occurrence counts. The table reports isolated
model-program sums in milliseconds:

| Case | Prefill control | Prefill equal-group | Decode control | Decode equal-group |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5 | 43.358206 | 12.435991 | 3.118781 | 2.504517 |
| Kimi K3 | 525.516804 | 377.373892 | 189.833032 | 189.460492 |
| GLM5.3 | 49.258302 | 48.978670 | 24.430844 | 24.437411 |
| V4.1 DP4 | 1,027.699755 | 206.138304 | 29.947994 | 27.287868 |

Qwen3.8 and Qwen3.6, which have no routed expert layers in these declarations,
are exactly unchanged. GLM decode's small increase is retained. Kimi and V4
include all four DP participants in their EP cohorts. Every isolated program
finished with zero live tasks; remaining buffer sums are zero apart from
sub-nanobyte floating-point roundoff.

The control completed in 73.877295 seconds and the candidate in 113.870892
seconds, within the declared budgets. These wall times measure the local
simulator run, not model latency. The changes in predicted cost demonstrate
the importance of the declared expert mapping, not measured hardware
speedups or model accuracy.

`model-convergence-20261029-balanced-unsupported.json` retains the separate
single-request GLM decode invocation. It exits 2 with
`balanced_equal_groups_require_equal_rank_workloads`, status `partial`, one
failed probe and no numerical cost. All three reports keep measured accuracy
`unassessed` and consume zero observations.

`model-convergence-20261029-balanced-graphs.json` additionally runs
`--level graph --only qwen35-decode glm53-decode v41-dp4-decode` on the same
clean runtime, forced balance and dense mapping, with two GLM requests.
Its per-probe/total budgets are 60/180 seconds:

| Graph control | Device graph span (ms) | Isolated sum (ms) | Result |
| --- | ---: | ---: | --- |
| Qwen3.5 decode | 2.504517033 | 2.504517033 | Completed, zero live tasks/programs |
| GLM5.3 two-request decode | 24.437411429 | 24.437411429 | Completed, zero live tasks/programs |
| V4.1 full EP cohort decode | unavailable | unavailable | Per-probe timeout |

The report is `partial`, exits 2 and completes in 70.878632 seconds. The first
two declarations use serial main-stream programs, so their graph spans agree
with the isolated sums within arithmetic roundoff. Their default catalog
communication still executes `ResourceEnvelope`; this does not establish
pairwise scheduling coverage. The V4 isolated census succeeded separately,
but its complete composed graph was not completed within this budget.
