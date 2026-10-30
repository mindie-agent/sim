# Model timing scoreboard

Development-run reports cited below are historical context and are not shipped. Current acceptance is documented in [verification](verification.md).

`tools/model_scoreboard.py` assembles independently observed model windows and
actual Sim graph probes. It reports retained values even when a comparison
cannot be made. The observation side never supplies durations, fitted rates,
or residual terms to the prediction worker.

The final report contains five conditional full-window comparisons, including
the separately authorized V4 Flash A3 workload, plus descriptor-conditioned
Kimi family comparisons. V4.1 remains a distinct row without a matching
capture; Flash is never renamed to V4.1. Earlier phase reports are preserved
as historical results. None of these descriptive comparisons certifies
six-model accuracy.

The later Kimi mechanism pass is documented in
`model-scoreboard-20261029-kimi-mechanism-review.md`.
It restores the independently declared MTE1 profile, adds role-specific
checkpoint precision, prices the previously missing SiTU population, and
executes four disjoint KDA semantic stage projections. Its current artifact is
`data/verification/model-scoreboard-20261029-mechanisms-kimi-family.json`;
the earlier numbers in this document remain historical records.

## Run the report

From the repository root:

```sh
PYTHONPATH=. .venv/bin/python tools/model_scoreboard.py \
  data/verification/model-scoreboard-20261029-final-input.json \
  -o data/verification/model-scoreboard-20261029-final.json \
  --markdown data/verification/model-scoreboard-20261029-final.md
```

The default reads the retained predictions named in the manifest. Add
`--run-predictions --timeout 40 --total-timeout 150` to execute the cases and
probe geometries again. This executes only entries with declared inputs.
V4.1 and Kimi missing-step rows remain visible.

The manifest may also declare `cohort_inputs`, each containing `label` and a
relative `report` path to an existing cohort comparison. These appear in the
same JSON/Markdown scoreboard as selected native-call campaigns. Each report,
rank and capture window stays separate; its native family rows retain complete
call counts, observed/predicted activity, signed gaps and the sum of absolute
descriptor gaps. Pass counts use the common per-call tiers without allowing
family cancellation to hide an outlier. Reference rows remain unscored. This
report-only assembly does not replace full-step predictions or rerun a source
report under a new runtime identity.

The child receives the case, probe geometry and execution budget only. It uses
the common `target_graphs` / `graph_costs` implementation for target-only
probes. The `scheduled_iteration` driver uses the ordinary
`Frontend(IterationPlan(RequestView(...)))` and `Execution.add/add_group`.
It retains draft generation, target verification, sampling, acceptance and
token commit; it does not rename these operations to bypass the intentionally
target-only `graph_costs` boundary. A child failure or timeout remains a failed
prediction in the report. Report assembly preserves measured values.
The distinct `scheduled_prefill` driver uses real prefill RequestViews with
`draft_prepare=True`: it executes the target, sampling/commit and subsequent
MTP preparation through final graph completion. An explicit `draft_layers`
declaration rejects a different catalog draft depth.

Each completed prediction records its input case hash, normalized case hash,
actual geometry, actual parallel configuration, quantized matrix and model
activation formats, hardware geometry, selected hardware profile, runtime
content hashes and live task/program counts. Manifest labels cannot relabel a
TP8/p4096 result as TP4/p256. Source changes during a run are disclosed by
before/after hashes; they do not silently become a clean-commit attribution.

## Retained measurements

| Model and capture | Known observed step | Scope and prediction status |
| --- | ---: | --- |
| Qwen3.5 A3, p16384/o128/c1, TP4, MTP1 | 23.718750 ms median, 64 intervals | Complete verify2 + draft1 iteration at context endpoints 16384/16512. |
| Qwen3.6 A3, p16384/o32/c1, TP4, MTP3 | 34.479625 ms median, 14 intervals | Complete verify4 + draft3 iteration at context endpoints 16384/16416. |
| Qwen3.8 A5, p16384/o32/c1, TP1, MTP3 | 25.661500 ms median, 10 windows | Complete verify4 + draft3 iteration at context endpoints 16384/16416. Only four directly retained measured family medians exist. |
| GLM-5.3 A3, p4096/o1/c1, TP8/EP1, no speculation | 1680.733399 ms, one complete profiled request | Conditional full-call versus target graph comparison, with helper/sampling/gap qualifications below. |
| V4 Flash A3, p4096/o1/c1, TP8/EP8, MTP1 | 575.235407 ms, one complete profiled request | Explicit target43 + BF16 MTP1 preparation, balanced expert routing. |
| V4.1 A5 | Unavailable | No corresponding complete same-checkpoint capture exists in this inventory. V4 Flash cannot replace it. |
| Kimi K3 A5, p32768/o64/c1, TP8/DP4/EP32 | Unavailable | The 6912.686953 ms capture mixes prefill and decode. No independently confirmed per-step boundary exists in the retained inputs. |

Available captures retain cost leaders and their actual statistic and scope.
Qwen3.5 leaders are means of actual per-name sums from a retained shortlist;
the shortlist was selected by count and is not guaranteed to contain the
global duration Top5. Qwen3.6 leaders are directly measured family medians,
not the older count-proportional per-name allocations. GLM and V4 Flash
leaders are original native-type activity sums from the full retained
single-request file.

Qwen3.5's 10.955450 ms `allreduceAicpuKernel` leader is explicitly marked
wrapper/control activity, not physical communication cost. Qwen3.6's
20.398236 ms `collective` family mixes native `hcom_` events with AICPU
all-reduce/all-gather wrappers under the original producer's classifier.
Count-proportional name allocations cannot split those durations. This mixed
family cannot be used as a ring/pairwise link-cost comparator.

V4 Flash has a 575.235407 ms complete p4096/o1/c1 span on TP8/EP8.
Its 44 attention calls comprise 43 target layers plus MTP. It is now a
separately authorized model row; the earlier report retains its original
reference status.

The Kimi native Top5 was computed in one bounded streaming pass over
286,988 events and the 110,031,770-byte source. It took 8.935 seconds and
36.5 MiB peak RSS. Activity totals sum to 9166.957818 ms, exceeding the
6912.686953 ms wall span because activities overlap. Its Top5 therefore
remains a whole-capture list, never a fabricated per-step list.

The durable inventory records original paths, hashes, producers, request
receipts, serving commands, retained step markers and model-geometry checks:

- `data/verification/model-scoreboard-20261029-observed-inventory.json`
- `data/verification/model-scoreboard-20261029-kimi-top.json`
- `data/verification/model-scoreboard-20261029-input.json`

## Final unified-source refresh

The final eight declared graph/iteration scenarios and three Kimi family workers all retain
the same unchanged package-content SHA256:
`e8609fa91badbe7b49d9ab7c36b3fa34adda60a2f1586db2deef73abbddfd18a`.
This identifies 184 package files (166 Python and 18 JSON): root revision `c2368372` plus the
profiler-owned stable working-tree mechanisms. It does not identify a clean
`c2368372` checkout. Each worker also retains its driver hash.

| Workload | Observed step ms | Final prediction ms | Signed gap % |
| --- | ---: | ---: | ---: |
| Qwen3.5 A3, verify2 + draft1 | 23.718750 | 3.468963 .. 3.470165 | -85.374594 .. -85.369529 |
| Qwen3.6 A3, verify4 + draft3 | 34.479625 | 10.764904 .. 10.765423 | -68.778942 .. -68.777437 |
| Qwen3.8 A5, verify4 + draft3 | 25.661500 | 14.716206 .. 14.716829 | -42.652588 .. -42.650162 |
| GLM-5.3 A3, prefill4096 | 1680.733399 | 921.407663 | -45.178238 |
| V4 Flash A3, prefill4096 + MTP1 | 575.235407 | 314.206973 | -45.377672 |

GLM's 78 attention occurrences now cost 477.451653 ms against the observed
1029.573832 ms, a conditional family gap of -552.122179 ms. The historical
31.605411 ms prediction is superseded for this source state, not deleted.
Other large GLM family gaps are grouped matrices -110.815931 ms and dense
matrices -97.537787 ms. V4 Flash's largest conditional family differences
are vector -116.739639 ms, native collective -109.626298 ms and attention
-67.506023 ms. These are activity/isolated comparisons; they cannot be added
as causal contributions to the graph-span residual.

The complete-step campaign took 21.762 seconds; its longest probe took
6.039 seconds. Peak process RSS was 77.828 MiB. All eight graphs finished
with zero live tasks and programs; isolated evaluation also left no live
tasks or allocations. V4 Flash retains exactly 44 attention occurrences,
43 target plus one MTP. `model-scoreboard-20261029-final-execution.json`
contains the receipts and `model-scoreboard-20261029-final-audit.json`
contains the independent consistency check.

Kimi's final references explicitly select `steady_matmul` for generic
constituent and one-group matrices. The unknown-histogram 28-group domain
retains its resource envelope. The fresh phase run remains a separate
control in `model-scoreboard-20261029-kimi-final-family.json`; the primary
new result is `model-scoreboard-20261029-kimi-steady-final-family.json`.

| Kimi family | Calls | Observed cumulative ms | Final predicted cumulative ms | Signed gap ms / % |
| --- | ---: | ---: | ---: | ---: |
| MlaPrologV3 | 1798 | 40.001485 | 33.173014 | -6.828471 / -17.070545 |
| GroupedMatmul | 27248 | 813.480705 | 1057.145272 | +243.664567 / +29.953331 |
| ChunkKdaFwd | 1104 native launches | 881.997838 | Unavailable | Unavailable |

GroupedMatmul's sum of absolute descriptor residuals is 423.848682 ms,
larger than its 243.664567 ms absolute aggregate difference. Its six
one-group geometries total 507.998379 observed versus 691.543403 predicted
ms; the 361 unknown-histogram geometries total 305.482326 versus 365.601869
ms. The 8192x7168x4608 one-group FP8 case is still +126.66%, while the
single-token 7168x3584 case is -48.23%. This spread remains a mapping and
scope diagnostic; a shared multiplier would obscure it.

The complete KDA semantic reference remains 0.219952078 ms. Its unbound
native launch stage prevents multiplying it by 1104. Kimi still has no
recovered complete step; V4.1 still has no same-checkpoint capture. The
scoreboard exposes those two gaps without hiding the useful family pairs.

## Historical phase full-call comparison

The independent GLM request receipt, profiling script and original device CSV
establish one p4096/o1/c1 request with 2392 positive device events. The serving
log establishes TP8/PP1/DP1, BF16 model activations and no speculation.
The raw typed rows establish INT8 dense matrix inputs/weights, INT4 routed
weights and BF16 attention/cache tensors. Fourteen independent model geometry
checks match the declared case. FFN/expert dimensions, detailed scale
granularity, normalization and routing parameters remain disclosed catalog
assumptions.

The prediction uses an explicit balanced integer expert allocation and
`phase_matmul`. It does not reconstruct the observed expert histogram or
vendor kernel mapping. The selected `Ascend910_9362` resource profile is an
assumption: the retained hardware log independently confirms A3, eight visible
64 GiB dies and two dies per package, but prints `Ascend910` without that
precise SKU suffix.

The observed 1680.733399 ms spans the first positive device task to the last
task in the one-request capture. It includes runtime helpers, sampling and
idle gaps. Sim's `device_graph` spans target-model work and excludes sampling
and token commit. Only one TP rank was retained, so observed cross-rank skew
is not reconstructed. These differences are explicit conditions on a useful
descriptive gap; absent binary provenance does not prohibit the comparison.

The first phase result is 786.527760 ms, a signed gap of -894.205639 ms
(-53.203300%) and absolute gap of 894.205639 ms (53.203300%). It is preserved
as a mechanism result, not corrected with a multiplier. The prediction is
stored separately from the report in
`model-scoreboard-20261029-glm53-phase-prediction.json`.

The whole-span difference does not identify its causes by itself. Existing
cost and occurrence ledgers expose two concrete next checks:

- The observed file has 157 `hcom_allReduce_` activities. The modeled graph has
  231 all-reduce invocations: 78 attention output reductions, 78 dense/shared
  reductions and 75 expert tensor reductions. The latter operate on top-k
  expanded rows before unpermutation; shared expert output is also reduced
  separately. These are raw native event counts, not proven unique logical
  calls. The source audit of the capture-declared vllm-ascend version
  `8fd0bc4885cb52e58791aaa8efacc571b9bd115d` supports one final token-row
  reduction after unpermutation and shared combination for the ordinary
  TP-only, non-SP path without a routed output transform. This source
  mechanism is independent of the timing discrepancy. The official source
  and blob identities are retained in
  `model-scoreboard-20261029-glm-moe-source.json`; the installed binary is
  not independently proven to equal that source.
- The observed file has 78 `SparseFlashAttention` activities totaling
  1029.573832 ms; the modeled 78 `latent_attention0` isolated calls total
  31.605411 ms. The report declares a coarse semantic-family comparison:
  signed gap -997.968421 ms, absolute gap 997.968421 ms, or 96.930243%.
  The same step and 78 logical occurrences make this useful for mechanism
  prioritization. Equal counts do not establish equal physical/native scope
  or tensor-exact accuracy. Precision, sparse selection, execution boundary
  and resource behavior remain explicit qualifications.

The communication audit reads the original event identities and keeps native
`COMMUNICATION` and `AI_CPU` domains separate. Native 157 all-reduce plus one
all-gather rows sum to 93.688830 ms and have the same interval union. AICPU
activities sum/union to 1552.326106 ms. Their time intersection is 86.319197 ms;
the union across both domains is 1559.695739 ms. Direct addition counts the
intersection twice and does not measure physical communication cost.

No complete event identity duplicate was found, and no original row was
removed. Every native hcom row lacks stream/task IDs, so there is no reliable
cross-domain operation ID for deduplicating calls. Most native intervals
overlap two AICPU activities, preventing a nearest-time or same-name join.
Wait columns are retained separately and never added again to duration.
The report embeds the audit and references
`model-scoreboard-20261029-glm-communication.json`.

## Historical phase speculative iterations

The three Qwen predictions execute the full fixed-proposal iteration.
Actual acceptance controls request progress; it does not discount target
verification or the declared proposal work in these isolated iterations.
The independently retained request lengths supply conservative context
endpoints. Each endpoint is run separately, with no fabricated median context.

| Model | Measured step median ms | Predicted endpoint spans ms | Signed gap at endpoints % |
| --- | ---: | ---: | ---: |
| Qwen3.5 A3 | 23.718750 | 3.472523 .. 3.473724 | -85.359588 .. -85.354522 |
| Qwen3.6 A3 | 34.479625 | 10.889103 .. 10.889622 | -68.418731 .. -68.417226 |
| Qwen3.8 A5 | 25.661500 | 18.975652 .. 18.976275 | -26.054002 .. -26.051576 |

These are two explicit context scenarios, not a mathematically proven bound
on all interior context timings. Deterministic scenario spans are compared
to observed medians with this statistic difference disclosed. The native
sampler-to-sampler window also contains host submission and runtime gaps,
which are outside the simulated device iteration. Catalog/draft dimensions,
quantization implementation, short-decode probability routing and declared
hardware rates remain analytic assumptions. Neither missing binary identity
nor unknown exact context disables the simulations.

The first Qwen3.5 endpoint pair has different package content hashes, although
each run's before/after hash is unchanged. Reporter development itself enters
the whole-package hash. The report retains both identities and does not
attribute the small endpoint spread solely to context.

Every endpoint retains actual Top5 isolated cost families, executed frontend
identity, plans, recipe/source metadata and zero remaining tasks/programs.
Five endpoint workers completed in 4.52 seconds total after the first
0.74-second Qwen3.5 run; each had a 40-second budget. These substantial
differences remain mechanism evidence, not values to repair by calibration
factors.

## Comparison contract

The required step conditions are model, hardware geometry, parallel layout,
phase, tokens, context, sequences, precision and executed branch. Nested null
conditions remain unknown. Different known conditions are incomparable.
Different timing boundaries can be compared only through an explicit named
boundary relation with assumptions. A successful qualified comparison is
`conditional`; independent heldout validation remains `not_assessed`.

Signed gap is `prediction - observation`, and absolute gap is its magnitude.
Percentages use the observed value as denominator. Missing comparisons do
not contribute zeros. The report retains signed and absolute residual sums
separately, without ranking models.

Cost leaders are side-by-side evidence. Names alone do not pair groups.
Tensor-exact group comparisons require equal homogeneous identity, timing
boundary, statistic, scope and occurrence count. A separately declared
semantic-family association can compare the same step and occurrence count
across qualified activity/isolated boundaries; it explicitly disclaims
tensor-exact native work and remains conditional. A sum of ten calls cannot
be compared to a one-call mean. Native activity sums and isolated model costs
are not additive decompositions of wall span.
Comparable family gaps are ordered by absolute residual, with incomparable
diagnostics afterward.

The new `family_bindings` diagnostic level explicitly associates observed
family statistics with a list of predicted recipes. It permits different
native/program invocation counts and records all members, statistics and
boundary assumptions. Qwen3.5/3.6 coarse rows use sums of class means over
the same clean-step population, which equal the mean of the summed activity;
they do not add class medians and call the result a median. Qwen3.8 retains
the directly available family medians. These coarse associations do not
replace stricter tensor or occurrence matching.

Qwen predicted Top5 now displays endpoint ranges and scenario coverage.
A missing endpoint remains visibly partial, including in cost-leader tables.
Mixed wrapper/native communication and unidentified `other` activity keep
their measured and predicted values but do not receive a physical-service
residual.

## Historical steady mapping controls

`model-scoreboard-20261029-steady.json` retains clean runtime probes at
`11fc9acf`, with the GLM node-identity correction and V4 Flash at `9e65140b`.
Every raw worker retains its package hash. The prior live phase cases
normalize to different case identities, so their total changes cannot be
attributed solely to matrix scheduling.

| Workload | Observed step ms | Steady prediction ms |
| --- | ---: | ---: |
| Qwen3.5 A3, verify2 + draft1 | 23.718750 | 3.461431 .. 3.462632 |
| Qwen3.6 A3, verify4 + draft3 | 34.479625 | 10.764329 .. 10.764848 |
| Qwen3.8 A5, verify4 + draft3 | 25.661500 | 14.715726 .. 14.716348 |
| GLM-5.3 A3, prefill4096 | 1680.733399 | 566.324506 |
| V4 Flash A3, prefill4096 + MTP1 | 575.235407 | 220.748207 |

Clean GLM controls separate two changes: phase mapping with the original
reduction boundary gives 904.985751 ms; the explicit `tp_join_reduce`
boundary gives 632.726923 ms; retaining that boundary and choosing steady
mapping gives 566.324506 ms. This algebraic reduction correction follows the
caller source, not a fit to 157 native communication events. A duplicate
attention/MoE node name initially failed both joined probes; the failure
receipts remain beside the corrected results.

The first V4 probability-routing run timed out at 40 seconds. Explicit
balanced routing supports the equal-group mapping and completes in 3.34
seconds with zero live tasks/programs. The catalog auxiliary declares three
DSpark layers and produced 46 attention calls, so that 232.993985 ms result
is retained as mismatched work. The actual paired case overrides the
auxiliary with one BF16 MTP layer, no Markov/confidence branch and causal
queries; its ledger has exactly 44 attention calls. Generic MTP feature
projection still approximates the native hyper-connection input path.
Neither equality of call count nor this explicit declaration proves exact
native fusion or checkpoint-binary identity.

## Historical Kimi phase references

`model-scoreboard-20261029-kimi-family.json` retains 369 descriptor groups,
including 367 GroupedMatmul geometries. Operand shapes and dtypes condition
independent analytic scenarios; measured durations enter only comparisons.
The prediction source includes concurrent profiler mechanisms, so its
per-mode live package hashes are reported separately from the clean
full-step runtime.

| Family | Actual calls | Observed cumulative ms | Conditional predicted cumulative ms | Signed / absolute gap ms |
| --- | ---: | ---: | ---: | ---: |
| MlaPrologV3 | 1798 | 40.001485 | 62.947310 | +22.945825 / 22.945825 |
| GroupedMatmul | 27248 | 813.480705 | 1718.010049 | +904.529344 / 904.529344 |
| ChunkKdaFwd | 1104 launches | 881.997838 | Unavailable | Unavailable |

These are mixed-capture activity populations, not model-step spans.
MLA's observed mean/P50 is 0.022247767/0.0222165 ms; its descriptor-conditioned
native resource DAG predicts 0.035009628 ms. GroupedMatmul retains each
descriptor's mean, P50 and prediction: the single-token FP8 7168-to-3584
case is about +1.71%, while an 8192-by-7168-by-4608 FP8 prefill case is
about +344.23%. Its one-group reference uses phase mapping; the 28-group
reference uses an aggregate resource envelope because group-list values
are absent. This difference points to mapping/overlap investigation; it
does not justify a common correction multiplier.
The grouped sum of absolute residuals across descriptor populations is
906.509810 ms, separately retained from the absolute aggregate difference
904.529344 ms, so opposite signed errors cannot disappear in a total.

The complete KDA semantic core predicts 0.219952078 ms with no live state
left. All 1104 native records share descriptors but do not identify their
launch stage. The complete-core prediction is therefore displayed without
multiplying it by 1104 or subtracting it from a per-launch mean.
Strict cohort mapping remains separately unpriced; that does not suppress
the explicitly conditioned MLA and matrix comparisons.

The durable reproduction tool separates observation extraction, analytic
workers and comparison assembly:

```sh
PYTHONPATH=.:/path/to/profiler .venv/bin/python tools/model_scoreboard_kimi.py \
  observe /path/to/kimi-op-summary.csv -o /tmp/kimi-observed.json
PYTHONPATH=. .venv/bin/python tools/model_scoreboard_kimi.py predict \
  --observed /tmp/kimi-observed.json --case examples/cases/kimi-k3-a5.json \
  --output-dir /tmp/kimi-predictions --matrix-recipe steady_matmul
PYTHONPATH=. .venv/bin/python tools/model_scoreboard_kimi.py assemble \
  --observed /tmp/kimi-observed.json \
  --predictions /tmp/kimi-predictions/prediction-mla.json \
    /tmp/kimi-predictions/prediction-grouped.json \
    /tmp/kimi-predictions/prediction-kda.json \
  -o /tmp/kimi-comparison.json
```

Prediction workers receive tensor descriptors and case declarations without
measured durations or occurrence counts. Assembly retains each original
prediction producer and identifies its own source separately; using the
new tool to read historical predictions does not rebrand them as fresh runs.
The source must contain the optional native family mechanisms used by those
declarations; unavailable implementations fail visibly.
New runs default to `steady_matmul`; use `--matrix-recipe phase_matmul`
only for an explicit control. The selected recipe travels in the worker
packet and assembled source metadata. Native source recipes retain precedence.

## Verification

The 15 focused tests exercise the actual short public graph worker, a real
complete Qwen speculative iteration, retained
GLM gap arithmetic, mixed-capture and speculative-branch exclusions, nested
unknowns, sum/mean/count mismatches, prediction relabeling, source-case identity
context endpoint comparisons, coarse family ranges with mixed wrapper
exclusion, partial-endpoint display, per-context boundary/source qualifiers
and real MTP prefill preparation. A captured one-group FP8 geometry also
checks that steady and phase service identical physical work while retaining
their different schedules; changing measured timing/count fields leaves
the worker packet unchanged. All 15 passed in 0.72 seconds.
The actual CLI produces seven distinct model rows: five full-window
comparisons, Kimi family comparisons with no invented step, and the
unavailable V4.1 capture.

The retained phase worker completed in 2.08 seconds wall time with
74.047 MiB peak RSS and no remaining tasks or programs. Its runtime source
identity is the package content hash recorded in the prediction, not a claim
that concurrent working-tree sources correspond to a clean commit. No
device workload, calibration fit, old-version deletion or formal accuracy
claim is part of this scoreboard run.
