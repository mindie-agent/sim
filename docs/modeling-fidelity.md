# Modeling fidelity and refinement policy

The objective is useful performance prediction across shapes, models, hardware
and execution configurations. Reimplementing each kernel, compiler and runtime
is not the objective. Nonzero timing error is expected. More detailed simulation
is justified by a better prediction or decision, not by a longer source ledger.

This policy follows the architecture review on 2026-10-28. Existing detailed
programs remain available; no prediction is silently changed to a cheaper model.
The next operator tasks should start from the simplest sufficient abstraction.

## What the industry implementations actually do

These are observations of pinned official documentation and selected source
files, not results from running their simulators or validating their accuracy.

| Implementation | Observed abstraction | Useful lesson for this simulator |
| --- | --- | --- |
| LLMCompass | Tensor/operator graphs, hardware parameters, optional roofline analysis and explicit GEMM mappings. Its tile simulator composes memory and compute using reuse, loop order and double buffering. | Preserve the physical effects of tiling and data movement without reconstructing a vendor compiler. A mapping search is a design-space result, not proof that a deployed kernel chooses it. |
| Microsoft Vidur | Separate compute, attention, network and CPU profiling. The execution-time predictor filters structural configuration and learns timings, including a random-forest implementation. Request simulation consumes the estimates. | Separate timing estimation from serving simulation. Reuse a small profiling domain instead of making every prediction a hardware experiment. Its model-partitioned data and implementation-specific overheads are not a cross-hardware analytic model to copy unchanged. |
| NVIDIA AISimulate | Separate forward-pass estimation, request Replay and configuration Sweeper. Op-level estimation composes collected/interpolated timings or supported analytical formulas; whole-forward and online-regression estimators have distinct scopes. | Expose the timing basis and its applicable domain. Keep graph composition, operator estimation and serving constraints separate. A forward estimate alone does not give TTFT/TPOT. |

AISimulate is the current development home named by AIConfigurator. The review
uses source at `17625dc8e6b0ef12b6da43fbf08f19cfd6e8b77e`, not a claim about an
installed or published wheel. Its empirical operator path can transfer
SOL-relative utilization between shapes/precisions. We do not adopt that
effective multiplier as a physical resource rate, nor import whole-forward
regression into the operator model.

Sources inspected:

- LLMCompass `2e015fd2ee750e6cad8d7152df52551e5b41ef20`:
  [guide](https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/docs/run.md),
  [matmul source](https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/software_model/matmul.py)
  (`roofline_model`, `Mapping`, tile read/compute/write composition).
- Vidur `abae7f63aa857300f5cdc6f5e0d27860cd24721b`:
  [profiling](https://github.com/microsoft/vidur/blob/abae7f63aa857300f5cdc6f5e0d27860cd24721b/docs/profiling.md),
  [feature/configuration selection](https://github.com/microsoft/vidur/blob/abae7f63aa857300f5cdc6f5e0d27860cd24721b/vidur/execution_time_predictor/sklearn_execution_time_predictor.py),
  [random forest](https://github.com/microsoft/vidur/blob/abae7f63aa857300f5cdc6f5e0d27860cd24721b/vidur/execution_time_predictor/random_forrest_execution_time_predictor.py).
- AISimulate `17625dc8e6b0ef12b6da43fbf08f19cfd6e8b77e`:
  [performance methods](https://github.com/ai-dynamo/aisimulate/blob/17625dc8e6b0ef12b6da43fbf08f19cfd6e8b77e/docs/perf-model/README.md),
  [op-level composition](https://github.com/ai-dynamo/aisimulate/blob/17625dc8e6b0ef12b6da43fbf08f19cfd6e8b77e/docs/perf-model/methods/op-level.md),
  [empirical implementation](https://github.com/ai-dynamo/aisimulate/blob/17625dc8e6b0ef12b6da43fbf08f19cfd6e8b77e/crates/core/src/perfmodel/operators/util_empirical.rs),
  [execution limitations](https://github.com/ai-dynamo/aisimulate/blob/17625dc8e6b0ef12b6da43fbf08f19cfd6e8b77e/docs/perf-model/limitations.md).

## Separate correctness, timing detail and calibration

Correctness concerns the specified computation and execution structure: tensor
shape/layout/precision, executed branch, materialized copies, recurrence,
kernel/fusion boundaries, ownership, capacity and communication dependencies.
An approximation cannot create free copies, execute before input arrival or
borrow idle cores for pinned work. Those are bugs even when a total looks close.

Timing detail is a choice within that structure:

| Level | Representation | When to use |
| --- | --- | --- |
| Resource envelope | Arithmetic and traffic served by declared resources, with explicit setup where applicable | Simple independent work, a first estimate, or an inexpensive screening baseline. It does not imply native kernel fidelity. |
| Phase/tile model | A small sequence/DAG of load, compute, reduction, exchange and store; tile tails, reuse and participating units where they matter | Preferred working model when overlap, limited parallelism, reduction or state movement changes trends. Reuse existing program classes. |
| Detailed local program | Instruction-class demand, finite queues or finer dependency stages | A diagnosed hotspot whose simpler model misses a material cost, bottleneck transition or configuration ordering. Keep it local. |

These are modeling choices, not new engines or a mandatory ladder. A
source-derived planner can emit a compact phase or envelope program while
retaining its actual call boundary, declared work and implementation identity.
`plan_required` means a selected implementation must bind a planner; it does
not mean that planner must enumerate instructions.

Calibration is independent of that choice. A phase model can use analytical
service rates or calibrated service rates. A detailed program can still have
uncertain rates. Missing measurements do not disable an implemented analytic
path, and a measured duration does not make a wrong dependency graph correct.
Measurement provenance belongs to calibration/comparison results, not to an
evidence-admission gate inside simulation.

## Review of the current architecture

Keep the common engine, semantic graph, hardware profiles and recipe registry.
Their separation supports NPU and GPU models without hardware-name branches.
Keep real source selection and operator boundaries. The per-owner supply and
input-readiness repairs fix structural errors, regardless of timing tolerance.

Change the default development pressure: a source line is evidence for a
mechanism, not a task to reproduce that line's instruction sequence. Shared
phase/tile primitives should carry most operators. Existing fine GDN programs
can be diagnostic references while reusable family models are evaluated.
Do not replace a selected implementation with a generic envelope merely because
it produces a closer number.

The current calibration interface also has limits:

- Exact `PrimitiveStore` timings are anchors for an identical declared variant.
  They do not provide interpolation across a shape family.
- The completion constraint currently takes the slower of physical service and
  primitive timing. It can lengthen an underestimate; it cannot repair an
  overestimated physical program or identify its resource occupancy.
- `fit-rate` identifies nonnegative setup plus work/rate for a measured transfer
  primitive. `fit-service` now provides a one-dimensional work-response curve for
  one declared resource invocation. It retains work and limits invocation supply
  on the common engine. Categorical conditions, units, participating owners and
  device/phase boundaries must match; domain misses follow the declared
  analytical/error policy. See `resource-services.md`.
- The first replay uses existing 4090 copy observations, with one heldout size.
  It does not establish arbitrary shape/stride/owner surfaces or GPU compute
  accuracy. Multidimensional surfaces and automatic dispatch selection remain
  future extensions, justified by the target operator families.

Fit only parameters identified by isolated training experiments; reserve shapes
and runs for validation. Separate categorical dispatch regions before numerical
interpolation. Never infer memory/compute occupancy from one opaque kernel
duration. Primitive service models are independent of operator timing detail.

## Bottom-up implementation order

These are composition layers, not an instruction-fidelity ladder. Reuse the
existing source planners and observations as each layer is checked.

| Layer | Deliverable and acceptance question | Current boundary / next work |
| --- | --- | --- |
| Physical services | Work units, supply, startup, owners and shared capacity. Can untouched work amounts be predicted without deleting traffic or inventing concurrency? | Common engine and first work-response curve implemented; retained copy replay plus independent contention/phase controls. Compute, reduction and other access domains still need their own service descriptions. |
| Operator families | Compact resource/phase/tile plans with real dispatch and kernel boundaries. Are dominant-family trends and repeated error useful within the declared domain? | Existing GEMM, attention, normalization, quantization, GDN and MoE planners remain. GQA, RMSNorm, residual RMSNorm and integer-routed grouped GEMM now bind through the ordinary frontend; see `operator-family-selection.md` for exact domains. Broader shape trends and hardware comparisons remain open. |
| Blocks and subgraphs | Compose projections, data movement, reductions, state and communication. Are producer/consumer layouts, fusion and stream dependencies correct? | `graph_costs` composes declared graphs, streams, external inputs and complete expert cohorts through the serving engine. Structural checks cover phase/cap preservation and real frontend boundaries; broader block timing validation remains open. |
| Pure model | Account for all operator occurrences and accumulated cost before request scheduling. Are cost concentration, unpaired families and repeated residuals explicit? | `graph --scope model` separates graph span, isolated sum and precedence/resource lower bounds. Capture scope and media geometry are explicit. Match remaining implementation domains and measurements; neither graph completion nor a bound establishes measured accuracy. |
| Serving constraints | Add arrivals, batching, cache, parallel placement and host work to evaluate feasibility under TTFT/TPOT constraints. | Existing serving engine is retained. Return to matched request/capture scenarios after lower-layer costs are useful; serving totals do not train operator parameters. |

Progress through these layers with declared approximations. Do not wait for zero
timing error on every primitive, and do not claim higher-layer accuracy from
lower-layer contract tests. Old implementations and raw measurements remain
preserved until the replacement has absorbed their mechanisms and data.

The public [study workflow](studies.md) now composes these layers across explicit
candidates and probes. It reports predicted cost concentration and comparable
configuration ordering, with optional original-case serving constraints.
Independent timing validation remains a separate command and report.

## Refine according to impact and decision stability

For a paired operator with modeled occurrence count `n`, use:

```text
predicted contribution = n * predicted duration
signed residual       = n * (predicted duration - observed median)
absolute residual     = n * abs(predicted duration - observed median)
```

Show both signed and absolute totals. Cancellation is not accuracy. Normalize
against a stated, matching scope. A fraction of the predicted isolated sum is a
priority diagnostic, not measured model error or a confidence bound. Multiple
captured signatures or predicted branches with one tensor contract are candidate
pairs, not independently summable observations. Unpaired costs stay unscored.

The retained 16K prefill prediction has 1,956 occurrences, 48 physical groups
and an isolated sum of 1544.207 ms. Its five largest family/recipe contributions
account for 77.607%: gate/up projection, QKVZ projection, down projection,
attention and linear output. Predicted share indicates where inspection can
matter; it does not establish that those operators are inaccurate.

| Native family | Kernel error | Signed residual across 48 modeled calls | Absolute residual / full predicted isolated sum |
| --- | ---: | ---: | ---: |
| FN convolution | +11.144% | +1.799 ms | 0.116% |
| Prepare | -20.256% | -14.983 ms | 0.970% |
| H | -15.752% | -6.131 ms | 0.397% |
| O | -30.421% | -7.486 ms | 0.485% |

The four conditional residuals sum to -26.802 ms signed and 30.399 ms absolute,
or 1.736% and 1.969% of that full predicted isolated sum. This calculation
projects four ABI-matched kernel medians onto modeled repetitions. Unknown
sequence values/dispatch flags and unmatched operators remain. It is not a
measured full-model error, an upper bound or critical-path sensitivity.
The comparison report's own denominator is only the four-family 126.839 ms;
the table above explicitly uses the separate full-model reference.

Do not keep refining FN solely to remove its 11% error. First ask whether a
simpler model predicts heldout shape trends, repeated cost and candidate
ordering adequately. Conversely, many individually small kernels can dominate
decode overhead; evaluate aggregated impact separately for prefill, decode and
the intended batch/context domain.

Prioritize work in this order:

1. Fix wrong structure or units, including exercised dispatch and ownership.
2. Inspect dominant family costs and large repeated residuals across the target
   workloads. Include communication and frequently repeated small kernels.
3. Check heldout shape/batch trends and configuration ranking. If plausible
   parameter variation reverses a decision, investigate that mechanism or
   benchmark the few competing configurations.
4. Add detail only with a discriminating counterexample and an expected effect.
   Compare its accuracy/trend benefit against simulation time and maintenance.
   Stop when the simpler abstraction meets the decision's stated tolerance.

There is no universal per-kernel zero-error target. Timing acceptance should
state the workload domain, boundary and useful tolerance, alongside trend and
ranking behavior. Structural tests and data coverage remain separate from that
evaluation. Whole-model measurement, when boundary-matched, checks accumulated
error; it never tunes operator parameters. TTFT/TPOT remain downstream serving
metrics used as deployment constraints.

## Reproduce the priority diagnostic

```sh
PYTHONPATH=. .venv/bin/python tools/modeling_priority.py \
  data/verification/qwen38-native-20261028-model-reviewed-rolling.json \
  data/verification/gdn-native-20261028-reviewed-rolling.json \
  -o data/verification/modeling-priority-20261028.json
```

This reads existing reports; it does not rerun simulation, calibrate parameters,
combine different report denominators or claim hardware acceptance. The output
keeps input hashes, ranked repeated costs, candidate matching limits, separate
signed/absolute residuals and unpaired predicted cost. No DSH wave or hardware
experiment is launched by this diagnostic.
