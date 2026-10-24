# Fine-grained operator execution plans

An execution plan describes a finite device-side implementation of one
operator: stages, dependencies, resource work, buffer requirements and the
origin of the estimate. It feeds the existing shared-resource event engine.
It is an opt-in alternative to the aggregate analytic cost of an ordinary
operator; existing predictions retain their previous path when it is disabled.

This layer is independent of a model catalog or a parallel strategy. The
examples use explicit synthetic hardware parameters and small operators. They
are arithmetic and scheduling demonstrations, not device measurements.

## Running an explicit operator DAG

```bash
python -m mindie_sim operator-plan --input examples/operator-plans/mixed.json --output /tmp/operator-plans-mixed.json
```

The input contains `machine`, a topologically ordered `operators` list, and an
`estimators` object. Operators use the existing `OpSpec` fields. `context` can
declare compute, activation and weight dtype plus timing scope. The estimator
configuration accepts `vector_profile`, `gemm_profile`, `cannbot_repo`,
`cannbot_revision` and `strict`. Profiles are explicit; passing
`vector_profile: true` selects the built-in, assumed RegBase scenario.

Declare finite, nonnegative baseline work (`flops`, `vector_ops` and/or
transfer bytes) for executable operators. Missing work is rejected rather than
silently falling back to a launch-only estimate. The optional context also
carries quantization; the BF16 dense GEMM provider refuses quantized contexts.

Each output operator retains its analytic baseline and adds
`operator_estimation`, including the selection or fallback reason. A selected
`execution_plan` contains stages, resource totals, provenance and its buffer
certificate. The report's `execution.trace` is the actual shared-resource
timeline; an operator's `isolated_plan_ms` is not a promise that the same plan
finishes that quickly when it competes with other work.

The ordinary prediction CLI accepts a separate estimator configuration:

```bash
python -m mindie_sim predict --input examples/a3_dense_predict.json --operator-config /path/to/estimators.json --output /tmp/prediction.json
```

Here `/path/to/estimators.json` contains only the `estimators` object's fields,
not the entire operator-DAG example. Profiles must match the prediction's
hardware and dtype. Omit `--operator-config` to preserve the previous path.

The equivalent Python entry is `PredictiveCostModel(machine,
op_estimator=create_operator_estimator(config))`; the factory is in
`mindie_sim.operator_estimation`. Custom providers implement
`estimate(op, machine, context) -> KernelEstimate`. The plan contract is in
`mindie_sim.operator_plan`.

## Accounting and timing

Each resource demand carries a unit. Compute work, transfer bytes, instruction
service and elapsed time are distinct quantities. The plan's resource summary
is an audit record derived from its stages, not another task to execute.

Only one device accounting path is active for an operator: its selected plan,
its existing calibrated cost or its original analytic fallback. A plan does
not execute alongside the aggregate parent work. The operator's completion
joins all terminal stages, so a dependent operator cannot start when only one
branch has finished.

`minimum_duration_ms` is a concurrent completion floor, not a resource rate
limit. Stages with a known per-core or local-pipe limit declare
`rate_limits_per_ms`; these use the engine's fixed reservations to constrain
actual service and retain resource pressure while other operators execute.
Demand units and rate units are work and work/ms respectively. The shared
physical capacity remains an upper bound on every declared rate.

Device launch and host submission have separate ownership. This feature models
device service and preserves the existing device launch term once. It does
not add scheduler, host submission, output processing or client transport
delays. Those require a separate framework task model.

Whole-kernel calibration is not apportioned into invented measured stage
durations. Existing calibration takes precedence over a new analytic plan.
Baseline and selected-plan records remain available for comparison, including
fallback reasons.

## Vector instruction streams

A Vector recipe explicitly declares its instruction stream, instruction
dependencies and architecture profile. The estimator accounts for shared
instruction issue, execution-pipe service and dependency latency. Independent
chains may overlap; dependent instructions wait for their inputs.

Instruction throughput, issue width, frequency, core count and transfer rates
are inputs with provenance. They are not inferred from a model name. A
synthetic profile demonstrates the algorithm but does not establish the
throughput of an Ascend SKU. Tail elements and core assignment must be
accounted for explicitly. Issue, SFU and dependency constraints are per-kernel
service floors. They are not separate globally shared instruction-port
resources: different kernels compete through the existing aggregate Vector
resource, without tracing cross-kernel instruction issue or SFU occupancy.

An optional CANNBot adapter can use a separately installed source checkout.
Normal imports and the default prediction path do not require CANNBot, torch,
torch_npu or an NPU. An adapter result records the backend version and profile.
If the external backend exposes only a whole-kernel duration, that duration is
reported as such; it is not expanded into an invented instruction timeline.
The native adapter accepts the `chip_peak` and `vector_core_conservative`
profiles. L2-aware cases and explicit L2 work are refused because this plan's
resource aliases do not include L2; the adapter does not relabel L2 bytes as
HBM traffic. Hardware variant and pinned source revision must match explicitly.
The target stage must also match the native profile's AIV count and HBM peak,
declare its native `vector_frequency_hz` in machine metadata, and have
sufficient MTE bandwidth. Native tiling requires matching UB capacity. Element
cardinality is checked against the operator; the caller still owns the full
semantic correspondence between the operator and its declared instruction case.

## Finite blocked GEMM

The initial Cube implementation covers explicit blocked BF16 GEMM. The caller
provides tile sizes and buffering. The planner checks L1, L0A, L0B and L0C
requirements before generating finite MTE2, MTE1, MMAD and output-store work.
Recognized Ascend A2/A3/A5 profiles require BM/BN/BK multiples of 16 for this
declared BF16 program. Smaller normalized test fragments are mathematical
fixtures, not evidence of legal Ascend MMAD instructions. Passing these shape
and capacity checks does not certify a compiler kernel implementation.

K loops, their final partial block, output tiles and core allocation are finite.
Buffer reuse depends on the last consumer of the old contents. This captures
startup, drain and slot-reuse dependencies that an aggregate Roofline or
steady-state initiation interval alone cannot express.

HBM traffic and local transfers have separate accounting. This initial plan
uses an explicit no-cache-credit scenario: each output tile's input requests
are charged to HBM, while the declared per-core L2 rate also constrains each
load. It does not claim compulsory-byte traffic or predict actual cache misses.
Adding cache reuse requires a separate declared cache policy rather than an
implicit hit-rate discount. All hierarchy capacities and rates are explicit
profile inputs.

This first implementation does not establish coverage for INT8, packed W4A8,
ragged grouped GEMM, automatic tile search or arbitrary layouts. Those require
their own dtype, metadata and work-conservation contracts rather than a
throughput multiplier applied to BF16.

## Captured kernel costs

A captured-kernel binding reuses one exact variant of an existing profiler
inventory (`mindie-profiler kernels`) as the completion floor of one operator.
The `profiled_kernels` configuration holds the inventory `db`, the
`environment_key` the caller accepts, and `bindings`. A binding names the
lowering it stands for (`family`, `recipe`, `phase`, `launch_shape`, `dtype`,
`layout`), the `variant_id` it resolves, and its own `timing_scope`, which must
be `profiled_device_kernel`. The captured scope belongs to the binding because
one run mixes captured kernels with operators priced by the run's own
measurement protocol; a run-wide captured scope would exclude every observation
that declares a real protocol, so `--timing-scope` still names only the run's
measurement protocols. An operator opts in with `metadata.profile_variant`; a
binding alone resolves only when the lowering key matches exactly and
unambiguously.

The provider reads the inventory with `mode=ro` and no `immutable` flag. That
inventory is incremental, and `immutable=1` returns the pre-WAL snapshot — a
superseded median, or no row for a freshly ingested variant — without reporting
anything. Hardware key, environment key, variant eligibility and the lowering
key must all match. Any mismatch is an unsupported reason, and an operator that
explicitly named a variant turns that reason into an error rather than a
different price.

### What a capture prices

The captured median is the stage's `minimum_duration_ms`: a concurrent
completion floor under the same rule as every other plan here. It is not a
second service interval added to the analytic cost, and the existing device
launch term is still charged once.

A capture records duration, not occupancy. The stage therefore charges the
operator's own declared work (`flops`, `vector_ops`, `hbm_bytes`, `mte2_bytes`,
`mte3_bytes`) as its shared-resource demand, recorded as `occupancy_basis:
declared_work_proxy` beside `resource_occupancy: not_captured`. This is a
declared-work proxy, never a measured occupancy; the capture cannot confirm it
and this repo does not report it as measured. Its purpose is that a captured
kernel is not a free task every concurrent operator overlaps at no cost. An
operator bound to a capture still declares its own work, and a captured
operator with no declared work has no occupancy to schedule and is refused like
any other operator without baseline work.

Status versus target: the proxy is only as informative as the declaration. When
a captured duration is dominated by something the declared work does not
express — launch behaviour, low block occupancy, intra-kernel synchronisation —
the floor dominates and concurrent operators still overlap it almost freely.
Charging whole-Die exclusivity instead would invent an occupancy the capture
never measured, so the gap stays visible rather than filled. Closing it needs
per-kernel occupancy in the capture itself (block dim and active-core
residency), not different arithmetic in this layer.

### Precedence and evidence

A dedicated measured calibration and the existing communication topology model
keep precedence over a captured binding, as they do over an analytic plan. The
operator's record then reports `superseded_profile_variant`, so an explicit
request is visible as superseded instead of disappearing. Declaring a capture
for an operator the same caller declared non-executing, already stage-planned
or explicitly serial is one self-contradictory declaration and is refused.

The calibration record keeps this repo's existing vocabulary.
`observations_used` is the number of captured calls the inventory aggregated —
the same quantity a measured served row reports as its sample count — and
`inside_envelope` is `True` because an exact variant match lies on the measured
support rather than outside it; neither field has meant runtime variance. The
provenance carries what those fields do not: `captured_calls`,
`captured_min_ns`, `captured_max_ns`, `captured_median_ns`,
`aggregated_observation_rows`, and `independent_trials`, which stays `None`
because a capture does not record whether repeated calls are independent
trials. `independent_accuracy_acceptance` stays `False`.

### Held-out evidence: an unclosed boundary

Reusing a known captured kernel and fitting an independent whole-model
benchmark are different purposes, and this layer does not classify splits.
`a3_calibration_store` owns the train/held-out rule for fits
(`calibration.heldout_fit_forbidden`) and `HeldoutGuard` excludes declared
held-out conditions from `fit_calibration`. Neither reaches a captured replay,
and their identities do not map today: a guard identity is
`family|canonical_shape` over a physical operator shape, while an inventory
variant is a digest of the kernel signature, capture context and capture group.
No declared correspondence exists between a kernel-level variant and an
operator-level condition key, so binding the guard here would either block
legitimate replay or match nothing while appearing to protect.

The boundary is therefore a configuration requirement rather than an automatic
check: an evaluation that publishes held-out accuracy must bind the captures it
replays to the declared held-out conditions it evaluates against, and state
which captures were replayed. Closing it in code needs a declared
variant-to-condition mapping in the evaluation configuration, not a second
split classifier in `profiler`.

### Validation

The CPU-side checks cover this layer's resource and precedence behaviour: the
captured median stays the completion floor and the launch term is charged once;
the plan's resource totals equal the operator's declared work and the shared
stage services it; a concurrent analytic operator is delayed by a captured
kernel's declared work; an explicit variant that cannot be applied is an error
instead of another price or a zero-cost plan; a captured binding and an exact
existing calibration both resolve in one run; and one real twice-captured
variant keeps its call count with its observed minimum and maximum. They run
against the committed A3 capture excerpt.

None of that is accuracy evidence. Whether replaying captured kernels improves a
whole-model prediction can only be answered on matching hardware, kernel
implementation, tiling and timing scope, by comparing error and configuration
ranking against frozen held-out device measurements before and after — and only
once an evaluation configuration binds the held-out boundary above.

## Coverage and fallback

Supported inputs select a plan with its source, assumptions and resource
summary. An unsupported operator or recipe keeps the legacy cost and reports
the refusal reason. Strict mode reports unsupported input as an error instead
of silently returning a different estimate. Existing calibration, fusion and
communication owners are preserved before invoking a provider, including in
strict mode.

Malformed plans are errors: missing or cyclic dependencies, unknown resources,
invalid units, negative or nonfinite work, and infeasible buffer allocations
must not become a successful zero-time estimate. Existing fusion and
communication stage paths retain their own accounting and precedence.

## Validation boundaries

Validation uses independently calculable instruction chains, finite GEMM
recurrences and a Vector-to-Cube-to-Vector dependency graph. The checks cover
resource conservation, terminal joins, fallback, optional-dependency isolation,
single charging of device launch and parent work, and compatibility with
existing calibration and stage paths.

These checks establish the behavior of the model. They do not establish
latency accuracy on an NPU. Accuracy evaluation needs matching hardware,
kernel implementation, tiling, timing scope and held-out measurements.
