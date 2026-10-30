# Primitive service curves

`ResourceService` is an optional, work-dependent service model on the existing
resource engine. It preserves native work and constrains the supply available
to one invocation. It does not replace the engine, derive occupancy for several
resources from one kernel time, or change any model's default recipe.

The first implementation has **one numerical coordinate: native work**. Hardware,
runtime/revision, kernel implementation, dtype, layout, replay mode, participating
units and any additional dispatch axes are fixed within a curve. Multiple strides,
tile policies or active-unit counts require separate declared curves. A general
multidimensional interpolation or automatic implementation selector is not present.

This is the first bottom-up extension from the industry review in
`modeling-fidelity.md`: isolated service estimation, then explicit phase/operator
composition, then block and pure-model costs. Calibration remains optional at
each level. Missing observations do not disable the analytical simulator.

## What is modeled

For native work `w`, the table supplies total invocation time `T(w)` and a
nonnegative startup `s`. Inside the training range, `T` is linear between
adjacent work knots. After startup, the program submits the unchanged physical
work to the engine with supply cap:

```text
service time   = T(w) - s
invocation cap = bound physical work / service time
```

The engine still enforces physical capacity, placement and contention. Two
independent streams consume the same shared memory resource. A pinned invocation
cannot borrow inactive owners. Its cap may exceed local physical supply, in
which case physics lengthens the prediction and the result records a
`capacity_limited_queries` count. The curve is not a guarantee that a declared
hardware configuration can realize its isolated time.

Treat startup as a response-model parameter within this domain, not a recovered
instruction latency. A copy kernel represented by memory service plus startup
is an explicit approximation. Concurrent accuracy does not follow from fitting
isolated timings: other instruction limits and interference may become relevant.
They require separate discriminating observations before adding complexity.

Outside the calibrated work range, the selected recipe's `outside_domain` policy
is either:

- `analytic` (default): use the declared physical resource and placement. Do not
  extrapolate the curve or extend its fitted startup beyond its domain.
- `error`: reject the explicitly selected empirical prediction outside its range.

The analytical result is reported as `analytic_outside_domain`. Different
categorical identities, boundaries, units or participating-unit counts are
binding errors, not opportunities for a nearest-neighbor substitution.

## Calibration input and output

`mindie-sim fit-service packet.json -o fitted.json` accepts:

| Field | Meaning |
| --- | --- |
| `resource`, `work_unit` | One native role and its unit, such as `hbm`/`bytes`, `cube`/`flops`, `vector`/`ops` |
| `identity` | `service_identity(case, op, replay_scope)` for the explicitly matched implementation |
| `boundary` | `device_kernel` for a whole invocation, or `resource_phase` for an isolated inner phase |
| `active_units` | Number of participating physical owners of that resource |
| `samples` | Records with `work`, `duration_ms`, `run_id`, `repeat_index`, `source`, `split` |
| `provenance` | Optional capture protocol, observed boundary and explicit binding assumptions |

Known work units are shared with hardware binding. Additional resource/mode units
are declared with `Device.work_units`; unspecified custom roles use `work`.
For example, a byte-copy instruction sharing a vector pipe can declare
`work_units: {"strided_copy": "bytes"}`. Its native bytes remain bytes; service
normalization into the shared pipe does not turn them into vector operations.

The fitter first takes a median within each run at each work coordinate, then a
median across runs. Repeated sampling of one run cannot outweigh independent
runs. Identical `(run_id, repeat_index, work)` records are deduplicated; conflicting
durations or train/heldout assignments are rejected. Only training samples supply
knots or the nonnegative affine startup estimate. At least two distinct work
coordinates are needed. The affine reference is reported separately; the service
curve uses the measured knots, not an end-to-end multiplier.

The result contains `model` (`mindie-sim.service-curve/1`), its content ID,
`affine_reference` and separate `heldout` predictions. Heldout coordinates outside
the range have no curve prediction. Changing heldout durations cannot change the
model. This separation does not guarantee that the experimental split covers
independent runs or devices; the adapter must retain that distinction.

An identity made from simulator metadata is not proof of a historical source
match. Preserve missing binary identity or run conditions in provenance, without
making them an analytical simulation admission gate.

## Use in an operator or a phase

An ordinary one-resource operator can bind the model in the normal runtime
recipe registry:

```python
case["runtime"]["kernels"]["recipes"]["copy_impl"] = {
    "kind": "resource_service",
    "model": fitted["model"],
    "outside_domain": "analytic",
    "units": [],  # All owners; a nonempty list pins physical owner IDs.
}
```

Its `OpSpec.recipe` is `copy_impl`, with an explicitly matched implementation
identity. Both `operator_costs` and `predict` execute that binding. A mixed
memory/compute kernel must keep its own phase model. Even two instruction modes
on one physical pipe cannot be identified by one opaque duration.

Source-bound operators still need `PlannedRecipe`; a service curve cannot bypass
kernel expansion or implementation selection. A planner may compose one measured
phase with analytical phases:

```python
load_context = context.for_phase(
    load_work, kernel="declared_load_phase", dtype="bf16",
    layout="contiguous", axes={"stride_bytes": 128},
)
program = KernelProgram((
    Stage("load", program=load_service.lower(load_context)),
    Stage("compute", context.bind(compute_work), after=("load",)),
    Stage("store", context.bind(store_work), after=("compute",)),
))
```

Here `load_service.model.boundary` must be `resource_phase`. `for_phase` binds the
phase's typed native work and clears enclosing setup/completion constraints.
`OperatorPlan.lower` applies those enclosing constraints once. Phase startup
belongs to each phase invocation; a whole-kernel measurement cannot be silently
reused as an inner phase. Delayed input readiness gates all nested service.

Diagnostics aggregate queries by model ID (`anchor`, `interpolated`,
`analytic_outside_domain`) and record physical-capacity conflicts. No per-token
query history is retained.

## Retained GPU copy replay

This command is CPU-only and reuses the October 27 capture:

```sh
.venv/bin/python tools/measure/replay_memory_service.py \
  data/measurements/gpu/sim-probe-20261027-234735.stdout \
  --memory-bytes-per-second 1e12 \
  --packet-output /tmp/copy-service-packet.json \
  -o /tmp/copy-service-validation.json
.venv/bin/python -m mindie_sim fit-service /tmp/copy-service-packet.json \
  -o /tmp/copy-service-model.json
```

`1e12 B/s` is a declared round-number physical ceiling for this control, not a
newly measured GPU specification. Required compute/memory-capacity fields in
the adapter are unused schema placeholders. Only memory work is exercised.
Changing a nonbinding ceiling leaves isolated predictions unchanged; it changes
the explicitly synthetic concurrent-resource control.

The original split uses source sizes 64/96/128 MiB for training and 112 MiB for
heldout evaluation. Work counts both read and write bytes. The adapter preserves
the five original trial identities. Each timing is a device-event batch average
over 24 launches, not 24 independent observations. Retained probe/PTX hashes
identify the source used for binding; the capture did not record the loaded
binary hash.

The production isolated path predicts **0.300842663 ms** at 112 MiB, against
observed median **0.301820000 ms**: **-0.324%**. Individual trial errors range
from **-6.162% to +2.904%**. The earlier affine method predicts 0.298891698 ms;
this small difference does not establish that interpolation is generally better.
There is one heldout size in the same capture, not an independent-run,
cross-device or concurrent-hardware validation.

The retained report is
`data/measurements/gpu/service-curve-validation-20261028.json`. It includes the
unchanged split, source hash, trained model, old affine reference, physical work
ledger, delayed-readiness checks, and serial/parallel copy composition. For two
112 MiB copies at the declared ceiling, the graph takes 0.469762048 ms in parallel
and 0.601685325 ms in sequence, while serving the same 469,762,048 bytes.
Those two values are conservation controls, not measured GPU concurrency.

This result does not calibrate general tensor copies, GEMM, attention, a complete
model, or TTFT/TPOT. It establishes a reusable interface and one bounded
primitive-domain replay before those higher layers are extended.
