# Source-derived operator extensions

The coordinator owns the engine, shared interfaces, frontend wiring, calibration
and integrated acceptance. Each operator author owns one planner module and its
tests. Source-derived plans may describe a different implementation than a
historical capture; record that distinction before comparing times.

## Choose the abstraction first

Read `modeling-fidelity.md`. Start with reusable resource or phase/tile models.
Preserve actual arithmetic, materialized traffic, branch, kernel boundaries,
data dependencies and relevant ownership/capacity limits. Use source to identify
these mechanisms; do not transcribe every intrinsic by default.

Describe the retained mechanisms and consequential approximations in the owned
operator report. A detailed queue/instruction model needs a counterexample where
the simpler model misses material repeated cost, a shape trend or a configuration
ordering. Small percentage errors alone do not justify further detail. A compact
`OperatorPlan` is a valid source planner; `plan_required` is not an instruction
inventory requirement. Timing calibration is independent of program granularity.

## Extension interface

Each module in `mindie_sim/prediction/operators/` exports:

```python
def make_op(params: dict) -> OpSpec: ...
def build(op: OpSpec, geometry: Geometry) -> OperatorPlan: ...
```

`make_op` is an explicit geometry adapter used by the campaign cases. It must not
read measurements, infer model names, or import a device runtime. `build` declares
physical work, including padding, conversions, repeated loads and workspace
roundtrips, using `Work(matrix={precision: flops}, resources={role: amount})`.
`Geometry` supplies compute-unit counts, resource-unit counts, per-unit buffer
sizes and buffer-unit counts.
It contains no vendor/model labels or measured latency.

Source `make_op` records set `source_boundary="kernel"`, or `"call"` for a
wrapper that requires kernel expansion. Both imply `plan_required`.
Unbound source records fail
before execution rather than inheriting a generic family or program envelope.
Source children declare `source_boundary="kernel"`; the terminal zero-work join
is a separate ordinary operator. A call without a bound expander is an error,
and repeated expansion does not turn physical children into new wrapper calls.

`OperatorPlan` pairs this work with `program(context)`. The program is built from
the existing `KernelProgram`, `PipelineProgram`, `ParallelProgram`, `Stage`,
`Allocation` and bounded transaction mechanisms. `context.bind(work)` converts
typed physical work into engine demand. BF16 and FP32 share the same matrix
resource and use their separately declared peaks. An undeclared peak is an input
error, never an inferred multiplier. Represent relevant reduction,
transcendental or memory-response costs with declared phases and service
parameters. An analytical aggregate is allowed if its work convention and limits
are explicit. Do not infer several resource rates from one opaque kernel time
or hide missing stages in a fitted setup time. Instruction inventories and
instruction-specific rates are optional refinements, not default requirements.

`OperatorPlan.lower` wraps setup and optional primitive completion timing. The
author's nested program does not need to apply either again.

An isolated one-resource phase can optionally use `ResourceService` through
`context.for_phase(work, kernel=..., dtype=..., layout=..., axes=...)`.
That context retains typed native work and ownership but clears the enclosing
setup/completion timing. The curve must have a `resource_phase` boundary and
matching categorical conditions; a whole-kernel measurement cannot stand in for
an unobserved inner phase. Other phases stay analytical. This is a calibration
choice, not a requirement for source planners. See `resource-services.md`.

The caller binds `PlannedRecipe(module.build)` with
`RecipeBook(recipes={implementation_id: ...})`. Both iteration execution and
isolated operator analysis prepare the physical OpSpec before pricing, so the
public work ledger uses the plan's totals. New modules are not auto-discovered or
activated. The coordinator reviews and binds them after validating dispatch,
layout and compatibility. Explicit source dispatch is allowed; model-name timing
tables and observation-presence conditions are not.

Generic semantic fusion and activation-conversion passes do not rewrite a source
planner's input or kernel boundaries. A fused source implementation must supply
its own explicit planner and input contract. Test the default public path as
well as the explicitly bound one: a private runner's binding cannot establish
that a frontend or runtime configuration selected the implementation.

## Executable checks

`tools/run_operator_cases.py` accepts the extension module and a campaign case
file. It exercises the public isolated execution path on declared synthetic
unified-memory and split-copy hardware, checking finite completion, work
conservation, buffer release and independence from graph identity.

```sh
PYTHONPATH=. /path/to/shared/venv/bin/python tools/run_operator_cases.py \
  mindie_sim.prediction.operators.example examples/operator-cases.json \
  -o /tmp/example-contract.json
```

Those synthetic rates have no relation to real hardware accuracy. Authors must
also add independent small checks appropriate to the abstraction: arithmetic,
address or numerical oracles, relevant tail and batch cases, and source line
references with file hashes. Do not recreate numerical kernel execution merely
to test a service envelope. A work-conservation check catches a divergent
program; it cannot establish that both the plan and program correctly represent
the source. The coordinator verifies that separately.

Programs must remain lazy in token count. Never materialize all scalar tasks.
Pin stages with `context.owned(owner, owner_count, roles)` or explicit
`Stage.placement`; inactive cores cannot supply a pinned CTA. `Allocation.units`
names the physical owners, and its size applies to EACH listed unit. Pool
capacity is per unit, with independent residency and release. Use
`context.allocate(name, pool, size, units)` for declared pools. Distributed
copy resources also require ownership; global memory remains shared.

A source wrapper with multiple kernel launches must export
`kernels(op) -> tuple[OpSpec, ...]` and bind
`PlannedRecipe(build, kernels)`. Children use local ids/dependencies and the same
recipe id; `build` selects their explicit kernel phase from metadata. Expansion
preserves caller dependencies, placement, capture metadata and a terminal join.
Both `operator_costs` and `Execution` expand before pricing. Eager execution
submits each real kernel; graph capture submits the captured group once.
The verifier checks conservation, input readiness and release for each expanded
kernel before aggregating the display totals; opposite child errors cannot cancel.
The case runner detects `module.kernels`. Verify host service independently:
`source_kernel_launches` metadata alone does not change execution.

The runner also delays input readiness by 1 ms. Duration must remain invariant
under that shift and no resource may be serviced before input arrival. Nested
lazy programs must gate their initial issue window; a later terminal join does
not repair an early DMA. `Stage.previous` can reference a named stage in the
previous pipeline tile when buffer reuse differs from same-stage pipe order.

Use `Device.service_modes` for multiple instruction classes on one physical
pipe. Keep bytes, lanes and instruction calls distinct in `Work`; mode binding
converts only service units. `InstructionInventory` is available for exact
source call/payload counts and independent capacity projections. It does not
provide unknown instruction latency, compiler occupancy or a dual-issue
schedule. Declare those limits rather than adding a fitted setup duration.

An extension with typed source services may export
`SERVICE_ROLES = {work_role: physical_resource}`. The case runner supplies
explicit synthetic rates for these roles; it does not add them to real hardware
profiles. Physical prediction requires declared rates, which may be analytical
assumptions or calibrated primitives. Missing measurement is never a gate.
Keep source lane counts, warp calls, shared-memory bytes and compiler-issued
instructions distinct. A single generic vector rate must not silently price
all of them as equivalent operations.

Tensor lowering identifies storage by producer and output port as well as
shape/stride/offset. Equal-shaped outputs from the same kernel are distinct.
Use `contiguous` for source `.contiguous()` semantics and `copy_into` for a
real assignment to caller-owned storage. Preserve the actual prefix view of a
padded destination and never relabel an actual output as a larger tensor.

## Handoff and acceptance

Every task delivers scoped files, a source map, executable cases, independent
checks, a resource ledger and a report of consequential approximations. Include
the reason for the selected modeling depth and what would justify refinement.
Where timings exist, report repeated residual contribution and heldout
shape/batch behavior as well as kernel percentage error. A passing test suite
does not establish timing accuracy; nonzero timing error does not by itself
require more source detail. Failed checks stay visible. Historical observations
are validation inputs unless an
explicit primitive training split was supplied. No TTFT/TPOT target or full-model
sum may change operator parameters.

The coordinator owns Git metadata and commits reviewed files; workers must not
create commits, rewire `.git`, fetch history or create scratch clones/bundles.
Case `expected_dispatch` entries must match `OpSpec.metadata["dispatch"]`.
An unsupported requested branch remains incomplete even if another branch passes.
The coordinator independently checks returned files and only then wires the
operator into the real graph. A finished DSH session is `returned`, never
`accepted`. The old implementation and observations remain preserved until
operator, block, pure model and serving-constraint acceptance is complete.
