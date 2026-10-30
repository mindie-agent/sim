# Compact steady two-stage pipeline

`SteadyPipeline` is a compact finite program for one producer stream and one
consumer stream. It preserves every full item's physical work while executing
`count` identical items with a constant number of Engine tasks instead of one
task per item. The only nonuniform item is an optional, explicitly declared
final tail.

```python
from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.program import Stage
from mindie_sim.prediction.steady_pipeline import SteadyPipeline

pipeline = SteadyPipeline(
    count=64,
    producer=Stage("load", {"hbm": 4096.0}, placement={"hbm": (0,)}),
    consumer=Stage("compute", {"cube": 2048.0}, placement={"cube": (0,)}),
    slots=2,
    tail_producer=Stage("load", {"hbm": 1024.0}, placement={"hbm": (0,)}),
    tail_consumer=Stage("compute", {"cube": 512.0}, placement={"cube": (0,)}),
    buffers=(Allocation("workspace", "hbm_pool", 1024.0, (0,)),),
)
completion = pipeline.start(engine, "layer", ready, owner="layer")
```

The primitive imports only the ordinary `Engine`, `Event`, `Allocation`,
`Stage` and `scope_allocations`. It does not create a second engine, history,
state machine or slot list.

## Supported declaration

`count` is a nonnegative integer number of identical full items. `slots` must
be at least two. The compact form assumes at least two resident item slots and
does not enumerate their occupancy; with disjoint producer and consumer
resources, two slots already permit the producer and consumer streams to
overlap. Extra slots do not change the uniform no-tail completion. A declared
tail requires exactly two slots: with more slots its producer can start before
the final full consumer, so the two-slot finite-tail equation does not apply.

A full producer or consumer is a plain `Stage` declaration. It may declare
only `name`, `work`, `caps` and `placement`. The following fields are refused:

- `after` and `previous`: the primitive owns the only cross-stream handoff.
- `latency`: no fixed steady latency is permitted.
- `program`: no nested program or slot machine.
- `acquire` and `free`: allocation lifetime belongs to the pipeline invocation.
- a true `coupled` flag on an input stage: the primitive sets coupling only on
  the aggregated steady task.

Producer and consumer resource names must be disjoint across `work`, `caps`
and `placement`. Aggregating shared names would merge contender counts,
placements or caps without an exact per-item schedule, so the declaration is
refused rather than silently merged.

`tail_producer` and `tail_consumer` are optional together in the two-slot domain.
They describe one exact nonuniform final item. Their resources must be a subset of the
corresponding full stage's resources, and their placement must match the full
stage for every declared tail resource. A tail may have a smaller demand, or
omit a resource whose work is zero for that item. Tail caps are passed through
exactly.

`buffers` are real invocation-resident `Allocation` objects supplied by the
caller. They are scoped with ordinary `scope_allocations(name, buffers)` so two
invocations cannot share an identity accidentally. Names must be unique. The
Engine performs capacity and owner-unit admission; the primitive invents no
buffer, unit or work-scaled residency.

## Executed timeline

For `N = count > 0` and an optional tail:

```text
ready
  -> reserve explicit buffers, if any
  -> producer(full)                              # item 1 producer
  -> coupled (N-1) * (producer + consumer)       # items 2..N producers,
                                                 # items 1..N-1 consumers
  -> without tail: consumer(full)                # item N consumer
  -> with tail:    consumer(full) || tail_producer
                   -> tail_consumer
  -> release explicit buffers, if any
  -> complete
```

With a tail, the tail producer overlaps the last full consumer. The tail
consumer waits for both. For `count = 0` with a tail pair, the tail producer
and tail consumer execute serially after readiness (and after any buffer
reservation). For `count = 0` without a tail, the program is empty: it completes
at readiness without reserving buffers or adding latency.

The steady task is submitted with `coupled=True`. Its producer and consumer
demands therefore maintain synchronized fractional progress under the Engine's
fair, capped and owner-unit allocation instead of letting the faster resource
consume all of its work before the slower stage. The Engine still recomputes
rates at external arrivals and completions; the primitive does not keep a
history or a private progress state.

## Work and time

Every full item contributes exactly one producer stage and one consumer stage.
The compact timeline therefore submits exactly:

- `N * producer.work` and `N * consumer.work`;
- plus the exact `tail_producer.work` and `tail_consumer.work` when declared.

There is no repeated-task cost, per-item latency, setup constant or scaled
average tail. The steady task contains `(N-1) * (producer.work + consumer.work)`
with the union of the two disjoint stages' caps and placements. It has no fixed
latency.

Let `load_ms` and `compute_ms` be the no-contention serial durations of the
full producer and consumer stages. For a stage with several resources this is
the maximum of that stage's declared resource services. The no-contention
completion is

```text
load_ms + (N-1) * max(load_ms, compute_ms) + compute_ms
```

With a tail whose serial producer and consumer times are `tail_load_ms` and
`tail_compute_ms`, the `(N-1)` steady intervals end first. The Nth full
consumer and the tail producer then start together, and the tail consumer
waits for both:

```text
load_ms + (N-1) * max(load_ms, compute_ms)
        + max(compute_ms, tail_load_ms) + tail_compute_ms
```

For `count = 0` with a tail the completion is
`tail_load_ms + tail_compute_ms` because the only item executes serially.

## Contention semantics and limits

The compact steady interval is a **backpressured fluid approximation**. Under
external competition, its completion is not claimed to equal arbitrary
discrete per-slot execution. The exact parts are the physical work, owner
placements, supply caps, allocation admission and release boundaries, and the
Engine's causal rate recomputation. The approximation is that all remaining
full-item producer and consumer work advances as one coupled flow rather than
as separately schedulable slots.

Use a detailed `PipelineProgram` when the question needs per-item stream
ordering, arbitrary discrete contention, unequal item work, nested programs or
tile-level residency. The compact primitive is a screening/finalist control for
the narrow uniform domain; it is not a replacement for the detailed program and
makes no source, compiler or native-kernel exactness claim.

`tests/test_steady_pipeline.py` covers the uniform N=1/N=2/N=1000 controls
against an independently built two-stage `PipelineProgram`, the closed form,
ready-gated service, exact tails against an N+1 detailed control, finite
buffer owner units and release boundaries, stage cap and placement union, an
external memory arrival, a ten-million-item run with a constant task window,
and the existing `tests/test_phase_ranking.py` 21 ms versus 30.1 ms overlap
counterexample. That test file is not changed.

Independent acceptance additionally compares 216 configurations with a finite
slot-retirement recurrence, including load-heavy, compute-heavy, balanced and
large tails. A three-slot counterexample completes in 103 ms while the two-slot
execution takes 112 ms; that unsupported tail geometry is explicitly rejected
instead of being advertised as exact.
