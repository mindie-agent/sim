# Architecture

The public entry point is `mindie_sim.predict`. Every supported deployment uses
`prediction.engine.Engine`. There is no separate GPU, A3, PD or calibrated engine.

The intended level is analytical resource/phase modeling plus causal event
simulation. Timing is approximate. Correct structure does not require reproducing
kernel instructions or a compiler schedule. Choose detail by repeated cost,
heldout trends and decision stability; see `modeling-fidelity.md` for the
industry comparison, refinement policy and current calibration limits.

## Contracts and ownership

| Layer | Input -> output | Responsibility |
| --- | --- | --- |
| `schema`, `catalog`, `backends` | native configuration -> `Case`, `ModelSpec`, `LayerPlan`, `Hardware` | Units, structure, representation and placement |
| `prediction.frontend` and block builders | scheduled request views -> `IterationWork` / `OpSpec` DAG | Arithmetic, storage traffic, logical dependencies and streams |
| `prediction.program`, `matrix` | physical work + implementation -> finite resource program | Tile tails, stage ordering, supply limits, allocations and lifetimes |
| `prediction.engine` | resource tasks -> completion events | Shared capacity, fair/FIFO service, releases and causal time |
| `scheduler`, `capacity`, `dispatch`, `disaggregated` | request progress + resource completion -> next iteration | Admission, cache ownership, batching, preemption and handoff |
| `metrics` | real commit events -> latency, throughput, SLO | Request-level accounting without token histories |
| `observations`, `primitives` | raw data -> explicit parameters or independent comparisons | Provenance, exact matching, train/heldout separation |
| `study` | explicit candidates + probes -> cost concentration and comparable rankings | Compose existing operator/graph/serving paths; retain failures and separate timing boundaries |
| `prediction.call`, `observations.call_alignment` | callback arguments/protocol -> graph cost; preserved events -> conditional comparison | Reuse the graph engine, preserve complete callback work and independent event identities |

Model labels and source revisions are provenance. Physical work does not branch
on them. Family adapters decode native configurations into structural records;
they contain no latency correction tables. Hardware profiles are data, not
alternate execution algorithms.

## Resource time

Rates are work per millisecond inside the engine. Each task declares its resource
work and dependencies. Independent resources may serve the same ready task at
once; all requested work and any minimum duration must finish before completion.
Consumers of a fair resource share its rate, redistributing unused capped supply.
FIFO resources serialize host submissions or other explicitly ordered service.

A dependency, stream order and contention are distinct constraints. Two independent
streams can contend for the same memory or link. Moving them to different streams
cannot delete bytes or FLOPs. Host launch tasks may run ahead of device work;
graph capture replaces groups of submissions with graph-replay submissions.

`Device.service_modes` binds distinct instruction work to a shared physical
pipe at an explicit rate. UB-to-UB copy bytes and register issue slots can
therefore contend with other vector instructions without becoming separate
unlimited resources. Work remains in its native units; only engine service is
converted to the pipe's units. Ownership follows that physical pipe.

A measured primitive duration is an optional completion constraint on an exact
implementation identity. It does not alter FLOPs or bytes. If resource service is
slower, the physical program still determines completion. Hardware-rate calibration
changes a declared resource rate instead. No end-to-end residual is applied.

Explicit `kernels.invocations` rules bind device startup and an inclusive
completion minimum to implementation axes. Both start at original device
readiness: completion is `max(minimum, startup + body)`. Logical joins and nested
resource phases do not gain another invocation. Host launch/replay and transport
propagation remain separate. Hardware execution-class declarations provide
low-priority defaults; runtime rules override them. A3 profiles carry separately
labeled user-reported completion minima. Undeclared classes remain zero; an
unexplained residual does not establish a startup parameter. See
`invocation-costs.md`.

`ResourceService` optionally interpolates a one-resource invocation's response
inside a declared work domain. It turns service time into a per-invocation supply
cap while preserving physical work, ownership and shared capacity. Native units
and phase boundaries are checked before binding; opaque mixed-resource timings
cannot be decomposed by this interface. `RecipeContext.for_phase` composes an
isolated phase without inheriting the enclosing kernel's startup or completion
constraint. See `resource-services.md` for schema, explicit analytical behavior
outside the domain and the retained GPU copy replay.

Observable whole-kernel service regimes additionally support small-payload,
per-call and per-group responses without pretending to identify every internal
resource from elapsed time. Their declared domains and optional resource-rate
scaling feed one inclusive completion constraint through the same engine.
Parameters from operator observations are supported; the comparison API itself
does not silently fit them. See [observable service regimes](observable-service-regimes.md).

## Finite programs

`KernelProgram` is a finite stage DAG. `PipelineProgram` lazily issues at most its
slot window, retaining stage tails and allocation lifetimes. Finished tasks and
waiter lists are released. Trace history is a bounded deque; request metrics use
constant-sized aggregates per request.

Execution retains graph dependencies for pending operators and expands their
per-core programs only when inputs, stream predecessors and submission are ready.
Collective participants share their rendezvous before that expansion. Inactive
model layers therefore do not each retain a full set of core pipelines.

`TiledMatmul` explicitly partitions M/N tiles across physical cores and K panels. It keeps accumulator
buffers across K panels and releases operand buffers after consumption. Declared
buffer capacities and residency are independent per physical unit. Tail panels retain their
real dimensions. Default tiles are 128 x 128 x 128 with two slots. Operand reuse
is idealized across relevant M/N tiles by default; disabling `reuse_inputs` or
`reuse_weights` charges reload traffic. This is not a reconstruction of a vendor's
GEMM implementation, cache replacement or instruction schedule.

The opt-in `hierarchical_matmul` and `mapped_matmul` recipes separate private
L1 refill tiles from inner matrix operand tiles. Independent finite producer
and consumer windows overlap refills with micro products while preserving
MTE1 service, per-owner L0A/B storage and accumulator lifetime. The finite
selector filters declared candidates by capacity and ranks their resource
lower bounds; it does not infer vendor dispatch. Equal independent head/group
matrices retain separate operand domains. Aggregate expert geometry keeps its
separate approximate family binding until actual unequal row counts are modeled.
See `operators/matrix-mapping.md`.

The explicit `phase_matmul` alternative retains the selected mapping's
per-owner work in a short load/operand/compute/output phase chain. It omits
within-owner K/micro overlap and dynamic tile residency. It is a declared
fidelity choice for full-model screening, not an equal-time optimization of
the detailed recipe. Source-bound plans keep precedence. See
`graph-runtime-diagnosis.md` for the complete V4 cohort and work comparisons.

`steady_matmul` preserves nominal refill/operand/cube overlap through a bounded
finite-state recurrence and exact translated-cycle jumps. It compresses regular
K panels and repeated per-owner output tiles into resource-coupled tasks, with
finite buffers and unchanged physical work. Nominal output tiles remain serial;
external contention is a declared fluid approximation. The individual-output
control remains selectable. See `operators/steady-matrix.md`.

An explicit quantized-grouped source call can supply lazy owner rectangles to
that same matrix core. A declared partitioned output program publishes local
accumulators, reserves per-vector buffers, loads scales and executes finite
output chunks. This preserves source AIC/AIV dependencies without inferring
the historical kernel's dispatch or scalar latency from an event residual.
See `operators/grouped-source-execution.md`.

`ResourceEnvelope` is the default for operations without a specialized stage
recipe. Its time is the maximum independent resource service, plus declared setup.
A source-derived `OpSpec` declares `source_boundary="kernel"` or `"call"`. Its exact recipe id must bind
a `PlannedRecipe` before isolated analysis or iteration execution; family/program
defaults and an explicit envelope cannot substitute for its work and dependencies.
Calls require a kernel expander; an unexpanded call cannot be priced as one
kernel. Already expanded kernels retain their boundary through repeated passes.
Generic conversion/fusion passes preserve these implementation boundaries.
A bound source planner may use a compact envelope or phase model when that is
sufficient for its declared work and constraints. The binding requirement
establishes selection, not a minimum instruction-level modeling depth.
A custom recipe can introduce serial stages, transaction limits, conversion work,
additional resource roles and real tensor lifetimes using the same engine.
`Stage.previous` expresses named dependencies on the prior pipeline tile in
addition to same-stage pipe order. A nested program must respect its parent's
ready event, including independent-looking prefetches. The public operator
verifier delays input readiness and rejects any earlier resource service.

Typed register inventories retain source calls, active lanes, UB payload and
barriers. Their issue/port/lane projections are independent capacity bounds,
not a compiler instruction schedule or a dependency-latency model. A masked
single-lane operation still consumes an instruction call.

`QueuedStore` models finite issue/response slots and address-granule dependencies
on that engine. Memory contention delays responses and therefore queue reuse.
`TiledPointwise` preserves independent unit streams, grouping only identical
ownership/address phases. Subprograms can be nested inside a stage; allocations
remain live until its nested program completes. Scalar transfer parameters and
their actual validation limits are documented in `scalar-transfers.md`.

Matrix waves limit both compute and declared distributed copy-engine service to
participating units. A tail wave cannot borrow inactive units' transfer supply.
Devices with unified memory service do not acquire an artificial copy-engine
bottleneck. `Device.resource_units` separates vector and matrix scheduling units.

`RecurrentVector` models GDN decode as load, decay, key dot, delta, rank-one
update, query dot and store. State remains resident across a sequence's dependent
tokens. Independent sequence/head tasks share a launch; each retains its own
length and state. Tail waves cap vector and distributed copy service. This
expresses elementary work and state dependencies, not a compiled instruction
schedule or calibrated reduction/synchronization latency.

## Semantic frontend

Attention, FFN and residual builders are registries. Supported building blocks
include GQA/windowed attention; latent MLA; compressed/window caches, index/top-k
selection and shared owners; GDN/KDA state transitions; dense and MoE feed-forward;
hyper connections and attention residuals; optional MTP/DSpark and vision towers.

GEMMs keep logical arithmetic, packed payload, scales, activation traffic and
compute precision separate. `activation_quantization` inserts explicit producers
of low-precision activation payloads and scales. Activation group geometry is
independent of weight blocks. Optional `norm_quant` and `residual_norm` passes
remove intermediate traffic only when consumers allow it; live residual outputs
remain. Packed weight dequantization is still an analytical in-kernel assumption
for formats that require it. It is not a reconstruction of vendor instructions.

`TensorLayout` carries shape, stride, offset and backing shape. Splits and
permutations retain address maps. Reshapes that cross storage gaps and consumers
requiring contiguous inputs introduce real copy operators. The current GDN
builder uses this mechanism for projection slices, full-batch QKV packing,
phase slices, native inputs/exports and the output gate. `TensorValue.output`
distinguishes multiple outputs from one producer so equal Q/K shapes do not
alias in the copy cache. A contiguous consumer reuses a view when possible;
concatenation and caller-owned output assignment still perform transfers.

Declared native fusion groups reuse that semantic graph. The optional
`native_fusion` lowering folds a selected complete group into one resource
program after capture boundaries are assigned. Child work and dependencies
remain; redundant startup and only source-confirmed internal materializations
are removed. Structure export preserves semantic members and their executable
resource boundary. Profiler cohorts reconcile repeated candidates with equal
tensor contracts and physical work; `cohort_comparison` prices that same
declared program without assigning individual layer occurrences. Explicit
recipe bindings participate in structure identity. See
[native fusion resources](native-fusion-resources.md) and
[structure attribution](profiler-structure.md) for comparison limits.

The explicit `gdn_prefill_arch35` implementation selects the retained
noninterleaved QKVZ/BA non-spec caller. Its native convolution and Prepare/H/O
use the same planners as isolated analysis. Mixed batches keep a decode prefix,
a native prefill suffix and the actual output concatenation. Projection,
zero-fill and output normalization retain padded rows, while the core processes
only actual rows. Decode recurrence remains a declared analytical mechanism.
Other projection, speculative mixed, PCP and state-precision branches require
their own implementations; selecting an unimplemented branch is an error.

`final_norm_scope=all` normalizes the whole forward output before selecting rows
for logits, including unfinished prefill chunks. `selected` declares the
alternative software ordering explicitly. Selection carries its payload and
index traffic. PCP normalizes local rows before the selected hidden-state gather.

A draft body precedes one target verification body. A declared acceptance trace,
distribution or mean determines integer commits, never a discount on target work.
Burst tokens share their real commit time, preserving long inter-token gaps.
Draft prefill populates draft cache after target processing. Noncausal block draft
behavior and persistent feature layout are analytical declarations, not a claim
that all serving implementations use that schedule. Standalone draft checkpoints
can be inspected, but cannot serve target requests without a target model.

## Serving and placement

Requests retain arrival, admission, first-token, last-token and completion times.
The scheduler supports open-loop arrivals and closed-loop chains, sequence and
token budgets, chunked prefill and priority/running-first ordering. Cache is
reserved before work. Preemption releases cache and explicitly recomputes it.
A prefix cache hit still recomputes the last hidden state needed for sampling.

DP replicas have distinct memory/compute resources. EP can span DP and includes
idle collective participants. TP, EP/ETP, PP, CP and PCP derive separate groups.
CP shards KV within TP; PCP adds a physical query-shard axis. Unsupported PCP/CP
or PCP/EP schedules are rejected rather than ignored.

Each PP stage simulates one representative local shard. Its TP/PCP group requires
matching physical owner topology, resource roles and service-mode bindings.
Rates and per-owner capacities may differ and use conservative member values.
Combining different core counts requires per-rank execution, which is not yet
implemented; the representative path rejects it instead of mixing one rank's
total rate with another rank's unit count. Separate DP/PP groups may have
different geometry. CP uses maximum
local work/capacity and exact block ownership where implemented; sparse selection
still uses a balanced selected-row assumption. It is not per-rank skew simulation.

P/D pools share a clock and engine. A destination reserves cache before transfer;
the source owns cache until completion. Transfer consumes the physical connector
and both memory services. Decode starts after handoff. Source/destination token
budgets and destination concurrency cause real backpressure. The connector volume
uses the destination shard layout; heterogeneous layout conversion is not modeled.

## Capacity and metrics

Weights are derived from matrix terms, quantized formats and parallel ownership.
Cache is derived from owner layers, block sizes, compression, windows, persistent
recurrent states and selected draft geometry. Measured total memory never replaces
these terms. Workspace estimates simultaneous layer tensors, residual history,
draft feature staging and the default vision score buffers. Compiler graph pools,
communication reserves and runtime allocations are explicit user parameters.

Public metrics use `device_token_commit`. Host scheduling, launch and queue work
that causes commit delay is included. Client transport, serialization and stream
consumption are outside this boundary. A service run at `client_stream` can be
shown next to prediction, but is marked non-comparable at that boundary.

TTFT includes queueing. TPOT is `(last - first)/(output_tokens - 1)` and does not
replace maximum ITL. Throughput uses the modeled serving window. Unserved requests
remain in the SLO denominator. One-token responses have no TPOT/ITL interval.

Operator analysis precedes this service layer. `operators` runs each distinct
physical operator through the same recipes and engine and sums isolated device
times. This sum is neither a serving latency nor an overlap-aware graph span.
Operator errors are prioritized by cumulative effect and trend/ranking failures
before using TTFT/TPOT as service constraints. Structural bugs are corrected
independently of that timing tolerance.

`graph_costs` composes a subgraph or pure target model through `Execution`,
without request scheduling or sampling. It preserves source kernel expansion,
barriers, streams, physical contention and complete cross-DP expert cohorts.
The default excludes host submissions; an explicit alternative includes their
declared costs. It reports graph duration separately from isolated cost sums
and precedence/resource lower bounds. Target capture scope is inherited and
media geometry is explicit. See `graph-analysis.md`.

Expert cohorts synchronize inputs and join completions around each participant's
actual recipe. Collective setup, invocation supply caps and phase dependencies
remain part of those programs; a cohort does not replace them with an aggregate
resource-envelope task.

`call_costs` composes a complete declared callback and finite repetitions through
this same graph path. It excludes serving scheduler/completion work and retains
the chosen eager or replay submission schedule. Historical binding and conditional
event comparisons are separate consumers; see [call analysis](call-analysis.md).

An explicit `ring_collective` recipe adds receive-forward dependencies between
transport rounds on the same Engine. Each rank waits for its own prior source
service and its predecessor's prior arrival, with propagation overlapping other
traffic. It creates no global round barrier. Shared framing is applied once per
hop/message. The current domain is uniform-chunk transport within complete
local groups; cross-DP round coordination and reduction compute are not supplied
by this recipe. See [collective rounds](operators/collective-rounds.md).

## Boundedness

Engine storage is O(active tasks + active pipeline windows + request records).
The active iteration DAG scales with layers and requests, not historical output
tokens. Per-request statistics and final rows scale with request count. Exact
primitive lookups use a bounded LRU and indexed SQLite reads. Large observations
are streamed; the CLI refuses metadata documents larger than 50 MiB.
