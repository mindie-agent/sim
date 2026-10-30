# GLM Decode control-path gap after the shared matrix fix

The [W4A8 pipeline correction](w4a8-matrix-pipeline.md) changes the retained
gate/down predictions to 16.110/8.954 µs versus 88.039/52.907 µs observed.
The remaining approximately 82% errors are still unresolved. PMU scalar times
are overlapping activity intervals; neither 54 µs nor 83 µs can be imported as
an additive service time or invocation floor.

This audit used local source and configuration only. It made no runtime changes
and launched no remote process or device workload.

## Ordinary balanced Forward does not automatically select MSD

A read-only `target_graphs(..., phase="decode", tokens=1, context=4096,
scope="body")` check on main revision `12884a12` used the retained GLM W4A8
case, `routing.policy="balanced"`, and `activation_quantization`:

| Software declaration | Gate/up occurrences | Down occurrences |
|---|---|---|
| Retained explicit native selectors | 75 `ascend_w4a8_msd` | 75 `ascend_w4a8_msd_grouped_down` |
| Same case with `runtime.kernels.lowerings` removed | 75 `balanced_grouped_gate_up` plus 75 separate SwiGLU | 75 `balanced_grouped_down` |

The deployment sweep's base configuration also selects ordinary steady matrix
recipes, not these native source programs. Therefore an improvement in the MSD
source branch does not automatically improve every ordinary balanced body.
`native_fusion.lower_native_fusions` exits unchanged without a selected fusion;
the down implementation likewise requires its selector after input quantization.

The present explicit bindings are:

```json
{
  "native_fusion": {
    "implementation": "declared_native_fusion",
    "options": {
      "roles": ["grouped_gate_up_activation_quant"],
      "program_profiles": {"grouped_gate_up_activation_quant": "ascend_w4a8_msd"}
    }
  },
  "grouped_gemm": {
    "implementation": "ascend_w4a8_msd_grouped_down",
    "recipes": ["grouped_down", "balanced_grouped_down"]
  }
}
```

A default execution-policy layer can select these programs from declared
implementation/quantization/layout traits and supported domains. It should
retain the selected implementation and assumptions in output. Balanced routing
alone establishes integer expert rows; it does not establish MSD, per-channel
quantization, a compatible fused output, or the historical loaded binary.

## What is already physically calculable

The down kernel and host tiler match the pinned source hashes in
`w4a8_down.SOURCE`. Their supported branch establishes:

| Mechanism | Known work/geometry | Existing shared mechanism | What is priced now |
|---|---|---|---|
| Group-count bulk DMA/reduction | 256 INT64 entries × 40 participating AIVs = 81,920 bytes; cast/reduce/cast = 20,480 declared vector operations | `Work`, per-owner `Stage`, MTE2_AIV/vector resources | Already included before the row-work guard, including idle AIVs |
| Active INT4 operands and column scales | Eight active banks; 16 gate or 96 down source output tiles; scale 8 bytes/column/tile | Shared matrix hierarchy/steady pipeline, MTE1 directional modes, FP16 publication | Included; no 256-bank weight read |
| Per-owner scalar group traversal | Down `Process` reads each count before `m<=0`; 256 × (20 AIC + 40 AIV) = 15,360 logical INT64 reads | Engine supports independent owner streams; a read-queue policy is still needed | Count is exposed, elapsed service is unresolved |
| C/V output credit | Host `cvParallNum=4`; the AIC waits after four outstanding outputs for AIV acknowledgment | Shared finite pipelines, allocations and Engine events | Current source programs use coarser global phase barriers |
| Vector scratch/assist/output | Source UB chunks, CV11/CV12 rotation and DMA payloads | Existing per-owner vector/MTE2/MTE3 stages and buffer capacities | Bulk service included; UB queue lifetimes and chunk overlap can be refined |

Exact source locations for the four-slot rule are host
`grouped_matmul_tiling.cpp:1912,2146`, kernel
`grouped_matmul_antiquant_a8w4_msd.h:326,359,389,439`. AIC publication must precede
the paired AIV consumption; that consumption releases the reusable slot. This
dependency model needs no invented flag latency. Replacing global barriers can
increase overlap and *reduce* duration, so it is not a guaranteed explanation
of the underprediction.

## Scalar reads and stores are different boundaries

The inspected down branch writes output using `DataCopyPad2DA8W4` and reads
count metadata through GM `GetValue`. It does not expose a per-element GM
scalar-store loop. The compatible local gate mirror contains
`quantScaleLocal.SetValue`, but that is a **local UB tensor** write; it is not
the BF16 UB-to-GM scalar-store path measured for GDN. The gate mirror is a
structural cross-check whose hash differs from the pinned gate declaration.

`transactions.QueuedStore` already models issue serialization, a finite queue,
same-address-granule hazards and response retirement. Reusing its exclusive
store hazard for consecutive read-only `GetValue` addresses would incorrectly
serialize reads to the same cache line. `FiniteTransfer` provides bounded
credits and response lifetime, but has no scalar issue resource, owner-aware
read cache, or source value-dependent scheduling. Those building blocks can
be generalized; their existing class names are not evidence that loads and
stores share parameters.

The retained A5 primitive in
`data/observations/gating-probe-20261028/native-queue-holdout/parameters.json`
is explicitly scoped to BF16 scalar UB-to-GM stores on `Ascend950DT_9582`, with
a retained compiler binary. Its eight slots, 128-byte granule, 91.25 ns response
and issue rate were identified from address/unit isolation. They are not A3
INT64 GM-load parameters and must not become the default GLM group-scan cost.

## Smallest remaining shared extension and required parameters

First, lower the source group traversal into a compact **descriptor-read/control
phase** with independently declared access count, width, stride, ownership and
value dependencies. Preserve empty-group visits and make each selected matrix
rectangle depend on its descriptor. Source count work can be exposed by default
without disguising unknown time as measured service.

Then extend the existing finite transaction machinery with an explicit read
policy: issue capacity, outstanding-read limit, response latency, cache-line
granule/residency policy, and dependencies on returned values. A read line may
be reused only under a declared cache policy. The 2,048-byte backing vector is
not 15,360 independent DRAM misses, nor does the monotone index prove all reads
are serialized. The next address is consecutive; the returned count controls
the empty-group branch and later offsets.

The current A3 profile declares bulk HBM/MTE rates, AIC/AIV counts, clock,
on-chip capacities and invocation minima. It does **not** declare applicable
scalar issue throughput, `GetValue` cache-hit/miss latency, outstanding loads,
or flag-instruction latency. Clock alone does not identify any of those;
source-level arithmetic/branches are not compiled instruction counts. Until
independent parameters exist, this extension should expose unpriced control
work and optional declared scenarios, while analytic simulation continues.

The four-slot producer/consumer dependency is implementable independently now.
The scalar elapsed-time model needs independent primitive parameters; it cannot
be identified by subtracting existing prediction from the whole native kernel.
Any later timing refinement must retain byte/work conservation, owner-local
contention, bounded Engine state, late-input causality, and validation on a
different group-count/active-expert mix from parameter identification.
