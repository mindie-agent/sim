# Invocation, communication and fusion evidence

This audit uses committed runtime `8cc3ff0b` and retained captures. Its compact
record is `data/verification/invocation-evidence-20261029.json`, including input
hashes, complete observation identities, controlled differences and work totals.
It does not fit parameters, reassign observations to training, or run hardware.

## Timing ownership

| Existing mechanism | Actual scope | What it does not identify |
| --- | --- | --- |
| `Device.launch_us` | Host FIFO service per eager expanded node, or fallback graph replay submission | Device kernel startup |
| `SubmissionPolicy.graph_replay_ms` | One host submission per captured stage/segment | Every kernel inside the replay |
| Scheduler/completion declarations | Serving scheduler and final completion boundary | Primitive callback execution |
| `ResourceService.startup_ms` | Response-model startup within its declared one-resource identity and work domain | A recovered instruction latency or generic invocation startup |
| `PrimitiveStore` duration | Implementation-matched device completion constraint from original readiness | Additional work or an additive delay after another floor |
| Ring/pairwise propagation | Each actual routed hop after wire service | Host submission or collective API control |
| `InvocationTiming` | Explicit device invocation startup and minimum completion constraint | A value inferred from an unexplained event residual |

The new invocation wrapper charges causal startup before body expansion.
Its completion constraint starts at original readiness, so the uncontended
duration is `max(minimum_ms, startup_ms + body_ms)`. The caller removes the
body's enclosing primitive floor before wrapping. Inner phases and tiles do not
inherit this boundary; a call expansion wraps its executable kernels, while a
logical join has no invocation cost.

Host, device and response-model parameters need separate ownership. A
`ResourceService` already includes its declared startup and total response.
Adding another parameter attributed to that same startup would double count it.
A caller must either choose one owner or declare independently identifiable
mechanisms.

Parameter selection should prefer a complete implementation/kernel identity,
then an explicitly declared recipe or family scenario. Hardware, implementation
revision, dtype/layout and replay scope belong to a measured identity. Model
names do not. An omitted parameter stays zero and does not block analytic work.

The isolated DSH commit `32be8ad1` and acceptance correction `53447718` implement
this wrapper. Acceptance found and corrected early body expansion when startup
was zero but input readiness was unresolved. A configuration typo or explicit
null is no longer silently converted into zero. The focused invocation,
resource-service and collective checks passed 57 tests. An additional 36-case
independent readiness/startup/floor/work recurrence matched, preserved physical
work and ended with zero live tasks. These are mechanism checks; public predictor
integration is a separate mainline change.

Independent mainline integration checks subsequently found two issues: an idle
collective sender lost startup despite receiving peer traffic, and an analytic
outside-domain response was incorrectly rejected as a duplicate startup. Both
were corrected. Nine public invocation checks and two public fusion controls
then passed. The compact record labels these working-tree checks separately
from the committed baseline inventory.

## Grouped events do not separate host and device startup

The retained grouped producer surrounds a complete callback or replay with NPU
events. Graph1 and Graph8 use separately captured graphs; Graph8 contains eight
sequential calls. Input preparation, quantization, conversion and capture occur
outside the timed callback. Complete-event samples remain `unassigned`.

For the small-group conditions, eager minus Graph1 is approximately
9.47--10.32 us. At 256 active groups the sign reverses: Graph1 exceeds eager by
3.65 us for gate/up and 5.25 us for down. Graph8 divided by eight is only
65.0--70.6% of Graph1 for the small-group conditions, rising to about 96% at
256 groups. This is descriptive normalization, not independent per-call
observations.

At M=32/G=8, Graph8 minus Graph1 is 172.552 us for gate/up and 167.357 us for
down. A one-time replay addition would cancel in this subtraction. This rejects
that mechanism as the sole explanation, but does not recover an additive
per-kernel startup. Queuing, dispatch, source control, native implementation,
cache behavior and event enqueue gaps remain mixed. The source-supported
matrix/output work is useful without filling that difference with a floor.

See `grouped-gap-analysis.md` and `grouped-dispatch-evidence.md` for the retained
producer and compatibility limits. A compatible open-source implementation does
not establish which historical binary or tiling executed.

## Real communication captures and their usable boundary

Two completed campaigns retain 72 conditions each:

- W16: rack06os5/os6, two nodes and 16 Ascend950DT_9582 ranks.
- W32: rack06os4--os7, four nodes and 32 ranks.

Both use BF16 HCCL with CANN 9.1.0, torch 2.10.0+cpu and torch_npu
2.10.0.post6. Rank placement, communicator readback, payload sizes and producer
protocol are retained. The selected HCCL engine is explicitly `not_observed`.
The new predictor does not inherit the campaigns' fit/train assignments.
Original 64 KiB/16 MiB fit holdouts remain recorded as historical metadata.

The producer preallocates buffers, records a start event, enqueues a batch of
asynchronous collectives, waits on those work handles, records an end event and
synchronizes. Its reported per-op time divides the complete event by batch
repetitions. Separate single-op conditions use one repetition. The original
host wall window also encloses event recording and synchronization, so subtracting
event time from host wall time is not a CPU launch measurement.

Selected medians in microseconds:

| Placement | Collective | Payload per rank | Batch/repetitions | Single event | Difference |
| --- | --- | ---: | ---: | ---: | ---: |
| W16 | Allreduce | 1 KiB | 122.519 | 376.210 | 253.690 |
| W16 | Allreduce | 1 MiB | 164.739 | 407.557 | 242.818 |
| W16 | Alltoall | 1 KiB | 203.806 | 399.335 | 195.530 |
| W16 | Alltoall | 1 MiB | 218.068 | 395.260 | 177.192 |
| W32 | Allreduce | 1 KiB | 160.401 | 428.432 | 268.031 |
| W32 | Allreduce | 1 MiB | 212.545 | 456.942 | 244.397 |
| W32 | Alltoall | 1 KiB | 339.281 | 553.255 | 213.974 |
| W32 | Alltoall | 1 MiB | 370.592 | 529.210 | 158.618 |

The batch uses 256 repetitions at 1 KiB and 16 repetitions at 1 MiB.
These differences demonstrate a protocol-sensitive call boundary. They cannot
separately price host launch, device startup, stream waits, peer entry or transport.
The existing affine campaign fits also cross a payload regime transition;
reported W32 heldout error reaches 47.6%. Their intercepts are not link
propagation constants, and their per-rank tensor slopes are not physical wire
bandwidths.

There is nevertheless a meaningful topology finding. The retained topology,
URMA registration logs and cross-node P2P throughput support UB/URMA transport
for these rack06 campaigns. The source report labels this as an inference, since
no device transport-counter trace was collected. The A5 SKU's automatic
inter-node default still declares RoCEv2_400G, 50 GB/s and 5 us; its intra-node
default declares UB_MESH, 392 GB/s and 1.8 us. Both have
`bandwidth_provenance="declared_peak"` and no calibrated bandwidth.
Explicit case links override these defaults. This finding therefore applies to
automatic profile materialization, not every example or deployment.

The next useful absorption boundary is a typed **collective response** with
fixed runtime, participant placement, payload semantics and invocation protocol.
It can reuse the existing response-model and common-engine concepts while
preserving its distinction from hop service. First compare complete original
events, retain unknown native algorithm, and separate local memory/control work
from transport ownership before combining a response constraint with a causal
ring/pairwise plan. This audit does not implement that extension or import the
historical fitted parameters into links.

## Fusion removes only private intermediate transfers

The committed ordinary frontend already applies
`activation_quantization -> norm_quant -> residual_norm` when declared in
`runtime.lowering_passes`. The generic passes require compatible dependencies,
stage/stream and geometry. A separately used normalization output prevents its
removal. A saved residual remains an output for its later consumer. Explicit
source plans are not rewritten by these generic passes.

The smallest useful representative is:

```text
residual_add -> rmsnorm -> MX quantize -> projection -> later residual consumer
```

For a 2x8 BF16 activation, an independent byte oracle removes 96 B:
32 B normalization output write, its 32 B quantizer read, and the 32 B residual
read by normalization. The 32 B saved residual write remains.
The real public graph reports HBM service 552 -> 456 B, with unchanged 512
matrix FLOPs and 180 vector operations, and zero live tasks. Its synthetic
resource-control span is 0.002616 -> 0.002536 ms; this is not measured accuracy.

The same ordinary catalog path, with only the pass list changed, gives:

| Model/probe | Removed HBM bytes | Removed operators |
| --- | ---: | ---: |
| Qwen3.8 prefill256 | 754,974,720 | 208 |
| Qwen3.8 decode1/context2048 | 2,949,120 | 208 |
| Qwen3.5 prefill256 | 167,772,160 | 120 |
| Qwen3.5 decode1/context2048 | 655,360 | 120 |

Matrix FLOPs, vector work and link payload are unchanged in all four controls.
These are declared-work comparisons, not full-model execution or speedup claims.

`examples/studies/norm-quant-fusion.json` is an executable two-candidate scenario
using the same catalog case. Only the lowering-pass list changes. Its prefill256
and decode1/context2048 probes are declared analysis workloads, not relabeled
capture events. The public `study` CLI completed all four evaluations with no
failures during integration. The compact record retains work and boundary
results; timing from that unfrozen development run is not promoted to a final
performance benchmark.

The retained `AddRmsNormDynamicMxQuant` rows expose FP8 payload, BF16 residual
and E8M0 scale outputs matching the known fused tensor geometry. They also list
a fourth FLOAT output with an empty shape. Thus the rows support the known
payload arithmetic but do not establish a complete output contract or a strict
whole-kernel timing comparison.

The present `mc2` recipe partitions dispatch/compute/combine into finite chunks
and streams. It retains the operators' physical HBM work and is not an
implementation of communication/GEMM fusion. Likewise, source MoE alignment
retains the packing copy and source combination retains required layout
materialization. A profiler subgraph label alone cannot remove either copy.
A new fused MoE path needs tensor identities, local residency/lifetimes,
communication ownership and all externally consumed outputs before changing
work or invocation count.
