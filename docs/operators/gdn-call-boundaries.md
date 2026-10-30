# Historical GDN call boundaries

`prediction.call_gdn.make_call` builds standalone device work from explicit
arguments. `observations.gdn_binding.bind_observation` interprets the reviewed
historical producer's arguments and returns that declaration. The builder has
no capture input, and the binding does not read durations or canary outcomes.
Both use the ordinary graph executor through `prediction.call.call_costs`.

The retained capture contains 60 conditions and 2,400 event samples. Its
`timed()` function records NPU events around the entire callback, including
device work enqueued from inside that callback. `device_graph` predictions
therefore remain distinct from `device_event_operator_call` observations.
Host dispatch gaps, allocator behavior, event recording and compiled binary
identity are not inferred from a matching Python function name.

## Supported arguments and device work

| Historical role | Conditions | Complete declared device work | Work outside the callback |
| --- | ---: | --- | --- |
| `gating` | 20 | One fused gating kernel, with FP32 `a`, `b`, `A_log`, `dt_bias`, `g` and `beta` | Input creation/transfers, warmup and event bookkeeping |
| `conv_prefill` | 16 | One native FN convolution launch, including device metadata, initial-state synchronization, bias/SiLU, final state writes | Input/weight/bias/metadata creation, initial state zero-fill, warmup and event bookkeeping |

Gating follows the retained wrapper's 64-row by 8-head tiles and one program
per declared vector unit. A short batch still launches all these programs.
Parameter loads depend on head masks, while `a`/`b` loads and output stores
also depend on row masks. The plan keeps those different masks and the padded
elementwise work. Its elementary vector-operation envelope is an analytical
abstraction, not a reconstruction of the compiler's instruction schedule.
Both outputs are FP32 here. The separately studied served BF16 scalar-store
path and its measured queue parameters do not enter this call.

The convolution producer passes bias and SiLU, equal sequence lengths, device
INT32 query offsets and cache indices `1..batch`, and device BOOL initial-state
flags that are all false. State has `batch+1` rows and `kernel-1` history slots;
row zero is reserved. The source FN planner preserves the generic biased
branch. Device-visible initial-state flags select snapshot workspace and
`SyncAll` even though false flags suppress the initial-state payload copy.
The wrapper uses uninitialized output/workspace allocations and returns the
native result without an additional output copy.

Each returned `GDNCall.ledger` lists the public callable, argument geometry,
device kernels, source references, host-only actions, outside-call preparation
and output/state effects. These fields describe scope; they do not manufacture
an observed binary identity.

## Remaining semantic mismatches

| Historical role | Conditions | Missing complete-call mechanism |
| --- | ---: | --- |
| `conv_update` | 4 | Native `runMode=1`, one query per sequence, caller-provided output and update-state movement |
| `scan_decode` | 4 | Q/K L2 normalization, per-call CPU tensor construction and H2D copy of actual lengths, native recurrent call and wrapper |

The served prefill block still includes QKV concatenation and cache-state
gather/scatter. Those remain serving-caller work and are absent from the
standalone composite. Conversely, reusing only the native recurrent kernel
would omit work explicitly inside the scan callback. The eight remaining
conditions stay visible as `semantic_mismatch`, with their original samples
and canaries intact.

The 16 original prefill scan canaries failed. They are preserved as recorded;
a later canary source does not rewrite this capture. All historical
conditions retain `split="unassigned"`; this implementation creates no
train/heldout assignment.

## Standalone `scan_prefill` composite

`prediction.gdn_composite.prefill` is the shared native prefill composite. The
serving `gdn_prefill.core` caller passes its packed Q/K/V views, its gathered
initial state and its serving gate/beta values; the standalone
`call_gdn` builder passes independent `q/k/v/g/beta/state` tensors prepared
outside the callback. Both paths use the same `gdn_prepare`, `chunk_fwd_h` and
`gdn_output` planners and the same caller materializations:

- Q/K/V BF16 BSND views become BNSD contiguous inputs.
- FP32 `g` becomes BNS. BF16 `beta` is promoted to FP32 and then becomes BNS.
- The initial FP32 V,K state is read by H; H writes the selected final state.
- The wrapper allocates separate `q_hat`, `k_hat`, `q_rstd`, `k_rstd`, `A`
  and `g_cumsum` outputs, even when Python ignores their returned values.
  Q/K hats and g first take their declared transpose/contiguous path, then
  all six values are copied into the caller buffers. A singleton transpose
  may be a free view; distinct destination storage still requires the copy.
- The selected no-sigmoid wrapper does not request `beta_eff`, so Prepare
  does not publish that optional output. `h` is not requested; O and final
  state already write caller buffers and receive no duplicate export copy.

The standalone declaration excludes projections, gating, convolution, cache
lookup and all input initialization. It contains no `qkv_concatenate`,
`gather_recurrent_state`, `scatter_recurrent_state` or serving output copy.
External inputs have no fabricated producer operator: their materializations
carry the source argument, current stream and full input/output byte counts in
metadata, and only the declared copy work enters the DAG.

`observations.gdn_binding` binds a historical `scan_prefill` only when the
reviewed producer SHA, the public callable and every recorded argument agree
with that composite's contract. It derives chunk count and FP32 state bytes
from the recorded batch/tokens/head geometry instead of trusting a label. The
explicit `make_call("scan_prefill", ...)` builder works with no observation.
Prebuilt `cu_seqlens` and chunk-index lists are prepared outside the callback;
the wrapper marshals them inside. The supported scale is `dk**-0.5`, with
no A-log or dt-bias input. Contradictory explicit arguments are rejected rather
than replaced by those selected defaults.

All 16 retained conditions bind and build 18 operators: 15 declared
materializations and three native kernels. This count describes the analytical
DAG, not a claim that the compiled dispatcher launches 18 device kernels.
Source dispatch can fuse or eliminate materializations; no compiled trace
establishes that behavior here.

Independent acceptance corrected the initial extraction's missing caller
copies, unnecessary singleton transposes, extra beta store and mixed-request
ownership. The original DSH run completed 13 of 16 predictions within its
budget, before these corrections. Those durations are superseded and are not
reported as predictions of the corrected implementation.

The six serving controls cover key/value head pairs (1,3) and (2,6), a 65-token
tail, ragged lengths (64,17), and a decode-prefix/prefill-suffix batch. Comparing
all existing OpSpec fields against the pre-extraction graph finds only the
intended `output_beta=False` change; each graph additionally has six caller
copies. In particular native Prepare/H/O retain only prefill request IDs.

For one token and one head, Q/K/V/g transposes retain contiguous storage.
The BF16-to-FP32 beta cast transfers 6 bytes. The six output payloads total
652 bytes, requiring another 1304 read-plus-write bytes. The test independently
checks this 1310-byte materialization total, actual tail rows rather than
chunk-padded output storage, native beta-store suppression and execution after
an external input-ready time of 7 ms. The call/native/tensor/graph focused
suite passed 164 tests in 7.73 seconds on October 29, 2026.

The corrected public `call` CLI was then run for the retained 4096-token,
batch-one geometry with each value-head count. Both reports keep
`boundary="device_graph"`, 18 operators, zero live tasks/programs and
`calibration="none"`:

| v heads | k heads | batch | tokens/seq | predicted ms | local wall seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 16 | 1 | 4096 | 0.662562 | 13.188 |
| 32 | 16 | 1 | 4096 | 0.523786 | 6.835 |

These are corrected device predictions, not event-window accuracy; the other
14 historical scan conditions have not been rerun after the correction.
The complete reports are retained as
`data/verification/gdn-scan-prefill-v48-20261029.json` and
`data/verification/gdn-scan-prefill-v32-20261029.json`.
An independent read-only review additionally checked 24 small geometries
across batch, token tails, key-head count and head grouping, including the
six output payloads and complete dependencies. Original failed canaries and
all observations remain unchanged.

## Accumulated cost and model-error limits

`data/verification/gdn-impact-summary-20261029.json` separates retained model
attribution from the corrected standalone calls. The older `edf9e156` Qwen3.8
prefill16K/batch1 reference has an isolated operator sum of 2368.328 ms:

| Scope within that reference | Isolated contribution | Share |
| --- | ---: | ---: |
| Native convolution plus Prepare/H/O | 126.839 ms | 5.356% |
| All GDN work except projections | 210.103 ms | 8.871% |
| QKVZ, beta/alpha and output projections | 713.817 ms | 30.140% |
| GDN subgraph including those projections, excluding input norm | 923.921 ms | 39.012% |

These rows overlap and are not additive. The four native kernels' retained
ABI-matched medians, projected over 48 modeled occurrences, give -26.802 ms
signed residual and 30.399 ms summed absolute residual: -1.132% and 1.284% of
that old isolated sum. This prioritizes native timing refinement relative to
matrix work; it is not measured whole-model error or a critical-path bound.

The corrected standalone 4096-token calls instead predict 0.662562/0.523786 ms
for 48/32 value heads, against historical full-event sample medians of
0.970389/0.726463 ms. Their conditional differences are -31.72%/-27.90%.
Device work and full event intervals still differ; host/event work and
compiled dispatch are unresolved. The original failed scan canaries remain
failed. No residual is converted to a timing parameter or an accuracy pass.
There is no corrected whole-model run at these same conditions in this
artifact, so the old 16K denominator cannot yield a current model-gap percentage.

## Reproduction and interpretation

The device declaration in `examples/calls/gdn-historical-device.json` selects
the registered nominal A5 physical resources and a one-rank context. The
small model and request fields supply schema context only; the declared
call DAG determines the work. No historical scalar-store fit, aggregate
operator fit, primitive duration or observation lookup enters this case.

```python
import json
from pathlib import Path

from mindie_sim.observations.gdn_binding import bind_observation
from mindie_sim.observations.historical import call_observations
from mindie_sim.prediction.call import call_costs

case = json.loads(Path("examples/calls/gdn-historical-device.json").read_text())
protocol = json.loads(Path("data/observations/protocols/gdn-call-protocol.json").read_text())
for observation in call_observations(protocol):
    binding = bind_observation(observation)
    if binding["status"] == "bound":
        call = binding["call"]
        prediction = call_costs(case, call.ops, recipes=call.recipes)
        # prediction["boundary"] remains "device_graph".
```

The retained prediction artifact is
`data/verification/gdn-call-device-predictions-20261029.json`. It reports each
condition, binding result, device prediction, resource ledger, active-state
counts and scope limitations. It contains no fitted parameter or accuracy
verdict. A separate comparison may describe the residual against the event
span under explicit assumptions; that does not establish strict boundary
equivalence or heldout accuracy.

The public comparison workflow reads the same real capture and preserves all
60 conditions, including the unsupported callbacks:

```sh
mindie-sim compare-calls examples/calls/gdn-historical-device.json \
  data/observations/protocols/gdn-call-protocol.json \
  -o data/verification/gdn-call-alignment-20261029.json
```

This report returns `partial` and exit code 2 while any callback is unsupported.
Its relative differences use the explicitly named sample-median statistic;
the source observations also retain median-of-trial-medians and the
producer's original summary. All predictions remain `device_graph` with
zero strict accuracy pairs.

The earlier October 29, 2026 public CLI run, before scan-prefill integration,
completed all 36 predictions and
retained 24 unsupported rows, with all 2,400 samples still unassigned and zero
failed prediction scenarios. It took 372.28 seconds wall time and 34.797 MiB
maximum resident memory. Every prediction drained its live tasks/programs and
reported no calibration. The earlier direct call run took 374.99 seconds and
35.391 MiB peak RSS; the focused call/native-convolution checks passed 91 tests.

Against the sample-median event span, the condition-wise median relative
difference is -41.520% for biased FN convolution and -94.502% for FP32 gating.
These are conditional differences across different timing boundaries, not
validation scores. The missing host/allocator/event contributions and nominal
elementary vector service are explicit; these observations do not identify
their individual magnitudes. No offset or multiplier is fitted to the gap.

The binder verifies the reviewed producer SHA before interpreting its
implicit defaults. A changed producer requires review of its argument
semantics. This constraint belongs to historical binding: callers can still
run analytical `make_call` declarations without any historical producer,
measurement or compiled binary.

The shared chunk composite is implemented in `prediction.gdn_composite` and
used by both the serving lowering and the standalone `scan_prefill` builder.
QKV concatenation and cache-state gather/scatter remain in the serving caller,
so the historical standalone call consumes independently prepared tensors
without importing extra serving work. The 16 scan-prefill conditions are now
part of the supported set; only the eight conv-update and scan-decode
conditions remain unsupported.

## Source anchors

- `data/observations/protocols/gdn-call-protocol.json` identifies the capture and producer.
  Producer SHA: `1662a58957250fed9043593b62a9c5b558783850c1016f66367cb5737cb43063`.
  Event timing is at lines 212-223; supported callbacks are at 514-535 and
  602-616.
- `data/observations/sources/gdn-image-c3d00a196905/fused_gdn_gating.py.txt`
  retains the image's gating kernel and wrapper at lines 14-101, SHA
  `5587bf3440e64eda125899f5de2d1ab609061e2b5336f847af101ada51d81058`.
- `data/observations/sources/gdn-prefill-image-c3d00a196905/sources.tar.gz`
  retains `ops/ascendc/_stable.py`. Its convolution wrapper is at 1018-1078,
  member SHA
  `d6be8820bb54e0177cf92e396099011402f97066b0974a2f1a8a90187f08aae0`.
  The retained `ops/ascendc/_runtime.py` uses uninitialized `torch.empty`
  workspace allocation at 446-452 and enqueues it on the current stream.
- `data/observations/sources/native-causal-conv-a4a7958/manifest.json`
  records the native FN source at revision
  `a4a795857868729fbdbb0118c1e06c8655a7b31b`.
- The same prefill archive retains `ops/ascendc/_aclnn_ctypes.py`, SHA
  `2aae5a0d7131f78f26a285f2b2cce2cbae3e2001fb20cf7e8fd373daf9e4cd8a`.
  Lines 2884-2958 allocate and forward the six requested output buffers,
  omit beta-eff/h for the selected flags, and allocate final state separately.
- `data/observations/sources/fla-a4a7958/chunk_gated_delta_rule_fwd/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp`,
  SHA `f7be8520e600693bd400f6875cebf6d4da9fd4be597cc60a55168df6f2c4362d`:
  lines 279-286 implement `ViewCopyIfPresent`; 618-669 allocate internal
  tensors and transform inputs; 676-696 execute Prepare/H/O; 699-739
  transform and copy requested outputs. These anchors establish declared
  storage work, not the identity or optimization of the captured binary.
