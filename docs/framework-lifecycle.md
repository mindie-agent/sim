# Framework lifecycle costs on the predictive timeline

The optional framework profile adds scheduling, input preparation, host
submission, output processing and client streaming to the existing resource
engine. These are executable dependency tasks, not an additive correction to
the final report. They can delay device work, contend for declared CPU service
and change the existing request TTFT, TPOT and E2E metrics.

All costs are explicit profile inputs with a required `source`. No vLLM host
cost, CPU core count or device measurement is inferred from a model name. The
example is a synthetic arithmetic fixture, not an Ascend or Kimi K3 benchmark.

## Run the integrated example

From the repository root:

```bash
python -m mindie_sim predict \
  --input examples/framework/case.json \
  --operator-config examples/framework/operator-config.json \
  --framework-config examples/framework/profile.json \
  --output /tmp/sim-framework.json
```

The case is one request, 16 input tokens and 3 output tokens, on a one-layer
Dense/GQA network with hidden size 16. Its machine and Vector/Cube profiles
come from `examples/operator-plans/mixed.json`. The operator configuration uses
`strict: false`: supported operators select finite plans, while other operators
retain an explicit legacy fallback. A framework profile also works without
`--operator-config`.

Omitting `--framework-config` preserves the previous path. An explicitly
all-zero profile, including `device_launch_us: 0`, also preserves the entire
legacy JSON. A nonzero profile enables the explicit accounting contract below;
missing coefficients then mean zero rather than an inferred runtime default.

## Five task kinds and four affine coefficients

Each task has four optional, nonnegative coefficients, all in microseconds:

```text
task_ms = (fixed_us
         + per_request_us * requests
         + per_token_us * tokens
         + per_kernel_us * kernels) / 1000
```

`fixed_us` applies once per task invocation, not once per request lifetime.
`per_request_us`, `per_token_us` and `per_kernel_us` use the counts below.

| Task kind | Request count | Token count | Kernel count |
| --- | --- | --- | --- |
| `schedule` | Selected requests in this iteration | Valid scheduled input tokens | Executable device OpSpecs, including legacy fallbacks |
| `input` | Same as `schedule` | Same as `schedule` | Same as `schedule` |
| `submit` | Selected requests, for each host call | Valid scheduled input tokens, for each host call | 1 for each eager submission, or 1 for the whole graph replay |
| `output` | Selected requests | Actual newly generated output tokens, summed across those requests | Executable device OpSpecs in this iteration |
| `stream` | 1 request per output burst | Actual output tokens in that request's burst | 0 |

Scheduled input tokens include Decode input and any declared speculative query
work; they are not the padded graph width. Output counts use accepted tokens
plus the fallback/bonus token, clipped to the request's remaining output.
They are not the number of speculative proposals or sampled requests. A
partial Prefill chunk can update request state without generating a token:
output handling still runs, but no empty stream burst is invented.

In eager mode, each executable device OpSpec, including legacy fallbacks,
receives one host submit.
Finite-plan stages do not add host submissions: an eight-stage GEMM plan is
still one device operation. Replaced preparation operations and nonexecuting
witnesses do not receive submissions. A replayed graph receives one host call
for all its device operations. Multiple native calls hidden inside one OpSpec
are currently refused rather than billed as one.

## Dependencies and sharing

The iteration's tasks follow these dependencies:

```text
previous engine output commit -> schedule -> input -> host submit(s)
                                                   -> device work
device/sample completion -> output processing -> next engine iteration
                          -> ordered client stream burst
```

Eager host submissions form a CPU sequence. Each device operation waits for
its own submission and its original graph predecessors. Submission of a later
operation does not wait for the preceding device operation to finish, so CPU
submission and device service can overlap. A graph uses one replay submission
gate and retains the device graph's own dependencies.

Output processing must finish before request state is committed, KV state is
released and the next engine iteration can proceed. Per-request stream tasks
wait for output processing and the previous stream burst of that request.
They do not gate subsequent engine scheduling. Their completion may still be
delayed by other work sharing the stream resource.

`cpu_scope` and `stream_scope` are independently `process` or `replica`.
Process scopes use the shared `cpu` and `dma` resources; replica scopes use
`r<replica>:cpu` and `r<replica>:dma`. These are unit-capacity service resources
with the engine's sharing policy, not a measured CPU-thread scheduler. Existing
`PrepareSpec` tasks can contend on the same process CPU/DMA resource. CPU and
stream resources remain distinct, so streaming does not implicitly occupy the
scheduler CPU. Resource identity, not a label in a report, determines sharing.

The token-budget scheduler remains synchronous, with one in-flight engine
iteration per replica. This feature does not add asynchronous multi-batch
scheduling or streaming/chunked device output.

## Device dispatch and accounting ownership

Explicit host tasks replace the eager cost model's previous host launch
intercept. `device_launch_us` declares the remaining device dispatch gap per
operation. Finite plans and ordinary operators both receive an explicit device
dispatch gate, charged once. Existing service work is retained without scaling
it to hide a second dispatch charge.

In graph mode the upstream graph device dispatch is retained; it is currently
the declared 2.5 microseconds per kernel, separate from the one host replay.
The profile's eager `device_launch_us` does not override that graph term.
An ordinary, uncalibrated communication operation retains the fixed setup
latency of the physical link selected by the existing cost model. Fabric setup
is not host submission and must not disappear when a host intercept is removed.

The opt-in path explicitly refuses costs whose host/device boundary cannot be
separated reliably: host/block-interval timing scopes, applied physical rows
with such or unknown actual scopes, MC2 composition, whole-span completion
envelopes, multiple-launch OpSpecs, and communication calibration/stage plans
with inseparable intercepts. A plan already including dispatch is also
refused. This prevents silently adding host work to an already measured span.
Rejections use the CLI's existing structured error JSON and nonzero exit code.

## Engine and client outputs

The original `metrics`, request timestamps and `slo_goodput` keep the
**engine output/state-commit** scope. Framework work before that commit is
included in those metrics. Client delivery is reported separately:

| JSON path | Meaning |
| --- | --- |
| `framework.profile` | Effective explicit coefficients, sources and scopes |
| `framework.metric_scope` | Scope of the existing engine metrics |
| `framework.client_metric_scope` | Ordered stream-completion scope |
| `framework.client_metrics` | TTFT, TPOT, burst ITL, E2E and throughput from client delivery |
| `traces[i].framework.submit_tasks` | Host call IDs; use this to verify eager vs graph counts |
| `traces[i].framework.device_end_ms` | Device/sample completion before output processing |
| `traces[i].framework.engine_commit_ms` | Output handling and engine state commit |
| `traces[i].framework.client_stream_end_ms` | Per-request delivery completion |
| `traces[i].events` | Actual task ready/start/end times, demands and piecewise resource service |
| `requests[i].engine_token_event_times_ms` | Per-token engine timestamps, including speculative bursts |
| `requests[i].client_token_times_ms` | One client timestamp per output burst |
| `requests[i].client_token_event_times_ms` | One client timestamp per generated token |
| `requests[i].client_ttft_ms/client_tpot_ms/client_e2e_ms` | Per-request client metrics |

A speculative burst can contain several tokens at one timestamp. Existing
`itl_ms` and `framework.client_metrics.itl_ms` describe burst intervals;
per-token event arrays preserve the zero intervals within a burst. A request
with one output token has no TPOT interval. Client completion does not delay
engine cache release or change the engine-scoped SLO decision.

## Reproducible synthetic result

The bundled eager example uses one microsecond of device dispatch per OpSpec.
Its three engine token commits are **63.0436, 109.4932 and 155.9588 us**.
Each stream burst costs `5 + 0.1 * 1 = 5.1 us`, giving client delivery at
**68.1436, 114.5932 and 161.0588 us**. Mean TPOT is **46.4576 us** for both
scopes in this case. The second iteration begins at the previous engine commit
63.0436 us, before the first client burst finishes at 68.1436 us.

This example has 11 executable OpSpecs and 11 eager host submissions per
iteration. Across three iterations, 18 of its 33 operator records select fine
plans; 15 explicitly fall back. These are coverage and arithmetic checks, not
an accuracy or speedup claim. No NPU or host-runtime calibration was performed.

A separate deterministic test uses an injected opaque 3 ms device backend.
With schedule 0.5 ms, input 0.5 ms, submit 2 ms, output 1 ms and stream 5 ms,
engine token times are 7 and 14 ms, while client delivery is 12 and 19 ms.
The second iteration starts at 7 ms. Those hand-calculable test values are
independent of the Dense CLI example above.
