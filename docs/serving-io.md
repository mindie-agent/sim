# Serving I/O and output boundaries

`runtime.serving_io` explicitly adds host preparation, host/device copies and
host result processing to ordinary `predict` and `predict_pd`. An omitted
declaration preserves the existing serving execution. Model/body graph probes
ignore this serving policy even when host kernel submission is included.

```json
{
  "runtime": {
    "serving_io": {
      "host_prepare_ms": 0.02,
      "h2d": {
        "bytes_per_s": 24000000000,
        "bytes_per_token": 8,
        "bytes_per_request": 16,
        "fixed_bytes": 0,
        "token_basis": "padded",
        "latency_us": 2
      },
      "d2h": {
        "bytes_per_s": 24000000000,
        "bytes_per_token": 4,
        "bytes_per_request": 0,
        "latency_us": 2
      },
      "host_result_ms": 0.03,
      "client_delivery_ms": 0.2
    }
  }
}
```

The numbers illustrate units; they are not measured defaults. Payload is
`fixed_bytes + bytes_per_token * tokens + bytes_per_request * requests` for one
representative shard. H2D counts the actual microbatch's valid or padded input
tokens, as declared. D2H counts committed output tokens and sampling requests,
including speculative bursts; padding and rejected proposals are not output.
An unfinished prefill chunk has no D2H output transfer.

Host preparation follows scheduling on the same per-replica FIFO CPU resource.
Host enqueue can overlap the input copy; device work waits for H2D readiness.
H2D and D2H have separate declared link resources and contend with their endpoint
stage's HBM. Transfer latency follows resource service. D2H follows the actual
device token commit. Host result processing follows the copy, and the next
scheduled step waits for both device work and host completion. The existing
`runtime.submission.completion_ms` remains a separate completion constraint.

Client delivery is included only when `client_delivery_ms` is supplied, including
an explicit zero. It is a declared per-batch propagation latency, with ordered
per-request delivery. It does not model network bandwidth, serialization work
or client consumption beyond that declaration. Regular decode continues after
host result readiness. A closed-loop successor is released after client delivery
plus its think gap when a client boundary is declared, or after host readiness
when only host I/O is declared. Without serving I/O the existing device-commit
release policy remains. PD pools have independent I/O policies and resources;
both must declare a client boundary to report client metrics for the full request.

## Metrics

`metric_boundaries.device_token_commit` uses actual token-commit events for both
request latency and the throughput window. An enabled I/O declaration adds
`host_result_ready`; an explicit client latency adds `client_delivery`. Each
entry contains the ordinary metric summary, including its own window and SLO
results. Per-request host/client timing appears in `requests[].boundaries`.
One-token replies still have no TPOT/ITL interval. Burst tokens share a delivery
time; their zero intra-burst intervals do not erase the longer preceding gap.

For compatibility, top-level `metrics` retains its previous scheduler-completion
throughput window and device-commit request latencies. Its window is labeled
`throughput_window_boundary: legacy_scheduler_completion`. Use the named
`metric_boundaries` for comparisons with a physical output boundary. In
particular, a final D2H tail changes host/client TPS without changing the device
commit window. The engine's final time also includes pending delivery and CPU
work and is not an interchangeable latency boundary.

## All-step summaries and graph dispatch

`execution.step_summary` accumulates every executed replica microbatch, regardless
of `trace_limit` or `event_limit`. It reports step/token/operator counts and
duration sum, mean, min and max, grouped by phase, replay scope and graph bucket.
PD exposes one summary per pool. Step duration runs from scheduled release to
step readiness; device duration ends after all device operators. Their sums can
include overlap and are not the serving makespan. State consists of aggregates
per configured group, not an iteration or token history.

Declared capture buckets continue to control graph replay. Decode batches that
fit use their padded bucket; batches larger than every bucket execute eagerly.
Explicit Eager mode uses valid rows without graph padding. I/O follows that same
dispatch choice, while output token counts stay tied to real commits. No observed
latency is used to select graph capture, eager execution or a timing correction.
