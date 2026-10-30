# Host enqueue and graph replay

`predict`, `predict_pd` and `graph_costs(include_submission=True)` use the same
host enqueue mechanism. A stage has a FIFO host queue. Enqueue depends on the
iteration scheduler and the previous enqueue, independently of the previous
device completion. Device execution waits for its enqueue, inputs and stream
predecessor. Thus enqueue service can run ahead and overlap device work.

An explicit `runtime.submission.kernel_launch_ms` takes precedence over the
legacy `machine.device.launch_us / 1000`. A captured region uses
`runtime.submission.graph_replay_ms` when supplied, otherwise the eager
parameter. Full capture charges one replay per pipeline stage per iteration;
piecewise capture charges one replay per declared segment. Capture is enabled
only by the case's graph declaration. Scheduler and completion costs remain
separate parameters.

Explicit host input/result processing, H2D/D2H resources and separately named
output boundaries are described in [Serving I/O](serving-io.md).

```json
{
  "runtime": {
    "submission": {
      "kernel_launch_ms": 0.002,
      "graph_replay_ms": 0.004,
      "scheduler_ms": 0,
      "completion_ms": 0
    }
  }
}
```

These numbers illustrate units and precedence; they are not calibrated values.
The Ascend profiles now exclude the previous nominal 20 us host cost unless the
user declares one. Zero means that undeclared host service is excluded from the
device-path scenario, not that hardware enqueue is free. An explicitly declared
20 us cost can still dominate many small eager kernels.

`execution.submission` reports whether submission is included, eager/replay
counts, aggregate host service and each stage's effective parameters. PD reports
these separately for prefill and decode pools. Aggregate service across pools
or pipeline stages is not elapsed time: independent host queues may overlap.
`graph_costs(include_submission=False)` also excludes asynchronous scheduler
preparation and completion cost.

The retained A5 empty-kernel experiment measures device task durations and
100-launch event spans. It does not isolate host enqueue service. Neither its
empty-kernel durations nor its roughly 4 us event spacing is automatically
installed as a host parameter; see `operators/a5-invocation-measurement.md`.
Device kernel invocation minima remain governed by the separate
`kernel_invocations` / `kernels.invocations` declarations.

`tests/test_submission_policy.py` verifies actual enqueue/device overlap,
runtime override precedence including explicit zero, one replay per captured
stage, independent PD costs with conserved handoff bytes, and exclusion of
background scheduling from device-only execution. These are causal scheduling
oracles, not observations or end-to-end accuracy measurements.
