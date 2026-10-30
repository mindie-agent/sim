# Complete callbacks and historical event alignment

`call` models a declared callback before request scheduling, using the same
recipes, resource services, streams and Engine as `graph` and `predict`.
It needs no capture, binary identity or calibration store. `compare-calls`
separately consumes observations and preserves the distinction between the
original event interval and a device or submitted graph.

## Standalone calls

```sh
.venv/bin/python -m mindie_sim call \
  examples/calls/activation-nominal.json examples/calls/swiglu-separate.json
.venv/bin/python -m mindie_sim call \
  examples/calls/gdn-historical-device.json examples/calls/gdn-gating.json
```

A declaration contains `builder`, `arguments` and an optional `protocol`.
The public `CALL_BUILDERS` registry resolves a structural builder. A new family
does not require another engine or model-name timing branch. `activation`
provides five API compositions; `gdn` provides FP32 gating and BF16 native
convolution prefill. See the detailed
[activation](operators/historical-activation-calls.md) and
[GDN](operators/gdn-call-boundaries.md) contracts.

```python
from mindie_sim import build_call, call_costs

call = build_call({
    "builder": "activation",
    "arguments": {
        "variant": "swiglu_separate", "tokens": 128, "intermediate": 4096,
    },
})
result = call_costs(case, call.ops, recipes=call.recipes, invocations=20)
```

Source calls expand into kernels before repetition. API analytical nodes remain
API analytical nodes: an unknown internal kernel count is not invented.
Intermediate tensors and their traffic remain in every invocation.
The default limit is 8,192 expanded operators for the whole repeated graph.
The report retains compact resource summaries and an optional bounded trace.

| Protocol | Declared execution |
| --- | --- |
| `mode="eager"` | Every expanded node submits to its stream; later host launches can overlap earlier device work. |
| `mode="graph"` | One replay contains all invocations, with one replay submission per participating stage. |
| `include_submission=False` | Device graph only; no request scheduler, host launch or completion tail. |
| `include_submission=True` | The graph plus declared launch/replay service; no serving scheduler or completion tail. |

The hardware case supplies eager launch service. A call's `submission` accepts
only `graph_replay_ms`, and only for graph mode. Explicit request scheduler or
completion fields are rejected. The case's serving submission policy is not
inherited. Repetitions preserve per-stream order; any cross-stream dependencies
must be declared. No implicit callback fence is added between different streams.
Current historical binders use the producer's single stream.

## Compare preserved event spans

```sh
.venv/bin/python -m mindie_sim compare-calls \
  examples/calls/activation-nominal.json data/observations/protocols/activation-call-protocol.json \
  --include-submission -o /tmp/activation-call-alignment.json
.venv/bin/python -m mindie_sim compare-calls \
  examples/calls/gdn-historical-device.json data/observations/protocols/gdn-call-protocol.json \
  -o /tmp/gdn-call-alignment.json
```

Source paths are relative to `--root` (default the working directory). Hashes are
verified and captures stream one condition at a time. Audited adapters bind
arguments and execution geometry, never timing values, canaries, sample roles
or model labels. Identical structural identities can reuse predictions inside
a report. Prediction keys also include the normalized case, builder ledger
and submission declaration.

Every condition retains its identity, source pointer, hardware, event boundary,
canary, run identity, sample roles and statistics. Burst durations are complete
event intervals: the importer restores the producer's recorded per-call average
once, and comparison never multiplies it again. Replay events are never divided
into invented kernel measurements.

Reports contain independent predictions, actual work and parameter provenance,
plus a callback ledger separating device work, outside preparation, host actions
and unresolved native dispatch/binary/event semantics. Conditional signed,
absolute and relative differences use each condition's sample median.
Per-family/protocol distributions describe conditions, not weighted workload
error. Unsupported callbacks and failed predictions remain visible.

The submitted scenario is a declaration, not a confidence interval or measured
upper bound. It does not reconstruct allocator behavior, internal dispatch,
event enqueue gaps, cache state or unrecorded host timelines. Resource parameters
are not adjusted from event residuals. The supplied machine stays visible; the
adapter does not assert that it is the capture's measured configuration.

Validation remains separately unassessed with zero strict pairs. No tolerance,
accuracy pass or new train/heldout split is created. All modeled conditions
return exit 0. Unsupported conditions or failed predictions produce a written
`status="partial"` report and exit 2. The default bounds are 4,096 conditions and
512 unique prediction evaluations. Exhausting prediction evaluations retains
remaining observations with explicit failures. Exceeding condition storage
aborts with an error. Raw captures and strict `accuracy` behavior are unchanged.

## Current historical scope

The 2026-10-29 activation run covers 1,080 conditions, 24,300 samples and
247 unique physical/protocol identities. Device and nominal submission scenarios
produce 494 graph evaluations. This establishes declared API dataflow, not the
loaded CANN kernel binaries.

GDN retains 60 conditions and 2,400 events. FP32 gating and biased SiLU
convolution prefill bind 36 conditions. The other 24 retain missing callback
mechanisms: convolution update, standalone prefill scan transforms, and decode
normalization/length transfer/native recurrence.

Grouped retains 30 conditions and 1,200 events. Its INT8 operands, FRACTAL_NZ
weights and scale tensors cannot use the existing BF16/ND planner. These remain
unsupported complete calls. No scoped result authorizes old-version cleanup.
