# Execution-plan identity and evidence replay

`execution-replay` adds a strict, report-only comparison layer around the
existing `operator-plan` graph. It records a complete plan identity, the legacy
and fine isolated durations, the actual shared-graph execution span, fallback
coverage, and explicitly matched whole-kernel observations.

It does **not** modify execution costs, import measurements into SQLite, fit
stage parameters, or distribute a whole-kernel observation across stages.
Existing `predict`, `KernelPlan.to_dict()` and default `operator-plan` outputs
retain their behavior and JSON fields.

```bash
python -m mindie_sim execution-replay \
  --input examples/execution-replay/mixed.json \
  --output replay-result.json
```

This bundled case has synthetic capacities and assumed Vector/Cube profiles,
with **no evidence**. It is a runnable CPU validation example, not NPU timing
data. Its original sim JSON includes three plans and a 0.0550192 ms graph;
heldout accuracy is unavailable (`count: 0`, error fields `null`).

## Input and supported measurement scope

The wrapper has four fields:

```json
{
  "schema_version": 1,
  "operator_plan": {"...": "an unchanged operator-plan input"},
  "protocol": {
    "timing_scope": "device_event_warm",
    "timer": "device_event",
    "execution_mode": "eager",
    "calls_per_sample": 1,
    "warmup": "warm",
    "completion": "isolated_whole_kernel",
    "includes_dispatch": true,
    "cache_state": "declared_unspecified"
  },
  "evidence": []
}
```

The full runnable `operator_plan` is in the example file; the placeholder above
is explanatory. `evidence` is optional. The protocol is mandatory and all eight
fields are required. `timing_scope` must equal the effective
`operator_plan.context.timing_scope`, whose default is `device_event_warm`.
Scope strings are not normalized or wildcarded. `cache_state` is a nonempty,
explicit declaration and is matched exactly.

The first version supports only **one eager, warm device-event whole-kernel
call, including the modeled device dispatch**. Batched/amortized, graph,
host-wall, cold and dispatch-excluded inputs are rejected. This is not a
completed adapter for all existing calibration protocols. A protocol declares
the intended measurement boundary; it does not certify that an evidence source
actually used it.

## Complete plan identity

Each selected fine plan has `identity` and `identity_sha256`. The fingerprint is
SHA-256 over finite JSON serialized with sorted mapping keys and preserved
array order. It locks:

- backend/version and the complete plan: stages, dependencies, work with units,
  rate limits, duration floors, dispatch ownership, buffers, tiling and provenance;
- operator shape lineage, work, effective and declared dtype, layout, recipe,
  stage/layer and metadata;
- the effective context, declared estimator/profile configuration, complete
  resolved hardware description and its independent `hardware_sha256`;
- exact timing scope, protocol, device launch term and millisecond duration unit.

The full hardware SHA is always present, including when the legacy
`hardware_key` is empty. An empty key therefore does not create a hardware
wildcard. The hardware remains the caller's declared/resolved hardware, not
physical-device attestation.

Runtime op IDs, request IDs, external predecessor dependencies and evidence
source/sample ID/outcome are excluded from the kernel identity. They do not
change the isolated kernel. Identical kernels at two op IDs may match the same
observation, which is counted once in aggregate errors.

This is deliberately a **strict configuration lock**, not a portable semantic
equivalence hash. It retains profile source descriptions, local provider paths,
other declaration text and JSON number representations. Changing any of those
can refuse an otherwise equivalent plan. A matching profile path or source
description does not mean the profile has been calibrated; the plan's original
assumption/provenance fields remain intact.

Public API:

```python
from mindie_sim.execution_replay import (
    execution_plan_identity, identity_sha256, execution_replay_report,
)

identity = execution_plan_identity(
    kernel_plan, op, machine, context,
    estimator_config=declared_estimators, protocol=declared_protocol,
    launch_ms=device_launch_ms,
)
fingerprint = identity_sha256(identity)
report = execution_replay_report(wrapper)
```

## Three timing witnesses

All durations and timestamps are milliseconds:

| Replay operator field | Meaning |
| --- | --- |
| `legacy_estimate_ms` | Existing cost-model scalar estimate, retained as a separate reference. |
| `legacy_isolated_ms` | Existing legacy cost lowered through the original resource engine with external predecessors removed. |
| `fine_isolated_ms` | Fine plan isolated duration, including modeled dispatch. `null` for fallback/preserved operators. |
| `shared_execution.dependency_ready_ms` | Maximum completion of the operator's actual external predecessor IDs. |
| `shared_execution.first_admission_ms` | Earliest internal task admission, including dispatch if present. |
| `shared_execution.complete_ms` | Actual operator completion in the shared graph; staged fallbacks may complete at a terminal stage instead of a parent gate. |
| `shared_execution.span_from_ready_ms` | Completion minus dependency-ready time: includes resource waiting after readiness, excludes predecessor execution. |
| `extra_execution_span_ms` | Shared span from readiness minus fine isolated duration. It is a signed execution difference, not a causal attribution of all delay to contention. |

A few-ULP subtraction residual in `extra_execution_span_ms` is reported as zero.
The full original graph and piecewise service rates are under
`operator_plan.execution`. Its `events` are named `trace` in this command.

For the serial synthetic example:

| Op | Legacy scalar | Legacy engine isolated | Fine isolated | External predecessor ready | Shared span from ready |
| --- | ---: | ---: | ---: | ---: | ---: |
| pre | 0.011240 | 0.010240 | 0.011752 | 0 | 0.011752 |
| gemm | 0.050152 | 0.049152 | 0.0371472 | 0.011752 | 0.0371472 |
| post | 0.005608 | 0.004608 | 0.006120 | 0.0488992 | 0.006120 |

The legacy scalar has an additive launch term, while the original legacy engine
may overlap that resource demand with service. Both are shown without changing
that legacy behavior. The final op's absolute completion at 0.0550192 ms is not
its kernel duration. All three extra execution spans in this serial graph are
zero. Independent concurrent kernels can have positive extra spans; the tests
check that case using the original engine.

## Evidence contract and error summaries

Each evidence record requires **all** these fields:

```json
{
  "sample_id": "collection-unique-id",
  "identity": {"...": "the complete identity captured for the actual executed kernel"},
  "identity_sha256": "sha256:...",
  "source": "explicit collection source",
  "split": "heldout",
  "source_class": "synthetic",
  "synthetic": true,
  "duration_ms": 0.01
}
```

The identity shown above is a placeholder. A real record must contain the full
identity and a correct hash, including hardware, dtype/layout, scope and protocol.
**Do not simply copy a modeled identity onto a different kernel measurement.**
Collection must establish those actual conditions. Op ID or shape-only records,
incomplete identities, stale profiles/hardware/protocols, duplicate sample IDs,
nonpositive/nonfinite durations and contradictory source-class/synthetic flags
are rejected. `source_class` is exactly `measured` or `synthetic`, and `synthetic`
must agree. An error percentage outside floating-point range also rejects the
sample explicitly; finite large-duration aggregates use scaled arithmetic.

Bad evidence records appear under `evidence[]` with `status: rejected` and a
reason; they do not prevent the independent analytical graph from running and
never enter error summaries. Invalid wrapper/protocol inputs return the CLI's
structured error with exit code 1. Fallback/preserved records have no fine
identity, fingerprint or fine time, and cannot be matched as fake fine plans.

Only exact identity matches that are declared `measured`, non-synthetic and
`split: heldout` enter `accuracy.heldout`. `train` observations appear separately
in `train_diagnostics`, and synthetic matches only validate the data flow. The
reported WAPE is `100 * sum(abs(prediction-observation)) / sum(observation)`;
P50/P95 are percentiles of per-observation absolute percentage errors. No
eligible data gives `count: 0` and `null` error values.

The comparison uses **isolated whole-kernel** predictions, never shared elapsed
time. Neither training nor heldout outcomes affect any cost or stage parameter.
Splits and collection provenance are caller declarations; this command cannot
certify collection authenticity or training independence. The tests exercise
those labels with constructed data and do not demonstrate NPU accuracy.

```bash
python -m pytest -q tests/test_execution_replay.py
```
