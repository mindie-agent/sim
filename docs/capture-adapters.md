# Retained activation and grouped-matrix observations

`observations.captures.iter_observations(manifest, root=...)` yields the existing
accuracy sample contract. Source and producer hashes bind interpretation; none
of the original hardware scripts are executed. JSON conditions and profiler CSV
rows are streamed. Captures remain in their existing locations.

Each sample has `sample_id`, `identity`, `duration_ms`, `run_id`, `split`,
`source` and `metadata`. The identity uses the same level, boundary, hardware,
implementation, phase, shape, dtype, layout and execution fields as
`observations.accuracy`. Source JSON pointers or profiler line/interval references
remain available. The capture hash identifies an acquisition run; trials within
one file do not become independent runs.

`iter_conditions` also retains failed/unavailable calls and excluded serving
intervals, with no invented zero-duration samples. `capture_inventory` reports
these conditions separately from readable samples and distinct identities.

## Protocols

| Manifest in `data/observations/protocols/` | Original scope | Normalized boundary |
| --- | --- | --- |
| `activation-call-protocol.json` | 270 activation conditions in each of two complete acquisitions | `npu_event_call` or `npu_event_burst` |
| `grouped-call-protocol.json` | 10 grouped GEMM shapes, eager and one/eight-call replay | `npu_event_call` or `npu_event_graph_replay` |
| `grouped-served-protocol.json` | Selected original c1/c4 serving profiler captures | `profiled_kernel_sum` |

All current imported samples remain **unassigned**. The captures did not declare
training/heldout roles for the current predictor; the old activation builder
also fitted timing parameters from these rows. Prior `measured`, canary, pass
and fitted-score fields never assign a new validation role.

The activation producer constructs tensors from `tokens` and `intermediate`.
`hidden` is an experiment context label, preserved in metadata; it does not add
a tensor dimension. Original hidden/batch/sequence replicates remain separate
source samples even when their physical identities match. Separate SwiGLU/GeGLU
closures contain activation and multiplication and therefore have subgraph
level. A fused or standalone call is an operator-level call boundary.

Each single-call event surrounds one Python API invocation. It can include
launch gaps and output allocation/lifetime behavior; it is not relabeled as an
isolated profiler kernel. Burst samples were recorded as event elapsed divided
by the number of calls. The adapter restores the actual batch boundary by
multiplying by the recorded count, retains that count in execution identity and
discloses the original six-decimal rounding.

The grouped probe's expert histogram comes from the exact producer: the first
`G` experts each receive `tokens_per_expert` rows and remaining experts receive
zero. It is not inferred from an average across all 256 experts. Quantization
and input/weight preparation occur outside the timed call. Eager output lifetime
and captured static graph buffers are different execution identities. An
eight-call graph observation remains one full replay event; the derived
`graph_replay_8calls_per_call` artifact field is not imported as a sample.
The probe phase is `primitive`: its served-shape labels do not make it a
captured serving step.

The serving adapter recomputes selected grouped-kernel sums from raw CSV,
between consecutive sampler markers, using decimal timestamp arithmetic.
It verifies nondecreasing source timestamps and retains only the active interval
ledger; an out-of-order source is an error. Source line references retain a
five-line preview and full count alongside the unchanged raw path.
It does not divide an authoritative whole-step timing among calls or scale
body/MTP contributions to match that timing. Intervals containing prefill-sized
grouped/attention calls stay excluded even if no clean interval remains.
The declared size separator is specific to these c1/c4 captures, not c8 or
arbitrary workloads. Tensor descriptors and call counts survive, but actual
serving `group_list` values and loaded kernel binary hashes remain unknown.
These are explicit `metadata.unresolved_semantics`, so the partial descriptors
cannot score descriptive or heldout prediction comparisons.

## Comparison and workflow

`observations.comparison.comparison_report(campaign)` accepts an observation
iterator and returns two independent sections:

- `descriptive`: exact-identity residuals using original observations, including
  unassigned/training samples. It reports their split counts, run medians and
  source previews. There is no tolerance or acceptance result.
- `validation`: the existing `accuracy_report`; only eligible heldout samples
  participate. Missing semantic fields are excluded, even if a caller changes
  their split label.

Duplicate source samples must agree on their semantic blockers as well as their
identity, timing and split. An older interpretation cannot hide a later missing
semantic field through input ordering; conflicting duplicates are an error.

Prediction identities come from the analyzed implementation and an explicit
binding. Observation identity is not copied into an unrelated prediction to
force a match. A device-kernel estimate cannot score an event-call, replay or
kernel-sum observation. Predictions stay available when observations are
unassigned, absent or semantically incomplete.

```python
from mindie_sim.observations.captures import iter_observations
from mindie_sim.observations.comparison import comparison_report

campaign["observations"] = iter_observations(protocol, root=repository_root)
report = comparison_report(campaign)
```

The import CLI writes a compact report and optionally streams normalized
samples. `--predictions` accepts a normal accuracy-input packet with explicit
prediction identities; without it, import coverage is reported with no invented
predictions.

```sh
PYTHONPATH=. .venv/bin/python tools/import_observations.py \
  data/observations/protocols/activation-call-protocol.json \
  --samples-output /tmp/activation-observations.jsonl \
  -o /tmp/activation-import.json

PYTHONPATH=. .venv/bin/python tools/import_observations.py \
  data/observations/protocols/grouped-call-protocol.json \
  --predictions explicitly-bound-predictions.json \
  -o /tmp/grouped-comparison.json
```

The actual retained import reports in `data/verification/` are:

| Report | Observed import result |
| --- | --- |
| `activation-observation-import-20261028.json` | 1,080 timing conditions: 540 single-call and 540 burst conditions; 24,300 samples, 247 distinct physical/protocol identities, two acquisitions |
| `grouped-observation-import-20261028.json` | 30 timing conditions, 1,200 samples, one acquisition |
| `grouped-served-import-20261028.json` | 21 intervals across two captures: 18 kernel-sum samples and three prefill-contaminated intervals |

These are import results, not timing acceptance. The default replay supplies no
prediction bindings, so it has no descriptive or validation matches. The
protocols are usable by the shared analysis/comparison workflow; exact
event-level host/replay programs and resolved serving route values remain
needed for those domains. Portable tests use unchanged selected raw rows and
the original producer texts; fixture selection provenance is retained alongside
them. No new device execution occurred.
