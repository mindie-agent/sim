# Independent accuracy regression

`observations.accuracy.accuracy_report` compares operator, subgraph and pure-model
predictions without fitting parameters or running hardware. The same comparison
contract applies at all three levels, while their identities and boundaries stay
separate. It complements `modeling_priority.py`, which ranks predicted costs, and
service validation, which compares request metrics.

No measurements are required to produce or retain a prediction. A report with
no matching heldout observations has unpaired points and no timing verdict.
Input `passed`, `accepted`, previously calculated errors and similar fields are
not used to compute outcomes.

## Input contract

An input packet contains `predictions`, `observations`, optional
`training_sample_ids`, `tolerances`, `relations`, `provenance` and `limits`.

Each prediction has:

- `point_id`: unique report identifier, also used by relation declarations.
- `identity`: `level`, `boundary`, `hardware`, `implementation`, `phase`,
  `shape`, `dtype`, `layout` and `execution`. Each field must have an explicit
  nonempty value. A truly inapplicable field uses an explicit `not_applicable`
  declaration; missing/null/empty fields remain unknown. Extra identity fields
  also participate in exact matching.
- `duration_ms`: predicted duration for that declared boundary.
- `occurrences`: optional positive integer, default one; used only for repeated
  residual accounting.

Each observation has the same `identity`, an immutable `sample_id`, `source`,
`run_id`, `duration_ms` and `split`, plus optional `metadata` for trial/protocol
details. Only `split: heldout` can score validation.
Known `training_sample_ids` are excluded even when a row is marked heldout.
Conflicting repeats or conflicting splits for the same sample ID are errors.
An explicit `metadata.unresolved_semantics` list also keeps a sample out of
validation: a missing route-value contract is not resolved by changing a split.
The adapter must retain source-derived sample IDs and calibration provenance;
the framework cannot discover undisclosed training use from timings alone.

Exact identity matching includes the entire declared boundary and configuration.
A device-kernel observation cannot score an operator wrapper, subgraph or model.
Unknown required identity fields remain incomplete and unpaired. There is
no nearest-shape lookup and no automatic equivalence between implementations.

Within a matched point, samples first form a median per supplied `run_id`, then
a median across those groups. Additional samples within one run cannot outweigh
other run groups. A run ID is a declared grouping, not proof of experimental
independence. Reports retain source-linked samples, metadata, group medians and
separate sample/group relative-error ranges;
neither dispersion nor timing tolerance is a confidence interval.

## Residuals and tolerances

For predicted duration `p`, heldout group-median duration `o` and occurrence
count `n`, the report computes:

```text
signed residual          = p - o
absolute residual        = abs(p - o)
relative error           = (p - o) / o
repeated signed residual = n * (p - o)
repeated absolute error  = n * abs(p - o)
```

Domain summaries keep signed and absolute totals separately. They group points
only when identity fields other than shape are identical, and retain their
selected observed-duration denominator. These sums describe the selected cases
and supplied repetitions. They are not a whole-model error bound, critical-path
impact or statistical population accuracy.

`tolerances.absolute_ms` and `tolerances.relative` are caller-declared acceptance
criteria for the **point median**:

```text
allowed error = max(absolute_ms, relative * observed_median_ms)
```

An omitted component contributes zero. With both omitted, the report calculates
residuals but leaves `within_tolerance` unset. Timing outcomes are counts of
paired points within/outside/without declared tolerance, never a global pass
that hides unpaired cases. Individual run-group errors remain visible even
when the median satisfies the declared criterion.

## Shape trends and configuration ordering

Relations reference prediction IDs. They require at least two fully paired
heldout points; training-only points do not complete a relation.

```json
[
  {
    "id": "token_scaling",
    "kind": "trend",
    "axis": "shape.tokens",
    "points": ["small", "medium", "large"]
  },
  {
    "id": "parallel_options",
    "kind": "ranking",
    "vary": ["execution.tp", "hardware.device_count"],
    "points": ["tp1", "tp2", "tp4"]
  }
]
```

A trend changes one numeric shape coordinate and requires every other identity
field to match. It reports adjacent direction agreement/reversal and the error
in predicted growth ratios. Direction agreement alone does not establish
correct growth magnitude.

A ranking changes only explicitly listed axes and keeps workload shape fixed.
Level, boundary and phase cannot be declared as varying axes. Other undeclared
domain changes produce `domain_mismatch`, not a timing comparison. Pairwise
ordering uses the declared `order_absolute_ms` / `order_relative` tie tolerance,
defaulting to exact ties when both are zero. Observed ties remain unresolved;
they do not prove either candidate better. A reversed order or a predicted tie
where observations distinguish candidates is reported as a disagreement.
Ranking counterexample output is capped at twenty pairs.

## CLI and retained GPU replay

For an existing input packet:

```sh
mindie-sim accuracy accuracy-input.json -o /tmp/accuracy-report.json
```

The source-tree `tools/accuracy_report.py` also accepts the same packet and
provides the retained-copy reconstruction convenience shown next.

The bundled adapter uses the existing 4090 capture and its frozen copy service
model. It does not read prior pass flags or prior error scores. It verifies the
capture and retained producer/PTX identities, preserves the original split and
excludes the frozen model's recorded training sources.

The original CUDA event spans 24 serial launches. The adapter reconstructs its
batch elapsed time as `recorded duration_ms * 24` and runs a corresponding
24-copy serial graph through public `graph_costs` using the frozen service curve.
Consequently it compares **event-batch durations**, rather than relabeling a
single-kernel prediction as a full callback or device event.

```sh
PYTHONPATH=. .venv/bin/python tools/accuracy_report.py \
  --copy-replay data/measurements/gpu/service-curve-validation-20261028.json \
  --relative-tolerance 0.05 \
  -o /tmp/copy-accuracy.json
```

The 5% value is an explicit illustrative criterion for this batch median, not a
universal operator target. The retained result is
`data/verification/accuracy-copy-20261028.json`. It has one heldout shape and
five trial event batches grouped under **one capture run**. The capture hash is
the run ID; trial indices remain sample metadata. Fifteen training samples at the other three shapes
are excluded. The batch-median relative error is approximately -0.324%; individual
trial-sample relative errors remain approximately -6.162% to +2.904%, so the
median result must not be read as every trial meeting 5%.

Scaling trends and implementation ordering remain **not assessed**: there is
only one heldout size and one measured implementation. These are trials within
the same capture, not independent capture runs. The adapter retains the existing
limitations concerning unrecorded binary identity, host launch gaps/cache state,
the caller-declared resource ceiling and generalization beyond this copy
protocol. No new hardware measurement or higher-level model acceptance is implied.

## Descriptive comparisons of unassigned observations

`observations.comparison.comparison_report` runs the same exact-identity join
for descriptive residuals while keeping the unchanged independent
`accuracy_report` as a separate `validation` section. Descriptive output has
source/split counts and residuals but no tolerance or acceptance conclusion.
Missing semantics remain unpaired; no prediction is disabled.

`capture-adapters.md` describes the retained activation/grouped source adapters,
their call/burst/replay/kernel-sum boundaries and the streaming import tool.
They keep the original samples `unassigned`; the importer does not decide a
training/heldout split on the user's behalf.
