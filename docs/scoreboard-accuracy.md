# Scoreboard criteria and comparison boundaries

The model scoreboard consumes the shared `mindie_sim.cohort_accuracy` policy.
An observed complete-event mean above 50 microseconds uses an inclusive relative
tolerance of 15%. At or below 50 microseconds, the inclusive absolute tolerance
is 15 microseconds. A small call within that allowance is reported as
`within_launch_floor_tolerance`. This is an acceptance allowance: it does not
attribute a measured launch overhead or add a timing floor to a prediction.

Whole-step comparisons always use relative 15%, including steps shorter than
50 microseconds. Explicit context endpoints are evaluated separately; the
range row passes only when both declared endpoints pass. This does not prove
the criterion for an unmodeled interior context.

## Conditional references remain conditional

The descriptive model scoreboard and the profiler's scored native cohorts have
different comparison boundaries. A scoreboard row marked conditional or
reference does not become a scored cohort because its prediction is nearby.
Its top-level `accuracy_pass` remains null and its scope explicitly excludes it
from the scored-cohort denominator. A homogeneous conditional population can
show the shared numeric result in `descriptive_tier`. That field retains the
conditional boundary and is not a claim of held-out or native-dispatch accuracy.

The four per-call fields use microseconds consistently:

- `observed_per_call_us`
- `predicted_per_call_us`
- `per_call_diff_us`
- `abs_error_per_call_us`

They require a retained positive call count. Missing counts remain unavailable.
Native-family activity and analytic programs may have different grouping;
their coarse-family residual remains available, but dividing either side by
an arbitrary count cannot create a per-call match.

Kimi family totals can contain hundreds of distinct descriptors. Each retained
descriptor receives its own conditional numeric tier. A pooled family mean
does not choose a tier or hide a slow descriptor. The family report retains
signed cumulative error, the sum of absolute descriptor residuals, and the
number of descriptors with and without a numeric comparison. The unresolved
1104-launch KDA population remains unpaired. Separately recovered closed calls
and their stage projections retain their declared boundaries; the stage rows
must not be added again to the whole-call activity.

## Refresh reports without re-executing predictions

The ordinary scoreboard command adds these fields when assembling its retained
manifest. It does not change the predictor:

```sh
PYTHONPATH=. .venv/bin/python tools/model_scoreboard.py \
  data/verification/model-scoreboard-20261029-final-input.json \
  -o /tmp/model-scoreboard-criteria.json \
  --markdown /tmp/model-scoreboard-criteria.md
```

A standalone Kimi descriptor report can be annotated without modifying its
prediction producer or recomputing any durations:

```sh
PYTHONPATH=. .venv/bin/python tools/model_scoreboard.py \
  --kimi-family-report data/verification/model-scoreboard-20261029-mechanisms-kimi-family.json \
  -o /tmp/kimi-descriptor-criteria.json \
  --markdown /tmp/kimi-descriptor-criteria.md
```

The new report records the input and annotator hashes. Original observations,
predictions, cumulative residuals, and their source identities are preserved.
`data/verification/scoreboard-criteria-20261029.json` records a real saved-report
replay; it is evidence of the reporting contract, not a new performance run.
