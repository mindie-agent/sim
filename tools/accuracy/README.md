# `tools/accuracy` — independent dataset assembly and frozen-holdout scoring

Offline evidence tooling for the A2/A3 accuracy round. It sits **outside** `mindie_sim/`:
the model package produces predictions, this package decides what was measured, freezes
predictions against a geometry-only manifest, and scores the frozen file against the frozen
outcomes. The scorer never imports model code, never fits and never re-selects a variant.

The authoritative field-by-field contract is [`schema.md`](schema.md). This file is the
operating manual.

## Layout

| file | role |
| --- | --- |
| `metrics.py` | the single definition of every identity key and metric: geometry key, shape identity, protocol signature, declared work model, WAPE/APE/coverage/penalized/containment, deterministic cluster bootstrap. Pure functions, no I/O. |
| `sources.py` | declared source registry (role, architecture, die evidence, protocol defaults, family/recipe vocabulary, ingest flag) and one reader per artifact shape. |
| `assemble_dataset.py` | normalises records to `mindie-sim.observations/1`, merges repeats, enforces the split, builds the alias groups, the derived outside-store table and `dataset-report.md`. |
| `freeze_predictions.py` | validates a geometry-only query manifest, records source/protocol/dataset/artifact identity, calls the public query/predict APIs (or imports an existing freeze), writes `predictions.frozen.json` + `freeze-receipt.json`. The `operator` entry kind evaluates one declared condition as `analytic` / `physical` / `curve` for the same timing scope. |
| `score_frozen.py` | verifies the freeze, joins predictions to observations, computes every metric cell, writes `score.json`, `bootstrap.json` and `acceptance.json`. |
| `heldout_manifests.py` | turns the split declarations into worker-ready heldout measurement manifests, validates them with each worker's `--dry-run`, proves they are disjoint from training, and writes the runbook + pre-flight guard. **Never runs a measurement.** |

## Quick start

```bash
# 1. Assemble the observation dataset (read-only over the measurement tree)
python -m tools.accuracy.assemble_dataset --out .workstreams/dataset

# 2. Freeze predictions for a declared geometry-only manifest.
#    `operator` entries evaluate analytic + physical + curve for one timing scope; the
#    physical layer is applied only when the artifact/integration policy applies it, and
#    every variant is recorded (unsupported / low-confidence included).
python -m tools.accuracy.freeze_predictions \
    --manifest queries/a3-test2-queries.json --out freeze/a3 \
    --physical-calibration freeze/train/a3-physical-calibration.json \
    --timing-scope device_event_warm_operator_batched_amortized \
    --curve-artifact freeze/train/curve-artifacts/generic.json \
    --calibration-db freeze/train/calibration-a3.sqlite \
    --allow-dirty                       # only when the tree is dirty on purpose

# 3. Score the frozen file (join only; never a model call)
python -m tools.accuracy.score_frozen \
    --dataset .workstreams/dataset --predictions freeze/a3/predictions.frozen.json \
    --receipt freeze/a3/freeze-receipt.json --out score/a3 \
    --primary-policy curve_first --emit-protocol-acceptance
```

Exit codes: **0** success, **2** acceptance failure (`--enforce` plus a limit flag) or a
partial freeze, **3** integrity violation. See `schema.md` §10.

Round-2 primary policy: `--primary-policy curve_physical_first` (alias `physical_first`)
takes a direct curve only when it is numeric **and** `production_eligible`, else the fitted
physical value that the integration policy applied, else `calibrated_db`, else `analytic`.
The receipt records a timestamp-free `predictions_content_sha256` so a post-measurement
re-freeze against the reassembled dataset can be proved identical.

Preparation for the blinded heldout campaigns (manifests + dry-run + guard, never a run):

```bash
python -m tools.accuracy.heldout_manifests --out .workstreams/heldout-campaign
python -m tools.accuracy.heldout_manifests --guard --receipt F/freeze-receipt.json \
    --dataset .workstreams/dataset          # exit 3 if the freeze is missing or stale
```

## What the dataset looks like

`observations.json` holds exactly one row per distinct physical condition (die + protocol +
recipe + shape). Repeated runs of the same condition merge into `runs` with every raw sample
retained; different protocols or different dies never merge. `duration_ms` is the **median of
per-trial medians** — the scored target — and the pooled-event median the old calibration
store uses is kept beside it as `diagnostics.pooled_median_ms` with the difference.

`dataset-manifest.json` is the census: every source file with its sha256, every excluded
record with its reason, every equal-payload alias group, the split-enforcement section
(including every row a campaign labelled train whose geometry is declared heldout), and the
content hash the freeze binds.

## What the score looks like

`acceptance.json` reports both roll-ups:

* the **all-variant** block — every variant of every condition, nothing filtered;
* the **primary path** — exactly one variant per condition (`curve_first`: direct curve if
  numeric, else calibrated DB, else analytic). This is the headline, because pooling
  `calibrated_db` and `db_and_curve` and counting generic rows several times is how the
  previous round's 17.83 % "overall" number was produced from 15.82 ms of observed time.

Cells are computed per family, per regime band, per die and per protocol signature, with a
cluster bootstrap over conditions (B=10000, seed=20261021) wherever n ≥ 5. Coverage always
divides by the number of **declared** conditions in scope, so an abstention is a coverage
miss and, in `wape_penalized_pct`, a 100 % error.

## A2/A3 symmetry

Nothing in the code knows a SKU. `hardware_key` is read from each artifact, peaks and band
edges come from the split declaration's `regime_rule`, and the family/recipe vocabulary is
per source. Adding A5 is a declaration plus measurements — no code change. The only
architecture-specific data in the package is the source registry: which campaign file lives
where, which physical die its lease exposed, and which protocol it declared.

## Tests

`tests/test_accuracy_tools.py` is Tier 0 only: pure functions, synthetic records, no
subprocess, no network, no device, no real measurement file. It covers the metric contracts
(WAPE denominator, nearest-rank p90, deterministic bootstrap, coverage denominator, interval
containment, tiny-sample guards), identity keys (including the exact declared geometry keys
and the shape/family aliases), the work-model evaluator, protocol canonicalisation, the
assembler's dedupe/split enforcement and the freeze/score integrity refusals.
