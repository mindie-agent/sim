# A3 operator shape campaign reference

This directory contains the configuration and commands for a nine-model
campaign. Generate the full plan and CPU dry-run locally before consuming
their outputs. **There are no measured A3 samples checked in here.** Real runs
require an externally managed device supplied by the coordinator.

Checked-in inputs:

- `campaign-config.json` — models, representative TP/EP/PP topologies, demand
  cases, sweep protocol and budget.
- `native-next-wave-capability.json` — four buildable native capability
  candidates (lastdim softmax, masked softmax, dynamic int8, grouped BF16 list)
  plus explicit pending FA/RoPE entries. Its small CPU dry-run example is
  `results/native-next-wave-capability-dry-run.json` (no measured samples).

Locally generated outputs, excluded from Git:

- `manifest.json` — frozen selected requests, exact train/heldout split,
  budget, pending key index and runner contract.  It does not contain latency.
- `conditions.json` — complete compact condition/lineage table (gzip+base64
  payload with a plain sorted condition-key index) covering every extracted
  condition, including executable-not-selected, conditional, pending and
  unsupported cases.  `load_conditions`/`materialize` resolve it.
- `inventory.json` — all nine model coverage, every emitted family/recipe,
  dimensions, execution variants and collection status (including non-GEMM
  families and unsupported/pending recipes).
- `unmeasured.json` — durable pending groups, bounded concrete pending index,
  derivation failures and inventory entries not selected by the budget.
- `coverage.md` — human-readable inventory/model/coverage summary.
- `results/dry-run-sample.json` — CPU validation of every selected request.
  Unsupported native shapes retain `pending_backend` with a reason. The
  result is explicitly `not_a3_measured` and contains no measured samples.
- `analysis-dry-run/` — analysis of the dry-run.  It reports no measured
  samples and exposes the analysis artifact shape without fitting timings.

Run these commands in order from the repository root. The first command
creates `manifest.json` and `conditions.json`; worker, analysis and materialize
commands require those generated inputs.

```bash
python -m mindie_sim.a3_operator_campaign plan \
  --config examples/operator-campaign/campaign-config.json \
  --output-dir examples/operator-campaign --budget 320
python -m mindie_sim.a3_operator_worker \
  --manifest examples/operator-campaign/manifest.json \
  --results examples/operator-campaign/results/dry-run-sample.json --dry-run
python -m mindie_sim.a3_operator_curves \
  --manifest examples/operator-campaign/manifest.json \
  --results examples/operator-campaign/results/dry-run-sample.json \
  --output-dir examples/operator-campaign/analysis-dry-run

# Materialize a specific missing condition into exact request(s).
python -m mindie_sim.a3_operator_campaign materialize \
  --conditions examples/operator-campaign/conditions.json \
  --condition-key cond:... --output /tmp/materialized.json
```

The measured production split adapter is:

```bash
python -m mindie_sim.a3_operator_curves --adapter linearity \
  --input train-results.json --output-dir analysis-train
```

It preserves the precommitted split, ignores the legacy measured-median
`analytic_ms` alias, treats 30 event repeats as one measurement, and writes
manifest/results/analysis/calibration/normalized artifacts.

See `docs/operator-shape-campaign.md` for condition identity, sweep/split
policy, worker recipes, protocol and analysis scope.  Actual results should be
kept with the run identifier and hardware evidence; `normalized-conditions.json`
is the adapter for the independent prospective acceptance script.
