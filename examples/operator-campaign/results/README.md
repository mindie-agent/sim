# Results

`dry-run-sample.json` is a CPU-only dry-run produced by
`python -m mindie_sim.a3_operator_worker --dry-run`.  It contains
`not_a3_measured=true`, `source_class=derived`, `raw_samples_ms=[]` and no
measured A3 sample.  It is retained to show the result schema and to validate
all 319 selected request recipes locally.

Real measured results must come from `sim_audit` under the existing
`defaultHostQueue` with a coordinator-injected device.  Keep the run id,
manifest, environment/versions, all raw trials and observed context together.
Do not copy heldout raw values into this directory or use them to choose fits.
