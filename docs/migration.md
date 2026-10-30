# 0.8 schema and replacement boundary

Version 0.8 has one maintained runtime and one case schema. The old A3 predictor,
legacy commands, case migration shim, historical source snapshots and temporary
campaign outputs are removed from the repository. Git history retains prior
releases; none is imported or executed by the current package.

## Current entry points

Use `predict`, `target_graphs`, `operator_costs`, `graph_costs` and `study` for
modeling. Cases declare `model`, `machine`, `runtime`, `workload` and optional
`slo`. PD cases declare `pools.prefill`, `pools.decode` and `connector`.
See [inputs](inputs.md) and the executable examples for the current contract.

All shipped `examples/cases/` use this schema. There is no `migrate-case` command
or accepted top-level `migration` metadata. Unknown execution fields are errors;
they cannot silently activate historical floors, measured pool selectors or
model-specific timing corrections. External old inputs must be rewritten using
explicit structural, quantization, topology and acceptance declarations.

## Data and validation

Current observation adapters use source-identified capture protocols in
`data/observations/protocols/`. Actual test inputs live in `tests/fixtures/`.
The selected release evidence is in `data/verification/`; process snapshots and
superseded predictions are excluded. External raw captures remain under their
original capture identities and do not become an executable compatibility layer.

Analytic prediction does not require measurements. Calibrated comparisons,
independent validation and unmodeled features remain separate. A Body pass is not
an assertion of full Serving or unseen-shape accuracy. See [verification](verification.md).
