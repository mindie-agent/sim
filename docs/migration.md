# Migration & Breaking Replacement Boundaries

## Single-Spine Runtime-Agnostic Migration (Zero-Shim Retirements)

`mindie-sim` now enforces a **single runtime-agnostic spine (`Core <-> RuntimeStackAdapter`)**
with zero compatibility shims or deprecation re-export stubs. Callers migrating
from pre-single-spine revisions must adopt the canonical paths and contracts below:

### 1. Retired Legacy Modules (Deleted With Zero Shims)

| Retired Module / Asset | Canonical Replacement |
| --- | --- |
| `mindie_sim.stack_registry` | [`mindie_sim/stacks/registry.py`](../mindie_sim/stacks/registry.py) (`resolve_stack_for_scenario`, `get_stack`, `list_stacks`, `register_stack`, `load_manifest_stack`) |
| `mindie_sim.stack_tracks` | [`mindie_sim/feature_mechanisms.py`](../mindie_sim/feature_mechanisms.py) & [`mindie_sim/prediction/results.py`](../mindie_sim/prediction/results.py) |
| `mindie_sim.runtime_profiles` | [`mindie_sim/stacks/vllm_ascend/profiles.py`](../mindie_sim/stacks/vllm_ascend/profiles.py) |
| `mindie_sim.catalog.va_speculative_rules` | [`mindie_sim/stacks/vllm_ascend/speculative.py`](../mindie_sim/stacks/vllm_ascend/speculative.py) |
| `mindie_sim.optimization.target_schedule` | [`mindie_sim/prediction/runtime/exposed_observer.py`](../mindie_sim/prediction/runtime/exposed_observer.py) (`ExposedObserver`, `attribute_stream0_intervals`) |
| `mindie_sim.export.script_audit` | [`mindie_sim/stacks/vllm_ascend/launch_codec.py`](../mindie_sim/stacks/vllm_ascend/launch_codec.py) (`VllmAscendLaunchCodec` accessed via `adapter.launch_codec`) |
| Legacy `stack_sources` JSON manifest | [`mindie_sim/stacks/vllm_ascend/manifest.json`](../mindie_sim/stacks/vllm_ascend/manifest.json) (`mindie-sim.stack-manifest/1`) |

### 2. Explicit 4-Mode `--deployment-mode` & Role-SLA Contract

- **No silent guessing**: `--deployment-mode` (`deployment_mode` in Python/MCP)
  accepts only four canonical values: `colocated`, `pd_disaggregated`,
  `pd_prefill`, and `pd_decode`. Legacy `"auto"` and `"pd"` values raise
  `ModelingError` immediately.
- **Role-appropriate SLAs (zero `5000.0 ms` TTFT coercion)**:
  - `colocated` / `pd_disaggregated`: `tpot_slo_ms` is required; omitted/`null`
    `ttft_slo_ms` remains explicitly unconstrained.
  - `pd_prefill`: `ttft_slo_ms` is required; passing `tpot_slo_ms` or `itl_ms`
    raises `pd_prefill_rejects_decode_sla`.
  - `pd_decode`: `tpot_slo_ms` and an explicit `ReadyStateAssumption` are
    required; passing `ttft_slo_ms` raises `pd_decode_ttft_unavailable`.

### 3. Domain-Aware Stack Routing (`--stack`)

- Omitting `--stack` routes by `WorkloadDomain`: LLM/VLM workloads default to
  `vllm-ascend`, and Diffusion/DiT workloads default to `diffusion-native`
  (never fabricating LLM KV-Cache on Diffusion models).
- Built-in stacks are `vllm-ascend`, `diffusion-native`, `ideal-target`, and
  `raw-eager` (plus custom `mindie-sim.stack-manifest/1` JSON manifests and
  `mindie_sim.stacks` Python entry points). Forcing a domain-mismatched stack
  raises `ModelingError("stack_domain_mismatch:...")`.

### 4. Export & Workbook Contract Changes

- **`--kernel-designs-json` retired**: Free-text prose overrides raise
  `ModelingError`; pass verified `mindie-rewrite-plan/1` JSON via
  `--rewrite-plan` instead.
- **Pure 4-sheet workbook (`0` hidden sheets)**: Sheet 2 (`Layer_Operator_Breakdown`)
  is locked to `A:AA` (27 visible columns; legacy `AB..AF` accounting columns
  removed), Sheet 3 (`Memory_Quant_Calculator`) defaults to `"As evaluated"`
  bound directly to `ResidentMemoryReceipt`, Sheet 4 (`Pending_Kernel_Details`)
  uses live visible Sheet 2 span formulas (with automatic visible `A:N` Group
  Attribution Table fallback when exceeding Excel limits), and all user-facing
  views filter strictly through [`mindie_sim/export/public_view.py`](../mindie_sim/export/public_view.py)
  recommending **`TileLang / AscendC / Catlass`**.

---

## 0.8 schema and replacement boundary

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
