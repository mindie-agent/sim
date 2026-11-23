# MindIE Sim (`sim`)

This repository owns the `mindie_sim` runtime and `mindie-sim` CLI. Product-level entry documentation and cross-repo routing live in the sibling [`mindie-agent`](../mindie-agent) repository.

## First-Screen Canonical Owner Map (`Rule #1`: Edit the Existing Owner)

**Rule #1**: *Edit the existing canonical owner in place — **never** create a parallel concept struct, `*_v2.py` duplicate, or `*_helper`/`*_mixin`/`*_part*` shard (`AG006`–`AG012` & Hub Lock-6 fail CI mechanically).*

| Domain / Concept | Canonical Types & Entrypoints | Single Authoritative Owner |
| --- | --- | --- |
| **L0 Contracts, Deployment & Co-Design** | `ScenarioContract`, `SLAContract`, `DeploymentMode`, `CheckpointQuantMap`, `RuntimeStackAdapter`, `CacheStateABI`, `TensorPrecisionABI`, `OwnerScheduleSpec`, `CoDesignOpportunity`, `RequestView`, `IterationPlan`, `ModelSpec`, `Case` | [`mindie_sim/schema/scenario.py`](mindie_sim/schema/scenario.py), [`mindie_sim/schema/checkpoint.py`](mindie_sim/schema/checkpoint.py), [`mindie_sim/schema/stack.py`](mindie_sim/schema/stack.py), [`mindie_sim/schema/operators.py`](mindie_sim/schema/operators.py) & [`mindie_sim/schema/`](mindie_sim/schema/) |
| **Pluggable Runtime Stacks (`Core <-> Adapter`)** | `resolve_stack_for_scenario`, `get_stack`, `load_manifest_stack`, `VllmAscendStackAdapter`, `DiffusionNativeStackAdapter`, `IdealTargetStackAdapter`, `RawEagerStackAdapter` | [`mindie_sim/stacks/registry.py`](mindie_sim/stacks/registry.py) & [`mindie_sim/stacks/`](mindie_sim/stacks/) *(domain-aware routing, `mindie_sim.stacks` entry points)* |
| **Causal Engine, Cohort Replay & Observer** | `Engine`, `Event`, `ExecutionBindings`, `LayerRunProgram`, `ExposedObserver`, `attribute_stream0_intervals` | [`mindie_sim/prediction/runtime/engine.py`](mindie_sim/prediction/runtime/engine.py), [`mindie_sim/prediction/execution/layer_runs.py`](mindie_sim/prediction/execution/layer_runs.py) & [`mindie_sim/prediction/runtime/exposed_observer.py`](mindie_sim/prediction/runtime/exposed_observer.py) |
| **L2 Residency & Unified Per-Die Memory Ledger** | `AdmissionMemoryView`, `ByteLedger`, `AddressRunPlan`, `ResidentMemoryReceipt` | [`mindie_sim/prediction/runtime/l2_residency.py`](mindie_sim/prediction/runtime/l2_residency.py), [`mindie_sim/prediction/runtime/memory_physics.py`](mindie_sim/prediction/runtime/memory_physics.py) & [`mindie_sim/prediction/capacity/ledger.py`](mindie_sim/prediction/capacity/ledger.py) |
| **Hardware Profiles & L0 Primitives** | `ascend.json`, `PrimitiveTable`, `CalibratedRateOverlay` | [`mindie_sim/backends/`](mindie_sim/backends/) & [`mindie_sim/primitives/`](mindie_sim/primitives/) |
| **Single-Rank NPU Operator Kernels** | `OperatorPlanningContext`, `FamilyRegistry`, `<family>/*` | [`mindie_sim/prediction/operators/`](mindie_sim/prediction/operators/) *(single-rank kernels ONLY)* |
| **Distributed Sharding & Collectives** | `SP`, `Local-TP`, `DCP`/`PCP` monoids, `EP`/`MC2` boundaries | [`mindie_sim/prediction/sharding/`](mindie_sim/prediction/sharding/) & [`mindie_sim/prediction/collectives/`](mindie_sim/prediction/collectives/) |
| **Fusion, Rewrite & Primal-Dual `L*`** | `RewritePipeline`, `OptimalityCertificate`, `certify_lower_bound` | [`mindie_sim/prediction/fusion/`](mindie_sim/prediction/fusion/), [`mindie_sim/rewrite/`](mindie_sim/rewrite/), [`mindie_sim/optimization/dual_bound.py`](mindie_sim/optimization/dual_bound.py) |
| **Catalog, Streaming Checkpoints & Compose** | `load_model`, `read_checkpoint_quant_map`, `lookup_curated_checkpoint`, `curated_checkpoints.json` (71 checkpoints), `compose_case` | [`mindie_sim/model_catalog.py`](mindie_sim/model_catalog.py), [`mindie_sim/catalog/checkpoint_reader.py`](mindie_sim/catalog/checkpoint_reader.py), [`mindie_sim/compose/`](mindie_sim/compose/) |
| **Full-Grid Solver & Pareto Sweep** | `solve_deployment`, `solve_optimal_deployment`, `sweep_topologies`, `DeploymentOutcome` | [`mindie_sim/solve.py`](mindie_sim/solve.py) & [`mindie_sim/sweep/`](mindie_sim/sweep/) |
| **Feature Waterfall & Mechanism Registry** | `compare_feature_tracks`, `register_optimization_mechanism`, `register_runtime_stack` | [`mindie_sim/feature_mechanisms.py`](mindie_sim/feature_mechanisms.py) |
| **Turnkey CLI, 4-Sheet Export & Public View** | `report`, `solve`, `export`, `public_view.py`, `xlsx_*.py`, `onboard-sku/operator` | [`mindie_sim/cli/`](mindie_sim/cli/), [`mindie_sim/export/`](mindie_sim/export/), [`mindie_sim/onboarding/`](mindie_sim/onboarding/) |
| **Observations & Profiler SSOT** | `observed_contracts`, `captures`, `calibration`, `service_fitting` | [`mindie_sim/observed_contracts/`](mindie_sim/observed_contracts/) & [`mindie_sim/observations/`](mindie_sim/observations/) |

Read [`docs/architecture.md`](docs/architecture.md), [`docs/observations.md`](docs/observations.md), [`docs/modeling-fidelity.md`](docs/modeling-fidelity.md), [`docs/migration.md`](docs/migration.md), and [`CONTRIBUTING.md`](CONTRIBUTING.md) ([English](CONTRIBUTING.en.md)) when changing engine semantics, stack adapters, calibration, or hardware profiles.

---

## 1. Causal Spine & Domain Invariants (`device-tool`)

- **Runtime-Agnostic Single Spine (`Core <-> RuntimeStackAdapter`)**: Keep L0–L3 Core (`schema/`, `catalog/`, `backends/`, `primitives/`, `prediction/`) 100% runtime-agnostic and acyclic (`AG010` layer DAG + package SCC checks). All inference-runtime specifics (`vllm-ascend`, `diffusion-native`, `ideal-target`, `raw-eager`, custom JSON manifests, and `mindie_sim.stacks` entry points) live strictly under [`mindie_sim/stacks/`](mindie_sim/stacks/), which never imports `compose/` or `export/`. Resolve default stacks by `WorkloadDomain` (`llm`/`vlm` -> `vllm-ascend`, `diffusion` -> `diffusion-native`) and reject incompatible forced stacks with `ModelingError("stack_domain_mismatch:...")`.
- **One Engine, One `PrimitiveTable`, Exact Cohort Replay & 4-Stage / 3-Owner Waterfall**: Evaluate `Before -> Graph_cfg ([1] ops_config) -> Graph_fwk ([2] framework_python) -> Graph_After ([3] kernel_dev_tune)` on the exact same `Engine` and immutable `PrimitiveTable` with native-layer cohort execution (`layer_runs.py`, `0 ns` drift on 93L/61L/48L models) and $O((N+G)\log(N+G))$ sweep-line attribution (`exposed_observer.py`). Keep `CoDesignOpportunity` (`MegaFusionBound`, `SpaceTimeTradeoffSpec`, `CrossHwOffloadSpec`) recommendations aligned with open Ascend/NPU kernel ecosystems (**`TileLang / AscendC / Catlass`**).
- **Explicit 4-Mode Deployment Contract & Role-SLA Validation**: Enforce canonical `--deployment-mode` in `{colocated, pd_disaggregated, pd_prefill, pd_decode}` across CLI, Python SDK, and MCP `sim_oracle` — zero silent `"auto"`/`"pd"` guessing and zero `5000.0 ms` TTFT coercion (`pd_decode` requires `ReadyStateAssumption` and rejects `ttft_slo_ms`; `pd_prefill` requires `ttft_slo_ms` and rejects `tpot_slo_ms`/`itl_ms`).
- **Checkpoint-Faithful Quantization & Unified Per-Die Memory Receipt**: Stream `> 50 MiB` ModelSlim/HF quantization metadata read-only via [`mindie_sim/catalog/checkpoint_reader.py`](mindie_sim/catalog/checkpoint_reader.py) and the packaged 71-checkpoint dataset ([`mindie_sim/catalog/data/curated_checkpoints.json`](mindie_sim/catalog/data/curated_checkpoints.json)); never silently degrade unknown weight tags to `BF16`. Bill per-native-layer `CacheStateABI` (MLA, DSA Indexer, KDA/GDN, PLE, `C8`/`FAQuant`/`QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL`, 7-family `RecoverSSM` snapshot rings) and simultaneous per-die HBM peaks once into a single `ResidentMemoryReceipt` shared identically across sizer, full-grid solver, and Sheet 3 (`"As evaluated"`).
- **Pure 4-Sheet Script-Free Export & Strict Allowlist**: Keep `.xlsx` exports to 4 visible sheets (`0` hidden sheets; optional 5th calibration sheet only with `--observations`/`--kernel-csv`), Sheet 2 strictly `A:AA` (27 cols), Sheet 3 defaulting to `"As evaluated"`, Sheet 4 using live visible Sheet 2 span formulas (with automatic visible `A:N` Group Attribution Table fallback if Excel limits are exceeded), and route all user/agent-facing projections through [`mindie_sim/export/public_view.py`](mindie_sim/export/public_view.py).
- Derive work and dependencies from structure with **100% structural/tensor generalization** (`0` `model_id` branches in `implementation_selection`, fusion passes, or `PhysicalRecipe`s; `model_scopes` distinguishes `target` vs `draft` subgraphs only). No serving floors, model-specific latency multipliers, body-total fits, `ResourceEnvelope` peak-division fallbacks, or measurement-presence admission checks; unmodeled core structures must raise `UnmodeledCapabilityError` (`mindie-unmodeled-structure-hint/1`).
- Verify **fresh upstream source `HEAD`s across the full Ascend stack** (`vllm-ascend`, `cann-recipes-infer`, `MindIE-SD`, `cann/ops-transformer`, `cann/ops-nn`, `fla-npu`, `flash-attn-npu`, `xllm` / `xllm-ops`, `torch-npu`, and open `TileLang / AscendC / Catlass` kernels) — mounting the actual owning repo whenever an operator lives outside `ops-transformer` — before altering operator graphs, fusion boundaries, or kernel ABIs; when stale issue assumptions conflict with current upstream source + real captures, follow real upstream source + captures and record the upstream evolution in the PR.
- Keep calibration separate from validation. Heldout measurements never enter a training lookup. Preserve original boundary, units, source, and implementation.
- Extend governed single-rank kernels under `mindie_sim/prediction/operators/<family>/` (or `mindie_sim/templates/`) and distributed tensor-placement/collective rewrites under `mindie_sim/prediction/sharding/`. Never add device operator builders or `OperatorPlan`/`PlannedRecipe` implementations in flat `mindie_sim/prediction/*.py` (`AG006` + `AG008` strictly ratchet `MAX_FLAT_PREDICTION_MODULES` downward).
- Prefer the simplest resource/phase/tile model that preserves relevant work, boundaries, and constraints. Instruction-level detail is optional and needs a material cost, trend, or ranking counterexample; nonzero timing error alone is not a reason to recreate a kernel/compiler. Calibration is a separate axis.
- **Zero Accidental Device Context (`DEV-ZERO-CTX`)**: Simulation must not load weights, import `torch_npu`, initialize NPU contexts, or start remote work. Device measurements require resource/VRAM prechecks and explicit bounded work.
- **Read-Only Analytical Execution & Clean Provenance (`DEV-READONLY-PATH`)**: Routine prediction/sweep/SLA/efficiency commands are read-only on inputs and never leak host `$HOME` paths, private intranet IPs, or container image SHA routing (`capture://` URIs and `{repository, revision, relative_path, source_sha256}` only).
- **Exact Hardware, Multi-Die Topology & Cross-Repo SSOT (`DEV-TOPOLOGY`)**: Enforce `buffer_units.l2 == 1` per physical Die, explicit A3 dual-Die (`SIO` vs `HCCS_SW` vs SuperPoD) and A5 `DAV_3510` (`DT` `4.0 TB/s` + `PR_9579` `1.6 TB/s` + `PR_958b` `1.4 TB/s`, `l0c->ub`, `K_8` UB mesh) capabilities, per-owner AIC/AIV placement, and `mindie-profiler` SSOT contract stability (`observed_contracts`, `subgraph_math`, `structure*`).
- **Cohesion-First Structural Governance, Bounded Memory & Differential Literals (`DEV-BOUNDED-MEM`)**:
  - **Cohesion Over Arbitrary Line Caps**: Keep a single cohesive physical domain together in one self-contained module (`~1,800` lines is a soft review guideline, NOT a hard ceiling—exceeding `1,800` lines is permitted when genuinely single-domain and declared in `[Role]`) with `[Role]`, `[Invariants]`, `[Timeouts & Locks]`, and `[Side-Effect Policy]` in the top 30 lines. **Never** slice cohesive classes or kernels into artificial `*_helpers.py`, `*_mixin.py`, or `*_part*.py` shards just to cut line counts; keep individual functions readable (`<= 150` lines, `<= 6` control-flow nesting depth, excluding declarative tables/dispatch maps) and package directories bounded (`<= 20` peer `.py` target).
  - Keep live execution memory bounded by active work (`<= 512 MiB` process RAM); stream large captures and never whole-read files `> 50 MiB`.
  - Write natural math/index literals (`0`, `1`, `2`, `0.5`, `100`, `1000`, `1024`, `1e-6..1e-12`) directly, never define tautological digit-word constants (`ZERO`, `ONE`, `SIXTY_FOUR`), and extract non-trivial domain constants to `UPPER_SNAKE_CASE` module symbols or `ascend.json` (`scripts/check_no_magic_numbers.py`).
  - Shard full-catalog / multi-SKU sweeps via `@pytest.mark.parametrize("hardware", SKUS)` and size wall-clock assertions for shared 2-core / 4-worker `pytest-xdist` CI runners; when touching `observed_contracts`, `subgraph_math`, `structure*`, or `source_boundary`, also run tests with `PYTHONPATH=".:../profiler"`.
- Tests must establish behavior or independent mathematical invariants, not merely type checks, static strings, documentation prose, or a fabricated external schema.
- Place tests in `tests/<domain>/test_<mechanism>.py` (`<= 600` lines) listed in `tests/TAXONOMY.toml`, with a module-level `tier0`/`tier1` mark; never encode issue/round/wave/SKU/ISA/model provenance in test file or function names (sweep SKUs/ISAs via `tests/_support/axes.py`), never import another test module, and never add root-level `tests/test_*.py` files (`scripts/check_test_orthogonality.py`).
- Maintain one current schema and runtime. Do not restore deleted A3 prediction paths, migration shims, or old-baseline fallbacks; see [`docs/migration.md`](docs/migration.md) and [`docs/archive/README.md`](docs/archive/README.md).
- Use Unix `LF` (`\n`) line endings everywhere.

---

## 2. Flat Sibling & Linked-Worktree Workflow

Develop inside an isolated linked worktree under `../.worktrees/` while keeping base checkouts on clean `main`:

```bash
git -C ../sim fetch origin
git -C ../sim worktree add -b feat/<topic> ../.worktrees/sim-<topic> origin/main
cd ../.worktrees/sim-<topic>
```

---

## 3. Verification Commands

```bash
# 1. Hygiene, 39-model independent catalog conservation, LF & differential literal gates
python3 .github/scripts/check_lf_eol.py
python3 -B scripts/check_no_magic_numbers.py --base origin/main --root .
python3 -B scripts/check_arch_governance.py --root . --strict
python3 -B scripts/check_test_orthogonality.py --root . --base origin/main   # tests/<domain>/ per tests/TAXONOMY.toml
python3 -B scripts/verify_m1.py hygiene
python3 -B scripts/verify_m1.py catalog-meta --require-models 39

# 2. Prebuild C accelerator & run maintained non-live-NPU test suite
python3 -B -c "import mindie_sim.prediction.runtime.accel_build as b; b.build_accel()"
pytest -q -n auto --dist worksteal -p no:cacheprovider -m "not live_npu"

# 3. Validate PR body against the device-tool design-principles checklist
python3 .github/scripts/check_pr_checklist.py --profile device-tool --body-file <pr_body.md>
```
