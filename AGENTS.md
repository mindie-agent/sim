# MindIE Sim (`sim`)

This repository owns the `mindie_sim` runtime and `mindie-sim` CLI. Product-level entry documentation and cross-repo routing live in the sibling [`mindie-agent`](../mindie-agent) repository.

## First-Screen Canonical Owner Map (`Rule #1`: Edit the Existing Owner)

**Rule #1**: *Edit the existing canonical owner in place — **never** create a parallel concept struct, `*_v2.py` duplicate, or `*_helper`/`*_mixin`/`*_part*` shard (`AG006`–`AG012` & Hub Lock-6 fail CI mechanically).*

| Domain / Concept | Canonical Types & Entrypoints | Single Authoritative Owner |
| --- | --- | --- |
| **Causal Event Engine & Resources** | `Engine`, `Event`, `ExecutionBindings`, `CostModel` | [`mindie_sim/prediction/runtime/engine.py`](mindie_sim/prediction/runtime/engine.py) & [`mindie_sim/prediction/execution/`](mindie_sim/prediction/execution/) |
| **Workload & Iteration Batch** | `RequestView`, `IterationPlan`, `ModelSpec`, `Case` | [`mindie_sim/schema/workload.py`](mindie_sim/schema/workload.py) & [`mindie_sim/schema/`](mindie_sim/schema/) |
| **L2 Residency & 3-Ledger Memory** | `AdmissionMemoryView`, `ByteLedger`, `AddressRunPlan` | [`mindie_sim/prediction/runtime/l2_residency.py`](mindie_sim/prediction/runtime/l2_residency.py) & [`mindie_sim/prediction/runtime/memory_physics.py`](mindie_sim/prediction/runtime/memory_physics.py) |
| **Hardware Profiles & L0 Primitives** | `ascend.json`, `PrimitiveTable`, `CalibratedRateOverlay` | [`mindie_sim/backends/`](mindie_sim/backends/) & [`mindie_sim/primitives/`](mindie_sim/primitives/) |
| **Single-Rank NPU Operator Kernels** | `OperatorPlanningContext`, `FamilyRegistry`, `<family>/*` | [`mindie_sim/prediction/operators/`](mindie_sim/prediction/operators/) *(single-rank kernels ONLY)* |
| **Distributed Sharding & Collectives** | `SP`, `Local-TP`, `DCP`/`PCP` monoids, `EP`/`MC2` boundaries | [`mindie_sim/prediction/sharding/`](mindie_sim/prediction/sharding/) & [`mindie_sim/prediction/collectives/`](mindie_sim/prediction/collectives/) |
| **Fusion, Rewrite & Primal-Dual `L*`** | `RewritePipeline`, `OptimalityCertificate`, `certify_lower_bound` | [`mindie_sim/prediction/fusion/`](mindie_sim/prediction/fusion/), [`mindie_sim/rewrite/`](mindie_sim/rewrite/), [`mindie_sim/optimization/dual_bound.py`](mindie_sim/optimization/dual_bound.py) |
| **Model Catalog, Quant & Compose** | `load_model`, `QUANTIZATION_PRESETS`, `compose_case` | [`mindie_sim/model_catalog.py`](mindie_sim/model_catalog.py), [`mindie_sim/catalog/`](mindie_sim/catalog/), [`mindie_sim/compose/`](mindie_sim/compose/) |
| **Deployment Solver & Pareto Sweep** | `solve_deployment`, `solve_optimal_deployment`, `sweep_topologies` | [`mindie_sim/solve.py`](mindie_sim/solve.py) & [`mindie_sim/sweep/`](mindie_sim/sweep/) |
| **Stack Tracks & Feature Waterfall** | `compare_feature_tracks`, `resolve_runtime_stack` | [`mindie_sim/feature_mechanisms.py`](mindie_sim/feature_mechanisms.py) & [`mindie_sim/stack_tracks.py`](mindie_sim/stack_tracks.py) |
| **Turnkey CLI, Exporters & Onboarding** | `report`, `quickstart`, `export`, `onboard-sku/operator` | [`mindie_sim/cli/`](mindie_sim/cli/), [`mindie_sim/export/`](mindie_sim/export/), [`mindie_sim/onboarding/`](mindie_sim/onboarding/) |
| **Observations & Profiler SSOT** | `observed_contracts`, `captures`, `calibration`, `service_fitting` | [`mindie_sim/observed_contracts/`](mindie_sim/observed_contracts/) & [`mindie_sim/observations/`](mindie_sim/observations/) |

Read [`docs/architecture.md`](docs/architecture.md), [`docs/observations.md`](docs/observations.md), [`docs/modeling-fidelity.md`](docs/modeling-fidelity.md), and [`CONTRIBUTING.md`](CONTRIBUTING.md) ([English](CONTRIBUTING.en.md)) when changing engine semantics, calibration, or hardware profiles.

---

## 1. Causal Spine & Domain Invariants (`device-tool`)

- Keep one causal execution engine for all hardware and deployments. Vendor names select input adapters, never performance branches in the execution loop.
- Derive work and dependencies from structure with **100% structural/tensor generalization** (`0` `model_id` branches in `implementation_selection`, fusion passes, or `PhysicalRecipe`s; `model_scopes` distinguishes `target` vs `draft` subgraphs only). No serving floors, model-specific latency multipliers, body-total fits, `ResourceEnvelope` peak-division fallbacks, or measurement-presence admission checks; unmodeled core structures must raise `UnmodeledCapabilityError` (`mindie-unmodeled-structure-hint/1`).
- Verify **fresh upstream source `HEAD`s across the full Ascend stack** (`vllm-ascend`, `cann-recipes-infer`, `MindIE-SD`, `cann/ops-transformer`, `cann/ops-nn`, `fla-npu`, `flash-attn-npu`, `cannbot-dsl`, `xllm` / `xllm-ops`, `torch-npu`) — mounting the actual owning repo whenever an operator lives outside `ops-transformer` — before altering operator graphs, fusion boundaries, or kernel ABIs; when stale issue assumptions conflict with current upstream source + real captures, follow real upstream source + captures and record the upstream evolution in the PR.
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
