# MindIE Sim (`sim`)

This repository owns the `mindie_sim` runtime and `mindie-sim` CLI. Product-level entry documentation and cross-repo routing live in the sibling [`mindie-agent`](../mindie-agent) repository.

Read [`docs/architecture.md`](docs/architecture.md) before changing the engine or semantic boundaries.
Read [`docs/observations.md`](docs/observations.md) before importing observations or changing calibration.
Read [`docs/modeling-fidelity.md`](docs/modeling-fidelity.md) when choosing operator detail or refinement work.
Read [`CONTRIBUTING.md`](CONTRIBUTING.md) ([English](CONTRIBUTING.en.md)) for the Ascend A2/A3/A5 calibration SOP, `PhysicalRecipe` conservation contract, and linked-worktree PR workflow.

---

## 1. Causal Spine & Domain Invariants (`device-tool`)

- Keep one causal execution engine for all hardware and deployments. Vendor names select input adapters, never performance branches in the execution loop.
- Derive work and dependencies from structure. No serving floors, model-specific latency multipliers, body-total fits, `ResourceEnvelope` peak-division fallbacks, or measurement-presence admission checks.
- Keep calibration separate from validation. Heldout measurements never enter a training lookup. Preserve original boundary, units, source, and implementation.
- Extend block/recipe registries (`Class A–E` or `PlannedRecipe`). A new model adapter produces existing structural records; a new mechanism supplies its own builder or executable recipe conserving prepared work across `mte2_aic`, `mte2_aiv`, `cube`, `vector`, `fixpipe`, `mte3`, `l2`, and `hbm`.
- Prefer the simplest resource/phase/tile model that preserves relevant work, boundaries, and constraints. Instruction-level detail is optional and needs a material cost, trend, or ranking counterexample; nonzero timing error alone is not a reason to recreate a kernel/compiler. Calibration is a separate axis.
- **Zero Accidental Device Context (`DEV-ZERO-CTX`)**: Simulation must not load weights, import `torch_npu`, initialize NPU contexts, or start remote work. Device measurements require resource/VRAM prechecks and explicit bounded work.
- **Read-Only Analytical Execution & Clean Provenance (`DEV-READONLY-PATH`)**: Routine prediction/sweep/SLA/efficiency commands are read-only on inputs and never leak host `$HOME` paths, private intranet IPs, or container image SHA routing (`capture://` URIs and `{repository, revision, relative_path, source_sha256}` only).
- **Exact Hardware & Multi-Die Topology (`DEV-TOPOLOGY`)**: Enforce `buffer_units.l2 == 1` per physical Die, explicit A3 dual-Die (`SIO` vs `HCCS_SW` vs SuperPoD) and A5 `DAV_3510` (`l0c->ub`, `K_8` UB mesh) capabilities, and per-owner AIC/AIV placement without borrowing idle sibling-core private pipes or SRAM.
- **Bounded Memory, Clean Modules & Differential Literals (`DEV-BOUNDED-MEM`)**:
  - Keep core modules cohesive and `< 800` lines with `[Role]`, `[Invariants]`, `[Timeouts & Locks]`, and `[Side-Effect Policy]` in their first 30 lines.
  - Keep live execution memory bounded by active work (`<= 512 MiB` process RAM); stream large captures and never whole-read files `> 50 MiB`.
  - No bare numeric literals in modified `mindie_sim/**/*.py` statements (`scripts/check_no_magic_numbers.py`); extract named module constants or `ascend.json` profile fields.
- Tests must establish behavior or independent mathematical invariants, not merely type checks, static strings, documentation prose, or a fabricated external schema.
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
# 1. Hygiene, 37-model independent catalog conservation, LF & differential literal gates
python3 .github/scripts/check_lf_eol.py
python3 -B scripts/check_no_magic_numbers.py --base origin/main --root .
python3 -B scripts/verify_m1.py hygiene
python3 -B scripts/verify_m1.py catalog-meta --require-models 37

# 2. Prebuild C accelerator & run maintained non-live-NPU test suite
python3 -B -c "import mindie_sim.prediction.runtime.accel_build as b; b.build_accel()"
pytest -q -n auto --dist worksteal -p no:cacheprovider -m "not live_npu"

# 3. Validate PR body against the device-tool design-principles checklist
python3 .github/scripts/check_pr_checklist.py --profile device-tool --body-file <pr_body.md>
```
