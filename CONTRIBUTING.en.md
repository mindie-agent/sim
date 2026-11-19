# Contributing to `mindie-sim`

简体中文完整版（含五域硅实测校准 SOP、`PhysicalRecipe` 守恒律、自定义 Attention 三级接入与 37+ 模型目录独立对账规程）：[**CONTRIBUTING.md**](CONTRIBUTING.md)

`mindie-sim` is the hardware-neutral, mechanism-first causal performance modeling engine for MindIE Agent (`requires-python >= 3.11`), supporting Ascend A2 (`DAV_2201`), Ascend A3 (`DAV_2201` dual-Die + CloudMatrix 384), and Ascend A5 (`DAV_3510` + Atlas 950 SuperPoD).

## 1. Flat Sibling & Linked-Worktree Workflow

Keep primary checkouts clean on `main` and create an isolated linked Git worktree under `../.worktrees/`:

```bash
git -C ../sim fetch origin
git -C ../sim worktree add -b feat/<topic> ../.worktrees/sim-<topic> origin/main
cd ../.worktrees/sim-<topic>
```

When launching a sandboxed coding CLI (e.g., Codex CLI or Claude Code), grant access to both `../sim` and `../.worktrees/sim-<topic>` so Git worktree metadata writes under `../sim/.git/worktrees/` succeed cleanly.

## 2. Seven Clean-Core Architecture Rules

1. **Single Causal Spine**: All single-op, subgraph, pure-model, 4D efficiency, SLA scaling, and PD/topology sweeps lower onto the single discrete-event `runtime.Engine` (`scripts/verify_m1.py hygiene`).
2. **Zero Silent Fallbacks & No `ResourceEnvelope` Peak-Division**: Default matrix lowering uses `SteadyMatmul`; unmodeled operator mechanisms or core structures fail explicitly (`UnmodeledCapabilityError` with `mindie-unmodeled-structure-hint/1` on public timing entry points); every computation/communication stage declares phased work.
3. **100% Structural Generalization, Full-Stack Upstream Truth & No LUTs**: Never branch on `model_id` / model names in `implementation_selection`, fusion passes, or `PhysicalRecipe`s (`model_scopes` distinguishes `target` vs `draft` subgraphs only), and never fit single-operator parameters against whole-step/body/e2e latency. Mount and verify the latest `HEAD` revision of the owning upstream repository across the Ascend stack (`vllm-ascend`, `cann-recipes-infer`, `MindIE-SD`, `cann/ops-transformer`, `cann/ops-nn`, `fla-npu`, `flash-attn-npu`, `cannbot-dsl`, `xllm` / `xllm-ops`, `torch-npu`) before changing operator graphs or ABIs; when stale issue descriptions conflict with fresh upstream code + real captures, follow real upstream code + captures and document the delta.
4. **Sanitized Provenance Only**: Never commit host `$HOME` paths, private lab IPs, or container image SHA dispatch branches; use `{repository, revision, relative_path, source_sha256}` and `capture://` URIs.
5. **Die-Shared L2 Invariant**: Whenever `device.buffers.l2` is declared, `device.buffer_units.l2` must equal `1`.
6. **Differential Numeric-Literal Gate**: Write natural mathematical and indexing literals (`0`, `1`, `2`, `0.5`, `100`, `1000`, `1024`, `1e-6..1e-12`, subscripts/slices) directly and **never** define tautological digit-word constants (`ZERO = 0`, `ONE = 1`, `SIXTY_FOUR = 64`); extract non-trivial domain invariants to `UPPER_SNAKE_CASE` module symbols carrying a real domain noun or `ascend.json` (`scripts/check_no_magic_numbers.py`).
7. **Agent-Native Module Hygiene, Cross-Repo SSOT & Unix `LF`**: Keep core Python modules `< 800` lines with `[Role]` / `[Invariants]` / `[Timeouts & Locks]` / `[Side-Effect Policy]` in the top 30 lines, `<= 512 MiB` peak RAM, `@pytest.mark.parametrize` SKU sharding for full-catalog sweeps on 2-core / 4-worker CI runners, `mindie-profiler` SSOT compatibility (`PYTHONPATH=.:../profiler`), and Unix `LF` (`\n`) line endings.

## 3. Local Verification & Installed-Wheel Smoke Test

```bash
# Run LF, magic-number, hygiene, and 37-model conservation gates
python3 .github/scripts/check_lf_eol.py
python3 -B scripts/check_no_magic_numbers.py --base origin/main --root .
python3 -B scripts/verify_m1.py hygiene
python3 -B scripts/verify_m1.py catalog-meta --require-models 37

# Prebuild C accelerator and run the maintained test suite
python3 -B -c "import mindie_sim.prediction.runtime.accel_build as b; b.build_accel()"
pytest -q -n auto --dist worksteal -p no:cacheprovider -m "not live_npu"

# Verify built wheel outside the repository checkout
uv build
tmpdir="$(mktemp -d)"
uv venv "$tmpdir/venv"
uv pip install --python "$tmpdir/venv/bin/python" dist/*.whl
(cd "$tmpdir" && "$tmpdir/venv/bin/mindie-sim" --help)
rm -rf "$tmpdir"
```

## 4. Pull Request Checklist (`device-tool` Profile)

Every PR must complete [`.github/pull_request_template.md`](.github/pull_request_template.md) and pass `python3 .github/scripts/check_pr_checklist.py --profile device-tool`:

- `DEV-ZERO-CTX`: Simulation and catalog/CLI paths never load weights, import `torch_npu`, or allocate NPU HBM.
- `DEV-READONLY-PATH`: Analytical paths do not mutate input artifacts or leak host paths/IPs; `TimeoutError` propagates explicitly before `OSError`.
- `DEV-TOPOLOGY`: Die-shared `L2` (`buffer_units.l2 == 1`), AIC/AIV owner placement, and A2/A3 dual-Die/A5 interconnect topologies remain exact.
- `DEV-BOUNDED-MEM`: Streaming I/O (`<= 512 MiB` RAM), `< 800`-line modules with 4-part headers, zero bare magic numbers, and Unix `LF` line endings.
