## 变更摘要 (Summary)

- Describe the root cause / physical mechanism and the resulting behavioral change.

## 设计原则 Review (Pre-PR Checklist — `device-tool`)

- [ ] **DEV-ZERO-CTX** — Simulation, compose, and catalog paths never load weights, dynamically import `torch_npu`, or initialize NPU contexts (`0 MiB` HBM delta).
- [ ] **DEV-READONLY-PATH** — Analytical CLI/API paths remain read-only on inputs, emit zero host `$HOME`/intranet-IP/image-SHA leaks, and propagate `TimeoutError` before `OSError`.
- [ ] **DEV-TOPOLOGY** — Physical Die/interconnect topology (`buffer_units.l2 == 1`, A2/A3 dual-Die `SIO`/`HCCS_SW`, A5 `DAV_3510` `DT` + `PR_9579/958b` `l0c->ub` & `K_8` UB mesh), per-owner `PhysicalRecipe` conservation (`0` `ResourceEnvelope` / `UnmodeledCapabilityError` on unmodeled structures), **100% structural generalization (`0` `model_id` kernel branches)**, fresh owning upstream repo `HEAD` verification (`vllm-ascend` / `cann-recipes-infer` / `MindIE-SD` / `ops-transformer` / `ops-nn` / `fla-npu` / `flash-attn-npu` / `TileLang` / `AscendC` / `Catlass`, etc.), and `mindie-profiler` SSOT contract compatibility all hold.
- [ ] **DEV-BOUNDED-MEM** — Streaming capture/checkpoint reads (`> 50 MiB` streamed, `<= 512 MiB` RAM), Cohesion-First modules with 4-section top-30-line headers (`AG001`–`AG015`, `~150` non-table-line soft baseline with `[Cohesive-Pipeline]` docstring contract for genuinely cohesive single-algorithm pipelines, hard nesting `<= 6`, zero artificial `*_helpers`/`*_mixin`/`*_part*` shards or multi-arg helper chops), `check_no_magic_numbers.py` (no tautological digit-word aliases), `@pytest.mark.parametrize` SKU sharding + shared 4-worker xdist wall-clock headroom, and Unix `LF` line endings all pass.

## 证据记录 (Evidence)

- **DEV-ZERO-CTX**: `<command / test and measured output, or not-applicable: docs-only>`
- **DEV-READONLY-PATH**: `<command / hygiene check output, or not-applicable: docs-only>`
- **DEV-TOPOLOGY**: `<upstream repo@revision checked, 0-model_id generalization, verify_m1/m2 & cross-repo profiler contract output, or not-applicable: docs-only>`
- **DEV-BOUNDED-MEM**: `<check_no_magic_numbers.py, parametrized test runtime & LF check output, or not-applicable: docs-only>`

## 验证结果 (Verification)

- Executed test suites (`pytest`, cross-repo `PYTHONPATH=.:../profiler` when touching SSOT/contracts, `verify_m1.py`, `verify_m2.py`, `check_no_magic_numbers.py`), wheel smoke test, LF check, and any unexecuted live-hardware scope.
