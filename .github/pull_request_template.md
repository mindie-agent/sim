## 变更摘要 (Summary)

- Describe the root cause / physical mechanism and the resulting behavioral change.

## 设计原则 Review (Pre-PR Checklist — `device-tool`)

- [ ] **DEV-ZERO-CTX** — Simulation, compose, and catalog paths never load weights, dynamically import `torch_npu`, or initialize NPU contexts (`0 MiB` HBM delta).
- [ ] **DEV-READONLY-PATH** — Analytical CLI/API paths remain read-only on inputs, emit zero host `$HOME`/intranet-IP/image-SHA leaks, and propagate `TimeoutError` before `OSError`.
- [ ] **DEV-TOPOLOGY** — Physical Die/interconnect topology (`buffer_units.l2 == 1`, A2/A3 dual-Die `SIO`/`HCCS_SW`, A5 `DAV_3510` `l0c->ub` & `K_8` UB mesh) and per-owner `PhysicalRecipe` stage conservation hold without `ResourceEnvelope` fallbacks.
- [ ] **DEV-BOUNDED-MEM** — Streaming capture reads (`<= 512 MiB` RAM), `< 800`-line modules with 4-section headers, `check_no_magic_numbers.py`, and Unix `LF` line endings all pass.

## 证据记录 (Evidence)

- **DEV-ZERO-CTX**: `<command / test and measured output, or not-applicable: docs-only>`
- **DEV-READONLY-PATH**: `<command / hygiene check output, or not-applicable: docs-only>`
- **DEV-TOPOLOGY**: `<verify_m1.py / recipe conservation / accuracy test output, or not-applicable: docs-only>`
- **DEV-BOUNDED-MEM**: `<check_no_magic_numbers.py / memory & LF check output, or not-applicable: docs-only>`

## 验证结果 (Verification)

- Executed test suites (`pytest`, `verify_m1.py`, `check_no_magic_numbers.py`), wheel smoke test, LF check, and any unexecuted live-hardware scope.
