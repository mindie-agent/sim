# mindie-sim 0.9.15

[中文](sim-0.9.15.md)

This patch release follows `v0.9.14` and fixes storage lifecycle leaks without changing simulation semantics, operator physics, or CLI contracts. It stops four classes of long-running cache/temporary-directory growth and makes the first build after upgrade reclaim historical duplicate artifacts automatically.

- **PATH-independent native accelerator cache identity (`mindie_sim/prediction/runtime/accel_build.py`, S-1)**:
  - Compiler identity now consists of the **resolved real compiler executable path**, its `stat` size/mtime, the `--version` line, an explicit `SDKROOT`, and the platform/pointer-width/endianness/library-extension identity. The caller's raw `os.environ["PATH"]` no longer enters `compiler-*.id` or `cache_digest`.
  - Calls from different virtualenvs, runtime generations, or benchmark slots therefore share one `cache_digest` whenever they resolve the same system compiler, hitting the compiled `<digest>-<generation>.<ext>` plus its `.json` manifest instead of recompiling the full C kernel for every `PATH`.
  - Source snapshots, build temporaries, manifest temporaries, and any hardlinked/copied candidate library from a failed publication are removed in `finally`; partial artifacts never leak.
  - Adds `_prune_engine_cache(cache_dir, keep_key=..., max_artifacts=16)`: oldest-first atomic rename-then-unlink retention keeps the newest 16 `<digest>-*` generation files and their manifests, bounds `compiler-*.id` to 16 entries, and removes `*.tmp` files older than 3600 seconds. On Windows, an older `.dll`/`.pyd` that is still mapped may reject rename/unlink; that eviction is deferred to a later run while the current build succeeds.
  - The cache-hit path also performs one bounded prune, so the first run after upgrade automatically reclaims historical 30+ duplicate `.dylib`/`.so` caches.

- **Deterministic wheel-verification workspace cleanup (`tools/verify_wheel.py`, S-2)**:
  - The `uv venv`, wheel install, CLI checks, and generated fixtures live in a workspace owned by the tool; success, error, and timeout paths all run `shutil.rmtree(..., ignore_errors=True)` in `finally`.
  - Adds `--work-dir` for a caller-owned workspace (never removed automatically) and `--keep-work` to retain an internal workspace for diagnosis. The default no longer leaves a full virtualenv behind.

- **Issue #34 budget-script temporary cleanup (`scripts/check_issue34_test_budget.py`, S-3)**:
  - When `--artifacts` is omitted, the internally created `issue34-budget-*` directory is removed in `finally`; an explicitly supplied directory remains caller-owned.
  - The script keeps its 200-line self-budget, four-section header, and existing timeout/no-retry semantics.

- **Test-suite temporary-directory ownership (S-4)**:
  - Adds `tests/temp_cleanup.py`: every test temporary root is registered through `tempdir()` and removed by both `pytest_sessionfinish` and process `atexit`; `nr-s3-accel-*` / `nr-s3-runtime-*` from `tests/conftest.py` join the same ownership.
  - `tests/nr_s6_round4_probe.py` replaces the `nr4-probe-*` leak with `TemporaryDirectory`, reads the abort record before returning, and deletes the workspace; `run_ranks` callers use `world.abort_record` instead of a removed path.
  - `r6cas-`, `r7cas-`, `r10...`, `nr-s3-r*-wheel-`, and `mp_` roots still use short directories where path length matters, but all are session-registered; orphan `nr4-probe-*` and `nr-s3-accel-*` directories are no longer produced.

- **Regression tests and release metadata**:
  - Adds `tests/test_storage_lifecycle_0_9_15.py`, covering: identical compiler identity/digest under two different `PATH` values plus a real cache hit; `_prune_engine_cache` bounding artifacts/manifests to `<= 16` while deferring a `PermissionError`; failed compilation/manifest publication leaving no candidate or temporary files; and `verify_wheel.py`, `nr_s6_round4_probe.py`, and the budget script leaving zero new entries in an isolated `TMPDIR`.
  - Aligns `pyproject.toml`, `mindie_sim.__version__`, both READMEs, and the release workflow to `0.9.15`.

Upgrade note: no manual cache deletion is required. The first `build-accel` after upgrade (or the first cache hit) computes the new identity and runs bounded pruning; a library still mapped by a Windows process is reclaimed by a later run after that process exits. The complete test record is `pytest -q -m "not live_npu"` plus the attached benchmark artifacts.
