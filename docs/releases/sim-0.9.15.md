# mindie-sim 0.9.15

[English](sim-0.9.15.en.md)

本版本是 `v0.9.14` 之后的存储生命周期修复版本，不改变模拟语义、算子物理模型或 CLI 契约。它修复四类长期运行后持续膨胀的缓存/临时目录泄漏，并让升级后的下一次构建自动回收历史重复产物。

- **原生加速核缓存身份去 `PATH` 化（`mindie_sim/prediction/runtime/accel_build.py`，S-1）**：
  - 编译器身份改为由**解析后的真实编译器可执行文件路径**、其 `stat` 大小/mtime、`--version` 版本行、显式 `SDKROOT`、平台/指针宽度/端序/库扩展名共同组成；调用方原始 `os.environ["PATH"]` 不再进入 `compiler-*.id` 或 `cache_digest`。
  - 因此从不同 virtualenv、运行时 generation 或基准 slot 调用时，只要解析到同一系统编译器，就得到完全相同的 `cache_digest`，直接命中已编译的 `<digest>-<generation>.<ext>` 与 `.json` manifest，避免为每个 `PATH` 重复完整 C 编译。
  - 编译快照、构建临时文件、manifest 临时文件以及发布失败时已 hardlink/copy 的候选库均在 `finally` 中清理；失败发布不会留下半成品。
  - 新增 `_prune_engine_cache(cache_dir, keep_key=..., max_artifacts=16)`：按 mtime 从旧到新，通过原子 rename 隔离后 unlink，保留最新 16 个 `<digest>-*` 生成文件及其 manifest；同时把 `compiler-*.id` 限制为 16 个、清理超过 3600 秒的 `*.tmp`。Windows 上正被进程映射的旧 `.dll`/`.pyd` 若拒绝 rename/unlink，则延迟到后续运行再回收，不影响当前构建。
  - 缓存命中路径也会执行一次有界修剪，升级后首次运行即可自动回收历史 30+ 份重复 `.dylib`/`.so` 缓存。

- **Wheel 验收工作区确定性清理（`tools/verify_wheel.py`，S-2）**：
  - `uv venv`、wheel 安装、CLI 验证和生成的 fixture 全部位于一个由本工具拥有的临时工作区；无论成功、异常还是超时，都在 `finally` 中 `shutil.rmtree(..., ignore_errors=True)`。
  - 新增 `--work-dir` 以复用调用方自有的工作区（该目录不会被自动删除），以及 `--keep-work` 在诊断时保留内部工作区；默认行为不再遗留完整 virtualenv。

- **Issue #34 预算脚本临时产物清理（`scripts/check_issue34_test_budget.py`，S-3）**：
  - 当未显式传入 `--artifacts` 时，脚本创建的内部 `issue34-budget-*` 目录会在 `finally` 中删除；显式传入的目录仍由调用方管理。
  - 脚本自身保持 200 行预算、四段式头注释与原有超时/无重试语义。

- **测试套件临时目录所有权（S-4）**：
  - 新增 `tests/temp_cleanup.py`：所有测试临时根通过 `tempdir()` 登记，并在 `pytest_sessionfinish` 与进程 `atexit` 时统一清理；`tests/conftest.py` 的 `nr-s3-accel-*` / `nr-s3-runtime-*` 也纳入该所有权。
  - `tests/nr_s6_round4_probe.py` 的 `nr4-probe-*` 改为 `TemporaryDirectory`，在返回前读取 abort 记录并删除工作区；`run_ranks` 调用方改用 `world.abort_record`，不再持有已删除的路径。
  - `r6cas-`、`r7cas-`、`r10...`、`nr-s3-r*-wheel-`、`mp_` 等短根目录在需要控制路径长度时仍使用短目录，但全部登记到会话清理；不再产生无主 `nr4-probe-*` 或 `nr-s3-accel-*` 目录。

- **回归测试与发布元数据**：
  - 新增 `tests/test_storage_lifecycle_0_9_15.py`，真实覆盖：两个不同 `PATH` 下同一编译器的身份/摘要一致并命中缓存；`_prune_engine_cache` 将产物/manifest 限制在 `<= 16` 且对 `PermissionError` 延迟回收；发布失败与快照失败不留下候选或临时文件；`verify_wheel.py`、`nr_s6_round4_probe.py` 与预算脚本在隔离 `TMPDIR` 中留下 0 个新条目。
  - `pyproject.toml`、`mindie_sim.__version__`、中英文 README 与 release workflow 统一升级到 `0.9.15`。

升级说明：无需手工删除旧缓存。升级后第一次 `build-accel`（或首次缓存命中）会按新身份计算摘要并执行有界修剪；仍被 Windows 进程映射的旧库会在进程退出后的后续运行中继续回收。完整测试以 `pytest -q -m "not live_npu"` 与随包基准记录为准。
