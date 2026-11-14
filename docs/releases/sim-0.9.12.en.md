# mindie-sim 0.9.12: Windows CP936 UTF-8 / LF I/O Determinism & Timeout Propagation

[中文](sim-0.9.12.md)

Hardens cross-platform CLI and catalog file I/O on Chinese OEM code-page (`CP936`) Windows, enforces strict `TimeoutError` propagation, and eliminates console popups in wheel verification:

- **Explicit UTF-8 & Unix `LF` CLI / Backend I/O (`mindie_sim/__main__.py`, `backends/ascend.py`, `backends/nvidia.py`)**: Fixes `read_json`, `emit`, and backend profile loaders to use explicit `encoding="utf-8"` and `newline="\n"`, with automatic `ensure_ascii=True` fallback when narrow console code pages (`CP936`) reject non-GBK Unicode characters.
- **Strict `TimeoutError` Propagation (`custom_hardware.py`, `model_catalog.py`)**: Re-raises `TimeoutError` before every `except OSError` boundary so Python 3.11+ timeout exceptions are never masked as unreadable config files or ignored optional SKUs.
- **Windowless UTF-8 Wheel Verification (`tools/verify_wheel.py`)**: Applies `CREATE_NO_WINDOW` and `encoding="utf-8", errors="replace"` across Windows subprocess invocations.

In-package `__version__` is aligned to `0.9.12`. Actual test and benchmark results are recorded in the packaged `benchmark-results.json`, `benchmark-results.md`, and CI runs.
