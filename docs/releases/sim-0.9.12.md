# mindie-sim 0.9.12：Windows CP936 UTF-8 / LF I/O 确定性与超时异常透传

[English](sim-0.9.12.en.md)

针对中文 Windows OEM 代码页（`CP936`）、零弹窗验证工具链及目录读取边界补齐跨平台确定性保护：

- **CLI 与后端配置显式 UTF-8 / Unix `LF` I/O（`mindie_sim/__main__.py`、`backends/ascend.py`、`backends/nvidia.py`）**：CLI JSON 输入读取与结果写出统一固定 `encoding="utf-8"` 与 `newline="\n"`，并在标准输出受限于宿主窄代码页（如 `CP936`）无法编码非 GBK Unicode 字符时自动降级为无损 `ensure_ascii=True` JSON，杜绝 `UnicodeDecodeError` / `UnicodeEncodeError` 与 Windows `CRLF` 换行漂移。
- **硬件与模型目录 `TimeoutError` 严格透传（`custom_hardware.py`、`model_catalog.py`）**：在所有 `except OSError` 边界前显式重抛 `TimeoutError`，避免将超时异常静默忽略或误标为配置文件损坏。
- **Windows Wheel 验收无黑框与 UTF-8 加固（`tools/verify_wheel.py`）**：Windows 子进程调用统一附带 `CREATE_NO_WINDOW` 与 `encoding="utf-8", errors="replace"`。

包内 `__version__` 同步为 `0.9.12`。本版本的实际测试与基准结果以随包 `benchmark-results.json`、`benchmark-results.md` 及对应 CI 记录为准。
