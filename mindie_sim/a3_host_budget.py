"""Cross-platform host execution budget and workload resource estimator for predictive simulations.

[Role] Estimate simulation host resource demand (`estimate_prediction_resources`) and refuse oversized prediction work on undersized hosts (`check_prediction_host`) before DAG allocation across Windows, macOS, and Linux.
[Invariants]
- Limits simulator host memory/CPU work, never modeled NPU HBM or model accuracy.
- Never hardcodes `sys.platform == "darwin"` to assume only macOS is a desktop or that Windows/Linux cannot OOM: capacity checks use actual host memory across Windows (`GlobalMemoryStatusEx`), Linux (`/proc/meminfo`), and macOS (`sysctl`/`sysconf`).
[Timeouts & Locks] Constant-time OS kernel memory queries; zero subprocesses, locks, or sleeps.
[Side-Effect Policy] Read host capacity only; never submit a remote job implicitly.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import os
from pathlib import Path
import sys
from typing import Any

from .a3_types import A3InputError, CaseSpec


def _windows_memory_bytes(kernel32: Any = None) -> int | None:
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx(dwLength=ctypes.sizeof(MemoryStatusEx))
    lib = kernel32 if kernel32 is not None else getattr(ctypes, "windll", None).kernel32
    if lib is None or not lib.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    avail = int(status.ullAvailPhys)
    total = int(status.ullTotalPhys)
    return avail if avail > 0 else (total if total > 0 else None)


def _linux_memory_bytes(meminfo_text: str | None = None) -> int | None:
    try:
        text = meminfo_text if meminfo_text is not None else Path("/proc/meminfo").read_text(encoding="utf-8")
        values_kb: dict[str, int] = {}
        for line in text.splitlines():
            key, sep, rest = line.partition(":")
            if not sep:
                continue
            tokens = rest.strip().split()
            if tokens and tokens[0].isdigit():
                values_kb[key.strip()] = int(tokens[0])
        avail_kb = values_kb.get("MemAvailable") or values_kb.get("MemTotal")
        if avail_kb and avail_kb > 0:
            return avail_kb * 1024
    except OSError:
        pass
    return None


def _posix_total_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, ValueError, OSError):
        pass
    return None


def host_memory_bytes(*, os_name: str | None = None, platform_name: str | None = None) -> int | None:
    """Return usable/physical host memory bytes across Windows, macOS, and Linux."""
    actual_os = os.name if os_name is None else os_name
    actual_platform = sys.platform if platform_name is None else platform_name
    if actual_os == "nt" or actual_platform.startswith("win"):
        return _windows_memory_bytes()
    if actual_platform.startswith("linux"):
        mem = _linux_memory_bytes()
        if mem is not None:
            return mem
    return _posix_total_memory_bytes()


def desktop_memory_bytes() -> int | None:
    """Backward-compatible alias used by callers and monkeypatch fixtures."""
    return host_memory_bytes()


def estimate_prediction_resources(case: CaseSpec | Mapping[str, Any], trace_detail: str = "bounded") -> dict[str, int | str]:
    """Compute OS-agnostic resource expectations (`WorkloadExpectation`) before running simulation."""
    if isinstance(case, Mapping):
        case = CaseSpec.from_mapping(case)
    chunk = max(
        1,
        min(
            case.runtime.max_num_batched_tokens,
            case.runtime.max_num_input_tokens or case.runtime.max_num_batched_tokens,
        ),
    )
    iterations = sum(
        math.ceil(max(0, request.prompt_tokens - request.prefix_tokens) / chunk)
        + request.output_tokens
        for request in case.workload.requests
    )
    layer_iterations = max(1, case.model.layers) * iterations
    work_per_16gib = 512 if trace_detail == "full" else 4096
    # Base runtime footprint (~160 MiB) + per-layer-iteration active working set.
    expected_memory_mb = max(
        160,
        int(math.ceil((layer_iterations / work_per_16gib) * 16384)),
    )
    if expected_memory_mb <= 512:
        tier, expected_cpu_cores = "light", 1
    elif expected_memory_mb <= 4096:
        tier, expected_cpu_cores = "medium", 4
    else:
        tier, expected_cpu_cores = "heavy", 8
    return {
        "tier": tier,
        "iterations": iterations,
        "layer_iterations": layer_iterations,
        "trace_detail": trace_detail,
        "expected_memory_mb": expected_memory_mb,
        "expected_cpu_cores": expected_cpu_cores,
    }


def check_prediction_host(
    case: CaseSpec,
    trace_detail: str = "bounded",
    *,
    host_capacity_bytes: int | None = None,
) -> dict[str, int | str]:
    """Validate that the target host's memory spec satisfies the simulation workload."""
    estimate = estimate_prediction_resources(case, trace_detail=trace_detail)
    capacity = desktop_memory_bytes() if host_capacity_bytes is None else int(host_capacity_bytes)
    if capacity is None or capacity <= 0:
        return estimate
    layer_iterations = int(estimate["layer_iterations"])
    work_per_16gib = 512 if trace_detail == "full" else 4096
    limit = max(128, int(capacity / (16 * 1024**3) * work_per_16gib))
    if layer_iterations > limit:
        raise A3InputError(
            "host_capacity_requires_remote:"
            f"layer_iterations={layer_iterations}>desktop_limit={limit};"
            f"expected_memory_mb={estimate['expected_memory_mb']};"
            "required_execution=remote_linux"
        )
    return {**estimate, "host_limit": limit}
