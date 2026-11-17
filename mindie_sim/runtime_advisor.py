"""Runtime environment probe, worker budgeting and agent compute-offload advisor.

[Role]
  Probe usable CPU cores, available RAM and C11 ``_engine_accel`` readiness using
  only the Python standard library; bound parallel worker counts within the
  remembered memory headroom (Principle 18); and emit actionable compute-offload
  guidance for calling AI agents and CLI users.
[Invariants]
  Zero non-stdlib dependencies (no ``psutil``); never mutates simulation physics
  or caches evaluation results; worker count is always ``>= 1`` and bounded by
  ``min(cpu_count, ram_budget_workers, task_count, MAX_AUTO_POOL_WORKERS)``.
  Container limits (cgroup v2 and v1 CPU quota / memory) always win over host
  totals, and one workflow uses exactly one flat worker pool.
[Timeouts & Locks]
  All probes are in-process syscalls or bounded ``<= 5s`` subprocess queries;
  ``except TimeoutError: raise`` precedes every ``except OSError:``.
[Side-Effect Policy]
  Pure probe by default; ``doctor_report(compile_accel=True)`` compiles into the
  configured ``MINDIE_CACHE_HOME``; ``emit_heavy_run_notice`` writes only to
  ``sys.stderr`` when warranted.
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from typing import Any, Mapping

BYTES_PER_GIB = 1024 * 1024 * 1024
KIB_PER_GIB = 1024 * 1024
LOW_CORE_THRESHOLD = 4
LOW_RAM_GIB_THRESHOLD = 8.0
RECOMMENDED_OFFLOAD_CORES = 32
RECOMMENDED_OFFLOAD_RAM_GIB = 64
DEFAULT_WORKER_RAM_GIB = 1.5
HOST_RESERVED_RAM_GIB = 2.0
MAX_AUTO_POOL_WORKERS = 32
HEAVY_WORK_UNIT_THRESHOLD = 8
SYSCTL_PROBE_TIMEOUT_S = 5
CGROUP_ROOT = "/sys/fs/cgroup"
CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
CGROUP_V2_MEMORY_MAX = "/sys/fs/cgroup/memory.max"
CGROUP_V2_MEMORY_CURRENT = "/sys/fs/cgroup/memory.current"
CGROUP_V1_CPU_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_CPU_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
CGROUP_V1_MEMORY_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
CGROUP_V1_MEMORY_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
CPUSET_EFFECTIVE = "/sys/fs/cgroup/cpuset.cpus.effective"
CGROUP_V2_CPU_FIELDS = 2
CGROUP_V1_CPU_FIELDS = 2
UNLIMITED_MARKERS = frozenset({"max", "-1"})
#: A cgroup memory limit at or above this value is the kernel's "no limit".
UNLIMITED_MEMORY_BYTES = 1 << 60
MIB_PER_GIB = 1024

#: Execution labels.  Build availability is never execution coverage.
BACKEND_C11 = "c11_accel"
BACKEND_PYTHON = "python_fallback"
BACKEND_PYTHON_REQUESTED = "python"
BACKEND_MIXED = "c11_accel+python"
C_NATIVE_BACKENDS = frozenset({"c_native", "c_native_ready"})


def _read_text(path: str) -> str | None:
    """Read one small sysfs/procfs cell, or ``None`` when unavailable."""
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except TimeoutError:
        raise
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    text = _read_text(path)
    if text is None or text in UNLIMITED_MARKERS or not text.isdigit():
        return None
    return int(text)


def _probe_cgroup_cpus() -> int | None:
    """Effective CPU quota of this process (cgroup v2, then v1, then cpuset)."""
    text = _read_text(CGROUP_V2_CPU_MAX)
    if text:
        fields = text.split()
        if len(fields) == CGROUP_V2_CPU_FIELDS and fields[0] not in UNLIMITED_MARKERS:
            period = int(fields[1]) if fields[1].isdigit() else 0
            quota = int(fields[0]) if fields[0].isdigit() else 0
            if period > 0 and quota > 0:
                return max(1, quota // period)
    quota = _read_int(CGROUP_V1_CPU_QUOTA)
    period = _read_int(CGROUP_V1_CPU_PERIOD)
    if quota is not None and period and quota > 0:
        return max(1, quota // period)
    return None


def _probe_affinity_cpus() -> int | None:
    """CPUs this process may actually run on (Linux/macOS sched affinity)."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        try:
            return max(1, len(getter(0)))
        except TimeoutError:
            raise
        except OSError:
            return None
    text = _read_text(CPUSET_EFFECTIVE)
    if text:
        total = 0
        for part in text.split(","):
            bounds = part.split("-")
            try:
                low = int(bounds[0])
                high = int(bounds[-1])
            except ValueError:
                continue
            total += high - low + 1
        if total > 0:
            return total
    return None


def usable_cpu_count() -> int:
    """Min of host CPUs, container CPU quota and process affinity."""
    candidates = [max(1, int(os.cpu_count() or 1))]
    for probe in (_probe_cgroup_cpus(), _probe_affinity_cpus()):
        if probe is not None and probe > 0:
            candidates.append(int(probe))
    return max(1, min(candidates))


def _cgroup_memory_gib() -> tuple[float, float] | None:
    limit = _read_int(CGROUP_V2_MEMORY_MAX)
    usage = _read_int(CGROUP_V2_MEMORY_CURRENT)
    if limit is None or limit >= UNLIMITED_MEMORY_BYTES:
        limit = _read_int(CGROUP_V1_MEMORY_LIMIT)
        usage = _read_int(CGROUP_V1_MEMORY_USAGE)
        if limit is not None and limit >= UNLIMITED_MEMORY_BYTES:
            limit = None
    if limit is None or limit <= 0:
        return None
    used = usage if usage is not None and usage >= 0 else 0
    return (round(limit / BYTES_PER_GIB, 2),
            round(max(0, limit - used) / BYTES_PER_GIB, 2))


def _proc_memory_gib() -> tuple[float, float] | None:
    text = _read_text("/proc/meminfo")
    if not text:
        return None
    total_kib = avail_kib = 0.0
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total_kib = float(line.split()[1])
        elif line.startswith("MemAvailable:"):
            avail_kib = float(line.split()[1])
    if total_kib <= 0.0:
        return None
    avail = avail_kib if avail_kib > 0.0 else total_kib * 0.5
    return (round(total_kib / KIB_PER_GIB, 2), round(avail / KIB_PER_GIB, 2))


class _MemStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_memory_gib() -> tuple[float, float] | None:
    try:
        stat = _MemStatusEx()
        stat.dwLength = ctypes.sizeof(_MemStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return (round(stat.ullTotalPhys / BYTES_PER_GIB, 2),
                    round(stat.ullAvailPhys / BYTES_PER_GIB, 2))
    except TimeoutError:
        raise
    except (AttributeError, OSError):
        return None
    return None


def _sysconf_memory_gib() -> tuple[float, float] | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    total_gib = (pages * page_size) / BYTES_PER_GIB
    avail = total_gib * 0.5
    if "SC_AVPHYS_PAGES" in os.sysconf_names:
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        if isinstance(avail_pages, int) and avail_pages > 0:
            avail = (avail_pages * page_size) / BYTES_PER_GIB
    return (round(total_gib, 2), round(avail, 2))


def _darwin_memory_gib() -> tuple[float, float] | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
            timeout=SYSCTL_PROBE_TIMEOUT_S, check=False)
    except TimeoutError:
        raise
    except OSError:
        return None
    if out.returncode != 0 or not out.stdout.strip().isdigit():
        return None
    total_gib = int(out.stdout.strip()) / BYTES_PER_GIB
    return (round(total_gib, 2), round(total_gib * 0.5, 2))


def probe_memory_gib() -> tuple[float, float]:
    """``(total_gib, available_gib)`` bounded by the container memory limit."""
    container = _cgroup_memory_gib()
    host = _proc_memory_gib()
    if host is None and os.name == "nt":
        host = _windows_memory_gib()
    if host is None:
        host = _sysconf_memory_gib()
    if host is None and sys.platform == "darwin":
        host = _darwin_memory_gib()
    if host is None:
        host = (0.0, 0.0)
    if container is None:
        return host
    total = min(host[0], container[0]) if host[0] > 0.0 else container[0]
    avail = min(host[1], container[1])
    return (round(total, 2), round(max(0.0, avail), 2))


def _compiler_identity() -> tuple[bool, str | None, str | None]:
    """``(available, basename, reason)`` without exposing a host path."""
    from .prediction.runtime import accel_build
    try:
        path, reason = accel_build.discover_compiler()
    except TimeoutError:
        raise
    except (OSError, RuntimeError) as error:
        return False, None, f"compiler_probe_failed:{error}"
    if path is None:
        return False, None, reason
    return True, os.path.basename(str(path)), None


def probe_host_environment(*, ensure_accel: bool = False) -> dict[str, Any]:
    """Inspect usable CPU/RAM headroom, compiler availability and build state."""
    from .prediction.runtime import accel_build
    cpu_count = usable_cpu_count()
    total_ram_gib, avail_ram_gib = probe_memory_gib()
    compiler_available, compiler_name, compiler_reason = _compiler_identity()
    build_status = accel_build.build_accel(
        check_only=not (ensure_accel and compiler_available))
    library_ready = bool(build_status.get("ok"))
    warnings: list[str] = []
    if not compiler_available:
        warnings.append(
            "No C compiler (clang/gcc/cc/cl.exe) detected on PATH; "
            "mindie-sim runs on the pure-Python Engine fallback (~20-60x slower)."
        )
    elif not library_ready:
        warnings.append(
            f"C11 _engine_accel is not built yet ({build_status.get('reason')}); "
            "run `mindie-sim doctor` or `mindie-sim build-accel` to prebuild."
        )
    # An existing usable binary stays available even without a compiler.
    return {
        # Build availability and execution coverage are separate facts.
        "engine_backend": BACKEND_PYTHON,
        "execution_backend": BACKEND_PYTHON,
        "c11_accel_available": library_ready,
        "c11_accel_active": False,
        "compiler_available": compiler_available,
        "compiler": compiler_name,
        "compiler_reason": compiler_reason,
        "library_ready": library_ready,
        "library_reason": (None if library_ready
                           else (build_status.get("reason") or "unavailable")),
        "accel_reason": None if library_ready else (
            build_status.get("reason") or "unavailable"),
        "host_cpu_count": cpu_count,
        "host_total_ram_gib": total_ram_gib,
        "host_available_ram_gib": avail_ram_gib,
        "platform": platform.platform(),
        "warnings": warnings,
    }


def resolve_jobs(
    requested: int | str | None,
    *,
    task_count: int,
    per_worker_ram_gib: float = DEFAULT_WORKER_RAM_GIB,
) -> int:
    """Resolve ``--jobs auto`` bounded by CPU quota, RAM headroom and task count."""
    tasks = max(1, int(task_count))
    if isinstance(requested, int) and not isinstance(requested, bool):
        return max(1, min(int(requested), tasks))
    text = str(requested or "auto").strip().lower()
    if text != "auto":
        try:
            return max(1, min(int(text), tasks))
        except ValueError:
            return 1
    cpu_count = usable_cpu_count()
    _total_gib, avail_gib = probe_memory_gib()
    if avail_gib > 0.0 and per_worker_ram_gib > 0.0:
        usable_gib = max(per_worker_ram_gib, avail_gib - HOST_RESERVED_RAM_GIB)
        ram_cap = max(1, int(usable_gib // per_worker_ram_gib))
    else:
        ram_cap = cpu_count
    return max(1, min(cpu_count, ram_cap, tasks, MAX_AUTO_POOL_WORKERS))


def canonical_engine_backend(status: Mapping[str, Any] | None) -> str:
    """Truthful label from real execution counters, never from build state.

    ``backend="python"`` stays Python even when a compiled library exists, and
    an attached-but-unused kernel is not reported as native execution.
    """
    if not isinstance(status, Mapping):
        return BACKEND_PYTHON
    if str(status.get("requested") or status.get("accel_backend") or "") in {
            BACKEND_PYTHON_REQUESTED, "py"}:
        return BACKEND_PYTHON_REQUESTED
    native_steps = int(status.get("native_steps") or 0)
    python_steps = int(status.get("python_steps") or 0)
    if native_steps <= 0:
        return BACKEND_PYTHON
    if python_steps > 0:
        return BACKEND_MIXED
    return BACKEND_C11


def aggregate_execution_telemetry(statuses) -> dict[str, Any]:
    """Sum worker execution counters into one honest coverage record."""
    native = python = build_true = 0
    reasons: list[str] = []
    for status in statuses or ():
        if not isinstance(status, Mapping):
            continue
        native += int(status.get("native_steps") or 0)
        python += int(status.get("python_steps") or 0)
        if status.get("c11_accel_available") or status.get("toolchain_ready"):
            build_true += 1
        reason = status.get("fallback_reason") or status.get("accel_reason")
        if reason and reason not in reasons:
            reasons.append(str(reason))
    if native and python:
        backend = BACKEND_MIXED
    elif native:
        backend = BACKEND_C11
    else:
        backend = BACKEND_PYTHON
    return {
        "execution_backend": backend,
        "native_steps": native,
        "python_steps": python,
        "workers_reporting_build": build_true,
        "fallback_reasons": reasons,
    }


def build_runtime_environment(
    *,
    engine: Any = None,
    workers_used: int = 1,
    estimated_work_units: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build ``(runtime_environment, agent_advisory)`` for CLI/API outputs."""
    from .prediction.runtime import accel_bridge
    host = probe_host_environment(ensure_accel=False)
    if engine is not None:
        status = accel_bridge.accel_status(engine)
        host["execution_backend"] = canonical_engine_backend(status)
        host["engine_backend"] = host["execution_backend"]
        host["native_steps"] = int(status.get("native_steps") or 0)
        host["python_steps"] = int(status.get("python_steps") or 0)
        host["accel_reason"] = status.get("fallback_reason")
        host["c11_accel_active"] = int(status.get("native_steps") or 0) > 0
    host["workers_used"] = max(1, int(workers_used))
    offload_recommended = (
        host["execution_backend"] == BACKEND_PYTHON
        or host["host_cpu_count"] <= LOW_CORE_THRESHOLD
        or (0.0 < host["host_available_ram_gib"] < LOW_RAM_GIB_THRESHOLD)
        or (estimated_work_units >= HEAVY_WORK_UNIT_THRESHOLD
            and host["host_cpu_count"] < RECOMMENDED_OFFLOAD_CORES // 2)
    )
    reasons: list[str] = list(host["warnings"])
    if host["host_cpu_count"] <= LOW_CORE_THRESHOLD:
        reasons.append(
            f"Only {host['host_cpu_count']} usable CPU cores; multi-SKU/SLA/sweep "
            f"workloads run faster on >= {RECOMMENDED_OFFLOAD_CORES}-core servers."
        )
    advisory = {
        "offload_recommended": offload_recommended,
        "severity": ("warn" if host["execution_backend"] == BACKEND_PYTHON
                     else ("info" if offload_recommended else "ok")),
        "reasons": reasons,
        "recommended_server_spec": {
            "min_cpu_cores": RECOMMENDED_OFFLOAD_CORES,
            "min_ram_gib": RECOMMENDED_OFFLOAD_RAM_GIB,
            "requires_c11_compiler": True,
        },
        "agent_steps": [
            "1. Probe fleet CPU/RAM inventory via the coordinator or `remote-dev` SSH.",
            "2. Select a Linux server with >= 32 usable cores, >= 64 GiB free RAM, gcc/clang.",
            "3. Run `mindie-sim doctor` on the remote host to build and verify C11 accel.",
            "4. Run `mindie-sim compare-skus ... --jobs auto` or `mindie-sim sla ... --jobs auto`.",
        ] if offload_recommended else [],
    }
    return host, advisory


def emit_heavy_run_notice(command: str, *, workers: int, work_units: int) -> None:
    """Emit an immediate one-line stderr notice before heavy degraded runs."""
    env, advisory = build_runtime_environment(
        workers_used=workers, estimated_work_units=work_units)
    if not advisory["offload_recommended"]:
        return
    reason = "; ".join(advisory["reasons"]) or "large simulation batch"
    sys.stderr.write(
        f"[mindie-sim:{command}] advisory: backend={env['execution_backend']} "
        f"workers={workers}/{env['host_cpu_count']}cpu "
        f"avail_ram={env['host_available_ram_gib']}GiB ({reason}). "
        f"Run `mindie-sim doctor` for offload guidance.\n")
    sys.stderr.flush()


def doctor_report(*, compile_accel: bool = True) -> dict[str, Any]:
    """Build and verify C11 ``_engine_accel`` and report agent readiness."""
    host = probe_host_environment(ensure_accel=compile_accel)
    recommended = resolve_jobs("auto", task_count=MAX_AUTO_POOL_WORKERS)
    _env, advisory = build_runtime_environment(
        workers_used=recommended,
        estimated_work_units=HEAVY_WORK_UNIT_THRESHOLD)
    return {
        "ok": bool(host["c11_accel_available"]),
        "runtime_environment": host,
        "recommended_auto_workers": recommended,
        "agent_offload_guidance": advisory,
    }
