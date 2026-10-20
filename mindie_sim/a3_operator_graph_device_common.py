"""Shared device-runtime errors, budgets, environment facts and artifacts.

[Role]
  Hold the cooperative budget clock, exact error/failure carriers, environment
  fact collection and the condition-artifact record shared by the A3 graph
  device preparation, timing and orchestration modules.  Torch is never
  imported at module import time.

[Invariants]
  * Budget ceilings are 24 s case, 1200 s global work, 60 s setup and 120 s
    cleanup; the supervisor minimum is 1380 s.
  * Environment records only facts actually returned by the installed stack;
    unavailable queries remain explicit errors/unsupported statuses.
  * Failure carriers preserve exact exception type/message/traceback and mark
    retry as not attempted.  A pending API remains distinct from a failure.
  * Condition metadata keeps native default-output, retain-last-only lifetime,
    no-pool-sharing and no-out-overload facts visible.

[Timeouts & Locks]
  The budget clock is cooperative host-side accounting with no locks or
  threads.  Native calls cannot be preempted; the outer supervisor is the hard
  bound.  Cleanup has its own 120 s deadline.

[Side-Effect Policy]
  Importing and calling these helpers allocates no device tensor and writes no
  file.  Queries are read-only device facts.
"""
import importlib.metadata
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping

def exception_info(exc: BaseException) -> dict[str, Any]:
    """Preserve the exact exception type/message plus traceback and no retry."""
    return {
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": str(exc),
        "repr": repr(exc),
        "traceback": "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__)),
        "retry": "not_attempted",
    }


class BudgetExceeded(RuntimeError):
    def __init__(self, stage: str, elapsed_s: float, limit_s: float) -> None:
        self.stage = stage
        self.elapsed_s = float(elapsed_s)
        self.limit_s = float(limit_s)
        super().__init__(
            f"{type(self).__name__} at {stage}: elapsed={self.elapsed_s:.6f}s "
            f"limit={self.limit_s:.6f}s")


class SetupBudgetExceeded(BudgetExceeded):
    pass


class GlobalWorkBudgetExceeded(BudgetExceeded):
    pass


class CaseBudgetExceeded(BudgetExceeded):
    pass


class CleanupBudgetExceeded(BudgetExceeded):
    pass


class DeviceSetupError(RuntimeError):
    pass


class PendingCondition(RuntimeError):
    """The exact installed API required by a condition is unavailable."""


class ConditionFailure(RuntimeError):
    def __init__(self, phase: str, cause: BaseException,
                 metadata: Mapping[str, Any], *, status: str = "failed",
                 stop_run: bool = False, reason: str | None = None) -> None:
        self.phase = phase
        self.cause = cause
        self.metadata = dict(metadata)
        self.status = status
        self.stop_run = stop_run
        self.reason = reason or f"{phase}: {cause}"
        super().__init__(self.reason)


class TrialFailure(RuntimeError):
    def __init__(self, trial: dict[str, Any], cause: BaseException, *,
                 stop_run: bool = False) -> None:
        self.trial = trial
        self.cause = cause
        self.error = exception_info(cause)
        self.stop_run = stop_run
        super().__init__(f"trial failed: {cause}")


class BudgetClock:
    """Cooperative setup/work/case/cleanup deadlines with recorded accounting."""

    def __init__(self, budgets: Mapping[str, Any]) -> None:
        self.budgets = {key: float(value) for key, value in budgets.items()
                        if key in ("case", "global_work", "setup", "cleanup")}
        for name, ceiling in (("case", 24.0), ("global_work", 1200.0),
                              ("setup", 60.0), ("cleanup", 120.0)):
            if self.budgets[name] > ceiling:
                raise ValueError(f"{name} budget exceeds {ceiling}")
        self.created = time.monotonic()
        self.setup_deadline = self.created + self.budgets["setup"]
        self.work_started: float | None = None
        self.work_deadline: float | None = None
        self.cleanup_started: float | None = None
        self.cleanup_deadline: float | None = None

    def check_setup(self, stage: str) -> None:
        elapsed = time.monotonic() - self.created
        if elapsed > self.budgets["setup"]:
            raise SetupBudgetExceeded(stage, elapsed, self.budgets["setup"])

    def start_work(self) -> None:
        self.work_started = time.monotonic()
        self.work_deadline = self.work_started + self.budgets["global_work"]

    def check_work(self, stage: str) -> None:
        if self.work_started is None:
            return
        elapsed = time.monotonic() - self.work_started
        if elapsed > self.budgets["global_work"]:
            raise GlobalWorkBudgetExceeded(stage, elapsed,
                                           self.budgets["global_work"])

    def work_elapsed(self) -> float:
        if self.work_started is None:
            return 0.0
        return time.monotonic() - self.work_started

    def start_cleanup(self) -> None:
        self.cleanup_started = time.monotonic()
        self.cleanup_deadline = self.cleanup_started + self.budgets["cleanup"]

    def check_cleanup(self, stage: str) -> None:
        if self.cleanup_started is None:
            raise RuntimeError("cleanup clock has not started")
        elapsed = time.monotonic() - self.cleanup_started
        if elapsed > self.budgets["cleanup"]:
            raise CleanupBudgetExceeded(stage, elapsed, self.budgets["cleanup"])

    def cleanup_elapsed(self) -> float:
        if self.cleanup_started is None:
            return 0.0
        return time.monotonic() - self.cleanup_started


@dataclass
class ConditionArtifact:
    condition: dict[str, Any]
    tensors: Any
    mode: str
    graph: Any | None
    events: tuple[Any, Any] | None
    capture_last: Any
    final_output: Any
    metadata: dict[str, Any]
    active_case_seconds: float


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        if all(item is None or isinstance(item, (bool, int, float, str))
               for item in value):
            return [item if not isinstance(item, float) or math.isfinite(item)
                    else None for item in value]
        return [repr(item)[:300] for item in value]
    return repr(value)[:500]


def _safe_call(label: str, function: Any) -> dict[str, Any]:
    try:
        return {"status": "ok", "label": label,
                "value": _json_safe(function())}
    except Exception as exc:
        return {"status": "error", "label": label,
                "error": exception_info(exc)}


def _memory_facts(torch: Any) -> dict[str, Any]:
    npu = getattr(torch, "npu", None)
    result: dict[str, Any] = {"source": "installed torch.npu allocator queries"}
    for name in ("memory_allocated", "max_memory_allocated", "memory_reserved",
                 "max_memory_reserved"):
        function = getattr(npu, name, None)
        if callable(function):
            result[name] = _safe_call(name, function)
        else:
            result[name] = {"status": "unsupported", "label": name}
    function = getattr(npu, "mem_get_info", None)
    if callable(function):
        result["mem_get_info"] = _safe_call("mem_get_info", function)
    else:
        result["mem_get_info"] = {"status": "unsupported", "label": "mem_get_info"}
    return result


def _device_properties(torch: Any, index: int) -> dict[str, Any]:
    npu = getattr(torch, "npu", None)
    result: dict[str, Any] = {}
    function = getattr(npu, "get_device_name", None)
    if callable(function):
        result["name"] = _safe_call("get_device_name", lambda: function(index))
    function = getattr(npu, "get_device_properties", None)
    if callable(function):
        try:
            properties = function(index)
        except Exception as exc:
            result["properties"] = {"status": "error",
                                    "error": exception_info(exc)}
        else:
            values: dict[str, Any] = {}
            for name in dir(properties):
                if name.startswith("_"):
                    continue
                try:
                    value = getattr(properties, name)
                except Exception as exc:
                    values[name] = {"status": "error", "message": str(exc)}
                    continue
                if callable(value):
                    continue
                values[name] = _json_safe(value)
            result["properties"] = {"status": "ok", "values": values}
    else:
        result["properties"] = {"status": "unsupported",
                                "label": "get_device_properties"}
    return result


def _collect_environment(torch: Any, torch_npu: Any, device: Any,
                         stream: Any, index: int) -> dict[str, Any]:
    npu = torch.npu
    package_versions: dict[str, Any] = {}
    for package in ("torch", "torch-npu", "torch_npu", "numpy"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
        except Exception as exc:
            package_versions[package] = {"error": str(exc)}
    initialized = getattr(getattr(torch_npu, "npu", None), "is_initialized", None)
    facts: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": getattr(torch, "__version__", None),
        "torch_git_version": getattr(getattr(torch, "version", None),
                                      "git_version", None),
        "torch_npu_version": getattr(torch_npu, "__version__", None),
        "package_versions": package_versions,
        "visible_devices_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "device_id_env": os.environ.get("ASCEND_DEVICE_ID"),
        "device": {
            "requested": str(device),
            "selected": str(device),
            "logical_index": index,
            "npu_is_available": _safe_call(
                "torch.npu.is_available", lambda: bool(npu.is_available())),
            "npu_is_initialized_after_setup": (
                _safe_call("torch_npu.npu.is_initialized",
                           lambda: bool(initialized()))
                if callable(initialized) else
                {"status": "unsupported", "label": "is_initialized"}),
            "device_count": _safe_call("torch.npu.device_count",
                                       lambda: int(npu.device_count())),
            "facts": _device_properties(torch, index),
        },
        "stream": {
            "repr": repr(stream),
            "created_once": True,
            "explicit": True,
        },
        "memory_at_setup": _memory_facts(torch),
        "clocks": {
            "host": "time.perf_counter_ns",
            "device_event": "torch.npu.Event(enable_timing=True)",
            "device_clock_rate": {
                "status": "not_independently_queried",
                "note": "actual device properties above are preserved verbatim; "
                        "no clock rate is invented",
            },
        },
    }
    return facts
