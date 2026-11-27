"""Isolated child-process Linux RSS / HWM measurement helper for solver & spine gates.

[Role] Spawn a fresh Python child process on Linux, record pre-workload ``VmRSS``
  and post-workload ``VmHWM`` from ``/proc/self/status`` (``delta = HWM - pre_rss``),
  and enforce bounded peak / incremental RSS ceilings (``< 250 MiB`` delta and
  ``<= 512 MiB`` HWM) without polluting the parent pytest worker's allocator state.
[Invariants] Never mutates parent ``os.environ``; invokes this module directly as
  a script (zero inline ``python -c`` blobs); fails loudly on non-Linux hosts or
  missing ``/proc/self/status`` metrics instead of returning fabricated zeroes;
  ``TimeoutError`` always propagates before ``OSError``.
[Timeouts & Locks] Bounded child subprocess timeout (default 120 s); no file locks.
[Side-Effect Policy] Read-only over repository files; spawns one ephemeral child
  process with isolated stdin/stdout pipes.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import gc
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

KIB_PER_MIB = 1024.0
DEFAULT_MAX_RSS_DELTA_MIB = 250.0
DEFAULT_MAX_RSS_HWM_MIB = 512.0
DEFAULT_RSS_PROBE_TIMEOUT_S = 120.0
_RESULT_PREFIX = "__MINDIE_RSS_PROBE_JSON__:"
_PROC_STATUS_PATH = Path("/proc/self/status")
_STATUS_KIB_RE = re.compile(r"^([A-Za-z0-9_]+):\s+(\d+)\s+kB\s*$", re.MULTILINE)


class RssBudgetError(ValueError):
    """Raised when child-process RSS delta or peak HWM exceeds its ceiling."""


@dataclass(frozen=True, slots=True)
class ChildRssMeasurement:
    """Exact pre-solve VmRSS and post-solve VmHWM snapshot from an isolated child."""

    pre_rss_kib: int
    pre_hwm_kib: int
    post_rss_kib: int
    post_hwm_kib: int
    result: Any

    @property
    def pre_rss_mib(self) -> float:
        return self.pre_rss_kib / KIB_PER_MIB

    @property
    def post_rss_mib(self) -> float:
        return self.post_rss_kib / KIB_PER_MIB

    @property
    def hwm_mib(self) -> float:
        return self.post_hwm_kib / KIB_PER_MIB

    @property
    def delta_hwm_mib(self) -> float:
        return (self.post_hwm_kib - self.pre_rss_kib) / KIB_PER_MIB


def read_proc_status_kib(field_name: str, *, status_path: Path = _PROC_STATUS_PATH) -> int:
    """Read an integer KiB metric (e.g. ``VmRSS`` or ``VmHWM``) from Linux procfs."""
    if not status_path.is_file():
        raise RuntimeError(f"rss_probe_requires_linux_proc_status:{status_path}")
    text = status_path.read_text(encoding="utf-8")
    for match in _STATUS_KIB_RE.finditer(text):
        if match.group(1) == field_name:
            return int(match.group(2))
    raise RuntimeError(f"rss_probe_proc_field_missing:{field_name}")


def _resolve_target(target: str) -> Any:
    module_name, sep, attr_path = target.partition(":")
    if not sep or not module_name or not attr_path:
        raise ValueError(f"rss_probe_invalid_target:{target!r}")
    obj: Any = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _build_child_env(env_overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    tests_root = repo_root / "tests"
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    parts = [str(repo_root), str(tests_root)]
    if existing_pp:
        parts.extend(p for p in existing_pp.split(os.pathsep) if p and p not in parts)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_overrides:
        for key, value in env_overrides.items():
            env[str(key)] = str(value)
    return env


def measure_child_rss(
    target: str,
    payload: Mapping[str, Any] | None = None,
    *,
    timeout_s: float = DEFAULT_RSS_PROBE_TIMEOUT_S,
    env_overrides: Mapping[str, str] | None = None,
) -> ChildRssMeasurement:
    """Run ``target(payload)`` in a fresh Python child and return VmRSS/VmHWM."""
    if sys.platform != "linux":
        raise RuntimeError(f"rss_probe_requires_linux:platform={sys.platform}")
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--target", target]
    serialized = json.dumps(dict(payload or {}), sort_keys=True)
    try:
        completed = subprocess.run(
            cmd,
            input=serialized,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
            check=False,
            env=_build_child_env(env_overrides),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except TimeoutError:
        raise
    except OSError as exc:
        raise RuntimeError(f"rss_probe_spawn_failed:{exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"rss_probe_child_failed:rc={completed.returncode}:"
            f"stderr={completed.stderr.strip()[-1000:]}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            raw = json.loads(line[len(_RESULT_PREFIX):])
            return ChildRssMeasurement(
                pre_rss_kib=int(raw["pre_rss_kib"]),
                pre_hwm_kib=int(raw["pre_hwm_kib"]),
                post_rss_kib=int(raw["post_rss_kib"]),
                post_hwm_kib=int(raw["post_hwm_kib"]),
                result=raw["result"],
            )
    raise RuntimeError("rss_probe_missing_result_marker")


def assert_rss_budget(
    measurement: ChildRssMeasurement,
    *,
    max_delta_mib: float = DEFAULT_MAX_RSS_DELTA_MIB,
    max_hwm_mib: float = DEFAULT_MAX_RSS_HWM_MIB,
) -> None:
    """Enforce strict incremental ``delta < max_delta_mib`` and ``HWM <= max_hwm_mib``."""
    if measurement.pre_rss_kib <= 0 or measurement.post_hwm_kib < measurement.pre_rss_kib:
        raise RssBudgetError(
            f"rss_measurement_invalid:pre_kib={measurement.pre_rss_kib}:"
            f"hwm_kib={measurement.post_hwm_kib}"
        )
    if measurement.delta_hwm_mib >= max_delta_mib:
        raise RssBudgetError(
            f"rss_delta_exceeded:delta_mib={measurement.delta_hwm_mib:.2f}:"
            f"limit_mib={max_delta_mib:.2f}:hwm_mib={measurement.hwm_mib:.2f}"
        )
    if measurement.hwm_mib > max_hwm_mib:
        raise RssBudgetError(
            f"rss_hwm_exceeded:hwm_mib={measurement.hwm_mib:.2f}:"
            f"limit_mib={max_hwm_mib:.2f}:delta_mib={measurement.delta_hwm_mib:.2f}"
        )


def tiny_engine_rss_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Minimal 1-layer real Engine workload verifying child RSS probe wiring."""
    from mindie_sim import api as sdk_api
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.results import StepRequest

    context = int(payload.get("context", 32))
    sku = str(payload.get("hardware", "a5"))
    case = compose_case("qwen3-32b", sku, layers=1, tp=1, dp=1)
    step = sdk_api.evaluate_step(
        case, StepRequest(phase="decode", tokens=1, context=context, sequences=1, scope="body"),
    )
    return {
        "ok": True,
        "span_ms": float(step.span_ms),
        "op_count": len(step.operators),
        "sku": str(case["machine"]["sku"]),
    }


def _child_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated Linux child RSS probe.")
    parser.add_argument("--target", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(sys.stdin.read() or "{}")
    fn = _resolve_target(args.target)
    gc.collect()
    pre_rss_kib = read_proc_status_kib("VmRSS")
    pre_hwm_kib = read_proc_status_kib("VmHWM")
    result = fn(payload)
    post_rss_kib = read_proc_status_kib("VmRSS")
    post_hwm_kib = read_proc_status_kib("VmHWM")
    record = {
        "pre_rss_kib": pre_rss_kib,
        "pre_hwm_kib": pre_hwm_kib,
        "post_rss_kib": post_rss_kib,
        "post_hwm_kib": post_hwm_kib,
        "result": result,
    }
    sys.stdout.write(f"{_RESULT_PREFIX}{json.dumps(record, sort_keys=True)}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_child_main())
