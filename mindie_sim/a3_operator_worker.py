"""Executable operator-campaign worker and CPU-only dry-run validator.

[Role]
  Consume a frozen campaign manifest and run one declared operator point per
  request on a coordinator-injected device.  Supported recipes are plain
  preallocated GEMM, vector/copy primitives, an explicitly named block-FP8
  reference decomposition, and probed torch_npu fused RMSNorm/SwiGLU routes.
  Unsupported exact recipes return pending with a reason and required backend.
  ``--dry-run`` validates the manifest/recipe wiring without importing torch or
  claiming a measured sample.

[Invariants]
  * No device is selected, leased, SSHed or mutated by this module.  The
    coordinator owns the visible device and sim_audit owns the NPU queue.
  * Every complete result retains all raw event samples, all trials, protocol,
    query context, observed API/backend/versions, timer/scope and preallocation.
  * Functional correctness is checked against a CPU reference sample before
    timing where an output exists.  A failed check aborts that request.
  * Native fused availability is probed, never inferred.  A reference
    decomposition is named as such and never labelled as the target kernel.
  * Pending conditions have no raw samples and are not counted as successes.
  * CPU dry-run results use source_class=derived and carry no measured claim.

[Timeouts & Locks]
  Each trial is bounded by configurable warmup/iterations; the outer
  coordinator bounds process lifetime.  No lock is held across a device call.

[Side-Effect Policy]
  Reads the explicit manifest, writes only the explicit results path, and
  allocates device tensors only when invoked in real mode.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .a3_types import A3InputError

from .a3_operator_worker_native import _native_softmax_geometry
from .a3_operator_worker_recipes import (
    PendingBackend, SUPPORTED_RECIPES, TorchNpuOperatorRunner,
    _dtype_bytes, _environment, _quant_scale_mode, _request_protocol,
    _scaled_expert_counts, _shape, _validate_request,
)
def _dry_run_results(manifest: Mapping[str, Any], *, run_id: str = "") -> dict[str, Any]:
    results = []
    for request in manifest.get("requests", ()):
        validation = _validate_request(request)
        status = validation["status"]
        result = {
            "measurement_id": request.get("measurement_id"),
            "request_key": request.get("request_key"),
            "condition_key": request.get("condition_key"),
            "condition_keys": request.get("lineage_condition_keys", [request.get("condition_key")]),
            "split": request.get("split"),
            "execution_kind": request.get("execution_kind"),
            "worker_recipe": request.get("worker_recipe"),
            "backend": request.get("backend"),
            "target_match": request.get("target_match"),
            "status": "dry_run_validated" if status == "dry_run_validated" else status,
            "reason": validation.get("reason", "manifest and recipe wiring validated; no device call"),
            "required_backend": validation.get("required_backend"),
            "source": "cpu_dry_run_validator",
            "source_class": "derived",
            "raw_samples_ms": [],
            "trials": [],
            "duration_ms": None,
            "analytic_ms": request.get("analytic_ms"),
            "query_context": request.get("query_context", {}),
            "observed_context": {"observed": False, "not_a3_measured": True},
            "environment": {"not_a3_measured": True, "dry_run": True},
            "protocol": _request_protocol(request),
            "correctness": {"status": "not_executed_cpu_dry_run"},
            "details": {"dry_run": True, "validated_recipe": validation.get("recipe"),
                        "shape": validation.get("shape")},
            "work": request.get("work", {}),
        }
        results.append(result)
    return {
        "manifest": str(manifest.get("campaign_id", "")),
        "run_id": run_id,
        "status": "dry_run",
        "not_a3_measured": True,
        "protocol": manifest.get("protocol", {}),
        "environment": {"not_a3_measured": True, "dry_run": True},
        "results": results,
        "summary": _result_summary(results),
    }


def _result_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "attempted": len(results),
        "completed": counts.get("completed", 0),
        "pending": counts.get("pending", 0) + counts.get("pending_backend", 0),
        "failed": counts.get("failed", 0),
        "dry_run_validated": counts.get("dry_run_validated", 0),
        "status_counts": dict(sorted(counts.items())),
        "measured_samples": sum(
            len(result.get("raw_samples_ms", ())) for result in results
            if result.get("status") == "completed"
        ),
    }


def run_manifest(manifest_path: str | Path, results_path: str | Path, *,
                 runner: Any = None, dry_run: bool = False, run_id: str = "",
                 warmup: int | None = None, iterations: int | None = None,
                 trials: int | None = None, max_cases: int | None = None) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    if dry_run:
        payload = _dry_run_results(manifest, run_id=run_id)
    elif runner is None:
        raise A3InputError("worker_real_run_requires_device_runner")
    else:
        results = []
        requests = list(manifest.get("requests", ()))
        if max_cases is not None:
            requests = requests[:max_cases]
        for request in requests:
            base = {
                "measurement_id": request.get("measurement_id"),
                "request_key": request.get("request_key"),
                "condition_key": request.get("condition_key"),
                "condition_keys": request.get("lineage_condition_keys", [request.get("condition_key")]),
                "split": request.get("split"),
                "execution_kind": request.get("execution_kind"),
                "worker_recipe": request.get("worker_recipe"),
                "backend": request.get("backend"),
                "target_match": request.get("target_match"),
                "query_context": request.get("query_context", {}),
                "protocol": _request_protocol(request),
                "work": request.get("work", {}),
            }
            try:
                result = runner.run(request)
                result["run_id"] = run_id or str(result.get("run_id") or request.get("run_id") or "")
                results.append({**base, **result})
            except PendingBackend as exc:
                results.append({
                    **base, "status": "pending", "reason": exc.reason,
                    "required_backend": exc.required_backend,
                    "source_class": "not_measured", "raw_samples_ms": [],
                    "observed_context": {"observed": False, "not_a3_measured": True},
                })
            except Exception as exc:  # noqa: BLE001 - runner boundary
                results.append({
                    **base, "status": "failed",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "source_class": "not_measured", "raw_samples_ms": [],
                    "observed_context": {"observed": False, "not_a3_measured": True},
                })
        payload = {
                "manifest": str(manifest_path),
            "run_id": run_id,
            "status": "completed" if all(r.get("status") == "completed" for r in results) else "partial",
            "not_a3_measured": False,
            "protocol": manifest.get("protocol", {}),
            "environment": getattr(runner, "environment", {}),
            "results": results,
            "summary": _result_summary(results),
        }
    temporary = Path(results_path).expanduser().with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(Path(results_path).expanduser())
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A3 operator campaign worker (device injected by coordinator)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=os.environ.get("MINDIE_OPERATOR_DEVICE", "npu:0"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--run-id", default=os.environ.get("MINDIE_RUN_ID", ""))
    args = parser.parse_args(argv)
    if args.dry_run:
        payload = run_manifest(args.manifest, args.results, dry_run=True, run_id=args.run_id)
    else:
        runner = TorchNpuOperatorRunner(
            device=args.device, warmup=args.warmup, iterations=args.iterations,
            trials=args.trials,
        )
        payload = run_manifest(
            args.manifest, args.results, runner=runner, run_id=args.run_id,
            max_cases=args.max_cases,
        )
    sys.stdout.write(json.dumps(payload["summary"], indent=1, sort_keys=True) + "\n")
    return 1 if payload["summary"].get("failed") else 0


__all__ = [
    "SUPPORTED_RECIPES", "PendingBackend",
    "TorchNpuOperatorRunner", "run_manifest", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
