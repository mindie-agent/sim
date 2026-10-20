"""Executable measurement runners and standalone manifest ingestion.

[Role]
  Consume manifest requests and produce raw timing repeats.  A managed
  ``torch_npu`` runner executes small GEMM/add/copy probes on the device the
  coordinator injected; a CPU runner exercises the local closed loop.  A
  standalone manifest runner writes results JSON without needing the local DB.

[Invariants]
  * CPU samples are ``derived`` and never claim A3 measured evidence.
  * The NPU runner preserves every event-timed repeat, protocol and environment;
    it does not collapse them into one fabricated sample.
  * int8 uses integer inputs, not ``randn``; buffers/outputs are preallocated.
  * Failures are returned as failed results and mark requests failed on ingest.
  * No runner selects a free device, creates a lease, SSHes or starts a service.

[Timeouts & Locks]
  Bounded warmup/iteration loops.  SQLite uses WAL + 5 s busy timeout.  No lock
  is held across an external run.

[Side-Effect Policy]
  Writes the explicitly supplied DB/results path only when invoked.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .a3_calibration_store import SQLiteCalibrationStore
from .a3_types import A3InputError


class MeasurementRunner(Protocol):
    def run(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class CpuMatmulRunner:
    """Small CPU reference loop for manifest/ingestion verification only."""

    def __init__(self, *, iterations: int = 5) -> None:
        self.iterations = max(1, int(iterations))

    def run(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if request.get("op_family") not in {"gemm", "qkv_projection", "moe_gate_up", "moe_down", "logits"}:
            raise A3InputError(f"cpu_runner_unsupported_op:{request.get('op_family')}")
        shape = request.get("shape", {})
        m = max(1, int(shape.get("m", shape.get("q", 1))))
        n = max(1, int(shape.get("n", shape.get("heads", 1))))
        k = max(1, int(shape.get("k", shape.get("dim", 1))))
        if m * n * k > 2_000_000:
            raise A3InputError("cpu_runner_shape_too_large_for_closed_loop")
        a = [[1.0] * k for _ in range(m)]
        b = [[1.0] * n for _ in range(k)]
        raw = []
        for _ in range(self.iterations):
            started = time.perf_counter()
            out = [[sum(a[i][j] * b[j][l] for j in range(k)) for l in range(n)] for i in range(m)]
            raw.append(max(1e-6, (time.perf_counter() - started) * 1000.0))
            if len(out) != m:
                raise A3InputError("cpu_runner_result_invalid")
        return {
            "status": "completed",
            "raw_samples_ms": raw,
            "duration_ms": statistics.median(raw),
            "analytic_ms": request.get("analytic_ms") or request.get("predicted_ms") or max(1e-6, statistics.median(raw)),
            "source": "cpu_reference_runner",
            "source_class": "derived",
            "run_id": request.get("run_id", ""),
            "split": "train",
            "timing_scope": "python_cpu_reference_only",
            "protocol": {"iterations": self.iterations},
            "environment": {"not_a3_measured": True},
            "details": {"not_a3_measured": True},
        }


class TorchNpuRunner:
    """Managed-device operator runner for GEMM/add/copy shape probes."""

    def __init__(self, *, device: str = "npu:0", warmup: int = 3, iterations: int = 10) -> None:
        try:
            import torch  # type: ignore
            import torch_npu  # type: ignore  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - optional runtime boundary
            raise A3InputError(f"torch_npu_unavailable:{type(exc).__name__}") from exc
        self.torch = torch
        self.device = device
        self.warmup = max(1, int(warmup))
        self.iterations = max(1, int(iterations))

    def _dtype(self, name: str) -> Any:
        key = str(name or "bf16").lower()
        mapping = {
            "bf16": self.torch.bfloat16, "bfloat16": self.torch.bfloat16,
            "fp16": self.torch.float16, "float16": self.torch.float16,
            "fp32": self.torch.float32, "float32": self.torch.float32,
            "int8": self.torch.int8,
        }
        if key not in mapping:
            raise A3InputError(f"torch_npu_dtype_unsupported:{name}")
        return mapping[key]

    def _inputs(self, request: Mapping[str, Any]) -> tuple[Any, Any, Any, str]:
        torch = self.torch
        op = str(request.get("op_family", ""))
        shape = request.get("shape", {})
        dtype = self._dtype(str(request.get("dtype", "bf16")))
        if op == "copy":
            elements = max(1, int(shape.get("elements", 1)))
            source = torch.zeros(elements, dtype=dtype, device=self.device)
            target = torch.empty_like(source)
            return source, target, None, "copy"
        if op == "add":
            elements = max(1, int(shape.get("elements", 1)))
            a = torch.zeros(elements, dtype=dtype, device=self.device)
            b = torch.zeros(elements, dtype=dtype, device=self.device)
            out = torch.empty_like(a)
            return a, b, out, "add"
        if op not in {
            "gemm", "qkv_projection", "o_projection", "gate_up_projection",
            "down_projection", "mla_q_a", "mla_kv_a", "mla_q_b", "mla_kv_b",
            "moe_gate_up", "moe_down", "shared_expert_gate_up",
            "shared_expert_down", "logits", "indexer_projection",
            "linear_q_projection", "linear_k_projection", "linear_v_projection",
            "linear_o_projection",
        }:
            raise A3InputError(f"torch_npu_runner_unsupported_op:{op}")
        m = max(1, int(shape.get("m", shape.get("q", 1))))
        n = max(1, int(shape.get("n", shape.get("heads", 1))))
        k = max(1, int(shape.get("k", shape.get("dim", 1))))
        if dtype == torch.int8:
            a = torch.randint(-8, 8, (m, k), dtype=dtype, device=self.device)
            b = torch.randint(-8, 8, (k, n), dtype=dtype, device=self.device)
            out = torch.empty((m, n), dtype=torch.int32, device=self.device)
            return a, b, out, "int8_mm"
        a = torch.randn((m, k), dtype=dtype, device=self.device)
        b = torch.randn((k, n), dtype=dtype, device=self.device)
        out = torch.empty((m, n), dtype=dtype, device=self.device)
        return a, b, out, "mm"

    def _call(self, a: Any, b: Any, out: Any, kind: str) -> None:
        torch = self.torch
        if kind == "copy":
            b.copy_(a)
        elif kind == "add":
            torch.add(a, b, out=out)
        elif kind == "int8_mm":
            torch._int_mm(a, b, out=out)  # type: ignore[attr-defined]
        else:
            torch.mm(a, b, out=out)

    def run(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        features = request.get("features") or {}
        weight_dtype = str(features.get("weight_dtype", request.get("dtype", "bf16"))).lower()
        quantization = str(features.get("quantization", "none")).lower()
        layout = str(request.get("layout", "contiguous")).lower()
        if layout not in {"", "contiguous"}:
            raise A3InputError(f"torch_npu_runner_layout_unsupported:{layout}")
        if quantization not in {"", "none"} or weight_dtype not in {"bf16", "fp16"}:
            raise A3InputError(
                f"torch_npu_runner_quantized_weight_unsupported:{weight_dtype}:{quantization}"
            )
        a, b, out, kind = self._inputs(request)
        for _ in range(self.warmup):
            self._call(a, b, out, kind)
        self.torch.npu.synchronize()
        raw: list[float] = []
        for _ in range(self.iterations):
            start = self.torch.npu.Event(enable_timing=True)
            end = self.torch.npu.Event(enable_timing=True)
            start.record()
            self._call(a, b, out, kind)
            end.record()
            self.torch.npu.synchronize()
            duration = float(start.elapsed_time(end))
            if duration <= 0:
                raise A3InputError("torch_npu_duration_invalid")
            raw.append(duration)
        ordered = sorted(raw)
        return {
            "status": "completed",
            "raw_samples_ms": raw,
            "duration_ms": statistics.median(raw),
            "analytic_ms": request.get("analytic_ms") or request.get("predicted_ms") or statistics.median(raw),
            "source": "torch_npu_managed_runner",
            "source_class": "measured",
            "run_id": request.get("run_id", ""),
            "revision": os.environ.get("MINDIE_SIM_RUNNER_REVISION", ""),
            "split": "train",
            "timing_scope": "device_event_warm",
            "protocol": {
                "measurement_scope": "event-timed warm repeated operator calls; includes launch/dispatch gaps",
                "warmup": self.warmup, "iterations": self.iterations,
                "preallocated": True, "timer": "device_event",
                "layout": layout, "weight_dtype": weight_dtype,
                "requested_compute_dtype": features.get("compute_dtype", ""),
                "attested_accumulation_dtype": "backend_default",
                "quantization": quantization,
            },
            "environment": {
                "device": self.device,
                "device_name": self.torch.npu.get_device_name(0) if hasattr(self.torch.npu, "get_device_name") else "",
                "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
            },
            "details": {
                "op_family": request.get("op_family"), "kind": kind,
                "min_ms": ordered[0], "max_ms": ordered[-1],
                "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
                "std_ms": statistics.pstdev(raw) if len(raw) > 1 else 0.0,
            },
        }


def record_prediction_misses(
    store: SQLiteCalibrationStore,
    prediction: Mapping[str, Any],
    *,
    hardware_key: str = "",
    run_id: str = "",
    max_requests: int = 256,
) -> dict[str, Any]:
    """Opt-in conversion of uncalibrated prediction ops into pending requests."""
    recorded = 0
    skipped = 0
    seen: set[tuple[Any, ...]] = set()
    for trace in prediction.get("traces", ()):
        for op in trace.get("ops", ()):
            if op.get("calibration_status") in {"calibrated_exact", "calibrated_interpolated"}:
                skipped += 1
                continue
            if recorded >= max_requests:
                skipped += 1
                continue
            curve_result = op.get("curve_result")
            if not isinstance(curve_result, Mapping):
                curve_result = {}
            missing = curve_result.get("missing_condition")
            transition = (
                str(curve_result.get("status") or "") == "transition_unresolved"
                and isinstance(missing, Mapping)
                and bool(missing.get("shape"))
            )
            protocol: Mapping[str, Any] = {}
            details: dict[str, Any] = {}
            op_family = str(op.get("family", "unknown"))
            recipe = str(op.get("recipe", ""))
            dtype = str(op.get("dtype", ""))
            layout = str(op.get("layout", "contiguous"))
            request_hardware = hardware_key
            features = (
                dict(op["features"])
                if isinstance(op.get("features"), Mapping) else {}
            )
            if transition:
                shape_map = {
                    str(key): float(value)
                    for key, value in dict(missing.get("shape") or {}).items()
                }
                op_family = str(missing.get("family") or op_family)
                recipe = str(missing.get("recipe") or recipe)
                precision = dict(missing.get("precision") or {})
                dtype = str(precision.get("input_dtype") or dtype)
                layout = str(missing.get("layout") or layout)
                request_hardware = str(
                    missing.get("hardware_key") or request_hardware
                )
                features = {**features, **precision}
                if isinstance(missing.get("protocol"), Mapping):
                    protocol = dict(missing["protocol"])
                details = {
                    "missing_condition": dict(missing),
                    "curve_transition": {
                        "curve_id": curve_result.get("matched_curve"),
                        "condition_key": curve_result.get("condition_key"),
                        "axis": curve_result.get("axis") or missing.get("axis"),
                        "value": curve_result.get("x") or missing.get("value"),
                        "lower_bound_ms": curve_result.get("lower_bound_ms"),
                        "upper_bound_ms": curve_result.get("upper_bound_ms"),
                        "uncertainty_ms": curve_result.get("uncertainty_ms"),
                        "reason": curve_result.get("reason"),
                        "protocol": dict(protocol),
                        "transfer_scope": curve_result.get("transfer_scope"),
                    },
                }
            else:
                shape_map = op.get("shape_features") or {}
                if not shape_map:
                    shape = op.get("launch_shape") or op.get("padded_shape") or []
                    if op.get("family") in {"attention_qk", "attention_pv"} and len(shape) >= 4:
                        shape_map = {"q": shape[0], "kv": shape[1], "heads": shape[2], "dim": shape[3]}
                    elif len(shape) >= 3:
                        shape_map = {"m": shape[-3], "n": shape[-2], "k": shape[-1]}
                    elif len(shape) == 2:
                        shape_map = {"elements": shape[0] * shape[1], "dim": shape[1]}
                    else:
                        shape_map = {"elements": shape[0] if shape else 1}
            dedupe_key = (
                op_family, op.get("phase"), recipe,
                tuple(sorted((str(k), float(v)) for k, v in shape_map.items())),
                dtype, layout,
                op.get("calibration_status"),
                tuple(sorted((str(k), str(v)) for k, v in protocol.items())),
            )
            if dedupe_key in seen:
                skipped += 1
                continue
            seen.add(dedupe_key)
            extrapolation = (
                1.0 if op.get("calibration_status") == "calibrated_extrapolated"
                else 0.0
            )
            duration = float(op.get("estimated_ms") or 0.0)
            reason = (
                "curve_transition_unresolved:"
                f"{curve_result.get('condition_key') or 'unknown'}"
                if transition else
                f"predictive_miss:{op.get('calibration_status', 'unknown')}"
            )
            store.record_measurement_request(
                op_family=op_family,
                phase=str(op.get("phase", "")),
                recipe=recipe,
                shape=shape_map,
                dtype=dtype,
                layout=layout,
                hardware_key=request_hardware,
                features=features or None,
                competition=(
                    op.get("competition")
                    if isinstance(op.get("competition"), Mapping) else None
                ),
                predicted_ms=duration if duration > 0 else None,
                analytic_ms=float(op.get("analytic_ms") or 0.0) or None,
                extrapolation=extrapolation,
                target_impact=min(
                    1.0,
                    duration / max(1.0, float(trace.get("duration_ms", 1.0))),
                ),
                reason=reason,
                run_id=run_id,
                details=details or None,
                protocol=protocol or None,
                timing_scope=str(protocol.get("timing_scope") or ""),
            )
            recorded += 1
    return {"recorded_requests": recorded, "skipped_calibrated": skipped}




def run_pending(
    store: SQLiteCalibrationStore,
    runner: MeasurementRunner,
    *,
    limit: int = 8,
    run_id: str = "",
) -> dict[str, Any]:
    pending = store.requests(status="pending", limit=limit)
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for request in pending:
        key = str(request["request_key"])
        effective_run = run_id or str(request.get("run_id", ""))
        store.update_request_status(key, status="running", run_id=effective_run)
        try:
            result = runner.run(request)
            analytic = result.get("analytic_ms", request.get("predicted_ms"))
            if analytic is None:
                raise A3InputError("analytic_ms_required")
            raw = result.get("raw_samples_ms")
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
                raw = [float(result["duration_ms"])]
            request_details = (
                dict(request["details"])
                if isinstance(request.get("details"), Mapping) else {}
            )
            details = {
                "case_id": key,
                **request_details,
                "features": {
                    **(request.get("features") or {}),
                    **(result.get("features") or {}),
                },
                "details": result.get("details", {}),
            }
            for field in ("timing_scope", "protocol", "environment"):
                value = result.get(field)
                if value is None or value == "" or value == {}:
                    value = request_details.get(field)
                if value is not None:
                    details[field] = value
            completed.append(store.complete_request_samples(
                key, raw_samples_ms=[float(value) for value in raw],
                analytic_ms=float(analytic),
                source=str(result.get("source", "runner")),
                source_class=str(result.get("source_class", "derived")),
                run_id=str(result.get("run_id") or effective_run),
                revision=str(result.get("revision", "")),
                split=str(result.get("split", "train")),
                features=result.get("features") if isinstance(result.get("features"), Mapping) else None,
                details=details,
            ))
        except Exception as exc:  # noqa: BLE001 - runner boundary is external
            reason = getattr(exc, "reason", f"{type(exc).__name__}:{exc}")
            failed.append(store.fail_request(key, reason=str(reason), run_id=effective_run))
    return {
        "status": "run_completed",
        "attempted": len(pending),
        "completed": completed,
        "failed": failed,
        "remaining_pending": len(store.requests(status="pending", limit=1_000_000)),
    }


def run_manifest_file(
    manifest_path: str | Path,
    results_path: str | Path,
    runner: MeasurementRunner,
    *,
    run_id: str = "",
) -> dict[str, Any]:
    """Run a manifest without a local DB and write ingestion-ready results."""
    manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    requests = manifest.get("requests", ())
    results = []
    for request in requests:
        effective_run = run_id or str(request.get("run_id", ""))
        try:
            result = runner.run(request)
            raw = result.get("raw_samples_ms")
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
                raise A3InputError("raw_samples_missing")
            request_details = (
                dict(request["details"])
                if isinstance(request.get("details"), Mapping) else {}
            )
            details = {
                **request_details,
                **(
                    dict(result["details"])
                    if isinstance(result.get("details"), Mapping) else {}
                ),
            }
            for field in ("timing_scope", "protocol", "environment"):
                value = result.get(field)
                if value is None or value == "" or value == {}:
                    value = (
                        request.get(field)
                        if request.get(field) not in (None, "", {})
                        else request_details.get(field)
                    )
                if value is not None:
                    details[field] = value
            results.append({
                **request,
                "status": "completed",
                "raw_samples_ms": list(raw),
                "duration_ms": float(result.get("duration_ms", statistics.median(raw))),
                "analytic_ms": float(result.get("analytic_ms") or request.get("analytic_ms") or request.get("predicted_ms") or statistics.median(raw)),
                "source": result.get("source", "runner"),
                "source_class": result.get("source_class", "derived"),
                "run_id": str(result.get("run_id") or effective_run),
                "revision": result.get("revision", ""),
                "split": result.get("split", "train"),
                "timing_scope": details.get("timing_scope", ""),
                "protocol": details.get("protocol", {}),
                "environment": details.get("environment", {}),
                "features": {
                    **(request.get("features") or {}),
                    **(result.get("features") or {}),
                },
                "details": details,
            })
        except Exception as exc:  # noqa: BLE001 - runner boundary is external
            results.append({
                **request,
                "status": "failed",
                "reason": getattr(exc, "reason", f"{type(exc).__name__}:{exc}"),
                "run_id": effective_run,
            })
    payload = {
        "manifest": str(manifest_path),
        "run_id": run_id,
        "results": results,
    }
    Path(results_path).expanduser().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"status": "written", "results": len(results),
            "completed": sum(1 for item in results if item["status"] == "completed"),
            "failed": sum(1 for item in results if item["status"] == "failed"),
            "path": str(results_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone manifest runner (no local DB required)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--runner", choices=("torch-npu", "cpu"), default="torch-npu")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args(argv)
    runner = TorchNpuRunner(device=args.device, warmup=args.warmup, iterations=args.iterations) \
        if args.runner == "torch-npu" else CpuMatmulRunner(iterations=args.iterations)
    payload = run_manifest_file(args.manifest, args.results, runner, run_id=args.run_id)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 1 if payload.get("failed", 0) else 0


__all__ = [
    "MeasurementRunner", "CpuMatmulRunner", "TorchNpuRunner", "run_pending",
    "run_manifest_file", "record_prediction_misses", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
