"""Import and summarize real A3 operator/CV measurements.

[Role]
  Convert the training measurement JSON produced by the managed A3 worker into
  raw repeats in ``SQLiteCalibrationStore``, preserving case, run, protocol,
  environment, split and details.  Build an analytic baseline with the same
  predictive cost model so calibration residuals are reproducible.  Also
  summarize paired C/V measurements into an explicit CV concurrency predictor.

[Invariants]
  * Every raw repeat is stored; re-importing the same run/case/repeat is
    idempotent.  Different runs at the same condition are not discarded.
  * Training is imported by default; heldout is skipped unless explicitly
    requested and is never used by ``as_calibration_store`` by default.
  * Measured samples are labelled by the JSON's real run protocol, not
    synthesized; logical bytes are not presented as measured HBM traffic.
  * The analytic baseline uses the same machine fixture and cost formulas as
    predictive modeling.
  * CV summaries are measured slowdowns under the declared stream overlap, not
    a universal contention discount.

[Timeouts & Locks]
  Pure JSON/SQLite ingestion with bounded loops.  No NPU/SSH/subprocess calls.

[Side-Effect Policy]
  Writes only the explicitly supplied SQLite database when importing.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .a3_calibration_store import SQLiteCalibrationStore
from .a3_cost import CalibrationStore, PredictiveCostModel
from .a3_resources import ResourceEngine, ResourceSpec
from .a3_types import A3InputError, MachineSpec
from .a3_workload import OpSpec, dtype_bytes


def machine_from_environment(environment: Mapping[str, Any]) -> MachineSpec:
    """Build a one-die fixture from the measurement environment."""
    device_name = str(environment.get("device_name", "Ascend910_9362"))
    common = str(environment.get("clock_common_before", ""))
    aic_match = re.search(r"Aicore Count\s*:\s*(\d+)", common)
    freq_match = re.search(r"Aicore Freq\(MHZ\)\s*:\s*(\d+)", common)
    aic = int(aic_match.group(1)) if aic_match else 20
    freq_mhz = float(freq_match.group(1)) if freq_match else 1650.0
    # Ascend910_9362 published HBM peak in the SKU catalog; this is a declared
    # hardware fixture, not inferred from the measured durations.
    hbm_bw = 1200e9
    cube = float(aic) * 8192.0 * freq_mhz * 1e6
    vector = float(aic * 2) * 256.0 * freq_mhz * 1e6
    return MachineSpec.from_mapping({
        "name": f"real-measurement-{device_name}",
        "hardware_key": device_name,
        "arch_family": "A3",
        "dies_per_package": 2,
        "num_devices": 1,
        "device": {
            "hbm_gib": 64.0,
            "cube_flops_per_s": cube,
            "vector_ops_per_s": vector,
            "hbm_bw_bytes_per_s": hbm_bw,
            "mte_bw_bytes_per_s": 2.0 * hbm_bw,
            "launch_us": 20.0,
            "l2_bytes": 201326592,
        },
        "compute_peaks": {"bf16": cube},
        "links": [{"name": "d2d", "capacity_gbps": 196, "members": [0]}],
    })


def _baseline_op(
    family: str, *, flops: float = 0.0, hbm_bytes: float = 0.0,
    vector_ops: float = 0.0, shape: Sequence[int], machine: MachineSpec,
    dtype: str = "bf16",
) -> float:
    token = max(1, int(shape[0]) if shape else 1)
    op = OpSpec(
        op_id=f"baseline.{family}", family=family, phase="operator",
        stage=0, layer=0, requests=(), semantic_shape=tuple(int(v) for v in shape),
        local_shape=tuple(int(v) for v in shape),
        padded_shape=tuple(int(v) for v in shape),
        launch_shape=tuple(int(v) for v in shape),
        flops=float(flops), hbm_bytes=float(hbm_bytes),
        mte2_bytes=float(hbm_bytes) / 2.0, mte3_bytes=float(hbm_bytes) / 2.0,
        vector_ops=float(vector_ops), recipe="baseline",
        dtype=dtype, layout="contiguous",
    )
    return PredictiveCostModel(
        machine, calibration=CalibrationStore(), compute_dtype=dtype,
        weight_dtype=dtype, activation_dtype=dtype,
    ).cost(op).analytic_ms


def baseline_gemm(
    machine: MachineSpec, m: int, n: int, k: int, *, dtype: str = "bf16",
) -> float:
    b = dtype_bytes(dtype)
    return _baseline_op(
        "gemm", flops=2.0 * m * n * k,
        hbm_bytes=(m * k + k * n + m * n) * b,
        shape=(m, n, k), machine=machine, dtype=dtype,
    )


def baseline_add(
    machine: MachineSpec, elements: int, *, dtype: str = "bf16",
) -> float:
    b = dtype_bytes(dtype)
    return _baseline_op(
        "add", hbm_bytes=3.0 * elements * b, vector_ops=float(elements),
        shape=(elements,), machine=machine, dtype=dtype,
    )


def baseline_copy(
    machine: MachineSpec, elements: int, *, dtype: str = "bf16",
) -> float:
    b = dtype_bytes(dtype)
    return _baseline_op(
        "copy", hbm_bytes=2.0 * elements * b, vector_ops=float(elements),
        shape=(elements,), machine=machine, dtype=dtype,
    )


def import_a3_measurements(
    store: SQLiteCalibrationStore,
    path: str | Path,
    *,
    include_heldout: bool = False,
) -> dict[str, Any]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("cases"), Sequence):
        raise A3InputError("a3_measurements_invalid")
    protocol = dict(data.get("protocol") or {})
    environment = dict(data.get("environment") or {})
    machine = machine_from_environment(environment)
    hardware_key = machine.hardware_key
    run_id = f"a3-{data.get('started_unix_s', 'unknown')}"
    imported = 0
    skipped = 0
    cases = 0
    by_kind: dict[str, int] = {}
    for case in data["cases"]:
        if not isinstance(case, Mapping) or case.get("status") != "completed":
            skipped += 1
            continue
        split = str(case.get("split", "train")).strip().lower()
        split = {"holdout": "heldout", "held_out": "heldout", "validation": "heldout"}.get(split, split)
        if split == "heldout" and not include_heldout:
            skipped += 1
            continue
        kind = str(case.get("kind", "unknown"))
        case_id = str(case.get("case_id", f"case-{cases}"))
        declared_dtype = str(
            case.get("dtype") or protocol.get("dtype") or "bf16"
        ).lower().replace("-", "_")
        dtype = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}.get(
            declared_dtype, declared_dtype,
        )
        common = {
            "dtype": dtype, "layout": str(case.get("layout") or "contiguous"),
            "hardware_key": hardware_key,
            "source": "a3_training_measurements",
            "source_class": "measured",
            "run_id": run_id, "split": split,
            "timing_scope": "device_event_warm",
            "protocol": protocol, "environment": environment,
        }
        common_features = {
            "weight_dtype": dtype, "activation_dtype": dtype,
            "compute_dtype": dtype, "accumulation_dtype": "backend_default",
            "quantization": "none",
        }
        if kind == "matmul":
            m, k, n = (int(v) for v in case.get("shape_mkn", (0, 0, 0)))
            analytic = baseline_gemm(machine, m, n, k, dtype=dtype)
            raw = case.get("raw_device_ms_per_op") or [
                sample.get("device_ms", 0.0) for sample in case.get("samples", ())
            ]
            for index, duration in enumerate(raw):
                store.record_sample(
                    op_family="gemm", phase="operator", recipe="torch_mm_out",
                    shape={"m": m, "n": n, "k": k}, duration_ms=float(duration),
                    analytic_ms=analytic, case_id=case_id, repeat_index=index,
                    features=common_features,
                    details={
                        "case_id": case_id, "raw_case": "matmul",
                        "call_path": case.get("call_path"),
                        "flops": case.get("flops"),
                        "logical_bytes_per_op": case.get("logical_bytes_per_op"),
                        "checksum": case.get("checksum"),
                        "timing_scope": protocol.get("measurement_scope"),
                    },
                    **common,
                )
                imported += 1
            cases += 1
            by_kind[kind] = by_kind.get(kind, 0) + len(raw)
        elif kind in {"add", "copy"}:
            shape = case.get("shape", ())
            elements = int(
                case.get("shape_n", case.get("n", case.get("elements", 0)))
                or (shape[0] if isinstance(shape, Sequence) and shape else 0)
            )
            if elements <= 0:
                skipped += 1
                continue
            analytic = (
                baseline_add(machine, elements, dtype=dtype) if kind == "add"
                else baseline_copy(machine, elements, dtype=dtype)
            )
            raw = case.get("raw_device_ms_per_op") or [
                sample.get("device_ms", 0.0) for sample in case.get("samples", ())
            ]
            for index, duration in enumerate(raw):
                store.record_sample(
                    op_family=kind, phase="operator", recipe=f"torch_{kind}",
                    shape={"elements": elements}, duration_ms=float(duration),
                    analytic_ms=analytic, case_id=case_id, repeat_index=index,
                    features=common_features,
                    details={
                        "case_id": case_id, "raw_case": kind,
                        "logical_bytes_per_op": case.get("logical_bytes_per_op"),
                        "checksum": case.get("checksum"),
                        "timing_scope": protocol.get("measurement_scope"),
                    },
                    **common,
                )
                imported += 1
            cases += 1
            by_kind[kind] = by_kind.get(kind, 0) + len(raw)
        elif kind == "paired_streams":
            mode = str(case.get("mode", "unknown"))
            c_shape = tuple(int(v) for v in case.get("c_shape_mkn", (0, 0, 0)))
            v_shape = tuple(int(v) for v in case.get("v_shape", (0,)))
            c_base = baseline_gemm(machine, c_shape[0], c_shape[2], c_shape[1]) if len(c_shape) == 3 else 0.0
            v_base = baseline_add(machine, v_shape[0]) if v_shape else 0.0
            c_repeats = max(1, int(case.get("c_repeats", 1)))
            v_repeats = max(1, int(case.get("v_repeats", 1)))
            paired_imported = 0
            for index, sample in enumerate(case.get("samples", ())):
                # Event ranges cover the repeated kernel loop; normalize to one
                # operator before comparing with the single-op analytic baseline.
                c_duration = float(sample.get("c_duration_ms", 0.0)) / c_repeats
                v_duration = float(sample.get("v_duration_ms", 0.0)) / v_repeats
                span = float(sample.get("active_device_span_ms", 0.0)) / max(c_repeats, v_repeats)
                competition = {
                    "mode": mode, "v_elements": v_shape[0] if v_shape else 0,
                    "requested_v_delay_ms": float(case.get("requested_v_delay_ms", 0.0)),
                    "c_repeats": c_repeats, "v_repeats": v_repeats,
                    "c_shape_mkn": list(c_shape),
                }
                streams = []
                if bool(sample.get("c_active", True)):
                    streams.append(("C", c_duration, c_base, {"m": c_shape[0], "n": c_shape[2], "k": c_shape[1]} if len(c_shape) == 3 else {"elements": 1}))
                if bool(sample.get("v_active", False)):
                    streams.append(("V", v_duration, v_base, {"elements": v_shape[0] if v_shape else 1}))
                streams.append(("span", span, max(c_base, v_base), {"c_elements": c_shape[0] * c_shape[1] * c_shape[2] if len(c_shape) == 3 else 1,
                                                                     "v_elements": v_shape[0] if v_shape else 1}))
                for stream, duration, analytic, shape in streams:
                    store.record_sample(
                        op_family=f"paired_{stream.lower()}", phase="concurrent",
                        recipe=f"two_streams_{mode}",
                        shape=shape, duration_ms=duration, analytic_ms=max(1e-12, analytic),
                        features={**common_features, "stream": stream, "mode": mode},
                        competition=competition,
                        case_id=f"{case_id}.{stream}", repeat_index=index,
                        details={
                            "case_id": case_id, "raw_case": "paired_streams",
                            "stream": stream, "mode": mode,
                            "sample": dict(sample),
                            "repeats": {"c": c_repeats, "v": v_repeats},
                            "timing_scope": protocol.get("concurrency_scope"),
                        },
                        **common,
                    )
                    imported += 1
                    paired_imported += 1
            cases += 1
            by_kind[kind] = by_kind.get(kind, 0) + paired_imported
        else:
            skipped += 1
    return {
        "status": "imported",
        "run_id": run_id,
        "hardware_key": hardware_key,
        "cases": cases,
        "measurements": imported,
        "skipped": skipped,
        "by_kind": by_kind,
        "include_heldout": include_heldout,
        "inventory": store.inventory(),
    }




__all__ = [
    "machine_from_environment", "baseline_gemm", "baseline_add", "baseline_copy",
    "import_linearity_calibration_cases",
    "import_a3_measurements", "CvCondition", "CvCalibration",
]


def _comm_baselines(placement: str, payload_bytes: int) -> tuple[float, float]:
    from .comm import default_transfer_channels, solve_collective_bound, solve_transfer_bound

    channels = default_transfer_channels("A3", dies_per_package=2)
    collective = solve_collective_bound(
        "all_reduce", float(payload_bytes), channels,
        total_ranks=2, ranks_per_node=2, dies_per_package=2,
    )
    channel = "d2d_intra_package" if "sio" in placement.lower() else "d2d_intra_node"
    transfer = solve_transfer_bound(channel, float(payload_bytes), channels)
    return (
        float(collective.get("lower_bound_us", 0.0) or 0.0) / 1000.0,
        float(transfer.get("lower_bound_us", 0.0) or 0.0) / 1000.0,
    )


def import_communication_measurements(
    store: SQLiteCalibrationStore,
    path: str | Path,
    *,
    include_heldout: bool = False,
) -> dict[str, Any]:
    """Import isolated/concurrent HCCL+P2P training cases with raw repeats."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise A3InputError("communication_measurements_invalid")
    runs = data.get("runs")
    if runs is None and isinstance(data.get("ranks"), Sequence):
        runs = [data]
    if not isinstance(runs, Sequence):
        raise A3InputError("communication_measurements_invalid")
    pending: list[dict[str, Any]] = []
    condition_keys: set[tuple[str, str]] = set()
    skipped = 0
    for run in runs:
        if not isinstance(run, Mapping) or run.get("status") != "completed":
            skipped += 1
            continue
        run_id = str(run.get("run_id") or f"comm-{run.get('started_unix_s', 'unknown')}")
        run_protocol = {
            key: run.get(key) for key in (
                "status", "source_script", "visible_devices",
                "topology", "child_exit_codes", "elapsed_seconds", "started_unix_s",
                "finished_unix_s",
            ) if run.get(key) is not None
        }
        for rank_block in run.get("ranks", ()):
            rank = int(rank_block.get("rank", 0) or 0)
            rank_environment = dict(rank_block.get("environment") or {})
            rank_protocol = dict(rank_block.get("protocol") or {})
            for case in rank_block.get("cases", ()):
                split = str(case.get("split", "train")).strip().lower()
                split = {"holdout": "heldout", "held_out": "heldout", "validation": "heldout"}.get(split, split)
                if split == "heldout" and not include_heldout:
                    skipped += 1
                    continue
                case_id = str(case.get("case_id", "comm-case"))
                placement = "sio" if "sio" in case_id.lower() else "hccssw"
                payload = int(case.get("tensor_payload_bytes", 0))
                if payload <= 0:
                    skipped += 1
                    continue
                condition_keys.add((run_id, case_id))
                repetitions = max(1, int(case.get("repetitions_per_lane", 1)))
                delay = float(case.get("requested_transfer_delay_ms", 0.0))
                c_base, p_base = _comm_baselines(placement, payload)
                mode = str(case.get("mode", "unknown"))
                features = {"placement": placement, "mode": mode}
                competition = {
                    "requested_transfer_delay_ms": delay,
                    "repetitions_per_lane": repetitions,
                    "concurrent": "plus" in mode,
                }
                for index, sample in enumerate(case.get("samples", ())):
                    lanes = []
                    if mode in {"all_reduce_only", "all_reduce_plus_p2p"} and sample.get("collective_duration_ms") is not None:
                        lanes.append(("all_reduce", float(sample["collective_duration_ms"]) / repetitions,
                                      c_base, "collective"))
                    if mode in {"p2p_only", "all_reduce_plus_p2p"} and sample.get("transfer_duration_ms") is not None:
                        lanes.append(("p2p", float(sample["transfer_duration_ms"]) / repetitions,
                                      p_base, "transfer"))
                    lanes.append(("comm_span", float(sample.get("active_device_span_ms", 0.0)) / repetitions,
                                  max(c_base, p_base), "span"))
                    for family, duration, analytic, lane in lanes:
                        if duration <= 0:
                            continue
                        pending.append({
                            "run_id": run_id, "case_id": case_id, "index": index,
                            "rank": rank, "family": family, "duration": duration,
                            "analytic": max(analytic, 1e-9), "lane": lane,
                            "placement": placement, "mode": mode, "payload": payload,
                            "repetitions": repetitions, "delay": delay,
                            "features": features, "competition": competition,
                            "split": split, "sample": dict(sample),
                            "case": dict(case), "run_protocol": run_protocol,
                            "rank_protocol": rank_protocol,
                            "rank_environment": rank_environment,
                            "visible_devices": run.get("visible_devices", ""),
                        })
    # Rank rows from one collective trial are correlated.  Store one critical-
    # rank sample per (case, lane, repeat), retain every rank sample in details,
    # and do not manufacture independent repeat counts from rank count.
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for item in pending:
        groups.setdefault((
            item["run_id"], item["case_id"], item["lane"], item["index"],
        ), []).append(item)
    imported = 0
    for (run_id, case_id, lane, index), rows in sorted(groups.items()):
        best = max(rows, key=lambda item: item["duration"])
        ranks = sorted({int(item["rank"]) for item in rows})
        raw_rank_samples = [
            {
                "rank": item["rank"], "duration_ms": item["duration"],
                "analytic_ms": item["analytic"], "sample": item["sample"],
            }
            for item in sorted(rows, key=lambda item: item["rank"])
        ]
        protocol = {
            **best["run_protocol"], **best["rank_protocol"],
            "case_protocol": {
                key: best["case"].get(key) for key in (
                    "kind", "mode", "collective_call", "transfer_call",
                    "tensor_payload_bytes", "repetitions_per_lane",
                    "requested_transfer_delay_ms", "split", "status",
                ) if best["case"].get(key) is not None
            },
            "repetitions_per_lane": best["repetitions"],
            "mode": best["mode"],
            "rank_count": len(ranks),
            "rank_samples_correlated": True,
        }
        try:
            from .a3_types import machine_from_environment as _machine_from_env
            _env_machine = _machine_from_env(best["rank_environment"])
        except Exception:
            _env_machine = None
        _device_name = str(
            best["rank_environment"].get("device_name", "")
            or (_env_machine.hardware_key if _env_machine is not None else "")
            or "unspecified"
        )
        _dtype_decl = str(
            best["rank_protocol"].get("dtype", "bf16")
        ).lower()
        _dtype = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
            _dtype_decl, _dtype_decl,
        )
        _backend = str(
            best["case"].get("execution_backend")
            or best["case"].get("requested_expansion")
            or best["rank_protocol"].get("execution_backend")
            or best["rank_protocol"].get("requested_expansion")
            or best["run_protocol"].get("execution_backend")
            or "unspecified"
        )
        _observed = str(
            best["case"].get("observed_execution")
            or best["case"].get("actual_execution")
            or best["rank_protocol"].get("observed_execution")
            or "unknown"
        )
        _collective = str(
            best["case"].get("collective")
            or best["rank_protocol"].get("collective")
            or ""
        )
        _reduction = str(
            best["case"].get("reduction")
            or best["rank_protocol"].get("reduction")
            or ""
        )
        _stages = (
            best["case"].get("execution_stages")
            or best["case"].get("execution_path")
            or best["rank_protocol"].get("execution_stages")
            or best["rank_protocol"].get("execution_path")
            or ()
        )
        environment = {
            **best["rank_environment"], "placement": best["placement"],
            "rank": ranks[0] if len(ranks) == 1 else ",".join(str(rank) for rank in ranks),
            "ranks": ranks, "visible_devices": best["visible_devices"],
            "run_id": run_id,
            "case_id": case_id,
        }
        store.record_sample(
            op_family=best["family"], phase="communication",
            recipe=f"two_rank_{best['mode']}_{best['placement']}",
            shape={"bytes": best["payload"], "ranks": len(ranks)},
            duration_ms=best["duration"], analytic_ms=best["analytic"],
            hardware_key=f"{_device_name}:{best['placement']}",
            dtype="", layout="lane",
            features={**best["features"], "ranks": len(ranks),
                      "dtype": _dtype, "algorithm": best["mode"],
                      "execution_backend": _backend,
                      "observed_execution": _observed,
                      "collective": _collective,
                      "reduction": _reduction,
                      "group_size": len(ranks),
                      "execution_stages": tuple(
                          best.get("execution_stages") or _stages
                      )},
            competition=best["competition"],
            source="a3_communication_training", source_class="measured",
            run_id=run_id, case_id=f"{case_id}:{lane}",
            repeat_index=index, split=best["split"],
            timing_scope="communication_lane",
            protocol=protocol, environment=environment,
            details={
                "case_id": case_id, "lane": lane,
                "raw_duration_ms": float(best["sample"].get(
                    "collective_duration_ms" if lane == "collective" else
                    "transfer_duration_ms" if lane == "transfer" else
                    "active_device_span_ms", 0.0,
                )),
                "sample": dict(best["sample"]),
                "rank_samples": raw_rank_samples,
                "rank_samples_correlated": True,
                "collective_call": best["case"].get("collective_call"),
                "transfer_call": best["case"].get("transfer_call"),
                "requested_release_ms": best["delay"],
                "timing_scope_detail": best["rank_protocol"].get("timing_scope"),
                "concurrency_scope": best["rank_protocol"].get("concurrency_scope"),
                "preallocated": best["rank_protocol"].get("preallocated"),
                "execution_backend": _backend,
            },
        )
        imported += 1
    conditions = len(condition_keys)
    return {
        "status": "imported",
        "measurements": imported,
        "conditions": conditions,
        "skipped": skipped,
        "include_heldout": include_heldout,
        "inventory": store.inventory(),
    }




from .a3_cv import CvCalibration, CvCondition
from .a3_communication import CommCondition, CommunicationCalibration

__all__ = [
    "machine_from_environment", "baseline_gemm", "baseline_add", "baseline_copy",
    "import_linearity_calibration_cases",
    "import_a3_measurements", "import_communication_measurements",
    "CvCondition", "CvCalibration", "CommCondition", "CommunicationCalibration",
]


def import_linearity_calibration_cases(
    store: SQLiteCalibrationStore,
    path: str | Path,
    *,
    include_heldout: bool = False,
) -> dict[str, Any]:
    """Import operator-owned linearity calibration cases without defaulting
    missing source/hardware/protocol facts to A3 measured.

    The export must carry per-case provenance.  Missing facts remain unknown
    (non-calibratable) rather than being replaced by A3/measured defaults.
    """
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("cases"), Sequence):
        raise A3InputError("linearity_calibration_export_invalid")
    data_environment = dict(data.get("environment") or {})
    data_protocol = dict(data.get("protocol") or {})
    imported = 0
    skipped = 0
    gaps: list[dict[str, Any]] = []
    for case in data["cases"]:
        if not isinstance(case, Mapping) or case.get("status") != "completed":
            skipped += 1
            continue
        split = str(case.get("split", data.get("split", "train"))).strip().lower()
        split = {"holdout": "heldout", "held_out": "heldout"}.get(split, split)
        if split == "heldout" and not include_heldout:
            skipped += 1
            continue
        raw = case.get("raw_device_ms_per_op")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            gaps.append({
                "case_id": case.get("case_id"),
                "reason": "raw_device_ms_per_op_missing",
            })
            skipped += 1
            continue
        environment = dict(case.get("environment") or data_environment)
        protocol = {
            **data_protocol, **dict(case.get("protocol") or {}),
        }
        device_name = str(
            environment.get("device_name")
            or environment.get("device")
            or ""
        )
        machine = None
        if device_name:
            machine = machine_from_environment(environment)
        shape_raw = case.get("shape_mkn")
        elements = 0
        m = k = n = 0
        if isinstance(shape_raw, Sequence) and len(shape_raw) == 3:
            m, k, n = (int(v) for v in shape_raw)
            shape_payload = {"m": m, "n": n, "k": k}
        else:
            shape_list = case.get("shape")
            if not isinstance(shape_list, Sequence) or len(shape_list) != 1:
                gaps.append({
                    "case_id": case.get("case_id"),
                    "reason": "shape_mkn_or_elements_missing",
                })
                skipped += 1
                continue
            elements = int(shape_list[0])
            shape_payload = {"elements": elements}
        dtype = str(case.get("dtype") or "unknown").lower().replace("-", "_")
        dtype = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}.get(
            dtype, dtype,
        )
        family = str(
            case.get("source_family") or case.get("kind") or "unknown"
        )
        recipe = str(
            case.get("source_recipe") or case.get("call_path") or "unknown"
        )
        source_class = str(
            case.get("source_class") or data.get("source_class") or "unknown"
        )
        source = str(
            case.get("source") or case.get("call_path")
            or "operator-linearity-export"
        )
        run_id = str(case.get("run_id") or data.get("run_id") or "")
        case_id = str(case.get("case_id") or "linearity-case")
        if machine is None:
            gaps.append({
                "case_id": case_id, "reason": "machine_identity_missing",
            })
            skipped += 1
            continue
        if shape_raw is not None and isinstance(shape_raw, Sequence):
            analytic = baseline_gemm(machine, m, n, k, dtype=dtype)
        elif family in {"add", "elementwise"}:
            analytic = baseline_add(machine, elements, dtype=dtype)
        elif family in {"copy", "memcpy"}:
            analytic = baseline_copy(machine, elements, dtype=dtype)
        else:
            gaps.append({
                "case_id": case_id, "reason": "unsupported_non_gemm_baseline",
            })
            skipped += 1
            continue
        for index, duration in enumerate(raw):
            store.record_sample(
                op_family=family, phase="operator", recipe=recipe,
                shape=shape_payload,
                duration_ms=float(duration), analytic_ms=float(analytic),
                dtype=dtype, layout=str(case.get("layout") or "unknown"),
                hardware_key=(machine.hardware_key if machine is not None else ""),
                features={
                    **dict(case.get("features") or {}),
                    "source_family": family, "source_recipe": recipe,
                    "native_precision_claim": case.get("native_precision_claim"),
                    "coverage_scope": case.get("coverage_scope"),
                    "analytic_baseline_omitted": False,
                },
                source=source, source_class=source_class,
                run_id=run_id, case_id=case_id, repeat_index=index,
                split=split,
                timing_scope=str(
                    case.get("timing_scope")
                    or protocol.get("timing_scope")
                    or protocol.get("measurement_scope")
                    or "device_event_warm"
                ),
                protocol=protocol, environment=environment,
                details={
                    "case_id": case_id, "raw_case": case.get("kind"),
                    "linearity_export": True,
                    "source_class": source_class,
                    "run_id": run_id,
                    "device_name": device_name,
                },
            )
            imported += 1
    return {
        "status": "imported" if imported else "empty",
        "imported_raw_samples": imported,
        "skipped_cases": skipped,
        "gaps": gaps,
        "source_class_defaulted": False,
        "hardware_defaulted": False,
    }


def import_linearity_split_results(
    store: SQLiteCalibrationStore,
    path: str | Path,
    *,
    include_heldout: bool = False,
) -> dict[str, Any]:
    """Adapt a linearity split-results JSON and import its calibration cases."""
    import tempfile

    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    from .a3_operator_curves_adapter import linearity_calibration_groups

    exported = linearity_calibration_groups(data)
    imported = 0
    skipped = 0
    gaps: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for group in exported.get("groups", ()):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="mindie-linearity-group-",
            delete=False,
        ) as handle:
            json.dump(group, handle)
            temporary = handle.name
        try:
            result = import_linearity_calibration_cases(
                store, temporary, include_heldout=include_heldout,
            )
        finally:
            try:
                Path(temporary).unlink()
            except OSError:
                pass
        imported += int(result.get("imported_raw_samples", 0) or 0)
        skipped += int(result.get("skipped_cases", 0) or 0)
        gaps.extend(result.get("gaps", ()))
        groups.append({
            "run_id": group.get("run_id"),
            "hardware_key": group.get("hardware_key"),
            "source_class": group.get("source_class"),
            "environment": group.get("environment"),
            "protocol": group.get("protocol"),
            "imported_raw_samples": result.get("imported_raw_samples", 0),
        })
    return {
        "status": "imported" if imported else "empty",
        "imported_raw_samples": imported,
        "skipped_cases": skipped,
        "gaps": gaps,
        "groups": groups,
        "adapter_group_count": len(exported.get("groups", ())),
        "adapter_excluded_cases": len(exported.get("excluded_cases", ())),
        "source_class_defaulted": False,
        "hardware_defaulted": False,
    }


__all__ = [*__all__, "import_linearity_split_results"]
