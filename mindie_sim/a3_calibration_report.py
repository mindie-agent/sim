"""Inventory and coverage reports over the lightweight calibration store.

[Role]
  Keep SQL/reporting code separate from sample ingestion.  Reports expose raw
  measurement counts, repeat statistics, observed shape/hardware/dtype/layout
  envelopes and optional query neighbor/extrapolation checks.

[Invariants]
  * Counts distinguish raw repeats, unique semantic conditions, calibratable
    measured repeats and separate runs.
  * Coverage is observed data, not a guarantee for unmeasured shapes.
  * Heldout rows are visible in coverage but excluded from default fitting.

[Timeouts & Locks]
  Read-only SQLite queries through the store's WAL connection.

[Side-Effect Policy]
  No writes and no device/network access.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def inventory_report(store: Any) -> dict[str, Any]:
    with store._connect() as connection:
        measurements = connection.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
        calibratable = connection.execute(
            """SELECT COUNT(*) FROM measurements
               WHERE source_class IN ('measured','observed') AND synthetic = 0"""
        ).fetchone()[0]
        conditions = connection.execute("SELECT COUNT(*) FROM sample_statistics").fetchone()[0]
        runs = connection.execute(
            "SELECT COUNT(DISTINCT run_id) FROM measurements WHERE run_id != ''"
        ).fetchone()[0]
        by_family = connection.execute(
            """SELECT op_family, phase, split, source_class, COUNT(*)
               FROM measurements GROUP BY op_family, phase, split, source_class
               ORDER BY op_family, phase, split"""
        ).fetchall()
        request_counts = connection.execute(
            "SELECT status, COUNT(*) FROM measurement_requests GROUP BY status"
        ).fetchall()
        repeats = connection.execute(
            "SELECT MIN(measurement_count), MAX(measurement_count) FROM sample_statistics"
        ).fetchone()
    return {
        "database": store.path,
        "measurements": int(measurements),
        "calibratable_measurements": int(calibratable),
        "conditions": int(conditions),
        "runs": int(runs),
        "repeats_per_condition": {
            "min": int(repeats[0] or 0), "max": int(repeats[1] or 0),
        },
        "by_family": [
            {"op_family": row[0], "phase": row[1], "split": row[2],
             "source_class": row[3], "measurements": int(row[4])}
            for row in by_family
        ],
        "requests": {str(row[0]): int(row[1]) for row in request_counts},
        "revision_is_metadata_only": True,
    }


def coverage_report(store: Any, *, queries: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    stats = store.statistics(split=None)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for record in stats:
        key = (record["op_family"], record["phase"])
        group = groups.setdefault(key, {
            "op_family": record["op_family"], "phase": record["phase"],
            "splits": {}, "shape_ranges": {},
            "hardware_keys": set(), "dtypes": set(), "layouts": set(),
        })
        entry = group["splits"].setdefault(record["split"], {
            "conditions": 0, "measurements": 0, "calibratable_measurements": 0, "runs": 0,
        })
        entry["conditions"] += 1
        entry["measurements"] += int(record["measurement_count"])
        entry["calibratable_measurements"] += int(record["calibratable_count"])
        entry["runs"] += int(record["run_count"])
        for name, value in record["shape"].items():
            current = group["shape_ranges"].setdefault(
                str(name), {"min": float(value), "max": float(value)}
            )
            current["min"] = min(current["min"], float(value))
            current["max"] = max(current["max"], float(value))
        if record["hardware_key"]:
            group["hardware_keys"].add(record["hardware_key"])
        if record["dtype"]:
            group["dtypes"].add(record["dtype"])
        if record["layout"]:
            group["layouts"].add(record["layout"])
    for group in groups.values():
        group["hardware_keys"] = sorted(group["hardware_keys"])
        group["dtypes"] = sorted(group["dtypes"])
        group["layouts"] = sorted(group["layouts"])
    payload: dict[str, Any] = {
        "groups": list(groups.values()),
        "note": "Observed shape/feature envelope only; unmeasured neighbors remain extrapolation.",
    }
    if queries:
        evaluations = []
        for query in queries:
            family = str(query.get("op_family", ""))
            phase = str(query.get("phase", ""))
            shape = {str(k): float(v) for k, v in (query.get("shape") or {}).items()}
            candidates = [
                record for record in stats
                if record["op_family"] == family and record["phase"] == phase
                and (not query.get("recipe") or record["recipe"] == query.get("recipe"))
                and (not query.get("hardware_key") or record["hardware_key"] == query.get("hardware_key"))
                and (not query.get("dtype") or record["dtype"] == query.get("dtype"))
                and (not query.get("layout") or record["layout"] == query.get("layout"))
            ]
            if not candidates:
                evaluations.append({"query": dict(query), "status": "no_coverage"})
                continue
            neighbor_rows = []
            for record in candidates:
                common = set(shape) & set(record["shape"])
                if not common:
                    continue
                distance = sum(
                    abs(math.log((abs(shape[key]) + 1.0) / (abs(record["shape"][key]) + 1.0)))
                    for key in common
                )
                neighbor_rows.append((distance, record, common))
            if not neighbor_rows:
                evaluations.append({"query": dict(query), "status": "no_common_dimension"})
                continue
            envelope = {}
            for key in set(shape):
                values = [record["shape"][key] for _, record, common in neighbor_rows if key in common]
                if values:
                    envelope[key] = [min(values), max(values)]
            inside = all(
                envelope[key][0] <= shape[key] <= envelope[key][1]
                for key in shape if key in envelope
            ) and set(shape).issubset({key for _, _, common in neighbor_rows for key in common})
            nearest = min(neighbor_rows, key=lambda item: item[0])
            evaluations.append({
                "query": dict(query),
                "status": "inside_envelope" if inside else "extrapolated",
                "nearest_distance": nearest[0],
                "nearest_sample_key": nearest[1]["sample_key"],
                "envelope": envelope,
            })
        payload["query_evaluations"] = evaluations
    return payload



def manifest_report(store: Any, *, limit: int = 32, status: str = "pending") -> dict[str, Any]:
    requests = store.requests(status=status, limit=limit)
    return {
        "database": store.path,
        "status": "ready" if requests else "empty",
        "requests": [
            {
                "request_key": request["request_key"],
                "request_id": request["request_id"],
                "hardware_key": request["hardware_key"],
                "op_family": request["op_family"], "phase": request["phase"],
                "recipe": request["recipe"], "shape": request["shape"],
                "dtype": request["dtype"], "layout": request["layout"],
                "features": request["features"],
                "competition": request["competition"],
                "priority": request["priority"],
                "query_count": request["query_count"],
                "predicted_ms": request["predicted_ms"],
                "analytic_ms": request["analytic_ms"],
                "attempt_count": request["attempt_count"],
                "error_ratio": request["error_ratio"],
                "extrapolation": request["extrapolation"],
                "target_impact": request["target_impact"],
                "run_id": request["run_id"],
                "request_status": request["status"],
                "details": request.get("details", {}),
                "protocol": (request.get("details") or {}).get("protocol", {}),
                "timing_scope": (
                    (request.get("details") or {}).get("timing_scope", "")
                ),
                "environment": (
                    (request.get("details") or {}).get("environment", {})
                ),
                "missing_condition": (
                    (request.get("details") or {}).get("missing_condition")
                ),
            }
            for request in requests
        ],
        "runner_contract": {
            "requirement": "measure declared semantic op/shape on declared hardware; return raw repeats and protocol",
            "no_npu_lease_created_by_sim": True,
            "managed_device_injection": "scheduler/coordinator injects the device; sim never SSHes or selects a free card",
            "ingest_command": "mindie-sim calibration ingest --db <db> --results <results.json>",
            "standalone_command": "python -m mindie_sim.a3_measurement --manifest <manifest.json> --results <results.json> --runner torch-npu",
            "result_contract": {
                "request_key": "string",
                "status": "completed|failed (optional; failed needs reason)",
                "duration_ms": "positive number or raw_samples_ms",
                "analytic_ms": "positive model baseline",
                "source": "non-empty string", "source_class": "measured|observed|derived",
                "run_id": "external run id", "split": "train|heldout",
                "timing_scope": "string", "protocol": "object",
                "environment": "object", "details": "object",
            },
        },
    }


def export_manifest_report(store: Any, path: str | Path, *, limit: int = 32) -> dict[str, Any]:
    manifest = store.manifest(limit=limit)
    Path(path).expanduser().write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"status": "written", "path": str(path), "requests": len(manifest["requests"])}


__all__ = [
    "inventory_report", "coverage_report", "manifest_report",
    "export_manifest_report",
]
