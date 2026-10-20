"""Inventory and coverage reports over the lightweight calibration store.

[Role]
  Keep SQL/reporting code separate from sample ingestion.  Reports expose raw
  measurement counts, repeat statistics, observed shape/hardware/dtype/layout
  envelopes and optional query neighbor/extrapolation checks.

[Invariants]
  * Counts distinguish raw repeats, unique semantic conditions, calibratable
    measured repeats, trials and separate runs.  The calibration target is
    ``duration_trial_median``; the pooled repeat median is reported only as a
    diagnostic (``estimator.pooled_event_median``).
  * Coverage is observed data, not a guarantee for unmeasured shapes.
  * Heldout rows are visible in coverage but are never fitted.
  * An empty hardware key is reported as unbound, never as a wildcard.
  * Split collisions (the same geometry under two split labels) are reported,
    never silently relabelled.

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
               WHERE source_class IN ('measured','observed') AND synthetic = 0
                 AND source != '' AND duration_ms > 0 AND analytic_ms > 0"""
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
        calibratable_conditions = connection.execute(
            "SELECT COUNT(*) FROM sample_statistics WHERE calibratable_count > 0"
        ).fetchone()[0]
        trials = connection.execute(
            """SELECT COUNT(*), COALESCE(MAX(trial_count), 0),
                      COALESCE(SUM(CASE WHEN trial_count > 1 THEN 1 ELSE 0 END), 0),
                      COALESCE(SUM(CASE WHEN duration_trial_median IS NOT NULL
                                         AND duration_median IS NOT NULL
                                         AND ABS(duration_median - duration_trial_median) > 1e-12
                                   THEN 1 ELSE 0 END), 0),
                      MIN(CASE WHEN duration_trial_median > 0
                               THEN duration_median / duration_trial_median END),
                      MAX(CASE WHEN duration_trial_median > 0
                               THEN duration_median / duration_trial_median END)
               FROM sample_statistics"""
        ).fetchone()
        hardware = connection.execute(
            """SELECT hardware_key, COUNT(*) FROM measurements
               GROUP BY hardware_key ORDER BY hardware_key"""
        ).fetchall()
        unbound_conditions = connection.execute(
            "SELECT COUNT(*) FROM sample_statistics WHERE hardware_key = ''"
        ).fetchone()[0]
        provenance = connection.execute(
            """SELECT COUNT(*),
                      COALESCE(SUM(CASE WHEN die NOT IN ('', 'unknown') THEN 1 ELSE 0 END), 0),
                      COALESCE(SUM(CASE WHEN visible_devices NOT IN ('', 'unknown') THEN 1 ELSE 0 END), 0),
                      COALESCE(SUM(CASE WHEN protocol_signature != '' THEN 1 ELSE 0 END), 0),
                      COALESCE(SUM(CASE WHEN timing_scope != '' THEN 1 ELSE 0 END), 0),
                      COALESCE(SUM(CASE WHEN run_id != '' THEN 1 ELSE 0 END), 0),
                      COALESCE(SUM(CASE WHEN source_class IN ('measured','observed')
                                         AND synthetic = 0 AND source != ''
                                   THEN 1 ELSE 0 END), 0)
               FROM measurements"""
        ).fetchone()
        split_counts = connection.execute(
            "SELECT split, COUNT(*) FROM measurements GROUP BY split ORDER BY split"
        ).fetchall()
        # L1: one physical condition carries exactly one split label.  Geometry
        # here is the semantic payload of ``sample_key`` minus the split, so a
        # condition measured as both train and heldout is visible as one row.
        collision_rows = connection.execute(
            """SELECT GROUP_CONCAT(DISTINCT split), MIN(sample_key), MIN(hardware_key),
                      MIN(op_family), MIN(phase)
               FROM measurements
               GROUP BY hardware_key, op_family, phase, recipe, shape_json, dtype,
                        layout, features_json, competition_json, timing_scope,
                        protocol_json, environment_json
               HAVING COUNT(DISTINCT split) > 1
               ORDER BY MIN(op_family), MIN(phase)
               LIMIT 8"""
        ).fetchall()
        collisions = connection.execute(
            """SELECT COUNT(*) FROM (
                   SELECT COUNT(DISTINCT split) AS labels
                   FROM measurements
                   GROUP BY hardware_key, op_family, phase, recipe, shape_json, dtype,
                            layout, features_json, competition_json, timing_scope,
                            protocol_json, environment_json
                   HAVING labels > 1
               )"""
        ).fetchone()[0]
    return {
        "database": store.path,
        "measurements": int(measurements),
        "calibratable_measurements": int(calibratable),
        "conditions": int(conditions),
        "calibratable_conditions": int(calibratable_conditions),
        "runs": int(runs),
        "repeats_per_condition": {
            "min": int(repeats[0] or 0), "max": int(repeats[1] or 0),
        },
        "estimator": {
            "calibration_target": "duration_trial_median = median over trials of (median over trial repeats)",
            "trial_identity": "one runner invocation, (run_id, case_id); repeats without a case id are single-repeat trials",
            "pooled_event_median": "duration_median is diagnostic only; fitted values never use it",
            "conditions": int(trials[0] or 0),
            "trial_count_max": int(trials[1] or 0),
            "multi_trial_conditions": int(trials[2] or 0),
            "pooled_target_mismatch_conditions": int(trials[3] or 0),
            "pooled_over_target_min": trials[4],
            "pooled_over_target_max": trials[5],
        },
        "hardware_identity": {
            "keys": [
                {"hardware_key": str(row[0]), "measurements": int(row[1])}
                for row in hardware
            ],
            "unbound_conditions": int(unbound_conditions),
            "unbound_key_policy": (
                "an empty hardware key never matches a query that names a machine; "
                "rejected observations are reported with their reason"
            ),
            "empty_key_is_wildcard": False,
        },
        "provenance": {
            "measurements": int(provenance[0] or 0),
            "with_die": int(provenance[1] or 0),
            "with_visible_devices": int(provenance[2] or 0),
            "with_protocol_signature": int(provenance[3] or 0),
            "with_timing_scope": int(provenance[4] or 0),
            "with_run_id": int(provenance[5] or 0),
            "attributable": int(provenance[6] or 0),
            "unattested_identity_label": "unknown",
        },
        "splits": {
            "measurements": {str(row[0]): int(row[1]) for row in split_counts},
            "heldout_fit_allowed": False,
            "geometry_conflicts": {
                "count": int(collisions),
                "examples": [
                    {
                        "splits": sorted(str(row[0]).split(",")),
                        "sample_key": str(row[1]), "hardware_key": str(row[2]),
                        "op_family": str(row[3]), "phase": str(row[4]),
                    }
                    for row in collision_rows
                ],
                "policy": (
                    "L1: a physical condition may carry only one split label; the "
                    "store reports collisions and never relabels rows"
                ),
            },
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
            "hardware_keys": set(), "unbound_conditions": 0,
            "dtypes": set(), "layouts": set(),
        })
        entry = group["splits"].setdefault(record["split"], {
            "conditions": 0, "measurements": 0, "calibratable_measurements": 0,
            "runs": 0, "trials": 0,
        })
        entry["conditions"] += 1
        entry["measurements"] += int(record["measurement_count"])
        entry["calibratable_measurements"] += int(record["calibratable_count"])
        entry["runs"] += int(record["run_count"])
        entry["trials"] += int(record["trial_count"] or 0)
        for name, value in record["shape"].items():
            current = group["shape_ranges"].setdefault(
                str(name), {"min": float(value), "max": float(value)}
            )
            current["min"] = min(current["min"], float(value))
            current["max"] = max(current["max"], float(value))
        if record["hardware_key"]:
            group["hardware_keys"].add(record["hardware_key"])
        else:
            group["unbound_conditions"] += 1
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
        "estimator": {
            "target": "median over trials of per-trial repeat medians",
            "weights": "conditions, not repeats or events",
        },
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
                "case_id": "optional trial id; repeats of one invocation share it",
                "timing_scope": "string", "protocol": "object",
                "die": "runner-attested die/Phy-ID; absent is stored as unknown",
                "visible_devices": "ASCEND_RT_VISIBLE_DEVICES; absent is stored as unknown",
                "environment": "object", "details": "object",
            },
            "estimator_contract": (
                "training target = median over trials of per-trial repeat medians; "
                "the pooled repeat median is diagnostic only"
            ),
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
