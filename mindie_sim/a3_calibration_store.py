"""SQLite raw-measurement store, coverage, and re-measurement loop.

[Role]
  Persist every raw measurement repeat (run/case/repeat), aggregate condition
  statistics for calibration, deduplicate missing-measurement requests, expose
  shape/feature coverage, and produce an executable manifest.  Ingest keeps
  train and heldout split labels separate.

[Invariants]
  * Independent measurements at the same semantic condition are never dropped:
    only an explicitly identified run/case/repeat is idempotent.
  * ``sample_key`` includes hardware, op family, phase, recipe, shape, dtype,
    layout, features, competition and split.  Revision/source are metadata.
  * Calibration statistics include only measured/observed, non-synthetic
    samples.  Synthetic-as-measured is rejected.
  * Raw repeats, timing protocol/environment and details survive ingestion.
  * Pending request deduplication is separate from measurement deduplication.
  * Heldout rows stay heldout and are excluded from the default calibration
    view.
  * Status is pending/running/completed/failed; no NPU job/lease is created.
  * One current contract: business tables are created directly from executable
    SQL.  No schema-version row, migration ladder or reflection gate exists.

[Timeouts & Locks]
  SQLite WAL and ``busy_timeout=5000`` on every connection.  No filesystem
  owner/stat probe, reflection ORM, global lock or wall-clock wait.

[Side-Effect Policy]
  Writes only the explicitly supplied SQLite path when recording.  No device,
  SSH, subprocess, network or model import.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .a3_cost import CalibrationObservation, CalibrationStore
from .a3_operator_curve_protocol import canonical_protocol
from .a3_store_types import (
    SampleRecord, _json, _measurement_key, _percentile, _semantic_key,
    _utc_now, canonical_timing_scope,
)
from .a3_types import A3InputError

STATUSES = ("pending", "running", "completed", "failed")
_REQUEST_COLUMNS = (
    "request_id", "request_key", "hardware_key", "op_family", "phase", "recipe",
    "shape_json", "dtype", "layout", "features_json", "competition_json", "status",
    "run_id", "priority", "query_count", "predicted_ms", "analytic_ms",
    "error_ratio", "extrapolation", "target_impact", "reason", "details_json",
    "attempt_count", "attempt_history_json", "created_at", "updated_at", "sample_id",
)


def _mapping_field(value: Any, reason: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise A3InputError(reason)
    return dict(value)


def _optional_mapping_field(value: Any, reason: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _mapping_field(value, reason)


def _shape_field(value: Any, reason: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise A3InputError(reason)
    shape: dict[str, float] = {}
    for key, item in value.items():
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise A3InputError(reason) from exc
        if not math.isfinite(number) or number <= 0:
            raise A3InputError(reason)
        shape[str(key)] = number
    return shape


class SQLiteCalibrationStore:
    """WAL SQLite raw-sample store; each operation opens a bounded connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._memory_connection: sqlite3.Connection | None = None
        self._ensure_tables()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self.path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(":memory:", timeout=5.0)
                self._configure(self._memory_connection)
                self._create_tables(self._memory_connection)
            yield self._memory_connection
            self._memory_connection.commit()
            return
        connection = sqlite3.connect(self.path, timeout=5.0)
        self._configure(connection)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _configure(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA synchronous=NORMAL")

    def _ensure_tables(self) -> None:
        with self._connect() as connection:
            self._create_tables(connection)

    @staticmethod
    def _create_tables(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                measurement_key TEXT NOT NULL UNIQUE,
                sample_key TEXT NOT NULL,
                hardware_key TEXT NOT NULL DEFAULT '',
                op_family TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT '',
                recipe TEXT NOT NULL DEFAULT '',
                shape_json TEXT NOT NULL,
                dtype TEXT NOT NULL DEFAULT '',
                layout TEXT NOT NULL DEFAULT '',
                features_json TEXT NOT NULL DEFAULT '{}',
                competition_json TEXT NOT NULL DEFAULT '{}',
                duration_ms REAL NOT NULL,
                analytic_ms REAL NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                source_class TEXT NOT NULL DEFAULT 'unknown',
                synthetic INTEGER NOT NULL DEFAULT 0,
                revision TEXT NOT NULL DEFAULT '',
                run_id TEXT NOT NULL DEFAULT '',
                case_id TEXT NOT NULL DEFAULT '',
                repeat_index INTEGER NOT NULL DEFAULT 0,
                split TEXT NOT NULL DEFAULT 'train',
                timing_scope TEXT NOT NULL DEFAULT '',
                protocol_json TEXT NOT NULL DEFAULT '{}',
                environment_json TEXT NOT NULL DEFAULT '{}',
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_measurements_sample
                ON measurements(sample_key, split);
            CREATE TABLE IF NOT EXISTS sample_statistics (
                sample_key TEXT PRIMARY KEY,
                hardware_key TEXT NOT NULL DEFAULT '',
                op_family TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT '',
                recipe TEXT NOT NULL DEFAULT '',
                shape_json TEXT NOT NULL,
                dtype TEXT NOT NULL DEFAULT '',
                layout TEXT NOT NULL DEFAULT '',
                features_json TEXT NOT NULL DEFAULT '{}',
                competition_json TEXT NOT NULL DEFAULT '{}',
                split TEXT NOT NULL DEFAULT 'train',
                measurement_count INTEGER NOT NULL DEFAULT 0,
                calibratable_count INTEGER NOT NULL DEFAULT 0,
                run_count INTEGER NOT NULL DEFAULT 0,
                run_ids_json TEXT NOT NULL DEFAULT '[]',
                duration_median REAL,
                duration_mean REAL,
                duration_min REAL,
                duration_max REAL,
                duration_p90 REAL,
                duration_std REAL,
                analytic_median REAL,
                source_classes_json TEXT NOT NULL DEFAULT '[]',
                synthetic_count INTEGER NOT NULL DEFAULT 0,
                timing_scope TEXT NOT NULL DEFAULT '',
                protocol_json TEXT NOT NULL DEFAULT '{}',
                environment_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS measurement_requests (
                request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_key TEXT NOT NULL UNIQUE,
                hardware_key TEXT NOT NULL DEFAULT '',
                op_family TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT '',
                recipe TEXT NOT NULL DEFAULT '',
                shape_json TEXT NOT NULL,
                dtype TEXT NOT NULL DEFAULT '',
                layout TEXT NOT NULL DEFAULT '',
                features_json TEXT NOT NULL DEFAULT '{}',
                competition_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                run_id TEXT NOT NULL DEFAULT '',
                priority REAL NOT NULL DEFAULT 0,
                query_count INTEGER NOT NULL DEFAULT 1,
                predicted_ms REAL,
                analytic_ms REAL,
                error_ratio REAL NOT NULL DEFAULT 0,
                extrapolation REAL NOT NULL DEFAULT 0,
                target_impact REAL NOT NULL DEFAULT 0,
                reason TEXT NOT NULL DEFAULT '',
                details_json TEXT NOT NULL DEFAULT '{}',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                attempt_history_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                sample_id INTEGER
            );
            CREATE INDEX IF NOT EXISTS ix_requests_status
                ON measurement_requests(status, priority DESC);
            """
        )

    # -- raw samples / statistics ---------------------------------------
    def record_sample(
        self,
        *,
        op_family: str,
        phase: str,
        recipe: str,
        shape: Mapping[str, Any],
        duration_ms: float,
        analytic_ms: float,
        dtype: str = "",
        layout: str = "",
        hardware_key: str = "",
        features: Mapping[str, Any] | None = None,
        competition: Mapping[str, Any] | None = None,
        source: str = "",
        source_class: str = "unknown",
        revision: str = "",
        run_id: str = "",
        case_id: str = "",
        repeat_index: int = 0,
        split: str = "train",
        synthetic: bool = False,
        timing_scope: str = "",
        protocol: Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        measurement_key: str = "",
    ) -> dict[str, Any]:
        source_class = str(source_class).strip().lower()
        if source_class not in {"measured", "observed", "derived", "synthetic", "unknown"}:
            raise A3InputError(f"calibration.source_class_invalid:{source_class}")
        if not str(op_family).strip():
            raise A3InputError("calibration.op_family_required")
        if synthetic and source_class in {"measured", "observed"}:
            raise A3InputError("calibration.synthetic_cannot_be_measured")
        if source_class in {"measured", "observed"} and not str(source).strip():
            raise A3InputError("calibration.measured_source_required")
        split = {"holdout": "heldout", "held_out": "heldout", "validation": "heldout"}.get(
            str(split).strip().lower(), str(split).strip().lower()
        )
        if split not in {"train", "heldout"}:
            raise A3InputError(f"calibration.split_invalid:{split}")
        try:
            duration = float(duration_ms)
            analytic = float(analytic_ms)
        except (TypeError, ValueError) as exc:
            raise A3InputError("calibration.duration_invalid") from exc
        if not math.isfinite(duration) or not math.isfinite(analytic):
            raise A3InputError("calibration.duration_invalid")
        if duration <= 0 or analytic <= 0:
            raise A3InputError("calibration.duration_invalid")
        shape_json = _shape_field(shape, "calibration.shape_invalid")
        features_json = _optional_mapping_field(features, "calibration.features_invalid")
        competition_json = _optional_mapping_field(
            competition, "calibration.competition_invalid",
        )
        protocol_json = canonical_protocol(
            _optional_mapping_field(protocol, "calibration.protocol_invalid")
        )
        environment_json = _optional_mapping_field(
            environment, "calibration.environment_invalid",
        )
        details_json = _optional_mapping_field(details, "calibration.details_invalid")
        timing_scope = canonical_timing_scope(timing_scope)
        payload = {
            "hardware_key": hardware_key, "op_family": op_family, "phase": phase,
            "recipe": recipe, "shape": shape_json, "dtype": dtype, "layout": layout,
            "features": features_json, "competition": competition_json, "split": split,
            "timing_scope": str(timing_scope),
            # Protocol and environment are semantic conditions, not display
            # metadata.  A different timer/allocation/cache convention must not
            # pool its residual with another measured condition.
            "protocol": protocol_json,
            "environment": environment_json,
        }
        key = _semantic_key(payload)
        mkey = measurement_key or _measurement_key(run_id, case_id, repeat_index, key)
        now = _utc_now()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT measurement_id FROM measurements WHERE measurement_key = ?",
                (mkey,),
            ).fetchone()
            if existing is not None:
                return {"status": "duplicate", "sample_key": key,
                        "measurement_key": mkey, "measurement_id": existing[0]}
            cursor = connection.execute(
                """INSERT INTO measurements(
                    measurement_key, sample_key, hardware_key, op_family, phase, recipe,
                    shape_json, dtype, layout, features_json, competition_json,
                    duration_ms, analytic_ms, source, source_class, synthetic,
                    revision, run_id, case_id, repeat_index, split, timing_scope,
                    protocol_json, environment_json, details_json, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mkey, key, hardware_key, op_family, phase, recipe, _json(shape_json),
                    dtype, layout, _json(features_json), _json(competition_json),
                    duration, analytic, source, source_class, int(synthetic),
                    revision, run_id, case_id, int(repeat_index), split, timing_scope,
                    _json(protocol_json), _json(environment_json), _json(details_json), now,
                ),
            )
            measurement_id = int(cursor.lastrowid)
            self._recompute_statistics(connection, key)
        return {"status": "inserted", "sample_key": key, "measurement_key": mkey,
                "measurement_id": measurement_id}

    @staticmethod
    def _recompute_statistics(connection: sqlite3.Connection, sample_key: str) -> None:
        row = connection.execute(
            """SELECT hardware_key, op_family, phase, recipe, shape_json, dtype,
                      layout, features_json, competition_json, split
               FROM measurements WHERE sample_key = ? LIMIT 1""",
            (sample_key,),
        ).fetchone()
        if row is None:
            return
        rows = connection.execute(
            """SELECT duration_ms, analytic_ms, source_class, synthetic, run_id,
                      timing_scope, protocol_json, environment_json
               FROM measurements WHERE sample_key = ?""",
            (sample_key,),
        ).fetchall()
        calibratable = [
            item for item in rows
            if item[2] in {"measured", "observed"} and not item[3]
        ]
        durations = [float(item[0]) for item in calibratable]
        analytics = [float(item[1]) for item in calibratable]
        run_ids = sorted({str(item[4]) for item in calibratable if item[4]})
        source_classes = sorted({str(item[2]) for item in rows})
        protocol = (
            canonical_protocol(json.loads(calibratable[-1][6]))
            if calibratable else {}
        )
        environment = json.loads(calibratable[-1][7]) if calibratable else {}
        timing_scope = str(calibratable[-1][5]) if calibratable else ""
        updated = _utc_now()
        connection.execute(
            """INSERT INTO sample_statistics(
                sample_key, hardware_key, op_family, phase, recipe, shape_json,
                dtype, layout, features_json, competition_json, split,
                measurement_count, calibratable_count, run_count, run_ids_json,
                duration_median, duration_mean, duration_min, duration_max,
                duration_p90, duration_std, analytic_median, source_classes_json,
                synthetic_count, timing_scope, protocol_json, environment_json, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(sample_key) DO UPDATE SET
                measurement_count=excluded.measurement_count,
                calibratable_count=excluded.calibratable_count,
                run_count=excluded.run_count, run_ids_json=excluded.run_ids_json,
                duration_median=excluded.duration_median, duration_mean=excluded.duration_mean,
                duration_min=excluded.duration_min, duration_max=excluded.duration_max,
                duration_p90=excluded.duration_p90, duration_std=excluded.duration_std,
                analytic_median=excluded.analytic_median,
                source_classes_json=excluded.source_classes_json,
                synthetic_count=excluded.synthetic_count,
                timing_scope=excluded.timing_scope, protocol_json=excluded.protocol_json,
                environment_json=excluded.environment_json, updated_at=excluded.updated_at""",
            (
                sample_key, row[0], row[1], row[2], row[3], row[4], row[5], row[6],
                row[7], row[8], row[9], len(rows), len(calibratable), len(run_ids),
                _json(run_ids), _percentile(durations, 0.5), statistics.fmean(durations) if durations else None,
                min(durations) if durations else None, max(durations) if durations else None,
                _percentile(durations, 0.9), statistics.pstdev(durations) if len(durations) > 1 else 0.0,
                _percentile(analytics, 0.5), _json(source_classes),
                sum(1 for item in rows if item[3]), timing_scope, _json(protocol),
                _json(environment), updated,
            ),
        )

    def samples(self, *, split: str | None = None) -> list[SampleRecord]:
        query = "SELECT * FROM measurements"
        params: tuple[Any, ...] = ()
        if split is not None:
            query += " WHERE split = ?"
            params = (split,)
        query += " ORDER BY measurement_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            result.append(SampleRecord(
                id=int(row[0]), measurement_key=str(row[1]), sample_key=str(row[2]),
                hardware_key=str(row[3]), op_family=str(row[4]), phase=str(row[5]),
                recipe=str(row[6]), shape=json.loads(row[7]), dtype=str(row[8]),
                layout=str(row[9]), features=json.loads(row[10]),
                competition=json.loads(row[11]), duration_ms=float(row[12]),
                analytic_ms=float(row[13]), source=str(row[14]),
                source_class=str(row[15]), synthetic=bool(row[16]),
                revision=str(row[17]), run_id=str(row[18]), case_id=str(row[19]),
                repeat_index=int(row[20]), split=str(row[21]), timing_scope=str(row[22]),
                protocol=json.loads(row[23]), environment=json.loads(row[24]),
                details=json.loads(row[25]),
            ))
        return result

    def statistics(self, *, split: str | None = "train") -> list[dict[str, Any]]:
        query = "SELECT * FROM sample_statistics"
        params: tuple[Any, ...] = ()
        if split is not None:
            query += " WHERE split = ?"
            params = (split,)
        query += " ORDER BY op_family, phase, sample_key"
        with self._connect() as connection:
            columns = [
                "sample_key", "hardware_key", "op_family", "phase", "recipe",
                "shape_json", "dtype", "layout", "features_json", "competition_json",
                "split", "measurement_count", "calibratable_count", "run_count",
                "run_ids_json", "duration_median", "duration_mean", "duration_min",
                "duration_max", "duration_p90", "duration_std", "analytic_median",
                "source_classes_json", "synthetic_count", "timing_scope",
                "protocol_json", "environment_json", "updated_at",
            ]
            rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            record = dict(zip(columns, row))
            for key in ("shape_json", "features_json", "competition_json",
                        "run_ids_json", "source_classes_json", "protocol_json",
                        "environment_json"):
                record[key.removesuffix("_json")] = json.loads(record.pop(key))
            result.append(record)
        return result

    def as_calibration_store(self, *, include_heldout: bool = False) -> CalibrationStore:
        observations = []
        for record in self.statistics(split=None if include_heldout else "train"):
            if int(record["calibratable_count"]) <= 0:
                continue
            if include_heldout and record["split"] != "train":
                pass
            observations.append(CalibrationObservation(
                op_family=record["op_family"], phase=record["phase"],
                recipe=record["recipe"],
                shape=tuple(sorted((str(k), float(v)) for k, v in record["shape"].items())),
                duration_ms=float(record["duration_median"]),
                analytic_ms=float(record["analytic_median"]),
                source_class="measured", source="sqlite_calibration_statistics",
                revision="", machine_key=record["hardware_key"],
                synthetic=False,
                metadata=tuple(sorted((str(k), str(v)) for k, v in record["competition"].items())),
                dtype=record["dtype"], layout=record["layout"],
                features=tuple(sorted((str(k), str(v)) for k, v in record["features"].items())),
                timing_scope=record["timing_scope"],
                protocol=json.dumps(
                    canonical_protocol(record.get("protocol") or {}),
                    sort_keys=True, default=str,
                ),
            ))
        return CalibrationStore(observations)

    def inventory(self) -> dict[str, Any]:
        from .a3_calibration_report import inventory_report

        return inventory_report(self)

    def coverage(self, queries: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        from .a3_calibration_report import coverage_report

        return coverage_report(self, queries=queries)

    # -- measurement requests -------------------------------------------
    def record_measurement_request(
        self,
        *,
        op_family: str,
        phase: str,
        recipe: str,
        shape: Mapping[str, Any],
        dtype: str = "",
        layout: str = "",
        hardware_key: str = "",
        features: Mapping[str, Any] | None = None,
        competition: Mapping[str, Any] | None = None,
        predicted_ms: float | None = None,
        analytic_ms: float | None = None,
        error_ratio: float = 0.0,
        extrapolation: float = 0.0,
        target_impact: float = 0.0,
        reason: str = "",
        run_id: str = "",
        details: Mapping[str, Any] | None = None,
        protocol: Mapping[str, Any] | None = None,
        timing_scope: str = "",
        environment: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not str(op_family).strip():
            raise A3InputError("calibration.op_family_required")
        shape_json = _shape_field(shape, "calibration.shape_invalid")
        features_json = _optional_mapping_field(features, "calibration.features_invalid")
        competition_json = _optional_mapping_field(
            competition, "calibration.competition_invalid",
        )
        protocol_json = canonical_protocol(
            _optional_mapping_field(protocol, "calibration.protocol_invalid")
        )
        timing_scope_text = canonical_timing_scope(timing_scope)
        environment_json = _optional_mapping_field(
            environment, "calibration.environment_invalid",
        )
        details_json = _optional_mapping_field(details, "calibration.details_invalid")
        if protocol_json:
            details_json["protocol"] = protocol_json
        if timing_scope_text:
            details_json["timing_scope"] = timing_scope_text
        if environment_json:
            details_json["environment"] = environment_json
        payload = {
            "hardware_key": hardware_key, "op_family": op_family, "phase": phase,
            "recipe": recipe, "shape": shape_json, "dtype": dtype, "layout": layout,
            "features": features_json, "competition": competition_json,
            "protocol": protocol_json, "timing_scope": timing_scope_text,
        }
        key = _semantic_key(payload)
        now = _utc_now()
        priority = (1.0 + max(0.0, float(error_ratio))) * (
            1.0 + max(0.0, float(extrapolation))
        ) * (1.0 + max(0.0, float(target_impact)))
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT request_id, query_count, priority, status FROM measurement_requests WHERE request_key = ?",
                (key,),
            ).fetchone()
            if existing is not None:
                request_id = int(existing[0])
                query_count = int(existing[1]) + 1
                new_priority = max(float(existing[2]), priority) + priority / query_count
                connection.execute(
                    """UPDATE measurement_requests
                       SET query_count = ?, priority = ?, error_ratio = MAX(error_ratio, ?),
                           extrapolation = MAX(extrapolation, ?), target_impact = MAX(target_impact, ?),
                           reason = CASE WHEN ? != '' THEN ? ELSE reason END,
                           details_json = CASE WHEN ? != '{}' THEN ? ELSE details_json END,
                           analytic_ms = COALESCE(analytic_ms, ?),
                           updated_at = ?
                       WHERE request_key = ?""",
                    (
                        query_count, new_priority, max(0.0, float(error_ratio)),
                        max(0.0, float(extrapolation)), max(0.0, float(target_impact)),
                        str(reason), str(reason), _json(details_json),
                        _json(details_json),
                        float(analytic_ms) if analytic_ms is not None else None,
                        now, key,
                    ),
                )
                return {
                    "status": "deduplicated", "request_key": key, "request_id": request_id,
                    "query_count": query_count, "priority": new_priority,
                    "request_status": str(existing[3]),
                }
            cursor = connection.execute(
                """INSERT INTO measurement_requests(
                    request_key, hardware_key, op_family, phase, recipe, shape_json,
                    dtype, layout, features_json, competition_json, status, run_id,
                    priority, query_count, predicted_ms, analytic_ms, error_ratio,
                    extrapolation, target_impact, reason, details_json, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    key, hardware_key, op_family, phase, recipe, _json(shape_json),
                    dtype, layout, _json(features_json), _json(competition_json),
                    "pending", run_id, priority, 1,
                    float(predicted_ms) if predicted_ms is not None else None,
                    float(analytic_ms) if analytic_ms is not None else None,
                    max(0.0, float(error_ratio)), max(0.0, float(extrapolation)),
                    max(0.0, float(target_impact)), reason or str(details_json or ""),
                    _json(details_json), now, now,
                ),
            )
            request_id = int(cursor.lastrowid)
        return {"status": "created", "request_key": key, "request_id": request_id,
                "query_count": 1, "priority": priority, "request_status": "pending"}

    def requests(self, *, status: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        query = "SELECT " + ", ".join(_REQUEST_COLUMNS) + " FROM measurement_requests"
        params: list[Any] = []
        if status is not None:
            if status not in STATUSES:
                raise A3InputError(f"calibration.status_invalid:{status}")
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY priority DESC, request_id ASC LIMIT ?"
        params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        result = []
        for row in rows:
            record = dict(zip(_REQUEST_COLUMNS, row))
            for key in ("shape_json", "features_json", "competition_json",
                        "details_json", "attempt_history_json"):
                record[key.removesuffix("_json")] = json.loads(record.pop(key))
            record["synthetic_allowed"] = False
            result.append(record)
        return result

    def _request_record(self, request_key: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT " + ", ".join(_REQUEST_COLUMNS)
                + " FROM measurement_requests WHERE request_key = ?",
                (request_key,),
            ).fetchone()
        if row is None:
            raise A3InputError(f"calibration.request_unknown:{request_key}")
        record = dict(zip(_REQUEST_COLUMNS, row))
        for key in ("shape_json", "features_json", "competition_json", "details_json"):
            record[key.removesuffix("_json")] = json.loads(record.pop(key))
        return record

    def _append_attempt(
        self, connection: sqlite3.Connection, request_key: str, status: str,
        run_id: str, reason: str,
    ) -> None:
        row = connection.execute(
            "SELECT attempt_count, attempt_history_json FROM measurement_requests WHERE request_key = ?",
            (request_key,),
        ).fetchone()
        if row is None:
            raise A3InputError(f"calibration.request_unknown:{request_key}")
        history = json.loads(row[1] or "[]")
        history.append({
            "attempt": int(row[0]) + 1, "status": status,
            "run_id": run_id, "reason": reason, "at": _utc_now(),
        })
        connection.execute(
            """UPDATE measurement_requests
               SET attempt_count = ?, attempt_history_json = ? WHERE request_key = ?""",
            (int(row[0]) + 1, _json(history[-50:]), request_key),
        )

    def update_request_status(
        self, request_key: str, *, status: str, run_id: str = "", reason: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status not in STATUSES:
            raise A3InputError(f"calibration.status_invalid:{status}")
        details_json = _optional_mapping_field(details, "calibration.details_invalid")
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE measurement_requests
                   SET status = ?, run_id = CASE WHEN ? != '' THEN ? ELSE run_id END,
                       reason = CASE WHEN ? != '' THEN ? ELSE reason END,
                       details_json = CASE WHEN ? != '{}' THEN ? ELSE details_json END,
                       updated_at = ?
                   WHERE request_key = ?""",
                (status, run_id, run_id, reason, reason,
                 _json(details_json), _json(details_json), _utc_now(), request_key),
            )
            if cursor.rowcount == 0:
                raise A3InputError(f"calibration.request_unknown:{request_key}")
            self._append_attempt(connection, request_key, status, run_id, reason)
        return {"status": "updated", "request_key": request_key, "request_status": status}

    def requeue_request(
        self, request_key: str, *, run_id: str = "", reason: str = "explicit_retry",
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM measurement_requests WHERE request_key = ?", (request_key,)
            ).fetchone()
            if row is None:
                raise A3InputError(f"calibration.request_unknown:{request_key}")
            if row[0] == "completed":
                raise A3InputError("calibration.requeue_completed_request")
            connection.execute(
                """UPDATE measurement_requests
                   SET status = 'pending', run_id = CASE WHEN ? != '' THEN ? ELSE run_id END,
                       updated_at = ? WHERE request_key = ?""",
                (run_id, run_id, _utc_now(), request_key),
            )
            self._append_attempt(connection, request_key, "pending", run_id, reason)
        return {"status": "requeued", "request_key": request_key, "request_status": "pending"}

    def complete_request_samples(
        self,
        request_key: str,
        *,
        raw_samples_ms: Sequence[float],
        analytic_ms: float,
        source: str,
        source_class: str = "measured",
        run_id: str = "",
        revision: str = "",
        split: str = "train",
        features: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = self._request_record(request_key)
        details_json = _optional_mapping_field(details, "calibration.details_invalid")
        raw = [float(value) for value in raw_samples_ms]
        if not raw or any(value <= 0 for value in raw):
            raise A3InputError("raw_samples_invalid")
        samples = []
        for index, duration in enumerate(raw):
            sample = self.record_sample(
                op_family=record["op_family"], phase=record["phase"], recipe=record["recipe"],
                shape=record["shape"], duration_ms=duration, analytic_ms=analytic_ms,
                dtype=record["dtype"], layout=record["layout"],
                hardware_key=record["hardware_key"],
                features={**record["features"], **dict(features or {})},
                competition=record["competition"], source=source,
                source_class=source_class, revision=revision, run_id=run_id,
                case_id=str(details_json.get("case_id", request_key)),
                repeat_index=index,
                split=split, synthetic=False,
                timing_scope=str(details_json.get("timing_scope", "")),
                protocol=details_json.get("protocol") if isinstance(details_json.get("protocol"), Mapping) else {},
                environment=details_json.get("environment") if isinstance(details_json.get("environment"), Mapping) else {},
                details=details_json,
            )
            samples.append(sample)
        with self._connect() as connection:
            connection.execute(
                """UPDATE measurement_requests
                   SET status = 'completed', sample_id = ?,
                       run_id = CASE WHEN ? != '' THEN ? ELSE run_id END,
                       details_json = ?, updated_at = ?
                   WHERE request_key = ?""",
                (samples[-1].get("measurement_id"), run_id, run_id,
                 _json(details_json), _utc_now(), request_key),
            )
            self._append_attempt(connection, request_key, "completed", run_id, "")
        return {
            "status": "completed", "request_key": request_key,
            "samples": samples, "sample_count": len(samples),
            "details_saved": bool(details_json),
        }

    def complete_request(
        self,
        request_key: str,
        *,
        duration_ms: float,
        analytic_ms: float,
        source: str,
        source_class: str = "measured",
        run_id: str = "",
        revision: str = "",
        split: str = "train",
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.complete_request_samples(
            request_key, raw_samples_ms=[float(duration_ms)], analytic_ms=analytic_ms,
            source=source, source_class=source_class, run_id=run_id,
            revision=revision, split=split, details=details,
        )

    def fail_request(
        self, request_key: str, *, reason: str, run_id: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.update_request_status(
            request_key, status="failed", run_id=run_id, reason=reason, details=details,
        )

    # -- manifest / ingestion -------------------------------------------
    def manifest(self, *, limit: int = 32, status: str = "pending") -> dict[str, Any]:
        from .a3_calibration_report import manifest_report

        return manifest_report(self, limit=limit, status=status)

    def export_manifest(self, path: str | Path, *, limit: int = 32) -> dict[str, Any]:
        from .a3_calibration_report import export_manifest_report

        return export_manifest_report(self, path, limit=limit)

    def ingest_results(self, results: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, Any]:
        items = results.get("results", ()) if isinstance(results, Mapping) else results
        completed = []
        failed = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping) or not item.get("request_key"):
                failed.append({"index": index, "reason": "request_key_required"})
                continue
            key = str(item["request_key"])
            run_id = str(item.get("run_id", ""))
            if str(item.get("status", "completed")).lower() in {"failed", "error"} or (
                item.get("duration_ms") is None and not item.get("raw_samples_ms")
            ):
                reason = str(item.get("reason", "runner_failed"))
                try:
                    failed.append(self.fail_request(
                        key, reason=reason, run_id=run_id,
                        details=item.get("details") if isinstance(item.get("details"), Mapping) else None,
                    ))
                except A3InputError as exc:
                    failed.append({"index": index, "reason": getattr(exc, "reason", str(exc))})
                continue
            try:
                record = self._request_record(key)
                analytic = item.get("analytic_ms")
                if analytic is None:
                    analytic = record.get("analytic_ms")
                if analytic is None:
                    raise A3InputError("analytic_ms_required")
                raw = item.get("raw_samples_ms")
                raw_values = (
                    [float(value) for value in raw]
                    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw
                    else [float(item["duration_ms"])]
                )
                request_details = (
                    dict(record["details"])
                    if isinstance(record.get("details"), Mapping) else {}
                )
                item_details = (
                    dict(item["details"])
                    if isinstance(item.get("details"), Mapping) else {}
                )
                details = {**request_details, **item_details}
                for field in ("timing_scope", "protocol", "environment"):
                    value = item.get(field)
                    if value is None or value == "" or value == {}:
                        value = details.get(field)
                    if value is not None:
                        details[field] = value
                completed.append(self.complete_request_samples(
                    key, raw_samples_ms=raw_values, analytic_ms=float(analytic),
                    source=str(item.get("source", "external_runner")),
                    source_class=str(item.get("source_class", "measured")),
                    run_id=run_id, revision=str(item.get("revision", "")),
                    split=str(item.get("split", "train")), details=details,
                    features=item.get("features") if isinstance(item.get("features"), Mapping) else None,
                ))
            except (A3InputError, KeyError, TypeError, ValueError) as exc:
                reason = getattr(exc, "reason", str(exc))
                try:
                    failed.append(self.fail_request(key, reason=str(reason), run_id=run_id))
                except A3InputError as nested:
                    failed.append({"index": index, "reason": getattr(nested, "reason", str(nested))})
        return {"status": "ingested", "completed": completed, "failed": failed}


__all__ = [
    "STATUSES", "SampleRecord", "SQLiteCalibrationStore",
]
