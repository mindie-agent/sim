"""SQLite raw-measurement store, coverage, and re-measurement loop.

[Role]
  Persist every raw measurement repeat (run/case/repeat), aggregate condition
  statistics for calibration, deduplicate missing-measurement requests, expose
  shape/feature coverage, and produce an executable manifest.  Ingest keeps
  train and heldout split labels separate.

[Estimator contract]
  The calibration target of a condition is ``duration_trial_median`` =
  ``median over trials of (median over that trial's raw repeats)``.  A trial is
  one runner invocation, identified by ``(run_id, case_id)``; a repeat without a
  ``case_id`` is its own single-repeat trial.  The acceptance scorer scores
  exactly this statistic (``median of trial medians``), so the pooled median
  over every repeat (``duration_median``) is retained as a *diagnostic* only and
  exported next to the target as ``pooled_over_target``.  The two differ
  whenever trials disagree or carry unequal repeat counts; a fit against the
  pooled median is fitted to a different functional than the score.

[Invariants]
  * Only ``source_class in {measured, observed}``, ``synthetic = false``,
    non-empty ``source``, positive duration/analytic and ``split = train``
    repeats may calibrate.  Heldout is write-only (leakage rule L5):
    ``as_calibration_store(include_heldout=True)`` is refused outright.
  * Independent measurements at the same semantic condition are never dropped:
    only an explicitly identified run/case/repeat is idempotent.
  * ``sample_key`` includes hardware, op family, phase, recipe, shape, dtype,
    layout, features, competition, protocol, environment, timing scope and
    split.  Revision, source, die and visible devices are provenance, not
    semantics.
  * An observation with an empty hardware key never responds to a query that
    names a machine (audit blocker B7).  ``MachineBoundCalibrationStore``
    refuses that wildcard and reports the refusal reason.
  * Provenance survives ingestion: run id, case id, repeat ordinal, die,
    visible devices, protocol signature, timing scope, source class and the raw
    repeat value are columns of ``measurements``.
  * Pending request deduplication is separate from measurement deduplication.
  * Heldout rows stay heldout and are excluded from the default calibration
    view.
  * Status is pending/running/completed/failed; no NPU job/lease is created.
  * The schema version lives in ``PRAGMA user_version``.  Migrations are
    additive ``ALTER TABLE ... ADD COLUMN`` statements only; aggregates are
    recomputed from the unmodified measurement rows.

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
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .a3_cost import CalibrationObservation, CalibrationResult, CalibrationStore
from .a3_operator_curve_protocol import canonical_protocol, protocol_signature
from .a3_store_types import (
    SampleRecord, _json, _measurement_key, _percentile, _semantic_key,
    _utc_now, canonical_timing_scope,
)
from .a3_types import A3InputError

STATUSES = ("pending", "running", "completed", "failed")
#: ``PRAGMA user_version`` of the current contract; additive migrations only.
SCHEMA_VERSION = 1
#: Explicit "not attested" state for provenance columns, never a wildcard.
UNKNOWN_IDENTITY = "unknown"
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


def _identity_text(value: Any) -> str:
    """Normalize an attested identity field; absence is ``""``, never a guess."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _attested(*values: Any) -> str:
    """First non-empty attestation, else the explicit ``unknown`` state."""
    for value in values:
        text = _identity_text(value)
        if text:
            return text
    return UNKNOWN_IDENTITY


def _protocol_signature(protocol: Mapping[str, Any] | None) -> str:
    """Canonical protocol signature (same helper the curve layer groups on)."""
    canonical = canonical_protocol(protocol)
    return protocol_signature(canonical) if canonical else ""


def _calibratable_repeat(item: Mapping[str, Any]) -> bool:
    """Leakage rule L5: measured/observed, attributable, non-synthetic, positive."""
    try:
        duration = float(item["duration_ms"])
        analytic = float(item["analytic_ms"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        str(item.get("source_class", "")) in {"measured", "observed"}
        and not item.get("synthetic")
        and bool(str(item.get("source", "")).strip())
        and math.isfinite(duration) and duration > 0
        and math.isfinite(analytic) and analytic > 0
    )


def _trial_medians(calibratable: Sequence[Mapping[str, Any]]) -> list[float]:
    """One trial = one runner invocation, keyed by ``(run_id, case_id)``.

    Raw repeats that share a case id are events of the same invocation and never
    become independent trials (leakage rule L7); repeats without a case id are
    independent single-repeat trials.
    """
    groups: dict[tuple[str, str], list[float]] = {}
    for item in calibratable:
        case_id = str(item.get("case_id", "") or "")
        trial = (
            (str(item.get("run_id", "") or ""), case_id)
            if case_id else ("", f"#{int(item['measurement_id'])}")
        )
        groups.setdefault(trial, []).append(float(item["duration_ms"]))
    medians = [_percentile(values, 0.5) for values in groups.values()]
    return [float(value) for value in medians if value is not None]


class MachineBoundCalibrationStore(CalibrationStore):
    """Calibration view whose machine identity is an exact join, never a wildcard.

    A store row imported without a hardware key must not calibrate a machine it
    was never measured on (audit blocker B7).  The view keeps every observation
    for inspection, but a query that names a machine only sees observations
    keyed to that same machine.  Refused observations are reported in the result
    evidence with an explicit ``hardware_key_unbound`` / ``hardware_key_mismatch``
    reason.  The store owns this guarantee rather than relying on the caller or
    on the cost matcher's current filtering.
    """

    def __init__(
        self,
        observations: Sequence[CalibrationObservation] = (),
        *,
        rejections: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(observations)
        self._rejections = tuple(dict(item) for item in rejections)

    def rejections(self) -> tuple[dict[str, Any], ...]:
        """Conditions excluded when this view was built, each with its reason."""
        return self._rejections

    def factor_for(self, **query: Any) -> CalibrationResult:
        machine_key = str(query.get("machine_key") or "").strip()
        if not machine_key:
            return super().factor_for(**query)
        bound = tuple(
            item for item in self.observations() if item.machine_key == machine_key
        )
        result = CalibrationStore(bound).factor_for(**query)
        family = str(query.get("op_family", ""))
        phase = str(query.get("phase", ""))
        refused = [
            item for item in self.observations()
            if item.machine_key != machine_key
            and item.op_family == family and item.phase == phase
        ]
        if not refused:
            return result
        evidence = dict(result.evidence or {})
        evidence["machine_key"] = machine_key
        evidence["refused_observations"] = {
            "count": len(refused),
            "unbound": sum(1 for item in refused if not item.machine_key),
            "mismatched": sum(1 for item in refused if item.machine_key),
            "examples": tuple(
                {
                    "recipe": item.recipe, "machine_key": item.machine_key,
                    "reason": (
                        "hardware_key_unbound" if not item.machine_key
                        else "hardware_key_mismatch"
                    ),
                }
                for item in refused[:8]
            ),
        }
        return replace(result, evidence=evidence)


class SQLiteCalibrationStore:
    """WAL SQLite raw-sample store; each operation opens a bounded connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._memory_connection: sqlite3.Connection | None = None
        self.schema_upgraded = False
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

    def _create_tables(self, connection: sqlite3.Connection) -> None:
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
                created_at TEXT NOT NULL,
                die TEXT NOT NULL DEFAULT 'unknown',
                visible_devices TEXT NOT NULL DEFAULT 'unknown',
                protocol_signature TEXT NOT NULL DEFAULT ''
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
                updated_at TEXT NOT NULL,
                duration_trial_median REAL,
                trial_count INTEGER NOT NULL DEFAULT 0,
                protocol_signature TEXT NOT NULL DEFAULT '',
                die_values_json TEXT NOT NULL DEFAULT '[]',
                visible_devices_json TEXT NOT NULL DEFAULT '[]'
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
        self._migrate(connection)

    # Columns added after the first frozen round-1 schema.  Append-only: the
    # migration never rewrites or renames an existing column, and existing rows
    # keep their measurement/sample keys.
    _SCHEMA_MIGRATIONS: Mapping[str, Mapping[str, str]] = {
        "measurements": {
            "die": "TEXT NOT NULL DEFAULT 'unknown'",
            "visible_devices": "TEXT NOT NULL DEFAULT 'unknown'",
            "protocol_signature": "TEXT NOT NULL DEFAULT ''",
        },
        "sample_statistics": {
            "duration_trial_median": "REAL",
            "trial_count": "INTEGER NOT NULL DEFAULT 0",
            "protocol_signature": "TEXT NOT NULL DEFAULT ''",
            "die_values_json": "TEXT NOT NULL DEFAULT '[]'",
            "visible_devices_json": "TEXT NOT NULL DEFAULT '[]'",
        },
    }

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Upgrade an older DB in place; aggregates are recomputed, rows untouched."""
        version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
        if version > SCHEMA_VERSION:
            raise A3InputError(
                f"calibration.schema_too_new:{version}>{SCHEMA_VERSION}"
            )
        if version == SCHEMA_VERSION:
            return
        added = 0
        provenance_backfill = False
        for table, columns in self._SCHEMA_MIGRATIONS.items():
            present = {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for name, declaration in columns.items():
                if name in present:
                    continue
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                )
                added += 1
                if table == "measurements" and name in {
                    "die", "visible_devices", "protocol_signature",
                }:
                    provenance_backfill = True
        if provenance_backfill:
            self._backfill_provenance(connection)
        if added:
            self.schema_upgraded = True
            # Trial medians and provenance aggregates are derived state: rebuild
            # them from the measurement rows so an upgraded DB is immediately
            # usable, without touching a single raw sample.
            keys = connection.execute(
                "SELECT DISTINCT sample_key FROM measurements ORDER BY sample_key"
            ).fetchall()
            for (key,) in keys:
                self._recompute_statistics(connection, str(key))
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _backfill_provenance(connection: sqlite3.Connection) -> None:
        """Fill newly added provenance columns from each row's retained evidence.

        Only the added columns are written; measured values, keys, splits and
        timestamps are never touched (no destructive rewrite).  An attestation
        already present in the row wins over the evidence fallback.
        """
        updates: list[tuple[str, str, str, int]] = []
        for row in connection.execute(
            """SELECT measurement_id, die, visible_devices, protocol_signature,
                      protocol_json, environment_json, details_json
               FROM measurements"""
        ):
            protocol_json = json.loads(row[4] or "{}")
            environment_json = json.loads(row[5] or "{}")
            details_json = json.loads(row[6] or "{}")
            die = _attested(
                "" if row[1] in {"", UNKNOWN_IDENTITY} else row[1],
                environment_json.get("die"), environment_json.get("die_id"),
                protocol_json.get("die"), details_json.get("die"),
            )
            visible = _attested(
                "" if row[2] in {"", UNKNOWN_IDENTITY} else row[2],
                environment_json.get("visible_devices"),
                details_json.get("visible_devices"),
            )
            signature = str(row[3] or "") or _protocol_signature(protocol_json)
            if (die, visible, signature) == (row[1], row[2], row[3]):
                continue
            updates.append((die, visible, signature, int(row[0])))
        if updates:
            connection.executemany(
                """UPDATE measurements
                   SET die = ?, visible_devices = ?, protocol_signature = ?
                   WHERE measurement_id = ?""",
                updates,
            )

    def schema_info(self) -> dict[str, Any]:
        """Current schema version and the additive columns it guarantees."""
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0] or 0)
            columns = {
                table: [str(row[1]) for row in connection.execute(
                    f"PRAGMA table_info({table})"
                )]
                for table in self._SCHEMA_MIGRATIONS
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "user_version": version,
            "upgraded_from_legacy_schema": bool(getattr(self, "schema_upgraded", False)),
            "columns": columns,
        }

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
        die: str = "",
        visible_devices: str = "",
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
        # Provenance, not semantics: an unattested die/device set is stored as
        # the explicit "unknown" state and never enters ``sample_key``, so a
        # missing attestation cannot fork or wildcard a measured condition.
        die_value = _attested(
            die, environment_json.get("die"), environment_json.get("die_id"),
            protocol_json.get("die"), details_json.get("die"),
        )
        visible_value = _attested(
            visible_devices, environment_json.get("visible_devices"),
            details_json.get("visible_devices"),
        )
        signature = _protocol_signature(protocol_json)
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
                    protocol_json, environment_json, details_json, created_at,
                    die, visible_devices, protocol_signature
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mkey, key, hardware_key, op_family, phase, recipe, _json(shape_json),
                    dtype, layout, _json(features_json), _json(competition_json),
                    duration, analytic, source, source_class, int(synthetic),
                    revision, run_id, case_id, int(repeat_index), split, timing_scope,
                    _json(protocol_json), _json(environment_json), _json(details_json), now,
                    die_value, visible_value, signature,
                ),
            )
            measurement_id = int(cursor.lastrowid)
            self._recompute_statistics(connection, key)
        return {"status": "inserted", "sample_key": key, "measurement_key": mkey,
                "measurement_id": measurement_id}

    _STAT_MEASUREMENT_COLUMNS = (
        "measurement_id", "duration_ms", "analytic_ms", "source",
        "source_class", "synthetic", "run_id", "case_id", "timing_scope",
        "protocol_json", "environment_json", "die", "visible_devices",
    )
    _SAMPLE_COLUMNS = (
        "measurement_id", "measurement_key", "sample_key", "hardware_key",
        "op_family", "phase", "recipe", "shape_json", "dtype", "layout",
        "features_json", "competition_json", "duration_ms", "analytic_ms",
        "source", "source_class", "synthetic", "revision", "run_id", "case_id",
        "repeat_index", "split", "timing_scope", "protocol_json",
        "environment_json", "details_json", "created_at", "die",
        "visible_devices", "protocol_signature",
    )
    _STATISTICS_COLUMNS = (
        "sample_key", "hardware_key", "op_family", "phase", "recipe",
        "shape_json", "dtype", "layout", "features_json", "competition_json",
        "split", "measurement_count", "calibratable_count", "run_count",
        "run_ids_json", "duration_median", "duration_mean", "duration_min",
        "duration_max", "duration_p90", "duration_std", "analytic_median",
        "source_classes_json", "synthetic_count", "timing_scope",
        "protocol_json", "environment_json", "updated_at",
        "duration_trial_median", "trial_count", "protocol_signature",
        "die_values_json", "visible_devices_json",
    )
    _JSON_COLUMNS = (
        "shape", "features", "competition", "run_ids", "source_classes",
        "protocol", "environment", "die_values", "visible_devices",
    )

    @classmethod
    def _recompute_statistics(cls, connection: sqlite3.Connection, sample_key: str) -> None:
        identity = connection.execute(
            """SELECT hardware_key, op_family, phase, recipe, shape_json, dtype,
                      layout, features_json, competition_json, split
               FROM measurements WHERE sample_key = ? LIMIT 1""",
            (sample_key,),
        ).fetchone()
        if identity is None:
            return
        rows = [
            dict(zip(cls._STAT_MEASUREMENT_COLUMNS, item))
            for item in connection.execute(
                "SELECT " + ", ".join(cls._STAT_MEASUREMENT_COLUMNS)
                + " FROM measurements WHERE sample_key = ? ORDER BY measurement_id",
                (sample_key,),
            ).fetchall()
        ]
        calibratable = [item for item in rows if _calibratable_repeat(item)]
        durations = [float(item["duration_ms"]) for item in calibratable]
        analytics = [float(item["analytic_ms"]) for item in calibratable]
        # Calibration target = median over trials of per-trial repeat medians.
        # The pooled median stays as the diagnostic column ``duration_median``.
        trials = _trial_medians(calibratable)
        run_ids = sorted({
            str(item["run_id"]) for item in calibratable if item["run_id"]
        })
        source_classes = sorted({str(item["source_class"]) for item in rows})
        protocol = (
            canonical_protocol(json.loads(calibratable[-1]["protocol_json"]))
            if calibratable else {}
        )
        environment = json.loads(calibratable[-1]["environment_json"]) if calibratable else {}
        timing_scope = str(calibratable[-1]["timing_scope"]) if calibratable else ""
        die_values = sorted({_attested(item["die"]) for item in rows})
        visible_values = sorted({
            _attested(item["visible_devices"]) for item in rows
        })
        updated = _utc_now()
        connection.execute(
            """INSERT INTO sample_statistics(
                sample_key, hardware_key, op_family, phase, recipe, shape_json,
                dtype, layout, features_json, competition_json, split,
                measurement_count, calibratable_count, run_count, run_ids_json,
                duration_median, duration_mean, duration_min, duration_max,
                duration_p90, duration_std, analytic_median, source_classes_json,
                synthetic_count, timing_scope, protocol_json, environment_json,
                updated_at, duration_trial_median, trial_count, protocol_signature,
                die_values_json, visible_devices_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                environment_json=excluded.environment_json, updated_at=excluded.updated_at,
                duration_trial_median=excluded.duration_trial_median,
                trial_count=excluded.trial_count,
                protocol_signature=excluded.protocol_signature,
                die_values_json=excluded.die_values_json,
                visible_devices_json=excluded.visible_devices_json""",
            (
                sample_key, identity[0], identity[1], identity[2], identity[3],
                identity[4], identity[5], identity[6], identity[7], identity[8],
                identity[9], len(rows), len(calibratable), len(run_ids),
                _json(run_ids), _percentile(durations, 0.5),
                statistics.fmean(durations) if durations else None,
                min(durations) if durations else None,
                max(durations) if durations else None,
                _percentile(durations, 0.9),
                statistics.pstdev(durations) if len(durations) > 1 else 0.0,
                _percentile(analytics, 0.5), _json(source_classes),
                sum(1 for item in rows if item["synthetic"]), timing_scope,
                _json(protocol), _json(environment), updated,
                _percentile(trials, 0.5), len(trials),
                _protocol_signature(protocol), _json(die_values),
                _json(visible_values),
            ),
        )

    def sample_rows(self, *, split: str | None = None) -> list[dict[str, Any]]:
        """Every stored sample row plus its provenance columns.

        ``SampleRecord`` predates the provenance columns; this view keeps die,
        visible devices and the protocol signature alongside the raw repeat so
        an import can be audited without re-reading the source JSON.
        """
        query = "SELECT " + ", ".join(self._SAMPLE_COLUMNS) + " FROM measurements"
        params: tuple[Any, ...] = ()
        if split is not None:
            query += " WHERE split = ?"
            params = (split,)
        query += " ORDER BY measurement_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            record = dict(zip(self._SAMPLE_COLUMNS, row))
            for key in ("shape", "features", "competition", "protocol",
                        "environment", "details"):
                record[key] = json.loads(record.pop(key + "_json"))
            result.append(record)
        return result

    def samples(self, *, split: str | None = None) -> list[SampleRecord]:
        return [
            SampleRecord(
                id=int(row["measurement_id"]),
                measurement_key=str(row["measurement_key"]),
                sample_key=str(row["sample_key"]),
                hardware_key=str(row["hardware_key"]),
                op_family=str(row["op_family"]), phase=str(row["phase"]),
                recipe=str(row["recipe"]), shape=row["shape"],
                dtype=str(row["dtype"]), layout=str(row["layout"]),
                features=row["features"], competition=row["competition"],
                duration_ms=float(row["duration_ms"]),
                analytic_ms=float(row["analytic_ms"]),
                source=str(row["source"]), source_class=str(row["source_class"]),
                synthetic=bool(row["synthetic"]), revision=str(row["revision"]),
                run_id=str(row["run_id"]), case_id=str(row["case_id"]),
                repeat_index=int(row["repeat_index"]), split=str(row["split"]),
                timing_scope=str(row["timing_scope"]),
                protocol=row["protocol"], environment=row["environment"],
                details=row["details"],
            )
            for row in self.sample_rows(split=split)
        ]

    def statistics(self, *, split: str | None = "train") -> list[dict[str, Any]]:
        query = "SELECT " + ", ".join(self._STATISTICS_COLUMNS) + " FROM sample_statistics"
        params: tuple[Any, ...] = ()
        if split is not None:
            query += " WHERE split = ?"
            params = (split,)
        query += " ORDER BY op_family, phase, sample_key"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            record = dict(zip(self._STATISTICS_COLUMNS, row))
            for key in self._JSON_COLUMNS:
                record[key] = json.loads(record.pop(key + "_json"))
            target = record["duration_trial_median"]
            if target is None:
                # Pre-trial-median aggregate: never silently keep fitting the
                # pooled statistic, fall back only to report the target as the
                # pooled median and let the estimator ratio stay 1.0.
                target = record["duration_median"]
            pooled = record["duration_median"]
            record["calibration_target_ms"] = target
            record["pooled_over_target"] = (
                float(pooled) / float(target)
                if pooled is not None and target not in (None, 0) else None
            )
            result.append(record)
        return result

    def as_calibration_store(
        self, *, include_heldout: bool = False, hardware_key: str = "",
    ) -> CalibrationStore:
        """Build the fitting view; heldout is refused, never silently included.

        ``hardware_key`` binds the view to one machine: unbound or
        differently-keyed conditions are excluded with an explicit rejection
        reason instead of wildcarding every machine (audit blocker B7).
        """
        if include_heldout:
            # Leakage rule L5: heldout outcomes are write-only.  The previous
            # ``pass`` here silently put them into the fit store.
            raise A3InputError("calibration.heldout_fit_forbidden")
        scope = str(hardware_key or "").strip()
        observations: list[CalibrationObservation] = []
        rejections: list[dict[str, Any]] = []
        for record in self.statistics(split=None):
            if record["split"] != "train":
                reason = "heldout_excluded_from_fit"
            elif int(record["calibratable_count"]) <= 0:
                reason = "no_calibratable_repeats"
            elif record["calibration_target_ms"] is None:
                reason = "calibration_target_missing"
            elif not str(record["hardware_key"]).strip():
                reason = "hardware_key_unbound" if scope else ""
            elif scope and str(record["hardware_key"]) != scope:
                reason = "hardware_key_mismatch"
            else:
                reason = ""
            if reason:
                rejections.append({
                    "sample_key": record["sample_key"],
                    "op_family": record["op_family"], "phase": record["phase"],
                    "recipe": record["recipe"], "split": record["split"],
                    "hardware_key": record["hardware_key"], "reason": reason,
                })
                continue
            observations.append(CalibrationObservation(
                op_family=record["op_family"], phase=record["phase"],
                recipe=record["recipe"],
                shape=tuple(sorted((str(k), float(v)) for k, v in record["shape"].items())),
                duration_ms=float(record["calibration_target_ms"]),
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
        return MachineBoundCalibrationStore(observations, rejections=rejections)

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
    "MachineBoundCalibrationStore", "SCHEMA_VERSION", "STATUSES", "SampleRecord",
    "SQLiteCalibrationStore", "UNKNOWN_IDENTITY",
]
