"""Tier-0 correctness tests for the SQLite calibration store.

Scope: the store contract only (pure SQLite, no device/network/subprocess):

* heldout is write-only and can never enter a fitting view (leakage rule L5);
* an unbound hardware key never wildcards a machine-named query (audit B7);
* the calibration target is ``median over trials of trial medians``, with the
  pooled repeat median retained as a diagnostic (``metrics.md`` §1.1);
* re-importing the same run/case/repeat is idempotent while different runs at
  the same condition are retained (L7);
* an old round-1 schema is upgraded additively and its aggregates are rebuilt
  from unmodified rows;
* per-row provenance (run/die/visible devices/protocol signature/timing scope/
  source class/raw repeat) survives a reopen.

The trial rows in these tests are explicit fixtures.  The migration tests use
the frozen round-1 training DB when it is mounted and a byte-identical legacy
schema (captured from that DB) when it is not; no measurement is invented.
"""
from __future__ import annotations

import sqlite3
import statistics
from pathlib import Path

import pytest

from mindie_sim.a3_calibration_store import (
    SCHEMA_VERSION, MachineBoundCalibrationStore, SQLiteCalibrationStore,
)
from mindie_sim.a3_operator_curve_protocol import protocol_signature
from mindie_sim.a3_types import A3InputError

ROUND1_DB = Path(
    "/Volumes/data/mindie-sim-data/a3/2026-10-20/"
    "independent-generalization-20261021/generic-training.sqlite"
)

CONDITION = {"op_family": "gemm", "phase": "operator", "shape": {"m": 8, "n": 16, "k": 16}}


def _record(store, *, recipe="r", hardware_key="Ascend910B4", run_id="run-1",
            case_id="", repeat_index=0, duration=1.0, analytic=1.0,
            source="tier0-fixture", source_class="measured", split="train",
            protocol=None, environment=None, details=None, synthetic=False,
            die="", visible_devices=""):
    return store.record_sample(
        **CONDITION, recipe=recipe, duration_ms=duration, analytic_ms=analytic,
        hardware_key=hardware_key, run_id=run_id, case_id=case_id,
        repeat_index=repeat_index, source=source, source_class=source_class,
        split=split, protocol=protocol, environment=environment,
        details=details, synthetic=synthetic, die=die,
        visible_devices=visible_devices,
    )


def _query(**overrides):
    query = {
        "op_family": "gemm", "phase": "operator", "recipe": "r",
        "shape": {"m": 8, "n": 16, "k": 16},
    }
    query.update(overrides)
    return query


# ---------------------------------------------------------------------------
# Heldout and unattributable rows never calibrate
# ---------------------------------------------------------------------------


def test_heldout_is_refused_with_an_explicit_reason():
    store = SQLiteCalibrationStore(":memory:")
    _record(store, recipe="train", duration=2.0)
    _record(store, recipe="heldout", duration=9.0, split="heldout", run_id="run-h")

    view = store.as_calibration_store()
    assert [observation.recipe for observation in view.observations()] == ["train"]
    rejections = view.rejections()
    assert len(rejections) == 1
    assert {
        key: rejections[0][key]
        for key in ("op_family", "phase", "recipe", "split", "hardware_key", "reason")
    } == {
        "op_family": "gemm", "phase": "operator", "recipe": "heldout",
        "split": "heldout", "hardware_key": "Ascend910B4",
        "reason": "heldout_excluded_from_fit",
    }

    # The previous no-op ``pass`` silently returned a store containing heldout.
    with pytest.raises(A3InputError, match="calibration.heldout_fit_forbidden"):
        store.as_calibration_store(include_heldout=True)


def test_unattributable_and_derived_rows_never_calibrate():
    store = SQLiteCalibrationStore(":memory:")
    _record(store, recipe="derived", source_class="derived")
    _record(store, recipe="synthetic", source_class="synthetic", synthetic=True)
    _record(store, recipe="unattributed", source_class="unknown")
    _record(store, recipe="measured", duration=2.0)

    stats = {record["recipe"]: record for record in store.statistics()}
    for recipe in ("derived", "synthetic", "unattributed"):
        assert stats[recipe]["calibratable_count"] == 0
    assert stats["measured"]["calibratable_count"] == 1
    inventory = store.inventory()
    assert inventory["calibratable_measurements"] == 1
    assert inventory["calibratable_conditions"] == 1
    assert inventory["provenance"]["attributable"] == 1

    view = store.as_calibration_store()
    assert [observation.recipe for observation in view.observations()] == ["measured"]
    rejections = {item["recipe"]: item["reason"] for item in view.rejections()}
    assert rejections == {
        "derived": "no_calibratable_repeats",
        "synthetic": "no_calibratable_repeats",
        "unattributed": "no_calibratable_repeats",
    }
    # A measured row without a source is rejected at ingestion, not silently kept.
    with pytest.raises(A3InputError, match="calibration.measured_source_required"):
        _record(store, recipe="no-source", source="", source_class="measured")


# ---------------------------------------------------------------------------
# Machine identity: an unbound key is not a wildcard
# ---------------------------------------------------------------------------


def test_unbound_machine_key_never_wildcards_a_keyed_query():
    store = SQLiteCalibrationStore(":memory:")
    # Same physical condition on three machines, one of them unattributed.
    _record(store, hardware_key="", duration=4.0)
    _record(store, hardware_key="Ascend910_9362", duration=2.0)
    _record(store, hardware_key="Ascend910B4", duration=1.5)

    view = store.as_calibration_store()
    assert isinstance(view, MachineBoundCalibrationStore)

    a2 = view.factor_for(machine_key="Ascend910B4", **_query())
    assert a2.status == "calibrated_exact"
    assert a2.factor == pytest.approx(1.5)
    refused = a2.evidence["refused_observations"]
    assert refused["count"] == 2
    assert refused["unbound"] == 1
    assert refused["mismatched"] == 1
    assert {example["reason"] for example in refused["examples"]} == {
        "hardware_key_unbound", "hardware_key_mismatch",
    }

    a3 = view.factor_for(machine_key="Ascend910_9362", **_query())
    assert a3.factor == pytest.approx(2.0)

    # No observation was measured on this machine: every candidate is refused
    # with its reason, and nothing is silently borrowed from a neighbour SKU.
    unknown_machine = view.factor_for(machine_key="Ascend910B5", **_query())
    assert unknown_machine.status.startswith("uncalibrated")
    assert unknown_machine.evidence["machine_key"] == "Ascend910B5"
    assert unknown_machine.evidence["refused_observations"]["count"] == 3
    assert unknown_machine.evidence["refused_observations"]["unbound"] == 1
    assert unknown_machine.evidence["refused_observations"]["mismatched"] == 2

    # A bound view already excludes the other machines with explicit reasons.
    bound = store.as_calibration_store(hardware_key="Ascend910B4")
    assert [observation.machine_key for observation in bound.observations()] == [
        "Ascend910B4"
    ]
    reasons = {item["sample_key"]: item["reason"] for item in bound.rejections()}
    assert sorted(reasons.values()) == ["hardware_key_mismatch", "hardware_key_unbound"]

    identity = store.inventory()["hardware_identity"]
    assert identity["unbound_conditions"] == 1
    assert identity["empty_key_is_wildcard"] is False
    assert {item["hardware_key"] for item in identity["keys"]} == {
        "", "Ascend910_9362", "Ascend910B4",
    }


# ---------------------------------------------------------------------------
# Estimator contract: median of trial medians, pooled median is diagnostic
# ---------------------------------------------------------------------------


def test_calibration_target_is_median_of_trial_medians():
    store = SQLiteCalibrationStore(":memory:")
    for index, duration in enumerate((1.0, 2.0, 3.0)):
        _record(store, recipe="two-trials", case_id="case-a",
                repeat_index=index, duration=duration)
    for index, duration in enumerate((10.0, 20.0)):
        _record(store, recipe="two-trials", case_id="case-b",
                repeat_index=index, duration=duration)
    # Same case id in a different run is a different trial, not a repeat.
    _record(store, recipe="cross-run", run_id="run-1", case_id="case-a",
            duration=1.0)
    _record(store, recipe="cross-run", run_id="run-2", case_id="case-a",
            duration=9.0)
    # No case id at all: each raw repeat is its own single-repeat trial.
    for index, duration in enumerate((1.0, 2.0, 3.0, 10.0, 20.0)):
        _record(store, recipe="no-trial-id", run_id="run-1",
                repeat_index=index, duration=duration)

    stats = {record["recipe"]: record for record in store.statistics()}

    two = stats["two-trials"]
    trial_medians = [statistics.median((1.0, 2.0, 3.0)), statistics.median((10.0, 20.0))]
    assert two["trial_count"] == 2
    assert two["duration_trial_median"] == pytest.approx(statistics.median(trial_medians))
    assert two["duration_trial_median"] == pytest.approx(8.5)
    assert two["duration_median"] == pytest.approx(
        statistics.median((1.0, 2.0, 3.0, 10.0, 20.0))
    )
    assert two["duration_median"] == pytest.approx(3.0)
    assert two["calibration_target_ms"] == pytest.approx(8.5)
    assert two["pooled_over_target"] == pytest.approx(3.0 / 8.5)

    cross = stats["cross-run"]
    assert cross["trial_count"] == 2
    assert cross["duration_trial_median"] == pytest.approx(5.0)

    pooled_only = stats["no-trial-id"]
    assert pooled_only["trial_count"] == 5
    assert pooled_only["duration_trial_median"] == pytest.approx(
        pooled_only["duration_median"]
    )

    observations = {
        observation.recipe: observation
        for observation in store.as_calibration_store().observations()
    }
    # The fitted value is the trial median, never the pooled median.
    assert observations["two-trials"].duration_ms == pytest.approx(8.5)
    assert observations["two-trials"].analytic_ms == pytest.approx(1.0)

    inventory = store.inventory()
    assert inventory["estimator"]["calibration_target"].startswith("duration_trial_median")
    assert inventory["estimator"]["multi_trial_conditions"] == 3
    assert inventory["estimator"]["pooled_target_mismatch_conditions"] == 1
    assert inventory["estimator"]["pooled_over_target_min"] == pytest.approx(3.0 / 8.5)
    coverage = store.coverage()
    trials = sum(
        split["trials"] for group in coverage["groups"]
        for split in group["splits"].values()
    )
    assert trials == 2 + 2 + 5


# ---------------------------------------------------------------------------
# Idempotence and duplicate handling
# ---------------------------------------------------------------------------


def test_reingest_same_run_case_repeat_is_idempotent_and_runs_are_kept(tmp_path):
    store = SQLiteCalibrationStore(tmp_path / "calibration.sqlite")
    first = _record(store, run_id="run-a", case_id="case-1", repeat_index=0,
                    duration=2.0)
    again = _record(store, run_id="run-a", case_id="case-1", repeat_index=0,
                    duration=2.0)
    assert first["status"] == "inserted"
    assert again["status"] == "duplicate"
    assert again["measurement_key"] == first["measurement_key"]
    assert store.inventory()["measurements"] == 1

    # A different run at the same condition is an independent trial and is kept.
    second = _record(store, run_id="run-b", case_id="case-1", repeat_index=0,
                     duration=6.0)
    assert second["status"] == "inserted"
    reopened = SQLiteCalibrationStore(tmp_path / "calibration.sqlite")
    assert reopened.inventory()["measurements"] == 2
    stats = reopened.statistics()[0]
    assert stats["run_count"] == 2
    assert stats["run_ids"] == ["run-a", "run-b"]
    assert stats["calibration_target_ms"] == pytest.approx(4.0)

    # The manifest/scorer path is idempotent too: the same result item ingested
    # twice must not duplicate raw repeats.
    request = reopened.record_measurement_request(
        **CONDITION, recipe="r", analytic_ms=1.0, hardware_key="Ascend910B4",
    )
    result = {
        "request_key": request["request_key"], "raw_samples_ms": [1.0, 2.0, 3.0],
        "analytic_ms": 1.0, "source": "tier0-runner", "source_class": "measured",
        "run_id": "run-ingest", "split": "train",
        "details": {"case_id": "ingest-case"},
    }
    before = reopened.inventory()["measurements"]
    first_ingest = reopened.ingest_results({"results": [result]})
    after_first = reopened.inventory()["measurements"]
    second_ingest = reopened.ingest_results({"results": [result]})
    after_second = reopened.inventory()["measurements"]
    assert len(first_ingest["completed"]) == 1 and not first_ingest["failed"]
    assert len(second_ingest["completed"]) == 1 and not second_ingest["failed"]
    assert after_first == before + 3
    assert after_second == after_first
    assert all(
        sample["status"] == "duplicate"
        for sample in second_ingest["completed"][0]["samples"]
    )


# ---------------------------------------------------------------------------
# Schema migration: additive, versioned, aggregates rebuilt
# ---------------------------------------------------------------------------

# Captured verbatim from the frozen round-1 DB
# (/Volumes/data/mindie-sim-data/a3/2026-10-20/independent-generalization-20261021/
# generic-training.sqlite, ``PRAGMA user_version = 0``).
LEGACY_MEASUREMENTS_DDL = """
CREATE TABLE measurements (
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
)
"""
LEGACY_STATISTICS_DDL = """
CREATE TABLE sample_statistics (
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
)
"""
LEGACY_REQUESTS_DDL = """
CREATE TABLE measurement_requests (
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
)
"""


def _write_legacy_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            LEGACY_MEASUREMENTS_DDL + ";" + LEGACY_STATISTICS_DDL + ";"
            + LEGACY_REQUESTS_DDL + ";"
            + "CREATE INDEX ix_measurements_sample ON measurements(sample_key, split);"
        )
        protocol = '{"iterations": 30, "timer": "device_event", "warmup": 10}'
        rows = [
            # two trials of the same condition, unequal repeat counts
            ("meas-1", "sample-1", "case-a", 0, 1.0),
            ("meas-2", "sample-1", "case-a", 1, 2.0),
            ("meas-3", "sample-1", "case-a", 2, 3.0),
            ("meas-4", "sample-1", "case-b", 0, 10.0),
            ("meas-5", "sample-1", "case-b", 1, 20.0),
        ]
        for measurement_key, sample_key, case_id, repeat, duration in rows:
            connection.execute(
                """INSERT INTO measurements(
                       measurement_key, sample_key, hardware_key, op_family, phase,
                       recipe, shape_json, dtype, layout, features_json,
                       competition_json, duration_ms, analytic_ms, source,
                       source_class, synthetic, revision, run_id, case_id,
                       repeat_index, split, timing_scope, protocol_json,
                       environment_json, details_json, created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    measurement_key, sample_key, "Ascend910_9362", "gemm",
                    "operator", "legacy", '{"m":8.0}', "", "", "{}", "{}",
                    duration, 1.0, "frozen-round1", "measured", 0, "",
                    "run-legacy", case_id, repeat, "train", "device_event_warm",
                    protocol, '{"visible_devices": "1"}', "{}",
                    "2026-10-20T20:06:54+00:00",
                ),
            )
        connection.commit()
    finally:
        connection.close()


def test_legacy_schema_is_upgraded_additively(tmp_path):
    db = tmp_path / "legacy.sqlite"
    _write_legacy_db(db)
    with sqlite3.connect(db) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        before = connection.execute(
            "SELECT measurement_key, duration_ms FROM measurements ORDER BY measurement_id"
        ).fetchall()

    store = SQLiteCalibrationStore(db)
    info = store.schema_info()
    assert info["user_version"] == SCHEMA_VERSION
    assert info["upgraded_from_legacy_schema"] is True
    assert {"die", "visible_devices", "protocol_signature"} <= set(
        info["columns"]["measurements"]
    )
    assert {
        "duration_trial_median", "trial_count", "protocol_signature",
        "die_values_json", "visible_devices_json",
    } <= set(info["columns"]["sample_statistics"])

    # Raw rows are untouched; only the derived aggregate is rebuilt.
    with sqlite3.connect(db) as connection:
        after = connection.execute(
            "SELECT measurement_key, duration_ms FROM measurements ORDER BY measurement_id"
        ).fetchall()
    assert after == before
    stats = store.statistics()[0]
    assert stats["measurement_count"] == 5
    assert stats["trial_count"] == 2
    assert stats["duration_median"] == pytest.approx(3.0)
    assert stats["duration_trial_median"] == pytest.approx(8.5)
    assert stats["calibration_target_ms"] == pytest.approx(8.5)
    assert stats["protocol_signature"] == protocol_signature(
        {"iterations": 30, "timer": "device_event", "warmup": 10}
    )
    provenance = store.sample_rows()[0]
    assert provenance["die"] == "unknown"
    assert provenance["visible_devices"] == "1"
    assert provenance["run_id"] == "run-legacy"
    assert provenance["timing_scope"] == "device_event_warm"
    assert provenance["source_class"] == "measured"

    # Re-opening an already current DB does not migrate again.
    reopened = SQLiteCalibrationStore(db)
    assert reopened.schema_info()["upgraded_from_legacy_schema"] is False
    assert reopened.inventory()["measurements"] == 5


@pytest.mark.skipif(not ROUND1_DB.exists(), reason="frozen round-1 training DB is not mounted")
def test_frozen_round1_db_migrates_without_changing_a_single_sample(tmp_path):
    db = tmp_path / "round1-copy.sqlite"
    source = sqlite3.connect(f"file:{ROUND1_DB}?mode=ro", uri=True)
    target = sqlite3.connect(db)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()

    with sqlite3.connect(db) as connection:
        rows_before = int(connection.execute(
            "SELECT COUNT(*) FROM measurements"
        ).fetchone()[0])
        keys_before = connection.execute(
            "SELECT sample_key, duration_ms, protocol_json FROM measurements "
            "ORDER BY measurement_id LIMIT 40"
        ).fetchall()
        expected_key = connection.execute(
            "SELECT sample_key FROM measurements ORDER BY measurement_id LIMIT 1"
        ).fetchone()[0]
        expected_duration = connection.execute(
            "SELECT duration_ms FROM measurements WHERE sample_key = ? "
            "ORDER BY measurement_id", (expected_key,),
        ).fetchall()
    expected_median = statistics.median(float(item[0]) for item in expected_duration)

    store = SQLiteCalibrationStore(db)
    assert store.schema_info()["upgraded_from_legacy_schema"] is True
    inventory = store.inventory()
    assert inventory["measurements"] == rows_before == 13710
    assert inventory["conditions"] == 457
    assert inventory["hardware_identity"]["keys"] == [
        {"hardware_key": "Ascend910_9362", "measurements": 13710}
    ]
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT sample_key, duration_ms, protocol_json FROM measurements "
            "ORDER BY measurement_id LIMIT 40"
        ).fetchall() == keys_before

    # Round-1 conditions are single-trial 30-event cases, so the trial median
    # equals the pooled median there; the contract itself must still be present.
    stats = {record["sample_key"]: record for record in store.statistics()}
    migrated = stats[expected_key]
    assert migrated["trial_count"] == 1
    assert migrated["duration_trial_median"] == pytest.approx(expected_median)
    assert migrated["duration_median"] == pytest.approx(expected_median)
    assert migrated["pooled_over_target"] == pytest.approx(1.0)
    assert all(record["trial_count"] == 1 for record in stats.values())
    assert all(record["protocol_signature"] for record in stats.values())

    view = store.as_calibration_store()
    assert len(view.observations()) == 457
    assert view.rejections() == ()
    provenance = store.sample_rows()[0]
    assert provenance["visible_devices"] == "1"
    assert provenance["die"] == "unknown"
    assert provenance["protocol_signature"].startswith('{"protocol"')


# ---------------------------------------------------------------------------
# Provenance round-trip
# ---------------------------------------------------------------------------


def test_provenance_survives_reopen_and_is_not_part_of_the_semantic_key(tmp_path):
    db = tmp_path / "provenance.sqlite"
    protocol = {"timer": "device_event", "warmup": 10, "iterations": 30}
    store = SQLiteCalibrationStore(db)
    explicit = _record(
        store, run_id="run-p", case_id="case-p", repeat_index=3, duration=2.5,
        protocol=protocol, environment={"device_name": "Ascend910_9362"},
        die="7", visible_devices="2,3",
        details={"raw_case": "matmul", "checksum": "abc"},
    )
    derived = _record(
        store, run_id="run-q", case_id="case-q", duration=4.5,
        protocol=protocol, environment={"device_name": "Ascend910_9362"},
        visible_devices="5", source_class="observed",
        source="observed-capture",
    )
    unattested = _record(
        store, run_id="run-r", case_id="case-r", duration=6.5,
        protocol=protocol, environment={"device_name": "Ascend910_9362"},
    )

    reopened = SQLiteCalibrationStore(db)
    rows = {row["measurement_key"]: row for row in reopened.sample_rows()}
    first = rows[explicit["measurement_key"]]
    assert first["run_id"] == "run-p"
    assert first["case_id"] == "case-p"
    assert first["repeat_index"] == 3
    assert first["die"] == "7"
    assert first["visible_devices"] == "2,3"
    assert first["protocol_signature"] == protocol_signature(protocol)
    assert first["protocol"] == protocol
    assert first["timing_scope"] == ""
    assert first["source_class"] == "measured"
    assert first["source"] == "tier0-fixture"
    assert first["duration_ms"] == pytest.approx(2.5)
    assert first["details"] == {"raw_case": "matmul", "checksum": "abc"}
    assert rows[derived["measurement_key"]]["visible_devices"] == "5"
    assert rows[derived["measurement_key"]]["die"] == "unknown"
    assert rows[unattested["measurement_key"]]["die"] == "unknown"
    assert rows[unattested["measurement_key"]]["visible_devices"] == "unknown"

    # die/visible devices are provenance: identical conditions keep one key, so
    # they cannot fork a measured condition and cannot create a second sample.
    assert explicit["sample_key"] == derived["sample_key"]
    assert explicit["sample_key"] == unattested["sample_key"]
    stats = reopened.statistics()[0]
    assert stats["measurement_count"] == 3
    assert stats["die_values"] == ["7", "unknown"]
    assert stats["visible_devices"] == ["2,3", "5", "unknown"]
    inventory = reopened.inventory()["provenance"]
    assert inventory == {
        "measurements": 3, "with_die": 1, "with_visible_devices": 2,
        "with_protocol_signature": 3, "with_timing_scope": 0,
        "with_run_id": 3, "attributable": 3, "unattested_identity_label": "unknown",
    }
