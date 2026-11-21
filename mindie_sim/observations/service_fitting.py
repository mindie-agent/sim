"""Service fitting APIs and capture evidence assembly.

[Role]
  Join declared calibration roles with located capture rows and expose fitters.
[Invariants]
  Only eligible independent-resource evidence enters training. Native
  measurement, declared plans and provenance retain their own boundaries.
[Timeouts & Locks]
  Explicit local capture reads only. No subprocesses or locks.
[Side-Effect Policy]
  Read-only on capture inputs. Returns evidence and fitted parameters.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json
import math
from statistics import median

from mindie_sim.observations.resource_services import (
    normalize_resource_service,
    resolve_nominal_rate,
)
from mindie_sim.prediction.services import ServiceCurve

from .service_fitting_capture_io import (
    _capture_float,
    _capture_int,
    capture_image_identity,
    load_native_cohort_validation,
    read_capture_locators,
)
from .service_fitting_capture_model import (
    bind_capture_native_plan,
    capture_measurement,
    join_capture_measurement_and_plan,
    logical_capture_work,
    parse_capture_abi,
)
from .service_fitting_regression import (
    fit_resource_service,
    fit_service_curve,
    nonnegative_affine,
)
from .service_fitting_roles import (
    _ROLE_IN_SAMPLE,
    calibration_role_partitions,
    fit_resource_service_from_role_records,
    load_role_ledger,
)


ACTUAL_EVIDENCE_SCHEMA = "mindie-sim.actual-resource-evidence/1"


def _capture_record_cohort(row):
    import json as _json
    fields = {}
    for key in ("contract_kind", "contract_match_status", "contract_checks",
                "contract_notes", "family_role", "family_status",
                "archive_event_count", "archive_mean_us", "cohort_window",
                "global_signature_count", "global_signature_mean_us",
                "full_cohort_window_reproduced", "clock_domain", "step_state",
                "locator_status", "locator_notes", "ambiguities"):
        value = row.get(key)
        if key in {"contract_checks", "contract_notes", "cohort_window",
                   "clock_domain", "step_state", "locator_notes",
                   "ambiguities"}:
            try:
                fields[key] = _json.loads(value) if value else None
            except (TypeError, ValueError):
                fields[key] = value
        else:
            fields[key] = value
    return fields


def _capture_locator(row):
    return {
        "csv_line": _capture_int(row.get("example_locator_csv_line")),
        "row_index": _capture_int(row.get("example_locator_row_index")),
        "task_id": row.get("example_locator_task_id"),
        "stream_id": row.get("example_locator_stream_id"),
        "device_id": row.get("example_locator_device_id"),
        "raw_start_us": _capture_float(row.get("raw_start_us")),
        "raw_duration_us": _capture_float(row.get("raw_duration_us")),
        "raw_end_us": _capture_float(row.get("raw_end_us")),
        "raw_op_type": row.get("raw_op_type"),
        "raw_input_shapes": row.get("raw_input_shapes"),
        "raw_output_shapes": row.get("raw_output_shapes"),
        "abi_signature": row.get("abi_signature"),
        "source_path": row.get("source_path"),
    }


def _locator_checks(raw, locator):
    checks = []
    for key, raw_key in (("device_id", "Device_id"),
                         ("task_id", "Task ID"),
                         ("stream_id", "Stream ID")):
        expected = str(locator.get(key) or "")
        actual = str(raw.get(raw_key) or "")
        checks.append({"name": key, "ok": expected == actual,
                       "expected": expected, "actual": actual})
    start = _capture_float(raw.get("Start Time(us)"))
    duration = _capture_float(raw.get("Duration(us)"))
    expected_start = locator.get("raw_start_us")
    expected_duration = locator.get("raw_duration_us")
    checks.append({
        "name": "start_time",
        "ok": (start is not None and expected_start is not None
               and abs(start - expected_start) <= 1e-6),
        "expected": expected_start, "actual": start,
    })
    checks.append({
        "name": "duration",
        "ok": (duration is not None and expected_duration is not None
               and abs(duration - expected_duration) <= 1e-6),
        "expected": expected_duration, "actual": duration,
    })
    abi = parse_capture_abi(raw)["abi_signature"]
    expected_abi = str(locator.get("abi_signature") or "")
    expected_sections = expected_abi.split("|")
    actual_sections = abi.split("|")
    tokens_match = (len(expected_sections) == len(actual_sections)
                    and all(
                        [token for token in expected.split(";") if token]
                        == [token for token in actual.split(";") if token]
                        for expected, actual in zip(expected_sections,
                                                    actual_sections)))
    checks.append({
        "name": "abi_signature", "ok": tokens_match,
        "expected": expected_abi, "actual": abi,
        "slot_compaction": tokens_match and abi != expected_abi,
        "detail": ("present_tensor_tokens_match"
                   + (";raw_field_compacts_empty_optional_slots"
                      if tokens_match and abi != expected_abi else "")),
    })
    return checks


def build_actual_evidence(role_records, native_validation, *,
                          with_sha256=False, max_cohorts=None,
                          role="in_sample"):
    """Join role ledger -> T03 validation -> same-boundary raw PMU rows.

    The function reads only ``in_sample`` captures.  Frozen-transfer and
    not-declared cohorts are registered but their raw captures are not opened.
    Every join check, participation record, exclusion, duplicate counter and
    source hash is returned so the training set is reproducible.
    """
    import hashlib
    if isinstance(native_validation, Mapping):
        native_rows = list(native_validation.get("rows") or ())
        native_meta = {key: native_validation.get(key)
                       for key in ("path", "sha256", "bytes", "count",
                                   "counts")}
    else:
        native_rows = list(native_validation)
        native_meta = {"path": None, "sha256": None, "bytes": None,
                       "count": len(native_rows), "counts": {}}
    native_by_cohort = {str(row.get("cohort_id")): row
                        for row in native_rows}
    partition = calibration_role_partitions(role_records)
    selected = list(partition["in_sample"])
    if max_cohorts is not None:
        selected = selected[:int(max_cohorts)]
    registration = {
        "role_counts": dict(partition["counts"]),
        "total": partition["total"],
        "in_sample_ids": [record["cohort_id"]
                          for record in partition["in_sample"]],
        "frozen_transfer_ids": [record["cohort_id"]
                                for record in partition["frozen_transfer"]],
        "not_declared_ids": [record["cohort_id"]
                            for record in partition["not_declared"]],
        "policy": partition["policy"],
        "read_capture_roles": [role],
    }
    excluded = []
    for record in partition["frozen_transfer"]:
        excluded.append({
            "cohort_id": record["cohort_id"],
            "role": record.get("role"),
            "reason": "role_frozen_transfer_not_training",
            "source_path": record.get("source_path"),
        })
    for record in partition["not_declared"]:
        excluded.append({
            "cohort_id": record["cohort_id"],
            "role": record.get("role"),
            "reason": "role_not_declared_neither_training_nor_holdout",
            "source_path": record.get("source_path"),
        })
    joined = []
    groups_by_capture = {}
    for record in selected:
        cohort_id = record["cohort_id"]
        row = native_by_cohort.get(cohort_id)
        if row is None:
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "native_validation_row_missing",
                "source_path": record.get("source_path"),
            })
            continue
        if str(row.get("calibration_relation") or "") != _ROLE_IN_SAMPLE:
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "native_validation_role_mismatch",
                "declared_role": row.get("calibration_relation"),
                "source_path": record.get("source_path"),
            })
            continue
        if str(row.get("case_id") or "") != str(record.get("case_id") or ""):
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "case_id_mismatch",
                "ledger_case": record.get("case_id"),
                "native_case": row.get("case_id"),
                "source_path": record.get("source_path"),
            })
            continue
        if str(row.get("source_path") or "") != str(record.get("source_path") or ""):
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "source_path_mismatch",
                "ledger_source": record.get("source_path"),
                "native_source": row.get("source_path"),
            })
            continue
        locator = _capture_locator(row)
        if not locator["csv_line"]:
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "locator_line_missing",
                "source_path": row.get("source_path"),
            })
            continue
        groups_by_capture.setdefault(locator["source_path"], []).append(
            (record, row, locator))
    capture_files = {}
    raw_by_key = {}
    for source_path, items in groups_by_capture.items():
        lines = [item[2]["csv_line"] for item in items]
        try:
            raw = read_capture_locators(source_path, lines,
                                        with_sha256=with_sha256)
        except ValueError as error:
            for record, row, locator in items:
                excluded.append({
                    "cohort_id": record["cohort_id"],
                    "role": record.get("role"),
                    "reason": "capture_unavailable",
                    "detail": str(error),
                    "source_path": source_path,
                })
            continue
        capture_files[source_path] = {
            "path": raw["path"], "bytes": raw["bytes"],
            "sha256": raw["sha256"], "rows_scanned": raw["rows_scanned"],
            "image": capture_image_identity(source_path),
        }
        for record, row, locator in items:
            raw_row = raw["rows"].get(locator["csv_line"])
            if raw_row is None:
                excluded.append({
                    "cohort_id": record["cohort_id"],
                    "role": record.get("role"),
                    "reason": "locator_row_missing",
                    "source_path": source_path,
                    "csv_line": locator["csv_line"],
                })
                continue
            raw_by_key[record["cohort_id"]] = raw_row
    for record in selected:
        cohort_id = record["cohort_id"]
        row = native_by_cohort.get(cohort_id)
        if row is None or cohort_id not in raw_by_key:
            continue
        raw_row = raw_by_key[cohort_id]
        locator = _capture_locator(row)
        image = capture_files.get(locator["source_path"], {}).get("image") or {}
        contract = _capture_record_cohort(row)
        measurement = capture_measurement(raw_row)
        work = logical_capture_work(raw_row)
        checks = _locator_checks(raw_row, locator)
        locator_verified = all(item["ok"] for item in checks)
        abi_slot_compaction = any(
            item.get("slot_compaction") for item in checks)
        exact = str(row.get("contract_match_status") or "") == "exact"
        plan = bind_capture_native_plan(raw_row,
                                        image_id=image.get("image_id"))
        pairing = join_capture_measurement_and_plan(measurement, plan)
        missing = []
        if not locator_verified:
            missing.append("raw_locator_and_abi_signature_verified")
        if not exact:
            missing.append("exact_geometry_contract")
        if str(row.get("full_cohort_window_reproduced") or "") != "True":
            missing.append("reproduced_full_cohort_window")
        if work.get("logical_work") is None:
            missing.append("logical_physical_work")
        if plan.get("status") != "predicted":
            missing.append("selected_native_program")
            missing.extend(
                "native_plan:" + str(item) for item in plan.get("reasons") or ())
        else:
            missing.append("dispatch_and_tiling_identity")
            missing.append("issued_work_measured")
        missing.append("owner_aggregation_and_active_owner_count")
        missing.append("independent_pipe_measurement")
        eligible = False
        if not locator_verified:
            reason = "raw_locator_or_abi_signature_mismatch"
        elif not exact:
            reason = "geometry_contract_not_exact"
        elif work.get("logical_work") is None:
            reason = "logical_physical_work_not_derivable"
        elif plan.get("status") != "predicted":
            reason = "selected_native_program_not_established"
        elif not plan.get("source_established"):
            missing.append("same_binary_source_identity")
            reason = "binary_source_not_established"
        else:
            reason = "measured_pipe_aggregation_and_dispatch_not_established"
        joined.append({
            "cohort_id": cohort_id,
            "case_id": record.get("case_id"),
            "role": record.get("role"),
            "source_path": locator["source_path"],
            "locator": locator,
            "native_validation": contract,
            "capture_image": image,
            "locator_checks": checks,
            "locator_verified": locator_verified,
            "abi_slot_compaction": abi_slot_compaction,
            "raw_measurement": measurement,
            "logical_work": work,
            "native_plan": plan,
            "measurement_plan_join": pairing,
            "training_eligibility": {
                "eligible": eligible,
                "reason": reason,
                "missing": list(dict.fromkeys(missing)),
            },
            "duplicates": {
                "source": "T03 native_cohort_validation global_signature_count",
                "global_signature_count": _capture_int(
                    row.get("global_signature_count")),
                "archive_event_count": _capture_int(
                    row.get("archive_event_count")),
                "global_signature_mean_us": _capture_float(
                    row.get("global_signature_mean_us")),
                "example_row_is_one_occurrence": True,
            },
        })
    groups = {}
    for item in joined:
        plan = item["native_plan"]
        key = (str(item["capture_image"].get("image_id")),
               str(item["raw_measurement"]["scheduling"].get("kernel_type")),
               str(item["native_validation"].get("contract_match_status")))
        group = groups.setdefault(key, {
            "image_id": key[0], "native_type": key[1],
            "contract_match_status": key[2], "cohorts": [],
        })
        group["cohorts"].append(item["cohort_id"])
    for group in groups.values():
        group["count"] = len(group["cohorts"])
        group["fit_eligible"] = False
        group["missing"] = [
            "independent_pipe_work_and_owner_aggregation",
            "same_dispatch_tiling_identity",
            "at_least_two_independent_work_sizes_for_one_resource",
        ]
    pairs = []
    by_type = {}
    for item in joined:
        by_type.setdefault(
            (item["source_path"],
             item["raw_measurement"]["scheduling"].get("kernel_type")),
            []).append(item)
    for (source_path, native_type), items in sorted(
            by_type.items(), key=lambda pair: (str(pair[0][0]), str(pair[0][1]))):
        if len(items) < 2:
            continue
        first, second = items[0], items[1]
        same_work = (
            first["logical_work"].get("logical_work")
            == second["logical_work"].get("logical_work"))
        pairs.append({
            "source_path": source_path,
            "native_type": native_type,
            "cohort_ids": [first["cohort_id"], second["cohort_id"]],
            "csv_lines": [first["locator"]["csv_line"],
                          second["locator"]["csv_line"]],
            "task_ids": [first["locator"]["task_id"],
                         second["locator"]["task_id"]],
            "stream_ids": [first["locator"]["stream_id"],
                           second["locator"]["stream_id"]],
            "logical_work": [first["logical_work"].get("logical_work"),
                             second["logical_work"].get("logical_work")],
            "block_num": [
                first["raw_measurement"].get("block_num"),
                second["raw_measurement"].get("block_num")],
            "kernel_duration_us": [
                first["raw_measurement"].get("kernel_duration_us"),
                second["raw_measurement"].get("kernel_duration_us")],
            "same_work": same_work,
            "same_owner_condition": False,
            "two_point_fit_eligible": False,
            "reason": ("overlapping_joint_kernel_window; different_work_and_"
                       "owner_conditions; measured_pipe_aggregation_not_declared"),
        })
    counts = {}
    for item in joined:
        key = item["native_validation"].get("contract_match_status")
        counts[key] = counts.get(key, 0) + 1
    reproduced = sum(
        1 for item in joined
        if item["native_validation"].get("full_cohort_window_reproduced")
        in {"True", True})
    training_inputs = []
    for item in joined:
        if item["training_eligibility"]["eligible"]:
            contract = {
                "cohort_id": item["cohort_id"],
                "role": item["role"],
                "scope": "independent_resource",
                "resources": [item["native_plan"].get("owner_phases", {}).get(
                    "mte2", {}).get("resource", "mte2")],
                "phase": "load",
                "work_unit": "bytes",
                "work": item["native_plan"].get("issued", {}).get("mte2_bytes"),
                "duration_us": item["raw_measurement"].get("kernel_duration_us"),
                "boundary": "device_kernel",
                "provenance": {"window": "resource_phase"},
            }
            training_inputs.append(contract)
    result = {
        "schema": ACTUAL_EVIDENCE_SCHEMA,
        "role_registration": registration,
        "native_validation": native_meta,
        "captures": capture_files,
        "participation": joined,
        "excluded": excluded,
        "counts": {
            "in_sample_joined": len(joined),
            "in_sample_excluded": len(excluded) - len(
                partition["frozen_transfer"]) - len(partition["not_declared"]),
            "contract_match_status": counts,
            "reproduced_full_cohort_window": reproduced,
            "not_reproduced_full_cohort_window": len(joined) - reproduced,
            "abi_slot_compaction": sum(
                1 for item in joined if item.get("abi_slot_compaction")),
        },
        "groups": list(groups.values()),
        "real_pairs": pairs,
        "training_inputs": training_inputs,
        "service": None,
        "fit": {
            "status": "underdetermined",
            "reason": ("no_in_sample_observation_has_an_independent_pipe_"
                       "measurement_with_declared_owner_aggregation_and_"
                       "verified_dispatch"),
            "training_inputs": len(training_inputs),
            "needed_evidence": [
                "selected_native_program_and_same_dispatch_tiling",
                "measured_issued_work_per_owner",
                "declared_owner_aggregation_and_active_owner_count",
                "independent_resource_pipe_measurement",
                "resource_duration_us_for_one_phase",
            ],
        },
        "policy": {
            "read_capture_roles": ["in_sample"],
            "frozen_transfer_capture_read": False,
            "not_declared_capture_read": False,
            "sealed_holdout_read": False,
            "cpu_simulated_duration_in_training": False,
            "target_duration_in_training": False,
            "logical_work_is_issued_work": False,
            "block_num_is_owner_count": False,
            "overlapping_pipe_times_added": False,
        },
    }
    canonical = json.loads(json.dumps(result, sort_keys=True,
                                      separators=(",", ":"),
                                      allow_nan=False))
    result["canonical_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()
    return result


def fit_actual_training_evidence(role_path, native_validation_path, *,
                                 with_sha256=False, max_cohorts=None):
    """Load the real role/native evidence and return one reproducible report."""
    import hashlib
    from pathlib import Path
    role_path = Path(role_path).expanduser().resolve()
    if not role_path.is_file():
        raise ValueError("role_ledger_source_missing:" + str(role_path))
    records = load_role_ledger(role_path)
    validation = load_native_cohort_validation(native_validation_path)
    evidence = build_actual_evidence(
        records, validation, with_sha256=with_sha256,
        max_cohorts=max_cohorts)
    evidence["role_ledger"] = {
        "path": str(role_path),
        "sha256": hashlib.sha256(role_path.read_bytes()).hexdigest(),
        "bytes": role_path.stat().st_size,
        "count": len(records),
    }
    canonical = json.loads(json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), allow_nan=False))
    evidence["canonical_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()
    if evidence["training_inputs"]:
        fitted = fit_resource_service_from_role_records(
            evidence["training_inputs"], resource="mte2", phase="load",
            work_unit="bytes", scope="independent_resource")
        evidence["fit"] = {
            "status": fitted.get("status"),
            "service": fitted.get("service"),
            "fit": fitted.get("fit"),
            "reason": fitted.get("reason"),
            "training_participation": fitted.get("training_participation"),
        }
    return evidence


__all__ = [
    "ACTUAL_EVIDENCE_SCHEMA",
    "build_actual_evidence",
    "bind_capture_native_plan",
    "capture_image_identity",
    "capture_measurement",
    "fit_actual_training_evidence",
    "fit_resource_service_from_role_records",
    "join_capture_measurement_and_plan",
    "load_native_cohort_validation",
    "load_role_ledger",
    "logical_capture_work",
    "parse_capture_abi",
    "read_capture_locators",
]
