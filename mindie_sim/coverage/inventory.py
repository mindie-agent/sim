"""Structural inventory reporting and bounded source readers.

[Role] Account for declared inventory records and their concrete source identities.
[Invariants] Keep duplicate groups and capture relationships in the inventory denominator.
[Timeouts & Locks] Bounded metadata reads and streaming hashes. No waits or locks.
[Side-Effect Policy] Reads declared inputs and returns JSON-ready coverage records.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys


MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_IDENTITY_FILE_BYTES = 64 * 1024 * 1024
MAX_INVENTORY_ROWS = 100_000
_LEDGER_SUMMARY_DIMENSIONS = 5
INVENTORY_SCHEMA = "mindie-sim.inventory-coverage/1"


def _pkg_limit(name: str, default: int) -> int:
    pkg = sys.modules.get("mindie_sim.coverage")
    return getattr(pkg, name, default) if pkg is not None else default


def source_identity(value):
    """Hash one bounded real input; the report keeps its concrete identity."""
    path = Path(value)
    size = path.stat().st_size
    if size > _pkg_limit("MAX_IDENTITY_FILE_BYTES", MAX_IDENTITY_FILE_BYTES):
        raise ValueError("inventory_source_exceeds_64_mib")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": size}


def _read_object(value, name, max_bytes=None):
    limit = _pkg_limit("MAX_METADATA_BYTES", MAX_METADATA_BYTES) if max_bytes is None else max_bytes
    source = source_identity(value) if isinstance(value, (str, Path)) else None
    if source is not None:
        if source["size_bytes"] > limit:
            raise ValueError(f"{name}_exceeds_16_mib")
        value = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}_requires_object")
    return value, source


def _int_field(container, key, *, required=False):
    value = container.get(key)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"inventory_field_requires_nonnegative_integer:{key}")
    return value


def _ledger_identity(value, expected_rows):
    source = source_identity(value)
    if source["size_bytes"] > _pkg_limit("MAX_IDENTITY_FILE_BYTES", MAX_IDENTITY_FILE_BYTES):
        raise ValueError("inventory_ledger_exceeds_64_mib")
    digest = sha256()
    rows = 0
    tiers, scopes, roles, suites, phases = (
        Counter() for _ in range(_LEDGER_SUMMARY_DIMENSIONS))
    group_counts, capture_counts, declared_sizes = Counter(), Counter(), {}
    a5_group_counts, a5_capture_counts = Counter(), Counter()
    a5_rows = 0
    with Path(value).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        base = {"audit_id", "hardware_tier", "suite", "phase", "capture_id"}
        new_schema = {"source_set_group_id", "source_set_group_size",
                      "source_set_role"} <= fields
        old_schema = {"duplicate_group_id", "duplicate_group_size",
                      "duplicate_role"} <= fields
        if not base <= fields or not (new_schema or old_schema):
            raise ValueError("inventory_ledger_missing_identity_columns")
        group_key = "source_set_group_id" if new_schema else "duplicate_group_id"
        size_key = "source_set_group_size" if new_schema else "duplicate_group_size"
        role_key = "source_set_role" if new_schema else "duplicate_role"
        scope_key = "in_a5_scope" if "in_a5_scope" in fields else None
        digest_fields = ["audit_id", "hardware_tier", "suite", "phase", "capture_id",
                         group_key, size_key, role_key]
        for row in reader:
            rows += 1
            if rows > _pkg_limit("MAX_INVENTORY_ROWS", MAX_INVENTORY_ROWS):
                raise ValueError("inventory_ledger_exceeds_100000_rows")
            digest.update("|".join(
                str(row.get(key, "")) for key in digest_fields).encode("utf-8") + b"\n")
            tier = row.get("hardware_tier", "")
            tiers[tier] += 1
            if scope_key is not None:
                in_a5 = str(row.get(scope_key, "")).strip().lower() in {
                    "true", "1", "yes"}
                scopes[row[scope_key]] += 1
            else:
                in_a5 = tier == "A5"
                scopes["A5" if in_a5 else "other"] += 1
            roles[row.get(role_key, "")] += 1
            suites[row.get("suite", "")] += 1
            phases[row.get("phase", "")] += 1
            a5_rows += int(in_a5)
            group = row.get(group_key, "")
            if group:
                group_counts[group] += 1
                if in_a5:
                    a5_group_counts[group] += 1
                try:
                    declared = int(row.get(size_key) or 1)
                except (TypeError, ValueError):
                    raise ValueError(
                        "inventory_ledger_duplicate_group_size_not_integer") from None
                if declared_sizes.setdefault(group, declared) != declared:
                    raise ValueError("inventory_ledger_duplicate_group_size_conflict")
            capture_id = row.get("capture_id", "")
            if capture_id:
                capture_counts[capture_id] += 1
                if in_a5:
                    a5_capture_counts[capture_id] += 1
    if rows != expected_rows:
        raise ValueError("inventory_ledger_row_count_mismatch")
    for group, count in group_counts.items():
        if declared_sizes.get(group) != count:
            raise ValueError("inventory_ledger_duplicate_group_size_mismatch")
    histogram = Counter(capture_counts.values())
    a5_histogram = Counter(a5_capture_counts.values())
    return {
        "source": source,
        "schema_family": "source_set" if new_schema else "duplicate_group",
        "rows": rows,
        "a5_rows": a5_rows,
        "a5_duplicate_groups": sum(
            1 for count in a5_group_counts.values() if count > 1),
        "a5_duplicate_extra_records": sum(
            count - 1 for count in a5_group_counts.values() if count > 1),
        "a5_captures": {
            "distinct": len(a5_capture_counts),
            "records_per_capture_histogram": {
                str(size): count for size, count in sorted(a5_histogram.items())},
        },
        "in_a5_scope_counts": dict(sorted(scopes.items())),
        "hardware_tier_counts": dict(sorted(tiers.items())),
        "suite_counts": dict(sorted(suites.items())),
        "phase_counts": dict(sorted(phases.items())),
        "duplicate_role_counts": dict(sorted(roles.items())),
        "duplicate_groups": len(group_counts),
        "multi_record_duplicate_groups": sum(
            1 for count in group_counts.values() if count > 1),
        "duplicate_extra_records": sum(
            count - 1 for count in group_counts.values()),
        "captures": {
            "distinct": len(capture_counts),
            "records_per_capture_histogram": {
                str(size): count for size, count in sorted(histogram.items())},
        },
        "identity_sha256_in_file_order": "sha256:" + digest.hexdigest(),
        "identity_fields": digest_fields,
    }


def _a5_record_recovery(value, expected_rows):
    source = source_identity(value)
    if source["size_bytes"] > _pkg_limit("MAX_IDENTITY_FILE_BYTES", MAX_IDENTITY_FILE_BYTES):
        raise ValueError("inventory_records_exceed_64_mib")
    flag_keys = (
        "full_per_call_10pct_validated", "per_step_runtime_state_restored",
        "exact_per_step_values_restored", "context_explicitly_unobserved",
        "source_binding_ambiguous")
    flags = {key: Counter() for key in flag_keys}
    present = set()
    configuration, evidence = Counter(), Counter()
    native_records = native_cohorts = 0
    total_rows = rows = 0
    with Path(value).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if "hardware_tier" not in fields:
            raise ValueError("inventory_records_missing_hardware_tier")
        present = {key for key in flag_keys if key in fields}
        if "configuration_recovery_status" in fields:
            present.add("configuration_recovery_status")
        evidence_key = ("current_evidence_kind" if "current_evidence_kind" in fields
                        else "record_grain" if "record_grain" in fields else None)
        for row in reader:
            total_rows += 1
            if total_rows > _pkg_limit("MAX_INVENTORY_ROWS", MAX_INVENTORY_ROWS):
                raise ValueError("inventory_records_exceed_100000_rows")
            if str(row.get("hardware_tier", "")).strip() != "A5":
                continue
            rows += 1
            for key in present & set(flag_keys):
                raw = str(row.get(key, "")).strip().lower()
                if raw in {"true", "1", "yes"}:
                    value = "true"
                elif raw in {"false", "0", "no"}:
                    value = "false"
                else:
                    raise ValueError(f"inventory_records_non_boolean_field:{key}")
                flags[key][value] += 1
            if "configuration_recovery_status" in present:
                configuration[row.get("configuration_recovery_status", "")] += 1
            if evidence_key:
                evidence[row.get(evidence_key, "")] += 1
            try:
                count = int(row.get("native_compared_cohorts") or 0)
            except ValueError:
                raise ValueError("inventory_records_native_compared_cohorts_not_integer") from None
            if count < 0:
                raise ValueError("inventory_records_native_compared_cohorts_negative")
            native_records += int(count > 0)
            native_cohorts += count
    if rows != expected_rows:
        raise ValueError("inventory_records_a5_row_count_mismatch")
    flag_counts = {}
    for key in flag_keys:
        if key in present:
            flag_counts[key] = dict(sorted(flags[key].items()))
        else:
            flag_counts[key] = {"not_declared": rows}
    return {
        "source": source,
        "source_rows_total": total_rows,
        "rows": rows,
        "record_scope": "A5",
        "flags": flag_counts,
        "native_evidence_records": native_records,
        "native_compared_cohorts_total": native_cohorts,
        "configuration_recovery_status_counts": dict(sorted(configuration.items())),
        "evidence_kind_counts": dict(sorted(evidence.items())),
        "historical_tier_note": (
            "This ledger does not supply a current 10% per-call verdict; "
            "a historical absolute_15us flag is not consumed here."),
    }


def _csv_records(value, name, required):
    source = source_identity(value)
    if source["size_bytes"] > _pkg_limit("MAX_IDENTITY_FILE_BYTES", MAX_IDENTITY_FILE_BYTES):
        raise ValueError(f"{name}_exceeds_64_mib")
    rows = []
    with Path(value).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(required) <= set(reader.fieldnames or ()):
            raise ValueError(f"{name}_missing_required_columns")
        for row in reader:
            rows.append(row)
            if len(rows) > _pkg_limit("MAX_INVENTORY_ROWS", MAX_INVENTORY_ROWS):
                raise ValueError(f"{name}_exceeds_100000_rows")
    return rows, source


def inventory_coverage(summary, captures=None, *, ledger=None, records=None,
                       captures_manifest=None):
    """Carry the 7337-record ledger identity, duplicate and capture relations.

    Accepts both the r1 summary shape and T03 r2 ``t03/full-recovery/2``.
    Capture counts come from the supplied manifest/summary version, never a
    compiled-in 4173.
    """
    summary, summary_source = _read_object(summary, "inventory_summary")
    total = _int_field(summary, "ledger_total_records")
    if total is None:
        total = _int_field(summary, "records_total", required=True)
    a5_records = _int_field(summary, "a5_records")
    legacy_captures = _int_field(summary, "a5_unique_event_captures")
    legacy_duplicate_extra = _int_field(summary, "a5_duplicate_extra_records")
    legacy_duplicate_groups = _int_field(summary, "a5_duplicate_groups")
    if a5_records is not None and a5_records > total:
        raise ValueError("inventory_a5_records_exceed_ledger_total")
    if a5_records is not None and legacy_captures is not None \
            and legacy_duplicate_extra is not None \
            and legacy_captures + legacy_duplicate_extra != a5_records:
        raise ValueError("inventory_a5_capture_duplicate_counts_do_not_reconcile")
    identity_counts = summary.get("a5_independent_capture_counts") or {}
    if not isinstance(identity_counts, Mapping):
        raise ValueError("inventory_a5_independent_capture_counts_requires_object")
    native_summary = summary.get("native") if isinstance(summary.get("native"), Mapping) else {}
    native_case_count = _int_field(summary, "a5_native_case_count")
    if native_case_count is None:
        native_case_count = native_summary.get("archive_cases")
    native_cohort_records = _int_field(summary, "a5_native_cohort_records")
    if native_cohort_records is None:
        native_cohort_records = native_summary.get("validation_rows")

    capture_report = capture_source = None
    if captures is not None:
        capture_report, capture_source = _read_object(captures, "captures_summary")
    captures_total = _int_field(summary, "captures_total")
    if capture_report is not None and _int_field(capture_report, "capture_count") is not None:
        captures_total = _int_field(capture_report, "capture_count")
    capture_count_source = ("captures_summary" if capture_report is not None
                            else "inventory_summary")
    a5_capture_count = None
    if capture_report is not None:
        tier_counts = capture_report.get("tier_capture_counts")
        if isinstance(tier_counts, Mapping):
            value = tier_counts.get("A5")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                a5_capture_count = value
    manifest_report = manifest_source = None
    if captures_manifest is not None:
        manifest_report, manifest_source = _read_object(
            captures_manifest, "captures_manifest")
        manifest_count = _int_field(manifest_report, "capture_count", required=True)
        manifest_lines = _int_field(manifest_report, "total_lines")
        if manifest_lines is not None and manifest_lines != manifest_count:
            raise ValueError("inventory_capture_manifest_total_lines_mismatch")
        if captures_total is not None and captures_total != manifest_count:
            raise ValueError("inventory_capture_manifest_count_mismatch")
        captures_total = manifest_count
        capture_count_source = "captures_manifest"

    result = {
        "schema": INVENTORY_SCHEMA,
        "source_schema": summary.get("schema"),
        "ledger": {
            "total_records": total,
            "records_are_not_independent_captures": True,
            "tier_counts": summary.get("tier_counts"),
            "a5_scope_records": a5_records,
            "outside_a5_scope_records": (total - a5_records
                                         if a5_records is not None else None),
            "captures_total": captures_total,
            "capture_count_source": capture_count_source,
            "a5_capture_count": a5_capture_count,
            "legacy_a5_unique_event_captures": legacy_captures,
            "legacy_a5_duplicate_extra_records": legacy_duplicate_extra,
            "legacy_a5_duplicate_groups": legacy_duplicate_groups,
            "a5_duplicate_identity_check": identity_counts.get("identity_check"),
            "source_set_duplicate_groups": _int_field(
                summary, "source_set_duplicate_groups"),
            "source_set_duplicate_extra_records": _int_field(
                summary, "source_set_duplicate_extra_records"),
            "source_relations_total": _int_field(summary, "source_relations_total"),
            "multi_source_records": _int_field(summary, "multi_source_records"),
            "derived_records": _int_field(summary, "derived_records"),
            "a5_native_case_count": native_case_count,
            "a5_native_cohort_records": native_cohort_records,
            "a5_records_with_native_evidence": _int_field(
                summary, "a5_records_with_native_evidence"),
            "a5_records_context_unobserved": _int_field(
                summary, "a5_records_context_unobserved"),
            "a5_records_runtime_state_not_restored": _int_field(
                summary, "a5_records_runtime_state_not_restored"),
            "a5_records_source_binding_ambiguous": _int_field(
                summary, "a5_records_source_binding_ambiguous"),
        },
        "unrecovered": {
            "runtime_state_not_restored_records": _int_field(
                summary, "a5_records_runtime_state_not_restored"),
            "context_unobserved_records": _int_field(
                summary, "a5_records_context_unobserved"),
            "source_binding_ambiguous_records": _int_field(
                summary, "a5_records_source_binding_ambiguous"),
            "historical_absolute_15us_is_not_this_round": True,
        },
        "sources": {"inventory_summary": summary_source},
        "identity_ledger": None,
        "a5_record_recovery": None,
        "capture_recovery": None,
    }
    if capture_report is not None:
        result["capture_recovery"] = {
            "capture_count": captures_total,
            "capture_count_source": capture_count_source,
            "tier_capture_counts": capture_report.get("tier_capture_counts"),
            "disposition_counts": capture_report.get("disposition_counts"),
            "schema_family_counts": capture_report.get("schema_family_counts"),
            "sidecar_candidate_calls_total": capture_report.get(
                "sidecar_candidate_calls_total"),
            "full_scan_verified_calls_total": capture_report.get(
                "full_scan_verified_calls_total"),
            "lower_bound_calls_total": capture_report.get("lower_bound_calls_total"),
            "upper_bound_calls_total": capture_report.get("upper_bound_calls_total"),
            "captures_with_full_scan": capture_report.get("captures_with_full_scan"),
            "captures_with_sidecar": capture_report.get("captures_with_sidecar"),
            "captures_with_abi": capture_report.get("captures_with_abi"),
            "captures_empty": capture_report.get("captures_empty"),
            "native_archive": capture_report.get("native_archive"),
            "recovery_depth_counts": capture_report.get("recovery_depth_counts"),
            "gap_counts": capture_report.get("captures_gap_counts"),
            "native_cohorts_validated": _int_field(
                capture_report, "native_cohorts_validated"),
            "native_windows_reproduced": _int_field(
                capture_report, "native_windows_reproduced"),
            "comparable_calls_lower_bound": capture_report.get(
                "comparable_calls_lower_bound"),
            "comparable_calls_upper_bound": capture_report.get(
                "comparable_calls_upper_bound"),
            "captures_step_values": capture_report.get("captures_step_values"),
        }
        result["sources"]["captures_summary"] = capture_source
    if manifest_report is not None:
        result["capture_manifest"] = {
            "capture_count": captures_total,
            "bucket_count": _int_field(manifest_report, "bucket_count"),
            "total_lines": _int_field(manifest_report, "total_lines"),
            "shard_count": len(manifest_report.get("shard_files") or ()),
        }
        result["sources"]["captures_manifest"] = manifest_source
    if ledger is not None:
        if a5_records is None:
            raise ValueError("inventory_ledger_requires_a5_record_count")
        ledger_report = _ledger_identity(ledger, total)
        if ledger_report["a5_rows"] != a5_records:
            raise ValueError("inventory_ledger_a5_rows_do_not_match_summary")
        if legacy_duplicate_extra is not None \
                and ledger_report["a5_duplicate_extra_records"] != legacy_duplicate_extra:
            raise ValueError("inventory_ledger_a5_duplicate_extra_mismatch")
        if legacy_duplicate_groups is not None \
                and ledger_report["a5_duplicate_groups"] != legacy_duplicate_groups:
            raise ValueError("inventory_ledger_a5_duplicate_groups_mismatch")
        if legacy_captures is not None \
                and ledger_report["a5_captures"]["distinct"] != legacy_captures:
            raise ValueError("inventory_ledger_a5_captures_mismatch")
        result["identity_ledger"] = ledger_report
        result["sources"]["identity_ledger"] = ledger_report["source"]
    if records is not None:
        if a5_records is None:
            raise ValueError("inventory_records_requires_a5_record_count")
        recovery = _a5_record_recovery(records, a5_records)
        result["a5_record_recovery"] = recovery
        result["sources"]["a5_records"] = recovery["source"]
        flags = recovery["flags"]
        result["unrecovered"].update({
            "runtime_state_not_restored_records": flags[
                "per_step_runtime_state_restored"].get("false", 0),
            "context_unobserved_records": flags[
                "context_explicitly_unobserved"].get("true", 0),
            "source_binding_ambiguous_records": flags[
                "source_binding_ambiguous"].get("true", 0),
            "full_per_call_10pct_validated_true_records": flags[
                "full_per_call_10pct_validated"].get("true", 0),
        })
    return result
