"""Link declared inventory dispositions and source records.

[Role] Resolve source edges and report record linkage for acceptance rows.
[Invariants] Treat ambiguous links as unresolved and preserve source identities.
[Timeouts & Locks] Bounded source reads and linkage processing. No waits or locks.
[Side-Effect Policy] Reads declared inputs and returns linkage evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os

from .inventory import _csv_records, _int_field, _read_object


def _json_list(value, name):
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}_must_be_json_list") from None
    if not isinstance(parsed, list):
        raise ValueError(f"{name}_must_be_json_list")
    return parsed


def _path_key(value):
    return os.path.realpath(str(value)) if value else ""


def _edge_list(row, name):
    values = _json_list(row.get("source_statuses"), name)
    edges, seen = [], set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name}_requires_edge_objects")
        source_id = value.get("source_id")
        capture_id = value.get("capture_id")
        event_source = value.get("event_source")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{name}_edge_requires_source_id")
        if not isinstance(capture_id, str) or not capture_id:
            raise ValueError(f"{name}_edge_requires_capture_id")
        key = (source_id, capture_id, _path_key(event_source or value.get("raw_path")))
        if key in seen:
            continue
        seen.add(key)
        edges.append({
            "source_id": source_id,
            "event_source_id": value.get("event_source_id"),
            "capture_id": capture_id,
            "event_source": event_source,
            "raw_path": value.get("raw_path"),
            "status": value.get("status"),
            "exists": value.get("exists"),
        })
    if not edges:
        raise ValueError(f"{name}_requires_source_status_edges")
    return edges


def _declared_source_edges(value):
    if value is None:
        return {}
    if isinstance(value, Mapping):
        value = value.get("source_edges", value)
    if isinstance(value, Mapping):
        rows = [dict(edge, item_id=item_id)
                for item_id, edges in value.items()
                for edge in (edges if isinstance(edges, list) else [edges])]
    elif isinstance(value, (list, tuple)):
        rows = list(value)
    else:
        raise ValueError("source_edges_requires_object_or_list")
    by_item = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("source_edge_requires_object")
        item_id = (row.get("item_id") or row.get("sample_id")
                   or row.get("observation_id") or row.get("window_id"))
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("source_edge_requires_item_id")
        source_id = row.get("source_id")
        source_path = row.get("source_path")
        capture_id = row.get("capture_id")
        if not source_id and not source_path:
            raise ValueError("source_edge_requires_source_id_or_path")
        by_item.setdefault(item_id, []).append({
            "source_id": source_id,
            "source_path": _path_key(source_path) if source_path else None,
            "capture_id": capture_id,
        })
    return by_item


def record_linkage(acceptance, *, dispositions, record_sources=None,
                   source_registry=None, captures_index=None,
                   captures_manifest=None, source_edges=None):
    """Link T16 sample/window/request facts to actual T03 source edges.

    A record may declare several sources and a source may be shared by direct
    and derived records.  Only the source edge actually selected by the
    comparison participates in capture/audit accounting; the complete source
    set stays for inventory traceability.  This is identity linkage, not
    per-call certification.
    """
    if not isinstance(acceptance, Mapping):
        raise ValueError("record_linkage_requires_acceptance")
    disposition_rows, disposition_source = _csv_records(
        dispositions, "record_dispositions",
        {"audit_id", "hardware_tier", "suite", "phase", "primary_raw_source",
         "primary_source_id", "source_ids", "event_source_ids",
         "evidence_capture_ids", "source_statuses", "derived", "derived_key",
         "source_set_group_id",
         "source_set_group_size", "source_set_role", "native_case_id"})
    records = {}
    for row in disposition_rows:
        audit_id = row.get("audit_id")
        if not audit_id or audit_id in records:
            raise ValueError("record_linkage_duplicate_or_missing_audit_id")
        edges = _edge_list(row, f"record_statuses[{audit_id}]")
        capture_ids = _json_list(row.get("evidence_capture_ids"),
                                 "record_capture_ids")
        edge_captures = {edge["capture_id"] for edge in edges}
        declared_captures = {str(value) for value in capture_ids}
        conflict = None
        if declared_captures != edge_captures:
            conflict = "evidence_capture_ids_mismatch"
        by_source = {}
        for edge in edges:
            by_source.setdefault(edge["source_id"], set()).add(edge["capture_id"])
        if any(len(values) > 1 for values in by_source.values()):
            conflict = conflict or "same_source_conflicting_capture"
        records[audit_id] = {
            "audit_id": audit_id,
            "hardware_tier": row.get("hardware_tier"),
            "suite": row.get("suite"),
            "phase": row.get("phase"),
            "record_grain": row.get("record_grain"),
            "primary_raw_source": row.get("primary_raw_source"),
            "primary_source_id": row.get("primary_source_id") or None,
            "source_ids": _json_list(row.get("source_ids"), "record_source_ids"),
            "event_source_ids": _json_list(
                row.get("event_source_ids"), "record_event_source_ids"),
            "capture_ids": capture_ids,
            "source_edges": edges,
            "edges_by_source": by_source,
            "edge_conflict": conflict,
            "derived": str(row.get("derived", "")).strip().lower() in {
                "true", "1", "yes"},
            "derived_key": row.get("derived_key") or None,
            "source_set_group_id": row.get("source_set_group_id") or None,
            "source_set_group_size": int(row.get("source_set_group_size") or 1),
            "source_set_role": row.get("source_set_role") or None,
            "native_case_id": row.get("native_case_id") or None,
        }

    sources = {"record_dispositions": disposition_source}
    path_to_sources = {}
    for audit_id, record in records.items():
        for edge in record["source_edges"]:
            source_id = edge["source_id"]
            for value in (edge["event_source"], edge["raw_path"]):
                path = _path_key(value)
                if path:
                    path_to_sources.setdefault(path, set()).add(source_id)
        primary_path = _path_key(record.get("primary_raw_source"))
        if primary_path:
            primary_source_id = record.get("primary_source_id")
            if primary_source_id and primary_source_id in record["edges_by_source"]:
                path_to_sources.setdefault(primary_path, set()).add(primary_source_id)
            elif not primary_source_id:
                matches = {edge["source_id"] for edge in record["source_edges"]
                           if _path_key(edge["event_source"]) == primary_path
                           or _path_key(edge["raw_path"]) == primary_path}
                if len(matches) == 1:
                    only = next(iter(matches))
                    path_to_sources.setdefault(primary_path, set()).add(only)

    audit_paths = {}
    if record_sources is not None:
        rows, source = _csv_records(
            record_sources, "record_sources",
            {"audit_id", "source_id", "raw_path", "canonical_path",
             "event_source"})
        sources["record_sources"] = source
        for row in rows:
            audit_id = row["audit_id"]
            if audit_id not in records:
                continue
            for key in (row.get("canonical_path"), row.get("event_source"),
                        row.get("raw_path")):
                path = _path_key(key)
                if path:
                    audit_paths.setdefault(audit_id, set()).add(path)
                    source_id = row.get("source_id")
                    if source_id:
                        path_to_sources.setdefault(path, set()).add(source_id)
    if source_registry is not None:
        rows, source = _csv_records(
            source_registry, "source_registry",
            {"source_id", "canonical_path", "event_sources", "record_ids"})
        sources["source_registry"] = source
        for row in rows:
            source_id = row["source_id"]
            paths = {_path_key(row.get("canonical_path"))}
            paths.update(_path_key(value) for value in _json_list(
                row.get("event_sources"), "source_event_sources"))
            for path in paths:
                if path:
                    path_to_sources.setdefault(path, set()).add(source_id)

    capture_by_id, capture_by_path, path_to_captures = {}, {}, {}
    if captures_index is not None:
        rows, source = _csv_records(
            captures_index, "captures_index",
            {"capture_id", "event_source", "tiers", "record_count"})
        sources["captures_index"] = source
        for row in rows:
            capture_id = row["capture_id"]
            if capture_id in capture_by_id:
                raise ValueError("record_linkage_duplicate_capture_id")
            capture_by_id[capture_id] = {
                "capture_id": capture_id,
                "event_source": row.get("event_source"),
                "tiers": _json_list(row.get("tiers"), "capture_tiers"),
                "record_count": int(row.get("record_count") or 0),
            }
            path = _path_key(row.get("event_source"))
            if path:
                capture_by_path[path] = capture_id
                path_to_captures.setdefault(path, set()).add(capture_id)
    manifest = None
    if captures_manifest is not None:
        manifest, manifest_source = _read_object(
            captures_manifest, "captures_manifest")
        manifest_count = _int_field(manifest, "capture_count", required=True)
        if capture_by_id and len(capture_by_id) != manifest_count:
            raise ValueError("record_linkage_capture_index_manifest_mismatch")
        manifest = {"capture_count": manifest_count, "source": manifest_source}
        sources["captures_manifest"] = manifest_source

    source_records, source_captures = {}, {}
    for audit_id, record in records.items():
        for edge in record["source_edges"]:
            source_records.setdefault(edge["source_id"], set()).add(audit_id)
            source_captures.setdefault(edge["source_id"], set()).add(
                edge["capture_id"])

    def resolve_reference(reference):
        source_id = reference.get("source_id")
        source_path = reference.get("source_path")
        capture_id = reference.get("capture_id")
        candidates = set()
        if source_id:
            candidates.add(source_id)
        elif source_path:
            candidates.update(path_to_sources.get(source_path, ()))
            if not candidates and source_path in capture_by_path:
                capture = capture_by_path[source_path]
                candidates.update(
                    sid for sid, captures in source_captures.items()
                    if capture in captures)
        else:
            return [], "source_edge_unresolved"
        if capture_id:
            filtered = {sid for sid in candidates
                        if capture_id in source_captures.get(sid, set())}
            if not filtered:
                return sorted(candidates), "declared_capture_conflicts_with_source"
            candidates = filtered
        if not candidates:
            return [], "source_not_in_ledger"
        if len(candidates) > 1:
            capture_sets = {
                tuple(sorted(source_captures.get(sid, ())))
                for sid in candidates}
            if len(capture_sets) != 1:
                return sorted(candidates), "ambiguous_source_alias"
        return sorted(candidates), None

    raw_declared = _declared_source_edges(source_edges)
    item_ids = {item.get("item_id") for item in acceptance.get("items", ())}
    observation_to_item, window_to_item = {}, {}
    for item in acceptance.get("items", ()):
        for observation_id in item.get("observation_ids") or ():
            observation_to_item[observation_id] = item.get("item_id")
        for window_id in item.get("observation_window_ids") or ():
            window_to_item[window_id] = item.get("item_id")
        for sample in item.get("samples", ()):
            if sample.get("observation_id"):
                observation_to_item[sample["observation_id"]] = item.get("item_id")
            if sample.get("window_id"):
                window_to_item[sample["window_id"]] = item.get("item_id")
    declared_edges = {}
    for key, edges in raw_declared.items():
        target = key
        if target not in item_ids:
            target = observation_to_item.get(key) or window_to_item.get(key)
        if target is None:
            raise ValueError(f"source_edge_unknown_item_reference:{key}")
        declared_edges.setdefault(target, []).extend(edges)

    def item_actual_edges(item):
        item_id = item.get("item_id")
        declared = declared_edges.get(item_id, ())
        references = []
        if declared:
            references = list(declared)
        else:
            identity = acceptance.get("source_identity") or {}
            source_path = _path_key(identity.get("path"))
            if not source_path:
                return [], ["no_source_identity"], False
            references = [{"source_path": source_path}]
        raw, errors, ambiguous = [], [], False
        for reference in references:
            source_ids, error = resolve_reference(reference)
            if error == "ambiguous_source_alias":
                ambiguous = True
                errors.append(error)
            elif error:
                errors.append(error)
            for source_id in source_ids:
                raw.append((source_id, reference.get("capture_id")))
        return raw, errors, ambiguous

    items = []
    matched_audits, matched_captures, matched_source_sets = set(), set(), set()
    duplicate_records, derived_records, native_cases = set(), set(), set()
    for item in acceptance.get("items", ()):
        source_path = _path_key(
            (acceptance.get("source_identity") or {}).get("path"))
        sample_ids = list(item.get("observation_ids") or ())
        window_ids = list(item.get("observation_window_ids") or ())
        for sample in item.get("samples", ()):
            if sample.get("window_id") and sample["window_id"] not in window_ids:
                window_ids.append(sample["window_id"])
        if not sample_ids:
            sample_ids = [sample.get("observation_id")
                          for sample in item.get("samples", ())
                          if sample.get("observation_id")]
        row = {
            "item_id": item.get("item_id"),
            "request_id": item.get("request_id"),
            "prediction_id": item.get("prediction_id"),
            "source_path": source_path or None,
            "source_capture_id": None,
            "observation_ids": sample_ids,
            "observation_window_ids": window_ids,
            "observation_samples": len(sample_ids),
            "selected_source_edges": [],
            "candidate_audit_ids": [],
            "candidate_capture_ids": [],
            "candidate_source_set_ids": [],
            "matched_audit_ids": [],
            "matched_record_count": 0,
            "independent_capture_count": 0,
            "independent_source_set_count": 0,
            "duplicate_reference_audit_ids": [],
            "derived_audit_ids": [],
            "native_case_ids": [],
            "status": "unmatched",
            "reason": None,
        }
        actual, reasons, ambiguous = item_actual_edges(item)
        if ambiguous:
            row.update(status="ambiguous",
                       reason=",".join(sorted(set(reasons))) or "ambiguous_source_alias",
                       source_candidates=[value[0] for value in actual])
            items.append(row)
            continue
        if not actual:
            row["reason"] = reasons[0] if reasons else "no_source_identity"
            if "declared_capture_conflicts_with_source" in reasons:
                row["status"] = "conflict"
            items.append(row)
            continue
        selected_audits, selected_captures, selected_conflicts = (
            set(), set(), list(reasons))
        selected_edges = []
        for source_id, expected_capture in actual:
            captures = source_captures.get(source_id, set())
            if not captures:
                selected_conflicts.append("source_edge_has_no_capture")
                continue
            if len(captures) > 1:
                selected_conflicts.append("same_source_multiple_captures")
            capture = sorted(captures)[0]
            if expected_capture and expected_capture not in captures:
                selected_conflicts.append("declared_capture_conflicts_with_source")
            for audit_id in source_records.get(source_id, ()):
                record = records[audit_id]
                if record["edge_conflict"]:
                    selected_conflicts.append(
                        f"record_capture_declaration_conflict:{audit_id}")
                    continue
                if capture not in record["edges_by_source"].get(source_id, set()):
                    selected_conflicts.append(
                        f"record_missing_actual_source_edge:{audit_id}")
                    continue
                selected_audits.add(audit_id)
                selected_captures.add(capture)
            selected_edges.append({"source_id": source_id, "capture_id": capture})
        if not selected_edges and not selected_audits:
            row["reason"] = (selected_conflicts[0] if selected_conflicts
                             else "source_not_in_ledger")
            items.append(row)
            continue
        selected = [records[audit_id] for audit_id in sorted(selected_audits)]
        source_sets = sorted({record["source_set_group_id"]
                              for record in selected if record["source_set_group_id"]})
        phases = sorted({record["phase"] for record in selected if record["phase"]})
        identity_phase = item.get("identity", {}).get("phase")
        if identity_phase and phases and identity_phase not in phases:
            selected_conflicts.append("phase_conflict")
        candidate_audits = [record["audit_id"] for record in selected]
        row.update({
            "selected_source_edges": selected_edges,
            "candidate_audit_ids": candidate_audits,
            "candidate_capture_ids": sorted(selected_captures),
            "candidate_source_set_ids": source_sets,
            "phase_values": phases,
        })
        if selected_conflicts:
            row.update(status="conflict",
                       reason=",".join(sorted(set(selected_conflicts))))
        else:
            row.update({
                "status": "matched",
                "matched_audit_ids": candidate_audits,
                "matched_record_count": len(selected),
                "independent_capture_count": len(selected_captures),
                "independent_source_set_count": len(source_sets),
                "duplicate_reference_audit_ids": [
                    record["audit_id"] for record in selected
                    if record["source_set_role"]
                    and record["source_set_role"] != "first_reference"],
                "derived_audit_ids": [record["audit_id"] for record in selected
                                      if record["derived"]],
                "native_case_ids": sorted({
                    record["native_case_id"] for record in selected
                    if record["native_case_id"]}),
            })
            matched_audits.update(row["matched_audit_ids"])
            matched_captures.update(selected_captures)
            matched_source_sets.update(source_sets)
            duplicate_records.update(row["duplicate_reference_audit_ids"])
            derived_records.update(row["derived_audit_ids"])
            native_cases.update(row["native_case_ids"])
        items.append(row)

    paired = int((acceptance.get("counts") or {}).get("paired_items") or 0)
    counts = acceptance.get("counts") or {}
    within = int(counts.get("within") or 0)
    outside = int(counts.get("outside") or 0)
    duration_denominator = int(
        counts.get("duration_denominator_items", within + outside) or 0)
    work_only = int(counts.get("work_only") or 0)
    matched_items = sum(row["status"] == "matched" for row in items)
    unmatched_items = sum(row["status"] == "unmatched" for row in items)
    conflict_items = sum(row["status"] == "conflict" for row in items)
    ambiguous_items = sum(row["status"] == "ambiguous" for row in items)
    status = ("all_paired_items_linked"
              if items and matched_items == len(items) == paired
              else "empty" if not items else "partial")
    return {
        "schema": "mindie-sim.record-linkage/1",
        "status": status,
        "ledger": {
            "records": len(records),
            "a5_records": sum(1 for record in records.values()
                              if record["hardware_tier"] == "A5"),
            "captures_manifest": manifest,
            "captures_index": len(capture_by_id) if capture_by_id else None,
        },
        "comparison": {
            "paired_items": paired,
            "duration_denominator_items": duration_denominator,
            "work_only_items": work_only,
            "matched_items": matched_items,
            "unmatched_items": unmatched_items,
            "conflict_items": conflict_items,
            "ambiguous_items": ambiguous_items,
            "observation_samples": sum(row["observation_samples"] for row in items),
            "matched_observation_samples": sum(
                row["observation_samples"] for row in items
                if row["status"] == "matched"),
        },
        "independent_evidence": {
            "precision_denominator_paired_items": duration_denominator,
            "duration_denominator_items": duration_denominator,
            "paired_items_retained": paired,
            "work_only_items_retained": work_only,
            "record_match_changes_precision_denominator": False,
            "matched_audit_records": len(matched_audits),
            "independent_capture_count": len(matched_captures),
            "independent_source_set_count": len(matched_source_sets),
            "duplicate_reference_records": len(duplicate_records),
            "derived_records": len(derived_records),
            "native_case_ids": sorted(native_cases),
            "statement": (
                "Ledger record matching follows the actual source/capture edge "
                "selected by the comparison. It is identity/coverage linkage, not "
                "per-call certification. Complete source sets and shared sources "
                "are inventory facts; duplicate source-set references and derived "
                "rows do not increase the paired-item precision denominator. The "
                "precision denominator is the real duration denominator "
                "(within+outside); work-only ABI pairs remain associated but add "
                "no duration score."),
        },
        "items": items,
        "unmatched_item_ids": [row["item_id"] for row in items
                               if row["status"] == "unmatched"],
        "conflict_item_ids": [row["item_id"] for row in items
                              if row["status"] == "conflict"],
        "ambiguous_item_ids": [row["item_id"] for row in items
                               if row["status"] == "ambiguous"],
        "sources": sources,
    }
