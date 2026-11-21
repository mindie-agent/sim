"""Public structural coverage separate from prediction accuracy.

[Role] Carry declared omissions, inventory and comparison coverage to prediction reports.
[Invariants] Partial costs cover modeled components. Missing measurements never gate execution.
[Timeouts & Locks] Bounded input reads. No waits or locks.
[Side-Effect Policy] Returns new JSON-ready coverage records.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import csv
from hashlib import sha256
import json
import os
from pathlib import Path

from .inventory import (
    INVENTORY_SCHEMA,
    MAX_IDENTITY_FILE_BYTES,
    MAX_INVENTORY_ROWS,
    MAX_METADATA_BYTES,
    inventory_coverage,
    source_identity,
    _a5_record_recovery,
    _csv_records,
    _int_field,
    _ledger_identity,
    _read_object,
)
from .linkage import (
    record_linkage,
    _declared_source_edges,
    _edge_list,
    _json_list,
    _path_key,
)


COMPARISON_COVERAGE_SCHEMA = "mindie-sim.comparison-coverage/1"


def model_coverage(model):
    metadata = dict(model.metadata)
    coverage = metadata.get("model_coverage", "complete")
    return {
        "model_coverage": coverage,
        "unmodeled_features": list(metadata.get("unmodeled_features", ())),
        "degraded_submodules": list(metadata.get("degraded_submodules", ())),
        "prediction_scope": ("known_components_with_declared_degradations"
                             if coverage == "partial" else "declared_model_structure"),
    }


def combine_coverage(models):
    reports = [model_coverage(model) for model in models]
    partial = any(report["model_coverage"] == "partial" for report in reports)
    return {
        "model_coverage": "partial" if partial else "complete",
        "unmodeled_features": sorted({gap for report in reports for gap in report["unmodeled_features"]}),
        "degraded_submodules": sorted({gap for report in reports for gap in report["degraded_submodules"]}),
        "prediction_scope": ("known_components_with_declared_degradations"
                             if partial else "declared_model_structure"),
    }


def comparison_coverage(inventory, acceptance, *, linkage=None):
    """Keep the comparison denominator distinct from the 7337-ledger inventory."""
    if not isinstance(inventory, Mapping) or not isinstance(acceptance, Mapping):
        raise ValueError("comparison_coverage_requires_inventory_and_acceptance")
    ledger = inventory.get("ledger")
    counts = acceptance.get("counts")
    if not isinstance(ledger, Mapping) or not isinstance(counts, Mapping):
        raise ValueError("comparison_coverage_inputs_are_incomplete")
    total = ledger.get("total_records")
    return {
        "schema": COMPARISON_COVERAGE_SCHEMA,
        "ledger_records": total,
        "ledger_a5_records": ledger.get("a5_scope_records"),
        "ledger_captures_total": ledger.get("captures_total"),
        "ledger_a5_event_captures": (ledger.get("a5_unique_event_captures")
                                     if ledger.get("a5_unique_event_captures") is not None
                                     else ledger.get("a5_capture_count")),
        "comparison": {
            "paired_items": counts.get("paired_items"),
            "duration_denominator_items": counts.get("duration_denominator_items"),
            "work_only": counts.get("work_only"),
            "within": counts.get("within"),
            "outside": counts.get("outside"),
            "observation_samples": counts.get("observation_samples"),
            "unpaired_observations": counts.get("unpaired_observations"),
            "unpaired_predictions": counts.get("unpaired_predictions"),
            "ambiguous": counts.get("ambiguous"),
            "condition_gap_items_in_denominator": counts.get(
                "condition_gap_items_in_denominator"),
            "source_status": acceptance.get("source_status"),
            "partial_coverage": acceptance.get("partial_coverage"),
            "truncated": acceptance.get("truncated"),
        },
        "source_coverage": {
            "source_status": acceptance.get("source_status"),
            "partial_coverage": acceptance.get("partial_coverage"),
            "truncated": acceptance.get("truncated"),
            "window": acceptance.get("window"),
            "window_count": len(acceptance.get("windows") or ()),
            "source_identity": acceptance.get("source_identity"),
            "observation_counts": acceptance.get("observation_counts"),
        },
        "inventory_linkage": (
            {
                "status": linkage["status"],
                "record_match_scope": "ledger_identity_link_only_not_call_certification",
                "paired_items": linkage["comparison"]["paired_items"],
                "matched_items": linkage["comparison"]["matched_items"],
                "unmatched_items": linkage["comparison"]["unmatched_items"],
                "conflict_items": linkage["comparison"]["conflict_items"],
                "matched_inventory_records": linkage["independent_evidence"][
                    "matched_audit_records"],
                "independent_capture_count": linkage["independent_evidence"][
                    "independent_capture_count"],
                "independent_source_set_count": linkage["independent_evidence"][
                    "independent_source_set_count"],
                "duplicate_reference_records": linkage["independent_evidence"][
                    "duplicate_reference_records"],
                "derived_records": linkage["independent_evidence"]["derived_records"],
                "precision_denominator_paired_items": linkage["independent_evidence"][
                    "precision_denominator_paired_items"],
                "duration_denominator_items": linkage["independent_evidence"][
                    "duration_denominator_items"],
                "paired_items_retained": linkage["independent_evidence"][
                    "paired_items_retained"],
                "work_only_items_retained": linkage["independent_evidence"][
                    "work_only_items_retained"],
                "record_match_changes_precision_denominator": False,
            }
            if linkage is not None else {
                "status": "not_declared",
                "reason": "record_dispositions_not_supplied",
                "matched_inventory_records": 0,
            }),
        "validated_fraction_of_ledger": None,
        "ledger_records_not_claimed_by_this_slice": total,
        "statement": (
            "This independent native slice validates only its duration-scored "
            "paired items under the source formal status and coverage. Work-only "
            "ABI pairs are retained as associations but add no duration verdict. "
            "Record matching is identity linkage, not per-call certification; it "
            "is not a 7337-record, duplicate-collapsed or heldout accuracy "
            "fraction."),
    }


__all__ = [
    "COMPARISON_COVERAGE_SCHEMA",
    "Counter",
    "INVENTORY_SCHEMA",
    "MAX_IDENTITY_FILE_BYTES",
    "MAX_INVENTORY_ROWS",
    "MAX_METADATA_BYTES",
    "Mapping",
    "Path",
    "annotations",
    "combine_coverage",
    "comparison_coverage",
    "csv",
    "inventory_coverage",
    "json",
    "model_coverage",
    "os",
    "record_linkage",
    "sha256",
    "source_identity",
]
