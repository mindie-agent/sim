"""Coverage and profiler correspondence for declared native calls.

[Role] Count the current normal emission of native-call boundaries and join
  captured profiler rows to the declared entry/kernel names they correspond to.
[Invariants] Only supplied actual emitted operators/records form the denominator;
  a self-written catalogue is never coverage.  ``bound`` requires a source
  declared/established entry, symbol or kernel plus a source sha256 and typed
  ABI.  A name match is correspondence evidence, never occurrence proof.
[Timeouts & Locks] Pure finite metadata arithmetic; no I/O, device or locks.
[Side-Effect Policy] Returns fresh dicts only; never mutates records or imports
  a hardware runtime, weights, the profiler or a model catalogue.
"""
from __future__ import annotations

from mindie_sim.schema.common import ModelingError


def _bound_state(evidence, source_sha256, typed):
    """True bound = callable/entry OR declared kernel, plus source and typed ABI."""
    has_call = (evidence.get("entry") in {"established", "source_declared"}
                or evidence.get("aclnn_symbol") in {"established", "source_declared"}
                or evidence.get("kernels") in {"source_declared", "partial"})
    reasons = []
    if not has_call:
        reasons.append("no_bound_entry_or_kernel")
    if not source_sha256:
        reasons.append("source_sha256_unresolved")
    if not isinstance(typed, dict) or not typed.get("signature_hash"):
        reasons.append("typed_abi_unresolved")
    return has_call and not reasons, reasons


def native_call_coverage(ops, *, bindings=None):
    """Exact per-source-family denominator for the current normal emission.

    Never treats a self-written catalogue as coverage: only the supplied actual
    emitted operators are counted.  ``bound`` requires an established or
    source-declared entry/symbol/kernel, not merely a returned record.
    """
    ops = tuple(ops)
    grouped = {}
    for index, op in enumerate(ops):
        binding = (bindings or {}).get(index) if bindings else None
        if bindings is None:
            from .native_binding import bind_native_call

            binding = bind_native_call(op)
        key = (op.family, op.source_boundary or "analytical", op.recipe or "-")
        row = grouped.setdefault(key, {
            "family": key[0], "source_boundary": key[1], "recipe": key[2],
            "emitted": 0, "bound": 0, "unresolved": 0,
            "with_entry": 0, "with_source_sha256": 0, "with_kernels": 0,
            "with_core": 0, "fully_paired": 0,
            "evidence": {}, "unbound_reasons": {},
        })
        row["emitted"] += 1
        if binding is None:
            row["unresolved"] += 1
            row["unbound_reasons"]["no_binding_record"] = (
                row["unbound_reasons"].get("no_binding_record", 0) + 1)
            continue
        evidence = binding.evidence
        is_bound, reasons = _bound_state(
            evidence, (binding.source or {}).get("sha256"), binding.typed)
        row["bound" if is_bound else "unresolved"] += 1
        row["fully_paired"] += int(
            is_bound and "installed_selected_binary_not_established"
            not in binding.unresolved)
        for reason in reasons:
            row["unbound_reasons"][reason] = row["unbound_reasons"].get(reason, 0) + 1
        row["with_entry"] += int(bool(binding.api.get("entry")))
        row["with_source_sha256"] += int(bool((binding.source or {}).get("sha256")))
        row["with_kernels"] += int(bool(binding.kernels))
        row["with_core"] += int(binding.execution_core is not None)
        for key_name, value in evidence.items():
            counter = row["evidence"].setdefault(key_name, {})
            counter[value] = counter.get(value, 0) + 1
    families = {}
    for row in grouped.values():
        family = families.setdefault(row["family"], {
            "family": row["family"], "emitted": 0, "bound": 0, "unresolved": 0})
        for key in ("emitted", "bound", "unresolved"):
            family[key] += row[key]
    return {
        "schema": "mindie-sim/native-call-coverage/1",
        "denominator": {"emitted_operators": len(ops),
                        "distinct_source_families": len(grouped),
                        "distinct_families": len(families)},
        "totals": {
            "bound": sum(row["bound"] for row in grouped.values()),
            "unresolved": sum(row["unresolved"] for row in grouped.values()),
            "with_entry": sum(row["with_entry"] for row in grouped.values()),
            "with_source_sha256": sum(row["with_source_sha256"] for row in grouped.values()),
            "with_kernels": sum(row["with_kernels"] for row in grouped.values()),
            "with_core": sum(row["with_core"] for row in grouped.values()),
            "fully_paired": sum(row["fully_paired"] for row in grouped.values()),
        },
        "by_family": [families[key] for key in sorted(families)],
        "by_source_family": [grouped[key] for key in sorted(grouped)],
        "unresolved_source_families": sorted(
            key for key, row in grouped.items() if row["unresolved"]),
        "non_claims": ["actual_normal_emission_only", "no_catalogue_counted_as_coverage"],
    }


def coverage_from_records(records, *, emitted=None):
    """Coverage of actual exported nodes; unrecorded emissions stay in the denominator.

    ``records`` are the BIND1 records present on the emitted structure.
    ``emitted`` is the actual emitted node/rank/operator count; when it exceeds
    ``len(records)`` the difference is explicitly unresolved rather than
    silently shrinking the denominator.
    """
    records = [record for record in records if isinstance(record, dict)]
    grouped = {}
    for record in records:
        key = (record.get("family"), record.get("source_boundary") or "analytical",
               record.get("recipe") or "-")
        row = grouped.setdefault(key, {
            "family": key[0], "source_boundary": key[1], "recipe": key[2],
            "emitted": 0, "bound": 0, "unresolved": 0,
            "with_entry": 0, "with_source_sha256": 0, "with_kernels": 0,
            "with_core": 0, "fully_paired": 0, "unbound_reasons": {}})
        row["emitted"] += 1
        evidence = record.get("evidence") or {}
        is_bound, reasons = _bound_state(
            evidence, (record.get("source") or {}).get("sha256"), record.get("typed"))
        row["bound" if is_bound else "unresolved"] += 1
        row["fully_paired"] += int(
            is_bound and "installed_selected_binary_not_established"
            not in (record.get("unresolved") or ()))
        for reason in reasons:
            row["unbound_reasons"][reason] = row["unbound_reasons"].get(reason, 0) + 1
        api = record.get("api") or {}
        row["with_entry"] += int(bool(api.get("entry")))
        row["with_source_sha256"] += int(bool((record.get("source") or {}).get("sha256")))
        row["with_kernels"] += int(bool(record.get("kernels")))
        row["with_core"] += int(record.get("execution_core") is not None)
    denominator = len(records) if emitted is None else int(emitted)
    if denominator < len(records):
        raise ModelingError("coverage_emitted_below_recorded")
    unrecorded = denominator - len(records)
    bound = sum(row["bound"] for row in grouped.values())
    unresolved_families = [key for key, row in grouped.items() if row["unresolved"]]
    if unrecorded:
        unresolved_families.append(("__unrecorded__", "none", "no_binding_record"))
    return {
        "schema": "mindie-sim/native-call-coverage/1",
        "denominator": {"emitted_operators": denominator,
                        "recorded_bindings": len(records),
                        "unrecorded_emitted": unrecorded,
                        "distinct_source_families": len(grouped)},
        "totals": {
            "bound": bound,
            "unresolved": denominator - bound,
            "with_entry": sum(row["with_entry"] for row in grouped.values()),
            "with_source_sha256": sum(row["with_source_sha256"] for row in grouped.values()),
            "with_kernels": sum(row["with_kernels"] for row in grouped.values()),
            "with_core": sum(row["with_core"] for row in grouped.values()),
            "fully_paired": sum(row["fully_paired"] for row in grouped.values()),
        },
        "by_source_family": [grouped[key] for key in sorted(grouped)],
        "unresolved_source_families": sorted(
            unresolved_families, key=lambda key: tuple(str(value) for value in key)),
        "non_claims": ["actual_exported_nodes_only", "no_catalogue_counted_as_coverage"],
    }


def join_profiler_rows(rows, bindings):
    """Join captured kernel rows to bindings by declared entry/kernel name.

    ``rows`` are mapping rows (at least ``name``/``op_type``).  The result is a
    pure descriptive join: a name match is correspondence evidence, not proof
    that the measured occurrence is the declared node.
    """
    kernels, entries = {}, {}
    for binding in bindings:
        if binding is None:
            continue
        if binding.api.get("entry"):
            entries.setdefault(binding.api["entry"], []).append(binding)
        for kernel in binding.kernels:
            kernels.setdefault(kernel["name"], []).append(binding)
    joined, unmatched = [], []
    for row in rows:
        name = str(row.get("name") or row.get("op_type") or "")
        candidates = kernels.get(name) or entries.get(name) or []
        record = {
            "row_name": name, "call_ids": sorted({c.call_id for c in candidates}),
            "basis": ("declared_child_kernel" if kernels.get(name)
                      else "declared_entry" if entries.get(name) else "unmapped"),
            "tensor_match_implied": False,
        }
        (joined if candidates else unmatched).append(record)
    return {
        "schema": "mindie-sim/native-call-profiler-join/1",
        "rows": len(rows), "joined": len(joined), "unmatched": len(unmatched),
        "records": joined, "unmatched_rows": unmatched,
        "non_claims": ["correspondence_not_occurrence_proof"],
    }
