"""Profiler observation normalizer, cross-rank verification, and raw capture sealer.

[Role] Verify joined ``mindie-profiler`` entries across all participating ranks
  (authoritative level parity, planned ``timed_nodes`` membership & topological
  dependency order, per-rank receipt & ``actual_shape`` / ``node_shapes``
  parity, and pre-hash privacy sanitization), write SHA-256-sealed raw capture
  JSON bytes when ``capture_out_dir`` is given, and emit normalized onboarding
  observation & ABI rejection rows bound to ``sim_case`` and ``measured_binding``.
[Invariants]
  * Never seals or hashes an entry whose level, ``timed_nodes``, or executed
    ``actual_shape`` conflicts with the plan or diverges across ranks.
  * Sanitizes ``actual_shape`` and runs ``assert_sanitized_payload`` on the
    capture payload BEFORE hashing or writing bytes to disk.
  * Embeds ``level``, ``timed_nodes``, ``actual_shape``, and ``plan_key`` into
    sealed raw capture bytes so Stage-3/4 verification cannot be spoofed by
    stripping manifest metadata.
[Timeouts & Locks] Pure in-memory normalization plus bounded atomic capture
  writes strictly inside caller-supplied ``capture_out_dir``; zero locks.
[Side-Effect Policy] Writes raw capture JSON files exclusively under
  ``capture_out_dir`` when explicitly provided.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import statistics
from typing import Any

from mindie_sim.schema.common import ModelingError

from ._profiler_geometry import (
    derive_writer_clock_operator,
    extract_abi_attempted_shape,
    extract_observed_shape,
    extract_shape_dims,
    quantile_ns,
    select_mode_candidates,
    validate_timed_nodes_sequence,
    verify_actual_geometry_matches_plan,
)
from ._profiler_lowering import parse_topo_options
from ._profiler_plan import _DEFAULT_SIM_MODEL, resolve_point_measured_binding
from ._profiler_records import (
    _ABI_REJECT_STATUSES,
    _OBSERVATION_SCHEMA,
    _RECEIPT_SCHEMA,
    _SUBGRAPH_RUN_SCHEMA,
    _collect_abi_and_buckets,
    _critical_path_durations_ns,
    _extract_ranks_from_bucket,
    _join_multirank_obs_entries,
    _lookup_entry_receipts,
    _sanitize_capture_shape,
)
from .contracts import (
    assert_sanitized_payload,
    canonical_json_bytes,
    sanitize_public_alias,
    sanitize_public_run_id,
    write_atomic_bytes,
)

_P99_QUANTILE = 0.99


def _extract_entry_timed_nodes(
    entry_receipts: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    *,
    cid: str,
    level: str,
    is_planned: bool,
    expected_timed_ops: Sequence[str] = (),
    expected_timed_edges: Sequence[Sequence[str]] = (),
) -> list[str]:
    if level == "L1" and not is_planned:
        for src in (*entry_receipts, *summaries):
            if isinstance(src, Mapping):
                tn = src.get("timed_nodes")
                if isinstance(tn, Sequence) and not isinstance(tn, (str, bytes)) and tn:
                    items, _ops = validate_timed_nodes_sequence(tn, cid=cid, level="L1")
                    return items
        return []

    eff_expected = (f"{cid}:op0",) if (level == "L1" and is_planned) else expected_timed_ops
    sources: list[Mapping[str, Any]] = list(entry_receipts)
    if not sources and not is_planned:
        sources = [s for s in summaries if isinstance(s, Mapping)]
    if not sources:
        raise ModelingError(f"profiler_timed_nodes_missing:{cid}:level={level}")

    parsed_lists: list[list[str]] = []
    op_lists: list[list[str]] = []
    for src in sources:
        tn = src.get("timed_nodes") if isinstance(src, Mapping) else None
        items, op_seq = validate_timed_nodes_sequence(
            tn,
            cid=cid,
            level=level,
            expected_ops=eff_expected,
            expected_edges=expected_timed_edges,
            error_prefix="profiler_timed_nodes",
            require_case_prefix=is_planned,
        )
        parsed_lists.append(items)
        op_lists.append(op_seq)

    first_set = frozenset(op_lists[0])
    if any(frozenset(ops) != first_set or len(ops) != len(op_lists[0]) for ops in op_lists[1:]):
        counts = [len(ops) for ops in op_lists]
        raise ModelingError(f"profiler_timed_nodes_rank_mismatch:{cid}:level={level}:counts={counts}")
    for summary in summaries:
        sum_tn = summary.get("timed_nodes") if isinstance(summary, Mapping) else None
        if sum_tn is not None:
            _sum_items, sum_ops = validate_timed_nodes_sequence(
                sum_tn,
                cid=cid,
                level=level,
                expected_ops=eff_expected,
                expected_edges=expected_timed_edges,
                error_prefix="profiler_timed_nodes_summary",
                require_case_prefix=is_planned,
            )
            if frozenset(sum_ops) != first_set:
                raise ModelingError(
                    f"profiler_timed_nodes_summary_mismatch:{cid}:level={level}"
                    f":summary={len(sum_ops)}:receipt={len(first_set)}"
                )
    return parsed_lists[0]


def _verify_entry_receipts_and_shape(
    entry: Mapping[str, Any],
    *,
    entry_receipts: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    first: Mapping[str, Any],
    comp: Mapping[str, Any],
    plan_pt: Mapping[str, Any] | None,
    binding_info: Mapping[str, Any] | None,
    exp_ranks: Sequence[int],
    cid: str,
) -> dict[str, Any]:
    receipt_ranks = {int(r) for r in (entry.get("receipt_ranks") or ())}
    if plan_pt is not None:
        if not entry_receipts or not set(exp_ranks).issubset(receipt_ranks):
            raise ModelingError(
                f"profiler_bench_receipt_missing:{cid}:expected_ranks={list(exp_ranks)}"
                f":got_ranks={sorted(receipt_ranks)}"
            )
    req_shape = extract_observed_shape(
        summary, first, comp, receipts=entry_receipts, allow_requested_fallback=(plan_pt is None),
    )
    if plan_pt is not None and not req_shape:
        raise ModelingError(f"profiler_actual_shape_missing:{cid}")
    if plan_pt is not None and binding_info is not None:
        mb = binding_info["measured_binding"]
        verify_actual_geometry_matches_plan(
            req_shape, plan_pt, expected_shape=binding_info["expected_shape"], measured_binding=mb, cid=cid
        )
        rank_sigs: set[bytes] = set()
        for rcpt in entry_receipts:
            rcpt_shape = extract_observed_shape(
                {}, {}, {}, receipts=(rcpt,), allow_requested_fallback=False,
            )
            if not rcpt_shape:
                raise ModelingError(f"profiler_actual_shape_missing:{cid}:rank={rcpt.get('rank')}")
            verify_actual_geometry_matches_plan(
                rcpt_shape, plan_pt, expected_shape=binding_info["expected_shape"], measured_binding=mb, cid=cid
            )
            rank_sigs.add(canonical_json_bytes(rcpt_shape))
        if len(rank_sigs) > 1:
            raise ModelingError(
                f"profiler_rank_actual_shape_mismatch:{cid}:ranks={sorted(receipt_ranks)}"
            )
    return req_shape


def _resolve_entry_level(
    group: Sequence[Mapping[str, Any]],
    entry_receipts: Sequence[Mapping[str, Any]],
    comps: Sequence[Mapping[str, Any]],
    binding_info: Mapping[str, Any] | None,
    *,
    cid: str,
) -> str:
    seen_levels: set[str] = set()
    for obs in group:
        if bool(obs.get("is_counterfactual")):
            raise ModelingError(f"profiler_counterfactual_forbidden:{cid}")
        if bool(obs.get("debug_cross_level")):
            raise ModelingError(f"profiler_debug_cross_level_forbidden:{cid}")
        raw_top = str(obs.get("level") or "").strip().upper()
        if raw_top:
            if raw_top not in {"L1", "L2", "L3"}:
                raise ModelingError(f"profiler_level_invalid:{cid}:level={raw_top}")
            seen_levels.add(raw_top)
        ident = obs.get("identity") if isinstance(obs.get("identity"), Mapping) else {}
        id_lvl = str(ident.get("level") or "").strip().upper()
        if id_lvl in {"L1", "L2", "L3"}:
            seen_levels.add(id_lvl)
    for src in (*entry_receipts, *comps):
        if isinstance(src, Mapping):
            if bool(src.get("is_counterfactual")):
                raise ModelingError(f"profiler_counterfactual_forbidden:{cid}")
            if bool(src.get("debug_cross_level")):
                raise ModelingError(f"profiler_debug_cross_level_forbidden:{cid}")
            raw_lvl = str(src.get("level") or "").strip().upper()
            if raw_lvl:
                if raw_lvl not in {"L1", "L2", "L3"}:
                    raise ModelingError(f"profiler_level_invalid:{cid}:level={raw_lvl}")
                seen_levels.add(raw_lvl)
    if binding_info is not None:
        known_level = str(binding_info["level"])
        for lvl in sorted(seen_levels):
            if lvl != known_level:
                raise ModelingError(
                    f"profiler_plan_level_mismatch:{cid}:expected={known_level}:got={lvl}"
                )
        return known_level
    if len(seen_levels) > 1:
        raise ModelingError(f"profiler_rank_level_mismatch:{cid}:levels={sorted(seen_levels)}")
    return next(iter(seen_levels)) if seen_levels else ""


def _seal_normalized_entry(
    entry: Mapping[str, Any],
    *,
    plan_doc: Mapping[str, Any],
    plan_pt: Mapping[str, Any] | None,
    sku: str,
    clean_ep: str,
    plan_family: str,
    plan_target_id: str,
    capture_out_dir: str | Path | None,
) -> dict[str, Any] | None:
    from .cliff_planner import validate_parallel_topology

    cid = str(entry["cid"])
    split = str(entry["split"])
    clock_op = str(entry["clock_op"])
    group = entry["group"]
    entry_receipts = list(entry.get("receipts") or ())
    comp = entry["comp"]
    comps = list(entry.get("comps") or ([comp] if comp else []))
    summary = entry["summary"]
    summaries = list(entry.get("summaries") or ([summary] if summary else []))
    ranks = list(entry["ranks"])
    pub_run_id = str(entry["pub_run_id"])
    acq_id = str(entry.get("acq_id") or "").strip()
    first = group[0]

    req_ranks = (
        validate_parallel_topology(str(plan_pt.get("parallel_topology") or "tp1"))
        if plan_pt is not None
        else len(ranks)
    )
    exp_ranks = list(range(req_ranks)) if req_ranks > 1 else ranks
    durations_ns = _critical_path_durations_ns(group, expected_ranks=exp_ranks, cid=cid)
    p50_ns = int(round(statistics.median(durations_ns)))
    p99_ns = quantile_ns(durations_ns, _P99_QUANTILE)
    plan_key = (
        summary.get("plan_key")
        if isinstance(summary.get("plan_key"), Mapping)
        else (first.get("plan_key") if isinstance(first.get("plan_key"), Mapping) else {})
    )
    binding_info = (
        resolve_point_measured_binding(plan_doc, plan_pt, sku=sku)
        if plan_pt is not None
        else None
    )
    known_level = _resolve_entry_level(group, entry_receipts, comps, binding_info, cid=cid)
    req_shape = _verify_entry_receipts_and_shape(
        entry,
        entry_receipts=entry_receipts,
        summary=summary,
        first=first,
        comp=comp,
        plan_pt=plan_pt,
        binding_info=binding_info,
        exp_ranks=exp_ranks,
        cid=cid,
    )
    assert_sanitized_payload(req_shape, field_name=f"actual_shape:{cid}")
    clean_shape = _sanitize_capture_shape(req_shape)
    dims = extract_shape_dims(
        clean_shape,
        plan_key,
        plan_pt=plan_pt,
        expected_shape=binding_info["expected_shape"] if binding_info else None,
        measured_binding=binding_info["measured_binding"] if binding_info else None,
        cid=cid,
    )
    ev_ids = sorted({str(r.get("sample_id") or f"{cid}-s{i}") for i, r in enumerate(group)})
    ident = first.get("identity") if isinstance(first.get("identity"), Mapping) else {}
    mb_map = binding_info["measured_binding"] if binding_info is not None else {}
    timed_nodes = _extract_entry_timed_nodes(
        entry_receipts,
        summaries,
        cid=cid,
        level=known_level or "L1",
        is_planned=(plan_pt is not None),
        expected_timed_ops=mb_map.get("timed_op_ids") or (),
        expected_timed_edges=mb_map.get("timed_edges") or (),
    )
    level = known_level if known_level else ("L1" if len(timed_nodes) <= 1 else "L2")

    if binding_info is not None:
        sim_fam, sim_case, row_phase = (
            binding_info["target_family"], binding_info["sim_case"], binding_info["phase"]
        )
    else:
        sim_fam = str(summary.get("op") or ("" if plan_family == "sku_suite" else plan_family) or "rmsnorm")
        sim_case = {
            "compose_model": _DEFAULT_SIM_MODEL, "compose_hardware": sku,
            "compose_options": parse_topo_options(dims["parallel_topology"]),
        }
        row_phase = str(first.get("phase") or ident.get("phase") or "decode")

    acq_root = sanitize_public_run_id(acq_id) if acq_id else f"acq-{split}-{pub_run_id}"
    cap_payload: dict[str, Any] = {
        "case_id": cid, "run_id": pub_run_id, "acquisition_root": acq_root,
        "split": split, "level": level,
        "sku": str(first.get("sku") or sku),
        "clock_domain": "device_monotonic", "clock_operator": clock_op,
        "start_event": "k_start", "end_event": "k_end",
        "device_ranks": ranks, "event_ids": ev_ids,
        "samples_ns": durations_ns, "measured_ns": p50_ns, "measured_p99_ns": p99_ns,
        "busy_union_ns": p50_ns if clock_op == "busy_union" else None,
        "elapsed_span_ns": p50_ns if clock_op == "elapsed_span" else None,
        "plan_key": dict(plan_key), "actual_shape": dict(clean_shape),
        "timed_nodes": list(timed_nodes), "target_family": sim_fam,
    }
    if binding_info is not None:
        if binding_info["target_op_id"]:
            cap_payload["target_op_id"] = binding_info["target_op_id"]
        if binding_info["target_op_ids"]:
            cap_payload["target_op_ids"] = list(binding_info["target_op_ids"])
        cap_payload["measured_binding"] = dict(binding_info["measured_binding"])
    assert_sanitized_payload(cap_payload, field_name=f"capture_payload:{cid}")
    cap_bytes = canonical_json_bytes(cap_payload)
    raw_sha = hashlib.sha256(cap_bytes).hexdigest()
    cap_uri = f"capture://{clean_ep}/{pub_run_id}/{raw_sha}.json"
    if capture_out_dir is not None:
        write_atomic_bytes(Path(capture_out_dir) / f"{raw_sha}.json", cap_bytes)

    row_out: dict[str, Any] = {
        "case_id": cid, "point_id": cid, "sku": str(first.get("sku") or sku),
        "split": split, "level": level, "phase": row_phase,
        "acquisition_root": acq_root, "run_uid": pub_run_id,
        "raw_sha256": raw_sha, "capture_uri": cap_uri,
        "clock_domain": "device_monotonic", "start_event": "k_start", "end_event": "k_end",
        "event_ids": ev_ids, "device_ranks": ranks, "clock_operator": clock_op,
        "measured_ns": p50_ns, "measured_p99_ns": p99_ns,
        "busy_union_ns": p50_ns if clock_op == "busy_union" else None,
        "elapsed_span_ns": p50_ns if clock_op == "elapsed_span" else None,
        "validated": all(bool(r.get("validated", True)) for r in group),
        "qualified": all(bool(r.get("qualified", True)) for r in group),
        "operator_id": plan_target_id,
        "implementation": str(
            ident.get("implementation") or (binding_info["implementation"] if binding_info else "") or summary.get("op") or "profiler_bench"
        ),
        "upstream_abi": str(plan_key.get("abi") or ident.get("implementation") or ""),
        "target_family": sim_fam, "sim_case": sim_case,
        "plan_key": dict(plan_key), "actual_shape": dict(clean_shape), **dims,
    }
    if timed_nodes:
        row_out["timed_nodes"] = timed_nodes
    if binding_info is not None:
        if binding_info["target_op_id"]:
            row_out["target_op_id"] = binding_info["target_op_id"]
        if binding_info["target_op_ids"]:
            row_out["target_op_ids"] = list(binding_info["target_op_ids"])
        row_out["measured_binding"] = dict(binding_info["measured_binding"])
    return row_out


def normalize_profiler_records(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    plan_doc: Mapping[str, Any],
    sku: str,
    endpoint_alias: str = "",
    capture_out_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert raw ``mindie-profiler`` JSONL/receipt rows into verified onboarding rows & ABI rows."""
    if not raw_records:
        return [], []
    planned_abi_by_id: dict[str, Mapping[str, Any]] = {
        str(pt["point_id"]): pt
        for pt in (plan_doc.get("abi_rejection_points") or ())
        if isinstance(pt, Mapping) and pt.get("point_id")
    }
    is_profiler_schema = any(
        str(r.get("schema") or "") in {_OBSERVATION_SCHEMA, _RECEIPT_SCHEMA, _SUBGRAPH_RUN_SCHEMA}
        or r.get("record_type") in {"observation", "case_complete", "receipt"}
        for r in raw_records
    )
    if not is_profiler_schema:
        obs_out = [dict(r) for r in raw_records if str(r.get("status") or "").lower() not in _ABI_REJECT_STATUSES]
        abi_out = [
            {
                "point_id": str(r.get("point_id") or r.get("case_id") or ""),
                "rejected": True,
                "abi_error": str(r.get("abi_error") or r.get("reason") or ""),
                "executed_shape": extract_abi_attempted_shape(
                    r, planned_abi_by_id.get(str(r.get("point_id") or r.get("case_id") or ""))
                ),
            }
            for r in raw_records
            if str(r.get("status") or "").lower() in _ABI_REJECT_STATUSES
        ]
        return obs_out, abi_out

    clean_ep = sanitize_public_alias(endpoint_alias or sku, sku)
    plan_family = str(plan_doc.get("family") or "").strip()
    plan_target_id = str(plan_doc.get("target_id") or "").strip()
    plan_pts_by_id: dict[str, Mapping[str, Any]] = {}
    for split_key in ("train_points", "heldout_points"):
        for pt in plan_doc.get(split_key) or ():
            if isinstance(pt, Mapping) and pt.get("point_id"):
                plan_pts_by_id[str(pt["point_id"])] = pt

    completes, obs_buckets, receipts_by_key, abi_rows = _collect_abi_and_buckets(
        raw_records, planned_abi_by_id=planned_abi_by_id
    )

    pre_entries: list[dict[str, Any]] = []
    for (cid, base_rid, split), mode_map in sorted(obs_buckets.items()):
        comp_list = completes.get((cid, base_rid)) or []
        comp = comp_list[0] if comp_list else {}
        summaries = [c["summary"] for c in comp_list if isinstance(c.get("summary"), Mapping)]
        summary = summaries[0] if summaries else {}
        plan_pt = plan_pts_by_id.get(cid)
        candidates = [
            (arm, tmode, derive_writer_clock_operator(group, summary, cid=cid, tmode=tmode), group)
            for (arm, tmode), group in sorted(mode_map.items())
        ]
        selected = select_mode_candidates(candidates, cid=cid, plan_pt=plan_pt)
        for _arm, _tmode, clock_op, group in selected:
            first = group[0]
            ranks = _extract_ranks_from_bucket(group, summary)
            sample_acqs = [str(r.get("acquisition_id") or "").strip() for r in group]
            nonempty_sample_acqs = {a for a in sample_acqs if a}
            if len(nonempty_sample_acqs) > 1:
                raise ModelingError(f"profiler_mixed_acquisition_ids:{cid}:{sorted(nonempty_sample_acqs)}")
            if nonempty_sample_acqs and any(not a for a in sample_acqs):
                raise ModelingError(f"profiler_sample_acquisition_id_missing:{cid}")
            comp_acqs = {str(c.get("acquisition_id") or "").strip() for c in comp_list if str(c.get("acquisition_id") or "").strip()}
            all_bucket_acqs = nonempty_sample_acqs | comp_acqs
            if len(all_bucket_acqs) > 1:
                raise ModelingError(f"profiler_mixed_acquisition_ids:{cid}:{sorted(all_bucket_acqs)}")
            acq_id = next(iter(all_bucket_acqs)) if all_bucket_acqs else ""
            obs_plan_id = str(first.get("plan_id") or comp.get("plan_id") or "").strip()
            acq_key = acq_id or obs_plan_id
            matched_rcpts, receipt_ranks = _lookup_entry_receipts(
                receipts_by_key, cid=cid, base_rid=base_rid, ranks=ranks,
                split=split, acq_id=acq_id, plan_id=obs_plan_id,
            )
            rcpt_acqs = {str(r.get("acquisition_id") or "").strip() for r in matched_rcpts if str(r.get("acquisition_id") or "").strip()}
            if len(all_bucket_acqs | rcpt_acqs) > 1:
                raise ModelingError(f"profiler_mixed_acquisition_ids:{cid}:{sorted(all_bucket_acqs | rcpt_acqs)}")
            if not acq_id and rcpt_acqs:
                acq_id = next(iter(rcpt_acqs))
                acq_key = acq_id or obs_plan_id
            pre_entries.append({
                "cid": cid, "base_rid": base_rid, "pub_run_id": sanitize_public_run_id(base_rid),
                "split": split, "clock_op": clock_op, "group": list(group),
                "receipts": matched_rcpts, "receipt_ranks": receipt_ranks,
                "comp": comp, "comps": comp_list, "summary": summary, "summaries": summaries,
                "ranks": ranks, "acq_id": acq_id, "acq_key": acq_key, "multi_rank_joined": False,
            })

    merged_entries = _join_multirank_obs_entries(pre_entries, plan_pts_by_id=plan_pts_by_id)
    normalized: list[dict[str, Any]] = []
    for entry in merged_entries:
        row_out = _seal_normalized_entry(
            entry,
            plan_doc=plan_doc,
            plan_pt=plan_pts_by_id.get(str(entry["cid"])),
            sku=sku,
            clean_ep=clean_ep,
            plan_family=plan_family,
            plan_target_id=plan_target_id,
            capture_out_dir=capture_out_dir,
        )
        if row_out is not None:
            normalized.append(row_out)
    return normalized, abi_rows


__all__ = [
    "normalize_profiler_records",
]
