"""Live ``mindie-profiler`` plan-consuming dispatch and real observation/ABI normalizer.

[Role] Dispatch translated L1/L2/L3 ``mindie-bench-subgraph-plan/1`` plans with
  ``train`` and heldout levels separated into independent acquisitions via
  ``mindie_profiler bench subgraph --plan`` (launching one worker per granted
  rank with a shared acquisition ID whenever ``world_size > 1``, routing every
  non-local ``endpoint_alias`` through sanctioned SSH transport, and reserving
  ``bench sku --domain`` strictly for Domain-5 fabric probes), collect all
  unsuffixed and ``.rankN`` observations/receipts, validate EVERY rank's ABI
  error class and attempted geometry before acceptance or deduplication, join
  multi-rank writer outputs across disjoint ranks, verify builder-reported
  ``actual_shape`` against the plan before sealing, and emit ``sim_case`` +
  ``measured_binding`` onboarding rows.
[Invariants]
  * Never overwrites writer-measured ``clock_operator`` or builder-reported
    ``actual_shape`` with the plan; fails closed before sealing any capture.
  * Non-local ``endpoint_alias`` ALWAYS routes over SSH, never local Python.
  * Train and heldout (and SKU L1/L2/L3 heldout levels) dispatch as independent
    acquisitions with disjoint run IDs and acquisition roots.
  * Multi-rank ``world_size > 1`` launches ``world_size`` parallel workers
    sharing one ``acquisition_id`` and joins only pairwise-disjoint rank sets.
  * Accepts ``rc == 2`` (``partial``) ONLY when every failing record on every
    rank matches a planned ABI rejection's exact error class and attempted shape.
  * ``TimeoutError`` always propagates before ``OSError``; windowless spawn.
[Timeouts & Locks] Synchronous bounded subprocesses; split acquisition timeouts
  sum to ``<= timeout_s``; multi-rank workers share one deadline and always reap
  in ``finally``; no locks.
[Side-Effect Policy] Writes ephemeral plan/output temp files cleaned up
  unconditionally; writes raw capture JSON bytes only to explicit ``capture_out_dir``.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

from mindie_sim.schema.common import ModelingError

from ._profiler_plan import (
    _DEFAULT_SIM_MODEL,
    assign_plan_point_level,
    build_profiler_subgraph_plan,
    derive_writer_clock_operator,
    extract_abi_attempted_shape,
    extract_observed_shape,
    extract_shape_dims,
    parse_run_id_and_arm,
    parse_topo_options,
    quantile_ns,
    resolve_point_measured_binding,
    select_mode_candidates,
    verify_actual_geometry_matches_plan,
)
from ._soc_probe import run_ssh_command
from .contracts import (
    DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
    NS_PER_MS,
    SHORT_DIGEST_LEN,
    canonical_json_bytes,
    sanitize_public_alias,
    sanitize_public_run_id,
    subprocess_window_kwargs,
    write_atomic_bytes,
)

_P99_QUANTILE = 0.99
_PARTIAL_EXIT_CODE = 2
_MAX_UNEXPECTED_REPORTED = 8
_ACQ_ID_HEX_LEN = 12
_PROC_REAP_GRACE_S = 5.0
_MIN_POLL_REMAINING_S = 0.05
_LOCAL_ENDPOINT_ALIASES = frozenset({"", "local", "localhost", "127.0.0.1", "::1"})
_ABI_REJECT_STATUSES = frozenset({"abi_rejected", "unsupported_binding"})
_PASSING_RECEIPT_STATUSES = frozenset({"ok", "skipped_stage", "skipped_non_participant"})
_ACCEPTED_ABI_STATUSES = frozenset({"ok", "unqualified"})
_OBSERVATION_SCHEMA = "mindie-bench-observation/1"
_RECEIPT_SCHEMA = "mindie-bench-receipt/1"
_SUBGRAPH_RUN_SCHEMA = "mindie-bench-subgraph-run/1"
_REMOTE_RECEIPT_DELIM = "__MINDIE_RECEIPT_JSON__"
_RANK_FILENAME_RE = re.compile(r"\.rank(\d+)\.(?:json|jsonl)$")


def _parse_rank_hint(filename: str) -> int | None:
    match = _RANK_FILENAME_RE.search(str(filename or ""))
    return int(match.group(1)) if match else None


def _is_remote_endpoint(endpoint_alias: str) -> bool:
    return endpoint_alias.strip().lower() not in _LOCAL_ENDPOINT_ALIASES


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _propagate_envelope_fields(
    item: dict[str, Any],
    envelope: Mapping[str, Any],
    *,
    key: str,
    rank_hint: int | None,
) -> dict[str, Any]:
    for field in ("run_id", "plan_id", "stage", "acquisition_id"):
        if envelope.get(field) and not item.get(field):
            item[field] = envelope[field]
    env_rank = envelope.get("rank")
    eff_rank = (
        int(env_rank)
        if isinstance(env_rank, int) and not isinstance(env_rank, bool)
        else rank_hint
    )
    if eff_rank is not None and item.get("rank") is None:
        item["rank"] = eff_rank
    if key != "observations" and not item.get("record_type"):
        item["record_type"] = "receipt"
    return item


def _parse_json_or_jsonl_records(
    text: str, *, rank_hint: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return [], {}
    try:
        raw = json.loads(stripped)
    except ValueError:
        rows: list[dict[str, Any]] = []
        for idx, line in enumerate(stripped.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                raise ModelingError(f"profiler_bench_invalid_jsonl:line={idx}:{exc}") from exc
            if not isinstance(rec, Mapping):
                raise ModelingError(f"profiler_bench_jsonl_record_not_mapping:line={idx}")
            item = dict(rec)
            if rank_hint is not None and item.get("rank") is None:
                item["rank"] = rank_hint
            rows.append(item)
        return rows, {}
    if isinstance(raw, Mapping):
        collected: list[dict[str, Any]] = []
        for key in ("observations", "records", "cases", "receipts", "abi_rejections"):
            val = raw.get(key)
            if isinstance(val, list):
                collected.extend(
                    _propagate_envelope_fields(dict(r), raw, key=key, rank_hint=rank_hint)
                    for r in val
                    if isinstance(r, Mapping)
                )
        if isinstance(raw.get("receipt"), Mapping):
            sub_env = raw["receipt"]
            for key in ("cases", "records", "receipts", "abi_rejections"):
                val = sub_env.get(key)
                if isinstance(val, list):
                    collected.extend(
                        _propagate_envelope_fields(dict(r), sub_env, key=key, rank_hint=rank_hint)
                        for r in val
                        if isinstance(r, Mapping)
                    )
        if not collected and (raw.get("record_type") or raw.get("case_id")):
            single = dict(raw)
            if rank_hint is not None and single.get("rank") is None:
                single["rank"] = rank_hint
            if not single.get("record_type") and str(single.get("schema") or "") in {_RECEIPT_SCHEMA, _SUBGRAPH_RUN_SCHEMA}:
                single["record_type"] = "receipt"
            collected.append(single)
        meta = dict(raw)
        if rank_hint is not None and meta.get("rank") is None:
            meta["rank"] = rank_hint
        return collected, meta
    if not isinstance(raw, list):
        raise ModelingError("profiler_bench_observations_not_list")
    return [dict(r) for r in raw if isinstance(r, Mapping)], {}


def _extract_ranks_from_bucket(group: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> list[int]:
    ranks_raw = summary.get("participating_ranks")
    if isinstance(ranks_raw, Sequence) and not isinstance(ranks_raw, (str, bytes)) and ranks_raw:
        parsed = sorted({int(r) for r in ranks_raw if not isinstance(r, bool)})
        if parsed:
            return parsed
    obs_ranks = sorted({
        int(r["rank"]) for r in group
        if r.get("rank") is not None and not isinstance(r.get("rank"), bool)
    })
    return obs_ranks if obs_ranks else [0]


def _critical_path_durations_ns(
    group: Sequence[Mapping[str, Any]], *, expected_ranks: Sequence[int], cid: str
) -> list[int]:
    if len(expected_ranks) <= 1:
        return sorted(
            int(round(float(r["duration_ms"]) * NS_PER_MS))
            for r in group
            if r.get("duration_ms") is not None and float(r["duration_ms"]) > 0.0
        )
    expected_set = {int(r) for r in expected_ranks}
    by_sample_slot: dict[tuple[int, int], dict[int, int]] = {}
    for row in group:
        raw_dur = row.get("duration_ms")
        if raw_dur is None or isinstance(raw_dur, bool) or float(raw_dur) <= 0.0:
            raise ModelingError(f"multirank_sample_duration_invalid:{cid}")
        rk = row.get("rank")
        trial = row.get("trial")
        rep = row.get("repeat_index")
        if (
            rk is None or isinstance(rk, bool) or not isinstance(rk, int)
            or trial is None or isinstance(trial, bool) or not isinstance(trial, int)
            or rep is None or isinstance(rep, bool) or not isinstance(rep, int)
        ):
            raise ModelingError(f"multirank_sample_coordinates_missing:{cid}")
        if rk not in expected_set:
            raise ModelingError(
                f"multirank_sample_unexpected_rank:{cid}:rank={rk}:expected={sorted(expected_set)}"
            )
        slot_key = (trial, rep)
        slot_ranks = by_sample_slot.setdefault(slot_key, {})
        if rk in slot_ranks:
            raise ModelingError(f"multirank_sample_duplicate_rank:{cid}:slot={slot_key}:rank={rk}")
        slot_ranks[rk] = int(round(float(raw_dur) * NS_PER_MS))
    if not by_sample_slot:
        raise ModelingError(f"multirank_incomplete_rank_samples:{cid}:empty")
    for slot_key, slot_ranks in sorted(by_sample_slot.items()):
        if set(slot_ranks) != expected_set:
            missing = sorted(expected_set - set(slot_ranks))
            raise ModelingError(
                f"multirank_incomplete_rank_samples:{cid}:slot={slot_key}:missing_ranks={missing}"
            )
    return sorted(max(slot_ranks.values()) for slot_ranks in by_sample_slot.values())


def _lookup_entry_receipts(
    receipts_by_key: Mapping[tuple[str, str, int | None], list[Mapping[str, Any]]],
    *,
    cid: str,
    base_rid: str,
    ranks: Sequence[int],
) -> list[Mapping[str, Any]]:
    matched: list[Mapping[str, Any]] = []
    seen_ids: set[int] = set()
    keys: list[tuple[str, str, int | None]] = [(cid, base_rid, int(rk)) for rk in ranks]
    keys.append((cid, base_rid, None))
    for key in keys:
        for rcpt in receipts_by_key.get(key) or ():
            obj_id = id(rcpt)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                matched.append(rcpt)
    return matched


def _join_multirank_obs_entries(
    raw_entries: list[dict[str, Any]],
    *,
    plan_pts_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from .cliff_planner import validate_parallel_topology

    passthrough: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for entry in raw_entries:
        cid = entry["cid"]
        plan_pt = plan_pts_by_id.get(cid)
        req_ranks = validate_parallel_topology(str((plan_pt or {}).get("parallel_topology") or "tp1"))
        if req_ranks <= 1:
            passthrough.append(entry)
            continue
        group_key = entry["acq_key"]
        grouped.setdefault((cid, entry["split"], group_key), []).append(entry)

    joined_out: list[dict[str, Any]] = list(passthrough)
    for (cid, split, group_key), members in sorted(grouped.items()):
        if len(members) == 1:
            joined_out.append(members[0])
            continue
        seen_ranks: set[int] = set()
        disjoint = True
        for m in members:
            m_ranks = set(m["ranks"])
            if seen_ranks & m_ranks:
                disjoint = False
                break
            seen_ranks |= m_ranks
        if not disjoint:
            joined_out.extend(members)
            continue
        clocks = {m["clock_op"] for m in members}
        if len(clocks) > 1:
            raise ModelingError(f"profiler_writer_clock_operator_conflict:{cid}:{sorted(clocks)}")
        merged_group: list[Mapping[str, Any]] = []
        merged_receipts: list[Mapping[str, Any]] = []
        base_rids: list[str] = []
        for m in members:
            merged_group.extend(m["group"])
            merged_receipts.extend(m.get("receipts") or ())
            base_rids.append(m["base_rid"])
        acq_id = next((m["acq_id"] for m in members if m["acq_id"]), "")
        if acq_id:
            pub_run_id = sanitize_public_run_id(acq_id)
        else:
            digest = hashlib.sha256(
                canonical_json_bytes([group_key, cid, split, sorted(base_rids)])
            ).hexdigest()[:SHORT_DIGEST_LEN]
            pub_run_id = f"mrank-{digest}"
        joined_out.append({
            "cid": cid,
            "base_rid": pub_run_id,
            "pub_run_id": pub_run_id,
            "split": split,
            "clock_op": next(iter(clocks)),
            "group": merged_group,
            "receipts": merged_receipts,
            "comp": members[0]["comp"],
            "summary": members[0]["summary"],
            "ranks": sorted(seen_ranks),
            "acq_id": acq_id,
            "acq_key": group_key,
            "multi_rank_joined": True,
        })
    return joined_out


def _collect_abi_and_buckets(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    planned_abi_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    dict[tuple[str, str, str], dict[tuple[str, str], list[Mapping[str, Any]]]],
    dict[tuple[str, str, int | None], list[Mapping[str, Any]]],
    list[dict[str, Any]],
]:
    completes: dict[tuple[str, str], Mapping[str, Any]] = {}
    obs_buckets: dict[tuple[str, str, str], dict[tuple[str, str], list[Mapping[str, Any]]]] = {}
    receipts_by_key: dict[tuple[str, str, int | None], list[Mapping[str, Any]]] = {}
    abi_rows: list[dict[str, Any]] = []
    seen_abi_tuples: set[tuple[str, str, bytes]] = set()

    for rec in raw_records:
        rtype = str(rec.get("record_type") or "")
        status = str(rec.get("status") or "ok").strip().lower()
        cid = str(rec.get("case_id") or rec.get("point_id") or "").strip()
        raw_run_id = str(rec.get("run_id") or "run-0").strip()

        if cid in planned_abi_by_id and (rtype == "observation" or status in _ACCEPTED_ABI_STATUSES):
            exp_err = str(planned_abi_by_id[cid].get("abi_expected_error") or "")
            raise ModelingError(
                f"profiler_abi_rejection_unexpectedly_accepted:{cid}:expected={exp_err}"
            )
        if status in _ABI_REJECT_STATUSES:
            if cid:
                err_text = str(rec.get("abi_error") or rec.get("reason") or "").strip()
                attempted = extract_abi_attempted_shape(rec, planned_abi_by_id.get(cid))
                shape_sig = canonical_json_bytes(attempted) if attempted is not None else b"null"
                dedup_key = (cid, err_text, shape_sig)
                if dedup_key not in seen_abi_tuples:
                    seen_abi_tuples.add(dedup_key)
                    abi_rows.append({
                        "point_id": cid, "rejected": True,
                        "abi_error": err_text,
                        "executed_shape": attempted,
                    })
            continue
        if cid in planned_abi_by_id and status not in _PASSING_RECEIPT_STATUSES:
            abi_rows.append({
                "point_id": cid, "rejected": True,
                "abi_error": str(rec.get("abi_error") or rec.get("reason") or status).strip(),
                "executed_shape": extract_abi_attempted_shape(rec, planned_abi_by_id.get(cid)),
            })
            continue
        if rtype == "case_complete":
            base_rid, _arm = parse_run_id_and_arm(raw_run_id)
            completes[(cid, base_rid)] = rec
        elif rtype == "observation":
            split = "heldout" if str(rec.get("split") or "") == "heldout" else "train"
            tmode = str(rec.get("timing_mode") or "default").strip()
            exec_map = rec.get("execution") if isinstance(rec.get("execution"), Mapping) else {}
            base_rid, arm = parse_run_id_and_arm(
                raw_run_id, tmode, rec.get("trial"), str(rec.get("source") or ""), exec_map
            )
            obs_buckets.setdefault((cid, base_rid, split), {}).setdefault((arm, tmode), []).append(rec)
        elif cid:
            if status not in _PASSING_RECEIPT_STATUSES:
                raise ModelingError(f"profiler_bench_receipt_failed:{cid}:{status}")
            base_rid, _arm = parse_run_id_and_arm(raw_run_id)
            raw_rk = rec.get("rank")
            rk_val = int(raw_rk) if isinstance(raw_rk, int) and not isinstance(raw_rk, bool) else None
            receipts_by_key.setdefault((cid, base_rid, rk_val), []).append(rec)
    return completes, obs_buckets, receipts_by_key, abi_rows


def _extract_entry_timed_nodes(
    entry_receipts: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> list[str]:
    for src in (*entry_receipts, summary):
        if isinstance(src, Mapping):
            tn = src.get("timed_nodes")
            if isinstance(tn, Sequence) and not isinstance(tn, (str, bytes)) and tn:
                return [str(x) for x in tn]
    return []


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
    summary = entry["summary"]
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
    if not durations_ns:
        return None
    p50_ns = int(round(statistics.median(durations_ns)))
    p99_ns = quantile_ns(durations_ns, _P99_QUANTILE)
    plan_key = (
        summary.get("plan_key")
        if isinstance(summary.get("plan_key"), Mapping)
        else (first.get("plan_key") if isinstance(first.get("plan_key"), Mapping) else {})
    )
    req_shape = extract_observed_shape(summary, first, comp, receipts=entry_receipts)
    binding_info = (
        resolve_point_measured_binding(plan_doc, plan_pt, sku=sku)
        if plan_pt is not None
        else None
    )
    if plan_pt is not None and binding_info is not None:
        verify_actual_geometry_matches_plan(
            req_shape, plan_pt, expected_shape=binding_info["expected_shape"], cid=cid
        )
        for rcpt in entry_receipts:
            rcpt_shape = extract_observed_shape({}, {}, {}, receipts=(rcpt,))
            if rcpt_shape:
                verify_actual_geometry_matches_plan(
                    rcpt_shape, plan_pt, expected_shape=binding_info["expected_shape"], cid=cid
                )
    dims = extract_shape_dims(
        req_shape,
        plan_key,
        plan_pt=plan_pt,
        expected_shape=binding_info["expected_shape"] if binding_info else None,
    )
    ev_ids = sorted({str(r.get("sample_id") or f"{cid}-s{i}") for i, r in enumerate(group)})
    timed_nodes = _extract_entry_timed_nodes(entry_receipts, summary)

    cap_payload: dict[str, Any] = {
        "case_id": cid, "run_id": pub_run_id, "split": split, "sku": str(first.get("sku") or sku),
        "clock_domain": "device_monotonic", "clock_operator": clock_op,
        "start_event": "k_start", "end_event": "k_end",
        "device_ranks": ranks, "event_ids": ev_ids,
        "samples_ns": durations_ns, "measured_ns": p50_ns, "measured_p99_ns": p99_ns,
        "busy_union_ns": p50_ns if clock_op == "busy_union" else None,
        "elapsed_span_ns": p50_ns if clock_op == "elapsed_span" else None,
        "plan_key": dict(plan_key), "actual_shape": dict(req_shape),
    }
    cap_bytes = canonical_json_bytes(cap_payload)
    raw_sha = hashlib.sha256(cap_bytes).hexdigest()
    cap_uri = f"capture://{clean_ep}/{pub_run_id}/{raw_sha}.json"
    if capture_out_dir is not None:
        write_atomic_bytes(Path(capture_out_dir) / f"{raw_sha}.json", cap_bytes)

    ident = first.get("identity") if isinstance(first.get("identity"), Mapping) else {}
    raw_level = str(first.get("level") or ident.get("level") or "").strip().upper()
    if raw_level in {"L1", "L2", "L3"}:
        level = raw_level
    elif binding_info is not None:
        level = binding_info["level"]
    else:
        level = "L1" if len(timed_nodes) <= 1 else "L2"

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
        "plan_key": dict(plan_key), "actual_shape": dict(req_shape), **dims,
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
        comp = completes.get((cid, base_rid)) or {}
        summary = comp.get("summary") if isinstance(comp.get("summary"), Mapping) else {}
        plan_pt = plan_pts_by_id.get(cid)
        candidates = [
            (arm, tmode, derive_writer_clock_operator(group, summary, cid=cid, tmode=tmode), group)
            for (arm, tmode), group in sorted(mode_map.items())
        ]
        selected = select_mode_candidates(candidates, cid=cid, plan_pt=plan_pt)
        for _arm, _tmode, clock_op, group in selected:
            first = group[0]
            ranks = _extract_ranks_from_bucket(group, summary)
            acq_id = str(first.get("acquisition_id") or comp.get("acquisition_id") or "").strip()
            acq_key = acq_id or str(first.get("plan_id") or comp.get("plan_id") or "").strip()
            matched_rcpts = _lookup_entry_receipts(
                receipts_by_key, cid=cid, base_rid=base_rid, ranks=ranks
            )
            pre_entries.append({
                "cid": cid, "base_rid": base_rid, "pub_run_id": sanitize_public_run_id(base_rid),
                "split": split, "clock_op": clock_op, "group": list(group),
                "receipts": matched_rcpts, "comp": comp, "summary": summary, "ranks": ranks,
                "acq_id": acq_id, "acq_key": acq_key, "multi_rank_joined": False,
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


def _verify_returned_plan_binding(
    plan_doc: Mapping[str, Any],
    receipt_metas: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    split_tag: str = "",
) -> None:
    expected_sha = str(plan_doc.get("plan_sha256") or "").strip()
    allowed_ids = {expected_sha, f"{expected_sha}:{split_tag}"} if split_tag else {expected_sha}
    for meta in receipt_metas:
        got_sha = str(meta.get("plan_sha256") or meta.get("plan_id") or "").strip()
        if got_sha and expected_sha and got_sha not in allowed_ids:
            raise ModelingError(f"profiler_bench_plan_sha_mismatch:expected={expected_sha}:got={got_sha}")
    planned_ids = {
        str(pt["point_id"])
        for split_key in ("train_points", "heldout_points", "abi_rejection_points")
        for pt in (plan_doc.get(split_key) or ())
        if isinstance(pt, Mapping) and pt.get("point_id")
    }
    if planned_ids and records:
        returned_ids = {
            str(r.get("case_id") or r.get("point_id") or "").strip()
            for r in records
            if str(r.get("case_id") or r.get("point_id") or "").strip()
        }
        if returned_ids and not (returned_ids & planned_ids):
            raise ModelingError(
                f"profiler_bench_unplanned_case_ids:matched=0:returned={len(returned_ids)}:planned={len(planned_ids)}"
            )


def _validate_dispatch_exit(
    returncode: int,
    receipt_metas: Sequence[Mapping[str, Any]],
    receipt_records: Sequence[Mapping[str, Any]],
    planned_abi_by_id: Mapping[str, Mapping[str, Any]],
    err_detail: str,
    *,
    is_remote: bool,
) -> None:
    from .cliff_planner import _check_abi_row_shape, _match_structured_abi_error

    prefix = "profiler_bench_remote_failed" if is_remote else "profiler_bench_failed"
    for rec in receipt_records:
        cid = str(rec.get("case_id") or rec.get("point_id") or "").strip()
        st = str(rec.get("status") or "").strip().lower()
        if cid in planned_abi_by_id and st in _ACCEPTED_ABI_STATUSES:
            raise ModelingError(f"profiler_abi_rejection_unexpectedly_accepted:{cid}:status={st}")
    if any(str(m.get("status") or "ok").strip().lower() not in {"ok", "partial"} for m in receipt_metas):
        raise ModelingError(f"{prefix}:rc={returncode}:receipt_status_invalid:{err_detail}")
    non_passing = [
        r for r in receipt_records
        if str(r.get("status") or "ok").strip().lower() not in _PASSING_RECEIPT_STATUSES
    ]
    violations: list[str] = []
    for rec in non_passing:
        cid = str(rec.get("case_id") or rec.get("point_id") or "unknown").strip()
        st = str(rec.get("status") or "").strip().lower()
        if st not in _ABI_REJECT_STATUSES or cid not in planned_abi_by_id:
            violations.append(f"{cid}={st}")
            continue
        planned_pt = planned_abi_by_id[cid]
        err_text = str(rec.get("abi_error") or rec.get("reason") or "").strip()
        expected_err = str(planned_pt.get("abi_expected_error") or "").strip()
        err_mismatch = _match_structured_abi_error(err_text, expected_err, cid)
        if err_mismatch:
            violations.append(err_mismatch)
            continue
        attempted = extract_abi_attempted_shape(rec, planned_pt)
        shape_err = _check_abi_row_shape({"executed_shape": attempted}, planned_pt)
        if shape_err:
            violations.append(f"{cid}:{shape_err}")
    if violations:
        raise ModelingError(
            f"profiler_bench_unexpected_partial_failure:{','.join(violations[:_MAX_UNEXPECTED_REPORTED])}"
        )
    if returncode == 0:
        if any(str(m.get("status") or "").strip().lower() == "partial" for m in receipt_metas) and not non_passing:
            raise ModelingError(f"{prefix}:rc=0:partial_without_planned_rejection:{err_detail}")
        return
    if returncode != _PARTIAL_EXIT_CODE or not receipt_metas:
        raise ModelingError(f"{prefix}:rc={returncode}:{err_detail}")
    if not non_passing:
        raise ModelingError(f"{prefix}:rc={returncode}:partial_without_planned_rejection:{err_detail}")


def _new_acq_id(tag: str) -> str:
    return f"acq-{tag or 'all'}-{uuid.uuid4().hex[:_ACQ_ID_HEX_LEN]}"


def _partition_by_world_size(
    plan_doc: Mapping[str, Any],
    *,
    base_tag: str,
    train_pts: Sequence[Mapping[str, Any]],
    held_pts: Sequence[Mapping[str, Any]],
    abi_pts: Sequence[Mapping[str, Any]],
) -> list[tuple[str, dict[str, Any], str]]:
    from .cliff_planner import validate_parallel_topology

    shared_acq = _new_acq_id(base_tag)
    active_pts = list(held_pts) if held_pts else list(train_pts)
    is_train = bool(train_pts) and not held_pts
    by_world: dict[int, list[Mapping[str, Any]]] = {}
    for pt in active_pts:
        ranks = validate_parallel_topology(str(pt.get("parallel_topology") or "tp1")) if isinstance(pt, Mapping) else 1
        by_world.setdefault(ranks, []).append(pt)
    if len(by_world) <= 1:
        return [(
            base_tag,
            {
                **dict(plan_doc),
                "train_points": list(train_pts),
                "heldout_points": list(held_pts),
                "abi_rejection_points": list(abi_pts),
            },
            shared_acq,
        )]
    out: list[tuple[str, dict[str, Any], str]] = []
    abi_attached = False
    for world_sz in sorted(by_world):
        pts_abi = list(abi_pts) if not abi_attached else []
        abi_attached = True
        sub_tag = f"{base_tag}-w{world_sz}"
        sub_doc = {
            **dict(plan_doc),
            "train_points": by_world[world_sz] if is_train else [],
            "heldout_points": [] if is_train else by_world[world_sz],
            "abi_rejection_points": pts_abi,
        }
        out.append((sub_tag, sub_doc, shared_acq))
    return out


def _split_disjoint_acquisitions(
    plan_doc: Mapping[str, Any], *, domain: int | None, dry_run: bool
) -> list[tuple[str, dict[str, Any], str]]:
    if domain is not None or dry_run:
        return [("", dict(plan_doc), _new_acq_id("all"))]
    train_pts = list(plan_doc.get("train_points") or ())
    held_pts = list(plan_doc.get("heldout_points") or ())
    abi_pts = list(plan_doc.get("abi_rejection_points") or ())
    if str(plan_doc.get("family") or "").strip() == "sku_suite" and held_pts:
        by_level: dict[str, list[Mapping[str, Any]]] = {"L1": [], "L2": [], "L3": []}
        for pt in held_pts:
            lvl = assign_plan_point_level(plan_doc, pt) if isinstance(pt, Mapping) else "L1"
            by_level.setdefault(lvl, []).append(pt)
        splits: list[tuple[str, dict[str, Any], str]] = []
        if train_pts:
            splits.extend(_partition_by_world_size(
                plan_doc, base_tag="train", train_pts=train_pts, held_pts=[], abi_pts=[],
            ))
        abi_attached = False
        for lvl, tag in (("L1", "heldout-l1"), ("L2", "heldout-l2"), ("L3", "heldout-l3")):
            if by_level.get(lvl):
                pts_abi = abi_pts if not abi_attached else []
                abi_attached = True
                splits.extend(_partition_by_world_size(
                    plan_doc, base_tag=tag, train_pts=[], held_pts=by_level[lvl], abi_pts=pts_abi,
                ))
        if abi_pts and not abi_attached:
            splits.append((
                "heldout-abi",
                {**dict(plan_doc), "train_points": [], "heldout_points": [], "abi_rejection_points": abi_pts},
                _new_acq_id("heldout-abi"),
            ))
        return splits or [("", dict(plan_doc), _new_acq_id("all"))]
    if train_pts and (held_pts or abi_pts):
        splits = _partition_by_world_size(
            plan_doc, base_tag="train", train_pts=train_pts, held_pts=[], abi_pts=[],
        )
        if held_pts:
            splits.extend(_partition_by_world_size(
                plan_doc, base_tag="heldout", train_pts=[], held_pts=held_pts, abi_pts=abi_pts,
            ))
        else:
            splits.append((
                "heldout",
                {**dict(plan_doc), "train_points": [], "heldout_points": [], "abi_rejection_points": abi_pts},
                _new_acq_id("heldout"),
            ))
        return splits
    if held_pts:
        return _partition_by_world_size(
            plan_doc, base_tag="heldout", train_pts=[], held_pts=held_pts, abi_pts=abi_pts,
        )
    return [("", dict(plan_doc), _new_acq_id("all"))]


def _aggregate_rank_returncodes(rcs: Sequence[int]) -> int:
    if all(rc == 0 for rc in rcs):
        return 0
    if all(rc in {0, _PARTIAL_EXIT_CODE} for rc in rcs) and any(rc == _PARTIAL_EXIT_CODE for rc in rcs):
        return _PARTIAL_EXIT_CODE
    for rc in rcs:
        if rc not in {0, _PARTIAL_EXIT_CODE}:
            return int(rc)
    return int(rcs[0]) if rcs else 1


def _run_local_workers(
    base_cmd: Sequence[str],
    *,
    prof_root: Path,
    base_env: Mapping[str, str],
    out_file: Path,
    obs_file: Path,
    world: int,
    dry_run: bool,
    domain: int | None,
    timeout_s: int,
) -> tuple[int, str]:
    if dry_run or domain is not None or world <= 1:
        cmd = [*base_cmd, "--out", str(out_file), "--observations", str(obs_file)]
        env = dict(base_env)
        if domain is None and world > 1:
            env.setdefault("RANK", "0")
            env.setdefault("LOCAL_RANK", "0")
            env["WORLD_SIZE"] = str(world)
        try:
            done = subprocess.run(
                cmd, cwd=str(prof_root), env=env, capture_output=True, text=True,
                timeout=timeout_s, check=False, **subprocess_window_kwargs(),
            )
            return int(done.returncode), (done.stderr or done.stdout).strip()
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"profiler_bench_timeout:timeout_s={timeout_s}") from exc
        except TimeoutError:
            raise
        except OSError as exc:
            raise ModelingError(f"profiler_bench_os_error:{exc}") from exc

    master_port = _pick_free_local_port()
    procs: list[subprocess.Popen[str]] = []
    deadline = time.monotonic() + float(timeout_s)
    try:
        for rank in range(world):
            rank_obs = obs_file.with_name(f"observations.rank{rank}.jsonl")
            cmd = [*base_cmd, "--out", str(out_file), "--observations", str(rank_obs)]
            env = dict(base_env)
            env.update({
                "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world),
                "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(master_port),
            })
            procs.append(
                subprocess.Popen(
                    cmd, cwd=str(prof_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, **subprocess_window_kwargs(),
                )
            )
        rcs: list[int] = []
        errs: list[str] = []
        for proc in procs:
            rem = max(_MIN_POLL_REMAINING_S, deadline - time.monotonic())
            out_txt, err_txt = proc.communicate(timeout=rem)
            rcs.append(int(proc.returncode or 0))
            detail = (err_txt or out_txt or "").strip()
            if detail and proc.returncode != 0:
                errs.append(detail)
        return _aggregate_rank_returncodes(rcs), "; ".join(errs)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"profiler_bench_timeout:timeout_s={timeout_s}") from exc
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"profiler_bench_os_error:{exc}") from exc
    finally:
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=_PROC_REAP_GRACE_S)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                        proc.wait(timeout=_PROC_REAP_GRACE_S)
                    except (OSError, subprocess.TimeoutExpired):
                        pass


def _build_remote_subgraph_cmd(
    *,
    plan_b64: str,
    plan_sha: str,
    acq_id: str,
    env_prefix: str,
    fence_env: str,
    phy_csv: str,
    world: int,
    dry_run: bool,
) -> str:
    phy_flag = f" --allowed-phy-ids {shlex.quote(phy_csv)}" if phy_csv else ""
    dry_flag = " --dry-run" if dry_run else ""
    common_env = (
        f"{env_prefix}{fence_env}"
        f"MINDIE_ONBOARDING_PLAN_JSON=\"$TMPD/plan.json\" "
        f"MINDIE_ONBOARDING_PLAN_SHA256={shlex.quote(plan_sha)} "
        f"MINDIE_ONBOARDING_ACQUISITION_ID={shlex.quote(acq_id)} "
    )
    if dry_run or world <= 1:
        rank_env = f"RANK=0 LOCAL_RANK=0 WORLD_SIZE={world} " if world > 1 else ""
        launch = (
            f"{common_env}{rank_env}"
            f"python3 -B -m mindie_profiler bench subgraph --plan \"$TMPD/plan.json\" "
            f"--out \"$TMPD/receipt.json\" --observations \"$TMPD/observations.jsonl\"{phy_flag}{dry_flag}; RC=$?; "
        )
    else:
        launch = (
            "MPORT=$(python3 -c 'import socket;s=socket.socket();s.bind((\"127.0.0.1\",0));print(s.getsockname()[1]);s.close()'); "
            f"PIDS=''; for R in $(seq 0 {world - 1}); do "
            f"{common_env}RANK=$R LOCAL_RANK=$R WORLD_SIZE={world} MASTER_ADDR=127.0.0.1 MASTER_PORT=$MPORT "
            f"python3 -B -m mindie_profiler bench subgraph --plan \"$TMPD/plan.json\" "
            f"--out \"$TMPD/receipt.json\" --observations \"$TMPD/observations.rank${{R}}.jsonl\"{phy_flag} & "
            "PIDS=\"$PIDS $!\"; done; "
            "RC=0; for P in $PIDS; do wait \"$P\"; W=$?; "
            "if [ \"$W\" -ne 0 ]; then if [ \"$RC\" -eq 0 ] || [ \"$RC\" -eq 2 -a \"$W\" -ne 2 ]; then RC=$W; fi; fi; done; "
        )
    return (
        f"TMPD=$(mktemp -d); trap 'rm -rf \"$TMPD\"' EXIT; "
        f"printf '%s' {shlex.quote(plan_b64)} | base64 -d > \"$TMPD/plan.json\"; "
        + launch
        + "for OF in \"$TMPD/observations.jsonl\" \"$TMPD\"/observations.rank*.jsonl; do "
        + "[ -s \"$OF\" ] && cat \"$OF\"; done; "
        + "for RF in \"$TMPD/receipt.json\" \"$TMPD\"/receipt.json.rank*.json; do "
        + f"[ -s \"$RF\" ] && printf '\\n{_REMOTE_RECEIPT_DELIM}\\n' && cat \"$RF\"; done; exit $RC"
    )


def _stamp_acquisition_id(records: list[dict[str, Any]], acq_id: str) -> None:
    for rec in records:
        rec.setdefault("acquisition_id", acq_id)


def _dispatch_remote_acquisition(
    sub_plan: Mapping[str, Any],
    *,
    plan_bytes: bytes,
    sku: str,
    split_tag: str,
    acq_id: str,
    endpoint_alias: str,
    phy_csv: str,
    fence_token: int | None,
    world: int,
    domain: int | None,
    dry_run: bool,
    timeout_s: int,
    planned_abi_by_id: Mapping[str, Mapping[str, Any]],
    capture_out_dir: str | Path | None,
    ssh_runner: Callable[[str, str, int], subprocess.CompletedProcess[str]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_ssh = ssh_runner if ssh_runner is not None else run_ssh_command
    env_prefix = f"ASCEND_RT_VISIBLE_DEVICES={shlex.quote(phy_csv)} " if phy_csv else ""
    fence_env = f"MINDIE_LEASE_FENCE_TOKEN={int(fence_token)} " if fence_token is not None else ""
    plan_b64 = base64.b64encode(plan_bytes).decode("ascii")
    if domain is not None:
        remote_cmd = (
            f"{env_prefix}{fence_env}"
            f"python3 -B -m mindie_profiler bench sku --sku {shlex.quote(sku)} --domain {int(domain)}"
            + (f" --allowed-phy-ids {shlex.quote(phy_csv)}" if phy_csv else "")
            + (" --dry-run" if dry_run else "")
        )
    else:
        remote_cmd = _build_remote_subgraph_cmd(
            plan_b64=plan_b64, plan_sha=str(sub_plan["plan_sha256"]),
            acq_id=acq_id, env_prefix=env_prefix, fence_env=fence_env,
            phy_csv=phy_csv, world=world, dry_run=dry_run,
        )
    done = active_ssh(endpoint_alias.strip(), remote_cmd, int(timeout_s))
    stdout_txt = done.stdout or ""
    obs_recs: list[dict[str, Any]] = []
    rcpt_recs: list[dict[str, Any]] = []
    receipt_metas: list[dict[str, Any]] = []
    if _REMOTE_RECEIPT_DELIM in stdout_txt:
        parts = stdout_txt.split(_REMOTE_RECEIPT_DELIM)
        obs_recs, _ = _parse_json_or_jsonl_records(parts[0])
        for seg in parts[1:]:
            recs, meta = _parse_json_or_jsonl_records(seg)
            rcpt_recs.extend(recs)
            if meta:
                receipt_metas.append(meta)
    else:
        recs, meta = _parse_json_or_jsonl_records(stdout_txt)
        rcpt_recs.extend(recs)
        if meta:
            receipt_metas.append(meta)
    _validate_dispatch_exit(
        done.returncode, receipt_metas, rcpt_recs, planned_abi_by_id,
        (done.stderr or done.stdout).strip(), is_remote=True,
    )
    raw_records = [*obs_recs, *rcpt_recs]
    _stamp_acquisition_id(raw_records, acq_id)
    if domain is None:
        _verify_returned_plan_binding(sub_plan, receipt_metas, raw_records, split_tag=split_tag)
    return normalize_profiler_records(
        raw_records, plan_doc=sub_plan, sku=sku,
        endpoint_alias=endpoint_alias, capture_out_dir=capture_out_dir,
    )


def _dispatch_local_acquisition(
    sub_plan: Mapping[str, Any],
    *,
    tmp: Path,
    plan_file: Path,
    out_file: Path,
    obs_file: Path,
    sku: str,
    split_tag: str,
    acq_id: str,
    profiler_root: str | Path,
    endpoint_alias: str,
    phy_csv: str,
    fence_token: int | None,
    world: int,
    domain: int | None,
    dry_run: bool,
    timeout_s: int,
    planned_abi_by_id: Mapping[str, Mapping[str, Any]],
    capture_out_dir: str | Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prof_root = Path(profiler_root).resolve()
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{prof_root}{os.pathsep}{existing_pp}" if existing_pp else str(prof_root)
    env["MINDIE_ONBOARDING_PLAN_JSON"] = str(plan_file)
    env["MINDIE_ONBOARDING_PLAN_SHA256"] = str(sub_plan["plan_sha256"])
    env["MINDIE_ONBOARDING_ACQUISITION_ID"] = acq_id
    if endpoint_alias.strip():
        env["MINDIE_ENDPOINT_ALIAS"] = endpoint_alias.strip()
    if fence_token is not None:
        env["MINDIE_LEASE_FENCE_TOKEN"] = str(int(fence_token))
    if phy_csv:
        env["ASCEND_RT_VISIBLE_DEVICES"] = phy_csv

    has_pkg_main = (prof_root / "mindie_profiler" / "__main__.py").is_file()
    if domain is not None:
        mod_file = prof_root / "mindie_profiler" / "bench" / "bench_sku_microbench.py"
        if not mod_file.is_file():
            raise ModelingError(f"profiler_bench_script_not_found:{mod_file}")
        cmd = [sys.executable, "-B", "-m", "mindie_profiler", "bench", "sku"] if has_pkg_main else [sys.executable, "-B", str(mod_file)]
        cmd.extend(["--sku", sku, "--domain", str(int(domain))])
    else:
        mod_file = prof_root / "mindie_profiler" / "bench" / "bench_archetype_subgraphs.py"
        if not mod_file.is_file() and not has_pkg_main:
            raise ModelingError(f"profiler_bench_subgraph_script_not_found:{mod_file}")
        cmd = [sys.executable, "-B", "-m", "mindie_profiler", "bench", "subgraph"] if has_pkg_main else [sys.executable, "-B", str(mod_file)]
        cmd.extend(["--plan", str(plan_file)])
    if phy_csv:
        cmd.extend(["--allowed-phy-ids", phy_csv])
    if dry_run:
        cmd.append("--dry-run")

    rc, err_txt = _run_local_workers(
        cmd, prof_root=prof_root, base_env=env, out_file=out_file, obs_file=obs_file,
        world=world, dry_run=dry_run, domain=domain, timeout_s=timeout_s,
    )
    obs_files = ([obs_file] if obs_file.is_file() else []) + sorted(tmp.glob("observations.rank*.jsonl"))
    receipt_files = ([out_file] if out_file.is_file() else []) + sorted(out_file.parent.glob(f"{out_file.name}.rank*.json"))
    obs_recs, rcpt_recs, receipt_metas = [], [], []
    for ofile in obs_files:
        recs, _ = _parse_json_or_jsonl_records(
            ofile.read_text(encoding="utf-8"), rank_hint=_parse_rank_hint(ofile.name)
        )
        obs_recs.extend(recs)
    for rfile in receipt_files:
        recs, meta = _parse_json_or_jsonl_records(
            rfile.read_text(encoding="utf-8"), rank_hint=_parse_rank_hint(rfile.name)
        )
        rcpt_recs.extend(recs)
        if meta:
            receipt_metas.append(meta)
    _validate_dispatch_exit(
        rc, receipt_metas, rcpt_recs, planned_abi_by_id, err_txt, is_remote=False,
    )
    if not obs_files and not receipt_files:
        raise ModelingError("profiler_bench_missing_output_json")
    combined_records = [*obs_recs, *rcpt_recs]
    _stamp_acquisition_id(combined_records, acq_id)
    if domain is None:
        _verify_returned_plan_binding(sub_plan, receipt_metas, combined_records, split_tag=split_tag)
    return normalize_profiler_records(
        combined_records, plan_doc=sub_plan, sku=sku,
        endpoint_alias=endpoint_alias, capture_out_dir=capture_out_dir,
    )


def _dispatch_single_acquisition(
    sub_plan: Mapping[str, Any],
    *,
    sku: str,
    split_tag: str,
    acq_id: str = "",
    profiler_root: str | Path | None,
    endpoint_alias: str,
    allowed_phy_ids: Sequence[int],
    fence_token: int | None,
    domain: int | None,
    dry_run: bool,
    timeout_s: int,
    capture_out_dir: str | Path | None,
    ssh_runner: Callable[[str, str, int], subprocess.CompletedProcess[str]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    use_ssh = ssh_runner is not None or _is_remote_endpoint(endpoint_alias)
    if not use_ssh and profiler_root is None:
        raise ModelingError("profiler_bench_requires_profiler_root")
    eff_acq_id = acq_id or _new_acq_id(split_tag)
    phy_csv = ",".join(str(int(d)) for d in allowed_phy_ids)
    subgraph_plan = (
        build_profiler_subgraph_plan(sub_plan, sku=sku, split_tag=split_tag, acquisition_id=eff_acq_id)
        if domain is None
        else dict(sub_plan)
    )
    world = max(1, int(subgraph_plan.get("world_size") or 1)) if domain is None else 1
    if not dry_run and domain is None and world > 1 and allowed_phy_ids and len(allowed_phy_ids) < world:
        raise ModelingError(
            f"profiler_bench_insufficient_granted_devices:required={world}:granted={len(allowed_phy_ids)}"
        )
    plan_bytes = canonical_json_bytes(subgraph_plan)
    planned_abi_by_id: dict[str, Mapping[str, Any]] = {
        str(pt["point_id"]): pt
        for pt in (sub_plan.get("abi_rejection_points") or ())
        if isinstance(pt, Mapping) and pt.get("point_id")
    }

    with tempfile.TemporaryDirectory(prefix="msim-prof-bench-") as tmp_dir:
        tmp = Path(tmp_dir)
        plan_file, out_file, obs_file = tmp / "onboarding_plan.json", tmp / "bench_receipt.json", tmp / "observations.jsonl"
        write_atomic_bytes(plan_file, plan_bytes)
        if use_ssh:
            return _dispatch_remote_acquisition(
                sub_plan, plan_bytes=plan_bytes, sku=sku, split_tag=split_tag, acq_id=eff_acq_id,
                endpoint_alias=endpoint_alias, phy_csv=phy_csv, fence_token=fence_token,
                world=world, domain=domain, dry_run=dry_run, timeout_s=timeout_s,
                planned_abi_by_id=planned_abi_by_id, capture_out_dir=capture_out_dir, ssh_runner=ssh_runner,
            )
        assert profiler_root is not None
        return _dispatch_local_acquisition(
            sub_plan, tmp=tmp, plan_file=plan_file, out_file=out_file, obs_file=obs_file,
            sku=sku, split_tag=split_tag, acq_id=eff_acq_id, profiler_root=profiler_root,
            endpoint_alias=endpoint_alias, phy_csv=phy_csv, fence_token=fence_token,
            world=world, domain=domain, dry_run=dry_run, timeout_s=timeout_s,
            planned_abi_by_id=planned_abi_by_id, capture_out_dir=capture_out_dir,
        )


def run_profiler_bench_subprocess(
    plan_doc: Mapping[str, Any],
    *,
    sku: str,
    profiler_root: str | Path | None = None,
    endpoint_alias: str = "",
    allowed_phy_ids: Sequence[int] = (),
    fence_token: int | None = None,
    domain: int | None = None,
    dry_run: bool = False,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
    capture_out_dir: str | Path | None = None,
    ssh_runner: Callable[[str, str, int], subprocess.CompletedProcess[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Dispatch versioned plan on leased devices (disjoint train/L1/L2/L3) and normalize."""
    if not isinstance(plan_doc, Mapping) or not plan_doc.get("plan_sha256"):
        raise ModelingError("profiler_bench_plan_doc_invalid")
    acquisitions = _split_disjoint_acquisitions(plan_doc, domain=domain, dry_run=dry_run)
    remaining = max(1, int(timeout_s))
    all_obs: list[dict[str, Any]] = []
    all_abi: list[dict[str, Any]] = []
    for idx, (split_tag, sub_plan, shared_acq_id) in enumerate(acquisitions):
        left = len(acquisitions) - idx
        step_budget = remaining if left <= 1 else max(1, remaining // left)
        remaining = max(1, remaining - step_budget)
        obs_part, abi_part = _dispatch_single_acquisition(
            sub_plan, sku=sku, split_tag=split_tag, acq_id=shared_acq_id,
            profiler_root=profiler_root, endpoint_alias=endpoint_alias,
            allowed_phy_ids=allowed_phy_ids, fence_token=fence_token,
            domain=domain, dry_run=dry_run, timeout_s=step_budget,
            capture_out_dir=capture_out_dir, ssh_runner=ssh_runner,
        )
        all_obs.extend(obs_part)
        all_abi.extend(abi_part)
    return all_obs, all_abi


__all__ = [
    "build_profiler_subgraph_plan",
    "normalize_profiler_records",
    "run_profiler_bench_subprocess",
]
