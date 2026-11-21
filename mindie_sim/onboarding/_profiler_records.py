"""Raw ``mindie-profiler`` JSON/JSONL record parsing, critical-path join, and bucket indexing.

[Role] Parse single- and multi-rank ``mindie-profiler`` JSON/JSONL observation
  and receipt files, sanitize capture-bound ``actual_shape`` payloads against
  an explicit allowlist, compute per-repeat critical-path durations across
  ranks with zero silent drops, filter and index rank receipts by split,
  acquisition, and plan context, collect planned ABI rejections, and join
  disjoint multi-rank observation entries.
[Invariants]
  * Rejects non-mapping JSON/JSONL records, non-finite/non-positive sample
    durations, missing/failed receipt statuses, diagnostic records masquerading
    as receipts, and unqualified or non-``ok`` observations/``case_complete``.
  * Preserves every rank's ``case_complete`` summaries and receipts during
    multi-rank joining so cross-rank level/timed-node checks inspect all ranks.
  * Never mutates global state or writes to disk.
[Timeouts & Locks] Pure in-memory parsing and aggregation; zero I/O, locks, or
  subprocesses.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from typing import Any

from mindie_sim.schema.common import ModelingError

from ._profiler_geometry import extract_abi_attempted_shape, parse_run_id_and_arm
from .contracts import (
    NS_PER_MS,
    SHORT_DIGEST_LEN,
    canonical_json_bytes,
    sanitize_public_run_id,
)

_ABI_REJECT_STATUSES = frozenset({"abi_rejected", "unsupported_binding"})
_PASSING_RECEIPT_STATUSES = frozenset({"ok", "skipped_stage", "skipped_non_participant"})
_ALLOWED_OBS_STATUSES = frozenset({"", "ok"})
_ACCEPTED_ABI_STATUSES = frozenset({"ok", "unqualified"})
_OBSERVATION_SCHEMA = "mindie-bench-observation/1"
_RECEIPT_SCHEMA = "mindie-bench-receipt/1"
_SUBGRAPH_RUN_SCHEMA = "mindie-bench-subgraph-run/1"
_RANK_FILENAME_RE = re.compile(r"\.rank(\d+)\.(?:json|jsonl)$")
_ALLOWED_CAPTURE_SHAPE_KEYS = frozenset({
    "tokens", "sequences", "batch_size", "query_tokens", "q_len",
    "seq_len", "kv_len", "context_len", "page_count", "block_table",
    "block_size", "paged", "heads", "kv_heads", "head_dim", "latent",
    "kv_lora_rank", "rope", "qk_rope_head_dim", "width", "byte_width",
    "m", "n", "k", "e", "matrix_groups", "topk", "top_k", "sparse_count",
    "enforce_exact_topk", "query_layout", "input_layout", "cache_layout",
    "output_layout", "query_dtype", "cache_dtype", "dtype", "partial_count",
    "nbytes", "link_bytes", "elements", "group_size", "parallel_topology",
    "batch_x_kv_heads",
})


def _sanitize_capture_shape(shape: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in shape.items():
        k_str = str(key)
        if k_str == "node_shapes" and isinstance(val, Mapping):
            clean_ns: dict[str, Any] = {}
            for nk, nv in val.items():
                if isinstance(nv, Mapping):
                    clean_ns[str(nk)] = _sanitize_capture_shape(nv)
            if clean_ns:
                out["node_shapes"] = clean_ns
        elif k_str in _ALLOWED_CAPTURE_SHAPE_KEYS:
            out[k_str] = val
    return out


def _parse_rank_hint(filename: str) -> int | None:
    match = _RANK_FILENAME_RE.search(str(filename or ""))
    return int(match.group(1)) if match else None


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
                for idx, r in enumerate(val):
                    if not isinstance(r, Mapping):
                        raise ModelingError(f"profiler_bench_record_not_mapping:key={key}:index={idx}")
                    collected.append(
                        _propagate_envelope_fields(dict(r), raw, key=key, rank_hint=rank_hint)
                    )
        if isinstance(raw.get("receipt"), Mapping):
            sub_env = raw["receipt"]
            for key in ("cases", "records", "receipts", "abi_rejections"):
                val = sub_env.get(key)
                if isinstance(val, list):
                    for idx, r in enumerate(val):
                        if not isinstance(r, Mapping):
                            raise ModelingError(f"profiler_bench_record_not_mapping:key=receipt.{key}:index={idx}")
                        collected.append(
                            _propagate_envelope_fields(dict(r), sub_env, key=key, rank_hint=rank_hint)
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
    out_list: list[dict[str, Any]] = []
    for idx, r in enumerate(raw):
        if not isinstance(r, Mapping):
            raise ModelingError(f"profiler_bench_record_not_mapping:index={idx}")
        out_list.append(dict(r))
    return out_list, {}


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


def _valid_duration_ms(raw_dur: Any) -> float | None:
    if raw_dur is None or isinstance(raw_dur, bool) or not isinstance(raw_dur, (int, float)):
        return None
    val = float(raw_dur)
    return val if math.isfinite(val) and val > 0.0 else None


def _critical_path_durations_ns(
    group: Sequence[Mapping[str, Any]], *, expected_ranks: Sequence[int], cid: str
) -> list[int]:
    if not group:
        raise ModelingError(f"profiler_sample_durations_empty:{cid}")
    if len(expected_ranks) <= 1:
        durations: list[int] = []
        for idx, row in enumerate(group):
            dur_ms = _valid_duration_ms(row.get("duration_ms"))
            if dur_ms is None:
                raise ModelingError(
                    f"profiler_sample_duration_invalid:{cid}:sample_id={row.get('sample_id') or idx}"
                    f":duration_ms={row.get('duration_ms')!r}"
                )
            durations.append(int(round(dur_ms * NS_PER_MS)))
        return sorted(durations)
    expected_set = {int(r) for r in expected_ranks}
    by_sample_slot: dict[tuple[int, int], dict[int, int]] = {}
    for row in group:
        dur_ms = _valid_duration_ms(row.get("duration_ms"))
        if dur_ms is None:
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
        slot_ranks[rk] = int(round(dur_ms * NS_PER_MS))
    if not by_sample_slot:
        raise ModelingError(f"multirank_incomplete_rank_samples:{cid}:empty")
    for slot_key, slot_ranks in sorted(by_sample_slot.items()):
        if set(slot_ranks) != expected_set:
            missing = sorted(expected_set - set(slot_ranks))
            raise ModelingError(
                f"multirank_incomplete_rank_samples:{cid}:slot={slot_key}:missing_ranks={missing}"
            )
    return sorted(max(slot_ranks.values()) for slot_ranks in by_sample_slot.values())


def _receipt_matches_context(
    rcpt: Mapping[str, Any], *, split: str, acq_id: str, plan_id: str
) -> bool:
    r_split = str(rcpt.get("split") or "").strip().lower()
    if r_split and r_split != split:
        return False
    r_acq = str(rcpt.get("acquisition_id") or "").strip()
    if r_acq and r_acq != acq_id:
        return False
    r_plan = str(rcpt.get("plan_id") or "").strip()
    if r_plan and r_plan != plan_id:
        return False
    return True


def _lookup_entry_receipts(
    receipts_by_key: Mapping[tuple[str, str, int | None], list[Mapping[str, Any]]],
    *,
    cid: str,
    base_rid: str,
    ranks: Sequence[int],
    split: str,
    acq_id: str,
    plan_id: str,
) -> tuple[list[Mapping[str, Any]], list[int]]:
    matched: list[Mapping[str, Any]] = []
    seen_ids: set[int] = set()
    covered_ranks: set[int] = set()
    for rk in ranks:
        rk_int = int(rk)
        rk_rcpts = [
            r for r in (receipts_by_key.get((cid, base_rid, rk_int)) or ())
            if _receipt_matches_context(r, split=split, acq_id=acq_id, plan_id=plan_id)
        ]
        if rk_rcpts:
            covered_ranks.add(rk_int)
            for rcpt in rk_rcpts:
                obj_id = id(rcpt)
                if obj_id not in seen_ids:
                    seen_ids.add(obj_id)
                    matched.append(rcpt)
    unsuffixed = [
        r for r in (receipts_by_key.get((cid, base_rid, None)) or ())
        if _receipt_matches_context(r, split=split, acq_id=acq_id, plan_id=plan_id)
    ]
    if unsuffixed:
        if len(ranks) == 1:
            covered_ranks.add(int(ranks[0]))
        for rcpt in unsuffixed:
            obj_id = id(rcpt)
            if obj_id not in seen_ids:
                seen_ids.add(obj_id)
                matched.append(rcpt)
    return matched, sorted(covered_ranks)


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
        merged_comps: list[Mapping[str, Any]] = []
        merged_summaries: list[Mapping[str, Any]] = []
        merged_receipt_ranks: set[int] = set()
        base_rids: list[str] = []
        for m in members:
            merged_group.extend(m["group"])
            merged_receipts.extend(m.get("receipts") or ())
            merged_comps.extend(m.get("comps") or ([m["comp"]] if m.get("comp") else []))
            merged_summaries.extend(m.get("summaries") or ([m["summary"]] if m.get("summary") else []))
            merged_receipt_ranks.update(int(r) for r in (m.get("receipt_ranks") or ()))
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
            "receipt_ranks": sorted(merged_receipt_ranks),
            "comp": members[0]["comp"],
            "comps": merged_comps,
            "summary": members[0]["summary"],
            "summaries": merged_summaries,
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
    dict[tuple[str, str], list[Mapping[str, Any]]],
    dict[tuple[str, str, str], dict[tuple[str, str], list[Mapping[str, Any]]]],
    dict[tuple[str, str, int | None], list[Mapping[str, Any]]],
    list[dict[str, Any]],
]:
    completes: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    obs_buckets: dict[tuple[str, str, str], dict[tuple[str, str], list[Mapping[str, Any]]]] = {}
    receipts_by_key: dict[tuple[str, str, int | None], list[Mapping[str, Any]]] = {}
    abi_rows: list[dict[str, Any]] = []
    seen_abi_tuples: set[tuple[str, str, bytes]] = set()

    for rec in raw_records:
        rtype = str(rec.get("record_type") or "")
        raw_status = rec.get("status")
        status = str(raw_status).strip().lower() if raw_status is not None else ""
        cid = str(rec.get("case_id") or rec.get("point_id") or "").strip()
        raw_run_id = str(rec.get("run_id") or "run-0").strip()
        schema = str(rec.get("schema") or "").strip()

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
            if status != "ok":
                raise ModelingError(
                    f"profiler_case_complete_failed:{cid}:{status or 'missing_status'}"
                )
            base_rid, _arm = parse_run_id_and_arm(raw_run_id)
            completes.setdefault((cid, base_rid), []).append(rec)
        elif rtype == "observation":
            if status not in _ALLOWED_OBS_STATUSES:
                raise ModelingError(f"profiler_observation_status_invalid:{cid}:status={status}")
            if rec.get("validated") is False or rec.get("qualified") is False:
                raise ModelingError(f"profiler_observation_unqualified:{cid}")
            split = "heldout" if str(rec.get("split") or "") == "heldout" else "train"
            tmode = str(rec.get("timing_mode") or "default").strip()
            exec_map = rec.get("execution") if isinstance(rec.get("execution"), Mapping) else {}
            base_rid, arm = parse_run_id_and_arm(
                raw_run_id, tmode, rec.get("trial"), str(rec.get("source") or ""), exec_map
            )
            obs_buckets.setdefault((cid, base_rid, split), {}).setdefault((arm, tmode), []).append(rec)
        elif cid and (rtype == "receipt" or (not rtype and schema in {_RECEIPT_SCHEMA, _SUBGRAPH_RUN_SCHEMA})):
            if status not in _PASSING_RECEIPT_STATUSES:
                raise ModelingError(f"profiler_bench_receipt_failed:{cid}:{status or 'missing_status'}")
            if status == "ok":
                base_rid, _arm = parse_run_id_and_arm(raw_run_id)
                raw_rk = rec.get("rank")
                rk_val = int(raw_rk) if isinstance(raw_rk, int) and not isinstance(raw_rk, bool) else None
                receipts_by_key.setdefault((cid, base_rid, rk_val), []).append(rec)
    return completes, obs_buckets, receipts_by_key, abi_rows


__all__ = [
    "_ABI_REJECT_STATUSES",
    "_ACCEPTED_ABI_STATUSES",
    "_OBSERVATION_SCHEMA",
    "_PASSING_RECEIPT_STATUSES",
    "_RECEIPT_SCHEMA",
    "_SUBGRAPH_RUN_SCHEMA",
    "_collect_abi_and_buckets",
    "_critical_path_durations_ns",
    "_extract_ranks_from_bucket",
    "_join_multirank_obs_entries",
    "_lookup_entry_receipts",
    "_parse_json_or_jsonl_records",
    "_parse_rank_hint",
    "_sanitize_capture_shape",
]
