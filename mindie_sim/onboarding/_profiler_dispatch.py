"""Live ``mindie-profiler`` plan-consuming dispatch and real observation/ABI normalizer.

[Role] Translate onboarding cliff/SKU plans into ``mindie-bench-subgraph-plan/1``
  plans, dispatch them through the plan-consuming ``mindie_profiler bench
  subgraph --plan`` entry point (both local subprocess and remote SSH on leased
  physical devices with ``fence_token``), reserve ``bench sku --domain`` strictly
  for Domain-5 fabric probes, verify returned ``plan_sha256`` and ``case_id``s,
  read BOTH observation JSONL and receipt JSON streams, normalize multi-trial
  ``run_id.<mode>.t<trial>`` repeats joined to ``case_complete`` with real
  ``participating_ranks``, and persist byte-identical SHA-256 capture files.
[Invariants]
  * Never drops ``plan_doc`` on either local or SSH dispatch; rejects returned
    ``plan_sha256`` / ``plan_id`` mismatches and disjoint case IDs.
  * Preserves real ``participating_ranks`` (never rewrites ``[2]`` to ``[0]``).
  * Reads both ``observations`` and ``receipt`` streams so separate ABI rejection
    receipts are never dropped when observations exist.
  * Hashes and writes the exact canonical JSON byte payload via
    ``write_atomic_bytes`` (zero ``capture_digest_mismatch`` self-corruption).
  * Uses opaque public ``ep-<12-hex>`` / ``run-<12-hex>`` IDs in URIs/rows.
  * ``TimeoutError`` always propagates before ``OSError``; windowless spawn.
[Timeouts & Locks] Synchronous bounded subprocesses; no locks.
[Side-Effect Policy] Writes ephemeral plan/output temp files cleaned up
  unconditionally; writes raw capture JSON bytes only to explicit ``capture_out_dir``.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from typing import Any

from mindie_sim.schema.common import ModelingError

from .contracts import (
    DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
    NS_PER_MS,
    canonical_json_bytes,
    sanitize_public_alias,
    sanitize_public_run_id,
    subprocess_window_kwargs,
    write_atomic_bytes,
)

_P99_QUANTILE = 0.99
_ABI_REJECT_STATUSES = frozenset({"abi_rejected", "unsupported_binding"})
_BUSY_UNION_MODES = frozenset({"device_kernel", "event_busy_union", "busy_union", "eager_device", "aclgraph_device"})
_OBSERVATION_SCHEMA = "mindie-bench-observation/1"
_RECEIPT_SCHEMA = "mindie-bench-receipt/1"
_SUBGRAPH_PLAN_SCHEMA = "mindie-bench-subgraph-plan/1"
_TRIAL_SUFFIX_RE = re.compile(r"\.[^.]+\.t\d+$")
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_REMOTE_RECEIPT_DELIM = "__MINDIE_RECEIPT_JSON__"


def _quantile_ns(sorted_ns: Sequence[int], q: float) -> int:
    if not sorted_ns:
        return 0
    if len(sorted_ns) == 1:
        return int(sorted_ns[0])
    pos = q * (len(sorted_ns) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return int(sorted_ns[lo])
    return int(round(sorted_ns[lo] * (1.0 - (pos - lo)) + sorted_ns[hi] * (pos - lo)))


def _base_run_id(run_id: str, timing_mode: str = "", trial: Any = None) -> str:
    clean = str(run_id or "run-0").strip()
    if timing_mode and trial is not None and not isinstance(trial, bool):
        suffix = f".{timing_mode}.t{int(trial)}"
        if clean.endswith(suffix):
            return clean[: -len(suffix)]
    return _TRIAL_SUFFIX_RE.sub("", clean)


def _parse_json_or_jsonl_records(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
            rows.append(dict(rec))
        return rows, {}
    if isinstance(raw, Mapping):
        collected: list[dict[str, Any]] = []
        for key in ("observations", "records", "cases", "receipts", "abi_rejections"):
            val = raw.get(key)
            if isinstance(val, list):
                collected.extend(dict(r) for r in val if isinstance(r, Mapping))
        if isinstance(raw.get("receipt"), Mapping):
            for key in ("cases", "receipts", "abi_rejections"):
                val = raw["receipt"].get(key)
                if isinstance(val, list):
                    collected.extend(dict(r) for r in val if isinstance(r, Mapping))
        if not collected and (raw.get("record_type") or raw.get("case_id")):
            collected.append(dict(raw))
        return collected, dict(raw)
    if not isinstance(raw, list):
        raise ModelingError("profiler_bench_observations_not_list")
    return [dict(r) for r in raw if isinstance(r, Mapping)], {}


def _extract_shape_dims(shape: Mapping[str, Any], plan_key: Mapping[str, Any]) -> dict[str, Any]:
    dims: dict[str, Any] = {}
    merged = {**dict(plan_key), **dict(shape)}
    for dst, candidates in (
        ("context_len", ("context_len", "context", "kv_len", "seq_len", "scanned_rows")),
        ("tokens", ("tokens", "m", "num_tokens")),
        ("batch_size", ("batch_size", "batch", "b", "sequences")),
        ("query_tokens", ("query_tokens", "q", "q_len")),
        ("topk", ("topk", "topk_items", "top_k")),
        ("batch_x_kv_heads", ("batch_x_kv_heads",)),
    ):
        for key in candidates:
            val = merged.get(key)
            if val is not None and not isinstance(val, bool):
                try:
                    dims[dst] = int(val)
                    break
                except (TypeError, ValueError):
                    continue
    dims.setdefault("batch_size", 1)
    dims.setdefault("query_tokens", 1)
    dims["parallel_topology"] = str(merged.get("parallel_topology") or "tp1")
    return dims


def _plan_world_size(plan_doc: Mapping[str, Any]) -> int:
    max_world = 1
    for split_key in ("train_points", "heldout_points"):
        for pt in plan_doc.get(split_key) or ():
            if not isinstance(pt, Mapping):
                continue
            topo = str(pt.get("parallel_topology") or "tp1").lower()
            parsed = {k: int(v) for k, v in _TOPO_TOKEN_RE.findall(topo)}
            w = parsed.get("tp", 1) * max(parsed.get("pcp", 1), parsed.get("dcp", 1)) * parsed.get("pp", 1) * parsed.get("dp", 1)
            max_world = max(max_world, w)
    return max_world


def build_profiler_subgraph_plan(plan_doc: Mapping[str, Any], *, sku: str) -> dict[str, Any]:
    """Translate an onboarding cliff/SKU plan into ``mindie-bench-subgraph-plan/1``."""
    if str(plan_doc.get("schema") or "") == _SUBGRAPH_PLAN_SCHEMA:
        out = dict(plan_doc)
        out.setdefault("plan_id", str(plan_doc.get("plan_sha256") or plan_doc.get("plan_id") or "onboard"))
        out.setdefault("plan_sha256", str(plan_doc.get("plan_sha256") or ""))
        out.setdefault("sku", sku)
        return out
    world = _plan_world_size(plan_doc)
    family = str(plan_doc.get("family") or "rmsnorm").strip() or "rmsnorm"
    op_name = "rmsnorm" if family == "sku_suite" else family
    cases: list[dict[str, Any]] = []
    for split_key in ("train_points", "heldout_points", "abi_rejection_points"):
        for pt in plan_doc.get(split_key) or ():
            if not isinstance(pt, Mapping) or not pt.get("point_id"):
                continue
            pid = str(pt["point_id"])
            axis = str(pt.get("axis") or "context_len")
            val = int(pt["value"])
            shape: dict[str, Any] = {
                "batch_size": int(pt.get("batch_size", 1)),
                "query_tokens": int(pt.get("query_tokens", 1)),
                "parallel_topology": str(pt.get("parallel_topology") or "tp1"),
                axis: val,
            }
            if pt.get("topk") is not None:
                shape["topk"] = int(pt["topk"])
            cases.append({
                "case_id": pid,
                "split": str(pt.get("split") or "heldout"),
                "phase": "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode",
                "archetype": str(pt.get("regime_tag") or "onboarding"),
                "clock_operator": str(pt.get("clock_operator") or "elapsed_span"),
                "nodes": [{
                    "node_id": f"{pid}:op0",
                    "kind": "operator",
                    "op": op_name,
                    "family": family,
                    "dtype": "bf16",
                    "shape": shape,
                }],
                "edges": [],
            })
    return {
        "schema": _SUBGRAPH_PLAN_SCHEMA,
        "plan_id": str(plan_doc["plan_sha256"]),
        "plan_sha256": str(plan_doc["plan_sha256"]),
        "target_id": str(plan_doc.get("target_id") or ""),
        "sku": sku,
        "stage": "s1_joint" if world > 1 else "s0_local",
        "world_size": world,
        "execution_modes": ["eager"],
        "cases": cases,
    }


def normalize_profiler_records(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    plan_doc: Mapping[str, Any],
    sku: str,
    endpoint_alias: str = "",
    capture_out_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert raw ``mindie-profiler`` JSONL/receipt rows into onboarding observations & ABI rows."""
    if not raw_records:
        return [], []
    is_profiler_schema = any(
        str(r.get("schema") or "") in {_OBSERVATION_SCHEMA, _RECEIPT_SCHEMA}
        or r.get("record_type") in {"observation", "case_complete"}
        for r in raw_records
    )
    if not is_profiler_schema:
        obs_out = [dict(r) for r in raw_records if str(r.get("status") or "").lower() not in _ABI_REJECT_STATUSES]
        abi_out = [
            {
                "point_id": str(r.get("point_id") or r.get("case_id") or ""),
                "rejected": True,
                "abi_error": str(r.get("abi_error") or r.get("reason") or ""),
                "executed_shape": r.get("executed_shape"),
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

    completes: dict[tuple[str, str], Mapping[str, Any]] = {}
    obs_groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    abi_rows: list[dict[str, Any]] = []
    seen_abi_ids: set[str] = set()

    for rec in raw_records:
        rtype = str(rec.get("record_type") or "")
        status = str(rec.get("status") or "ok").strip().lower()
        cid = str(rec.get("case_id") or rec.get("point_id") or "").strip()
        raw_run_id = str(rec.get("run_id") or "run-0").strip()
        if status in _ABI_REJECT_STATUSES:
            if cid and cid not in seen_abi_ids:
                seen_abi_ids.add(cid)
                abi_rows.append({
                    "point_id": cid,
                    "rejected": True,
                    "abi_error": str(rec.get("abi_error") or rec.get("reason") or ""),
                    "executed_shape": rec.get("executed_shape") or rec.get("requested_shape") or rec.get("actual_shape"),
                })
            continue
        if rtype == "case_complete":
            completes[(cid, _base_run_id(raw_run_id))] = rec
        elif rtype == "observation":
            split = "heldout" if str(rec.get("split") or "") == "heldout" else "train"
            tmode = str(rec.get("timing_mode") or "default").strip()
            base_rid = _base_run_id(raw_run_id, tmode, rec.get("trial"))
            obs_groups.setdefault((cid, base_rid, split, tmode), []).append(rec)

    normalized: list[dict[str, Any]] = []
    for (cid, base_rid, split, tmode), group in sorted(obs_groups.items()):
        comp = completes.get((cid, base_rid)) or {}
        summary = comp.get("summary") if isinstance(comp.get("summary"), Mapping) else {}
        first = group[0]
        durations_ns = sorted(
            int(round(float(r["duration_ms"]) * NS_PER_MS))
            for r in group
            if r.get("duration_ms") is not None and float(r["duration_ms"]) > 0.0
        )
        if not durations_ns:
            continue
        p50_ns = int(round(statistics.median(durations_ns)))
        p99_ns = _quantile_ns(durations_ns, _P99_QUANTILE)
        plan_pt = plan_pts_by_id.get(cid)
        boundary_name = str(first.get("boundary") or "device_kernel").strip()
        if plan_pt and plan_pt.get("clock_operator"):
            clock_op = str(plan_pt["clock_operator"])
        else:
            clock_op = "busy_union" if (boundary_name in _BUSY_UNION_MODES or tmode in _BUSY_UNION_MODES) else "elapsed_span"
        plan_key = summary.get("plan_key") if isinstance(summary.get("plan_key"), Mapping) else (first.get("plan_key") if isinstance(first.get("plan_key"), Mapping) else {})
        req_shape = summary.get("actual_shape") or summary.get("requested_shape") or first.get("requested_shape") or {}
        if not isinstance(req_shape, Mapping):
            req_shape = {}
        dims = _extract_shape_dims(req_shape, plan_key)

        ranks_raw = summary.get("participating_ranks")
        if not (isinstance(ranks_raw, Sequence) and not isinstance(ranks_raw, (str, bytes)) and ranks_raw):
            obs_ranks = sorted({int(r["rank"]) for r in group if r.get("rank") is not None and not isinstance(r.get("rank"), bool)})
            ranks_raw = obs_ranks if obs_ranks else [0]
        ranks = [int(r) for r in ranks_raw if not isinstance(r, bool)] or [0]
        ev_ids = [str(r.get("sample_id") or f"{cid}-s{i}") for i, r in enumerate(group)]
        pub_run_id = sanitize_public_run_id(base_rid)

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
        acq_root = f"acq-{split}-{pub_run_id}-{cid}"
        row: dict[str, Any] = {
            "case_id": cid, "point_id": cid, "sku": str(first.get("sku") or sku),
            "split": split, "level": "L1" if str(ident.get("level") or "operator") == "operator" else "L2",
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
            "implementation": str(ident.get("implementation") or summary.get("op") or "profiler_bench"),
            "upstream_abi": str(plan_key.get("abi") or ident.get("implementation") or ""),
            "target_family": str(summary.get("op") or plan_family),
            "plan_key": dict(plan_key), "actual_shape": dict(req_shape),
            **dims,
        }
        normalized.append(row)
    return normalized, abi_rows


def _verify_returned_plan_binding(
    plan_doc: Mapping[str, Any],
    receipt_meta: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    expected_sha = str(plan_doc.get("plan_sha256") or "").strip()
    got_sha = str(receipt_meta.get("plan_sha256") or receipt_meta.get("plan_id") or "").strip()
    if got_sha and expected_sha and got_sha != expected_sha:
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
    """Dispatch versioned plan on leased devices to ``mindie-profiler`` and normalize receipts."""
    if not isinstance(plan_doc, Mapping) or not plan_doc.get("plan_sha256"):
        raise ModelingError("profiler_bench_plan_doc_invalid")
    if profiler_root is None and ssh_runner is None:
        raise ModelingError("profiler_bench_requires_profiler_root")

    phy_csv = ",".join(str(int(d)) for d in allowed_phy_ids)
    subgraph_plan = build_profiler_subgraph_plan(plan_doc, sku=sku) if domain is None else dict(plan_doc)
    plan_bytes = canonical_json_bytes(subgraph_plan)

    with tempfile.TemporaryDirectory(prefix="msim-prof-bench-") as tmp_dir:
        tmp = Path(tmp_dir)
        plan_file = tmp / "onboarding_plan.json"
        out_file = tmp / "bench_receipt.json"
        obs_file = tmp / "observations.jsonl"
        write_atomic_bytes(plan_file, plan_bytes)

        if ssh_runner is not None and endpoint_alias.strip():
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
                remote_cmd = (
                    f"TMPD=$(mktemp -d); trap 'rm -rf \"$TMPD\"' EXIT; "
                    f"printf '%s' {shlex.quote(plan_b64)} | base64 -d > \"$TMPD/plan.json\"; "
                    f"{env_prefix}{fence_env}"
                    f"MINDIE_ONBOARDING_PLAN_JSON=\"$TMPD/plan.json\" "
                    f"MINDIE_ONBOARDING_PLAN_SHA256={shlex.quote(str(plan_doc['plan_sha256']))} "
                    f"python3 -B -m mindie_profiler bench subgraph --plan \"$TMPD/plan.json\" "
                    f"--out \"$TMPD/receipt.json\" --observations \"$TMPD/observations.jsonl\""
                    + (f" --allowed-phy-ids {shlex.quote(phy_csv)}" if phy_csv else "")
                    + (" --dry-run" if dry_run else "")
                    + "; [ -s \"$TMPD/observations.jsonl\" ] && cat \"$TMPD/observations.jsonl\"; "
                    + f"printf '\\n{_REMOTE_RECEIPT_DELIM}\\n'; "
                    + "[ -s \"$TMPD/receipt.json\" ] && cat \"$TMPD/receipt.json\" || true"
                )
            done = ssh_runner(endpoint_alias.strip(), remote_cmd, int(timeout_s))
            if done.returncode != 0:
                raise ModelingError(f"profiler_bench_remote_failed:rc={done.returncode}:{(done.stderr or done.stdout).strip()}")
            stdout_txt = done.stdout or ""
            if _REMOTE_RECEIPT_DELIM in stdout_txt:
                obs_part, rcpt_part = stdout_txt.split(_REMOTE_RECEIPT_DELIM, 1)
                obs_recs, _ = _parse_json_or_jsonl_records(obs_part)
                rcpt_recs, receipt_meta = _parse_json_or_jsonl_records(rcpt_part)
                raw_records = [*obs_recs, *rcpt_recs]
            else:
                raw_records, receipt_meta = _parse_json_or_jsonl_records(stdout_txt)
            if domain is None:
                _verify_returned_plan_binding(plan_doc, receipt_meta, raw_records)
            return normalize_profiler_records(
                raw_records, plan_doc=plan_doc, sku=sku,
                endpoint_alias=endpoint_alias, capture_out_dir=capture_out_dir,
            )

        assert profiler_root is not None
        prof_root = Path(profiler_root).resolve()
        env = dict(os.environ)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{prof_root}{os.pathsep}{existing_pp}" if existing_pp else str(prof_root)
        env["MINDIE_ONBOARDING_PLAN_JSON"] = str(plan_file)
        env["MINDIE_ONBOARDING_PLAN_SHA256"] = str(plan_doc["plan_sha256"])
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
            cmd = (
                [sys.executable, "-B", "-m", "mindie_profiler", "bench", "sku"]
                if has_pkg_main
                else [sys.executable, "-B", str(mod_file)]
            )
            cmd.extend(["--sku", sku, "--domain", str(int(domain)), "--out", str(out_file), "--observations", str(obs_file)])
        else:
            mod_file = prof_root / "mindie_profiler" / "bench" / "bench_archetype_subgraphs.py"
            if not mod_file.is_file() and not has_pkg_main:
                raise ModelingError(f"profiler_bench_subgraph_script_not_found:{mod_file}")
            cmd = (
                [sys.executable, "-B", "-m", "mindie_profiler", "bench", "subgraph"]
                if has_pkg_main
                else [sys.executable, "-B", str(mod_file)]
            )
            cmd.extend(["--plan", str(plan_file), "--out", str(out_file), "--observations", str(obs_file)])
        if phy_csv:
            cmd.extend(["--allowed-phy-ids", phy_csv])
        if dry_run:
            cmd.append("--dry-run")
        try:
            done = subprocess.run(
                cmd, cwd=str(prof_root), env=env, capture_output=True, text=True,
                timeout=timeout_s, check=False, **subprocess_window_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"profiler_bench_timeout:timeout_s={timeout_s}") from exc
        except TimeoutError:
            raise
        except OSError as exc:
            raise ModelingError(f"profiler_bench_os_error:{exc}") from exc
        if done.returncode != 0:
            raise ModelingError(f"profiler_bench_failed:rc={done.returncode}:{(done.stderr or done.stdout).strip()}")

        combined_records: list[dict[str, Any]] = []
        receipt_meta: dict[str, Any] = {}
        if obs_file.is_file():
            obs_recs, _ = _parse_json_or_jsonl_records(obs_file.read_text(encoding="utf-8"))
            combined_records.extend(obs_recs)
        if out_file.is_file():
            out_recs, receipt_meta = _parse_json_or_jsonl_records(out_file.read_text(encoding="utf-8"))
            combined_records.extend(out_recs)
        if not obs_file.is_file() and not out_file.is_file():
            raise ModelingError("profiler_bench_missing_output_json")
        if domain is None:
            _verify_returned_plan_binding(plan_doc, receipt_meta, combined_records)
        return normalize_profiler_records(
            combined_records, plan_doc=plan_doc, sku=sku,
            endpoint_alias=endpoint_alias, capture_out_dir=capture_out_dir,
        )


__all__ = [
    "build_profiler_subgraph_plan",
    "normalize_profiler_records",
    "run_profiler_bench_subprocess",
]
