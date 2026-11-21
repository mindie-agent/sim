"""Live ``mindie-profiler`` plan-consuming dispatch and real observation/ABI normalizer.

[Role] Dispatch translated ``mindie-bench-subgraph-plan/1`` plans with train
  and heldout separated into independent acquisitions via
  ``mindie_profiler bench subgraph --plan`` (routing every non-local
  ``endpoint_alias`` through sanctioned SSH transport and reserving
  ``bench sku --domain`` strictly for Domain-5 fabric probes), collect both
  unsuffixed and ``.rankN.json`` receipts (accepting ``exit 2`` ``partial`` ONLY
  when every failure matches a planned ABI rejection), normalize multi-trial
  writer observations into sealed capture files + ``sim_case``-bound onboarding
  rows, and re-export ``build_profiler_subgraph_plan``.
[Invariants]
  * Never overwrites writer-measured ``clock_operator`` with the plan's clock;
    fails closed on mismatch before sealing any capture.
  * Non-local ``endpoint_alias`` ALWAYS routes over SSH, never local Python.
  * Train and heldout dispatch as independent acquisitions with disjoint run IDs.
  * Accepts ``rc == 2`` (``partial``) ONLY when every failing record matches a
    planned ABI rejection point; collects all ``.rankN.json`` receipts.
  * ``TimeoutError`` always propagates before ``OSError``; windowless spawn.
[Timeouts & Locks] Synchronous bounded subprocesses; split acquisition timeouts
  sum to ``<= timeout_s``; no locks.
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
import shlex
import statistics
import subprocess
import sys
import tempfile
from typing import Any

from mindie_sim.schema.common import ModelingError

from ._profiler_plan import (
    _DEFAULT_SIM_MODEL,
    build_profiler_subgraph_plan,
    derive_writer_clock_operator,
    extract_observed_shape,
    extract_shape_dims,
    parse_run_id_and_arm,
    parse_topo_options,
    quantile_ns,
    resolve_point_binding_and_shape,
    select_mode_candidates,
)
from ._soc_probe import run_ssh_command
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
_PARTIAL_EXIT_CODE = 2
_SPLIT_TIMEOUT_DIVISOR = 2
_MAX_UNEXPECTED_REPORTED = 8
_LOCAL_ENDPOINT_ALIASES = frozenset({"", "local", "localhost", "127.0.0.1", "::1"})
_ABI_REJECT_STATUSES = frozenset({"abi_rejected", "unsupported_binding"})
_PASSING_RECEIPT_STATUSES = frozenset({"ok", "skipped_stage", "skipped_non_participant"})
_OBSERVATION_SCHEMA = "mindie-bench-observation/1"
_RECEIPT_SCHEMA = "mindie-bench-receipt/1"
_SUBGRAPH_RUN_SCHEMA = "mindie-bench-subgraph-run/1"
_REMOTE_RECEIPT_DELIM = "__MINDIE_RECEIPT_JSON__"


def _is_remote_endpoint(endpoint_alias: str) -> bool:
    return endpoint_alias.strip().lower() not in _LOCAL_ENDPOINT_ALIASES


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
            for key in ("cases", "records", "receipts", "abi_rejections"):
                val = raw["receipt"].get(key)
                if isinstance(val, list):
                    collected.extend(dict(r) for r in val if isinstance(r, Mapping))
        if not collected and (raw.get("record_type") or raw.get("case_id")):
            collected.append(dict(raw))
        return collected, dict(raw)
    if not isinstance(raw, list):
        raise ModelingError("profiler_bench_observations_not_list")
    return [dict(r) for r in raw if isinstance(r, Mapping)], {}


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
        str(r.get("schema") or "") in {_OBSERVATION_SCHEMA, _RECEIPT_SCHEMA, _SUBGRAPH_RUN_SCHEMA}
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
    obs_buckets: dict[tuple[str, str, str], dict[tuple[str, str], list[Mapping[str, Any]]]] = {}
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
                    "point_id": cid, "rejected": True,
                    "abi_error": str(rec.get("abi_error") or rec.get("reason") or ""),
                    "executed_shape": rec.get("executed_shape") or rec.get("requested_shape") or rec.get("actual_shape"),
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

    normalized: list[dict[str, Any]] = []
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
            durations_ns = sorted(
                int(round(float(r["duration_ms"]) * NS_PER_MS))
                for r in group
                if r.get("duration_ms") is not None and float(r["duration_ms"]) > 0.0
            )
            if not durations_ns:
                continue
            p50_ns = int(round(statistics.median(durations_ns)))
            p99_ns = quantile_ns(durations_ns, _P99_QUANTILE)
            plan_key = (
                summary.get("plan_key")
                if isinstance(summary.get("plan_key"), Mapping)
                else (first.get("plan_key") if isinstance(first.get("plan_key"), Mapping) else {})
            )
            req_shape = extract_observed_shape(summary, first)
            dims = extract_shape_dims(req_shape, plan_key)

            ranks_raw = summary.get("participating_ranks")
            if not (isinstance(ranks_raw, Sequence) and not isinstance(ranks_raw, (str, bytes)) and ranks_raw):
                obs_ranks = sorted({
                    int(r["rank"]) for r in group
                    if r.get("rank") is not None and not isinstance(r.get("rank"), bool)
                })
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
            raw_level = str(first.get("level") or ident.get("level") or "operator").strip()
            timed_nodes = summary.get("timed_nodes") if isinstance(summary.get("timed_nodes"), Sequence) else ()
            level = raw_level.upper() if raw_level.upper() in {"L1", "L2", "L3"} else ("L1" if (raw_level.lower() == "operator" or len(timed_nodes) <= 1) else "L2")

            if plan_pt is not None:
                _pf, _pr, _pb, _imp, sim_fam, _sh, sim_case = resolve_point_binding_and_shape(
                    plan_doc, plan_pt, sku=sku
                )
            else:
                sim_fam = str(summary.get("op") or ("" if plan_family == "sku_suite" else plan_family) or "rmsnorm")
                sim_case = {
                    "compose_model": _DEFAULT_SIM_MODEL, "compose_hardware": sku,
                    "compose_options": parse_topo_options(dims["parallel_topology"]),
                }
            normalized.append({
                "case_id": cid, "point_id": cid, "sku": str(first.get("sku") or sku),
                "split": split, "level": level,
                "acquisition_root": f"acq-{split}-{pub_run_id}-{cid}", "run_uid": pub_run_id,
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
                "target_family": sim_fam, "sim_case": sim_case,
                "plan_key": dict(plan_key), "actual_shape": dict(req_shape), **dims,
            })
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
    planned_abi_ids: set[str],
    err_detail: str,
    *,
    is_remote: bool,
) -> None:
    prefix = "profiler_bench_remote_failed" if is_remote else "profiler_bench_failed"
    if returncode == 0:
        return
    if returncode != _PARTIAL_EXIT_CODE or not receipt_metas:
        raise ModelingError(f"{prefix}:rc={returncode}:{err_detail}")
    if any(str(m.get("status") or "").strip().lower() not in {"ok", "partial"} for m in receipt_metas):
        raise ModelingError(f"{prefix}:rc={returncode}:receipt_status_invalid:{err_detail}")
    non_passing = [
        r for r in receipt_records
        if str(r.get("status") or "ok").strip().lower() not in _PASSING_RECEIPT_STATUSES
    ]
    if not non_passing:
        raise ModelingError(f"{prefix}:rc={returncode}:partial_without_planned_rejection:{err_detail}")
    unexpected = [
        f"{r.get('case_id') or r.get('point_id') or 'unknown'}={r.get('status')}"
        for r in non_passing
        if str(r.get("status") or "").strip().lower() not in _ABI_REJECT_STATUSES
        or str(r.get("case_id") or r.get("point_id") or "").strip() not in planned_abi_ids
    ]
    if unexpected:
        raise ModelingError(
            f"profiler_bench_unexpected_partial_failure:{','.join(unexpected[:_MAX_UNEXPECTED_REPORTED])}"
        )


def _split_disjoint_acquisitions(
    plan_doc: Mapping[str, Any], *, domain: int | None, dry_run: bool
) -> list[tuple[str, dict[str, Any]]]:
    if domain is not None or dry_run:
        return [("", dict(plan_doc))]
    has_train = bool(plan_doc.get("train_points"))
    has_held = bool(plan_doc.get("heldout_points") or plan_doc.get("abi_rejection_points"))
    if has_train and has_held:
        return [
            ("train", {**dict(plan_doc), "heldout_points": [], "abi_rejection_points": []}),
            ("heldout", {**dict(plan_doc), "train_points": []}),
        ]
    return [("", dict(plan_doc))]


def _dispatch_single_acquisition(
    sub_plan: Mapping[str, Any],
    *,
    sku: str,
    split_tag: str,
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
    phy_csv = ",".join(str(int(d)) for d in allowed_phy_ids)
    subgraph_plan = build_profiler_subgraph_plan(sub_plan, sku=sku, split_tag=split_tag) if domain is None else dict(sub_plan)
    plan_bytes = canonical_json_bytes(subgraph_plan)
    planned_abi_ids = {
        str(pt["point_id"])
        for pt in (sub_plan.get("abi_rejection_points") or ())
        if isinstance(pt, Mapping) and pt.get("point_id")
    }

    with tempfile.TemporaryDirectory(prefix="msim-prof-bench-") as tmp_dir:
        tmp = Path(tmp_dir)
        plan_file, out_file, obs_file = tmp / "onboarding_plan.json", tmp / "bench_receipt.json", tmp / "observations.jsonl"
        write_atomic_bytes(plan_file, plan_bytes)

        if use_ssh:
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
                remote_cmd = (
                    f"TMPD=$(mktemp -d); trap 'rm -rf \"$TMPD\"' EXIT; "
                    f"printf '%s' {shlex.quote(plan_b64)} | base64 -d > \"$TMPD/plan.json\"; "
                    f"{env_prefix}{fence_env}"
                    f"MINDIE_ONBOARDING_PLAN_JSON=\"$TMPD/plan.json\" "
                    f"MINDIE_ONBOARDING_PLAN_SHA256={shlex.quote(str(sub_plan['plan_sha256']))} "
                    f"python3 -B -m mindie_profiler bench subgraph --plan \"$TMPD/plan.json\" "
                    f"--out \"$TMPD/receipt.json\" --observations \"$TMPD/observations.jsonl\""
                    + (f" --allowed-phy-ids {shlex.quote(phy_csv)}" if phy_csv else "")
                    + (" --dry-run" if dry_run else "")
                    + "; RC=$?; [ -s \"$TMPD/observations.jsonl\" ] && cat \"$TMPD/observations.jsonl\"; "
                    + "for RF in \"$TMPD/receipt.json\" \"$TMPD\"/receipt.json.rank*.json; do "
                    + f"[ -s \"$RF\" ] && printf '\\n{_REMOTE_RECEIPT_DELIM}\\n' && cat \"$RF\"; done; exit $RC"
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
                done.returncode, receipt_metas, rcpt_recs, planned_abi_ids,
                (done.stderr or done.stdout).strip(), is_remote=True,
            )
            raw_records = [*obs_recs, *rcpt_recs]
            if domain is None:
                _verify_returned_plan_binding(sub_plan, receipt_metas, raw_records, split_tag=split_tag)
            return normalize_profiler_records(
                raw_records, plan_doc=sub_plan, sku=sku,
                endpoint_alias=endpoint_alias, capture_out_dir=capture_out_dir,
            )

        assert profiler_root is not None
        prof_root = Path(profiler_root).resolve()
        env = dict(os.environ)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{prof_root}{os.pathsep}{existing_pp}" if existing_pp else str(prof_root)
        env["MINDIE_ONBOARDING_PLAN_JSON"] = str(plan_file)
        env["MINDIE_ONBOARDING_PLAN_SHA256"] = str(sub_plan["plan_sha256"])
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
            cmd.extend(["--sku", sku, "--domain", str(int(domain)), "--out", str(out_file), "--observations", str(obs_file)])
        else:
            mod_file = prof_root / "mindie_profiler" / "bench" / "bench_archetype_subgraphs.py"
            if not mod_file.is_file() and not has_pkg_main:
                raise ModelingError(f"profiler_bench_subgraph_script_not_found:{mod_file}")
            cmd = [sys.executable, "-B", "-m", "mindie_profiler", "bench", "subgraph"] if has_pkg_main else [sys.executable, "-B", str(mod_file)]
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

        receipt_files = ([out_file] if out_file.is_file() else []) + sorted(out_file.parent.glob(f"{out_file.name}.rank*.json"))
        obs_recs, rcpt_recs, receipt_metas = [], [], []
        if obs_file.is_file():
            obs_recs, _ = _parse_json_or_jsonl_records(obs_file.read_text(encoding="utf-8"))
        for rfile in receipt_files:
            recs, meta = _parse_json_or_jsonl_records(rfile.read_text(encoding="utf-8"))
            rcpt_recs.extend(recs)
            if meta:
                receipt_metas.append(meta)
        _validate_dispatch_exit(
            done.returncode, receipt_metas, rcpt_recs, planned_abi_ids,
            (done.stderr or done.stdout).strip(), is_remote=False,
        )
        if not obs_file.is_file() and not receipt_files:
            raise ModelingError("profiler_bench_missing_output_json")
        combined_records = [*obs_recs, *rcpt_recs]
        if domain is None:
            _verify_returned_plan_binding(sub_plan, receipt_metas, combined_records, split_tag=split_tag)
        return normalize_profiler_records(
            combined_records, plan_doc=sub_plan, sku=sku,
            endpoint_alias=endpoint_alias, capture_out_dir=capture_out_dir,
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
    """Dispatch versioned plan on leased devices (train & heldout independently) and normalize."""
    if not isinstance(plan_doc, Mapping) or not plan_doc.get("plan_sha256"):
        raise ModelingError("profiler_bench_plan_doc_invalid")
    acquisitions = _split_disjoint_acquisitions(plan_doc, domain=domain, dry_run=dry_run)
    remaining = max(1, int(timeout_s))
    all_obs: list[dict[str, Any]] = []
    all_abi: list[dict[str, Any]] = []
    for idx, (split_tag, sub_plan) in enumerate(acquisitions):
        left = len(acquisitions) - idx
        step_budget = remaining if left <= 1 else max(1, remaining // _SPLIT_TIMEOUT_DIVISOR)
        remaining = max(1, remaining - step_budget)
        obs_part, abi_part = _dispatch_single_acquisition(
            sub_plan, sku=sku, split_tag=split_tag, profiler_root=profiler_root,
            endpoint_alias=endpoint_alias, allowed_phy_ids=allowed_phy_ids,
            fence_token=fence_token, domain=domain, dry_run=dry_run,
            timeout_s=step_budget, capture_out_dir=capture_out_dir, ssh_runner=ssh_runner,
        )
        all_obs.extend(obs_part)
        all_abi.extend(abi_part)
    return all_obs, all_abi


__all__ = [
    "build_profiler_subgraph_plan",
    "normalize_profiler_records",
    "run_profiler_bench_subprocess",
]
