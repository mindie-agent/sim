"""Live ``mindie-profiler`` plan dispatch and observation/ABI receipt normalizer.

[Role] Dispatch versioned onboarding cliff/SKU plans to ``mindie-profiler``
  (forwarding leased physical device IDs, ``fence_token``, ``endpoint_alias``,
  and materialized plan JSON) and normalize real ``mindie-bench-observation/1``
  / ``mindie-bench-receipt/1`` JSONL records into sealed onboarding
  observations (with computed ``p50``/``p99``, explicit ``clock_operator``,
  verified ``raw_sha256``, ``event_ids``, ``device_ranks``, and ABI rejections).
[Invariants] Never drops ``plan_doc``; forwards leased ``allowed_phy_ids`` and
  ``fence_token``; ``TimeoutError`` always propagates before ``OSError``;
  windowless spawn flags on Windows; zero fabricated timings.
[Timeouts & Locks] Synchronous bounded subprocesses; no locks.
[Side-Effect Policy] Writes ephemeral plan/output temp files cleaned up
  unconditionally; writes raw capture JSONs only to explicit ``capture_out_dir``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
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
    subprocess_window_kwargs,
    write_atomic_json,
)

_P99_QUANTILE = 0.99
_ABI_REJECT_STATUSES = frozenset({"abi_rejected", "unsupported_binding"})
_BUSY_UNION_MODES = frozenset({"device_kernel", "event_busy_union", "busy_union"})
_OBSERVATION_SCHEMA = "mindie-bench-observation/1"


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


def _parse_json_or_jsonl_records(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
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
        return rows
    if isinstance(raw, Mapping):
        for key in ("observations", "records", "cases"):
            if isinstance(raw.get(key), list):
                return [dict(r) for r in raw[key] if isinstance(r, Mapping)]
        return []
    if not isinstance(raw, list):
        raise ModelingError("profiler_bench_observations_not_list")
    return [dict(r) for r in raw if isinstance(r, Mapping)]


def _extract_shape_dims(
    shape: Mapping[str, Any], plan_key: Mapping[str, Any]
) -> dict[str, Any]:
    dims: dict[str, Any] = {}
    merged = {**dict(plan_key), **dict(shape)}
    for dst, candidates in (
        ("context_len", ("context_len", "context", "kv_len", "seq_len", "scanned_rows")),
        ("tokens", ("tokens", "m", "num_tokens")),
        ("batch_size", ("batch_size", "batch", "b", "sequences")),
        ("query_tokens", ("query_tokens", "q", "q_len")),
        ("topk", ("topk", "topk_items")),
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
        str(r.get("schema") or "") in {_OBSERVATION_SCHEMA, "mindie-bench-receipt/1"}
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
    completes: dict[tuple[str, str], Mapping[str, Any]] = {}
    obs_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    abi_rows: list[dict[str, Any]] = []

    for rec in raw_records:
        rtype = str(rec.get("record_type") or "")
        status = str(rec.get("status") or "ok").strip().lower()
        cid = str(rec.get("case_id") or "").strip()
        run_id = str(rec.get("run_id") or "run-0").strip()
        if status in _ABI_REJECT_STATUSES:
            abi_rows.append({
                "point_id": cid,
                "rejected": True,
                "abi_error": str(rec.get("abi_error") or rec.get("reason") or ""),
                "executed_shape": rec.get("requested_shape") or rec.get("actual_shape"),
            })
            continue
        if rtype == "case_complete":
            completes[(cid, run_id)] = rec
        elif rtype == "observation":
            split = "heldout" if str(rec.get("split") or "") == "heldout" else "train"
            obs_groups.setdefault((cid, run_id, split), []).append(rec)

    normalized: list[dict[str, Any]] = []
    for (cid, run_id, split), group in sorted(obs_groups.items()):
        comp = completes.get((cid, run_id)) or {}
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
        boundary_name = str(first.get("boundary") or "device_kernel").strip()
        timing_mode = str(first.get("timing_mode") or "").strip()
        clock_op = "busy_union" if (boundary_name in _BUSY_UNION_MODES or timing_mode in _BUSY_UNION_MODES) else "elapsed_span"
        plan_key = summary.get("plan_key") if isinstance(summary.get("plan_key"), Mapping) else (first.get("plan_key") if isinstance(first.get("plan_key"), Mapping) else {})
        req_shape = summary.get("actual_shape") or summary.get("requested_shape") or first.get("requested_shape") or {}
        if not isinstance(req_shape, Mapping):
            req_shape = {}
        dims = _extract_shape_dims(req_shape, plan_key)
        ranks_raw = summary.get("participating_ranks") or ([first["rank"]] if first.get("rank") is not None else [0])
        ranks = [int(r) for r in ranks_raw if not isinstance(r, bool)] or [0]
        ev_ids = [str(r.get("sample_id") or f"{cid}-s{i}") for i, r in enumerate(group)]
        cap_payload = {
            "case_id": cid, "run_id": run_id, "split": split, "sku": str(first.get("sku") or sku),
            "clock_operator": clock_op, "samples_ns": durations_ns,
            "plan_key": dict(plan_key), "actual_shape": dict(req_shape),
        }
        cap_bytes = canonical_json_bytes(cap_payload)
        raw_sha = hashlib.sha256(cap_bytes).hexdigest()
        cap_uri = f"capture://{clean_ep}/{run_id}/{raw_sha}.json"
        if capture_out_dir is not None:
            write_atomic_json(Path(capture_out_dir) / f"{raw_sha}.json", cap_payload)
        ident = first.get("identity") if isinstance(first.get("identity"), Mapping) else {}
        acq_root = f"acq-{split}-{run_id}-{cid}"
        row: dict[str, Any] = {
            "case_id": cid,
            "point_id": cid,
            "sku": str(first.get("sku") or sku),
            "split": split,
            "level": "L1" if str(ident.get("level") or "operator") == "operator" else "L2",
            "acquisition_root": acq_root,
            "run_uid": run_id,
            "raw_sha256": raw_sha,
            "capture_uri": cap_uri,
            "event_ids": ev_ids,
            "device_ranks": ranks,
            "clock_operator": clock_op,
            "measured_ns": p50_ns,
            "measured_p99_ns": p99_ns,
            "busy_union_ns": p50_ns if clock_op == "busy_union" else None,
            "elapsed_span_ns": p50_ns if clock_op == "elapsed_span" else None,
            "validated": all(bool(r.get("validated", True)) for r in group),
            "qualified": all(bool(r.get("qualified", True)) for r in group),
            "implementation": str(ident.get("implementation") or summary.get("op") or "profiler_bench"),
            "target_family": str(summary.get("op") or plan_family),
            "plan_key": dict(plan_key),
            "actual_shape": dict(req_shape),
            **dims,
        }
        normalized.append(row)
    return normalized, abi_rows


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
    with tempfile.TemporaryDirectory(prefix="msim-prof-bench-") as tmp_dir:
        tmp = Path(tmp_dir)
        plan_file = tmp / "onboarding_plan.json"
        out_file = tmp / "bench_receipt.json"
        obs_file = tmp / "observations.jsonl"
        write_atomic_json(plan_file, dict(plan_doc))

        if ssh_runner is not None and endpoint_alias.strip():
            env_prefix = (
                f"ASCEND_RT_VISIBLE_DEVICES={shlex.quote(phy_csv)} "
                if phy_csv else ""
            )
            fence_env = f"MINDIE_LEASE_FENCE_TOKEN={int(fence_token)} " if fence_token is not None else ""
            remote_cmd = (
                f"{env_prefix}{fence_env}"
                f"python3 -B -m mindie_profiler bench sku --sku {shlex.quote(sku)}"
                + (f" --allowed-phy-ids {shlex.quote(phy_csv)}" if phy_csv else "")
                + (f" --domain {int(domain)}" if domain is not None else "")
                + (" --dry-run" if dry_run else "")
            )
            done = ssh_runner(endpoint_alias.strip(), remote_cmd, int(timeout_s))
            if done.returncode != 0:
                raise ModelingError(f"profiler_bench_remote_failed:rc={done.returncode}:{(done.stderr or done.stdout).strip()}")
            raw_records = _parse_json_or_jsonl_records(done.stdout or "")
            return normalize_profiler_records(
                raw_records, plan_doc=plan_doc, sku=sku,
                endpoint_alias=endpoint_alias, capture_out_dir=capture_out_dir,
            )

        assert profiler_root is not None
        prof_root = Path(profiler_root).resolve()
        mod_file = prof_root / "mindie_profiler" / "bench" / "bench_sku_microbench.py"
        if not mod_file.is_file():
            raise ModelingError(f"profiler_bench_script_not_found:{mod_file}")
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

        is_subgraph_plan = str(plan_doc.get("schema") or "") == "mindie-bench-subgraph-plan/1"
        has_pkg_main = (prof_root / "mindie_profiler" / "__main__.py").is_file()
        if is_subgraph_plan:
            cmd = [
                sys.executable, "-B", "-m", "mindie_profiler", "bench", "subgraph",
                "--plan", str(plan_file), "--out", str(out_file), "--observations", str(obs_file),
            ]
        else:
            cmd = (
                [sys.executable, "-B", "-m", "mindie_profiler", "bench", "sku"]
                if has_pkg_main
                else [
                    sys.executable, "-B", "-c",
                    "import sys; from mindie_profiler.bench.bench_sku_microbench import main; raise SystemExit(main(sys.argv[1:]))",
                ]
            )
            cmd.extend(["--sku", sku, "--out", str(out_file), "--observations", str(obs_file)])
            if domain is not None:
                cmd.extend(["--domain", str(int(domain))])
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
        target_file = obs_file if obs_file.is_file() else out_file
        if not target_file.is_file():
            raise ModelingError("profiler_bench_missing_output_json")
        raw_records = _parse_json_or_jsonl_records(target_file.read_text(encoding="utf-8"))
        return normalize_profiler_records(
            raw_records, plan_doc=plan_doc, sku=sku,
            endpoint_alias=endpoint_alias, capture_out_dir=capture_out_dir,
        )


__all__ = ["normalize_profiler_records", "run_profiler_bench_subprocess"]
