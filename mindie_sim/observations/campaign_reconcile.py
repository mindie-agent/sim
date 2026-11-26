"""Fail-closed silicon campaign reconciliation, matrix enrichment, and version diff.

[Role] Own packaged ``mindie-bench-campaign/1`` integrity verification, observed
  AICore clock normalization, thermal-throttle/out-of-band rejection, traced
  ``busy_union`` validation, bound ``L*`` non-inversion checking, exact plan
  multiset & ``OpOccurrence`` conservation, cross-plan training-manifest heldout
  independence, DAG-replayed step delta enrichment, and cross-version diffing
  (plus all legacy fleet 400K verification helpers promoted from ``tools/``).
[Invariants]
  * Rejects any caller-supplied ``predicted_*`` or ``sim_reference`` key and any
    private provenance leak (IPv4/IPv6, ``$HOME``, ``/dev/davinci*``, registries).
  * Verifies ``campaign_sha256``, immutable per-run shard ``observations_sha256``,
    ``raw_sha256``, and ``plan_sha256`` fail-closed before scoring.
  * ``clock_operator == "busy_union"`` requires traced device kernel intervals
    (``traced_interval_verified=True``); raw ACLGraph/event-pair spans without
    traced intervals fail closed.
  * ``L*`` checking binds strictly to identical ``(timed_nodes, participating_ranks,
    clock_operator, normalized_clock_mhz)`` and enforces ``0 < L* <= sim, P50, P99``.
  * A single missing planned case, single ``OpOccurrence`` mismatch, single
    pointwise tolerance breach, or single ``train``/``heldout`` geometry overlap
    (including against ``training_manifest_sha256``) fails the gate.
  * Step deltas replay the execution DAG (longest path), never linearly adding
    overlapping multi-stream kernel durations.
[Timeouts & Locks] Pure deterministic CPU verification and bounded local reads
  (<= 50 MiB JSON, streaming SHA-256 on shards); no locks or device contexts.
[Side-Effect Policy] Read-only except when CLI explicitly writes ``--out``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from mindie_sim.backends.ascend import profiles as ascend_profiles
from mindie_sim.observations.bench_packets import PacketError, assert_public_provenance
from mindie_sim.observations.comparison import apply_capture_observed_clock
from mindie_sim.schema.bench_matrix import (
    BENCH_CAMPAIGN_V1_SCHEMA,
    BENCH_PLAN_V2_SCHEMA,
    BenchPlanV2,
    CampaignCaseRecord,
    CampaignManifest,
    HardwareNominalSpec,
    OperatorMatrixManifest,
    OperatorMatrixRow,
    StepSummary,
    assert_no_forbidden_prediction_keys,
    canonical_sha256,
    compute_campaign_sha256,
    compute_case_raw_sha256,
    compute_plan_sha256,
)
from mindie_sim.schema.common import ModelingError

REL_TOL = 0.10
ABS_FLOOR_NS = 2000.0
L1_MEDIAN_APE_MAX_PCT = 8.0
L1_P90_APE_MAX_PCT = 12.0
L1_CLIFF_REL_TOL = 0.15
L1_CLIFF_ABS_FLOOR_NS = 3000.0
P90_QUANTILE = 0.9
NS_PER_US = 1000.0
US_PER_MS = 1000.0
US_PER_SECOND = 1_000_000.0
DCP_MAX_VERIFIED_KV = 32768
DCP_MAX_ABS_ERR = 0.05
FAIL_SENTINEL_APE = 999.0
DIGEST_PREFIX_LEN = 16
PCT_SCALE = 100.0
ROUND_APE_DIGITS = 4
DEFAULT_Q_LEN = 1
DEFAULT_RECURRENT_BATCH = 1
GDN_REC_DEFAULT_H = 16
KDA_REC_DEFAULT_H = 8
MAX_DIFF_PREVIEW = 5
MIN_POSITIVE = 0.0
P99_ORDER_EPS = 1e-9
SHARD_DURATION_TOL_US = 1e-3
CLOCK_MATCH_REL_TOL = 1e-3
MIN_QUALIFIED_CLOCK_RATIO = 0.90
MAX_QUALIFIED_CLOCK_RATIO = 1.15
SHARD_HASH_CHUNK_BYTES = 1024 * 1024
MAX_CAMPAIGN_JSON_BYTES = 50 * 1024 * 1024
CLI_RECONCILE_FAIL_EXIT = 1
CLI_ERROR_EXIT = 2
DEFAULT_TIMED_NODES: tuple[str, ...] = ("n0",)
DEFAULT_PARTICIPATING_RANKS: tuple[int, ...] = (0,)
ABORT_SUFFIXES: tuple[str, ...] = ("_ABORT", "_TIMEOUT")
ABORT_ERR_PREFIXES: tuple[str, ...] = ("worker_abort_", "timeout_")


class CampaignIntegrityError(ValueError):
    """Fail-closed integrity, anti-spoofing, provenance, or conservation error."""


def nearest_rank_p90(values: Sequence[float]) -> float:
    """Return canonical nearest-rank P90 (index ceil(0.9 * n) - 1) without pre-rounding."""
    if not values:
        return MIN_POSITIVE
    ordered = sorted(float(v) for v in values)
    idx = max(0, min(len(ordered) - 1, math.ceil(P90_QUANTILE * len(ordered)) - 1))
    return float(ordered[idx])


def canonical_capture_sha256(doc: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 of a legacy capture JSON payload excluding raw_sha256."""
    body = {k: v for k, v in doc.items() if k != "raw_sha256"}
    raw = (json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_verified_capture(path: Path) -> tuple[dict[str, Any], str]:
    """Load capture JSON and fail closed if inner raw_sha256 does not verify."""
    if not path.is_file():
        raise FileNotFoundError(f"missing_required_capture:{path.name}")
    raw_bytes = path.read_bytes()
    file_sha = hashlib.sha256(raw_bytes).hexdigest()
    doc = json.loads(raw_bytes.decode("utf-8"))
    stored = str(doc.get("raw_sha256") or "")
    if not stored:
        raise ValueError(f"capture_missing_raw_sha256:{path.name}")
    calc = canonical_capture_sha256(doc)
    if stored != calc:
        raise ValueError(
            f"capture_inner_digest_mismatch:{path.name}:{stored[:DIGEST_PREFIX_LEN]}!={calc[:DIGEST_PREFIX_LEN]}"
        )
    return doc, file_sha


def _positive_clock_from_dict(entry: Any) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    for key in ("cur_freq_mhz", "observed_clock_mhz", "aicore_freq_mhz", "observed_aicore_freq_mhz"):
        val = entry.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)) and float(val) > MIN_POSITIVE:
            return float(val)
    return None


def extract_observed_clock_mhz(
    doc: Mapping[str, Any], fallback_mhz: float | None = None,
) -> float | None:
    """Extract capture-recorded AICore frequency (MHz), falling back if unavailable."""
    for key in ("clock_start", "clock_end", "clock", "version_fingerprint"):
        val = _positive_clock_from_dict(doc.get(key))
        if val is not None:
            return val
    top = _positive_clock_from_dict(doc)
    if top is not None:
        return top
    ops_clks = doc.get("clocks_by_op")
    if isinstance(ops_clks, Mapping):
        for entry in ops_clks.values():
            if isinstance(entry, Mapping):
                for sub in ("start", "end"):
                    val = _positive_clock_from_dict(entry.get(sub))
                    if val is not None:
                        return val
    ranks = doc.get("rank_to_die_map")
    if isinstance(ranks, Sequence):
        for entry in ranks:
            if isinstance(entry, Mapping):
                val = _positive_clock_from_dict(entry.get("clock"))
                if val is not None:
                    return val
    return fallback_mhz


def apply_sku_observed_clock(
    case_dict: Mapping[str, Any], observed_mhz: float | None,
) -> dict[str, Any]:
    """Scale core-clock-coupled machine peaks/rates to the observed board clock."""
    if observed_mhz is None or not math.isfinite(observed_mhz) or observed_mhz <= MIN_POSITIVE:
        return dict(case_dict)
    return apply_capture_observed_clock(case_dict, {"observed_clock_mhz": float(observed_mhz)})


def score_verdict(
    case_id: str, family: str, split: str, meas_p50: float, meas_p99: float,
    pred_us: float, lstar_us: float, capture_uri: str,
    *, acquisition_uid: str = "", boundary: str = "cold", geometry_key: str = "",
) -> dict[str, Any]:
    """Score one prediction against silicon P50/P99 with strict L* > 0 and P50/P99 non-inversion."""
    cid = str(case_id or f"{family}_{split}_unnamed")
    finite_ok = (
        math.isfinite(meas_p50) and meas_p50 > MIN_POSITIVE
        and math.isfinite(meas_p99) and meas_p99 > MIN_POSITIVE
        and meas_p99 + P99_ORDER_EPS >= meas_p50
        and math.isfinite(pred_us) and pred_us > MIN_POSITIVE
        and math.isfinite(lstar_us) and lstar_us > MIN_POSITIVE
    )
    if not finite_ok:
        return {
            "case_id": cid, "family": family, "split": split, "status": "fail",
            "l1_cliff_status": "fail",
            "acquisition_uid": acquisition_uid, "boundary": boundary, "geometry_key": geometry_key,
            "meas_us": float(meas_p50) if math.isfinite(meas_p50) else None,
            "meas_p99_us": float(meas_p99) if math.isfinite(meas_p99) else None,
            "pred_us": float(pred_us) if math.isfinite(pred_us) else None,
            "lstar_us": float(lstar_us) if math.isfinite(lstar_us) else None,
            "ape_pct": FAIL_SENTINEL_APE, "inv_sim": True, "inv_obs": True, "capture_uri": capture_uri,
        }
    meas_ns = meas_p50 * NS_PER_US
    p99_ns = meas_p99 * NS_PER_US
    pred_ns = pred_us * NS_PER_US
    lstar_ns = lstar_us * NS_PER_US
    err_ns = abs(pred_ns - meas_ns)
    tol_ns = max(REL_TOL * meas_ns, ABS_FLOOR_NS)
    cliff_tol_ns = max(L1_CLIFF_REL_TOL * meas_ns, L1_CLIFF_ABS_FLOOR_NS)
    inv_sim = lstar_ns > pred_ns
    inv_obs = (lstar_ns > meas_ns) or (lstar_ns > p99_ns)
    passed = (err_ns <= tol_ns) and (not inv_sim) and (not inv_obs)
    cliff_passed = (err_ns <= cliff_tol_ns) and (not inv_sim) and (not inv_obs)
    ape_pct = (err_ns / meas_ns) * PCT_SCALE
    return {
        "case_id": cid, "family": family, "split": split,
        "status": "pass" if passed else "fail",
        "l1_cliff_status": "pass" if cliff_passed else "fail",
        "acquisition_uid": acquisition_uid, "boundary": boundary, "geometry_key": geometry_key,
        "meas_us": float(meas_p50), "meas_p99_us": float(meas_p99),
        "pred_us": float(pred_us), "lstar_us": float(lstar_us),
        "ape_pct": round(ape_pct, ROUND_APE_DIGITS), "inv_sim": inv_sim, "inv_obs": inv_obs,
        "capture_uri": capture_uri,
    }


def is_worker_diagnostic_record(record: Mapping[str, Any]) -> bool:
    """Detect process-level abort/timeout diagnostics that are not shape verdicts."""
    cid = str(record.get("case_id") or "")
    err = str(record.get("error") or "")
    return cid.endswith(ABORT_SUFFIXES) or err.startswith(ABORT_ERR_PREFIXES)


def dcp_reference_verified(record: Mapping[str, Any]) -> tuple[bool, str]:
    """Require an executed numerical reference receipt for DCP/PCP records."""
    if record.get("status") == "blocked" or record.get("error"):
        return False, str(record.get("error") or "blocked")
    if not record.get("total_us") or not record.get("fia_us") or not record.get("comm_us") or not record.get("merge_us"):
        return False, "dcp_pcp_missing_component_timings"
    raw_tot_kv = record.get("total_kv")
    if isinstance(raw_tot_kv, bool) or not isinstance(raw_tot_kv, (int, float)) or not math.isfinite(float(raw_tot_kv)) or int(raw_tot_kv) <= 0:
        return False, f"dcp_pcp_invalid_total_kv:{raw_tot_kv!r}"
    tot_kv = int(raw_tot_kv)
    raw_err = record.get("max_abs_err")
    if isinstance(raw_err, bool) or not isinstance(raw_err, (int, float)) or not math.isfinite(float(raw_err)) or float(raw_err) < 0.0:
        return False, f"dcp_pcp_invalid_max_abs_err:{raw_err!r}"
    err_val = float(raw_err)
    if tot_kv > DCP_MAX_VERIFIED_KV and err_val <= MIN_POSITIVE:
        return False, "dcp_pcp_reference_not_executed_L_gt_32768"
    if type(record.get("verified")) is not bool or not record["verified"] or err_val > DCP_MAX_ABS_ERR:
        return False, f"dcp_pcp_reference_failed:err={err_val}"
    return True, ""


def recurrent_executed_geometry(
    op_name: str, record: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Normalize executed (batch, heads, q_len) for recurrent linear-attention records."""
    raw_shape = record.get("actual_executed_shape")
    shape = raw_shape if isinstance(raw_shape, Mapping) else {}
    default_h = GDN_REC_DEFAULT_H if op_name == "gdn_recurrent" else KDA_REC_DEFAULT_H
    batch_candidates = (record.get("B"), record.get("batch"), shape.get("batch"), shape.get("B"), record.get("seqlen"))
    heads_candidates = (record.get("H"), record.get("heads"), shape.get("heads"), shape.get("H"))
    q_candidates = (record.get("q_len"), shape.get("q_len"), shape.get("T"))
    batch = next((int(v) for v in batch_candidates if v is not None and not isinstance(v, bool) and int(v) > 0), DEFAULT_RECURRENT_BATCH)
    heads = next((int(v) for v in heads_candidates if v is not None and not isinstance(v, bool) and int(v) > 0), default_h)
    q_len = next((int(v) for v in q_candidates if v is not None and not isinstance(v, bool) and int(v) > 0), DEFAULT_Q_LEN)
    return batch, heads, q_len


def seq_geometry_key(op_name: str, record: Mapping[str, Any]) -> str:
    """Canonical geometry string independent of split-tagged case_id."""
    if op_name == "ragged_suite":
        sub = str(record.get("sub_op") or "")
        ql = tuple(int(x) for x in record.get("q_lens", ()))
        sl = tuple(int(x) for x in record.get("seq_lens", ()))
        return f"{op_name}:{sub}:q={ql}:kv={sl}"
    if op_name == "dcp_lse_attention_merge":
        return f"{op_name}:cp={int(record.get('cp', 0))}:q={int(record.get('q_len', DEFAULT_Q_LEN))}"
    if op_name in {"gdn_recurrent", "kda_recurrent"}:
        batch, heads, q_len = recurrent_executed_geometry(op_name, record)
        return f"{op_name}:B={batch}:H={heads}:q={q_len}"
    return f"{op_name}:L={int(record.get('seqlen', 0))}:q={int(record.get('q_len', DEFAULT_Q_LEN))}"


def reconcile_sku_accounting(
    expected_keys: Counter[tuple[str, str, str]],
    rows: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assert exact (acquisition_uid, case_id, boundary) multiset parity and split disjointness."""
    actual_keys: Counter[tuple[str, str, str]] = Counter()
    for bucket in (rows, blocked, diagnostics):
        for item in bucket:
            key = (str(item.get("acquisition_uid", "")), str(item.get("case_id", "")), str(item.get("boundary", "")))
            actual_keys[key] += 1
    if actual_keys != expected_keys:
        missing = list((expected_keys - actual_keys).elements())[:MAX_DIFF_PREVIEW]
        extra = list((actual_keys - expected_keys).elements())[:MAX_DIFF_PREVIEW]
        raise AssertionError(f"multiset_reconciliation_failed:missing={missing}:extra={extra}")

    anchor_geoms: set[str] = set()
    heldout_geoms: set[str] = set()
    for item in (*rows, *blocked):
        gk = str(item.get("geometry_key") or "")
        if not gk:
            continue
        if item.get("split") in {"anchor", "train"}:
            anchor_geoms.add(gk)
        elif item.get("split") == "heldout":
            heldout_geoms.add(gk)
    overlap = sorted(anchor_geoms & heldout_geoms)
    if overlap:
        raise AssertionError(f"anchor_heldout_geometry_overlap:{overlap[:MAX_DIFF_PREVIEW]}")

    return {
        "expected_records": sum(expected_keys.values()),
        "reconciled_records": sum(actual_keys.values()),
        "scored_cases": len(rows),
        "blocked_cases": len(blocked),
        "worker_diagnostics": len(diagnostics),
        "anchor_heldout_geometry_overlap": 0,
    }


def _resolve_contained_path(base_dir: Path, rel_path: str) -> Path:
    """Resolve a shard/observations path strictly inside ``base_dir`` (no traversal)."""
    rel = Path(rel_path)
    if not str(rel_path).strip() or rel.is_absolute():
        raise CampaignIntegrityError(f"shard_path_escapes_campaign_dir:{rel_path}")
    root = base_dir.resolve()
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise CampaignIntegrityError(f"shard_path_escapes_campaign_dir:{rel_path}")
    return target


def _stream_verify_shard(
    path: Path,
    combined_hasher: Any | None = None,
) -> tuple[str, int, dict[tuple[str, str, str], list[float]]]:
    """Stream-hash one JSONL shard line-by-line and index ``(case_id, digest, acq_uid)`` timings."""
    file_hasher = hashlib.sha256()
    line_count = 0
    case_durations: dict[tuple[str, str, str], list[float]] = {}
    with path.open("rb") as stream:
        for raw_line in stream:
            file_hasher.update(raw_line)
            if combined_hasher is not None:
                combined_hasher.update(raw_line)
            if not raw_line.strip():
                continue
            line_count += 1
            try:
                rec = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CampaignIntegrityError(f"shard_jsonl_invalid:{path.name}:{line_count}") from exc
            if not isinstance(rec, Mapping):
                raise CampaignIntegrityError(f"shard_jsonl_object_required:{path.name}:{line_count}")
            cid = str(rec.get("case_id") or "")
            dg = str(rec.get("canonical_key_digest") or "")
            acq = str(rec.get("acquisition_uid") or "")
            if cid and dg and acq:
                durs = case_durations.setdefault((cid, dg, acq), [])
                dur_ms = rec.get("duration_ms")
                if isinstance(dur_ms, (int, float)) and not isinstance(dur_ms, bool):
                    val_ms = float(dur_ms)
                    if math.isfinite(val_ms) and val_ms > MIN_POSITIVE:
                        durs.append(val_ms * US_PER_MS)
    return file_hasher.hexdigest(), line_count, case_durations


def _check_shard_case_bindings(
    manifest: CampaignManifest,
    shard_durations: Mapping[tuple[str, str, str], list[float]],
    present_run_ids: set[str],
    present_shard_paths: set[str],
) -> None:
    """Verify every measured ``ok`` case exists in its shard with consistent ``p50_us``."""
    for case in manifest.cases:
        if case.status != "ok":
            continue
        is_external_hit = (
            case.cache_status == "hit"
            and case.measured_in_run_id not in present_run_ids
            and case.run_shard not in present_shard_paths
        )
        if is_external_hit:
            continue
        durs = shard_durations.get((case.case_id, case.canonical_key_digest, case.acquisition_uid))
        if not durs:
            raise CampaignIntegrityError(f"case_shard_record_missing:{case.case_id}:{case.acquisition_uid}")
        lo, hi = min(durs), max(durs)
        if case.p50_us < lo - SHARD_DURATION_TOL_US or case.p50_us > hi + SHARD_DURATION_TOL_US:
            raise CampaignIntegrityError(
                f"case_shard_p50_mismatch:{case.case_id}:p50={case.p50_us}:shard_range=[{lo},{hi}]"
            )


def _verify_campaign_shards(manifest: CampaignManifest, base_dir: Path) -> None:
    """Verify path containment, streaming SHA-256 seals, record counts, and case bindings."""
    shard_durations: dict[tuple[str, str, str], list[float]] = {}
    present_run_ids: set[str] = set()
    present_shard_paths: set[str] = set()
    if manifest.run_shards:
        combined = hashlib.sha256()
        for shard in manifest.run_shards:
            shard_path = _resolve_contained_path(base_dir, shard.path)
            if not shard_path.is_file():
                raise CampaignIntegrityError(f"missing_run_shard:{shard.path}")
            actual_sha, line_count, rec_map = _stream_verify_shard(shard_path, combined)
            if actual_sha != shard.observations_sha256:
                raise CampaignIntegrityError(
                    f"run_shard_sha256_mismatch:{shard.path}:"
                    f"{shard.observations_sha256[:DIGEST_PREFIX_LEN]}!={actual_sha[:DIGEST_PREFIX_LEN]}"
                )
            if line_count != shard.record_count:
                raise CampaignIntegrityError(
                    f"run_shard_record_count_mismatch:{shard.path}:declared={shard.record_count}:actual={line_count}"
                )
            present_run_ids.add(shard.run_id)
            present_shard_paths.add(shard.path)
            for key, vals in rec_map.items():
                shard_durations.setdefault(key, []).extend(vals)
        expected_obs_sha = (
            manifest.run_shards[0].observations_sha256
            if len(manifest.run_shards) == 1 else combined.hexdigest()
        )
        if manifest.observations_sha256 != expected_obs_sha:
            raise CampaignIntegrityError(
                f"observations_sha256_mismatch:{manifest.observations_sha256[:DIGEST_PREFIX_LEN]}!={expected_obs_sha[:DIGEST_PREFIX_LEN]}"
            )
        _check_shard_case_bindings(manifest, shard_durations, present_run_ids, present_shard_paths)
        return

    obs_path = _resolve_contained_path(base_dir, manifest.observations_jsonl)
    if not obs_path.is_file():
        raise CampaignIntegrityError(f"missing_observations_jsonl:{manifest.observations_jsonl}")
    actual_obs_sha, _, rec_map = _stream_verify_shard(obs_path)
    if actual_obs_sha != manifest.observations_sha256:
        raise CampaignIntegrityError(
            f"observations_sha256_mismatch:{manifest.observations_sha256[:DIGEST_PREFIX_LEN]}!={actual_obs_sha[:DIGEST_PREFIX_LEN]}"
        )
    present_shard_paths.add(manifest.observations_jsonl)
    _check_shard_case_bindings(manifest, rec_map, present_run_ids, present_shard_paths)


def _load_bounded_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing_json_file:{path}")
    if path.stat().st_size > MAX_CAMPAIGN_JSON_BYTES:
        raise CampaignIntegrityError(f"json_exceeds_50_mib:{path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CampaignIntegrityError(f"json_object_required:{path.name}")
    return data


def nominal_spec_for_sku(sku: str) -> HardwareNominalSpec:
    """Load strictly un-derated public datasheet ``HardwareNominalSpec`` from ``ascend.json``."""
    profs = ascend_profiles()
    profile = profs.get(sku)
    if profile is None:
        raise CampaignIntegrityError(f"unknown_sku_for_nominal_spec:{sku}")
    return HardwareNominalSpec.from_ascend_profile(sku, profile)


def verify_clock_in_band(
    sku: str,
    observed_clock_mhz: float,
    *,
    nominal_spec: HardwareNominalSpec | None = None,
    min_ratio: float = MIN_QUALIFIED_CLOCK_RATIO,
    max_ratio: float = MAX_QUALIFIED_CLOCK_RATIO,
) -> float:
    """Reject thermally throttled or out-of-band AICore frequencies from qualified hits."""
    spec = nominal_spec or nominal_spec_for_sku(sku)
    if not math.isfinite(observed_clock_mhz) or observed_clock_mhz <= 0.0:
        raise CampaignIntegrityError(f"invalid_observed_clock_mhz:{observed_clock_mhz!r}")
    ratio = observed_clock_mhz / spec.nominal_clock_mhz
    if ratio < min_ratio:
        raise CampaignIntegrityError(
            f"clock_thermally_throttled:{sku}:observed={observed_clock_mhz}MHz:nominal={spec.nominal_clock_mhz}MHz"
        )
    if ratio > max_ratio:
        raise CampaignIntegrityError(
            f"clock_out_of_band:{sku}:observed={observed_clock_mhz}MHz:nominal={spec.nominal_clock_mhz}MHz"
        )
    return ratio


def verify_campaign_integrity(
    raw_campaign: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
    verify_case_raw_sha256: bool = True,
) -> CampaignManifest:
    """Verify anti-spoofing, public provenance, campaign/shard/raw SHA-256 seals."""
    try:
        assert_no_forbidden_prediction_keys(raw_campaign, "$")
    except ModelingError as exc:
        raise CampaignIntegrityError(str(exc)) from exc
    try:
        assert_public_provenance(raw_campaign, "$")
    except PacketError as exc:
        raise CampaignIntegrityError(str(exc)) from exc
    if raw_campaign.get("schema") != BENCH_CAMPAIGN_V1_SCHEMA:
        raise CampaignIntegrityError(f"campaign_schema_invalid:{raw_campaign.get('schema')!r}")
    recorded_sha = str(raw_campaign.get("campaign_sha256") or "")
    expected_sha = compute_campaign_sha256(raw_campaign)
    if not recorded_sha or recorded_sha != expected_sha:
        raise CampaignIntegrityError(
            f"campaign_sha256_mismatch:{recorded_sha[:DIGEST_PREFIX_LEN]}!={expected_sha[:DIGEST_PREFIX_LEN]}"
        )
    try:
        manifest = CampaignManifest.from_mapping(raw_campaign, verify_seal=True)
    except ModelingError as exc:
        raise CampaignIntegrityError(str(exc)) from exc

    if verify_case_raw_sha256:
        raw_cases = raw_campaign.get("cases", ())
        for raw_case in raw_cases:
            if not isinstance(raw_case, Mapping):
                raise CampaignIntegrityError("campaign_case_mapping_required")
            stored_raw_sha = str(raw_case.get("raw_sha256") or "")
            calc_raw_sha = compute_case_raw_sha256(raw_case)
            if stored_raw_sha != calc_raw_sha:
                raise CampaignIntegrityError(
                    f"case_raw_sha256_mismatch:{raw_case.get('case_id')}:"
                    f"{stored_raw_sha[:DIGEST_PREFIX_LEN]}!={calc_raw_sha[:DIGEST_PREFIX_LEN]}"
                )

    if base_dir is not None:
        _verify_campaign_shards(manifest, base_dir)
    return manifest


def _collect_training_manifest_sets(
    training_manifest: Mapping[str, Any],
    train_digests: set[str],
    train_geoms: set[str],
) -> None:
    for item in training_manifest.get("canonical_key_digests", ()):
        if item:
            train_digests.add(str(item))
    for item in training_manifest.get("geometry_keys", ()):
        if item:
            train_geoms.add(str(item))
    for item in training_manifest.get("cases", ()):
        if not isinstance(item, Mapping):
            continue
        sp = item.get("split")
        if sp is not None and str(sp) not in {"train", "anchor"}:
            continue
        dg = str(item.get("canonical_key_digest") or "")
        gk = str(item.get("geometry_key") or "")
        if dg:
            train_digests.add(dg)
        if gk:
            train_geoms.add(gk)


def verify_training_manifest_independence(
    manifest: CampaignManifest,
    training_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Enforce intra-campaign and cross-plan ``training_manifest_sha256`` heldout disjointness."""
    train_digests: set[str] = set()
    train_geoms: set[str] = set()
    heldout_digests: set[str] = set()
    heldout_geoms: set[str] = set()
    for case in manifest.cases:
        if case.split == "train":
            train_digests.add(case.canonical_key_digest)
            if case.geometry_key:
                train_geoms.add(case.geometry_key)
        elif case.split == "heldout":
            heldout_digests.add(case.canonical_key_digest)
            if case.geometry_key:
                heldout_geoms.add(case.geometry_key)
    for blk in manifest.blocked_cases:
        sp = str(blk.get("split") or "")
        dg = str(blk.get("canonical_key_digest") or "")
        gk = str(blk.get("geometry_key") or "")
        if sp == "train":
            if dg:
                train_digests.add(dg)
            if gk:
                train_geoms.add(gk)
        elif sp == "heldout":
            if dg:
                heldout_digests.add(dg)
            if gk:
                heldout_geoms.add(gk)

    if manifest.training_manifest_sha256 is not None and training_manifest is None and (heldout_digests or heldout_geoms):
        raise CampaignIntegrityError("training_manifest_required_for_heldout_verification")

    if training_manifest is not None:
        tm_sha = canonical_sha256(training_manifest)
        if manifest.training_manifest_sha256 and manifest.training_manifest_sha256 != tm_sha:
            raise CampaignIntegrityError(
                f"training_manifest_sha256_mismatch:{manifest.training_manifest_sha256[:DIGEST_PREFIX_LEN]}!={tm_sha[:DIGEST_PREFIX_LEN]}"
            )
        _collect_training_manifest_sets(training_manifest, train_digests, train_geoms)

    digest_overlap = sorted(train_digests & heldout_digests)
    if digest_overlap:
        raise CampaignIntegrityError(f"train_heldout_digest_overlap:{digest_overlap[:MAX_DIFF_PREVIEW]}")
    geom_overlap = sorted(train_geoms & heldout_geoms)
    if geom_overlap:
        raise CampaignIntegrityError(f"train_heldout_geometry_overlap:{geom_overlap[:MAX_DIFF_PREVIEW]}")


def verify_plan_and_occurrence_conservation(
    manifest: CampaignManifest,
    plan: BenchPlanV2,
    *,
    matrix: OperatorMatrixManifest | None = None,
) -> dict[str, int]:
    """Verify ``plan_sha256``, exact planned case multiset (with split), and ``OpOccurrence`` conservation."""
    plan.verify_seal()
    if manifest.plan_sha256 != plan.plan_sha256:
        raise CampaignIntegrityError(
            f"plan_sha256_mismatch:campaign={manifest.plan_sha256[:DIGEST_PREFIX_LEN]}:plan={plan.plan_sha256[:DIGEST_PREFIX_LEN]}"
        )
    expected_by_cbs: Counter[tuple[str, str, str]] = Counter()
    expected_full: Counter[tuple[str, str, str, str]] = Counter()
    for pcase in plan.cases:
        expected_by_cbs[(pcase.case_id, pcase.boundary, pcase.split)] += 1
        expected_full[(pcase.acquisition_uid, pcase.case_id, pcase.boundary, pcase.split)] += 1

    actual_by_cbs: Counter[tuple[str, str, str]] = Counter()
    actual_non_hit_full: Counter[tuple[str, str, str, str]] = Counter()
    has_cache_hits = False
    for rec in manifest.cases:
        actual_by_cbs[(rec.case_id, rec.boundary, rec.split)] += 1
        if rec.cache_status == "hit":
            has_cache_hits = True
        else:
            actual_non_hit_full[(rec.acquisition_uid, rec.case_id, rec.boundary, rec.split)] += 1
    for blk in manifest.blocked_cases:
        cid = str(blk.get("case_id", ""))
        bnd = str(blk.get("boundary", ""))
        sp = str(blk.get("split", ""))
        acq = str(blk.get("acquisition_uid", ""))
        blk_status = str(blk.get("status", "")).strip().lower()
        if sp == "abi_rejection" and (blk.get("abi_rejected") is False or blk_status == "failed"):
            raise CampaignIntegrityError(f"abi_rejection_unexpected_pass:{cid}")
        actual_by_cbs[(cid, bnd, sp)] += 1
        if str(blk.get("cache_status", "")) == "hit":
            has_cache_hits = True
        else:
            actual_non_hit_full[(acq, cid, bnd, sp)] += 1
    for diag in manifest.worker_diagnostics:
        cid = str(diag.get("case_id", ""))
        bnd = str(diag.get("boundary", ""))
        sp = str(diag.get("split", ""))
        acq = str(diag.get("acquisition_uid", ""))
        actual_by_cbs[(cid, bnd, sp)] += 1
        actual_non_hit_full[(acq, cid, bnd, sp)] += 1

    if actual_by_cbs != expected_by_cbs:
        missing = list((expected_by_cbs - actual_by_cbs).elements())[:MAX_DIFF_PREVIEW]
        extra = list((actual_by_cbs - expected_by_cbs).elements())[:MAX_DIFF_PREVIEW]
        raise CampaignIntegrityError(f"planned_case_multiset_mismatch:missing={missing}:extra={extra}")
    if not has_cache_hits and actual_non_hit_full != expected_full:
        missing = list((expected_full - actual_non_hit_full).elements())[:MAX_DIFF_PREVIEW]
        extra = list((actual_non_hit_full - expected_full).elements())[:MAX_DIFF_PREVIEW]
        raise CampaignIntegrityError(f"planned_case_multiset_mismatch:missing={missing}:extra={extra}")
    if has_cache_hits:
        extra_non_hit = list((actual_non_hit_full - expected_full).elements())[:MAX_DIFF_PREVIEW]
        if extra_non_hit:
            raise CampaignIntegrityError(f"planned_case_multiset_mismatch:extra_measured={extra_non_hit}")

    occ_checked = 0
    if matrix is not None:
        if matrix.plan_sha256 is not None and matrix.plan_sha256 != plan.plan_sha256:
            raise CampaignIntegrityError("matrix_plan_sha256_mismatch")
        plan_occ: Counter[tuple[str, str, str, str, int, int, int | None, str]] = Counter()
        for cid, entries in plan.reverse_index.items():
            for entry in entries:
                plan_occ[(str(cid), *entry.occurrence_key())] += entry.occurrence_count
        matrix_occ: Counter[tuple[str, str, str, str, int, int, int | None, str]] = Counter()
        for row in matrix.rows:
            matrix_occ[(row.bench_case_id, *row.occurrence_key())] += row.occurrence_count
        if plan_occ != matrix_occ:
            missing_occ = list((plan_occ - matrix_occ).elements())[:MAX_DIFF_PREVIEW]
            extra_occ = list((matrix_occ - plan_occ).elements())[:MAX_DIFF_PREVIEW]
            raise CampaignIntegrityError(
                f"occurrence_multiset_mismatch:missing={missing_occ}:extra={extra_occ}"
            )
        occ_checked = sum(plan_occ.values())

    return {
        "planned_cases": sum(expected_by_cbs.values()),
        "reconciled_cases": sum(actual_by_cbs.values()),
        "verified_occurrences": occ_checked,
    }


def replay_step_dag_us(
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
    durations_us: Mapping[str, float],
) -> float:
    """Compute exact critical-path step span (us) by replaying the execution DAG."""
    unique_nodes = list(dict.fromkeys(str(n) for n in node_ids))
    if not unique_nodes:
        return 0.0
    preds: dict[str, list[str]] = {n: [] for n in unique_nodes}
    succs: dict[str, list[str]] = {n: [] for n in unique_nodes}
    indeg: dict[str, int] = {n: 0 for n in unique_nodes}
    for src, dst in edges:
        s, d = str(src), str(dst)
        if s not in preds or d not in preds:
            raise CampaignIntegrityError(f"dag_edge_unknown_node:{s}->{d}")
        preds[d].append(s)
        succs[s].append(d)
        indeg[d] += 1
    queue = deque(n for n in unique_nodes if indeg[n] == 0)
    finish: dict[str, float] = {}
    visited = 0
    while queue:
        cur = queue.popleft()
        visited += 1
        if cur not in durations_us:
            raise CampaignIntegrityError(f"dag_missing_node_duration:{cur}")
        dur = float(durations_us[cur])
        if not math.isfinite(dur) or dur < 0.0:
            raise CampaignIntegrityError(f"dag_negative_or_nonfinite_duration:{cur}:{dur!r}")
        start = max((finish[p] for p in preds[cur]), default=0.0)
        finish[cur] = start + dur
        for nxt in succs[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if visited != len(unique_nodes):
        raise CampaignIntegrityError("dag_contains_cycle")
    return max(finish.values(), default=0.0)


@dataclass(frozen=True, slots=True)
class CasePredictionReference:
    """Independent ``PredictionSpine`` + ``L*`` binding for one campaign case."""

    case_id: str
    pred_us: float
    l_star_us: float
    timed_nodes: tuple[str, ...]
    participating_ranks: tuple[int, ...]
    clock_operator: str
    normalized_clock_mhz: float


def _check_case_clock_and_lstar_binding(
    case: CampaignCaseRecord,
    pred_ref: CasePredictionReference,
    sku: str,
    nominal_spec: HardwareNominalSpec,
) -> None:
    verify_clock_in_band(sku, case.observed_clock_mhz, nominal_spec=nominal_spec)
    if case.clock_operator == "busy_union" and not case.traced_interval_verified:
        raise CampaignIntegrityError(
            f"busy_union_requires_traced_intervals:{case.case_id}"
        )
    if tuple(pred_ref.timed_nodes) != tuple(case.timed_nodes):
        raise CampaignIntegrityError(
            f"l_star_binding_timed_nodes_mismatch:{case.case_id}:"
            f"pred={pred_ref.timed_nodes}:obs={case.timed_nodes}"
        )
    if tuple(pred_ref.participating_ranks) != tuple(case.participating_ranks):
        raise CampaignIntegrityError(
            f"l_star_binding_ranks_mismatch:{case.case_id}:"
            f"pred={pred_ref.participating_ranks}:obs={case.participating_ranks}"
        )
    if str(pred_ref.clock_operator) != str(case.clock_operator):
        raise CampaignIntegrityError(
            f"l_star_binding_clock_operator_mismatch:{case.case_id}:"
            f"pred={pred_ref.clock_operator}:obs={case.clock_operator}"
        )
    clk_err = abs(pred_ref.normalized_clock_mhz - case.observed_clock_mhz) / max(case.observed_clock_mhz, P99_ORDER_EPS)
    if clk_err > CLOCK_MATCH_REL_TOL:
        raise CampaignIntegrityError(
            f"l_star_binding_clock_mhz_mismatch:{case.case_id}:"
            f"pred={pred_ref.normalized_clock_mhz}:obs={case.observed_clock_mhz}"
        )


def _row_prediction_reference(
    row: OperatorMatrixRow,
    nominal_specs: Mapping[str, HardwareNominalSpec],
) -> CasePredictionReference:
    spec = nominal_specs.get(row.sku) or nominal_spec_for_sku(row.sku)
    ranks = (
        row.participating_ranks
        if row.participating_ranks
        else ((row.rank,) if row.rank is not None else DEFAULT_PARTICIPATING_RANKS)
    )
    nodes = row.timed_nodes if row.timed_nodes else DEFAULT_TIMED_NODES
    clk_mhz = row.normalized_clock_mhz if row.normalized_clock_mhz is not None else spec.nominal_clock_mhz
    return CasePredictionReference(
        case_id=row.bench_case_id,
        pred_us=row.per_call_stage0_us,
        l_star_us=row.l_star_us,
        timed_nodes=nodes,
        participating_ranks=ranks,
        clock_operator=row.clock_operator,
        normalized_clock_mhz=clk_mhz,
    )


def _predictions_from_matrix(
    manifest: CampaignManifest,
    matrix: OperatorMatrixManifest,
) -> dict[str, CasePredictionReference]:
    """Build non-circular ``CasePredictionReference`` bindings purely from matrix rows."""
    del manifest
    refs: dict[str, CasePredictionReference] = {}
    for row in matrix.rows:
        candidate = _row_prediction_reference(row, matrix.nominal_specs)
        prev = refs.get(row.bench_case_id)
        if prev is None:
            refs[row.bench_case_id] = candidate
            continue
        same = (
            abs(prev.pred_us - candidate.pred_us) <= P99_ORDER_EPS
            and abs(prev.l_star_us - candidate.l_star_us) <= P99_ORDER_EPS
            and prev.timed_nodes == candidate.timed_nodes
            and prev.participating_ranks == candidate.participating_ranks
            and prev.clock_operator == candidate.clock_operator
            and abs(prev.normalized_clock_mhz - candidate.normalized_clock_mhz) <= P99_ORDER_EPS
        )
        if not same:
            raise CampaignIntegrityError(f"matrix_case_prediction_conflict:{row.bench_case_id}")
    return refs


def _build_prediction_map(
    manifest: CampaignManifest,
    matrix: OperatorMatrixManifest | None,
    predictions: Mapping[str, CasePredictionReference | Mapping[str, Any]] | None,
) -> dict[str, CasePredictionReference]:
    pred_map: dict[str, CasePredictionReference] = {}
    if matrix is not None:
        pred_map.update(_predictions_from_matrix(manifest, matrix))
    if predictions is not None:
        for cid, raw_ref in predictions.items():
            if isinstance(raw_ref, CasePredictionReference):
                pred_map[str(cid)] = raw_ref
            else:
                pred_map[str(cid)] = CasePredictionReference(
                    case_id=str(cid),
                    pred_us=float(raw_ref["pred_us"]),
                    l_star_us=float(raw_ref["l_star_us"]),
                    timed_nodes=tuple(raw_ref["timed_nodes"]),
                    participating_ranks=tuple(int(r) for r in raw_ref["participating_ranks"]),
                    clock_operator=str(raw_ref["clock_operator"]),
                    normalized_clock_mhz=float(raw_ref["normalized_clock_mhz"]),
                )
    return pred_map


def _score_campaign_cases(
    manifest: CampaignManifest,
    sku: str,
    spec: HardwareNominalSpec,
    pred_map: Mapping[str, CasePredictionReference],
    predictor: Callable[[CampaignCaseRecord, float], CasePredictionReference] | None,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str], list[str], list[str], int]:
    verdicts: list[dict[str, Any]] = []
    pointwise_failures: list[str] = []
    l_star_inversions: list[str] = []
    unscored_cases: list[str] = []
    disqualified_cases: list[str] = []
    failed_cases: list[str] = []
    ok_cases_count = 0

    for case in manifest.cases:
        if case.split == "abi_rejection" and case.status == "ok":
            raise CampaignIntegrityError(f"abi_rejection_unexpected_pass:{case.case_id}")
        if case.clock_operator == "busy_union" and not case.traced_interval_verified:
            raise CampaignIntegrityError(f"busy_union_requires_traced_intervals:{case.case_id}")
        if case.status == "failed":
            failed_cases.append(case.case_id)
            continue
        if case.status != "ok":
            continue
        ok_cases_count += 1
        verify_clock_in_band(sku, case.observed_clock_mhz, nominal_spec=spec)
        if not case.qualified:
            disqualified_cases.append(case.case_id)
            continue
        ref = pred_map.get(case.case_id)
        if ref is None and predictor is not None:
            ref = predictor(case, case.observed_clock_mhz)
        if ref is None:
            unscored_cases.append(case.case_id)
            continue
        _check_case_clock_and_lstar_binding(case, ref, sku, spec)
        verdict = score_verdict(
            case.case_id,
            case.family or case.bench_builder or "operator",
            case.split,
            case.p50_us,
            case.p99_us,
            ref.pred_us,
            ref.l_star_us,
            case.capture_uri,
            acquisition_uid=case.acquisition_uid,
            boundary=case.boundary,
            geometry_key=case.geometry_key or case.canonical_key_digest,
        )
        verdict["canonical_key_digest"] = case.canonical_key_digest
        verdict["observed_clock_mhz"] = case.observed_clock_mhz
        verdict["clock_operator"] = case.clock_operator
        verdicts.append(verdict)
        if verdict["inv_sim"] or verdict["inv_obs"]:
            l_star_inversions.append(case.case_id)
        if verdict["status"] != "pass":
            pointwise_failures.append(case.case_id)

    return (
        verdicts,
        pointwise_failures,
        l_star_inversions,
        unscored_cases,
        disqualified_cases,
        failed_cases,
        ok_cases_count,
    )


def reconcile_campaign(
    campaign: CampaignManifest | Mapping[str, Any] | str | Path,
    *,
    plan: BenchPlanV2 | Mapping[str, Any] | str | Path | None = None,
    matrix: OperatorMatrixManifest | None = None,
    predictions: Mapping[str, CasePredictionReference | Mapping[str, Any]] | None = None,
    predictor: Callable[[CampaignCaseRecord, float], CasePredictionReference] | None = None,
    training_manifest: Mapping[str, Any] | None = None,
    base_dir: Path | None = None,
    nominal_spec: HardwareNominalSpec | None = None,
    strict: bool = True,
    verify_case_raw_sha256: bool = True,
    integrity_only: bool = False,
) -> dict[str, Any]:
    """Fail-closed campaign reconciliation across integrity, conservation, clock, and ``L*``."""
    resolved_dir = base_dir
    if isinstance(campaign, (str, Path)):
        camp_path = Path(campaign)
        resolved_dir = resolved_dir or camp_path.parent
        raw_campaign: Mapping[str, Any] = _load_bounded_json(camp_path)
    elif isinstance(campaign, CampaignManifest):
        raw_campaign = campaign.to_dict(include_seal=True)
    else:
        raw_campaign = campaign

    manifest = verify_campaign_integrity(
        raw_campaign,
        base_dir=resolved_dir,
        verify_case_raw_sha256=verify_case_raw_sha256,
    )
    verify_training_manifest_independence(manifest, training_manifest=training_manifest)

    parsed_plan: BenchPlanV2 | None = None
    if plan is not None:
        if isinstance(plan, (str, Path)):
            raw_plan = _load_bounded_json(Path(plan))
            parsed_plan = BenchPlanV2.from_mapping(raw_plan, verify_seal=True)
        elif isinstance(plan, BenchPlanV2):
            parsed_plan = plan
        else:
            parsed_plan = BenchPlanV2.from_mapping(plan, verify_seal=True)

    conservation: dict[str, Any] = (
        {"plan_verified": True, **verify_plan_and_occurrence_conservation(manifest, parsed_plan, matrix=matrix)}
        if parsed_plan is not None else {
            "plan_verified": False,
            "planned_cases": None,
            "reconciled_cases": len(manifest.cases) + len(manifest.blocked_cases) + len(manifest.worker_diagnostics),
            "verified_occurrences": 0,
        }
    )

    sku = manifest.version_fingerprint.sku
    spec = nominal_spec or nominal_spec_for_sku(sku)
    pred_map = _build_prediction_map(manifest, matrix, predictions)
    (
        verdicts,
        pointwise_failures,
        l_star_inversions,
        unscored_cases,
        disqualified_cases,
        failed_cases,
        ok_cases_count,
    ) = _score_campaign_cases(manifest, sku, spec, pred_map, predictor)

    apes = [float(v["ape_pct"]) for v in verdicts if v["ape_pct"] is not None]
    median_ape = sorted(apes)[len(apes) // 2] if apes else 0.0
    p90_ape = nearest_rank_p90(apes) if apes else 0.0
    coverage_ok = True if integrity_only else (not unscored_cases and (len(verdicts) > 0 if ok_cases_count > 0 else True))
    passed = (
        not pointwise_failures
        and not l_star_inversions
        and not disqualified_cases
        and not failed_cases
        and not manifest.worker_diagnostics
        and coverage_ok
    )

    report = {
        "schema": "mindie-sim.campaign-reconciliation/1",
        "campaign_id": manifest.campaign_id,
        "env_fingerprint_id": manifest.env_fingerprint_id,
        "plan_id": manifest.plan_id,
        "plan_sha256": manifest.plan_sha256,
        "sku": sku,
        "cann_version": manifest.version_fingerprint.cann_version,
        "va_commit": manifest.version_fingerprint.va_commit,
        "integrity_only": integrity_only,
        "passed": passed,
        "conservation": conservation,
        "scored_count": len(verdicts),
        "unscored_cases": unscored_cases,
        "disqualified_cases": disqualified_cases,
        "failed_cases": failed_cases,
        "worker_diagnostics_count": len(manifest.worker_diagnostics),
        "pointwise_failures": pointwise_failures,
        "l_star_inversions": l_star_inversions,
        "median_ape_pct": round(median_ape, ROUND_APE_DIGITS),
        "p90_ape_pct": round(p90_ape, ROUND_APE_DIGITS),
        "verdicts": verdicts,
    }
    if strict and not passed:
        raise CampaignIntegrityError(
            f"campaign_reconciliation_failed:pointwise_failures={pointwise_failures[:MAX_DIFF_PREVIEW]}:"
            f"l_star_inversions={l_star_inversions[:MAX_DIFF_PREVIEW]}:"
            f"disqualified={disqualified_cases[:MAX_DIFF_PREVIEW]}:"
            f"failed={failed_cases[:MAX_DIFF_PREVIEW]}:"
            f"worker_diagnostics={len(manifest.worker_diagnostics)}:"
            f"unscored={unscored_cases[:MAX_DIFF_PREVIEW]}"
        )
    return report


def _register_row_dag_durations(
    durations_by_cell: dict[str, dict[str, float]],
    row: OperatorMatrixRow,
    duration_us: float,
) -> None:
    cell_map = durations_by_cell[row.cell_id]
    cell_map[row.op_id] = duration_us
    if row.occurrence_count > 1:
        for occ_idx in range(1, row.occurrence_count + 1):
            cell_map[f"{row.op_id}#{occ_idx}"] = duration_us


def enrich_operator_matrix(
    matrix: OperatorMatrixManifest,
    campaign: CampaignManifest,
    reconciliation_report: Mapping[str, Any] | None = None,
    *,
    step_dags: Mapping[str, tuple[Sequence[str], Sequence[tuple[str, str]]]] | None = None,
) -> OperatorMatrixManifest:
    """Attach reconciled qualified silicon timings and nominal MFU/MBU onto matrix rows."""
    del reconciliation_report
    by_case: dict[str, CampaignCaseRecord] = {
        c.case_id: c for c in campaign.cases if c.status == "ok" and c.qualified
    }

    new_rows: list[OperatorMatrixRow] = []
    cell_row_durations: dict[str, dict[str, float]] = defaultdict(dict)
    for row in matrix.rows:
        case = by_case.get(row.bench_case_id)
        if case is None:
            new_rows.append(replace(
                row,
                measured_us=None,
                measured_p99_us=None,
                measured_mfu_pct=None,
                measured_mbu_rw_pct=None,
                measured_mbu_read_pct=None,
                measured_link_bu_pct=None,
                ape_pct=None,
                within_tolerance=None,
                l_star_inverted=None,
                env_fingerprint_id=None,
                capture_uri=None,
            ))
            _register_row_dag_durations(cell_row_durations, row, row.per_call_stage0_us)
            continue
        spec = matrix.nominal_specs.get(row.sku) or nominal_spec_for_sku(row.sku)
        m_mfu, m_mbu_rw, m_mbu_rd, m_link = spec.utilization_pcts(
            case.p50_us,
            cube_flops=float(row.cube_flops),
            cube_dtype=row.cube_dtype,
            hbm_read_bytes=float(row.hbm_read_bytes),
            hbm_write_bytes=float(row.hbm_write_bytes),
            link_bytes=float(row.link_bytes),
            link_tier=row.link_tier,
        )
        row_verdict = score_verdict(
            case.case_id,
            row.family or case.family or "operator",
            case.split,
            case.p50_us,
            case.p99_us,
            row.per_call_stage0_us,
            row.l_star_us,
            case.capture_uri,
            acquisition_uid=case.acquisition_uid,
            boundary=case.boundary,
            geometry_key=case.geometry_key or case.canonical_key_digest,
        )
        new_rows.append(replace(
            row,
            measured_us=case.p50_us,
            measured_p99_us=case.p99_us,
            measured_mfu_pct=round(m_mfu, ROUND_APE_DIGITS),
            measured_mbu_rw_pct=round(m_mbu_rw, ROUND_APE_DIGITS),
            measured_mbu_read_pct=round(m_mbu_rd, ROUND_APE_DIGITS),
            measured_link_bu_pct=round(m_link, ROUND_APE_DIGITS),
            ape_pct=float(row_verdict["ape_pct"]),
            within_tolerance=bool(row_verdict["status"] == "pass"),
            l_star_inverted=bool(row_verdict["inv_sim"] or row_verdict["inv_obs"]),
            env_fingerprint_id=campaign.env_fingerprint_id,
            capture_uri=case.capture_uri,
        ))
        _register_row_dag_durations(cell_row_durations, row, case.p50_us)

    new_steps: list[StepSummary] = []
    for step in matrix.step_summaries:
        spec = matrix.nominal_specs.get(step.sku) or nominal_spec_for_sku(step.sku)
        dag_spec = step_dags.get(step.cell_id) if step_dags is not None else None
        if dag_spec is not None:
            dag_nodes, dag_edges = dag_spec
            meas_step = replay_step_dag_us(dag_nodes, dag_edges, cell_row_durations.get(step.cell_id, {}))
            dur_s = meas_step / US_PER_SECOND
            hbm_denom = float(step.num_dies) * dur_s * spec.hbm_peak_bytes_per_s
            cube_denom = float(step.num_dies) * dur_s * spec.cube_peak_for_dtype(step.cube_dtype)
            meas_model_mbu = (step.irreducible_model_bytes / hbm_denom) * PCT_SCALE if hbm_denom > 0.0 else 0.0
            meas_bus_util = (step.total_step_hbm_rw_bytes / hbm_denom) * PCT_SCALE if hbm_denom > 0.0 else 0.0
            meas_mfu = (step.total_step_cube_flops / cube_denom) * PCT_SCALE if (cube_denom > 0.0 and step.total_step_cube_flops > 0) else 0.0
            meas_link = (
                (step.total_step_link_bytes / (float(step.num_dies) * dur_s * spec.link_peak_for_tier(step.link_tier))) * PCT_SCALE
                if (dur_s > 0.0 and step.total_step_link_bytes > 0) else 0.0
            )
            new_steps.append(replace(
                step,
                measured_step_us=meas_step,
                step_delta_us=meas_step - step.sim_step_us,
                measured_model_mbu_pct=round(meas_model_mbu, ROUND_APE_DIGITS),
                measured_hbm_bus_util_pct=round(meas_bus_util, ROUND_APE_DIGITS),
                step_measured_mfu_pct=round(meas_mfu, ROUND_APE_DIGITS),
                step_measured_link_bu_pct=round(meas_link, ROUND_APE_DIGITS),
                dag_replayed=True,
                env_fingerprint_id=campaign.env_fingerprint_id,
            ))
        else:
            new_steps.append(replace(
                step,
                measured_step_us=None,
                step_delta_us=None,
                measured_model_mbu_pct=None,
                measured_hbm_bus_util_pct=None,
                step_measured_mfu_pct=None,
                step_measured_link_bu_pct=None,
                dag_replayed=False,
                env_fingerprint_id=None,
            ))

    return replace(
        matrix,
        rows=tuple(new_rows),
        step_summaries=tuple(new_steps),
        campaign_id=campaign.campaign_id,
        env_fingerprint_id=campaign.env_fingerprint_id,
    )


def _load_campaign_for_diff(
    source: CampaignManifest | Mapping[str, Any] | str | Path,
    explicit_dir: Path | None,
) -> CampaignManifest:
    if isinstance(source, (str, Path)):
        src_path = Path(source)
        return verify_campaign_integrity(_load_bounded_json(src_path), base_dir=explicit_dir or src_path.parent)
    raw = source.to_dict(include_seal=True) if isinstance(source, CampaignManifest) else source
    return verify_campaign_integrity(raw, base_dir=explicit_dir)


def diff_campaigns(
    baseline: CampaignManifest | Mapping[str, Any] | str | Path,
    candidate: CampaignManifest | Mapping[str, Any] | str | Path,
    *,
    baseline_dir: Path | None = None,
    candidate_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare two version-fingerprinted campaigns on identical qualified execution semantics."""
    base = _load_campaign_for_diff(baseline, baseline_dir)
    cand = _load_campaign_for_diff(candidate, candidate_dir)

    base_fp = base.version_fingerprint.to_dict()
    cand_fp = cand.version_fingerprint.to_dict()
    env_diff = {
        k: {"baseline": base_fp.get(k), "candidate": cand_fp.get(k)}
        for k in sorted(set(base_fp) | set(cand_fp))
        if base_fp.get(k) != cand_fp.get(k)
    }

    base_ok = [c for c in base.cases if c.status == "ok" and c.qualified]
    cand_ok = [c for c in cand.cases if c.status == "ok" and c.qualified]
    skipped_unqual_base = sorted(c.case_id for c in base.cases if c.status == "ok" and not c.qualified)
    skipped_unqual_cand = sorted(c.case_id for c in cand.cases if c.status == "ok" and not c.qualified)

    base_by_full = {(c.canonical_key_digest, c.capture_mode, c.residency, c.clock_operator): c for c in base_ok}
    cand_by_full = {(c.canonical_key_digest, c.capture_mode, c.residency, c.clock_operator): c for c in cand_ok}
    base_by_kern = {(c.canonical_key_digest, c.capture_mode, c.residency): c for c in base_ok}
    cand_by_kern = {(c.canonical_key_digest, c.capture_mode, c.residency): c for c in cand_ok}
    base_blocked = {str(b.get("canonical_key_digest", "")) for b in base.blocked_cases if b.get("canonical_key_digest")}
    cand_blocked = {str(b.get("canonical_key_digest", "")) for b in cand.blocked_cases if b.get("canonical_key_digest")}

    incompatible_clock_op: list[dict[str, Any]] = []
    for kkey in sorted(set(base_by_kern) & set(cand_by_kern)):
        bc, cc = base_by_kern[kkey], cand_by_kern[kkey]
        if bc.clock_operator != cc.clock_operator:
            incompatible_clock_op.append({
                "case_id": cc.case_id,
                "canonical_key_digest": cc.canonical_key_digest,
                "baseline_clock_operator": bc.clock_operator,
                "candidate_clock_operator": cc.clock_operator,
            })

    kernel_deltas: list[dict[str, Any]] = []
    clock_mismatch_cases: list[str] = []
    for key in sorted(set(base_by_full) & set(cand_by_full)):
        bc, cc = base_by_full[key], cand_by_full[key]
        clock_ratio = cc.observed_clock_mhz / bc.observed_clock_mhz
        clk_err = abs(cc.observed_clock_mhz - bc.observed_clock_mhz) / max(bc.observed_clock_mhz, P99_ORDER_EPS)
        clock_compatible = clk_err <= CLOCK_MATCH_REL_TOL
        delta_us = cc.p50_us - bc.p50_us
        if clock_compatible:
            delta_pct: float | None = round((delta_us / bc.p50_us) * PCT_SCALE, ROUND_APE_DIGITS)
            speedup: float | None = round(bc.p50_us / cc.p50_us, ROUND_APE_DIGITS)
        else:
            clock_mismatch_cases.append(cc.case_id)
            delta_pct = None
            speedup = None
        kernel_deltas.append({
            "case_id": cc.case_id,
            "canonical_key_digest": cc.canonical_key_digest,
            "capture_mode": cc.capture_mode,
            "residency": cc.residency,
            "clock_operator": cc.clock_operator,
            "upstream_abi": cc.upstream_abi or bc.upstream_abi,
            "baseline_p50_us": bc.p50_us,
            "candidate_p50_us": cc.p50_us,
            "baseline_clock_mhz": bc.observed_clock_mhz,
            "candidate_clock_mhz": cc.observed_clock_mhz,
            "clock_ratio": round(clock_ratio, ROUND_APE_DIGITS),
            "clock_compatible": clock_compatible,
            "delta_us": round(delta_us, ROUND_APE_DIGITS),
            "delta_pct": delta_pct,
            "speedup_ratio": speedup,
        })

    base_ok_digests = {c.canonical_key_digest for c in base_ok}
    cand_ok_digests = {c.canonical_key_digest for c in cand_ok}
    return {
        "schema": "mindie-sim.campaign-diff/1",
        "baseline_campaign_id": base.campaign_id,
        "candidate_campaign_id": cand.campaign_id,
        "baseline_env_fingerprint_id": base.env_fingerprint_id,
        "candidate_env_fingerprint_id": cand.env_fingerprint_id,
        "version_fingerprint_diff": env_diff,
        "matched_kernels": len(kernel_deltas),
        "incompatible_clock_operator_pairs": incompatible_clock_op,
        "clock_mismatch_cases": clock_mismatch_cases,
        "skipped_unqualified_baseline": skipped_unqual_base,
        "skipped_unqualified_candidate": skipped_unqual_cand,
        "newly_supported": sorted(cand_ok_digests & base_blocked),
        "newly_broken": sorted(base_ok_digests & cand_blocked),
        "kernel_deltas": kernel_deltas,
    }


def reconcile_main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "mindie-sim obs reconcile-campaign",
) -> int:
    """CLI entrypoint for ``mindie-sim obs reconcile-campaign``."""
    parser = argparse.ArgumentParser(prog=prog, description="Fail-closed reconciliation of a mindie-bench-campaign/1 manifest.")
    parser.add_argument("--campaign", type=Path, required=True, help="Path to campaign_manifest.json")
    parser.add_argument("--plan", type=Path, default=None, help="Optional mindie-bench-plan/2 JSON")
    parser.add_argument("--predictions", type=Path, default=None, help="Optional predictions JSON keyed by case_id")
    parser.add_argument("--training-manifest", type=Path, default=None, help="Optional frozen training manifest JSON")
    parser.add_argument("--integrity-only", action="store_true", default=False, help="Verify seals, shards, conservation, and clocks without scoring predictions")
    parser.add_argument("--non-strict", dest="strict", action="store_false", default=True)
    parser.add_argument("--out", "-o", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        preds = _load_bounded_json(args.predictions) if args.predictions else None
        t_manifest = _load_bounded_json(args.training_manifest) if args.training_manifest else None
        report = reconcile_campaign(
            args.campaign,
            plan=args.plan,
            predictions=preds,
            training_manifest=t_manifest,
            strict=bool(args.strict),
            integrity_only=bool(args.integrity_only),
        )
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0 if report["passed"] else CLI_RECONCILE_FAIL_EXIT
    except TimeoutError:
        raise
    except (CampaignIntegrityError, ModelingError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return CLI_ERROR_EXIT


def diff_main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "mindie-sim obs diff-campaigns",
) -> int:
    """CLI entrypoint for ``mindie-sim obs diff-campaigns``."""
    parser = argparse.ArgumentParser(prog=prog, description="Cross-version diff between two mindie-bench-campaign/1 manifests.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline campaign_manifest.json")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate campaign_manifest.json")
    parser.add_argument("--out", "-o", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = diff_campaigns(args.baseline, args.candidate)
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0
    except TimeoutError:
        raise
    except (CampaignIntegrityError, ModelingError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return CLI_ERROR_EXIT


__all__ = [
    "ABS_FLOOR_NS",
    "DCP_MAX_ABS_ERR",
    "DCP_MAX_VERIFIED_KV",
    "DIGEST_PREFIX_LEN",
    "FAIL_SENTINEL_APE",
    "L1_CLIFF_ABS_FLOOR_NS",
    "L1_CLIFF_REL_TOL",
    "L1_MEDIAN_APE_MAX_PCT",
    "L1_P90_APE_MAX_PCT",
    "MAX_QUALIFIED_CLOCK_RATIO",
    "MIN_QUALIFIED_CLOCK_RATIO",
    "NS_PER_US",
    "P90_QUANTILE",
    "PCT_SCALE",
    "REL_TOL",
    "ROUND_APE_DIGITS",
    "CampaignIntegrityError",
    "CasePredictionReference",
    "apply_sku_observed_clock",
    "canonical_capture_sha256",
    "dcp_reference_verified",
    "diff_campaigns",
    "diff_main",
    "enrich_operator_matrix",
    "extract_observed_clock_mhz",
    "is_worker_diagnostic_record",
    "load_verified_capture",
    "nearest_rank_p90",
    "nominal_spec_for_sku",
    "reconcile_campaign",
    "reconcile_main",
    "reconcile_sku_accounting",
    "recurrent_executed_geometry",
    "replay_step_dag_us",
    "score_verdict",
    "seq_geometry_key",
    "verify_campaign_integrity",
    "verify_clock_in_band",
    "verify_plan_and_occurrence_conservation",
    "verify_training_manifest_independence",
]
