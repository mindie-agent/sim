"""Verification reconciliation, clock alignment, and digest audit helpers.

[Role] Enforce fail-closed inner-digest verification, capture-recorded board clock
  alignment, strict P50/P99 Lean L* non-inversion checking (with L* > 0), split
  geometry disjointness, and exact (acquisition_uid, case_id, boundary) multiset
  reconciliation for the all-SKU 512..400K fleet verification producer.
[Invariants]
  * Every capture JSON MUST pass canonical raw_sha256 verification before replay.
  * Every scored verdict MUST satisfy finite > 0 duration, finite > 0 Lean L*,
    and L* <= P50 (and P99) with zero inversions.
  * Anchor and Heldout splits MUST be strictly disjoint by actual operator geometry.
  * Every enumerated input (acquisition_uid, case_id, boundary) record maps 1-to-1
    into cases, blocked_cases, or worker_diagnostics with zero silent drops.
[Timeouts & Locks] Pure deterministic CPU helpers; read-only file hashing; no locks.
[Side-Effect Policy] Pure validation and scoring helpers without persistent writes.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from mindie_sim.observations.comparison import apply_capture_observed_clock

REL_TOL = 0.10
ABS_FLOOR_NS = 2000.0
L1_MEDIAN_APE_MAX_PCT = 8.0
L1_P90_APE_MAX_PCT = 12.0
L1_CLIFF_REL_TOL = 0.15
L1_CLIFF_ABS_FLOOR_NS = 3000.0
P90_QUANTILE = 0.9
NS_PER_US = 1000.0
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
ABORT_SUFFIXES: tuple[str, ...] = ("_ABORT", "_TIMEOUT")
ABORT_ERR_PREFIXES: tuple[str, ...] = ("worker_abort_", "timeout_")


def nearest_rank_p90(values: Sequence[float]) -> float:
    """Return canonical nearest-rank P90 (index ceil(0.9 * n) - 1) without pre-rounding."""
    if not values:
        return MIN_POSITIVE
    ordered = sorted(float(v) for v in values)
    idx = max(0, min(len(ordered) - 1, math.ceil(P90_QUANTILE * len(ordered)) - 1))
    return float(ordered[idx])


def canonical_capture_sha256(doc: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 of a capture JSON payload excluding raw_sha256."""
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
    for key in ("cur_freq_mhz", "aicore_freq_mhz", "observed_aicore_freq_mhz"):
        val = entry.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)) and float(val) > MIN_POSITIVE:
            return float(val)
    return None


def extract_observed_clock_mhz(
    doc: Mapping[str, Any], fallback_mhz: float | None = None,
) -> float | None:
    """Extract capture-recorded AICore frequency (MHz), falling back if unavailable."""
    for key in ("clock_start", "clock_end", "clock"):
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
    return apply_capture_observed_clock(case_dict, {"observed_aicore_freq_mhz": float(observed_mhz)})


def score_verdict(
    case_id: str, family: str, split: str, meas_p50: float, meas_p99: float,
    pred_us: float, lstar_us: float, capture_uri: str,
    *, acquisition_uid: str = "", boundary: str = "cold", geometry_key: str = "",
) -> dict[str, Any]:
    """Score one prediction against silicon P50/P99 with strict L* > 0 and P50 non-inversion."""
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
    batch_candidates = (
        record.get("B"),
        record.get("batch"),
        shape.get("batch"),
        shape.get("B"),
        record.get("seqlen"),
    )
    heads_candidates = (
        record.get("H"),
        record.get("heads"),
        shape.get("heads"),
        shape.get("H"),
    )
    q_candidates = (
        record.get("q_len"),
        shape.get("q_len"),
        shape.get("T"),
    )
    batch = next(
        (int(val) for val in batch_candidates
         if val is not None and not isinstance(val, bool) and int(val) > 0),
        DEFAULT_RECURRENT_BATCH,
    )
    heads = next(
        (int(val) for val in heads_candidates
         if val is not None and not isinstance(val, bool) and int(val) > 0),
        default_h,
    )
    q_len = next(
        (int(val) for val in q_candidates
         if val is not None and not isinstance(val, bool) and int(val) > 0),
        DEFAULT_Q_LEN,
    )
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
        if item.get("split") == "anchor":
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
