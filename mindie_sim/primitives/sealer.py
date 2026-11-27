"""Single canonical L0 primitive calibration sealer inside the installed package.

[Role] Own full-envelope SHA-256 sealing of ``mindie-l0-primitives/1`` tables,
  canonical raw capture serialization, opaque public endpoint alias hashing,
  per-metric conflict diagnostics (distinguishing ``missing_samples`` vs
  ``p99_exceeds_nominal`` vs ``cv_exceeds_5pct`` vs ``silicon_profile_conflict``),
  ``SKU x CANN x primitive x mode x metric`` coverage summaries, and manifest
  updates. Both ``mindie_sim.onboarding`` and ``tools/bench_l0_primitives.py``
  delegate here so installed wheels work identically without ``tools/``.
[Invariants] Never inflates ``certified_upper_capacity`` above hardware
  declarations; enforces ``cv_pct <= 5.0%`` and ``p99 <= certified_upper_capacity``
  (for ``rate_upper``) or ``p05 >= certified_upper_capacity`` (for
  ``latency_lower``); never leaks raw IPv4 quads, hostnames, usernames, or
  host ``$HOME`` paths into emitted tables or manifests; zero ``AG003``/``AG004``
  violations.
[Timeouts & Locks] Pure deterministic CPU-only statistics and JSON/SHA-256
  sealing; zero device contexts, subprocesses, or locks.
[Side-Effect Policy] Zero side effects on import; filesystem writes occur only
  in explicit ``update_manifest`` calls.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from mindie_sim.backends import ascend
from mindie_sim.primitives.schema import (
    MAX_CALIBRATED_CV_PCT,
    PRIMITIVE_SCHEMA_VERSION,
    compute_envelope_sha256,
)
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.hardware import Hardware

_ARCH_PRESET_35 = "npu_arch_3510"
_ARCH_PRESET_22 = "npu_arch_2201"
_GIB = 1024 * 1024 * 1024
_GBPS_TO_BPS = 1e9
_PERCENT_SCALE = 100.0
_TRIM_FRACTION = 0.10
_MIN_STDEV_SAMPLES = 2
_ROUND_STAT_DIGITS = 6
_ROUND_CV_DIGITS = 4
_Q05 = 0.05
_Q50 = 0.50
_Q95 = 0.95
_Q99 = 0.99
_OPAQUE_ALIAS_HEX = 12
_DEFAULT_SECTOR_BYTES = 32
_DEFAULT_FRACTAL_BYTES = 512
_DEFAULT_GRANULE_BYTES = 128
_DEFAULT_VEC_WIDTH_BYTES = 256
_DEFAULT_FP32_LANES = 64
_DEFAULT_BF16_LANES = 128
_DEFAULT_PIPE_M_FRACTAL_THRESH = 10
_DEFAULT_SCALE_GROUP = 32
_DEFAULT_HIST_TRUNK_35 = 16384
_DEFAULT_TLA_THREADS_35 = 2048
_DEFAULT_CATLASS_THREADS_35 = 1024
_DEFAULT_WARP_SIZE = 32
_DEFAULT_EVENT_IDS = 8
_DEFAULT_BARRIER_AIV = 8
_DEFAULT_BARRIER_AIC = 9
_DEFAULT_BARRIER_SUB = 10
_DEFAULT_FFTS_MAX_FLAG = 7
_DEFAULT_REVERSE_DEPTH = 15
_DEFAULT_MAX_MUTEX_IDS = 28
_DEFAULT_QUEUE_DEPTH = 64
_DEFAULT_HCCL_PAYLOAD_35 = 480
_DEFAULT_HCCL_WIRE_35 = 512
_DEFAULT_HCCL_FRAME_22 = 1024
_DEFAULT_SORTALL_LENGTHS = (64, 128, 256, 512, 1024, 2048)
_SORT_PROBE_PASSES = 4
_TILE_16 = 16
_TILE_8 = 8
_TILE_32 = 32
_TILE_64 = 64
_DOUBLE_FACTOR = 2.0
_DEFAULT_UNIT_ONE = 1
_MAX_HBM_RW_ALPHA = 1.0

PRIVATE_CAPTURE_RE = re.compile(
    r"(?<![\d.])(?:\d{1,3}[._-]){3}\d{1,3}(?![\d.])|/(?:Users|home|root)/[^\s\"']*|[^\s\"']*@[^\s\"']+|\$\{?HOME\}?"
)
_OPAQUE_ALIAS_RE = re.compile(r"^ep-[0-9a-f]{12}$")
_OPAQUE_CAMPAIGN_RE = re.compile(r"^camp-[0-9a-f]{12}$")

SKU_DEFAULT_FILES = {
    "Ascend910B4": "arch22_a2_910b4.json",
    "Ascend910_9362": "arch22_a3_9362.json",
    "Ascend910_9382": "arch22_a3_9382.json",
    "Ascend950PR_9579": "arch35_a5_950pr.json",
    "Ascend950DT_9582": "arch35_a5_950dt.json",
}


def sanitize_record_alias(alias: str, sku: str) -> str:
    """Return an opaque, deterministic public endpoint digest (zero raw hostnames/IPs/users)."""
    raw = str(alias or "").strip()
    if _OPAQUE_ALIAS_RE.fullmatch(raw):
        return raw
    seed = f"{raw or sku}|{sku}".encode("utf-8")
    return f"ep-{hashlib.sha256(seed).hexdigest()[:_OPAQUE_ALIAS_HEX]}"


def sanitize_record_campaign(campaign_id: str) -> str:
    """Return an opaque, deterministic public campaign digest (zero raw usernames/leases)."""
    raw = str(campaign_id or "").strip()
    if _OPAQUE_CAMPAIGN_RE.fullmatch(raw):
        return raw
    seed = (raw or "campaign").encode("utf-8")
    return f"camp-{hashlib.sha256(seed).hexdigest()[:_OPAQUE_ALIAS_HEX]}"


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] * (1.0 - (pos - lo)) + sorted_vals[hi] * (pos - lo))


def summarize_samples_with_reason(
    samples: Sequence[float],
    *,
    certified_bound: float,
    bound_kind: str = "rate_upper",
    max_admissible: float | None = None,
) -> tuple[dict[str, Any], str]:
    """Compute band statistics plus explicit status/conflict reason."""
    clean = sorted(float(x) for x in samples if math.isfinite(x) and x >= 0.0)
    nom = round(float(certified_bound), _ROUND_STAT_DIGITS)
    if not clean:
        return {"certified_upper_capacity": nom, "bound_kind": bound_kind}, "missing_samples"
    n = len(clean)
    mean_v = sum(clean) / n
    cv_pct = (
        (_PERCENT_SCALE * statistics.stdev(clean) / mean_v)
        if (n >= _MIN_STDEV_SAMPLES and mean_v > 0.0)
        else 0.0
    )
    trim_k = int(math.floor(n * _TRIM_FRACTION))
    trimmed = clean[trim_k : n - trim_k] if n - _MIN_STDEV_SAMPLES * trim_k >= _MIN_STDEV_SAMPLES else clean
    t_mean = sum(trimmed) / len(trimmed)
    t_cv = (
        (_PERCENT_SCALE * statistics.stdev(trimmed) / t_mean)
        if (len(trimmed) >= _MIN_STDEV_SAMPLES and t_mean > 0.0)
        else 0.0
    )
    p05, p50, p95, p99 = (_quantile(clean, q) for q in (_Q05, _Q50, _Q95, _Q99))
    if bound_kind == "rate_upper":
        if p05 <= 0.0:
            return {"certified_upper_capacity": nom, "bound_kind": bound_kind}, "p05_nonpositive"
        if p99 > float(certified_bound):
            return (
                {"certified_upper_capacity": nom, "bound_kind": bound_kind},
                f"p99_exceeds_nominal:p99={round(p99, _ROUND_CV_DIGITS)}:nominal={nom}",
            )
    elif p05 < float(certified_bound):
        return (
            {"certified_upper_capacity": nom, "bound_kind": bound_kind},
            f"p05_below_floor:p05={round(p05, _ROUND_CV_DIGITS)}:floor={nom}",
        )
    if max_admissible is not None and p99 > float(max_admissible):
        return (
            {"certified_upper_capacity": nom, "bound_kind": bound_kind},
            f"p99_exceeds_admissible:p99={round(p99, _ROUND_CV_DIGITS)}:max={round(float(max_admissible), _ROUND_STAT_DIGITS)}",
        )
    band = {
        "p05": round(p05, _ROUND_STAT_DIGITS),
        "p50": round(p50, _ROUND_STAT_DIGITS),
        "p95": round(p95, _ROUND_STAT_DIGITS),
        "p99": round(p99, _ROUND_STAT_DIGITS),
        "trimmed_mean": round(t_mean, _ROUND_STAT_DIGITS),
        "trimmed_cv_pct": round(t_cv, _ROUND_CV_DIGITS),
        "cv_pct": round(cv_pct, _ROUND_CV_DIGITS),
        "n_samples": n,
        "certified_upper_capacity": nom,
        "bound_kind": bound_kind,
    }
    if cv_pct > MAX_CALIBRATED_CV_PCT:
        return band, f"cv_exceeds_5pct:cv_pct={round(cv_pct, _ROUND_CV_DIGITS)}"
    return band, "calibrated"


def summarize_samples(
    samples: Sequence[float],
    *,
    certified_bound: float,
    bound_kind: str = "rate_upper",
) -> dict[str, Any]:
    """Compute untrimmed (p05,p50,p95,p99,cv_pct) + 10%-trimmed stats under hardware ceiling."""
    band, _reason = summarize_samples_with_reason(
        samples, certified_bound=certified_bound, bound_kind=bound_kind
    )
    return band


def _service_rate_or_default(die_dict: Mapping[str, Any], mode_key: str, default: float) -> float:
    modes = die_dict.get("service_modes")
    if isinstance(modes, Mapping):
        entry = modes.get(mode_key)
        if isinstance(entry, Mapping) and entry.get("rate") is not None:
            return float(entry["rate"])
        if isinstance(entry, (list, tuple)) and len(entry) >= _MIN_STDEV_SAMPLES:
            return float(entry[1])
    return float(default)


def load_sku_hardware_spec(sku: str) -> dict[str, Any]:
    """Load authoritative hardware bounds for ``sku`` from ``ascend.json`` / ``Hardware``."""
    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": _DEFAULT_UNIT_ONE}))
    die = hw.dies[0]
    dp = datapath_for_die(die)
    die_dict = die.to_dict()
    bufs = dict(die.buffers)
    rates = dict(die.rates)
    tc = {k: dict(v) for k, v in hw.transfer_channels}
    h2d, d2h, d2d = tc.get("h2d", {}), tc.get("d2h", {}), tc.get("d2d_intra_node", {})
    h2d_solo = float(h2d.get("solo_effective_uni_bw_gbps") or h2d.get("uni_peak_bw_gbps") or 0.0) * _GBPS_TO_BPS
    d2h_solo = float(d2h.get("solo_effective_uni_bw_gbps") or d2h.get("uni_peak_bw_gbps") or 0.0) * _GBPS_TO_BPS
    h2d_rc = float(h2d.get("node_peak_bw_gbps") or 0.0) * _GBPS_TO_BPS or h2d_solo
    d2h_rc = float(d2h.get("node_peak_bw_gbps") or 0.0) * _GBPS_TO_BPS or d2h_solo
    link_uni = float(d2d.get("uni_peak_bw_gbps") or d2d.get("effective_uni_bw_gbps") or 0.0) * _GBPS_TO_BPS
    peaks_raw = die_dict.get("compute_peaks") if isinstance(die_dict.get("compute_peaks"), Mapping) else {}
    ma = getattr(dp, "memory_access", None)
    return {
        "sku": sku,
        "arch_preset": str(die.compute_arch.preset or _ARCH_PRESET_22),
        "aic_count": int(dp.aic_count),
        "aiv_count": int(dp.aiv_count),
        "hbm_gib": int(die.memory_bytes // _GIB),
        "l2_bytes": int(bufs.get("l2", 0)),
        "ub_bytes": int(bufs.get("ub", 0)),
        "l1_bytes": int(bufs.get("l1", 0)),
        "l0a_bytes": int(bufs.get("l0a", 0)),
        "l0b_bytes": int(bufs.get("l0b", 0)),
        "l0c_bytes": int(bufs.get("l0c", 0)),
        "sector_bytes": int(ma.sector_bytes) if ma is not None else _DEFAULT_SECTOR_BYTES,
        "hbm_burst_granule_bytes": int(ma.hbm_burst_granule_bytes) if ma is not None else None,
        "hbm_scatter_row_open_bytes": int(ma.hbm_scatter_row_open_bytes) if ma is not None else None,
        "l2_line_bytes": int(ma.l2_line_bytes) if ma is not None else None,
        "l2_sector_bytes": int(ma.l2_sector_bytes) if ma is not None else _DEFAULT_GRANULE_BYTES,
        "has_l0c_to_ub": bool(dp.has_l0c_to_ub),
        "has_ub_to_l1": bool(dp.has_ub_to_l1),
        "simt_ub_carveout_bytes": int(dp.simt_ub_carveout_bytes),
        "hbm_read_bytes_per_s": float(dp.hbm_read_bytes_per_s or rates.get("hbm", 0.0)),
        "hbm_write_bytes_per_s": float(dp.hbm_write_bytes_per_s or rates.get("hbm", 0.0)),
        "l2_bytes_per_s": float(rates.get("l2", rates.get("mte2_aiv", 0.0) * _DOUBLE_FACTOR)),
        "mte1_l0a_bytes_per_s": _service_rate_or_default(die_dict, "mte1_l0a", rates.get("mte1", 0.0)),
        "mte1_l0b_bytes_per_s": _service_rate_or_default(die_dict, "mte1_l0b", rates.get("mte1", 0.0)),
        "mte2_aic_bytes_per_s": _service_rate_or_default(die_dict, "mte2_aic_l2", rates.get("mte2_aic", 0.0)),
        "mte2_aiv_bytes_per_s": _service_rate_or_default(die_dict, "mte2_aiv_l2", rates.get("mte2_aiv", 0.0)),
        "mte3_bytes_per_s": float(rates.get("mte3", 0.0)),
        "atomic_rmw_bytes_per_s": _service_rate_or_default(
            die_dict, "hbm_atomic_rmw", dp.hbm_write_bytes_per_s or rates.get("hbm", 0.0)
        ),
        "fixpipe_bytes_per_s": float(rates.get("fixpipe", 0.0)),
        "vector_ops_per_s": float(rates.get("vector", 0.0)),
        "vec_alu_ops_per_s": float(rates.get("reg_alu_lanes", rates.get("vector", 0.0) * _DOUBLE_FACTOR)),
        "vec_sfu_ops_per_s": float(rates.get("reg_special_lanes", rates.get("vector", 0.0))),
        "vec_reduce_ops_per_s": float(rates.get("reg_reduce_lanes", rates.get("vector", 0.0))),
        "cast_bytes_per_s": float(rates.get("ub_write_bytes", rates.get("mte2_aiv", 0.0))),
        "link_bytes_per_s": link_uni if link_uni > 0.0 else None,
        "host_h2d_bytes_per_s": h2d_solo if h2d_solo > 0.0 else None,
        "host_d2h_bytes_per_s": d2h_solo if d2h_solo > 0.0 else None,
        "host_rc_h2d_bytes_per_s": h2d_rc if h2d_rc > 0.0 else None,
        "host_rc_d2h_bytes_per_s": d2h_rc if d2h_rc > 0.0 else None,
        "peaks": {str(k): float(v) for k, v in peaks_raw.items()},
    }


def _cell_status(metrics: Mapping[str, Mapping[str, Any] | None]) -> str:
    if not metrics:
        return "unmeasured"
    for band in metrics.values():
        if band is None or band.get("p50") is None or band.get("cv_pct") is None:
            return "unmeasured"
        if float(band["cv_pct"]) > MAX_CALIBRATED_CV_PCT:
            return "unmeasured"
    return "calibrated"


def _clean_band(b: Mapping[str, Any]) -> dict[str, Any]:
    if b.get("p50") is not None and b.get("cv_pct") is not None and float(b["cv_pct"]) <= MAX_CALIBRATED_CV_PCT:
        return dict(b)
    return {"certified_upper_capacity": b["certified_upper_capacity"], "bound_kind": b.get("bound_kind", "rate_upper")}


def _sanitize_capture_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return PRIVATE_CAPTURE_RE.sub("redacted", obj)
    if isinstance(obj, Mapping):
        return {PRIVATE_CAPTURE_RE.sub("redacted", str(k)): _sanitize_capture_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_capture_obj(v) for v in obj]
    return obj


def canonical_capture_bytes(sku: str, raw_measurements: Mapping[str, Any] | None) -> bytes:
    """Serialize sanitized raw calibration samples/summary to canonical JSON bytes."""
    payload = _sanitize_capture_obj(raw_measurements) if raw_measurements else {"sku": sku, "status": "unmeasured"}
    body = {"tag": "campaign", "payload": payload}
    return (json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _bound_only(value: float | None, kind: str = "rate_upper") -> dict[str, Any] | None:
    if value is None:
        return None
    return {"certified_upper_capacity": round(float(value), _ROUND_STAT_DIGITS), "bound_kind": kind}


def _build_p06_mmad_modes(
    p06r: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    aic: int,
    is_arch35: bool,
    allow: bool,
    band_fn: Any,
) -> tuple[dict[str, Any], bool]:
    p06_modes: dict[str, Any] = {}
    any_p06 = False
    mode_defs = [
        ("bf16", _TILE_16, _TILE_16, _TILE_16, _TILE_16, spec["peaks"].get("bf16", 0.0) / aic),
        ("fp16", _TILE_16, _TILE_16, _TILE_16, _TILE_16, spec["peaks"].get("fp16", 0.0) / aic),
        ("fp32", _TILE_16, _TILE_16, _TILE_8, _TILE_16, spec["peaks"].get("fp32", 0.0) / aic),
        ("hf32", _TILE_16, _TILE_16, _TILE_8, _TILE_16, spec["peaks"].get("hf32", 0.0) / aic),
        ("int8", _TILE_16, _TILE_16, _TILE_32, _TILE_16, spec["peaks"].get("int8", 0.0) / aic),
    ]
    if is_arch35:
        for mode in ("fp8_e4m3", "fp8_e5m2"):
            cap = spec["peaks"].get(mode, 0.0) / aic
            mode_defs.append((mode, _TILE_16, _TILE_16, _TILE_32, _TILE_16, cap))
        mx_cap = spec["peaks"].get("fp8_e4m3", spec["peaks"].get("int8", 0.0)) / aic
        hif8_cap = spec["peaks"].get("hifloat8", mx_cap * aic) / aic
        mode_defs.extend([
            ("hifloat8", _TILE_16, _TILE_16, _TILE_32, _TILE_16, hif8_cap),
            ("mxfp8", _TILE_16, _TILE_16, _TILE_32, _TILE_16, mx_cap),
            ("mxfp4", _TILE_16, _TILE_16, _TILE_64, _TILE_16, mx_cap * _DOUBLE_FACTOR),
        ])
    raw_cube_modes = p06r.get("cube_only_modes") or p06r.get("modes") or {}
    for m_name, tm, tn, tk, miss, cap in mode_defs:
        if cap <= 0.0:
            continue
        b = band_fn("P06_mmad", "cube_only_rate", raw_cube_modes.get(m_name), cap, allow_meas=allow, mode=m_name)
        ms = _cell_status({"r": b})
        any_p06 = any_p06 or (ms == "calibrated")
        entry: dict[str, Any] = {
            "status": ms, "tile_m": tm, "tile_n": tn, "tile_k": tk, "m_iss": miss,
            "cube_only_rate": b if ms == "calibrated" else _clean_band(b),
        }
        if m_name in {"mxfp8", "mxfp4"}:
            entry["scale_group"] = _DEFAULT_SCALE_GROUP
        p06_modes[m_name] = entry
    return p06_modes, any_p06


def _build_p10_sort_cell(
    p10r: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    aiv: int,
    is_arch35: bool,
    allow: bool,
    conflict_meta: Mapping[str, Any],
    band_fn: Any,
    make_cell_fn: Any,
) -> dict[str, Any]:
    sort_per_aiv_cap = spec["vector_ops_per_s"] / aiv
    valid_algs = ("sort32_mrgsort4", "masked_hist256") if is_arch35 else ("sort32_mrgsort4",)
    raw_alg = str(p10r.get("measured_algorithm") or "").strip()
    target_alg = raw_alg if raw_alg in valid_algs else "sort32_mrgsort4"
    raw_per_aiv = p10r.get("sort_elems_per_s_per_aiv")
    if raw_per_aiv is None and p10r.get("sort_elems_per_s") is not None:
        raw_die = p10r["sort_elems_per_s"]
        raw_per_aiv = [float(v) / aiv for v in raw_die] if isinstance(raw_die, Sequence) and not isinstance(raw_die, (str, bytes)) else raw_die
    passes = (
        _SORT_PROBE_PASSES
        if (target_alg == "sort32_mrgsort4" and p10r.get("work_unit") != "elem_passes_per_s")
        else 1
    )
    if isinstance(raw_per_aiv, Sequence) and not isinstance(raw_per_aiv, (str, bytes)):
        p10_samples: Sequence[float] | None = [float(v) * float(passes) for v in raw_per_aiv]
    else:
        p10_samples = raw_per_aiv
    raw_modes = p10r.get("modes") if isinstance(p10r.get("modes"), Mapping) else {}
    p10_modes: dict[str, Any] = {}
    any_calibrated = False
    for alg in valid_algs:
        alg_samples = raw_modes.get(alg) if alg in raw_modes else (p10_samples if alg == target_alg else None)
        b = band_fn(
            "P10_sort_topk",
            "sort_elems_per_s_per_aiv",
            alg_samples,
            sort_per_aiv_cap,
            allow_meas=allow,
            mode=alg,
        )
        st = _cell_status({"r": b})
        any_calibrated = any_calibrated or (st == "calibrated")
        p10_modes[alg] = {
            "status": st,
            "sort_elems_per_s_per_aiv": b if st == "calibrated" else _clean_band(b),
        }
    primary_band = p10_modes[target_alg]["sort_elems_per_s_per_aiv"]
    return make_cell_fn(
        "P10_sort_topk",
        {"rate": "elem_passes_per_s_per_aiv"},
        {
            "sort_algorithm": "sort32_mrgsort4",
            "masked_hist256_supported": is_arch35,
            "work_unit": "elem_passes_per_s",
            "supported_sortall_lengths": list(_DEFAULT_SORTALL_LENGTHS),
            "histogram_trunk_elems": _DEFAULT_HIST_TRUNK_35 if is_arch35 else 0,
            **conflict_meta,
        },
        {"sort_elems_per_s_per_aiv": primary_band},
        status_override="calibrated" if any_calibrated else "unmeasured",
        modes=p10_modes,
    )


def _assemble_primitive_cells(
    *,
    raw: Mapping[str, Any],
    spec: Mapping[str, Any],
    aic: int,
    aiv: int,
    is_arch35: bool,
    allow: bool,
    conflict_meta: Mapping[str, Any],
    band_fn: Any,
    make_cell_fn: Any,
) -> dict[str, Any]:
    """[Cohesive-Pipeline] Assemble the full hardware primitive cell dictionary across P01-P20 in one pass."""
    p: dict[str, Any] = {}
    p01r, p02r, p03r, p04r, p05r, p06r = (
        raw.get(k, {})
        for k in ("P01_datacopy", "P02_load_data", "P03_fixpipe", "P04_mte3_store", "P05_l2_residency_reuse", "P06_mmad")
    )
    p09r, p12r, p14r, p15r, p20r = (
        raw.get(k, {})
        for k in ("P09_vec_reduce_cast", "P12_intracore_flag", "P14_mutex_block_mmad", "P15_rts_stars", "P20_host_dma_sync")
    )
    p01_params: dict[str, Any] = {
        "sector_bytes": spec.get("sector_bytes", _DEFAULT_SECTOR_BYTES),
        "l2_cache_ctl_supported": is_arch35,
        "burst_turn_penalty": [],
        **conflict_meta,
    }
    if spec.get("hbm_burst_granule_bytes") is not None:
        p01_params["hbm_burst_granule_bytes"] = spec["hbm_burst_granule_bytes"]
    if spec.get("hbm_scatter_row_open_bytes") is not None:
        p01_params["hbm_scatter_row_open_bytes"] = spec["hbm_scatter_row_open_bytes"]
    p["P01_datacopy"] = make_cell_fn(
        "P01_datacopy", {"rate": "bytes_per_s", "penalty": "bytes", "alpha": "dimensionless"},
        p01_params,
        {
            "hbm_read_bytes_per_s": band_fn("P01_datacopy", "hbm_read_bytes_per_s", p01r.get("hbm_read_bytes_per_s"), spec["hbm_read_bytes_per_s"], allow_meas=allow),
            "hbm_write_bytes_per_s": band_fn("P01_datacopy", "hbm_write_bytes_per_s", p01r.get("hbm_write_bytes_per_s"), spec["hbm_write_bytes_per_s"], allow_meas=allow),
            "mte2_aiv_hit_bytes_per_s_per_core": band_fn("P01_datacopy", "mte2_aiv_hit_bytes_per_s_per_core", p01r.get("mte2_aiv_hit_bytes_per_s_per_core"), spec["mte2_aiv_bytes_per_s"] / aiv, allow_meas=allow),
        },
        {
            "hbm_rw_alpha": band_fn("P01_datacopy", "hbm_rw_alpha", p01r.get("hbm_rw_alpha"), 0.0, kind="latency_lower", allow_meas=allow, max_admissible=_MAX_HBM_RW_ALPHA),
            "mte2_aic_hit_bytes_per_s_per_core": band_fn("P01_datacopy", "mte2_aic_hit_bytes_per_s_per_core", p01r.get("mte2_aic_hit_bytes_per_s_per_core"), spec["mte2_aic_bytes_per_s"] / aic, allow_meas=allow),
        },
    )
    l0b_trans_cap = (spec["mte1_l0b_bytes_per_s"] / aic) / (_DOUBLE_FACTOR if is_arch35 else 1.0)
    p["P02_load_data"] = make_cell_fn(
        "P02_load_data", {"rate": "bytes_per_s_per_aic", "latency": "ns"},
        {"fractal_bytes": _DEFAULT_FRACTAL_BYTES, "mx_scale_supported": is_arch35,
         "transpose_factor": max(1.0, float(p02r.get("transpose_factor", 1.0))), **conflict_meta},
        {
            "l1_to_l0a_bytes_per_s_per_aic": band_fn("P02_load_data", "l1_to_l0a_bytes_per_s_per_aic", p02r.get("l1_to_l0a_bytes_per_s_per_aic"), spec["mte1_l0a_bytes_per_s"] / aic, allow_meas=allow),
            "l1_to_l0b_bytes_per_s_per_aic": band_fn("P02_load_data", "l1_to_l0b_bytes_per_s_per_aic", p02r.get("l1_to_l0b_bytes_per_s_per_aic"), spec["mte1_l0b_bytes_per_s"] / aic, allow_meas=allow),
        },
        {
            "l1_to_l0a_transpose_bytes_per_s_per_aic": band_fn("P02_load_data", "l1_to_l0a_transpose_bytes_per_s_per_aic", p02r.get("l1_to_l0a_transpose_bytes_per_s_per_aic"), spec["mte1_l0a_bytes_per_s"] / aic, allow_meas=allow),
            "l1_to_l0b_transpose_bytes_per_s_per_aic": band_fn("P02_load_data", "l1_to_l0b_transpose_bytes_per_s_per_aic", p02r.get("l1_to_l0b_transpose_bytes_per_s_per_aic"), l0b_trans_cap, allow_meas=allow),
            "setup_ns": band_fn("P02_load_data", "setup_ns", p02r.get("setup_ns"), 0.0, kind="latency_lower", allow_meas=allow),
        },
    )
    fixp_out_aic = spec["fixpipe_bytes_per_s"] / aic
    fixp_in_aic = fixp_out_aic * _DOUBLE_FACTOR
    p["P03_fixpipe"] = make_cell_fn(
        "P03_fixpipe", {"rate": "bytes_per_s_per_aic", "latency": "ns"},
        {"has_l0c_to_ub": spec["has_l0c_to_ub"], "has_l0c_to_l1": True,
         "ub_split_n_align_elems": _DEFAULT_SECTOR_BYTES, **conflict_meta},
        {
            "fixpipe_in_bytes_per_s_per_aic": band_fn("P03_fixpipe", "fixpipe_in_bytes_per_s_per_aic", p03r.get("fixpipe_in_bytes_per_s_per_aic"), fixp_in_aic, allow_meas=allow),
            "fixpipe_out_bytes_per_s_per_aic": band_fn("P03_fixpipe", "fixpipe_out_bytes_per_s_per_aic", p03r.get("fixpipe_out_bytes_per_s_per_aic"), fixp_out_aic, allow_meas=allow),
        },
        {
            "setup_ns": band_fn("P03_fixpipe", "setup_ns", p03r.get("setup_ns"), 0.0, kind="latency_lower", allow_meas=allow),
        },
    )
    p["P04_mte3_store"] = make_cell_fn(
        "P04_mte3_store", {"rate": "bytes_per_s", "latency": "ns"}, {"has_ub_to_l1": spec["has_ub_to_l1"], **conflict_meta},
        {
            "mte3_store_bytes_per_s_per_aiv": band_fn("P04_mte3_store", "mte3_store_bytes_per_s_per_aiv", p04r.get("mte3_warm_bytes_per_s_per_aiv"), spec["mte3_bytes_per_s"] / aiv, allow_meas=allow),
            "atomic_rmw_bytes_per_s": band_fn("P04_mte3_store", "atomic_rmw_bytes_per_s", p04r.get("atomic_rmw_bytes_per_s"), spec["atomic_rmw_bytes_per_s"], allow_meas=allow),
        },
        {"setup_ns": _bound_only(0.0, "latency_lower"), "atomic_contention_ns": _bound_only(0.0, "latency_lower")},
    )
    p05_params: dict[str, Any] = {
        "l2_bytes": spec["l2_bytes"],
        "granule_bytes": spec.get("l2_sector_bytes", _DEFAULT_GRANULE_BYTES),
        "buffer_units_l2": _DEFAULT_UNIT_ONE,
        "cache_disable_supported": is_arch35,
        "ws_sweep_bytes_per_s": p05r.get("ws_sweep_bytes_per_s", {}),
        **conflict_meta,
    }
    if spec.get("l2_line_bytes") is not None:
        p05_params["l2_line_bytes"] = spec["l2_line_bytes"]
    if spec.get("l2_sector_bytes") is not None:
        p05_params["l2_sector_bytes"] = spec["l2_sector_bytes"]
    p["P05_l2_residency_reuse"] = make_cell_fn(
        "P05_l2_residency_reuse", {"rate": "bytes_per_s", "fraction": "dimensionless", "capacity": "bytes"},
        p05_params,
        {
            "l2_die_bytes_per_s": band_fn("P05_l2_residency_reuse", "l2_die_bytes_per_s", p05r.get("l2_die_bytes_per_s"), spec["l2_bytes_per_s"], allow_meas=allow),
            "l2_effective_fraction": band_fn("P05_l2_residency_reuse", "l2_effective_fraction", p05r.get("l2_effective_fraction"), 1.0, allow_meas=allow),
        },
    )
    p06_modes, any_p06 = _build_p06_mmad_modes(
        p06r, spec, aic=aic, is_arch35=is_arch35, allow=allow, band_fn=band_fn
    )
    p["P06_mmad"] = make_cell_fn(
        "P06_mmad", {"rate": "flops_per_s_per_aic", "latency": "ns"},
        {"pipe_m_barrier_fractal_threshold": _DEFAULT_PIPE_M_FRACTAL_THRESH, "small_m_bf16_ns": p06r.get("small_m_bf16_ns", {}), **conflict_meta}, {},
        {"pipe_m_barrier_ns": band_fn("P06_mmad", "pipe_m_barrier_ns", p06r.get("pipe_m_barrier_ns"), 0.0, kind="latency_lower", allow_meas=allow)},
        status_override="calibrated" if any_p06 else "unmeasured", modes=p06_modes,
    )
    p["P07_vec_alu"] = make_cell_fn("P07_vec_alu", {"rate": "ops_per_s_per_aiv"}, {"vector_width_bytes": _DEFAULT_VEC_WIDTH_BYTES, "fp32_lanes": _DEFAULT_FP32_LANES, "bf16_lanes": _DEFAULT_BF16_LANES, **conflict_meta}, {"vec_alu_ops_per_s_per_aiv": band_fn("P07_vec_alu", "vec_alu_ops_per_s_per_aiv", raw.get("P07_vec_alu", {}).get("vec_alu_ops_per_s_per_aiv"), spec["vec_alu_ops_per_s"] / aiv, allow_meas=allow)}, {"raw_latency_cycles": _bound_only(0.0, "latency_lower")})
    p["P08_vec_sfu"] = make_cell_fn("P08_vec_sfu", {"rate": "ops_per_s_per_aiv"}, {"vector_width_bytes": _DEFAULT_VEC_WIDTH_BYTES, "fp32_lanes": _DEFAULT_FP32_LANES, "bf16_lanes": _DEFAULT_BF16_LANES, **conflict_meta}, {"vec_sfu_ops_per_s_per_aiv": band_fn("P08_vec_sfu", "vec_sfu_ops_per_s_per_aiv", raw.get("P08_vec_sfu", {}).get("vec_sfu_ops_per_s_per_aiv"), spec["vec_sfu_ops_per_s"] / aiv, allow_meas=allow)}, {"raw_latency_cycles": _bound_only(0.0, "latency_lower")})
    p["P09_vec_reduce_cast"] = make_cell_fn("P09_vec_reduce_cast", {"rate": "ops_per_s_per_aiv", "cast_rate": "bytes_per_s_per_aiv"}, {"mx_scale_group": _DEFAULT_SCALE_GROUP, "mx_dynamic_quant_supported": is_arch35, **conflict_meta}, {"vec_reduce_ops_per_s_per_aiv": band_fn("P09_vec_reduce_cast", "vec_reduce_ops_per_s_per_aiv", p09r.get("vec_reduce_ops_per_s_per_aiv"), spec["vec_reduce_ops_per_s"] / aiv, allow_meas=allow), "cast_bytes_per_s_per_aiv": band_fn("P09_vec_reduce_cast", "cast_bytes_per_s_per_aiv", p09r.get("cast_bytes_per_s_per_aiv"), spec["cast_bytes_per_s"] / aiv, allow_meas=allow)})
    p["P10_sort_topk"] = _build_p10_sort_cell(
        raw.get("P10_sort_topk", {}), spec, aiv=aiv, is_arch35=is_arch35, allow=allow,
        conflict_meta=conflict_meta, band_fn=band_fn, make_cell_fn=make_cell_fn,
    )
    vec_cap = max(1.0, spec["vector_ops_per_s"] / aiv)
    p["P11_simt"] = make_cell_fn("P11_simt", {"rate": "ops_per_s_per_aiv", "latency": "ns"}, {"supported": is_arch35, "simt_ub_carveout_bytes": spec["simt_ub_carveout_bytes"], "tla_max_threads": _DEFAULT_TLA_THREADS_35 if is_arch35 else 0, "catlass_max_threads": _DEFAULT_CATLASS_THREADS_35 if is_arch35 else 0, "warp_size": _DEFAULT_WARP_SIZE}, {}, {"simt_ops_per_s_per_aiv": _bound_only(vec_cap if is_arch35 else 0.0), "syncthreads_ns": _bound_only(0.0, "latency_lower"), "simt_launch_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured" if is_arch35 else "unsupported")
    p["P12_intracore_flag"] = make_cell_fn(
        "P12_intracore_flag", {"latency": "ns"},
        {"event_ids": _DEFAULT_EVENT_IDS, "unit_flag_requires_single_l0c_stage": True, **conflict_meta},
        {"flag_ns": band_fn("P12_intracore_flag", "flag_ns", p12r.get("flag_ns"), 0.0, kind="latency_lower", allow_meas=allow)},
        {
            "unitflag_ns": band_fn("P12_intracore_flag", "unitflag_ns", p12r.get("unitflag_ns"), 0.0, kind="latency_lower", allow_meas=allow),
            "flag_mte2_v_ns": band_fn("P12_intracore_flag", "flag_mte2_v_ns", p12r.get("flag_mte2_v_ns"), 0.0, kind="latency_lower", allow_meas=allow),
            "flag_v_mte3_ns": band_fn("P12_intracore_flag", "flag_v_mte3_ns", p12r.get("flag_v_mte3_ns"), 0.0, kind="latency_lower", allow_meas=allow),
            "flag_m_fix_ns": band_fn("P12_intracore_flag", "flag_m_fix_ns", p12r.get("flag_m_fix_ns"), 0.0, kind="latency_lower", allow_meas=allow),
            "pipe_v_barrier_ns": band_fn("P12_intracore_flag", "pipe_v_barrier_ns", p12r.get("pipe_v_barrier_ns"), 0.0, kind="latency_lower", allow_meas=allow),
        },
    )
    p["P13_crosscore_sync"] = make_cell_fn("P13_crosscore_sync", {"latency": "ns"}, {"aiv_inter_block_barrier_id": _DEFAULT_BARRIER_AIV, "aic_inter_block_barrier_id": _DEFAULT_BARRIER_AIC, "intra_subblock_barrier_id": _DEFAULT_BARRIER_SUB, "ffts_max_flag_id": _DEFAULT_FFTS_MAX_FLAG, "reverse_depth": _DEFAULT_REVERSE_DEPTH}, {}, {"mode2_paired_ns": _bound_only(0.0, "latency_lower"), "mode1_subblock_ns": _bound_only(0.0, "latency_lower"), "mode0_aic_barrier_ns": _bound_only(0.0, "latency_lower"), "mode0_aiv_barrier_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p14_extra = (
        {"mutex_xpipe_handoff_ns": band_fn("P14_mutex_block_mmad", "mutex_xpipe_handoff_ns", p14r.get("mutex_xpipe_handoff_ns"), 0.0, kind="latency_lower", allow_meas=allow)}
        if (is_arch35 and p14r.get("mutex_xpipe_handoff_ns")) else (
            {} if is_arch35 else {"mutex_overhead_ns": _bound_only(0.0, "latency_lower")}
        )
    )
    p["P14_mutex_block_mmad"] = make_cell_fn(
        "P14_mutex_block_mmad", {"latency": "ns"},
        {"max_mutex_ids": _DEFAULT_MAX_MUTEX_IDS if is_arch35 else 0, "unit_flag_requires_single_l0c_stage": True, **conflict_meta},
        {"mutex_overhead_ns": band_fn("P14_mutex_block_mmad", "mutex_overhead_ns", p14r.get("mutex_overhead_ns"), 0.0, kind="latency_lower", allow_meas=allow)} if is_arch35 else {},
        p14_extra,
        status_override=None if is_arch35 else "unsupported",
    )
    p["P15_rts_stars"] = make_cell_fn("P15_rts_stars", {"latency": "ns"}, {"queue_scope": "die", "queue_depth": _DEFAULT_QUEUE_DEPTH, "lanes": _DEFAULT_UNIT_ONE, **conflict_meta}, {"issue_interval_ns": band_fn("P15_rts_stars", "issue_interval_ns", p15r.get("issue_interval_ns"), 0.0, kind="latency_lower", allow_meas=allow), "aiv_first_issue_ns": band_fn("P15_rts_stars", "aiv_first_issue_ns", p15r.get("aiv_first_issue_ns"), 0.0, kind="latency_lower", allow_meas=allow), "aic_first_issue_ns": band_fn("P15_rts_stars", "aic_first_issue_ns", p15r.get("aic_first_issue_ns"), 0.0, kind="latency_lower", allow_meas=allow)}, {"host_submit_ns": band_fn("P15_rts_stars", "host_submit_ns", p15r.get("host_submit_ns"), 0.0, kind="latency_lower", allow_meas=allow)})
    p["P16_multistream_core_control"] = make_cell_fn("P16_multistream_core_control", {"latency": "ns"}, {"modes": ["disjoint", "shared_cap"], **conflict_meta}, {"stream_event_ns": band_fn("P16_multistream_core_control", "stream_event_ns", raw.get("P16_multistream_core_control", {}).get("stream_event_ns"), 0.0, kind="latency_lower", allow_meas=allow)}, {"set_limit_ns": _bound_only(0.0, "latency_lower")})
    p["P17_aicpu_metadata_launch"] = make_cell_fn("P17_aicpu_metadata_launch", {"latency": "ns"}, {"aclgraph_cacheable": True}, {}, {"host_select_ns": _bound_only(0.0, "latency_lower"), "aicpu_base_ns": _bound_only(0.0, "latency_lower"), "gm_pub_ns": _bound_only(0.0, "latency_lower"), "eager_d2h_sync_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P18_hccl_collective_band"] = make_cell_fn("P18_hccl_collective_band", {"rate": "bytes_per_s", "latency": "ns"}, {"payload_bytes_per_frame": _DEFAULT_HCCL_PAYLOAD_35 if is_arch35 else _DEFAULT_HCCL_FRAME_22, "wire_bytes_per_frame": _DEFAULT_HCCL_WIRE_35 if is_arch35 else _DEFAULT_HCCL_FRAME_22, "short_final_frame_policy": "exact_plus_header"}, {}, {"link_bytes_per_s": _bound_only(spec["link_bytes_per_s"]), "startup_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P19_direct_peer_window"] = make_cell_fn("P19_direct_peer_window", {"rate": "bytes_per_s", "latency": "ns"}, {"transport": "a5_ub_shmem_mte" if is_arch35 else "a3_sio_hccs", "token_payload_bytes_per_frame": _DEFAULT_HCCL_PAYLOAD_35, "token_wire_bytes_per_frame": _DEFAULT_HCCL_WIRE_35}, {}, {"peer_bytes_per_s": _bound_only(spec["link_bytes_per_s"]), "status_sync_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P20_host_dma_sync"] = make_cell_fn(
        "P20_host_dma_sync", {"rate": "bytes_per_s", "latency": "ns"}, {"contention_scope": "root_complex", **conflict_meta},
        {
            "host_h2d_bytes_per_s": band_fn("P20_host_dma_sync", "host_h2d_bytes_per_s", p20r.get("host_h2d_bytes_per_s"), spec["host_h2d_bytes_per_s"], allow_meas=allow),
            "host_d2h_bytes_per_s": band_fn("P20_host_dma_sync", "host_d2h_bytes_per_s", p20r.get("host_d2h_bytes_per_s"), spec["host_d2h_bytes_per_s"], allow_meas=allow),
        },
        {
            "host_rc_h2d_bytes_per_s": _bound_only(spec["host_rc_h2d_bytes_per_s"]),
            "host_rc_d2h_bytes_per_s": _bound_only(spec["host_rc_d2h_bytes_per_s"]),
            "pinned_dma_startup_ns": band_fn("P20_host_dma_sync", "pinned_dma_startup_ns", p20r.get("pinned_dma_startup_ns"), 0.0, kind="latency_lower", allow_meas=allow),
            "pageable_dma_startup_ns": band_fn("P20_host_dma_sync", "pageable_dma_startup_ns", p20r.get("pageable_dma_startup_ns"), 0.0, kind="latency_lower", allow_meas=allow),
            "eager_d2h_sync_ns": band_fn("P20_host_dma_sync", "eager_d2h_sync_ns", p20r.get("eager_d2h_sync_ns"), 0.0, kind="latency_lower", allow_meas=allow),
        },
    )
    return p


def build_primitive_table_with_diagnostics(
    *,
    sku: str,
    cann_version: str,
    endpoint_alias: str,
    campaign_id: str,
    raw_measurements: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Assemble sealed ``mindie-l0-primitives/1`` table plus explicit per-metric conflicts & coverage."""
    spec = load_sku_hardware_spec(sku)
    alias = sanitize_record_alias(endpoint_alias, sku)
    clean_campaign = sanitize_record_campaign(campaign_id)
    aic, aiv = max(_DEFAULT_UNIT_ONE, spec["aic_count"]), max(_DEFAULT_UNIT_ONE, spec["aiv_count"])
    is_arch35 = spec["arch_preset"] == _ARCH_PRESET_35
    raw = dict(raw_measurements or {})
    sample_dev = str(raw.get("device_name") or "").strip()
    if sample_dev and sample_dev != sku:
        raise ValueError(f"bench_sample_sku_mismatch:expected={sku}:got={sample_dev}")
    obs = raw.get("observed_silicon") if isinstance(raw.get("observed_silicon"), Mapping) else {}
    obs_aic = obs.get("observed_aic_count")
    core_mismatch = isinstance(obs_aic, int) and not isinstance(obs_aic, bool) and obs_aic > 0 and obs_aic != aic
    conflict_meta = (
        {"silicon_profile_conflict": {
            "observed_aic_count": obs_aic, "profile_aic_count": aic,
            "observed_aicore_freq_mhz": obs.get("observed_aicore_freq_mhz"),
        }} if core_mismatch else {}
    )
    cap_bytes = canonical_capture_bytes(sku, raw_measurements)
    top_capture = f"capture://{alias}/{clean_campaign}/{hashlib.sha256(cap_bytes).hexdigest()}"
    metric_conflicts: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    def _band(
        pid: str, metric: str, samples: Sequence[float] | None, bound: float | None,
        *, kind: str = "rate_upper", allow_meas: bool = True, mode: str = "",
        max_admissible: float | None = None,
    ) -> dict[str, Any] | None:
        if bound is None:
            reason = "undeclared_hardware_bound"
            out = None
        elif not allow_meas:
            reason = "silicon_profile_conflict"
            out = _bound_only(bound, kind)
        elif not samples:
            reason = "missing_samples"
            out = _bound_only(bound, kind)
        else:
            out, reason = summarize_samples_with_reason(
                samples, certified_bound=bound, bound_kind=kind, max_admissible=max_admissible,
            )
        coverage_rows.append({
            "sku": sku, "cann_version": cann_version, "primitive": pid,
            "mode": mode, "metric": metric, "status": "calibrated" if reason == "calibrated" else "unmeasured",
            "reason": reason,
        })
        if reason not in {"calibrated", "missing_samples"}:
            metric_conflicts.append({
                "sku": sku, "cann_version": cann_version, "primitive": pid,
                "mode": mode, "metric": metric, "reason": reason,
            })
        return out

    def _make_cell(
        pid: str, units: dict[str, str], params: dict[str, Any],
        core_metrics: dict[str, dict[str, Any] | None],
        extra_metrics: dict[str, dict[str, Any] | None] | None = None,
        *, status_override: str | None = None, modes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        st = status_override if status_override is not None else _cell_status(core_metrics)
        merged = {
            k: (v if st == "calibrated" else _clean_band(v))
            for k, v in core_metrics.items() if v is not None
        }
        for k, v in (extra_metrics or {}).items():
            if v is not None:
                merged[k] = _clean_band(v)
        ev = top_capture if st == "calibrated" else ""
        return {"status": st, "units": units, "parameters": params, "metrics": merged, "modes": modes or {}, "evidence_id": ev}

    p = _assemble_primitive_cells(
        raw=raw, spec=spec, aic=aic, aiv=aiv, is_arch35=is_arch35,
        allow=not core_mismatch, conflict_meta=conflict_meta,
        band_fn=_band, make_cell_fn=_make_cell,
    )
    payload: dict[str, Any] = {
        "schema_version": PRIMITIVE_SCHEMA_VERSION, "arch_preset": spec["arch_preset"],
        "sku": sku, "cann_version": cann_version,
        "provenance": {
            "campaign_id": clean_campaign, "endpoint_alias": alias, "capture_uri": top_capture,
            "calibrated": any(c["status"] == "calibrated" for c in p.values()),
            "collected_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "silicon_fingerprint": {k: spec[k] for k in ("aic_count", "aiv_count", "hbm_gib", "l2_bytes", "ub_bytes", "l1_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes")},
        },
        "primitives": p,
    }
    payload["envelope_sha256"] = compute_envelope_sha256(payload)
    return payload, metric_conflicts, coverage_rows


def build_primitive_table_payload(
    *,
    sku: str,
    cann_version: str,
    endpoint_alias: str,
    campaign_id: str,
    raw_measurements: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble a full-envelope SHA-256 sealed ``mindie-l0-primitives/1`` table."""
    payload, _conflicts, _coverage = build_primitive_table_with_diagnostics(
        sku=sku,
        cann_version=cann_version,
        endpoint_alias=endpoint_alias,
        campaign_id=campaign_id,
        raw_measurements=raw_measurements,
    )
    return payload


def update_manifest(manifest_path: Path, table_path: Path, payload: Mapping[str, Any]) -> None:
    """Update ``data/primitives/manifest.json`` atomically with one sealed table entry."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"schema_version": PRIMITIVE_SCHEMA_VERSION, "tables": {}}
    sku = str(payload["sku"])
    manifest.setdefault("tables", {})[sku] = {
        "file": table_path.name,
        "sku": sku,
        "compatible_skus": [sku],
        "arch_preset": payload["arch_preset"],
        "cann_version": payload["cann_version"],
        "calibrated": bool(payload["provenance"]["calibrated"]),
        "calibrated_primitive_count": sum(1 for c in payload["primitives"].values() if c["status"] == "calibrated"),
        "total_primitives": len(payload["primitives"]),
        "envelope_sha256": payload["envelope_sha256"],
        "capture_uri": payload["provenance"]["capture_uri"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "MAX_CALIBRATED_CV_PCT",
    "PRIMITIVE_SCHEMA_VERSION",
    "PRIVATE_CAPTURE_RE",
    "SKU_DEFAULT_FILES",
    "build_primitive_table_payload",
    "build_primitive_table_with_diagnostics",
    "canonical_capture_bytes",
    "compute_envelope_sha256",
    "load_sku_hardware_spec",
    "sanitize_record_alias",
    "sanitize_record_campaign",
    "summarize_samples",
    "summarize_samples_with_reason",
    "update_manifest",
]
