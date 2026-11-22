#!/usr/bin/env python3
"""Hardware bound derivation, metric statistics, and BurstTurnRule builder.

[Role] Plane-1 calibration helper owning SKU hardware upper-bound extraction,
  core-clock-only bound scaling, UncertaintyBand CV/sample-count summarization,
  and pairwise-disjoint half-open BurstTurnRule derivation for L0 tables.
[Invariants] Strictly offline Plane-1 tool outside ``mindie_sim/``. Enforces
  MIN_CALIBRATED_SAMPLES >= 8, rejects NaN/nonpositive samples, checks
  p99 <= certified_upper_capacity and optional max_admissible limits without
  clipping residuals, and derives directional PCIe caps without inflating by
  copied effective rates.
[Timeouts & Locks] Pure in-memory numerical computation; zero I/O or locks.
[Side-Effect Policy] Zero filesystem or external side effects.
"""
from __future__ import annotations

import math
import statistics
from typing import Any

MAX_CALIBRATED_CV_PCT = 5.0
MIN_CALIBRATED_SAMPLES = 8
TRIM_FRACTION = 0.10
GIB = 1024 * 1024 * 1024
PCT_SCALE = 100.0
Q_P05 = 0.05
Q_P50 = 0.50
Q_P95 = 0.95
Q_P99 = 0.99
ROUND_METRIC_DIGITS = 6
ROUND_CV_DIGITS = 4
BURST_TURN_ROUND_DIGITS = 2
GBPS_TO_BPS = 1e9
MHZ_TO_HZ = 1e6
BF16_FLOPS_PER_CYCLE_PER_AIC = 8192.0
LOAD_DATA_BYTES_PER_CYCLE = 256.0
DAV22_UB_CAST_BYTES_PER_VEC_OP = 6.0
SIMT_WARP_SCHEDULERS = 4
SIMT_UB_CARVEOUT_MIN_BYTES = 32768
DEFAULT_LINK_BPS = 2.5e10
DEFAULT_HOST_DMA_BPS = 3.15e10
DEFAULT_HOST_RC_BPS = 2.0e11
MIN_STDEV_SAMPLES = 2
TWO = 2.0

CORE_CLOCK_COUPLED_RATE_KEYS = (
    "l2_bytes_per_s",
    "mte1_l0a_bytes_per_s",
    "mte1_l0b_bytes_per_s",
    "mte1_l0b_transpose_bytes_per_s",
    "mte2_aic_bytes_per_s",
    "mte2_aiv_bytes_per_s",
    "mte3_bytes_per_s",
    "fixpipe_bytes_per_s",
    "vector_ops_per_s",
    "vec_alu_ops_per_s",
    "vec_sfu_ops_per_s",
    "vec_reduce_ops_per_s",
    "cast_bytes_per_s",
    "simt_warp_insts_per_s_per_aiv",
)


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi or sorted_vals[lo] == sorted_vals[hi]:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def summarize_metric(
    samples: list[float] | None, *, certified_bound: float, bound_kind: str = "rate_upper",
    allow_meas: bool = True, max_admissible: float | None = None, allow_zero: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize samples into an UncertaintyBand dict plus an explicit RC-1 diagnostic record."""
    nom = round(float(certified_bound), ROUND_METRIC_DIGITS)
    bound_only = {"certified_upper_capacity": nom, "bound_kind": bound_kind}
    if not samples or not allow_meas:
        return bound_only, {"status": "unmeasured", "nominal_bound": nom}
    clean: list[float] = []
    invalid = 0
    for x in samples:
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(float(x)):
            invalid += 1
            continue
        fv = float(x)
        if fv < 0.0 or (fv == 0.0 and not allow_zero):
            invalid += 1
            continue
        clean.append(fv)
    if invalid > 0 or len(clean) < MIN_CALIBRATED_SAMPLES:
        return bound_only, {
            "status": "insufficient_or_invalid_samples",
            "valid_samples": len(clean), "invalid_samples": invalid,
            "min_required": MIN_CALIBRATED_SAMPLES, "nominal_bound": nom,
        }
    clean.sort()
    n = len(clean)
    mean_v = sum(clean) / n
    cv_pct = (PCT_SCALE * statistics.stdev(clean) / mean_v) if (n >= MIN_STDEV_SAMPLES and mean_v > 0.0) else 0.0
    trim_k = int(math.floor(n * TRIM_FRACTION))
    trimmed = clean[trim_k : n - trim_k] if n - MIN_STDEV_SAMPLES * trim_k >= MIN_STDEV_SAMPLES else clean
    t_mean = sum(trimmed) / len(trimmed)
    t_cv = (PCT_SCALE * statistics.stdev(trimmed) / t_mean) if (len(trimmed) >= MIN_STDEV_SAMPLES and t_mean > 0.0) else 0.0
    p05, p50, p95, p99 = (_quantile(clean, q) for q in (Q_P05, Q_P50, Q_P95, Q_P99))
    exceeded = (p99 > float(certified_bound) or p05 <= 0.0) if bound_kind == "rate_upper" else (p05 < float(certified_bound))
    above_max = max_admissible is not None and p99 > float(max_admissible)
    if exceeded or above_max:
        diag: dict[str, Any] = {
            "status": "p99_exceeds_nominal_conflict",
            "observed_p05": round(p05, ROUND_METRIC_DIGITS),
            "observed_p50": round(p50, ROUND_METRIC_DIGITS),
            "observed_p99": round(p99, ROUND_METRIC_DIGITS),
            "cv_pct": round(cv_pct, ROUND_CV_DIGITS),
            "nominal_bound": nom,
        }
        if max_admissible is not None:
            diag["admissible_max"] = round(float(max_admissible), ROUND_METRIC_DIGITS)
        return bound_only, diag
    band = {
        "p05": round(p05, ROUND_METRIC_DIGITS), "p50": round(p50, ROUND_METRIC_DIGITS),
        "p95": round(p95, ROUND_METRIC_DIGITS), "p99": round(p99, ROUND_METRIC_DIGITS),
        "trimmed_mean": round(t_mean, ROUND_METRIC_DIGITS), "trimmed_cv_pct": round(t_cv, ROUND_CV_DIGITS),
        "cv_pct": round(cv_pct, ROUND_CV_DIGITS), "n_samples": n,
        "certified_upper_capacity": nom, "bound_kind": bound_kind,
    }
    if cv_pct > MAX_CALIBRATED_CV_PCT:
        return bound_only, {
            "status": "high_cv_unsealed",
            "observed_p50": round(p50, ROUND_METRIC_DIGITS),
            "observed_p99": round(p99, ROUND_METRIC_DIGITS),
            "cv_pct": round(cv_pct, ROUND_CV_DIGITS),
            "nominal_bound": nom,
        }
    return band, {"status": "calibrated", "cv_pct": round(cv_pct, ROUND_CV_DIGITS), "nominal_bound": nom}


def summarize_samples(samples: list[float], *, certified_bound: float, bound_kind: str = "rate_upper") -> dict[str, Any]:
    """Compatibility wrapper returning the sealed UncertaintyBand mapping."""
    band, _diag = summarize_metric(samples, certified_bound=certified_bound, bound_kind=bound_kind)
    return band


PCIE_GEN4X16_UNI_GBPS = 31.5
PCIE_GEN5X16_UNI_GBPS = 64.0
DAV35_SFU_LANES_PER_CYCLE = 64.0


def directional_pcie_bps(chan: dict[str, Any]) -> float:
    uni = float(chan.get("uni_peak_bw_gbps") or 0.0)
    bi = float(chan.get("bi_peak_bw_gbps") or 0.0)
    proto = str(chan.get("protocol") or "").lower()
    proto_cap = PCIE_GEN5X16_UNI_GBPS if "gen5" in proto else (PCIE_GEN4X16_UNI_GBPS if "gen4" in proto else 0.0)
    if uni > 0.0:
        cap = min(uni, proto_cap) if proto_cap > 0.0 else uni
        return cap * GBPS_TO_BPS
    if proto_cap > 0.0:
        return proto_cap * GBPS_TO_BPS
    return (bi / TWO) * GBPS_TO_BPS if bi > 0.0 else 0.0


def load_sku_hardware_spec(sku: str) -> dict[str, Any]:
    from mindie_sim.backends import ascend
    from mindie_sim.schema.datapath import datapath_for_die
    from mindie_sim.schema.hardware import Hardware

    hw = Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))
    die = hw.dies[0]
    dp = datapath_for_die(die)
    bufs, rates, sm = dict(die.buffers), dict(die.rates), dict(die.service_modes)
    tc = {k: dict(v) for k, v in hw.transfer_channels}
    h2d, d2h, d2d = tc.get("h2d", {}), tc.get("d2h", {}), tc.get("d2d_intra_node", {})
    h2d_solo, d2h_solo = directional_pcie_bps(h2d), directional_pcie_bps(d2h)
    h2d_rc = float(h2d.get("node_peak_bw_gbps") or 0.0) * GBPS_TO_BPS or h2d_solo
    d2h_rc = float(d2h.get("node_peak_bw_gbps") or 0.0) * GBPS_TO_BPS or d2h_solo
    link_uni = float(d2d.get("uni_peak_bw_gbps") or d2d.get("effective_uni_bw_gbps") or 0.0) * GBPS_TO_BPS
    is_arch35 = str(die.compute_arch.preset or "npu_arch_2201") == "npu_arch_3510"
    aic_cnt = max(1, int(dp.aic_count))
    peaks = {str(k): float(v) for k, v in dict(die.compute_peaks).items()}
    bf16_peak = float(peaks.get("bf16", 0.0))
    nom_clock_hz = bf16_peak / (aic_cnt * BF16_FLOPS_PER_CYCLE_PER_AIC) if bf16_peak > 0.0 else 0.0
    mte1_arch_bps = LOAD_DATA_BYTES_PER_CYCLE * nom_clock_hz * aic_cnt if nom_clock_hz > 0.0 else 0.0
    mte1_l0a_raw = float(sm["mte1_l0a"][1]) if "mte1_l0a" in sm else float(rates.get("mte1", 0.0))
    mte1_l0a_bps = max(mte1_l0a_raw, mte1_arch_bps) if is_arch35 else mte1_l0a_raw
    mte1_l0b_raw = float(sm["mte1_l0b"][1]) if "mte1_l0b" in sm else float(rates.get("mte1", 0.0))
    mte1_l0b_bps = max(mte1_l0a_bps, mte1_l0b_raw) if is_arch35 else mte1_l0b_raw
    mte1_l0b_t_bps = float(sm["mte1_l0b_transpose"][1]) if "mte1_l0b_transpose" in sm else mte1_l0b_bps
    mte2_aic_raw = float(sm["mte2_aic_l2"][1]) if "mte2_aic_l2" in sm else (
        float(sm["mte2_aic_cached"][1]) if "mte2_aic_cached" in sm else float(rates.get("mte2_aic", 0.0))
    )
    mte2_aiv_raw = float(sm["mte2_aiv_l2"][1]) if "mte2_aiv_l2" in sm else (
        float(sm["mte2_aiv_cached"][1]) if "mte2_aiv_cached" in sm else float(rates.get("mte2_aiv", 0.0))
    )
    l2_die_bps = float(rates.get("l2") or max(mte2_aic_raw, mte2_aiv_raw))
    mte2_aic_bps = min(mte2_aic_raw, l2_die_bps) if (mte2_aic_raw > 0.0 and l2_die_bps > 0.0) else (mte2_aic_raw or l2_die_bps)
    mte2_aiv_bps = min(mte2_aiv_raw, l2_die_bps) if (mte2_aiv_raw > 0.0 and l2_die_bps > 0.0) else (mte2_aiv_raw or l2_die_bps)
    cast_raw = float(rates.get("ub_write_bytes", l2_die_bps))
    cast_bps = cast_raw if is_arch35 else max(cast_raw, float(rates.get("vector", 0.0)) * DAV22_UB_CAST_BYTES_PER_VEC_OP * TWO)
    simt_warp_cap = SIMT_WARP_SCHEDULERS * nom_clock_hz if (is_arch35 and nom_clock_hz > 0.0) else 0.0
    sfu_raw = float(rates.get("reg_special_lanes", rates.get("vector", 0.0)))
    sfu_arch35_bps = DAV35_SFU_LANES_PER_CYCLE * nom_clock_hz * int(dp.aiv_count) if (is_arch35 and nom_clock_hz > 0.0) else 0.0
    vec_sfu_bps = max(sfu_raw, sfu_arch35_bps) if is_arch35 else sfu_raw
    return {
        "sku": sku, "arch_preset": str(die.compute_arch.preset or "npu_arch_2201"),
        "dies_per_package": max(1, int(getattr(hw, "dies_per_package", 1) or 1)),
        "aic_count": int(dp.aic_count), "aiv_count": int(dp.aiv_count),
        "hbm_gib": int(die.memory_bytes // GIB),
        "l2_bytes": int(bufs.get("l2", 0)), "ub_bytes": int(bufs.get("ub", 0)),
        "l1_bytes": int(bufs.get("l1", 0)), "l0a_bytes": int(bufs.get("l0a", 0)),
        "l0b_bytes": int(bufs.get("l0b", 0)), "l0c_bytes": int(bufs.get("l0c", 0)),
        "has_l0c_to_ub": bool(dp.has_l0c_to_ub), "has_ub_to_l1": bool(dp.has_ub_to_l1),
        "simt_ub_carveout_bytes": max(SIMT_UB_CARVEOUT_MIN_BYTES, int(dp.simt_ub_carveout_bytes)) if is_arch35 else 0,
        "hbm_read_bytes_per_s": float(dp.hbm_read_bytes_per_s or rates.get("hbm", 0.0)),
        "hbm_write_bytes_per_s": float(dp.hbm_write_bytes_per_s or rates.get("hbm", 0.0)),
        "l2_bytes_per_s": l2_die_bps,
        "mte1_l0a_bytes_per_s": mte1_l0a_bps,
        "mte1_l0b_bytes_per_s": mte1_l0b_bps,
        "mte1_l0b_transpose_bytes_per_s": mte1_l0b_t_bps,
        "mte2_aic_bytes_per_s": mte2_aic_bps,
        "mte2_aiv_bytes_per_s": mte2_aiv_bps,
        "mte3_bytes_per_s": float(rates.get("mte3", 0.0)),
        "atomic_rmw_bytes_per_s": float(sm["hbm_atomic_rmw"][1]) if "hbm_atomic_rmw" in sm else float(dp.hbm_write_bytes_per_s or rates.get("hbm", 0.0)),
        "fixpipe_bytes_per_s": float(rates.get("fixpipe", 0.0)),
        "vector_ops_per_s": float(rates.get("vector", 0.0)),
        "vec_alu_ops_per_s": float(rates.get("vector", 0.0)),
        "vec_sfu_ops_per_s": vec_sfu_bps,
        "vec_reduce_ops_per_s": float(rates.get("reg_reduce_lanes", rates.get("vector", 0.0))),
        "cast_bytes_per_s": cast_bps,
        "simt_warp_insts_per_s_per_aiv": simt_warp_cap,
        "link_bytes_per_s": link_uni or DEFAULT_LINK_BPS,
        "host_h2d_bytes_per_s": h2d_solo or DEFAULT_HOST_DMA_BPS,
        "host_d2h_bytes_per_s": d2h_solo or DEFAULT_HOST_DMA_BPS,
        "host_rc_h2d_bytes_per_s": h2d_rc or DEFAULT_HOST_RC_BPS,
        "host_rc_d2h_bytes_per_s": d2h_rc or DEFAULT_HOST_RC_BPS,
        "peaks": peaks,
    }


def apply_core_clock_scale(
    spec: dict[str, Any], observed_mhz: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Scale core-clock-coupled upper bounds to observed_aicore_freq_mhz; keep HBM/links unscaled."""
    aic = max(1, int(spec["aic_count"]))
    bf16_peak = float(spec["peaks"].get("bf16", 0.0))
    nom_mhz = (bf16_peak / (aic * BF16_FLOPS_PER_CYCLE_PER_AIC)) / MHZ_TO_HZ if bf16_peak > 0.0 else 0.0
    obs_val = float(observed_mhz) if isinstance(observed_mhz, (int, float)) and observed_mhz > 0 else 0.0
    ratio = (obs_val / nom_mhz) if (obs_val > 0.0 and nom_mhz > 0.0) else 1.0
    scaled = dict(spec)
    if ratio != 1.0:
        for key in CORE_CLOCK_COUPLED_RATE_KEYS:
            scaled[key] = float(spec[key]) * ratio
        scaled["peaks"] = {k: float(v) * ratio for k, v in spec["peaks"].items()}
    meta = {
        "nominal_cube_freq_mhz": round(nom_mhz, ROUND_CV_DIGITS) if nom_mhz > 0.0 else None,
        "observed_aicore_freq_mhz": int(obs_val) if obs_val > 0.0 else None,
        "core_clock_scale_ratio": round(ratio, ROUND_METRIC_DIGITS),
    }
    return scaled, meta


def derive_burst_turn_rules(
    surface: list[dict[str, Any]] | None, *, peak_read_bps: float, granule_bytes: int, evidence_id: str = "",
) -> list[dict[str, Any]]:
    """Derive pairwise-disjoint half-open BurstTurnRule entries for sub-granule strided accesses."""
    if not surface or peak_read_bps <= 0.0:
        return []
    by_row_stride: dict[int, dict[int, list[float]]] = {}
    for item in surface:
        rb = int(item.get("row_bytes", 0) or 0)
        sb = int(item.get("stride_bytes", 0) or 0)
        bps = float(item.get("bps_p50", 0.0) or 0.0)
        if 0 < rb < granule_bytes and sb >= granule_bytes and sb > rb and bps > 0.0:
            by_row_stride.setdefault(rb, {}).setdefault(sb, []).append(bps)
    sorted_rbs = sorted(by_row_stride)
    rules: list[dict[str, Any]] = []
    for idx, rb in enumerate(sorted_rbs):
        min_rb = rb
        max_rb = sorted_rbs[idx + 1] if idx + 1 < len(sorted_rbs) else granule_bytes
        sorted_sbs = sorted(by_row_stride[rb])
        for s_idx, sb in enumerate(sorted_sbs):
            eff_bps = statistics.median(by_row_stride[rb][sb])
            eff_bytes = float(rb) * (peak_read_bps / max(1.0, eff_bps))
            max_phys_penalty = float(max(0, min(sb - rb, granule_bytes - rb)))
            penalty = min(max_phys_penalty, max(0.0, eff_bytes - float(rb)))
            max_sb = sorted_sbs[s_idx + 1] if s_idx + 1 < len(sorted_sbs) else None
            rules.append({
                "min_row_bytes": min_rb,
                "max_row_bytes": max_rb,
                "min_stride_bytes": sb,
                "max_stride_bytes": max_sb,
                "penalty_bytes": round(penalty, BURST_TURN_ROUND_DIGITS),
                "evidence_id": evidence_id,
            })
    return rules
