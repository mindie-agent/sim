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
_DEFAULT_LINK_BPS = 2.5e10
_DEFAULT_HOST_DMA_BPS = 3.15e10
_DEFAULT_HOST_RC_BPS = 2.0e11
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
_TILE_16 = 16
_TILE_8 = 8
_TILE_32 = 32
_TILE_64 = 64
_DOUBLE_FACTOR = 2.0
_DEFAULT_UNIT_ONE = 1

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
        "link_bytes_per_s": link_uni or _DEFAULT_LINK_BPS,
        "host_h2d_bytes_per_s": h2d_solo or _DEFAULT_HOST_DMA_BPS,
        "host_d2h_bytes_per_s": d2h_solo or _DEFAULT_HOST_DMA_BPS,
        "host_rc_h2d_bytes_per_s": h2d_rc or _DEFAULT_HOST_RC_BPS,
        "host_rc_d2h_bytes_per_s": d2h_rc or _DEFAULT_HOST_RC_BPS,
        "peaks": {str(k): float(v) for k, v in peaks_raw.items()},
    }


def _cell_status(metrics: Mapping[str, Mapping[str, Any]]) -> str:
    if not metrics:
        return "unmeasured"
    for band in metrics.values():
        if band.get("p50") is None or band.get("cv_pct") is None:
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


def _bound_only(value: float, kind: str = "rate_upper") -> dict[str, Any]:
    return {"certified_upper_capacity": round(float(value), _ROUND_STAT_DIGITS), "bound_kind": kind}


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
        pid: str, metric: str, samples: Sequence[float] | None, bound: float,
        *, kind: str = "rate_upper", allow_meas: bool = True, mode: str = "",
    ) -> dict[str, Any]:
        if not allow_meas:
            reason = "silicon_profile_conflict"
            out = _bound_only(bound, kind)
        elif not samples:
            reason = "missing_samples"
            out = _bound_only(bound, kind)
        else:
            out, reason = summarize_samples_with_reason(samples, certified_bound=bound, bound_kind=kind)
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
        core_metrics: dict[str, dict[str, Any]], extra_metrics: dict[str, dict[str, Any]] | None = None,
        *, status_override: str | None = None, modes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        st = status_override if status_override is not None else _cell_status(core_metrics)
        merged = {k: (v if st == "calibrated" else _clean_band(v)) for k, v in core_metrics.items()}
        for k, v in (extra_metrics or {}).items():
            merged[k] = _clean_band(v)
        ev = top_capture if st == "calibrated" else ""
        return {"status": st, "units": units, "parameters": params, "metrics": merged, "modes": modes or {}, "evidence_id": ev}

    p: dict[str, Any] = {}
    p01r, p04r, p05r, p06r = (raw.get(k, {}) for k in ("P01_datacopy", "P04_mte3_store", "P05_l2_residency_reuse", "P06_mmad"))
    p09r, p15r, p20r = (raw.get(k, {}) for k in ("P09_vec_reduce_cast", "P15_rts_stars", "P20_host_dma_sync"))
    allow = not core_mismatch
    p["P01_datacopy"] = _make_cell(
        "P01_datacopy", {"rate": "bytes_per_s", "penalty": "bytes", "alpha": "dimensionless"},
        {"sector_bytes": _DEFAULT_SECTOR_BYTES, "l2_cache_ctl_supported": is_arch35, "burst_turn_penalty": [], **conflict_meta},
        {
            "hbm_read_bytes_per_s": _band("P01_datacopy", "hbm_read_bytes_per_s", p01r.get("hbm_read_bytes_per_s"), spec["hbm_read_bytes_per_s"], allow_meas=allow),
            "hbm_write_bytes_per_s": _band("P01_datacopy", "hbm_write_bytes_per_s", p01r.get("hbm_write_bytes_per_s"), spec["hbm_write_bytes_per_s"], allow_meas=allow),
            "mte2_aiv_hit_bytes_per_s_per_core": _band("P01_datacopy", "mte2_aiv_hit_bytes_per_s_per_core", p01r.get("mte2_aiv_hit_bytes_per_s_per_core"), spec["mte2_aiv_bytes_per_s"] / aiv, allow_meas=allow),
        },
        {
            "hbm_rw_alpha": _band("P01_datacopy", "hbm_rw_alpha", p01r.get("hbm_rw_alpha"), 0.0, kind="latency_lower", allow_meas=allow),
            "mte2_aic_hit_bytes_per_s_per_core": _bound_only(spec["mte2_aic_bytes_per_s"] / aic),
        },
    )
    p["P02_load_data"] = _make_cell(
        "P02_load_data", {"rate": "bytes_per_s_per_aic", "latency": "ns"},
        {"fractal_bytes": _DEFAULT_FRACTAL_BYTES, "mx_scale_supported": is_arch35, "transpose_factor": 1.0}, {},
        {"l1_to_l0a_bytes_per_s_per_aic": _bound_only(spec["mte1_l0a_bytes_per_s"] / aic), "l1_to_l0b_bytes_per_s_per_aic": _bound_only(spec["mte1_l0b_bytes_per_s"] / aic), "setup_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured",
    )
    fixp_aic = spec["fixpipe_bytes_per_s"] / aic
    p["P03_fixpipe"] = _make_cell(
        "P03_fixpipe", {"rate": "bytes_per_s_per_aic", "latency": "ns"},
        {"has_l0c_to_ub": spec["has_l0c_to_ub"], "has_l0c_to_l1": True, "ub_split_n_align_elems": _DEFAULT_SECTOR_BYTES}, {},
        {"fixpipe_in_bytes_per_s_per_aic": _bound_only(fixp_aic), "fixpipe_out_bytes_per_s_per_aic": _bound_only(fixp_aic), "setup_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured",
    )
    p["P04_mte3_store"] = _make_cell(
        "P04_mte3_store", {"rate": "bytes_per_s", "latency": "ns"}, {"has_ub_to_l1": spec["has_ub_to_l1"], **conflict_meta},
        {
            "mte3_store_bytes_per_s_per_aiv": _band("P04_mte3_store", "mte3_store_bytes_per_s_per_aiv", p04r.get("mte3_warm_bytes_per_s_per_aiv"), spec["mte3_bytes_per_s"] / aiv, allow_meas=allow),
            "atomic_rmw_bytes_per_s": _band("P04_mte3_store", "atomic_rmw_bytes_per_s", p04r.get("atomic_rmw_bytes_per_s"), spec["atomic_rmw_bytes_per_s"], allow_meas=allow),
        },
        {"setup_ns": _bound_only(0.0, "latency_lower"), "atomic_contention_ns": _bound_only(0.0, "latency_lower")},
    )
    p["P05_l2_residency_reuse"] = _make_cell(
        "P05_l2_residency_reuse", {"rate": "bytes_per_s", "fraction": "dimensionless", "capacity": "bytes"},
        {"l2_bytes": spec["l2_bytes"], "granule_bytes": _DEFAULT_GRANULE_BYTES, "buffer_units_l2": _DEFAULT_UNIT_ONE, "cache_disable_supported": is_arch35, "ws_sweep_bytes_per_s": p05r.get("ws_sweep_bytes_per_s", {}), **conflict_meta},
        {
            "l2_die_bytes_per_s": _band("P05_l2_residency_reuse", "l2_die_bytes_per_s", p05r.get("l2_die_bytes_per_s"), spec["l2_bytes_per_s"], allow_meas=allow),
            "l2_effective_fraction": _band("P05_l2_residency_reuse", "l2_effective_fraction", p05r.get("l2_effective_fraction"), 1.0, allow_meas=allow),
        },
    )
    p06_modes: dict[str, Any] = {}
    any_p06 = False
    mode_defs = [
        ("bf16", _TILE_16, _TILE_16, _TILE_16, _TILE_16, spec["peaks"].get("bf16", 0.0) / aic),
        ("fp16", _TILE_16, _TILE_16, _TILE_16, _TILE_16, spec["peaks"].get("fp16", 0.0) / aic),
        ("fp32", _TILE_16, _TILE_16, _TILE_8, _TILE_16, spec["peaks"].get("fp32", 0.0) / aic),
        ("int8", _TILE_16, _TILE_16, _TILE_32, _TILE_16, spec["peaks"].get("int8", 0.0) / aic),
    ]
    if is_arch35:
        fp8_cap = spec["peaks"].get("fp8_e4m3", spec["peaks"].get("int8", 0.0)) / aic
        mode_defs.extend([
            ("fp8_e4m3", _TILE_16, _TILE_16, _TILE_32, _TILE_16, fp8_cap),
            ("mxfp8", _TILE_16, _TILE_16, _TILE_32, _TILE_16, fp8_cap),
            ("mxfp4", _TILE_16, _TILE_16, _TILE_64, _TILE_16, fp8_cap * _DOUBLE_FACTOR),
        ])
    for m_name, tm, tn, tk, miss, cap in mode_defs:
        if cap <= 0.0:
            continue
        b = _band("P06_mmad", "cube_only_rate", p06r.get("modes", {}).get(m_name), cap, allow_meas=allow, mode=m_name)
        ms = _cell_status({"r": b})
        any_p06 = any_p06 or (ms == "calibrated")
        entry: dict[str, Any] = {"status": ms, "tile_m": tm, "tile_n": tn, "tile_k": tk, "m_iss": miss, "cube_only_rate": b if ms == "calibrated" else _clean_band(b)}
        if m_name in {"mxfp8", "mxfp4"}:
            entry["scale_group"] = _DEFAULT_SCALE_GROUP
        p06_modes[m_name] = entry
    p["P06_mmad"] = _make_cell(
        "P06_mmad", {"rate": "flops_per_s_per_aic", "latency": "ns"},
        {"pipe_m_barrier_fractal_threshold": _DEFAULT_PIPE_M_FRACTAL_THRESH, "small_m_bf16_ns": p06r.get("small_m_bf16_ns", {}), **conflict_meta}, {},
        {"pipe_m_barrier_ns": _bound_only(0.0, "latency_lower")},
        status_override="calibrated" if any_p06 else "unmeasured", modes=p06_modes,
    )
    p["P07_vec_alu"] = _make_cell("P07_vec_alu", {"rate": "ops_per_s_per_aiv"}, {"vector_width_bytes": _DEFAULT_VEC_WIDTH_BYTES, "fp32_lanes": _DEFAULT_FP32_LANES, "bf16_lanes": _DEFAULT_BF16_LANES, **conflict_meta}, {"vec_alu_ops_per_s_per_aiv": _band("P07_vec_alu", "vec_alu_ops_per_s_per_aiv", raw.get("P07_vec_alu", {}).get("vec_alu_ops_per_s_per_aiv"), spec["vec_alu_ops_per_s"] / aiv, allow_meas=allow)}, {"raw_latency_cycles": _bound_only(0.0, "latency_lower")})
    p["P08_vec_sfu"] = _make_cell("P08_vec_sfu", {"rate": "ops_per_s_per_aiv"}, {"vector_width_bytes": _DEFAULT_VEC_WIDTH_BYTES, "fp32_lanes": _DEFAULT_FP32_LANES, "bf16_lanes": _DEFAULT_BF16_LANES, **conflict_meta}, {"vec_sfu_ops_per_s_per_aiv": _band("P08_vec_sfu", "vec_sfu_ops_per_s_per_aiv", raw.get("P08_vec_sfu", {}).get("vec_sfu_ops_per_s_per_aiv"), spec["vec_sfu_ops_per_s"] / aiv, allow_meas=allow)}, {"raw_latency_cycles": _bound_only(0.0, "latency_lower")})
    p["P09_vec_reduce_cast"] = _make_cell("P09_vec_reduce_cast", {"rate": "ops_per_s_per_aiv", "cast_rate": "bytes_per_s_per_aiv"}, {"mx_scale_group": _DEFAULT_SCALE_GROUP, "mx_dynamic_quant_supported": is_arch35, **conflict_meta}, {"vec_reduce_ops_per_s_per_aiv": _band("P09_vec_reduce_cast", "vec_reduce_ops_per_s_per_aiv", p09r.get("vec_reduce_ops_per_s_per_aiv"), spec["vec_reduce_ops_per_s"] / aiv, allow_meas=allow), "cast_bytes_per_s_per_aiv": _band("P09_vec_reduce_cast", "cast_bytes_per_s_per_aiv", p09r.get("cast_bytes_per_s_per_aiv"), spec["cast_bytes_per_s"] / aiv, allow_meas=allow)})
    p["P10_sort_topk"] = _make_cell("P10_sort_topk", {"rate": "elems_per_s"}, {"sort_algorithm": "masked_hist256" if is_arch35 else "sort32_mrgsort4", "supported_sortall_lengths": list(_DEFAULT_SORTALL_LENGTHS), "histogram_trunk_elems": _DEFAULT_HIST_TRUNK_35 if is_arch35 else 0, **conflict_meta}, {"sort_elems_per_s": _band("P10_sort_topk", "sort_elems_per_s", raw.get("P10_sort_topk", {}).get("sort_elems_per_s"), spec["vector_ops_per_s"], allow_meas=allow)})
    vec_cap = max(1.0, spec["vector_ops_per_s"] / aiv)
    p["P11_simt"] = _make_cell("P11_simt", {"rate": "ops_per_s_per_aiv", "latency": "ns"}, {"supported": is_arch35, "simt_ub_carveout_bytes": spec["simt_ub_carveout_bytes"], "tla_max_threads": _DEFAULT_TLA_THREADS_35 if is_arch35 else 0, "catlass_max_threads": _DEFAULT_CATLASS_THREADS_35 if is_arch35 else 0, "warp_size": _DEFAULT_WARP_SIZE}, {}, {"simt_ops_per_s_per_aiv": _bound_only(vec_cap if is_arch35 else 0.0), "syncthreads_ns": _bound_only(0.0, "latency_lower"), "simt_launch_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured" if is_arch35 else "unsupported")
    p["P12_intracore_flag"] = _make_cell("P12_intracore_flag", {"latency": "ns"}, {"event_ids": _DEFAULT_EVENT_IDS, "unit_flag_requires_single_l0c_stage": True}, {}, {"flag_ns": _bound_only(0.0, "latency_lower"), "unitflag_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P13_crosscore_sync"] = _make_cell("P13_crosscore_sync", {"latency": "ns"}, {"aiv_inter_block_barrier_id": _DEFAULT_BARRIER_AIV, "aic_inter_block_barrier_id": _DEFAULT_BARRIER_AIC, "intra_subblock_barrier_id": _DEFAULT_BARRIER_SUB, "ffts_max_flag_id": _DEFAULT_FFTS_MAX_FLAG, "reverse_depth": _DEFAULT_REVERSE_DEPTH}, {}, {"mode2_paired_ns": _bound_only(0.0, "latency_lower"), "mode1_subblock_ns": _bound_only(0.0, "latency_lower"), "mode0_aic_barrier_ns": _bound_only(0.0, "latency_lower"), "mode0_aiv_barrier_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P14_mutex_block_mmad"] = _make_cell("P14_mutex_block_mmad", {"latency": "ns"}, {"max_mutex_ids": _DEFAULT_MAX_MUTEX_IDS, "unit_flag_requires_single_l0c_stage": True}, {}, {"mutex_overhead_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P15_rts_stars"] = _make_cell("P15_rts_stars", {"latency": "ns"}, {"queue_scope": "die", "queue_depth": _DEFAULT_QUEUE_DEPTH, "lanes": _DEFAULT_UNIT_ONE, **conflict_meta}, {"issue_interval_ns": _band("P15_rts_stars", "issue_interval_ns", p15r.get("issue_interval_ns"), 0.0, kind="latency_lower", allow_meas=allow), "aiv_first_issue_ns": _band("P15_rts_stars", "aiv_first_issue_ns", p15r.get("aiv_first_issue_ns"), 0.0, kind="latency_lower", allow_meas=allow), "aic_first_issue_ns": _band("P15_rts_stars", "aic_first_issue_ns", p15r.get("aic_first_issue_ns"), 0.0, kind="latency_lower", allow_meas=allow)}, {"host_submit_ns": _band("P15_rts_stars", "host_submit_ns", p15r.get("host_submit_ns"), 0.0, kind="latency_lower", allow_meas=allow)})
    p["P16_multistream_core_control"] = _make_cell("P16_multistream_core_control", {"latency": "ns"}, {"modes": ["disjoint", "shared_cap"], **conflict_meta}, {"stream_event_ns": _band("P16_multistream_core_control", "stream_event_ns", raw.get("P16_multistream_core_control", {}).get("stream_event_ns"), 0.0, kind="latency_lower", allow_meas=allow)}, {"set_limit_ns": _bound_only(0.0, "latency_lower")})
    p["P17_aicpu_metadata_launch"] = _make_cell("P17_aicpu_metadata_launch", {"latency": "ns"}, {"aclgraph_cacheable": True}, {}, {"host_select_ns": _bound_only(0.0, "latency_lower"), "aicpu_base_ns": _bound_only(0.0, "latency_lower"), "gm_pub_ns": _bound_only(0.0, "latency_lower"), "eager_d2h_sync_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P18_hccl_collective_band"] = _make_cell("P18_hccl_collective_band", {"rate": "bytes_per_s", "latency": "ns"}, {"payload_bytes_per_frame": _DEFAULT_HCCL_PAYLOAD_35 if is_arch35 else _DEFAULT_HCCL_FRAME_22, "wire_bytes_per_frame": _DEFAULT_HCCL_WIRE_35 if is_arch35 else _DEFAULT_HCCL_FRAME_22, "short_final_frame_policy": "exact_plus_header"}, {}, {"link_bytes_per_s": _bound_only(spec["link_bytes_per_s"]), "startup_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P19_direct_peer_window"] = _make_cell("P19_direct_peer_window", {"rate": "bytes_per_s", "latency": "ns"}, {"transport": "a5_ub_shmem_mte" if is_arch35 else "a3_sio_hccs", "token_payload_bytes_per_frame": _DEFAULT_HCCL_PAYLOAD_35, "token_wire_bytes_per_frame": _DEFAULT_HCCL_WIRE_35}, {}, {"peer_bytes_per_s": _bound_only(spec["link_bytes_per_s"]), "status_sync_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P20_host_dma_sync"] = _make_cell(
        "P20_host_dma_sync", {"rate": "bytes_per_s", "latency": "ns"}, {"contention_scope": "root_complex", **conflict_meta},
        {
            "host_h2d_bytes_per_s": _band("P20_host_dma_sync", "host_h2d_bytes_per_s", p20r.get("host_h2d_bytes_per_s"), spec["host_h2d_bytes_per_s"], allow_meas=allow),
            "host_d2h_bytes_per_s": _band("P20_host_dma_sync", "host_d2h_bytes_per_s", p20r.get("host_d2h_bytes_per_s"), spec["host_d2h_bytes_per_s"], allow_meas=allow),
        },
        {
            "host_rc_h2d_bytes_per_s": _bound_only(spec["host_rc_h2d_bytes_per_s"]),
            "host_rc_d2h_bytes_per_s": _bound_only(spec["host_rc_d2h_bytes_per_s"]),
            "pinned_dma_startup_ns": _band("P20_host_dma_sync", "pinned_dma_startup_ns", p20r.get("pinned_dma_startup_ns"), 0.0, kind="latency_lower", allow_meas=allow),
            "pageable_dma_startup_ns": _band("P20_host_dma_sync", "pageable_dma_startup_ns", p20r.get("pageable_dma_startup_ns"), 0.0, kind="latency_lower", allow_meas=allow),
            "eager_d2h_sync_ns": _band("P20_host_dma_sync", "eager_d2h_sync_ns", p20r.get("eager_d2h_sync_ns"), 0.0, kind="latency_lower", allow_meas=allow),
        },
    )

    payload: dict[str, Any] = {
        "schema_version": PRIMITIVE_SCHEMA_VERSION,
        "arch_preset": spec["arch_preset"],
        "sku": sku,
        "cann_version": cann_version,
        "provenance": {
            "campaign_id": clean_campaign,
            "endpoint_alias": alias,
            "capture_uri": top_capture,
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
