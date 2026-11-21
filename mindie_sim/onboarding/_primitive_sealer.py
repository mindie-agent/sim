"""SHA-256 primitive table sealing for Stage 2 SKU onboarding calibration.

[Role] Assemble and validate ``mindie-l0-primitives/1`` JSON tables across all
  20 primitives (``P01..P20``) from bounded L0 micro-benchmark and fabric/DMA
  samples, enforcing ``>= 3x L2`` ``ColdRing`` rotation, ``cv_pct <= 5.0%``,
  and ``certified_upper_capacity >= p99``.
[Invariants] Every sealed table passes ``parse_primitive_table(..., verify_envelope=True)``.
  Zero raw IPs, hostnames, usernames, or ``$HOME`` paths enter the envelope.
[Timeouts & Locks] Pure CPU statistics and SHA-256 hashing; no I/O or locks.
[Side-Effect Policy] Returns new dicts; never mutates hardware profiles.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import statistics
from typing import Any

from mindie_sim.backends import ascend
from mindie_sim.primitives.schema import (
    PRIMITIVE_SCHEMA_VERSION,
    compute_envelope_sha256,
    parse_primitive_table,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.hardware import Hardware

from ._soc_probe import ARCH_PRESET_35
from .contracts import (
    COLD_RING_L2_MULTIPLIER,
    DISPLAY_ROUND_DIGITS,
    GIB_BYTES,
    MAX_CALIBRATED_CV_PCT,
    PERCENT_SCALE,
    assert_sanitized_payload,
    build_capture_uri,
    canonical_json_bytes,
    sanitize_public_alias,
    utc_now_iso,
)

TRIM_FRACTION = 0.10
QUANTILE_P05 = 0.05
QUANTILE_P50 = 0.50
QUANTILE_P95 = 0.95
QUANTILE_P99 = 0.99
ROUND_METRIC_DIGITS = 6
SECTOR_BYTES = 32
FRACTAL_BYTES = 512
L2_GRANULE_BYTES = 128
VECTOR_WIDTH_BYTES = 256
FP32_LANES = 64
BF16_LANES = 128
MX_SCALE_GROUP = 32
STARS_QUEUE_DEPTH = 64
PIPE_M_BARRIER_THRESHOLD = 10
EVENT_ID_SLOTS = 8
AIV_BARRIER_ID = 8
AIC_BARRIER_ID = 9
SUBBLOCK_BARRIER_ID = 10
FFTS_MAX_FLAG_ID = 7
REVERSE_SYNC_DEPTH = 15
MAX_MUTEX_IDS = 28
FRAME_PAYLOAD_DIRECT = 480
FRAME_WIRE_DIRECT = 512
CUBE_TILE_M = 16
CUBE_TILE_N = 16
CUBE_TILE_K_BF16 = 16
CUBE_TILE_K_FP32 = 8
CUBE_TILE_K_INT8 = 32
CUBE_ISSUE_M = 16
SIMT_TLA_THREADS = 2048
SIMT_CATLASS_THREADS = 1024
SIMT_WARP_SIZE = 32
HISTOGRAM_TRUNK_ELEMS = 16384
SORTALL_LENGTHS = (64, 128, 256, 512, 1024, 2048)
DEFAULT_LINK_FALLBACK_BPS = 25_000_000_000.0
DEFAULT_HOST_DMA_BPS = 31_500_000_000.0
DEFAULT_HOST_RC_DMA_BPS = 200_000_000_000.0


def materialize_sku_hardware(
    sku: str,
    *,
    profile_override: Mapping[str, Any] | None = None,
) -> Hardware:
    """Materialize ``Hardware`` for a registered or freshly scaffolded SKU."""
    if profile_override is not None:
        entry = (
            profile_override.get(sku, profile_override)
            if isinstance(profile_override, Mapping)
            else {}
        )
        if "device" in entry:
            return Hardware.from_mapping({
                "sku": sku,
                "architecture": entry["architecture"],
                "dies_per_package": int(entry.get("dies_per_package", 1)),
                "ranks_per_node": max(1, int(entry.get("dies_per_package", 1))),
                "num_devices": 1,
                "device": deepcopy(entry["device"]),
                "transfer_channels": deepcopy(entry.get("transfer_channels", {})),
            })
    return Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))


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


def _summarize_band(
    samples: Sequence[float] | None,
    *,
    certified_bound: float,
    bound_kind: str = "rate_upper",
) -> dict[str, Any]:
    nom = round(float(certified_bound), ROUND_METRIC_DIGITS)
    if not samples:
        return {"certified_upper_capacity": nom, "bound_kind": bound_kind}
    clean = sorted(float(x) for x in samples if math.isfinite(float(x)) and float(x) >= 0.0)
    if not clean:
        return {"certified_upper_capacity": nom, "bound_kind": bound_kind}
    n = len(clean)
    mean_v = sum(clean) / n
    cv_pct = (PERCENT_SCALE * statistics.stdev(clean) / mean_v) if (n >= 2 and mean_v > 0.0) else 0.0
    trim_k = int(math.floor(n * TRIM_FRACTION))
    trimmed = clean[trim_k : n - trim_k] if n - 2 * trim_k >= 2 else clean
    t_mean = sum(trimmed) / len(trimmed)
    p05 = _quantile(clean, QUANTILE_P05)
    p50 = _quantile(clean, QUANTILE_P50)
    p95 = _quantile(clean, QUANTILE_P95)
    p99 = _quantile(clean, QUANTILE_P99)
    exceeded = (
        (p99 > float(certified_bound) or p05 <= 0.0)
        if bound_kind == "rate_upper"
        else (p05 < float(certified_bound))
    )
    if exceeded or cv_pct > MAX_CALIBRATED_CV_PCT:
        return {"certified_upper_capacity": nom, "bound_kind": bound_kind}
    return {
        "p05": round(p05, ROUND_METRIC_DIGITS),
        "p50": round(p50, ROUND_METRIC_DIGITS),
        "p95": round(p95, ROUND_METRIC_DIGITS),
        "p99": round(p99, ROUND_METRIC_DIGITS),
        "trimmed_mean": round(t_mean, ROUND_METRIC_DIGITS),
        "cv_pct": round(cv_pct, DISPLAY_ROUND_DIGITS),
        "n_samples": n,
        "certified_upper_capacity": nom,
        "bound_kind": bound_kind,
    }


def _bound_only(value: float, bound_kind: str = "rate_upper") -> dict[str, Any]:
    return {"certified_upper_capacity": round(float(value), ROUND_METRIC_DIGITS), "bound_kind": bound_kind}


def seal_primitive_table(
    *,
    sku: str,
    cann_version: str,
    endpoint_alias: str,
    campaign_id: str,
    raw_l0: Mapping[str, Any] | None,
    raw_fabric: Mapping[str, Any] | None,
    profile_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Build, validate, and SHA-256-seal a complete ``P01..P20`` primitive table."""
    hw = materialize_sku_hardware(sku, profile_override=profile_override)
    die = hw.dies[0]
    dp = datapath_for_die(die)
    bufs = dict(die.buffers)
    rates = dict(die.rates)
    tc = {k: dict(v) for k, v in hw.transfer_channels}
    aic, aiv = max(1, int(dp.aic_count)), max(1, int(dp.aiv_count))
    is_arch35 = str(dp.arch_id) == ARCH_PRESET_35
    l0 = dict(raw_l0 or {})
    fab = dict(raw_fabric or {})

    sample_dev = str(l0.get("device_name") or "").strip()
    if sample_dev and sample_dev != sku:
        raise ModelingError(f"silicon_profile_conflict:sku_mismatch:expected={sku}:got={sample_dev}")
    cold_mult = l0.get("cold_ring_l2_multiplier")
    if cold_mult is not None and int(cold_mult) < COLD_RING_L2_MULTIPLIER:
        raise ModelingError(f"cold_ring_l2_multiplier_too_small:{cold_mult}")

    clean_alias = sanitize_public_alias(endpoint_alias or sku, sku)
    cap_body = {"sku": sku, "l0": l0, "fabric": fab}
    assert_sanitized_payload(cap_body, field_name="l0_capture_body")
    capture_uri, _ = build_capture_uri(clean_alias, campaign_id, canonical_json_bytes(cap_body))

    hbm_r = float(dp.hbm_read_bytes_per_s or rates.get("hbm", 0.0))
    hbm_w = float(dp.hbm_write_bytes_per_s or rates.get("hbm", 0.0))
    l2_bps = float(rates.get("l2", rates.get("mte2_aiv", 0.0) * 2.0))
    mte1_bps = float(rates.get("mte1", 0.0))
    mte2_aic_bps = float(rates.get("mte2_aic", 0.0))
    mte2_aiv_bps = float(rates.get("mte2_aiv", 0.0))
    mte3_bps = float(rates.get("mte3", 0.0))
    fixp_bps = float(rates.get("fixpipe", 0.0))
    vec_bps = float(rates.get("vector", 0.0))
    h2d, d2h, d2d = tc.get("h2d", {}), tc.get("d2h", {}), tc.get("d2d_intra_node", {})
    link_bps = float(d2d.get("uni_peak_bw_gbps") or d2d.get("effective_uni_bw_gbps") or 0.0) * 1e9 or DEFAULT_LINK_FALLBACK_BPS
    h2d_bps = float(h2d.get("solo_effective_uni_bw_gbps") or h2d.get("uni_peak_bw_gbps") or 0.0) * 1e9 or DEFAULT_HOST_DMA_BPS
    d2h_bps = float(d2h.get("solo_effective_uni_bw_gbps") or d2h.get("uni_peak_bw_gbps") or 0.0) * 1e9 or DEFAULT_HOST_DMA_BPS
    h2d_rc = float(h2d.get("node_peak_bw_gbps") or 0.0) * 1e9 or DEFAULT_HOST_RC_DMA_BPS
    d2h_rc = float(d2h.get("node_peak_bw_gbps") or 0.0) * 1e9 or DEFAULT_HOST_RC_DMA_BPS

    def _cell(
        units: dict[str, str],
        params: dict[str, Any],
        core: dict[str, dict[str, Any]],
        extra: dict[str, dict[str, Any]] | None = None,
        *,
        status_override: str | None = None,
        modes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status_override is not None:
            st = status_override
        elif core and all(b.get("p50") is not None for b in core.values()):
            st = "calibrated"
        else:
            st = "unmeasured"
        return {
            "status": st,
            "units": units,
            "parameters": params,
            "metrics": {**core, **(extra or {})},
            "modes": modes or {},
            "evidence_id": capture_uri if st == "calibrated" else "",
        }

    p01r, p04r, p05r, p06r = (l0.get(k, {}) for k in ("P01_datacopy", "P04_mte3_store", "P05_l2_residency_reuse", "P06_mmad"))
    p07r, p08r, p09r, p10r = (l0.get(k, {}) for k in ("P07_vec_alu", "P08_vec_sfu", "P09_vec_reduce_cast", "P10_sort_topk"))
    p15r, p16r = l0.get("P15_rts_stars", {}), l0.get("P16_multistream_core_control", {})
    p18r = fab.get("P18_hccl_collective_band", l0.get("P18_hccl_collective_band", {}))
    p19r = fab.get("P19_direct_peer_window", l0.get("P19_direct_peer_window", {}))
    p20r = fab.get("P20_host_dma_sync", l0.get("P20_host_dma_sync", {}))

    p: dict[str, Any] = {}
    p["P01_datacopy"] = _cell(
        {"rate": "bytes_per_s", "alpha": "dimensionless"},
        {"sector_bytes": SECTOR_BYTES, "l2_cache_ctl_supported": is_arch35},
        {
            "hbm_read_bytes_per_s": _summarize_band(p01r.get("hbm_read_bytes_per_s"), certified_bound=hbm_r),
            "hbm_write_bytes_per_s": _summarize_band(p01r.get("hbm_write_bytes_per_s"), certified_bound=hbm_w),
            "mte2_aiv_hit_bytes_per_s_per_core": _summarize_band(p01r.get("mte2_aiv_hit_bytes_per_s_per_core"), certified_bound=mte2_aiv_bps / aiv),
        },
        {
            "hbm_rw_alpha": _summarize_band(p01r.get("hbm_rw_alpha"), certified_bound=0.0, bound_kind="latency_lower"),
            "mte2_aic_hit_bytes_per_s_per_core": _bound_only(mte2_aic_bps / aic),
        },
    )
    p["P02_load_data"] = _cell(
        {"rate": "bytes_per_s_per_aic", "latency": "ns"},
        {"fractal_bytes": FRACTAL_BYTES, "mx_scale_supported": is_arch35}, {},
        {"l1_to_l0a_bytes_per_s_per_aic": _bound_only(mte1_bps / aic), "l1_to_l0b_bytes_per_s_per_aic": _bound_only(mte1_bps / aic), "setup_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured",
    )
    p["P03_fixpipe"] = _cell(
        {"rate": "bytes_per_s_per_aic", "latency": "ns"},
        {"has_l0c_to_ub": bool(dp.has_l0c_to_ub), "has_l0c_to_l1": True}, {},
        {"fixpipe_in_bytes_per_s_per_aic": _bound_only(fixp_bps / aic), "fixpipe_out_bytes_per_s_per_aic": _bound_only(fixp_bps / aic), "setup_ns": _bound_only(0.0, "latency_lower")},
        status_override="unmeasured",
    )
    p["P04_mte3_store"] = _cell(
        {"rate": "bytes_per_s", "latency": "ns"}, {"has_ub_to_l1": bool(dp.has_ub_to_l1)},
        {
            "mte3_store_bytes_per_s_per_aiv": _summarize_band(p04r.get("mte3_warm_bytes_per_s_per_aiv"), certified_bound=mte3_bps / aiv),
            "atomic_rmw_bytes_per_s": _summarize_band(p04r.get("atomic_rmw_bytes_per_s"), certified_bound=hbm_w),
        },
    )
    p["P05_l2_residency_reuse"] = _cell(
        {"rate": "bytes_per_s", "fraction": "dimensionless"},
        {"l2_bytes": int(bufs.get("l2", 0)), "granule_bytes": L2_GRANULE_BYTES, "buffer_units_l2": 1},
        {
            "l2_die_bytes_per_s": _summarize_band(p05r.get("l2_die_bytes_per_s"), certified_bound=l2_bps),
            "l2_effective_fraction": _summarize_band(p05r.get("l2_effective_fraction"), certified_bound=1.0),
        },
    )
    p06_modes: dict[str, Any] = {}
    any_p06 = False
    mode_specs = [
        ("bf16", CUBE_TILE_M, CUBE_TILE_N, CUBE_TILE_K_BF16),
        ("fp16", CUBE_TILE_M, CUBE_TILE_N, CUBE_TILE_K_BF16),
        ("fp32", CUBE_TILE_M, CUBE_TILE_N, CUBE_TILE_K_FP32),
        ("int8", CUBE_TILE_M, CUBE_TILE_N, CUBE_TILE_K_INT8),
    ]
    if is_arch35:
        mode_specs.append(("fp8_e4m3", CUBE_TILE_M, CUBE_TILE_N, CUBE_TILE_K_INT8))
    for m_name, tm, tn, tk in mode_specs:
        try:
            cap_per_aic = float(die.peak(m_name)) / aic
        except ModelingError:
            continue
        band = _summarize_band(p06r.get("modes", {}).get(m_name), certified_bound=cap_per_aic)
        ms = "calibrated" if band.get("p50") is not None else "unmeasured"
        any_p06 = any_p06 or (ms == "calibrated")
        p06_modes[m_name] = {"status": ms, "tile_m": tm, "tile_n": tn, "tile_k": tk, "m_iss": CUBE_ISSUE_M, "cube_only_rate": band}
    p["P06_mmad"] = _cell(
        {"rate": "flops_per_s_per_aic", "latency": "ns"},
        {"pipe_m_barrier_fractal_threshold": PIPE_M_BARRIER_THRESHOLD}, {},
        {"pipe_m_barrier_ns": _bound_only(0.0, "latency_lower")},
        status_override="calibrated" if any_p06 else "unmeasured", modes=p06_modes,
    )
    alu_cap = float(rates.get("reg_alu_lanes", vec_bps * 2.0)) / aiv
    sfu_cap = float(rates.get("reg_special_lanes", vec_bps)) / aiv
    red_cap = float(rates.get("reg_reduce_lanes", vec_bps)) / aiv
    cast_cap = float(rates.get("ub_write_bytes", mte2_aiv_bps)) / aiv
    p["P07_vec_alu"] = _cell({"rate": "ops_per_s_per_aiv"}, {"vector_width_bytes": VECTOR_WIDTH_BYTES, "fp32_lanes": FP32_LANES, "bf16_lanes": BF16_LANES}, {"vec_alu_ops_per_s_per_aiv": _summarize_band(p07r.get("vec_alu_ops_per_s_per_aiv"), certified_bound=alu_cap)})
    p["P08_vec_sfu"] = _cell({"rate": "ops_per_s_per_aiv"}, {"vector_width_bytes": VECTOR_WIDTH_BYTES, "fp32_lanes": FP32_LANES, "bf16_lanes": BF16_LANES}, {"vec_sfu_ops_per_s_per_aiv": _summarize_band(p08r.get("vec_sfu_ops_per_s_per_aiv"), certified_bound=sfu_cap)})
    p["P09_vec_reduce_cast"] = _cell(
        {"rate": "ops_per_s_per_aiv", "cast_rate": "bytes_per_s_per_aiv"},
        {"mx_scale_group": MX_SCALE_GROUP, "mx_dynamic_quant_supported": is_arch35},
        {
            "vec_reduce_ops_per_s_per_aiv": _summarize_band(p09r.get("vec_reduce_ops_per_s_per_aiv"), certified_bound=red_cap),
            "cast_bytes_per_s_per_aiv": _summarize_band(p09r.get("cast_bytes_per_s_per_aiv"), certified_bound=cast_cap),
        },
    )
    p["P10_sort_topk"] = _cell(
        {"rate": "elems_per_s"},
        {"sort_algorithm": "masked_hist256" if is_arch35 else "sort32_mrgsort4", "supported_sortall_lengths": list(SORTALL_LENGTHS), "histogram_trunk_elems": HISTOGRAM_TRUNK_ELEMS if is_arch35 else 0},
        {"sort_elems_per_s": _summarize_band(p10r.get("sort_elems_per_s"), certified_bound=vec_bps)},
    )
    p["P11_simt"] = _cell(
        {"rate": "ops_per_s_per_aiv", "latency": "ns"},
        {"supported": is_arch35, "simt_ub_carveout_bytes": int(dp.simt_ub_carveout_bytes), "tla_max_threads": SIMT_TLA_THREADS if is_arch35 else 0, "catlass_max_threads": SIMT_CATLASS_THREADS if is_arch35 else 0, "warp_size": SIMT_WARP_SIZE},
        {}, {"simt_ops_per_s_per_aiv": _bound_only(max(1.0, vec_bps / aiv) if is_arch35 else 0.0)},
        status_override="unmeasured" if is_arch35 else "unsupported",
    )
    p["P12_intracore_flag"] = _cell({"latency": "ns"}, {"event_ids": EVENT_ID_SLOTS}, {}, {"flag_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P13_crosscore_sync"] = _cell({"latency": "ns"}, {"aiv_inter_block_barrier_id": AIV_BARRIER_ID, "aic_inter_block_barrier_id": AIC_BARRIER_ID, "intra_subblock_barrier_id": SUBBLOCK_BARRIER_ID, "ffts_max_flag_id": FFTS_MAX_FLAG_ID, "reverse_depth": REVERSE_SYNC_DEPTH}, {}, {"mode2_paired_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P14_mutex_block_mmad"] = _cell({"latency": "ns"}, {"max_mutex_ids": MAX_MUTEX_IDS}, {}, {"mutex_overhead_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P15_rts_stars"] = _cell(
        {"latency": "ns"}, {"queue_scope": "die", "queue_depth": STARS_QUEUE_DEPTH, "lanes": 1},
        {
            "issue_interval_ns": _summarize_band(p15r.get("issue_interval_ns"), certified_bound=0.0, bound_kind="latency_lower"),
            "aiv_first_issue_ns": _summarize_band(p15r.get("aiv_first_issue_ns"), certified_bound=0.0, bound_kind="latency_lower"),
            "aic_first_issue_ns": _summarize_band(p15r.get("aic_first_issue_ns"), certified_bound=0.0, bound_kind="latency_lower"),
        },
        {"host_submit_ns": _summarize_band(p15r.get("host_submit_ns"), certified_bound=0.0, bound_kind="latency_lower")},
    )
    p["P16_multistream_core_control"] = _cell({"latency": "ns"}, {"modes": ["disjoint", "shared_cap"]}, {"stream_event_ns": _summarize_band(p16r.get("stream_event_ns"), certified_bound=0.0, bound_kind="latency_lower")}, {"set_limit_ns": _bound_only(0.0, "latency_lower")})
    p["P17_aicpu_metadata_launch"] = _cell({"latency": "ns"}, {"aclgraph_cacheable": True}, {}, {"host_select_ns": _bound_only(0.0, "latency_lower")}, status_override="unmeasured")
    p["P18_hccl_collective_band"] = _cell(
        {"rate": "bytes_per_s", "latency": "ns"},
        {"payload_bytes_per_frame": FRAME_PAYLOAD_DIRECT if is_arch35 else 1024, "wire_bytes_per_frame": FRAME_WIRE_DIRECT if is_arch35 else 1024, "short_final_frame_policy": "exact_plus_header"},
        {"link_bytes_per_s": _summarize_band(p18r.get("link_bytes_per_s"), certified_bound=link_bps)},
        {"startup_ns": _summarize_band(p18r.get("startup_ns"), certified_bound=0.0, bound_kind="latency_lower")},
    )
    p["P19_direct_peer_window"] = _cell(
        {"rate": "bytes_per_s", "latency": "ns"}, {"transport": "ub_shmem_mte" if is_arch35 else "sio_hccs"},
        {"peer_bytes_per_s": _summarize_band(p19r.get("peer_bytes_per_s"), certified_bound=link_bps)},
        {"status_sync_ns": _summarize_band(p19r.get("status_sync_ns"), certified_bound=0.0, bound_kind="latency_lower")},
    )
    p["P20_host_dma_sync"] = _cell(
        {"rate": "bytes_per_s", "latency": "ns"}, {"contention_scope": "root_complex"},
        {
            "host_h2d_bytes_per_s": _summarize_band(p20r.get("host_h2d_bytes_per_s"), certified_bound=h2d_bps),
            "host_d2h_bytes_per_s": _summarize_band(p20r.get("host_d2h_bytes_per_s"), certified_bound=d2h_bps),
        },
        {
            "host_rc_h2d_bytes_per_s": _bound_only(h2d_rc),
            "host_rc_d2h_bytes_per_s": _bound_only(d2h_rc),
            "pinned_dma_startup_ns": _summarize_band(p20r.get("pinned_dma_startup_ns"), certified_bound=0.0, bound_kind="latency_lower"),
        },
    )
    payload: dict[str, Any] = {
        "schema_version": PRIMITIVE_SCHEMA_VERSION,
        "arch_preset": str(dp.arch_id),
        "sku": sku,
        "cann_version": cann_version,
        "provenance": {
            "campaign_id": campaign_id,
            "endpoint_alias": clean_alias,
            "capture_uri": capture_uri,
            "calibrated": any(c["status"] == "calibrated" for c in p.values()),
            "collected_utc": utc_now_iso(),
            "silicon_fingerprint": {
                "aic_count": aic, "aiv_count": aiv,
                "hbm_gib": max(1, int(die.memory_bytes // GIB_BYTES)),
                "l2_bytes": int(bufs.get("l2", 0)), "ub_bytes": int(bufs.get("ub", 0)),
                "l1_bytes": int(bufs.get("l1", 0)), "l0a_bytes": int(bufs.get("l0a", 0)),
                "l0b_bytes": int(bufs.get("l0b", 0)), "l0c_bytes": int(bufs.get("l0c", 0)),
            },
        },
        "primitives": p,
    }
    payload["envelope_sha256"] = compute_envelope_sha256(payload)
    parse_primitive_table(payload, verify_envelope=True)
    return payload, capture_uri


__all__ = ["materialize_sku_hardware", "seal_primitive_table"]
