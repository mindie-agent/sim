#!/usr/bin/env python3
"""Offline Ascend A2/A3/A5 NPU L0 micro-benchmark calibration suite.

[Role] Plane-1 offline calibration producer executing bounded synthetic NPU
  micro-benchmarks on idle physical Ascend A2/A3/A5 dies via endpoint alias
  and emitting full-envelope SHA-256 sealed ``data/primitives/*.json`` tables
  plus ``data/primitives/manifest.json`` with honest SKU x CANN x primitive x
  mode x metric coverage.
[Invariants] Strictly offline Plane-1 tool outside ``mindie_sim/``. Checks
  ``npu-smi info`` before touching any die (including peer die for HCCL
  P18/P19); caps HBM <= 4 GiB; enforces cv_pct <= 5.0% and
  certified_upper_capacity >= p99; surfaces any p99 > nominal conflict loudly
  in ``metric_diagnostics`` without silently inflating bounds; never emits
  raw IPs, hostnames, usernames, or host $HOME paths.
[Timeouts & Locks] Bounded SSH/remote subprocess execution with unconditional
  remote temporary directory cleanup on exit; targets idle NPU dies only.
[Side-Effect Policy] Writes only the requested ``--out`` JSON table, sibling
  ``captures/*.capture.json``, and ``--manifest`` index.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any

_PARENTS = Path(__file__).resolve().parents
REPO_ROOT = _PARENTS[1] if len(_PARENTS) > 1 else _PARENTS[0]
if (REPO_ROOT / "mindie_sim").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_l0_bounds import (
    MAX_CALIBRATED_CV_PCT,
    ROUND_METRIC_DIGITS,
    SIMT_UB_CARVEOUT_MIN_BYTES,
    SIMT_WARP_SCHEDULERS,
    TWO,
    apply_core_clock_scale as _apply_core_clock_scale,
    derive_burst_turn_rules as _derive_burst_turn_rules,
    load_sku_hardware_spec as _load_sku_hardware_spec,
    summarize_metric,
    summarize_samples,
)
from _bench_l0_probes import DEFAULT_REPEATS, DEFAULT_WARMUP, RESULT_MARKER, run_local_npu_microbenchmarks
from _bench_l0_remote import (
    JSON_INDENT,
    PRIMITIVE_SCHEMA_VERSION,
    assert_no_privacy_leak,
    canonical_capture_bytes,
    check_idle_die as _check_idle_die,
    check_local_idle_die as _check_local_idle_die,
    coverage_summary as _coverage_summary,
    load_endpoint as _load_endpoint,
    parse_npu_smi_idle_die,
    run_remote_calibration as _run_remote_calibration,
    sanitize_capture_obj as _sanitize_capture_obj,
    sanitize_record_alias,
    ssh_run as _ssh_run,
    unwrap_raw_measurements,
    update_manifest,
    validate_quantitative_curves,
)

SECTOR_BYTES = 32
FRACTAL_BYTES = 512
UB_SPLIT_N_ALIGN = 32
L2_LINE_BYTES = 512
ARCH22_L2_GRANULE_BYTES = 512
ARCH35_L2_GRANULE_BYTES = 128
ARCH22_HBM_BURST_GRANULE_BYTES = 512
ARCH35_HBM_BURST_GRANULE_BYTES = 256
ARCH35_HBM_SCATTER_ROW_OPEN_BYTES = 2048
FIXPIPE_IN_TO_OUT_RATIO = 2.0
ARCH35_NZ2DN_STRIDE_GT1_DIV = 4
ARCH35_TRANSPOSE_L0B_FACTOR = 0.5
ARCH35_MX_SCALE_BYTES_PER_CYC = 32
HCCL_TWO_RANK_GROUP = 2
ADMISSIBLE_ALPHA_MAX = 1.0
TILE_1 = 1
TILE_4 = 4
TILE_8 = 8
TILE_16 = 16
TILE_32 = 32
TILE_64 = 64
DAV22_FP32_RATIO_VS_BF16 = 4.0
DAV35_FP32_RATIO_VS_BF16 = 16.0
HF32_RATIO_VS_BF16 = 2.0
PIPE_M_BARRIER_FRACTALS = 10
VEC_WIDTH_BYTES = 256
FP32_LANES = 64
BF16_LANES = 128
DAV22_UB_BANK_GROUPS = 16
DAV22_UB_BANKS_PER_GROUP = 3
DAV22_UB_BANK_SIZE_BYTES = 4096
DAV35_UB_BANK_GROUPS = 8
DAV35_UB_BANKS_PER_GROUP = 2
DAV35_UB_BANK_SIZE_BYTES = 16384
MX_SCALE_GROUP = 32
ARCH35_HIST_TRUNK = 16384
SORTALL_LENGTHS = (64, 128, 256, 512, 1024, 2048)
SORT_PROBE_LENGTH = 1024
SORT_PROBE_PASSES = 4
TLA_MAX_THREADS = 2048
CATLASS_MAX_THREADS = 1024
SIMT_WARP_SIZE = 32
EVENT_IDS_COUNT = 8
AIV_BARRIER_ID = 8
AIC_BARRIER_ID = 9
INTRA_SUBBLOCK_BARRIER_ID = 10
FFTS_MAX_FLAG_ID = 7
FFTS_HW_FLAG_IDS = 16
FFTS_REVERSE_DEPTH = 15
HW_MUTEX_IDS = 32
MAX_MUTEX_IDS = 28
STARS_QUEUE_DEPTH = 64
ARCH35_FRAME_PAYLOAD = 480
ARCH35_FRAME_WIRE = 512
ARCH22_FRAME_BYTES = 1024
DEFAULT_L2_ARG = 134217728
DEFAULT_AIC_ARG = 32
DEFAULT_AIV_ARG = 64
ARCH22_CORE_MODES = ("bf16", "fp16", "fp32", "int8")
ARCH35_EXTRA_MODES = ("fp8_e4m3", "fp8_e5m2", "hifloat8", "mxfp8", "mxfp4")

SKU_DEFAULT_FILES = {
    "Ascend910B4": "arch22_a2_910b4.json",
    "Ascend910_9362": "arch22_a3_9362.json",
    "Ascend910_9382": "arch22_a3_9382.json",
    "Ascend950PR_9579": "arch35_a5_950pr.json",
    "Ascend950DT_9582": "arch35_a5_950dt.json",
}


def compute_envelope_sha256(payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "envelope_sha256"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cell_status(bands: dict[str, dict[str, Any]]) -> str:
    if not bands:
        return "unmeasured"
    for b in bands.values():
        if b.get("p50") is None or b.get("cv_pct") is None or float(b["cv_pct"]) > MAX_CALIBRATED_CV_PCT:
            return "unmeasured"
    return "calibrated"


def _clean_band(b: dict[str, Any]) -> dict[str, Any]:
    if b.get("p50") is not None and b.get("cv_pct") is not None and float(b["cv_pct"]) <= MAX_CALIBRATED_CV_PCT:
        return b
    return {"certified_upper_capacity": b["certified_upper_capacity"], "bound_kind": b.get("bound_kind", "rate_upper")}


def _bound_only(value: float, kind: str = "rate_upper") -> dict[str, Any]:
    return {"certified_upper_capacity": round(float(value), ROUND_METRIC_DIGITS), "bound_kind": kind}


def _peer_transport_name(dies_per_package: int, is_arch35: bool) -> str:
    if is_arch35:
        return "a5_ub_shmem_mte"
    return "a3_sio_hccs" if dies_per_package >= 2 else "a2_hccs_mesh"


def _filter_p01_scaling_curve(
    raw_curve: Any, *, hbm_cap: float, mte2_cap: float, warm: bool, allow_core: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Validate active-core scaling points against certified bounds and profile consistency."""
    if not isinstance(raw_curve, dict) or not raw_curve:
        return {}, {}
    if not allow_core:
        return {}, {"status": "quarantined_silicon_profile_conflict"}
    eps = 1.0 + 1e-6
    clean: dict[str, float] = {}
    for k, v in raw_curve.items():
        if not str(k).isdigit() or not isinstance(v, (int, float)) or isinstance(v, bool):
            return {}, {"status": "quarantined_invalid_entry", "key": str(k)}
        kc, fv = int(k), float(v)
        if kc < 1 or not math.isfinite(fv) or fv <= 0.0:
            return {}, {"status": "quarantined_nonpositive", "key": str(k)}
        if not warm and hbm_cap > 0.0 and fv > hbm_cap * eps:
            return {}, {"status": "quarantined_exceeds_hbm_capacity", "key": str(k), "observed": fv, "capacity": hbm_cap}
        if mte2_cap > 0.0 and fv > kc * mte2_cap * eps:
            return {}, {"status": "quarantined_exceeds_mte2_capacity", "key": str(k), "observed": fv, "capacity": kc * mte2_cap}
        clean[str(kc)] = fv
    return clean, {}


def _make_cell(
    units: dict[str, str], params: dict[str, Any],
    core: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    extra: dict[str, tuple[dict[str, Any], dict[str, Any]]] | None = None,
    *, top_capture: str, status_override: str | None = None,
    modes: dict[str, Any] | None = None, unmeasured_reason: str = "",
) -> dict[str, Any]:
    core_bands = {k: pair[0] for k, pair in core.items()}
    diags = {k: pair[1] for k, pair in core.items()}
    st = status_override if status_override is not None else _cell_status(core_bands)
    merged = {k: (v if st == "calibrated" else _clean_band(v)) for k, v in core_bands.items()}
    for k, (eb, ed) in (extra or {}).items():
        merged[k] = _clean_band(eb)
        diags[k] = ed
    p_out = dict(params)
    if diags:
        p_out["metric_diagnostics"] = diags
    if st == "unmeasured" and unmeasured_reason:
        p_out["unmeasured_reason"] = unmeasured_reason
    return {
        "status": st, "units": units, "parameters": p_out, "metrics": merged,
        "modes": modes or {}, "evidence_id": top_capture if st == "calibrated" else "",
    }


def _build_memory_primitives(
    ctx: dict[str, Any], raw: dict[str, Any],
) -> dict[str, Any]:
    spec, aic, aiv, is_arch35 = ctx["spec"], ctx["aic"], ctx["aiv"], ctx["is_arch35"]
    _m, _u, _cell = ctx["m"], ctx["u"], ctx["cell"]
    conflict_meta, ini_meta, unm_hw = ctx["conflict_meta"], ctx["ini_meta"], ctx["unm_hw"]
    p01r, p02r, p03r, p04r, p05r = (
        raw.get(k, {}) for k in ("P01_datacopy", "P02_load_data", "P03_fixpipe", "P04_mte3_store", "P05_l2_residency_reuse")
    )
    hbm_gran = ARCH35_HBM_BURST_GRANULE_BYTES if is_arch35 else ARCH22_HBM_BURST_GRANULE_BYTES
    l2_gran = ARCH35_L2_GRANULE_BYTES if is_arch35 else ARCH22_L2_GRANULE_BYTES
    cache_mech = "va_bits_62_60" if is_arch35 else "l2_cache_alter_dual_va"
    rd_samples = p01r.get("hbm_read_bytes_per_s") or []
    peak_rd = statistics.median(rd_samples) if rd_samples else spec["hbm_read_bytes_per_s"]
    bt_rules = _derive_burst_turn_rules(
        p01r.get("strided_cold_surface"), peak_read_bps=peak_rd, granule_bytes=hbm_gran, evidence_id=ctx["top_capture"],
    )
    fixp_out_aic = spec["fixpipe_bytes_per_s"] / aic
    fixp_in_aic = fixp_out_aic * FIXPIPE_IN_TO_OUT_RATIO
    idx_upd_band, idx_upd_diag = _m(p04r.get("indexed_update_bytes_per_s"), spec["l2_bytes_per_s"])
    mte2_aic_cap = spec["mte2_aic_bytes_per_s"] / aic
    cold_curve, cold_diag = _filter_p01_scaling_curve(
        p01r.get("mte2_aic_cold_scaling_bps"),
        hbm_cap=spec["hbm_read_bytes_per_s"], mte2_cap=mte2_aic_cap, warm=False, allow_core=ctx["allow_core"],
    )
    warm_curve, warm_diag = _filter_p01_scaling_curve(
        p01r.get("mte2_aic_warm_scaling_bps"),
        hbm_cap=spec["hbm_read_bytes_per_s"], mte2_cap=mte2_aic_cap, warm=True, allow_core=ctx["allow_core"],
    )
    curve_diags = {k: v for k, v in (("cold", cold_diag), ("warm", warm_diag)) if v}
    return {
        "P01_datacopy": _cell(
            {"rate": "bytes_per_s", "penalty": "bytes", "alpha": "dimensionless"},
            {"sector_bytes": SECTOR_BYTES, "hbm_burst_granule_bytes": hbm_gran,
             "hbm_scatter_row_open_bytes": ARCH35_HBM_SCATTER_ROW_OPEN_BYTES if is_arch35 else 0,
             "l2_cache_ctl_supported": True, "l2_cache_ctl_mechanism": cache_mech,
             "burst_turn_penalty": bt_rules, "strided_cold_surface": p01r.get("strided_cold_surface", []),
             "gm_alignment_cold_bps": p01r.get("gm_alignment_cold_bps", {}),
             "hbm_read_datacopy_blocked_bps_p50": p01r.get("hbm_read_datacopy_blocked_bps_p50"),
             "payload_ramp_bytes_per_s": p01r.get("payload_ramp_bytes_per_s", {}),
             "gather_v2_cold_surface": p01r.get("gather_v2_cold_surface", {}),
             "mte2_aic_cold_scaling_bps": cold_curve,
             "mte2_aic_warm_scaling_bps": warm_curve,
             **({"scaling_curve_diagnostics": curve_diags} if curve_diags else {}),
             **ini_meta, **conflict_meta},
            {"hbm_read_bytes_per_s": _m(p01r.get("hbm_read_bytes_per_s"), spec["hbm_read_bytes_per_s"]),
             "hbm_write_bytes_per_s": _m(p01r.get("hbm_write_bytes_per_s"), spec["hbm_write_bytes_per_s"]),
             "mte2_aiv_hit_bytes_per_s_per_core": _m(p01r.get("mte2_aiv_hit_bytes_per_s_per_core"), spec["mte2_aiv_bytes_per_s"] / aiv, per_core=True)},
            {"hbm_rw_alpha": _m(p01r.get("hbm_rw_alpha"), 0.0, "latency_lower", max_admissible=ADMISSIBLE_ALPHA_MAX, allow_zero=True),
             "hbm_rw11_bytes_per_s": _m(p01r.get("hbm_rw11_bytes_per_s"), spec["hbm_read_bytes_per_s"]),
             "hbm_rw21_bytes_per_s": _m(p01r.get("hbm_rw21_bytes_per_s"), spec["hbm_read_bytes_per_s"]),
             "mte2_aic_hit_bytes_per_s_per_core": _m(p01r.get("mte2_aic_hit_bytes_per_s_per_core"), mte2_aic_cap, per_core=True)},
            unmeasured_reason=unm_hw,
        ),
        "P02_load_data": _cell(
            {"rate": "bytes_per_s_per_aic", "latency": "ns"},
            {"fractal_bytes": FRACTAL_BYTES, "mx_scale_supported": is_arch35,
             "mx_scale_bytes_per_cyc": ARCH35_MX_SCALE_BYTES_PER_CYC if is_arch35 else 0,
             "transpose_factor": max(1.0, float(p02r.get("transpose_factor", 1.0))),
             "measured_l0a_transpose_ratio": p02r.get("measured_l0a_transpose_ratio"),
             "measured_l0b_transpose_ratio": p02r.get("measured_l0b_transpose_ratio"),
             "transpose_l0b_factor": ARCH35_TRANSPOSE_L0B_FACTOR if is_arch35 else 1.0, **conflict_meta},
            {"l1_to_l0a_bytes_per_s_per_aic": _m(p02r.get("l1_to_l0a_bytes_per_s_per_aic"), spec["mte1_l0a_bytes_per_s"] / aic, per_core=True),
             "l1_to_l0b_bytes_per_s_per_aic": _m(p02r.get("l1_to_l0b_bytes_per_s_per_aic"), spec["mte1_l0b_bytes_per_s"] / aic, per_core=True)},
            {"l1_to_l0a_transpose_bytes_per_s_per_aic": _m(p02r.get("l1_to_l0a_transpose_bytes_per_s_per_aic"), spec["mte1_l0a_bytes_per_s"] / aic, per_core=True),
             "l1_to_l0b_transpose_bytes_per_s_per_aic": _m(p02r.get("l1_to_l0b_transpose_bytes_per_s_per_aic"), spec["mte1_l0b_transpose_bytes_per_s"] / aic, per_core=True),
             "setup_ns": _m(p02r.get("setup_ns"), 0.0, "latency_lower")},
            unmeasured_reason=unm_hw,
        ),
        "P03_fixpipe": _cell(
            {"rate": "bytes_per_s_per_aic", "latency": "ns"},
            {"has_l0c_to_ub": spec["has_l0c_to_ub"], "has_l0c_to_l1": True, "ub_split_n_align_elems": UB_SPLIT_N_ALIGN,
             "dual_target_ub_supported": is_arch35, "split_m_supported": is_arch35,
             "nz2dn_stride_gt1_div": ARCH35_NZ2DN_STRIDE_GT1_DIV if is_arch35 else 1, **conflict_meta},
            {"fixpipe_in_bytes_per_s_per_aic": _m(p03r.get("fixpipe_in_bytes_per_s_per_aic"), fixp_in_aic, per_core=True),
             "fixpipe_out_bytes_per_s_per_aic": _m(p03r.get("fixpipe_out_bytes_per_s_per_aic"), fixp_out_aic, per_core=True)},
            {"setup_ns": _m(p03r.get("setup_ns"), 0.0, "latency_lower")},
            unmeasured_reason=unm_hw,
        ),
        "P04_mte3_store": _cell(
            {"rate": "bytes_per_s", "latency": "ns"},
            {"has_ub_to_l1": spec["has_ub_to_l1"],
             "indexed_update_diagnostic": {"band": idx_upd_band, "diagnostic": idx_upd_diag}, **conflict_meta},
            {"mte3_store_bytes_per_s_per_aiv": _m(p04r.get("mte3_warm_bytes_per_s_per_aiv"), spec["mte3_bytes_per_s"] / aiv, per_core=True)},
            {"atomic_rmw_bytes_per_s": _m(p04r.get("hbm_atomic_native_bytes_per_s"), spec["atomic_rmw_bytes_per_s"]),
             "setup_ns": _u(0.0, "latency_lower"), "atomic_contention_ns": _u(0.0, "latency_lower")},
            unmeasured_reason=unm_hw,
        ),
        "P05_l2_residency_reuse": _cell(
            {"rate": "bytes_per_s", "fraction": "dimensionless", "capacity": "bytes"},
            {"l2_bytes": spec["l2_bytes"], "l2_line_bytes": L2_LINE_BYTES, "l2_sector_bytes": l2_gran,
             "granule_bytes": l2_gran, "buffer_units_l2": 1,
             "cache_disable_supported": True, "cache_disable_mechanism": cache_mech,
             "cache_persistent_supported": is_arch35,
             "ws_sweep_bytes_per_s": p05r.get("ws_sweep_bytes_per_s", {}),
             "l2_stride_conflict": p05r.get("l2_stride_conflict", []), **conflict_meta},
            {"l2_die_bytes_per_s": _m(p05r.get("l2_die_bytes_per_s"), spec["l2_bytes_per_s"])},
            {"l2_effective_fraction": _m(p05r.get("l2_effective_fraction"), 1.0)},
            unmeasured_reason=unm_hw,
        ),
    }


def _build_p06_mmad_modes(
    ctx: dict[str, Any], p06r: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    spec, aic, is_arch35 = ctx["spec"], ctx["aic"], ctx["is_arch35"]
    _m, top_capture = ctx["m"], ctx["top_capture"]
    p06_modes: dict[str, Any] = {}
    cal_modes, unm_modes, conf_modes = [], [], []
    unsupported_modes = ["int4"] if is_arch35 else list(ARCH35_EXTRA_MODES)
    bf16_cap = spec["peaks"].get("bf16", 0.0) / aic
    int8_cap = spec["peaks"].get("int8", 0.0) / aic
    fp32_cube_k = TILE_1 if is_arch35 else TILE_4
    fp32_cap = (bf16_cap / DAV35_FP32_RATIO_VS_BF16) if is_arch35 else max(spec["peaks"].get("fp32", 0.0) / aic, bf16_cap / DAV22_FP32_RATIO_VS_BF16)
    hf32_cap = bf16_cap / HF32_RATIO_VS_BF16
    mode_defs = [
        ("bf16", TILE_16, TILE_16, TILE_16, TILE_16, TILE_16, TILE_16, bf16_cap),
        ("fp16", TILE_16, TILE_16, TILE_16, TILE_16, TILE_16, TILE_16, spec["peaks"].get("fp16", 0.0) / aic),
        ("fp32", TILE_16, TILE_16, TILE_8, fp32_cube_k, TILE_8, TILE_16, fp32_cap),
        ("hf32", TILE_16, TILE_16, TILE_8, TILE_8, TILE_8, TILE_16, hf32_cap),
        ("int8", TILE_16, TILE_16, TILE_32, TILE_32, TILE_32, TILE_16, int8_cap),
    ]
    if is_arch35:
        fp8_cap = spec["peaks"].get("fp8_e4m3", spec["peaks"].get("int8", 0.0)) / aic
        fp8_e5_cap = spec["peaks"].get("fp8_e5m2", fp8_cap * aic) / aic
        mode_defs.extend([
            ("fp8_e4m3", TILE_16, TILE_16, TILE_32, TILE_32, TILE_32, TILE_16, fp8_cap),
            ("fp8_e5m2", TILE_16, TILE_16, TILE_32, TILE_32, TILE_32, TILE_16, fp8_e5_cap),
            ("hifloat8", TILE_16, TILE_16, TILE_32, TILE_32, TILE_32, TILE_16, fp8_cap),
            ("mxfp8", TILE_16, TILE_16, TILE_32, TILE_32, TILE_32, TILE_16, fp8_cap),
            ("mxfp4", TILE_16, TILE_16, TILE_64, TILE_64, TILE_64, TILE_16, fp8_cap * TWO),
        ])
    else:
        mode_defs.append(("int4", TILE_16, TILE_16, TILE_64, TILE_64, TILE_64, TILE_16, int8_cap * TWO))
    p06_mode_diags: dict[str, Any] = {}
    gemm_e2e_summary: dict[str, Any] = {}
    raw_cube_modes = p06r.get("cube_only_modes") or {}
    raw_e2e_modes = p06r.get("gemm_e2e_modes") or p06r.get("modes") or {}
    for m_name, tm, tn, tk, ck, k0, miss, cap in mode_defs:
        if cap <= 0.0:
            continue
        if m_name in raw_e2e_modes:
            eb, ed = _m(raw_e2e_modes.get(m_name), cap, per_core=True)
            gemm_e2e_summary[m_name] = {"band": eb, "diagnostic": ed}
        b, d = _m(raw_cube_modes.get(m_name), cap, per_core=True)
        ms = _cell_status({"r": b})
        p06_mode_diags[m_name] = d
        if ms == "calibrated":
            cal_modes.append(m_name)
        elif d.get("status") == "p99_exceeds_nominal_conflict":
            conf_modes.append(m_name)
        else:
            unm_modes.append(m_name)
        entry: dict[str, Any] = {
            "status": ms, "tile_m": tm, "tile_n": tn, "tile_k": tk,
            "cube_k": ck, "k_0": k0, "m_iss": miss,
            "cube_only_rate": b if ms == "calibrated" else _clean_band(b),
        }
        if ms == "calibrated":
            entry["evidence_id"] = top_capture
        if m_name in {"mxfp8", "mxfp4"}:
            entry["scale_group"] = MX_SCALE_GROUP
        p06_modes[m_name] = entry
    core_p06_ok = all(m in cal_modes for m in ARCH22_CORE_MODES) and not conf_modes
    p06_cov = {
        "calibrated_modes": cal_modes, "unmeasured_modes": unm_modes, "unsupported_modes": unsupported_modes,
        "conflict_modes": conf_modes, "calibrated_count": len(cal_modes), "supported_count": len(p06_modes),
        "mode_diagnostics": p06_mode_diags,
    }
    return p06_modes, p06_cov, gemm_e2e_summary, core_p06_ok


def _build_compute_primitives(
    ctx: dict[str, Any], raw: dict[str, Any],
) -> dict[str, Any]:
    spec, aiv, is_arch35 = ctx["spec"], ctx["aiv"], ctx["is_arch35"]
    _m, _u, _cell, top_capture = ctx["m"], ctx["u"], ctx["cell"], ctx["top_capture"]
    conflict_meta, unm_hw = ctx["conflict_meta"], ctx["unm_hw"]
    p06r, p09r, p10r, p11r = (raw.get(k, {}) for k in ("P06_mmad", "P09_vec_reduce_cast", "P10_sort_topk", "P11_simt"))
    ub_meta = {
        "ub_bank_groups": DAV35_UB_BANK_GROUPS if is_arch35 else DAV22_UB_BANK_GROUPS,
        "ub_banks_per_group": DAV35_UB_BANKS_PER_GROUP if is_arch35 else DAV22_UB_BANKS_PER_GROUP,
        "ub_bank_size_bytes": DAV35_UB_BANK_SIZE_BYTES if is_arch35 else DAV22_UB_BANK_SIZE_BYTES,
    }
    p06_modes, p06_cov, gemm_e2e_summary, core_p06_ok = _build_p06_mmad_modes(ctx, p06r)
    sort_per_aiv_cap = spec["vector_ops_per_s"] / aiv
    p10_raw = p10r.get("sort_elems_per_s_per_aiv")
    p10_passes = SORT_PROBE_PASSES if (p10r.get("measured_algorithm") == "sort32_mrgsort4" and p10r.get("work_unit") != "elem_passes_per_s") else 1
    p10_samples = [float(v) * float(p10_passes) for v in p10_raw] if isinstance(p10_raw, list) else p10_raw
    sort_band, sort_diag = _m(p10_samples, sort_per_aiv_cap, per_core=True)
    sort_st = _cell_status({"r": sort_band})
    p07r = raw.get("P07_vec_alu", {})
    p07_samples = p07r.get("vec_alu_ops_per_s_per_aiv")
    p10_modes: dict[str, Any] = {
        "sort32_mrgsort4": {
            "status": sort_st,
            "sort_elems_per_s_per_aiv": sort_band if sort_st == "calibrated" else _clean_band(sort_band),
            **({"evidence_id": top_capture} if sort_st == "calibrated" else {}),
        },
    }
    if is_arch35:
        p10_modes["masked_hist256"] = {
            "status": "unmeasured",
            "sort_elems_per_s_per_aiv": _bound_only(sort_per_aiv_cap),
            "unmeasured_reason": "requires_histogram_microkernel",
        }
    simt_cap = max(1.0, spec["simt_warp_insts_per_s_per_aiv"]) if is_arch35 else 0.0
    simt_samples = p11r.get("simt_warp_instructions_per_s_per_aiv")
    return {
        "P06_mmad": _cell(
            {"rate": "flops_per_s_per_aic", "latency": "ns"},
            {"pipe_m_barrier_fractal_threshold": PIPE_M_BARRIER_FRACTALS, "fp32_allow_hf32_recorded": True,
             "gemm_e2e_flops_per_s_per_aic": gemm_e2e_summary,
             "small_m_bf16_ns": p06r.get("small_m_bf16_ns", {}),
             "small_m_bf16_warm_ns": p06r.get("small_m_bf16_warm_ns", {}), "mode_coverage": p06_cov, **conflict_meta},
            {}, {"pipe_m_barrier_ns": _m(p06r.get("pipe_m_barrier_ns"), 0.0, "latency_lower")},
            status_override="calibrated" if core_p06_ok else "unmeasured", modes=p06_modes, unmeasured_reason=unm_hw,
        ),
        "P07_vec_alu": _cell(
            {"rate": "ops_per_s_per_aiv"},
            {"vector_width_bytes": VEC_WIDTH_BYTES, "fp32_lanes": FP32_LANES, "bf16_lanes": BF16_LANES,
             "work_unit": "elementary_alu_ops_per_s", "probe_ops_per_element": TWO, **ub_meta, **conflict_meta},
            {"vec_alu_ops_per_s_per_aiv": _m(p07_samples, spec["vec_alu_ops_per_s"] / aiv, per_core=True)},
            {"raw_latency_cycles": _u(0.0, "latency_lower")}, unmeasured_reason=unm_hw,
        ),
        "P08_vec_sfu": _cell(
            {"rate": "ops_per_s_per_aiv"},
            {"vector_width_bytes": VEC_WIDTH_BYTES, "fp32_lanes": FP32_LANES, "bf16_lanes": BF16_LANES, **ub_meta, **conflict_meta},
            {"vec_sfu_ops_per_s_per_aiv": _m(raw.get("P08_vec_sfu", {}).get("vec_sfu_ops_per_s_per_aiv"), spec["vec_sfu_ops_per_s"] / aiv, per_core=True)},
            {"raw_latency_cycles": _u(0.0, "latency_lower")}, unmeasured_reason=unm_hw,
        ),
        "P09_vec_reduce_cast": _cell(
            {"rate": "ops_per_s_per_aiv", "cast_rate": "bytes_per_s_per_aiv"},
            {"mx_scale_group": MX_SCALE_GROUP, "mx_dynamic_quant_supported": is_arch35, **ub_meta, **conflict_meta},
            {"vec_reduce_ops_per_s_per_aiv": _m(p09r.get("vec_reduce_ops_per_s_per_aiv"), spec["vec_reduce_ops_per_s"] / aiv, per_core=True),
             "cast_bytes_per_s_per_aiv": _m(p09r.get("cast_bytes_per_s_per_aiv"), spec["cast_bytes_per_s"] / aiv, per_core=True)},
            unmeasured_reason=unm_hw,
        ),
        "P10_sort_topk": _cell(
            {"rate": "elem_passes_per_s_per_aiv"},
            {"sort_algorithm": "sort32_mrgsort4", "masked_hist256_supported": is_arch35,
             "supported_sortall_lengths": list(SORTALL_LENGTHS),
             "work_unit": "elem_passes_per_s", "probe_sort_length": SORT_PROBE_LENGTH, "probe_sort_passes": SORT_PROBE_PASSES,
             "histogram_trunk_elems": ARCH35_HIST_TRUNK if is_arch35 else 0,
             "sort_die_elems_per_s_p50": p10r.get("sort_die_elems_per_s_p50"), **conflict_meta},
            {"sort_elems_per_s_per_aiv": (sort_band, sort_diag)},
            {"sort_elems_per_s": _u(sort_per_aiv_cap)},
            modes=p10_modes, unmeasured_reason=unm_hw,
        ),
        "P11_simt": _cell(
            {"rate": "warp_instructions_per_s_per_aiv", "latency": "ns"},
            {"supported": is_arch35, "simt_ub_carveout_bytes": spec["simt_ub_carveout_bytes"],
             "simt_ub_carveout_min_bytes": SIMT_UB_CARVEOUT_MIN_BYTES if is_arch35 else 0,
             "warp_schedulers": SIMT_WARP_SCHEDULERS if is_arch35 else 0,
             "tla_max_threads": TLA_MAX_THREADS if is_arch35 else 0,
             "catlass_max_threads": CATLASS_MAX_THREADS if is_arch35 else 0, "warp_size": SIMT_WARP_SIZE,
             "simt_scalar_flops_per_s_per_aiv_p50": p11r.get("simt_scalar_flops_per_s_per_aiv_p50"), **conflict_meta},
            {"simt_warp_instructions_per_s_per_aiv": _m(simt_samples, simt_cap, per_core=True),
             "syncthreads_ns": _m(p11r.get("syncthreads_ns"), 0.0, "latency_lower"),
             "simt_launch_ns": _m(p11r.get("simt_launch_ns"), 0.0, "latency_lower")} if is_arch35 else {},
            {"simt_ops_per_s_per_aiv": _m(simt_samples, simt_cap, per_core=True)} if is_arch35 else {
                "simt_warp_instructions_per_s_per_aiv": _u(0.0), "simt_ops_per_s_per_aiv": _u(0.0),
                "syncthreads_ns": _u(0.0, "latency_lower"), "simt_launch_ns": _u(0.0, "latency_lower"),
            },
            status_override=None if is_arch35 else "unsupported", unmeasured_reason="requires_ascendc_simt_kernel" if is_arch35 else "",
        ),
    }


def _build_sync_and_link_primitives(
    ctx: dict[str, Any], raw: dict[str, Any],
) -> dict[str, Any]:
    spec, sku, is_arch35 = ctx["spec"], ctx["sku"], ctx["is_arch35"]
    _m, _u, _cell = ctx["m"], ctx["u"], ctx["cell"]
    conflict_meta, unm_hw, has_raw = ctx["conflict_meta"], ctx["unm_hw"], ctx["has_raw"]
    p12r, p13r, p14r, p15r, p17r = (
        raw.get(k, {}) for k in (
            "P12_intracore_flag", "P13_crosscore_sync", "P14_mutex_block_mmad",
            "P15_rts_stars", "P17_aicpu_metadata_launch",
        )
    )
    p18r, p19r, p20r = (raw.get(k, {}) for k in ("P18_hccl_collective_band", "P19_direct_peer_window", "P20_host_dma_sync"))
    p13_extra: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        "mode1_subblock_ns": _u(0.0, "latency_lower"),
        "mode0_aic_barrier_ns": _m(p13r.get("mode0_aic_barrier_ns"), 0.0, "latency_lower"),
    }
    if is_arch35:
        p13_extra["mode4_intra_block_ns"] = _m(p13r.get("mode4_intra_block_ns"), 0.0, "latency_lower")
    p18_curves = validate_quantitative_curves(p18r.get("curves_by_topology"), max_bw_bps=spec["link_bytes_per_s"])
    p19_curves = validate_quantitative_curves(p19r.get("curves_by_topology"), max_bw_bps=spec["link_bytes_per_s"])
    p20_curves = validate_quantitative_curves(p20r.get("dma_curve_by_size"), max_bw_bps=max(spec["host_h2d_bytes_per_s"], spec["host_d2h_bytes_per_s"]))
    p19_duplex_diag = {}
    for dk, bnd, kind in (
        ("hccl_duplex_bytes_per_s", spec["link_bytes_per_s"], "rate_upper"),
        ("hccl_duplex_4k_ns", 0.0, "latency_lower"),
        ("hccl_duplex_4k_blocked_ns", 0.0, "latency_lower"),
    ):
        if p19r.get(dk):
            db, dd = _m(p19r.get(dk), bnd, kind)
            p19_duplex_diag[dk] = {"band": db, "diagnostic": dd}
    return {
        "P12_intracore_flag": _cell(
            {"latency": "ns"}, {"event_ids": EVENT_IDS_COUNT, "unit_flag_requires_single_l0c_stage": True, **conflict_meta},
            {"flag_ns": _m(p12r.get("flag_ns"), 0.0, "latency_lower")},
            {"unitflag_ns": _u(0.0, "latency_lower"), "flag_mte2_v_ns": _m(p12r.get("flag_mte2_v_ns"), 0.0, "latency_lower"),
             "flag_v_mte3_ns": _m(p12r.get("flag_v_mte3_ns"), 0.0, "latency_lower"), "flag_m_fix_ns": _m(p12r.get("flag_m_fix_ns"), 0.0, "latency_lower"),
             "pipe_v_barrier_ns": _m(p12r.get("pipe_v_barrier_ns"), 0.0, "latency_lower")},
            unmeasured_reason=unm_hw,
        ),
        "P13_crosscore_sync": _cell(
            {"latency": "ns"},
            {"aiv_inter_block_barrier_id": AIV_BARRIER_ID, "aic_inter_block_barrier_id": AIC_BARRIER_ID,
             "intra_subblock_barrier_id": INTRA_SUBBLOCK_BARRIER_ID, "ffts_max_flag_id": FFTS_MAX_FLAG_ID,
             "hw_flag_ids": FFTS_HW_FLAG_IDS, "reverse_depth": FFTS_REVERSE_DEPTH, "mode4_supported": is_arch35, **conflict_meta},
            {"mode2_paired_ns": _m(p13r.get("mode2_paired_ns"), 0.0, "latency_lower"),
             "mode0_aiv_barrier_ns": _m(p13r.get("mode0_aiv_barrier_ns"), 0.0, "latency_lower")},
            p13_extra, unmeasured_reason=unm_hw,
        ),
        "P14_mutex_block_mmad": _cell(
            {"latency": "ns"},
            {"supported": is_arch35, "hw_mutex_ids": HW_MUTEX_IDS if is_arch35 else 0,
             "max_mutex_ids": MAX_MUTEX_IDS if is_arch35 else 0, "unit_flag_requires_single_l0c_stage": True, **conflict_meta},
            {"mutex_overhead_ns": _m(p14r.get("mutex_overhead_ns"), 0.0, "latency_lower")} if is_arch35 else {},
            ({"mutex_xpipe_handoff_ns": _m(p14r.get("mutex_xpipe_handoff_ns"), 0.0, "latency_lower")}
             if (is_arch35 and p14r.get("mutex_xpipe_handoff_ns")) else (
                 {} if is_arch35 else {"mutex_overhead_ns": _u(0.0, "latency_lower")}
             )),
            status_override=None if is_arch35 else "unsupported", unmeasured_reason=unm_hw if is_arch35 else "",
        ),
        "P15_rts_stars": _cell(
            {"latency": "ns"},
            {"queue_scope": "die", "queue_depth": STARS_QUEUE_DEPTH, "lanes": 1,
             "first_issue_includes_min_kernel": True, "issue_interval_includes_min_kernel": True,
             "issue_interval_owner": "aiv", **conflict_meta},
            {"issue_interval_ns": _m(p15r.get("issue_interval_ns"), 0.0, "latency_lower"),
             "aiv_first_issue_ns": _m(p15r.get("aiv_first_issue_ns"), 0.0, "latency_lower"),
             "aic_first_issue_ns": _m(p15r.get("aic_first_issue_ns"), 0.0, "latency_lower")},
            {"host_submit_ns": _m(p15r.get("host_submit_ns"), 0.0, "latency_lower"),
             "host_submit_burst_ns": _m(p15r.get("host_submit_burst_ns"), 0.0, "latency_lower"),
             "npugraph_issue_ns": _m(p15r.get("npugraph_issue_ns"), 0.0, "latency_lower")},
            unmeasured_reason=unm_hw,
        ),
        "P16_multistream_core_control": _cell(
            {"latency": "ns"}, {"modes": ["disjoint", "shared_cap"], **conflict_meta},
            {"stream_event_ns": _m(raw.get("P16_multistream_core_control", {}).get("stream_event_ns"), 0.0, "latency_lower")},
            {"set_limit_ns": _u(0.0, "latency_lower")}, unmeasured_reason=unm_hw,
        ),
        "P17_aicpu_metadata_launch": _cell(
            {"latency": "ns"}, {"aclgraph_cacheable": True, **conflict_meta},
            {"host_select_ns": _m(p17r.get("metadata_select_ns"), 0.0, "latency_lower")},
            {"aicpu_base_ns": _u(0.0, "latency_lower"), "gm_pub_ns": _u(0.0, "latency_lower"),
             "eager_d2h_sync_ns": _m(p20r.get("eager_d2h_sync_ns"), 0.0, "latency_lower")},
            unmeasured_reason="requires_aicpu_metadata_select_probe" if has_raw else unm_hw,
        ),
        "P18_hccl_collective_band": _cell(
            {"rate": "bytes_per_s", "latency": "ns"},
            {"payload_bytes_per_frame": ARCH22_FRAME_BYTES, "wire_bytes_per_frame": ARCH22_FRAME_BYTES,
             "short_final_frame_policy": "exact_plus_header",
             "group_min": HCCL_TWO_RANK_GROUP, "group_max": HCCL_TWO_RANK_GROUP,
             "measured_world_size": int(p18r.get("measured_world_size") or HCCL_TWO_RANK_GROUP),
             "measured_collective": str(p18r.get("measured_collective") or "all_reduce"),
             "measured_scope": "package" if int(spec.get("dies_per_package", 1)) >= 2 else "node",
             "bandwidth_definition": "two_rank_allreduce_busbw_equals_algbw",
             "curves_by_topology": p18_curves,
             "mc2_contention": p18r.get("mc2_contention", {}), **conflict_meta},
            {"link_bytes_per_s": _m(p18r.get("link_bytes_per_s"), spec["link_bytes_per_s"]),
             "startup_ns": _m(p18r.get("startup_ns"), 0.0, "latency_lower")},
            {"startup_blocked_ns": _m(p18r.get("startup_blocked_ns"), 0.0, "latency_lower")},
            unmeasured_reason="requires_multirank_peer_die" if has_raw else unm_hw,
        ),
        "P19_direct_peer_window": _cell(
            {"rate": "bytes_per_s", "latency": "ns"},
            {"transport": _peer_transport_name(int(spec.get("dies_per_package", 1)), is_arch35),
             "token_payload_bytes_per_frame": ARCH35_FRAME_PAYLOAD if is_arch35 else ARCH22_FRAME_BYTES,
             "token_wire_bytes_per_frame": ARCH35_FRAME_WIRE if is_arch35 else ARCH22_FRAME_BYTES,
             "hccl_duplex_diagnostics": p19_duplex_diag,
             "curves_by_topology": p19_curves, **conflict_meta},
            {"peer_bytes_per_s": _m(p19r.get("direct_window_peer_bytes_per_s"), spec["link_bytes_per_s"]),
             "status_sync_ns": _m(p19r.get("direct_window_status_sync_ns"), 0.0, "latency_lower")},
            unmeasured_reason="requires_direct_window_kernel" if has_raw else unm_hw,
        ),
        "P20_host_dma_sync": _cell(
            {"rate": "bytes_per_s", "latency": "ns"},
            {"contention_scope": "root_complex", "dma_curve_by_size": p20_curves, **conflict_meta},
            {"host_h2d_bytes_per_s": _m(p20r.get("host_h2d_bytes_per_s"), spec["host_h2d_bytes_per_s"]),
             "host_d2h_bytes_per_s": _m(p20r.get("host_d2h_bytes_per_s"), spec["host_d2h_bytes_per_s"])},
            {"host_rc_h2d_bytes_per_s": _u(spec["host_rc_h2d_bytes_per_s"]),
             "host_rc_d2h_bytes_per_s": _u(spec["host_rc_d2h_bytes_per_s"]),
             "pinned_h2d_startup_ns": _m(p20r.get("pinned_h2d_startup_ns"), 0.0, "latency_lower"),
             "pinned_d2h_startup_ns": _m(p20r.get("pinned_d2h_startup_ns"), 0.0, "latency_lower"),
             "pageable_h2d_startup_ns": _m(p20r.get("pageable_h2d_startup_ns"), 0.0, "latency_lower"),
             "pageable_d2h_startup_ns": _m(p20r.get("pageable_d2h_startup_ns"), 0.0, "latency_lower"),
             "pinned_dma_startup_ns": _u(0.0, "latency_lower"),
             "pageable_dma_startup_ns": _u(0.0, "latency_lower"),
             "eager_d2h_sync_ns": _m(p20r.get("eager_d2h_sync_ns"), 0.0, "latency_lower")},
            unmeasured_reason=unm_hw,
        ),
    }


def build_primitive_table_payload(
    *, sku: str, cann_version: str, endpoint_alias: str, campaign_id: str,
    raw_measurements: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble a full-envelope SHA-256 sealed mindie-l0-primitives/1 table with RC-1 coverage."""
    base_spec = _load_sku_hardware_spec(sku)
    alias = sanitize_record_alias(endpoint_alias, sku)
    aic, aiv = max(1, base_spec["aic_count"]), max(1, base_spec["aiv_count"])
    raw = _sanitize_capture_obj(raw_measurements) if raw_measurements else {}
    sample_dev = str(raw.get("device_name") or "").strip()
    if sample_dev and sample_dev != sku:
        raise ValueError(f"bench_sample_sku_mismatch:expected={sku}:got={sample_dev}")
    rt_cann = str(raw.get("runtime_cann_version") or "").strip()
    if rt_cann and cann_version and cann_version != "unmeasured" and cann_version != rt_cann:
        raise ValueError(f"bench_cann_version_mismatch:runtime={rt_cann}:arg={cann_version}")
    obs = raw.get("observed_silicon") if isinstance(raw.get("observed_silicon"), dict) else {}
    obs_aic = obs.get("observed_aic_count")
    core_mismatch = isinstance(obs_aic, int) and obs_aic > 0 and obs_aic != aic
    conflict_meta = (
        {"silicon_profile_conflict": {
            "observed_aic_count": obs_aic, "profile_aic_count": aic,
            "observed_aicore_freq_mhz": obs.get("observed_aicore_freq_mhz"),
        }} if core_mismatch else {}
    )
    spec, clock_meta = _apply_core_clock_scale(base_spec, obs.get("observed_aicore_freq_mhz"))
    raw_ini = str(raw.get("ini_filename") or f"{sku}.ini")
    ini_name = os.path.basename(raw_ini.replace("\\", "/")) or f"{sku}.ini"
    ini_meta = {
        "ini_sha256": str(raw.get("ini_sha256") or ""),
        "ini_filename": ini_name,
        "runtime_cann_version": rt_cann,
        "bisheng_version": str(raw.get("bisheng_version") or ""),
        "observed_cur_freq_mhz": obs.get("observed_cur_freq_mhz"),
        **clock_meta,
    }
    cap_bytes = canonical_capture_bytes(sku, raw_measurements)
    top_capture = f"capture://{alias}/{campaign_id}/{hashlib.sha256(cap_bytes).hexdigest()}"
    allow_core = not core_mismatch
    ctx = {
        "sku": sku, "spec": spec, "aic": aic, "aiv": aiv, "top_capture": top_capture,
        "is_arch35": spec["arch_preset"] == "npu_arch_3510", "allow_core": allow_core,
        "conflict_meta": conflict_meta, "ini_meta": ini_meta, "has_raw": bool(raw_measurements),
        "unm_hw": "no_reachable_physical_hardware" if not raw_measurements else "requires_ascendc_microkernel",
        "m": lambda s, b, kind="rate_upper", *, per_core=False, max_admissible=None, allow_zero=False: summarize_metric(
            s, certified_bound=b, bound_kind=kind, allow_meas=(allow_core if per_core else True),
            max_admissible=max_admissible, allow_zero=allow_zero,
        ),
        "u": lambda b, kind="rate_upper": (_bound_only(b, kind), {"status": "unmeasured", "nominal_bound": round(float(b), ROUND_METRIC_DIGITS)}),
        "cell": lambda units, params, core, extra=None, **kw: _make_cell(units, params, core, extra, top_capture=top_capture, **kw),
    }
    p: dict[str, Any] = {}
    p.update(_build_memory_primitives(ctx, raw))
    p.update(_build_compute_primitives(ctx, raw))
    p.update(_build_sync_and_link_primitives(ctx, raw))
    collected_utc = str(raw.get("collected_utc") or "").strip() or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    resolved_cann = rt_cann if (rt_cann and (not cann_version or cann_version == "unmeasured")) else cann_version
    payload: dict[str, Any] = {
        "schema_version": PRIMITIVE_SCHEMA_VERSION,
        "arch_preset": spec["arch_preset"],
        "sku": sku,
        "cann_version": resolved_cann,
        "provenance": {
            "campaign_id": campaign_id, "endpoint_alias": alias, "capture_uri": top_capture,
            "calibrated": any(c["status"] == "calibrated" for c in p.values()),
            "collected_utc": collected_utc,
            "clock_calibration": clock_meta,
            "silicon_fingerprint": {k: spec[k] for k in ("aic_count", "aiv_count", "hbm_gib", "l2_bytes", "ub_bytes", "l1_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes")},
        },
        "primitives": p,
    }
    payload["envelope_sha256"] = compute_envelope_sha256(payload)
    assert_no_privacy_leak(json.dumps(payload, sort_keys=True), label="table")
    return payload


def run_remote_calibration(args: argparse.Namespace) -> dict[str, Any]:
    return _run_remote_calibration(args, _load_sku_hardware_spec(args.sku))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Ascend L0 primitive micro-benchmark calibration suite")
    for flag in ("--local-npu", "--emit-unmeasured"):
        parser.add_argument(flag, action="store_true")
    for flag, default in (
        ("--endpoint-alias", ""), ("--record-alias", ""), ("--container", ""), ("--python-bin", "python3"),
        ("--visible-devices", ""), ("--sku", ""), ("--arch", "npu_arch_2201"), ("--cann-version", "unmeasured"),
        ("--campaign-id", "camp-202611-l0"), ("--primitives", "all"), ("--out", ""),
        ("--manifest", "data/primitives/manifest.json"), ("--from-raw-json", ""),
        ("--profile-json", ""),
    ):
        parser.add_argument(flag, default=default)
    for flag, default in (
        ("--npu-id", 0), ("--chip-id", 0), ("--device-id", 0),
        ("--peer-npu-id", 0), ("--peer-chip-id", 1), ("--peer-device-id", -1),
        ("--l2-bytes", DEFAULT_L2_ARG), ("--aic-count", DEFAULT_AIC_ARG), ("--aiv-count", DEFAULT_AIV_ARG),
        ("--warmup", DEFAULT_WARMUP), ("--repeats", DEFAULT_REPEATS),
    ):
        parser.add_argument(flag, type=int, default=default)
    args = parser.parse_args(argv)

    with _register_profile_json(args.sku, args.profile_json):
        if args.local_npu:
            if not args.sku:
                raise SystemExit("--sku is required with --local-npu to bind hardware geometry")
            spec = _load_sku_hardware_spec(args.sku)
            vis = str(args.visible_devices or "").strip()
            obs = _check_local_idle_die(int(args.npu_id), int(args.chip_id), int(args.device_id), vis)
            if int(args.peer_device_id) >= 0:
                _check_local_idle_die(int(args.peer_npu_id), int(args.peer_chip_id), int(args.peer_device_id), vis)
            if vis:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = vis
            raw = run_local_npu_microbenchmarks(
                device_id=args.device_id, peer_device_id=args.peer_device_id,
                l2_bytes=int(spec["l2_bytes"]), aic_count=int(spec["aic_count"]), aiv_count=int(spec["aiv_count"]),
                arch_preset=str(spec["arch_preset"]), warmup=args.warmup, repeats=args.repeats,
                primitives=str(args.primitives or "all"),
            )
            raw["observed_silicon"] = obs
            print(RESULT_MARKER + json.dumps(raw, sort_keys=True, separators=(",", ":")))
            return 0

        if not args.sku:
            raise SystemExit("--sku is required")
        out_path = Path(args.out) if args.out else (REPO_ROOT / "data" / "primitives" / SKU_DEFAULT_FILES[args.sku])
        manifest_path = Path(args.manifest) if Path(args.manifest).is_absolute() else (REPO_ROOT / args.manifest)
        if args.from_raw_json:
            raw_meas = unwrap_raw_measurements(json.loads(Path(args.from_raw_json).read_text(encoding="utf-8")))
        else:
            raw_meas = None if args.emit_unmeasured else run_remote_calibration(args)
        rec_alias = args.record_alias or sanitize_record_alias(args.endpoint_alias or args.sku, args.sku)
        payload = build_primitive_table_payload(
            sku=args.sku, cann_version=args.cann_version, endpoint_alias=rec_alias,
            campaign_id=args.campaign_id, raw_measurements=raw_meas,
        )
        from mindie_sim.primitives.schema import parse_primitive_table

        parse_primitive_table(payload)
        cap_bytes = canonical_capture_bytes(args.sku, raw_meas)
        table_text = json.dumps(payload, indent=JSON_INDENT, sort_keys=True) + "\n"
        assert_no_privacy_leak(table_text, label="table_write")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cap_path = out_path.parent / "captures" / f"{out_path.stem}.capture.json"
        cap_path.parent.mkdir(parents=True, exist_ok=True)
        cap_path.write_bytes(cap_bytes)
        out_path.write_text(table_text, encoding="utf-8")
        update_manifest(manifest_path, out_path, payload)
        cal = [k for k, v in payload["primitives"].items() if v["status"] == "calibrated"]
        rel_out = os.path.relpath(out_path, REPO_ROOT) if out_path.resolve().is_relative_to(REPO_ROOT.resolve()) else out_path.name
        print(json.dumps({"sku": args.sku, "out": rel_out, "calibrated_count": len(cal), "calibrated": cal, "coverage": _coverage_summary(payload), "envelope_sha256": payload["envelope_sha256"]}, indent=JSON_INDENT))
        return 0


@contextmanager
def _register_profile_json(sku: str, profile_json: str):
    if not profile_json:
        yield
        return
    from mindie_sim.backends import ascend_profiles

    raw = json.loads(Path(profile_json).read_text(encoding="utf-8"))
    entry = raw[sku] if isinstance(raw, dict) and sku in raw and isinstance(raw[sku], dict) else raw
    if not isinstance(entry, dict) or "device" not in entry:
        raise ValueError(f"bench_profile_json_invalid:{sku}")
    reg = ascend_profiles()
    already = sku in reg
    if not already:
        reg[sku] = deepcopy(entry)
    try:
        yield
    finally:
        if not already:
            reg.pop(sku, None)


if __name__ == "__main__":
    sys.exit(main())
