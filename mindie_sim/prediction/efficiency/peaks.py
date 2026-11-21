"""Hardware peak rates and operator work bounds.

[Role] Resolve stage and link supply and derive nominal per-pipe service bounds
  from operator reports.
[Invariants] Rates retain their declared hardware and rank boundaries. Work
  and modeled invocation costs retain their original units and provenance.
[Timeouts & Locks] In-memory lookups and arithmetic with no device locks.
[Side-Effect Policy] Input cases and reports are read-only.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.hardware import Device

from ..structure import parse_case


DEFAULT_TOKENS = 1
DEFAULT_CONTEXT = 4096
DEFAULT_SEQUENCES = 1
DEFAULT_OUTPUT_BUDGET = 1024
DEFAULT_STAGE = 0
DEFAULT_REPLICA = 0

_VECTOR_FP16_FMA_FLOPS_PER_ISSUE = 4.0
_LAUNCH_DIAGNOSTIC_MIN_THRESHOLD_MS = 0.002
_LAUNCH_DIAGNOSTIC_SHORT_DURATION_MS = 0.006
_LAUNCH_DIAGNOSTIC_DURATION_FACTOR = 2.5
_LAUNCH_DIAGNOSTIC_MAX_WORK_SHARE = 0.25

_HBM_ROLES = frozenset({"hbm", "memory", "gm_hbm_cacheline_128b"})
_CUBE_ROLES = frozenset({"cube", "aic"})
_VEC_ROLES = frozenset({"vector", "vbitsort32", "scalar_store_issue", "ub_copy", "aiv", "reg_issue_slots", "reg_alu_lanes", "reg_special_lanes", "reg_reduce_lanes", "reg_shuffle_lanes", "reg_data_writes", "reg_ub_accesses", "reg_ub_stores"})
_DTYPE_ALIASES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "fp8_e4m3": "fp8", "fp8_e5m2": "fp8"}
_BOUNDS = ("memory_bound", "cube_bound", "vector_bound", "link_bound", "launch_bound")
_KINDS = ("cube", "vector", "cv_fused", "communication", "memory_only")
_SCALE_KEYS = ("hbm_bw_scale", "cube_flops_scale", "vector_ops_scale", "link_bw_scale", "hbm_capacity_scale")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _check_pos_float(name: str, val: Any) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or float(val) <= 0.0:
        raise ValueError(f"{name}_must_be_positive:{val}")
    return float(val)


def _validate_geometry(phase: str, tokens: int, context: int, sequences: int, scope: str) -> None:
    if phase not in {"prefill", "decode"} or scope not in {"body", "model"}:
        raise ValueError(f"efficiency_invalid_phase_or_scope:{phase}:{scope}")
    for name, val, minimum in (("tokens", tokens, 1), ("sequences", sequences, 1), ("context", context, 0)):
        if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
            raise ValueError(f"efficiency_invalid_{name}:{val}")


def _resolve_link_bw_bytes_per_ms(case: CaseSpec, scope: str) -> float:
    """Effective machine-wide link bandwidth in bytes/ms (slowest configured link)."""
    links = parse_case(case).machine.links
    if not links:
        return 0.0
    return float(min(link.capacity_bytes_per_ms for link in links))


def _resolve_link_peak_bw_bytes_per_ms(case: CaseSpec, scope: str) -> float:
    """Peak machine-wide link bandwidth before configured utilization is applied."""
    links = parse_case(case).machine.links
    if not links:
        return 0.0
    return float(min(
        getattr(link, "effective_peak_capacity_bytes_per_ms", link.capacity_bytes_per_ms)
        for link in links
    ))


def _role_scale_factor(role: str, *, hbm_bw_scale: float, cube_flops_scale: float, vector_ops_scale: float, link_bw_scale: float) -> float:
    """Scale only the named physical supply; MTE/L2/copy stay independent unless explicitly coupled."""
    r = str(role or "").strip()
    if r in _HBM_ROLES or r.startswith(("hbm", "gm_hbm")):
        return hbm_bw_scale
    if r in _CUBE_ROLES or r.startswith("cube"):
        return cube_flops_scale
    if r in _VEC_ROLES or r.startswith(("vector", "vreg_", "sreg_", "reg_", "ub_", "aiv")):
        return vector_ops_scale
    return link_bw_scale if (r == "link" or r.startswith("link")) else 1.0


def _resolve_cube_peak(hw: Mapping[str, Any], dtype: str, default_dtype: str = "bf16") -> float:
    target = str(dtype or default_dtype or hw.get("default_dtype") or "bf16").strip().lower()
    canonical, peaks = _DTYPE_ALIASES.get(target, target), hw.get("compute_peaks", {})
    try:
        return float(peaks[canonical])
    except KeyError:
        try:
            return float(peaks[target])
        except KeyError:
            return min(float(d.peak(target)) for d in (hw.get("dies") or (hw["die"],)))


def extract_hardware_peaks(case: CaseSpec | Mapping[str, Any], *, stage: int = DEFAULT_STAGE, replica: int = DEFAULT_REPLICA) -> dict[str, Any]:
    """Extract conservative stage-level Cube, Vector, HBM, MTE, and Link peak rates in SI units (/s)."""
    parsed = parse_case(case)
    p = parsed.runtime.parallel
    width, start = p.tp * p.pcp, (replica * p.pp + stage) * (p.tp * p.pcp)
    dies = tuple(parsed.machine.dies[start:start + width]) or (parsed.machine.dies[0],)
    common_dtypes = set(dict(dies[0].compute_peaks))
    for d in dies[1:]:
        common_dtypes &= set(dict(d.compute_peaks))
    compute_peaks, rates_list = {dt: min(float(d.peak(dt)) for d in dies) for dt in sorted(common_dtypes)}, [dict(d.rates) for d in dies]
    vec_peak, hbm_peak = min(float(r["vector"]) for r in rates_list), min(float(r["hbm"]) for r in rates_list)
    mte1_peak, fixpipe_peak = min(float(r.get("mte1", 0.0)) for r in rates_list), min(float(r.get("fixpipe", 0.0)) for r in rates_list)
    mte2_peak, mte3_peak = min(float(r.get("mte2_aic", r.get("mte2", r["hbm"]))) for r in rates_list), min(float(r.get("mte3", r["hbm"])) for r in rates_list)
    links = parsed.machine.links
    active_width = max(p.tp, p.ep if p.ep > 1 else 1)
    intra_links = [lnk for lnk in links
                   if "intra_package" in lnk.channel
                   or "intra_node" in lnk.channel
                   or len(lnk.members) <= parsed.machine.dies_per_package]
    intra_bw = max((
        lnk.capacity_bytes_per_ms * 1000.0 * (
            max(1, min(active_width, len(lnk.members)) - 1)
            if lnk.semantics == "per_pair" else 1.0)
        for lnk in intra_links), default=0.0)
    inter_bw = min((lnk.capacity_bytes_per_ms * 1000.0 for lnk in links),
                   default=0.0)
    link_peak = (
        intra_bw if (active_width <= max(parsed.machine.dies_per_package,
                                         parsed.machine.ranks_per_node)
                     and intra_bw > 0.0) else inter_bw
    ) if links else 0.0
    is_lane_issue = "lane operation per cycle" in str(dict(parsed.machine.assumptions).get("vector_issue", ""))
    return {
        "die": dies[0], "dies": dies, "compute_peaks": compute_peaks, "default_dtype": parsed.model.compute_dtype or "bf16",
        "vector_peak_ops_per_s": vec_peak, "vector_fp32_fma_flops_per_s": (2.0 if is_lane_issue else 1.0) * vec_peak, "vector_fp16_fma_flops_per_s": (_VECTOR_FP16_FMA_FLOPS_PER_ISSUE if is_lane_issue else 1.0) * vec_peak,
        "hbm_peak_bytes_per_s": hbm_peak, "mte1_peak_bytes_per_s": mte1_peak, "mte2_peak_bytes_per_s": mte2_peak,
        "mte3_peak_bytes_per_s": mte3_peak, "fixpipe_peak_bytes_per_s": fixpipe_peak, "link_peak_bytes_per_s": float(link_peak),
        "intra_package_link_bytes_per_s": float(intra_bw or link_peak), "inter_package_link_bytes_per_s": float(inter_bw or link_peak), "dies_per_package": parsed.machine.dies_per_package,
    }


def _operator_invocation_bounds(op: Mapping[str, Any], row: Mapping[str, Any], parsed: CaseSpec, dies: Sequence[Device], duration_ms: float, flops: float, vector_ops: float, link_bytes: float) -> tuple[float, float]:
    """Return (fixed_startup_ms, inclusive_minimum_ms) without host launch_us."""
    basis, startup_ms, minimum_ms = row.get("basis") or {}, 0.0, 0.0
    inv_models = basis.get("invocation_models") or ()
    if inv_models:
        startup_ms, minimum_ms = max(float(i.get("startup_ms", 0.0) or 0.0) for i in inv_models), max(float(i.get("minimum_ms", 0.0) or 0.0) for i in inv_models)
    else:
        meta = dict(op.get("metadata") or {})
        explicit_cls = meta.get("invocation_class") or (meta.get("program_execution") or {}).get("execution_class")
        inv_rules, matched_rule = parsed.runtime.kernels.get("invocations", ()), False
        if inv_rules:
            ident = {"hardware": parsed.machine.hardware_key or parsed.machine.sku, "runtime": parsed.runtime.vllm_impl, "revision": parsed.runtime.vllm_version, "format": parsed.runtime.kernel_format, "kernel": meta.get("kernel", meta.get("kernel_name", op.get("recipe", ""))), "implementation": meta.get("implementation"), "dtype": op.get("dtype") or parsed.model.compute_dtype, "layout": op.get("layout", ""), "replay_scope": meta.get("graph_replay_scope", "none"), "recipe": op.get("recipe", ""), "family": op.get("family", ""), "phase": op.get("phase", "")}
            cands = [(len(r.get("match", {})), r) for r in inv_rules if all(ident.get(k) == v for k, v in (r.get("match") or {}).items())]
            if cands:
                top = [c[1] for c in cands if c[0] == max(x[0] for x in cands)]
                if len(top) == 1:
                    startup_ms, minimum_ms, matched_rule = float(top[0].get("startup_ms", 0.0) or 0.0), float(top[0].get("minimum_ms", 0.0) or 0.0), True
        if not matched_rule and not (link_bytes > 0 and not explicit_cls):
            classes = (explicit_cls,) if explicit_cls else tuple(r for r, amt in (("cube", flops), ("vector", vector_ops)) if amt > 0)
            for d in dies:
                inv_map = dict(d.kernel_invocations)
                for cls_name in (c for c in classes if c in inv_map):
                    startup_ms, minimum_ms = max(startup_ms, float(inv_map[cls_name].startup_ms)), max(minimum_ms, float(inv_map[cls_name].minimum_ms))
    return min(startup_ms, duration_ms), min(minimum_ms, duration_ms)


def diagnostic_legacy_invocation_bounds(basis: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit diagnostic replay of archived service-regime floors.

    Never called by the ordinary analysis path.  The returned fixed/minimum
    values retain the source rule id and are marked ``production_prediction``
    false so legacy whole-call floors cannot silently re-enter timing.
    """
    fixed_ms, minimum_ms, sources = 0.0, 0.0, []
    for reg in basis.get("service_regimes", ()) or ():
        queries, resp = reg.get("queries", {}), reg.get("response", {})
        if (queries.get("observable_response") or queries.get("joint_native_service")) and float(resp.get("fixed_us", 0.0) or 0.0) > 0:
            sources.append(str(reg.get("rule_id") or reg.get("id") or "service_regime"))
            if resp.get("terms") or resp.get("services"):
                fixed_ms = max(fixed_ms, float(resp["fixed_us"]) / 1000.0)
            else:
                minimum_ms = max(minimum_ms, float(resp["fixed_us"]) / 1000.0)
    return {"fixed_ms": fixed_ms, "minimum_ms": minimum_ms,
            "source_rule_ids": sources, "production_prediction": False}


def legacy_service_regime_diagnostics(case: CaseSpec | Mapping[str, Any]) -> dict[str, Any] | None:
    """Expose archived service-regime rules as diagnostic-only, never as production price."""
    parsed = parse_case(case)
    metadata = dict(parsed.runtime.metadata)
    existing = metadata.get("legacy_service_regime_diagnostics")
    if isinstance(existing, Mapping):
        return deepcopy(dict(existing))
    rules = list(parsed.runtime.kernels.get("service_regimes", ()) or ())
    if not rules:
        return None
    return {"production_prediction": False, "scaled": False,
            "source": "runtime.kernels.service_regimes",
            "rules": deepcopy(rules)}


def _resolve_operator_link_peak(parsed, op, hw) -> float:
    """Physical per-rank link peak for this operator's declared group axis."""
    metadata = dict(op.get("metadata") or ())
    axis = str(metadata.get("group_axis") or "tp")
    parallel = parsed.runtime.parallel
    width = max(1, int(getattr(parallel, axis, 1) or 1))
    links = parsed.machine.links
    candidates = [lnk for lnk in links
                  if lnk.members and min(len(lnk.members), width) >= width]
    if not candidates:
        candidates = list(links)
    if not candidates:
        return float(hw.get("link_peak_bytes_per_s", 0.0) or 0.0)
    within_node = width <= max(1, parsed.machine.ranks_per_node)
    selected = [lnk for lnk in candidates
                if ("intra_node" in lnk.channel or "intra_package" in lnk.channel)
                == within_node] or candidates
    link = min(selected, key=lambda item: item.capacity_bytes_per_ms)
    # A link that explicitly declares a usable fraction is bounded by that
    # configured capacity; a profile that only publishes a nameplate peak keeps
    # the peak service boundary used by its silicon anchors.
    if getattr(link, "bw_utilization", None) is not None:
        peak = float(link.capacity_bytes_per_ms
                     or link.peak_capacity_bytes_per_ms) * 1000.0
    else:
        peak = float(link.peak_capacity_bytes_per_ms
                     or link.capacity_bytes_per_ms) * 1000.0
    if link.semantics == "per_pair":
        peak *= max(1, width - 1)
    elif link.semantics == "shared":
        peak /= max(1, width)
    return peak


def physical_link_service(comm_info: Mapping[str, Any]) -> tuple[float, float] | None:
    """Exact per-physical-link byte service from the communication ledger.

    Returns ``(max(wire_bytes / peak_bytes_per_ms), sum(wire_bytes))`` only
    when every published resource carries its own matching capacity.  The
    numerator and denominator then cover the same rank boundary, so utilization
    is never repaired by clipping or by averaging uneven traffic away.
    """
    resources = dict(comm_info.get("resources") or {})
    wire, capacity = {}, {}
    for name, item in resources.items():
        if not isinstance(item, Mapping):
            continue
        amount = float(item.get("wire_bytes", 0.0) or 0.0)
        peak = float(item.get("peak_bytes_per_ms", 0.0) or 0.0)
        if amount > 0.0:
            wire[str(name)] = amount
        if peak > 0.0:
            capacity[str(name)] = peak
    if not wire or not set(wire) <= set(capacity):
        return None
    return (max(wire[name] / capacity[name] for name in wire),
            math.fsum(wire.values()))


def analyze_operator_row(row: Mapping[str, Any], case: CaseSpec | Mapping[str, Any], *, peaks: Mapping[str, Any] | None = None, observed_duration_ms: float | None = None, pmu: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Compute 4D efficiency (Cube_MFU, Vec_MFU, MBU, Link_Util) & CV-overlap for one operator."""
    parsed, op, count = parse_case(case), row["operator"], int(row.get("occurrences", 1))
    if isinstance(row.get("occurrences", 1), bool) or count < 1:
        raise ValueError(f"operator_occurrences_must_be_positive:{row.get('occurrences')}")
    raw_dur = observed_duration_ms if observed_duration_ms is not None else row["duration_ms"]
    if isinstance(raw_dur, bool) or not isinstance(raw_dur, (int, float)) or not math.isfinite(raw_dur) or float(raw_dur) < 0.0:
        raise ValueError(f"operator_duration_must_be_nonnegative_finite:{raw_dur}")
    duration_ms, stage, replica = float(raw_dur), int(op.get("stage", 0)), int(row.get("replica", 0))
    hw = dict(peaks) if peaks is not None else extract_hardware_peaks(parsed, stage=stage, replica=replica)
    dies, ref_work, has_jn = hw.get("dies") or (hw["die"],), dict(row.get("analytical_reference_work") or {}), bool(row.get("joint_native_service"))
    raw_meta = op.get("metadata") or ()
    metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {k: v for k, v in raw_meta}
    res_work, serviced = dict(metadata.get("resource_work") or {}), dict(row.get("serviced") or {})

    serv_hbm, serv_vec = (sum(float(v) for k, v in serviced.items() if k.startswith("memory@")) if not has_jn else 0.0), (sum(float(v) for k, v in serviced.items() if k.startswith("vector@")) if not has_jn else 0.0)
    serv_mte1, serv_mte2, serv_mte3 = (sum(float(v) for k, v in serviced.items() if k.startswith("mte1@")) if not has_jn else 0.0), (sum(float(v) for k, v in serviced.items() if k.startswith("mte2")) if not has_jn else 0.0), (sum(float(v) for k, v in serviced.items() if k.startswith("mte3@")) if not has_jn else 0.0)
    comm_info = dict(row.get("communication") or {})
    comm_wire = comm_info.get("wire_by_resource") or {}
    comm_res = dict(comm_info.get("resources") or {})
    _non_lnk = ("memory@", "vector@", "matrix@", "mte", "fixpipe@", "l2@", "ub", "reg_", "sreg_", "vreg_", "scalar_", "vbitsort", "host", "scheduler@", "native_call_service")
    if has_jn:
        serv_link = 0.0
    elif comm_wire:
        pair_sources = {str(key).split(":pair", 1)[0]
                        for key in comm_wire if ":pair" in str(key)}
        rank_div = max(1, len(comm_info.get("useful_bytes_by_rank") or ())
                       or len(pair_sources))
        serv_link = sum(float(v) for v in comm_wire.values()) / rank_div
    elif comm_info.get("wire_bytes") is not None:
        is_per_port = any(":tx:port" in k or ":rx:port" in k for k in comm_res)
        rank_div = (max(1, len(comm_info.get("useful_bytes_by_rank") or ()) or len(dies)) if is_per_port else 1)
        serv_link = float(comm_info["wire_bytes"]) / rank_div
    else:
        lnk_items = {k: float(v) for k, v in serviced.items() if not any(k.startswith(p) for p in _non_lnk) and not k.endswith(":switch_plane")}
        tx_items = [v for k, v in lnk_items.items() if ":tx:port" in k]
        serv_link = (sum(tx_items) / max(1, len(dies))) if tx_items else sum(lnk_items.values())

    m_raw = op.get("matrix_flops") or ()
    matrix_flops = {str(k): float(v) for k, v in (m_raw.items() if isinstance(m_raw, Mapping) else m_raw) if v}
    flops, dtype = max(float(op.get("flops", 0.0) or 0.0) + float(res_work.get("cube", 0.0) or 0.0), sum(matrix_flops.values()), float(ref_work.get("cube", 0.0) or 0.0)), str(op.get("dtype") or hw["default_dtype"] or "bf16")
    t_cube_ideal_ms = sum((fl / _resolve_cube_peak(hw, dt, dtype)) * 1000.0 for dt, fl in matrix_flops.items()) if matrix_flops else ((flops / _resolve_cube_peak(hw, dtype, hw["default_dtype"])) * 1000.0 if flops > 0 else 0.0)
    vector_ops = max(float(op.get("vector_ops", 0.0) or 0.0) + float(res_work.get("vector", 0.0) or 0.0), serv_vec, float(ref_work.get("vector", 0.0) or 0.0))
    t_vec_ideal_ms = (vector_ops / hw["vector_peak_ops_per_s"]) * 1000.0 if vector_ops > 0 else 0.0
    hbm_bytes = max(float(op.get("hbm_bytes", 0.0) or 0.0) + float(res_work.get("hbm", 0.0) or 0.0), serv_hbm, float(ref_work.get("hbm", 0.0) or 0.0))
    mte1_bytes = max(float(metadata.get("mte1_bytes", 0.0) or 0.0) + float(res_work.get("mte1", 0.0) or 0.0), serv_mte1, float(ref_work.get("mte1", 0.0) or 0.0))
    mte2_bytes = max(float(op.get("mte2_bytes", 0.0) or 0.0) + float(res_work.get("mte2", 0.0) or 0.0), serv_mte2, float(ref_work.get("mte2", 0.0) or 0.0))
    mte3_bytes = max(float(op.get("mte3_bytes", 0.0) or 0.0) + float(res_work.get("mte3", 0.0) or 0.0), serv_mte3, float(ref_work.get("mte3", 0.0) or 0.0))
    t_hbm_ideal_ms = (hbm_bytes / hw["hbm_peak_bytes_per_s"]) * 1000.0 if hbm_bytes > 0 else 0.0
    logical_link_bytes = max(float(op.get("link_bytes", 0.0) or 0.0) + float(res_work.get("link", 0.0) or 0.0), float(ref_work.get("link", 0.0) or 0.0))
    physical_link = physical_link_service(comm_info)
    link_bytes = max(logical_link_bytes, serv_link)
    if physical_link is not None:
        # The physical wire ledger drives the service bound; the reported
        # logical link volume stays the declared per-rank payload.
        t_link_ideal_ms, _physical_link_bytes = physical_link
    else:
        op_link_peak = _resolve_operator_link_peak(parsed, op, hw)
        t_link_ideal_ms = ((link_bytes / op_link_peak) * 1000.0
                           if (link_bytes > 0.0 and op_link_peak > 0.0)
                           else 0.0)

    total_ms, safe_t = duration_ms * count, max(duration_ms, 1e-12)
    operator_kind = "cv_fused" if (flops > 0 and vector_ops > 0) else "cube" if flops > 0 else "vector" if vector_ops > 0 else "communication" if link_bytes > 0 else "memory_only"
    fixed_ms, minimum_ms = _operator_invocation_bounds(op, row, parsed, dies, duration_ms, flops, vector_ops, link_bytes)
    roofline_ideal_ms, launch_thresh = max(t_hbm_ideal_ms, t_cube_ideal_ms, t_vec_ideal_ms, t_link_ideal_ms), max(fixed_ms, minimum_ms, _LAUNCH_DIAGNOSTIC_MIN_THRESHOLD_MS)
    pipe_ideals = {"memory_bound": t_hbm_ideal_ms, "cube_bound": t_cube_ideal_ms, "vector_bound": t_vec_ideal_ms, "link_bound": t_link_ideal_ms}
    dominant_bound = "launch_bound" if (roofline_ideal_ms == 0.0 or (roofline_ideal_ms < launch_thresh and duration_ms <= max(_LAUNCH_DIAGNOSTIC_SHORT_DURATION_MS, _LAUNCH_DIAGNOSTIC_DURATION_FACTOR * launch_thresh) and roofline_ideal_ms / safe_t < _LAUNCH_DIAGNOSTIC_MAX_WORK_SHARE)) else max(pipe_ideals, key=pipe_ideals.__getitem__)
    cv_dominant_bound = cv_roofline_efficiency = cv_compute_share_ratio = cv_overlap_ratio = None
    roofline_efficiency = roofline_ideal_ms / max(duration_ms - fixed_ms, 1e-9) if roofline_ideal_ms > 0 else 0.0
    if operator_kind == "cv_fused":
        cv_dominant_bound, cv_roofline_efficiency, denom_comp = f"cv_{dominant_bound}", roofline_efficiency, t_cube_ideal_ms + t_vec_ideal_ms
        cv_compute_share_ratio = (t_vec_ideal_ms / denom_comp) if denom_comp > 0 else 0.5
        pmu_data = pmu or row.get("pmu")
        if isinstance(pmu_data, Mapping) and ("aic_mac_time_us" in pmu_data or "aiv_vec_time_us" in pmu_data or "compute_span_us" in pmu_data):
            t_c, t_v = float(pmu_data.get("aic_mac_time_us", t_cube_ideal_ms * 1000.0)) / 1000.0, float(pmu_data.get("aiv_vec_time_us", t_vec_ideal_ms * 1000.0)) / 1000.0
            t_span = float(pmu_data["compute_span_us"]) / 1000.0 if "compute_span_us" in pmu_data else (max(t_c + float(pmu_data.get("aic_wait_time_us", 0.0)) / 1000.0, t_v + float(pmu_data.get("aiv_wait_time_us", 0.0)) / 1000.0) if ("aic_wait_time_us" in pmu_data or "aiv_wait_time_us" in pmu_data) else max(duration_ms - fixed_ms, max(t_c, t_v)))
            cv_overlap_ratio = _clamp((t_c + t_v - t_span) / min(t_c, t_v), 0.0, 1.0) if min(t_c, t_v) > 0 else 0.0
        elif denom_comp > max(duration_ms - fixed_ms, 0.0) and min(t_cube_ideal_ms, t_vec_ideal_ms) > 0:
            cv_overlap_ratio = _clamp((denom_comp - max(duration_ms - fixed_ms, max(t_cube_ideal_ms, t_vec_ideal_ms))) / min(t_cube_ideal_ms, t_vec_ideal_ms), 0.0, 1.0)
    return {
        "physical_id": row.get("physical_id", ""), "op_id": op.get("op_id", ""), "family": op.get("family", ""), "recipe": op.get("recipe", ""),
        "structure_block": str(dict(op.get("metadata") or ()).get("structure_block", "")),
        "program": row.get("program", ""), "phase": op.get("phase", ""), "stage": stage, "layer": int(op.get("layer", 0)), "dtype": dtype,
        "local_shape": tuple(op.get("local_shape") or ()), "occurrences": count, "duration_ms": duration_ms, "total_ms": total_ms,
        "fixed_ms": fixed_ms, "minimum_ms": minimum_ms, "work_basis": "analytical_reference" if has_jn else "executed_service",
        "analytical_reference_work": ref_work, "flops": flops, "matrix_flops": matrix_flops, "vector_ops": vector_ops, "hbm_bytes": hbm_bytes,
        "mte1_bytes": mte1_bytes, "mte2_bytes": mte2_bytes, "mte3_bytes": mte3_bytes, "link_bytes": link_bytes, "logical_link_bytes": logical_link_bytes,
        "t_cube_ideal_ms": t_cube_ideal_ms, "t_vec_ideal_ms": t_vec_ideal_ms, "t_hbm_ideal_ms": t_hbm_ideal_ms, "t_link_ideal_ms": t_link_ideal_ms,
        "roofline_ideal_ms": roofline_ideal_ms, "cube_mfu": (t_cube_ideal_ms / safe_t) if flops > 0 else None, "vec_mfu": (t_vec_ideal_ms / safe_t) if vector_ops > 0 else None,
        "mbu": (t_hbm_ideal_ms / safe_t) if hbm_bytes > 0 else None, "link_util": (t_link_ideal_ms / safe_t) if link_bytes > 0 else None,
        "roofline_efficiency": roofline_efficiency, "operator_kind": operator_kind, "dominant_bound": dominant_bound,
        "cv_dominant_bound": cv_dominant_bound, "cv_roofline_efficiency": cv_roofline_efficiency, "cv_compute_share_ratio": cv_compute_share_ratio, "cv_overlap_ratio": cv_overlap_ratio,
    }
