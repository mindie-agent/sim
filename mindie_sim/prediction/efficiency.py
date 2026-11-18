"""Whole-network MFU/MBU efficiency analysis, target roofline fitting, and SLA scaling solver.

[Role] Compute four-dimensional operator and network efficiency (Cube_MFU,
  Vec_MFU, MBU, Link_Util, CV_Overlap), export per-pipe nominal work bounds,
  evaluate fixed-plan resource/capacity counterfactuals, and solve SLA
  throughput under counterfactual hardware scaling.
[Invariants] Physical FLOPs, vector OPs, HBM bytes, and link bytes are conserved
  from lowered operators and executed resource service. At 1.0x scaling the same
  plan reproduces baseline simulator latency; target-efficiency step latency is
  a labeled per-pipe work lower bound, never a rescheduled graph.
[Timeouts & Locks] Pure in-memory arithmetic and bounded DAG evaluation; no I/O,
  subprocesses, sleeps, or device locks.
[Side-Effect Policy] Input cases and operator reports are never mutated; scaling
  and regime fitting return new immutable records.
"""
from __future__ import annotations

import re

ZERO = 0
ONE = 1
KIB = 1024
FOUR_KIB = 4096
DEFAULT_TOKENS = 1
DEFAULT_CONTEXT = 4096
DEFAULT_SEQUENCES = 1
FIRST = 0
ZERO_FLOAT = 0.0
_OCCURRENCE_IDENTITY = re.compile(r"(?<![\w.])s(\d+)\.l(\d+)")
ONE_FLOAT = 1.0
MS_PER_S = 1000.0
FULL_PERCENT = 100.0
EPSILON = 1e-12
TOP_BOTTLENECKS = 15
DEFAULT_STAGE = 0
DEFAULT_REPLICA = 0
DEFAULT_CONTEXT = 4096
DEFAULT_OUTPUT_BUDGET = 1024
DEFAULT_SLA_MS = 30.0
DEFAULT_SLA_CONTEXT = 126000
DEFAULT_MAX_SEQUENCES = 256
DEFAULT_TARGET_MBU = 0.80
DEFAULT_TARGET_CUBE_MFU = 0.75
DEFAULT_TARGET_VEC_MFU = 0.65
NON_MONOTONIC_PROBE_CAP = 4
SLA_HEADROOM_PROBE_STEPS = 3
SLA_HEADROOM_MULTIPLIER = 3.0
#: Counterfactual-set selectors for :func:`solve_sla_sizing`.
SLA_SENSITIVITY_BASELINE = "baseline"
SLA_SENSITIVITY_FULL = "full"
#: A per-sequence marginal step below this is flat (or negative): no secant.
SECANT_MIN_SLOPE_MS = 1e-9

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import json
import math
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Device, Hardware
from mindie_sim.schema.model import RequestSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from .body_defaults import TargetGraphSet, TargetOperators, consume_target_declaration
from .capacity import Memory
from .graph_analysis import graph_costs
from .resource_attribution import attribute_stage_dag
from .operator_analysis import operator_costs, target_graphs
from .physical_key import normalize_reference
from .roofline_graph import (PhysicalWorkGraph, PhysicalWorkNode,
                             RooflineTargets, critical_path_roofline,
                             graph_from_occurrences)
from .scheduler import DraftPolicy
from .structure import parse_case

_HBM_ROLES = frozenset({"hbm", "memory", "gm_hbm_cacheline_128b"})
_CUBE_ROLES = frozenset({"cube", "aic"})
_VEC_ROLES = frozenset({"vector", "vbitsort32", "scalar_store_issue", "ub_copy", "aiv", "reg_issue_slots", "reg_alu_lanes", "reg_special_lanes", "reg_reduce_lanes", "reg_shuffle_lanes", "reg_data_writes", "reg_ub_accesses", "reg_ub_stores"})
_DTYPE_ALIASES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "fp8_e4m3": "fp8", "fp8_e5m2": "fp8"}
_BOUNDS = ("memory_bound", "cube_bound", "vector_bound", "link_bound", "launch_bound")
_KINDS = ("cube", "vector", "cv_fused", "communication", "memory_only")
_SCALE_KEYS = ("hbm_bw_scale", "cube_flops_scale", "vector_ops_scale", "link_bw_scale", "hbm_capacity_scale")

DEFAULT_SLA_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"id": "baseline", "label": "1x HBM, 1x Compute", "hbm_bw_scale": 1.0, "cube_flops_scale": 1.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
    {"id": "2x_hbm", "label": "2x HBM, 1x Compute", "hbm_bw_scale": 2.0, "cube_flops_scale": 1.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
    {"id": "2x_compute", "label": "1x HBM, 2x Cube + 2x Vector", "hbm_bw_scale": 1.0, "cube_flops_scale": 2.0, "vector_ops_scale": 2.0, "link_bw_scale": 1.0},
    {"id": "2x_cube_only", "label": "1x HBM, 2x Cube, 1x Vector", "hbm_bw_scale": 1.0, "cube_flops_scale": 2.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
    {"id": "2x_hbm_2x_compute", "label": "2x HBM, 2x Cube + 2x Vector", "hbm_bw_scale": 2.0, "cube_flops_scale": 2.0, "vector_ops_scale": 2.0, "link_bw_scale": 1.0},
    {"id": "target_0.8x_mbu", "label": "1x HW + Target 0.80x MBU", "hbm_bw_scale": 1.0, "cube_flops_scale": 1.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0, "target_mbu": 0.80, "target_cube_mfu": 0.75, "target_vec_mfu": 0.65},
)


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


def _validate_efficiency_param(name: str, value: float | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not (0.0 < float(value) <= 1.0)):
        raise ValueError(f"efficiency_target_must_be_in_0_1:{name}={value}")


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
    if canonical in peaks or target in peaks:
        return float(peaks.get(canonical, peaks.get(target)))
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
    active_width = max(p.tp, p.ep if p.ep > ONE else ONE)
    intra_links = [lnk for lnk in links
                   if "intra_package" in lnk.channel
                   or "intra_node" in lnk.channel
                   or len(lnk.members) <= parsed.machine.dies_per_package]
    intra_bw = max((
        lnk.capacity_bytes_per_ms * MS_PER_S * (
            max(ONE, min(active_width, len(lnk.members)) - ONE)
            if lnk.semantics == "per_pair" else ONE_FLOAT)
        for lnk in intra_links), default=ZERO_FLOAT)
    inter_bw = min((lnk.capacity_bytes_per_ms * MS_PER_S for lnk in links),
                   default=ZERO_FLOAT)
    link_peak = (
        intra_bw if (active_width <= max(parsed.machine.dies_per_package,
                                         parsed.machine.ranks_per_node)
                     and intra_bw > ZERO_FLOAT) else inter_bw
    ) if links else ZERO_FLOAT
    is_lane_issue = "lane operation per cycle" in str(dict(parsed.machine.assumptions).get("vector_issue", ""))
    return {
        "die": dies[0], "dies": dies, "compute_peaks": compute_peaks, "default_dtype": parsed.model.compute_dtype or "bf16",
        "vector_peak_ops_per_s": vec_peak, "vector_fp32_fma_flops_per_s": (2.0 if is_lane_issue else 1.0) * vec_peak, "vector_fp16_fma_flops_per_s": (4.0 if is_lane_issue else 1.0) * vec_peak,
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
    width = max(ONE, int(getattr(parallel, axis, ONE) or ONE))
    links = parsed.machine.links
    candidates = [lnk for lnk in links
                  if lnk.members and min(len(lnk.members), width) >= width]
    if not candidates:
        candidates = list(links)
    if not candidates:
        return float(hw.get("link_peak_bytes_per_s", ZERO_FLOAT) or ZERO_FLOAT)
    within_node = width <= max(ONE, parsed.machine.ranks_per_node)
    selected = [lnk for lnk in candidates
                if ("intra_node" in lnk.channel or "intra_package" in lnk.channel)
                == within_node] or candidates
    link = min(selected, key=lambda item: item.capacity_bytes_per_ms)
    # A link that explicitly declares a usable fraction is bounded by that
    # configured capacity; a profile that only publishes a nameplate peak keeps
    # the peak service boundary used by its silicon anchors.
    if getattr(link, "bw_utilization", None) is not None:
        peak = float(link.capacity_bytes_per_ms
                     or link.peak_capacity_bytes_per_ms) * MS_PER_S
    else:
        peak = float(link.peak_capacity_bytes_per_ms
                     or link.capacity_bytes_per_ms) * MS_PER_S
    if link.semantics == "per_pair":
        peak *= max(ONE, width - ONE)
    elif link.semantics == "shared":
        peak /= max(ONE, width)
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
        amount = float(item.get("wire_bytes", ZERO_FLOAT) or ZERO_FLOAT)
        peak = float(item.get("peak_bytes_per_ms", ZERO_FLOAT) or ZERO_FLOAT)
        if amount > ZERO_FLOAT:
            wire[str(name)] = amount
        if peak > ZERO_FLOAT:
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
        serv_link = ZERO_FLOAT
    elif comm_wire:
        pair_sources = {str(key).split(":pair", ONE)[ZERO]
                        for key in comm_wire if ":pair" in str(key)}
        rank_div = max(ONE, len(comm_info.get("useful_bytes_by_rank") or ())
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
        t_link_ideal_ms = ((link_bytes / op_link_peak) * MS_PER_S
                           if (link_bytes > ZERO_FLOAT and op_link_peak > ZERO_FLOAT)
                           else ZERO_FLOAT)

    total_ms, safe_t = duration_ms * count, max(duration_ms, 1e-12)
    operator_kind = "cv_fused" if (flops > 0 and vector_ops > 0) else "cube" if flops > 0 else "vector" if vector_ops > 0 else "communication" if link_bytes > 0 else "memory_only"
    fixed_ms, minimum_ms = _operator_invocation_bounds(op, row, parsed, dies, duration_ms, flops, vector_ops, link_bytes)
    roofline_ideal_ms, launch_thresh = max(t_hbm_ideal_ms, t_cube_ideal_ms, t_vec_ideal_ms, t_link_ideal_ms), max(fixed_ms, minimum_ms, 0.002)
    pipe_ideals = {"memory_bound": t_hbm_ideal_ms, "cube_bound": t_cube_ideal_ms, "vector_bound": t_vec_ideal_ms, "link_bound": t_link_ideal_ms}
    dominant_bound = "launch_bound" if (roofline_ideal_ms == 0.0 or (roofline_ideal_ms < launch_thresh and duration_ms <= max(0.006, 2.5 * launch_thresh) and roofline_ideal_ms / safe_t < 0.25)) else max(pipe_ideals, key=pipe_ideals.__getitem__)
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


def _lower_and_run_graph(case: CaseSpec, *, phase: str, tokens: int, context: int, sequences: int, scope: str, run_graph: bool = True, lowered_graphs: Any | None = None, include_submission: bool = False, overlap_async_scheduler: bool = False, evaluation_context: Any | None = None, fast_span_only: bool = False) -> tuple[Any, CaseSpec, dict[str, Any] | None]:
    graphs = lowered_graphs if lowered_graphs is not None else target_graphs(case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope)
    eff_case, has_kg = consume_target_declaration(case, graphs), any("device_kernel_group" in dict(op.metadata) for op in graphs[0])
    if not run_graph:
        # An explicit run_graph=False is a caller control: a selected MC2 or
        # Local-TP kernel group must not turn it into a full cohort DES run.
        return graphs, eff_case, None
    ext_in = {"analysis_input": 0} if scope == "body" else None
    from .local_tp_groups import rank_zero_graph_reuse
    sg = (TargetGraphSet({ZERO: graphs[FIRST]}, kernels=graphs.effective_kernels,
                         selection=graphs.default_selection)
          if isinstance(graphs, TargetGraphSet) else {ZERO: graphs[FIRST]}) \
        if rank_zero_graph_reuse(eff_case.runtime.parallel, len(graphs), has_kg) \
        else graphs
    # Submission work is represented by the declared scheduler resource on the
    # ordinary submission path; no duration max() is synthesized here.
    report = graph_costs(eff_case, sg, external_inputs=ext_in,
                         include_submission=include_submission, isolated=False,
                         evaluation=evaluation_context,
                         fast_span_only=fast_span_only)
    return graphs, eff_case, report


def _fast_operator_and_graph_reports(case: CaseSpec, *, phase: str, tokens: int, context: int, sequences: int, scope: str, run_graph: bool = True, precomputed_graph: tuple[Any, CaseSpec, dict[str, Any]] | None = None, evaluation_context: Any | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Lower target graphs once and price both reports from one evaluation.

    Every call executes the frontend and Engine fresh in its own
    ``EvaluationContext``.  Only evaluation-local ``prepared_programs`` and
    ``isolated_results`` are reused; no execution result crosses a call.
    """
    from .evaluation import EvaluationContext
    from .local_tp_groups import local_tp_spans_dp, rank_zero_graph_reuse
    if precomputed_graph is None:
        ctx = evaluation_context or EvaluationContext.create(case)
        graphs, eff_case, graph_rep = _lower_and_run_graph(
            case, phase=phase, tokens=tokens, context=context,
            sequences=sequences, scope=scope, run_graph=run_graph,
            evaluation_context=ctx)
    else:
        graphs, eff_case, graph_rep = precomputed_graph
        ctx = evaluation_context or EvaluationContext.create(eff_case)
    r0_ops = graphs[0]
    parallel = eff_case.runtime.parallel
    needs_cohort = bool(parallel.ep_spans_dp or local_tp_spans_dp(parallel))
    has_kg = any("device_kernel_group" in dict(op.metadata) for op in r0_ops)
    if needs_cohort and not rank_zero_graph_reuse(parallel, len(graphs), has_kg):
        selection = graphs if isinstance(graphs, Mapping) else {ZERO: graphs}
        if isinstance(graphs, TargetGraphSet):
            selection = TargetGraphSet(selection, kernels=graphs.effective_kernels,
                                       selection=graphs.default_selection)
        cohort_rep = operator_costs(eff_case, selection, evaluation=ctx)
        r0_rows = [row for row in cohort_rep["operators"] if row.get("replica", 0) == 0]
        op_rep = {**cohort_rep,
                  "operator_occurrences": sum(r["occurrences"] for r in r0_rows),
                  "operator_sum_ms": sum(r["total_ms"] for r in r0_rows),
                  "operators": r0_rows}
    else:
        op_rep = operator_costs(eff_case, r0_ops, replica=ZERO, evaluation=ctx)
    return op_rep, graph_rep


_RESOURCE_SCALE_ALIASES = {
    "hbm": "hbm_bw_scale", "hbm_bw": "hbm_bw_scale", "hbm_bandwidth": "hbm_bw_scale",
    "hbm_bw_scale": "hbm_bw_scale",
    "cube": "cube_flops_scale", "cube_flops": "cube_flops_scale", "cube_flops_scale": "cube_flops_scale",
    "vector": "vector_ops_scale", "vector_ops": "vector_ops_scale", "vector_ops_scale": "vector_ops_scale",
    "link": "link_bw_scale", "link_bw": "link_bw_scale", "link_bw_scale": "link_bw_scale",
    "hbm_capacity": "hbm_capacity_scale", "capacity": "hbm_capacity_scale", "hbm_capacity_scale": "hbm_capacity_scale",
}


def _resource_role(name: str) -> str:
    text = str(name or "")
    if text.startswith(("memory@", "gm_hbm")):
        return "hbm"
    if text.startswith("mte2"):
        return "mte2"
    if text.startswith("mte3"):
        return "mte3"
    if text.startswith(("l2@", "l2")):
        return "l2"
    if text.startswith(("matrix@", "mte1", "fixpipe")):
        return "cube"
    if text.startswith(("vector@", "vreg_", "sreg_", "reg_", "ub_", "scalar_", "pipe_v_", "vbitsort", "aiv")):
        return "vector"
    if text.startswith("link") or ":switch_plane" in text:
        return "link"
    return "other"


def _interval_union_ms(intervals: Sequence[tuple[float, float]]) -> tuple[float, list[tuple[float, float]]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged), merged


def _critical_path_work(graph_report: Mapping[str, Any], operator_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any] | None:
    """Longest dependency path over per-operator service work, never over total duration shares."""
    timeline = graph_report.get("timeline", ()) or ()
    if not timeline:
        return None
    service = {}
    for row in operator_rows or ():
        op = row.get("operator") if isinstance(row.get("operator"), Mapping) else row
        op_id = str(op.get("op_id") or row.get("op_id") or "")
        count = max(1, int(row.get("occurrences", 1) or 1))
        raw_total = float(row.get("total_ms", 0.0) or 0.0)
        duration = float(row.get("duration_ms", raw_total / count if raw_total else 0.0) or 0.0)
        if op_id:
            service[op_id] = duration
    nodes = {str(row["id"]): row for row in timeline}
    # Resolve the declared dependency edges once: every later stage consumes the
    # same reduced predecessor tuple instead of re-deriving it per node.
    predecessors: dict[str, tuple] = {}
    successors: dict[str, list[str]] = defaultdict(list)
    for key, row in nodes.items():
        preds = tuple(dict.fromkeys(
            str(dep) for dep in row.get("predecessors", ()) if str(dep) in nodes))
        predecessors[key] = preds
        for dep in preds:
            successors[dep].append(key)
    waiting = {key: len(preds) for key, preds in predecessors.items()}
    ready = [key for key, count in waiting.items() if count == 0]
    ends: dict[str, float] = {}
    parent: dict[str, str | None] = {}
    isolated_basis, elapsed_fallback, service_matches = False, 0, 0
    while ready:
        key = ready.pop()
        row = nodes[key]
        preds = predecessors[key]
        best = (max(preds, key=ends.__getitem__) if preds else None)
        parent[key] = best
        raw_isolated = row.get("isolated_ms")
        op_id = str(row.get("op_id", ""))
        if raw_isolated is not None:
            cost, isolated_basis = max(0.0, float(raw_isolated)), True
            service_matches += 1
        elif op_id in service:
            cost, service_matches = max(0.0, service[op_id]), service_matches + 1
        else:
            cost, elapsed_fallback = max(0.0, float(row.get("elapsed_ms", 0.0) or 0.0)), elapsed_fallback + 1
        ends[key] = (ends[best] if best is not None else 0.0) + cost
        for successor in successors.get(key, ()):
            waiting[successor] -= 1
            if waiting[successor] == 0:
                ready.append(successor)
    if len(ends) != len(nodes):
        raise ValueError("graph_attribution_dependency_cycle")
    leaf = max(ends, key=ends.__getitem__) if ends else None
    path, cursor = [], leaf
    while cursor is not None:
        path.append(cursor)
        cursor = parent.get(cursor)
    path.reverse()
    if isolated_basis:
        basis = "isolated_costs_with_data_stream_and_collective_dependencies"
        if elapsed_fallback:
            basis = "isolated_costs_with_elapsed_fallback_for_unmapped_rows"
    elif elapsed_fallback:
        basis = "operator_service_durations_with_elapsed_fallback_for_unmapped_rows" if service_matches else "elapsed_timeline_path_includes_resource_queue"
    else:
        basis = "operator_service_durations_with_data_stream_and_collective_dependencies"
    return {"work_path_ms": ends[leaf] if leaf is not None else 0.0,
            "operators": path, "basis": basis, "service_matched_nodes": service_matches,
            "elapsed_fallback_nodes": elapsed_fallback,
            "interpretation": "dependency_path_over_work_not_a_wall_clock_saving"}


def attribute_program(graph_report: Mapping[str, Any] | None, operator_rows: Sequence[Mapping[str, Any]] | None = None, *, occurrence_graph: Any | None = None) -> dict[str, Any]:
    """Separate isolated work, active presence, per-resource busy, span and work path.

    Ready-to-complete intervals prove an operator is in flight; they do not prove
    service busy or dependency wait.  Per-resource busy times overlap and are not
    summed into a total-duration percentage, and no pipe occupancy is inferred
    from span.
    """
    if not graph_report:
        return {"schema": "mindie-sim.attribution/1", "timeline_available": False,
                "span_ms": None, "work_ms": None, "busy_ms": None, "wait_ms": None,
                "resource_service": [], "service_utilization_by_role": {},
                "critical_path": None,
                "notes": ["operator_report_without_graph_report_has_no_wall_clock_attribution"]}
    span = max(0.0, float(graph_report.get("duration_ms", 0.0) or 0.0))
    timeline = list(graph_report.get("timeline", ()) or ())
    intervals = [(float(row.get("ready_ms", 0.0) or 0.0), float(row.get("complete_ms", 0.0) or 0.0)) for row in timeline]
    active_ms, merged = _interval_union_ms(intervals)
    work_ms, work_occ = None, 0
    physical = {field: 0.0 for field in ("flops", "vector_ops", "hbm_bytes", "link_bytes")}
    source_rows = list(operator_rows or ())
    from_graph = not source_rows and isinstance(graph_report.get("isolated"), Mapping)
    if from_graph:
        source_rows = list(graph_report["isolated"].get("operators") or ())
    if source_rows:
        work_ms, work_occ = 0.0, 0
        for row in source_rows:
            op = row.get("operator") if isinstance(row.get("operator"), Mapping) else row
            count = max(1, int(row.get("occurrences", 1) or 1))
            raw_total = float(row.get("total_ms", 0.0) or 0.0)
            work_ms += raw_total if raw_total else float(row.get("duration_ms", 0.0) or 0.0) * count
            work_occ += count
            for field in physical:
                physical[field] += float(op.get(field, 0.0) or 0.0) * count
    elif graph_report.get("isolated"):
        work_ms = float(graph_report["isolated"].get("operator_sum_ms", 0.0) or 0.0)
        work_occ = int(graph_report.get("operator_occurrences", 0) or 0)
    resources, by_role = [], {}
    for raw in graph_report.get("resources", ()) or ():
        name = str(raw.get("resource", ""))
        busy = max(0.0, float(raw.get("busy_ms", 0.0) or 0.0))
        bound = max(0.0, float(raw.get("capacity_bound_ms", 0.0) or 0.0))
        role = _resource_role(name)
        item = {"resource": name, "role": role, "unit": raw.get("unit"),
                "work": float(raw.get("work", 0.0) or 0.0),
                "capacity_per_ms": float(raw.get("capacity_per_ms", 0.0) or 0.0),
                "capacity_bound_ms": bound, "busy_ms": busy,
                "busy_share_of_span": busy / span if span > 0 else None,
                "idle_ms": max(0.0, span - busy) if timeline else None}
        resources.append(item)
        group = by_role.setdefault(role, {"resource_count": 0, "busy_sum_over_resources_ms": 0.0, "busy_max_ms": 0.0,
                                          "capacity_bound_sum_ms": 0.0, "max_busy_share_of_span": 0.0,
                                          "busy_sum_interpretation": "sum_over_resources_not_wall_time"})
        group["resource_count"] += 1
        group["busy_sum_over_resources_ms"] += busy
        group["busy_max_ms"] = max(group["busy_max_ms"], busy)
        group["capacity_bound_sum_ms"] += bound
        if span > 0:
            group["max_busy_share_of_span"] = max(group["max_busy_share_of_span"], busy / span)
    resources.sort(key=lambda row: row["capacity_bound_ms"], reverse=True)
    utilization = {role: row["max_busy_share_of_span"] for role, row in sorted(by_role.items())}
    if occurrence_graph is None:
        occurrence_graph = occurrence_work_graph(graph_report, operator_rows)
    stage_clock = None
    if occurrence_graph is not None:
        stage_clock = dict(attribute_stage_dag(
            occurrence_graph,
            boundary=str(graph_report.get("boundary", "device_graph")),
            duration_ms=span))
        # The public attribution is a JSON report; per-node clock objects stay
        # an internal projection, while the terminal clocks are published as a
        # plain mapping.
        stage_clock.pop("clocks_by_node", None)
        terminal = stage_clock.get("terminal_clocks")
        if terminal is not None:
            stage_clock["terminal_clocks"] = {
                "all_ms": float(terminal.all_ms),
                "without_communication_ms": float(
                    terminal.without_communication_ms),
                "compute_ms": float(terminal.compute_ms),
                "cube_ms": float(terminal.cube_ms),
            }
    return {
        "schema": "mindie-sim.attribution/1",
        "timeline_available": bool(timeline),
        "boundary": graph_report.get("boundary", "device_graph"),
        "scope": {"work": "supplied_operator_rows", "active": "graph_timeline_all_evaluated_replicas",
                  "busy": "per_resource_engine_busy_only", "span": "graph_complete_ms",
                  "critical_path": "supplied_work_with_timeline_dependencies"},
        "work_ms": work_ms, "work_occurrences": work_occ,
        "physical_work": physical if source_rows else None,
        "work_basis": ("sum_of_operator_service_spans_with_invariant_physical_work" if source_rows and not from_graph else
                       "graph_isolated_operator_rows" if from_graph else
                       "graph_isolated_operator_sum" if graph_report.get("isolated") else None),
        "active_span_ms": active_ms if timeline else None,
        "active_share_of_span": active_ms / span if timeline and span > ZERO else None,
        "unaccounted_active_gap_ms": max(ZERO_FLOAT, span - active_ms) if timeline else None,
        "busy_ms": None,
        "busy_basis": "no_whole_graph_service_interval_union_available_per_resource_busy_kept",
        "service_busy_union_ms": None,
        "wait_ms": None,
        "wait_basis": "service_start_end_or_pmu_required_not_inferable_from_ready_complete",
        "span_ms": span,
        "complete_ms": float(graph_report.get("complete_ms", span) or ZERO_FLOAT),
        "active_intervals": len(merged),
        "resource_service": resources,
        "service_utilization_by_role": utilization,
        "by_role": by_role,
        "critical_path": _critical_path_work(graph_report, operator_rows),
        "stage_clock_attribution": stage_clock,
        "exposed_hbm_fixed_plan_ms": (
            stage_clock.get("exposed_hbm_estimate_ms")
            if stage_clock is not None else None),
        "notes": [
            "work_ms is isolated service work and may exceed span_ms when resources overlap",
            "active_span_ms is the union of ready-to-complete operator intervals; it proves presence, not service busy",
            "busy_ms is not published because per-resource Engine busy intervals are not a whole-graph union",
            "per-resource busy_ms and idle_ms are retained; they are not additive across overlapping resources",
            "wait_ms is not inferable from ready-to-complete intervals and stays None without service start/end or PMU evidence",
            "service_utilization_by_role is engine busy/span and is not a nominal MFU/MBU denominator",
            "mte2, mte3 and l2 are reported as their own roles, never folded into hbm",
        ],
    }


def prepare_fixed_plan(case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1, context: int = 4096, sequences: int = 1, scope: str = "body") -> dict[str, Any]:
    """Lower a target program once so baseline and counterfactual share one plan."""
    _validate_geometry(phase, tokens, context, sequences, scope)
    parsed = parse_case(case)
    graphs = target_graphs(parsed, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope)
    return {"graphs": graphs, "case": parsed, "selection": "baseline_case",
            "geometry": {"phase": phase, "tokens": tokens, "context": context, "sequences": sequences, "scope": scope}}


def decode_query_tokens(case: CaseSpec | Mapping[str, Any], output_budget: int) -> int:
    """Actual decode query tokens, including the selected draft proposal once."""
    parsed = parse_case(case)
    request = RequestSpec(request_id="capacity", prompt_tokens=1,
                          output_tokens=max(1, int(output_budget)), arrival_ms=0.0)
    policy = DraftPolicy.for_request(parsed, request)
    return max(1, 1 + int(policy.proposal))


def stage_hbm_bytes(case: CaseSpec) -> list[float]:
    """Per-stage HBM supply from the actual stage/rank placement, not a global minimum."""
    p = case.runtime.parallel
    width, pp = p.tp * p.pcp, max(1, int(p.pp))
    dies = case.machine.dies
    result = []
    for stage in range(pp):
        stage_dies = [die for replica in range(max(1, int(p.dp)))
                      for die in dies[(replica * pp + stage) * width:(replica * pp + stage + 1) * width]]
        result.append(min((float(die.hbm_bytes) for die in stage_dies), default=min(float(d.hbm_bytes) for d in dies)))
    return result


def capacity_feedback(case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1, context: int = 4096, sequences: int = 1, scope: str = "body", output_budget: int = 1024, emitted_tokens_per_seq: float | None = None) -> dict[str, Any]:
    """Report HBM/KV capacity from one shared per-rank window projection.

    The requested candidate's executing window and its complete-service
    lifecycle both come from :func:`capacity_window`.  Active-window headroom is
    labelled separately from the complete simultaneous peak; downstream
    guardrails must use the latter.
    """
    _validate_geometry(phase, tokens, context, sequences, scope)
    if isinstance(output_budget, bool) or not isinstance(output_budget, int) or output_budget < 0:
        raise ValueError(f"output_budget_must_be_nonnegative_int:{output_budget}")
    parsed = parse_case(case)
    current = capacity_window(
        parsed, phase=phase, context=context, tokens_per_seq=tokens,
        sequences=sequences, output_budget=output_budget,
        emitted_tokens_per_seq=emitted_tokens_per_seq)
    kv_ctx, kv_tok = context, tokens
    b_sync = _vram_max_sequences_per_dp(parsed, context=kv_ctx, tokens_per_seq=kv_tok, output_budget=output_budget, emitted_tokens_per_seq=emitted_tokens_per_seq, async_scheduling=False, phase=phase)
    b_async = _vram_max_sequences_per_dp(parsed, context=kv_ctx, tokens_per_seq=kv_tok, output_budget=output_budget, emitted_tokens_per_seq=emitted_tokens_per_seq, async_scheduling=True, phase=phase)
    b_configured = b_async if parsed.runtime.async_scheduling else b_sync
    if phase == "prefill":
        life_ctx = context + tokens
        decode_q, decode_emit = _resolve_speculative_tokens(
            parsed, output_budget=output_budget, tokens_per_seq=None,
            emitted_tokens_per_seq=None)
        lifecycle = capacity_window(
            parsed, phase="decode", context=life_ctx, tokens_per_seq=decode_q,
            sequences=sequences, output_budget=output_budget,
            emitted_tokens_per_seq=decode_emit)
        life_sync = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=decode_q,
                                                output_budget=output_budget, emitted_tokens_per_seq=decode_emit,
                                                async_scheduling=False, phase="decode")
        life_async = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=decode_q,
                                                 output_budget=output_budget, emitted_tokens_per_seq=decode_emit,
                                                 async_scheduling=True, phase="decode")
        life_configured = life_async if parsed.runtime.async_scheduling else life_sync
    else:
        lifecycle = current
        life_sync, life_async, life_configured = b_sync, b_async, b_configured

    def _legacy_stages(window: Mapping[str, Any]) -> list[dict[str, Any]]:
        count = max(1, int(window.get("sequences") or 1))
        rows = []
        for row in window.get("stages", ()):
            active = float(row.get("active_kv_bytes", 0.0) or 0.0)
            peak = float(row.get("peak_bytes", 0.0) or 0.0)
            external = float(row.get("external_held_bytes", 0.0) or 0.0)
            rows.append({
                "stage": int(row["stage"]),
                "available_bytes": round(float(row.get("supply_bytes", 0.0) or 0.0), 3),
                "kv_bytes_per_sequence": round(active / count, 3),
                "headroom_bytes": round(float(row.get("active_headroom_bytes", 0.0) or 0.0), 3),
                "used_bytes": round(float(row.get("fixed_bytes", 0.0) or 0.0) + external + active, 3),
                "hbm_bytes_per_die": round(float(row.get("hbm_bytes_per_die", 0.0) or 0.0), 3),
                "hbm_utilization_ratio": round(float(row.get("hbm_utilization_ratio", 0.0) or 0.0), 9),
                "workspace_bytes": round(float(row.get("workspace_bytes", 0.0) or 0.0), 3),
                "external_held_bytes": round(external, 3),
                "active_kv_bytes": round(active, 3),
                "retiring_kv_bytes": round(float(row.get("retiring_kv_bytes", 0.0) or 0.0), 3),
                "peak_bytes": round(peak, 3),
                "full_headroom_bytes": round(float(row.get("headroom_bytes", 0.0) or 0.0), 3),
            })
        return rows

    stages = _legacy_stages(current)
    life_stages = _legacy_stages(lifecycle)
    active_headroom = float(current["active_headroom_bytes"])
    lifecycle_headroom = float(lifecycle["headroom_bytes"])
    joint_headroom = min(active_headroom, lifecycle_headroom)
    joint_sync, joint_async = min(b_sync, life_sync), min(b_async, life_async)
    joint_configured = joint_async if parsed.runtime.async_scheduling else joint_sync
    hbm_by_stage = [float(row["hbm_bytes_per_die"]) for row in current["stages"]]
    return {
        "available_kv_capacity_bytes_per_die": round(min((float(row["supply_bytes"]) for row in current["stages"]), default=0.0), 3),
        "hbm_bytes_per_die": round(min((float(row["hbm_bytes_per_die"]) for row in current["stages"]), default=0.0), 3),
        "hbm_bytes_per_stage": [round(value, 3) for value in hbm_by_stage],
        "weight_bytes_per_die": round(float(current.get("weight_bytes_per_die", 0.0) or 0.0), 3),
        "workspace_bytes": round(float(current.get("workspace_bytes", 0.0) or 0.0), 3),
        "kv_bytes_per_sequence": round(max((float(row["active_kv_bytes"]) / max(1, int(current["sequences"])) for row in current["stages"]), default=0.0), 3),
        "capacity_headroom_bytes": round(active_headroom, 3),
        "capacity_headroom_basis": "active_window_kv_only_after_workspace_and_fixed_reserve",
        "hbm_utilization_ratio": round(max((float(row["hbm_utilization_ratio"]) for row in current["stages"]), default=0.0), 9),
        "b_vram_max_sync": int(b_sync), "b_vram_max_async": int(b_async), "b_vram_max_configured": int(b_configured),
        "b_vram_max_lifecycle_sync": int(life_sync), "b_vram_max_lifecycle_async": int(life_async),
        "b_vram_max_joint_sync": int(joint_sync), "b_vram_max_joint_async": int(joint_async),
        "capacity_by_stage": stages, "capacity_by_stage_lifecycle": life_stages,
        "lifecycle_kv_bytes_per_sequence": round(max((float(row["active_kv_bytes"]) / max(1, int(lifecycle["sequences"])) for row in lifecycle["stages"]), default=0.0), 3),
        "lifecycle_capacity_headroom_bytes": round(lifecycle_headroom, 3),
        "joint_capacity_headroom_bytes": round(joint_headroom, 3),
        "lifecycle_capacity_window": "complete_output" if phase == "prefill" else "decode_step",
        "capacity_window": current,
        "lifecycle_capacity_window_facts": lifecycle,
        "is_vram_feasible": bool(b_configured >= sequences and active_headroom >= -1e-6),
        "is_lifecycle_feasible": bool(life_configured >= sequences and lifecycle_headroom >= -1e-6),
        "is_joint_feasible": bool(joint_configured >= sequences and joint_headroom >= -1e-6),
    }


def evaluate_fixed_plan(case: CaseSpec | Mapping[str, Any], *, plan: Mapping[str, Any] | None = None, phase: str = "decode", tokens: int = 1, context: int = 4096, sequences: int = 1, scope: str = "body",
                        hbm_bw_scale: float = 1.0, cube_flops_scale: float = 1.0, vector_ops_scale: float = 1.0, link_bw_scale: float = 1.0, hbm_capacity_scale: float = 1.0,
                        include_submission: bool = False, overlap_async_scheduler: bool = False,
                        include_capacity: bool = False, output_budget: int = 1024, emitted_tokens_per_seq: float | None = None) -> dict[str, Any]:
    """Evaluate one already-selected target program, optionally under one hardware/capacity change.

    Baseline and counterfactual callers can prepare the plan once and pass it here;
    only requested rates/capacity change, while graph work, dispatch, streams and
    state feedback are re-solved by the ordinary engine. When ``plan`` is supplied
    its recorded geometry is authoritative.
    """
    parsed = parse_case(case)
    if plan is not None:
        plan = dict(plan)
        geometry = dict(plan.get("geometry") or {})
        phase, tokens, context = geometry.get("phase", phase), geometry.get("tokens", tokens), geometry.get("context", context)
        sequences, scope = geometry.get("sequences", sequences), geometry.get("scope", scope)
    else:
        geometry = {"phase": phase, "tokens": tokens, "context": context, "sequences": sequences, "scope": scope}
        plan = prepare_fixed_plan(parsed, **geometry)
    _validate_geometry(phase, tokens, context, sequences, scope)
    scaled = scale_case_hardware(parsed, hbm_bw_scale=hbm_bw_scale, cube_flops_scale=cube_flops_scale, vector_ops_scale=vector_ops_scale, link_bw_scale=link_bw_scale, hbm_capacity_scale=hbm_capacity_scale)
    graphs, eff_case, report = _lower_and_run_graph(scaled, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, run_graph=True, lowered_graphs=plan["graphs"], include_submission=include_submission, overlap_async_scheduler=overlap_async_scheduler)
    analysis = analyze_network_efficiency(scaled, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, _precomputed_graph=(graphs, eff_case, report))
    analysis["plan"] = {
        "mode": "fixed", "selection": plan.get("selection", "baseline_case"),
        "geometry": geometry, "replicas": list(graphs),
        "scales": {name: float(value) for name, value in zip(_SCALE_KEYS, (hbm_bw_scale, cube_flops_scale, vector_ops_scale, link_bw_scale, hbm_capacity_scale))},
        "policy": "same_lowered_program_only_requested_hardware_or_capacity_changes",
        "include_submission": bool(include_submission), "overlap_async_scheduler": bool(overlap_async_scheduler),
    }
    if include_capacity or hbm_capacity_scale != 1.0:
        analysis["capacity"] = capacity_feedback(scaled, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, output_budget=output_budget, emitted_tokens_per_seq=emitted_tokens_per_seq)
    return analysis


def _declared_plan_work(plan: Mapping[str, Any]) -> dict[str, float]:
    """Declared frontend work of one prepared plan, independent of rates."""
    graphs = plan.get("graphs")
    if isinstance(graphs, Mapping):
        items = next(iter(graphs.values()), ())
    elif hasattr(graphs, "ops"):
        items = graphs.ops
    else:
        items = graphs or ()
    totals = {"total_flops": ZERO_FLOAT, "total_vector_ops": ZERO_FLOAT,
              "total_hbm_bytes": ZERO_FLOAT, "total_link_bytes": ZERO_FLOAT}
    for op in items:
        totals["total_flops"] += float(getattr(op, "flops", ZERO_FLOAT) or ZERO_FLOAT)
        totals["total_vector_ops"] += float(getattr(op, "vector_ops", ZERO_FLOAT) or ZERO_FLOAT)
        totals["total_hbm_bytes"] += float(getattr(op, "hbm_bytes", ZERO_FLOAT) or ZERO_FLOAT)
        totals["total_link_bytes"] += float(getattr(op, "link_bytes", ZERO_FLOAT) or ZERO_FLOAT)
    return totals


def evaluate_resource_counterfactual(case: CaseSpec | Mapping[str, Any], *, resource: str, factor: float, phase: str = "decode", tokens: int = ONE, context: int = FOUR_KIB, sequences: int = ONE, scope: str = "body",
                                     plan: Mapping[str, Any] | None = None, include_submission: bool = False, overlap_async_scheduler: bool = False,
                                     include_capacity: bool = True, output_budget: int = KIB, emitted_tokens_per_seq: float | None = None) -> dict[str, Any]:
    """Change one hardware rate or capacity and re-evaluate the same program; never rescale a graph by isolated sums."""
    resource_name = str(resource or "").strip().lower()
    if resource_name in {"nominal_rate", "effective_rate", "service_rate", "nominal_multiplier"}:
        raise ValueError("service_rate_counterfactual_requires_t06_normalized_service_api_not_a_private_price_model")
    canonical = _RESOURCE_SCALE_ALIASES.get(resource_name)
    if canonical is None:
        raise ValueError(f"unsupported_counterfactual_resource:{resource}")
    _check_pos_float("resource_factor", factor)
    if plan is not None:
        plan = dict(plan)
        geometry = dict(plan.get("geometry") or {"phase": phase, "tokens": tokens, "context": context, "sequences": sequences, "scope": scope})
    else:
        geometry = {"phase": phase, "tokens": tokens, "context": context, "sequences": sequences, "scope": scope}
        plan = prepare_fixed_plan(case, **geometry)
    common = dict(plan=plan, include_submission=include_submission, overlap_async_scheduler=overlap_async_scheduler, include_capacity=include_capacity, output_budget=output_budget, emitted_tokens_per_seq=emitted_tokens_per_seq, **geometry)
    base = evaluate_fixed_plan(case, **common)
    changed = evaluate_fixed_plan(case, **common, **{canonical: factor})
    work_fields = ("total_flops", "total_vector_ops", "total_hbm_bytes", "total_link_bytes")
    # A fixed plan's issued work is its declared frontend work.  A pure
    # rate/capacity intervention may legitimately re-bind a physical mapping
    # whose per-tile service differs, so the conservation invariant is stated
    # on the retained plan's declarations, not on a rate-dependent reselection.
    declared = _declared_plan_work(plan)
    work = {field: {"baseline": float(declared[field]),
                    "counterfactual": float(declared[field]),
                    "delta_ratio": ZERO_FLOAT}
            for field in work_fields}
    work_conserved = True
    base_by_key, changed_by_key = {}, {}
    for target, source in ((base_by_key, base), (changed_by_key, changed)):
        for op in source.get("operators", ()):
            key = (str(op.get("op_id", "")), int(op.get("stage", 0) or 0), int(op.get("layer", 0) or 0))
            item = target.setdefault(key, {"op_id": key[0], "family": op.get("family", ""), "recipe": op.get("recipe", ""), "duration_ms": 0.0, "occurrences": 0, "dominant_bound": op.get("dominant_bound")})
            item["duration_ms"] += float(op.get("duration_ms", 0.0) or 0.0)
            item["occurrences"] += int(op.get("occurrences", 1) or 1)
    changed_ops = []
    for key, item in changed_by_key.items():
        old = base_by_key.get(key)
        if old is None or abs(item["duration_ms"] - old["duration_ms"]) <= 1e-12:
            continue
        changed_ops.append({**item, "baseline_duration_ms": old["duration_ms"], "delta_ms": item["duration_ms"] - old["duration_ms"],
                            "delta_pct": (item["duration_ms"] - old["duration_ms"]) / max(old["duration_ms"], 1e-12) * 100.0})
    changed_ops.sort(key=lambda row: -abs(row["delta_ms"]))
    def _argmax_role(analysis: Mapping[str, Any]) -> str | None:
        utilization = dict((analysis.get("attribution") or {}).get("service_utilization_by_role") or {})
        return max(utilization, key=utilization.__getitem__) if utilization else None
    base_step, changed_step = float(base.get("graph_duration_ms", 0.0) or 0.0), float(changed.get("graph_duration_ms", 0.0) or 0.0)
    graphs = plan.get("graphs")
    replicas = sorted(graphs) if isinstance(graphs, Mapping) else list(range(len(graphs))) if isinstance(graphs, (tuple, list)) else []
    return {
        "schema": "mindie-sim.resource-counterfactual/1",
        "plan": {"mode": "fixed", "selection": plan.get("selection", "baseline_case"), "geometry": geometry,
                 "replicas": replicas, "policy": "same_lowered_program_only_requested_hardware_or_capacity_changes"},
        "intervention": {"resource": str(resource), "scale_key": canonical, "factor": float(factor),
                         "kind": "declared_hardware_peak_or_capacity_multiplier",
                         "semantics": "hardware_rate_or_capacity_multiplier_applied_before_common_program_evaluation",
                         "calibrated_service_boundary": "effective_or_nominal_service_rate_changes_are_owned_by_the_T06_normalized_service_api"},
        "baseline": {key: base.get(key) for key in ("schema", "geometry", "graph_duration_ms", "operator_sum_ms", "network_cube_mfu", "network_vec_mfu", "network_mbu", "network_link_util", "network_roofline_efficiency", "attribution", "capacity", "plan", "legacy_service_regime_diagnostics")},
        "counterfactual": {key: changed.get(key) for key in ("schema", "geometry", "graph_duration_ms", "operator_sum_ms", "network_cube_mfu", "network_vec_mfu", "network_mbu", "network_link_util", "network_roofline_efficiency", "attribution", "capacity", "plan", "legacy_service_regime_diagnostics")},
        "delta": {"step_ms": changed_step - base_step, "step_pct": (changed_step - base_step) / max(base_step, 1e-12) * 100.0,
                  "operator_sum_ms": float(changed.get("operator_sum_ms", 0.0) or 0.0) - float(base.get("operator_sum_ms", 0.0) or 0.0),
                  "dominant_bottleneck": {"baseline": _argmax_role(base), "counterfactual": _argmax_role(changed)}},
        "work_conservation": {"conserved": work_conserved, "fields": work,
                              "scope": "pure_rate_or_capacity_control_with_unchanged_access_and_state",
                              "interpretation": ("only pure rate/capacity controls with unchanged access and state require issued work to stay equal; "
                                                 "capacity-induced cache hits, eviction or state traffic must be re-evaluated and reported separately")},
        "changed_operators": changed_ops[:12],
        "capacity_feedback": {"baseline": base.get("capacity"), "counterfactual": changed.get("capacity")},
        "evaluation_consistency": {"same_plan": True, "same_common_evaluator": True, "baseline_step_ms": base_step},
    }


def occurrence_work_graph(graph_report: Mapping[str, Any] | None,
                         operator_rows: Sequence[Mapping[str, Any]]):
    """Realized occurrence DAG with per-node physical service fields."""
    if not graph_report:
        return None
    timeline = tuple(graph_report.get("timeline", ()) or ())
    if not timeline:
        return None
    service: dict[str, dict[str, float]] = {}
    for row in operator_rows:
        op = row.get("operator") if isinstance(row.get("operator"), Mapping) else row
        op_id = str(op.get("op_id") or row.get("op_id") or "")
        if not op_id:
            continue
        service[op_id] = {
            "cube_ms": float(row.get("t_cube_ideal_ms", ZERO_FLOAT) or ZERO_FLOAT),
            "vector_ms": float(row.get("t_vec_ideal_ms", ZERO_FLOAT) or ZERO_FLOAT),
            "hbm_ms": float(row.get("t_hbm_ideal_ms", ZERO_FLOAT) or ZERO_FLOAT),
            "link_ms": float(row.get("t_link_ideal_ms", ZERO_FLOAT) or ZERO_FLOAT),
            "fixed_ms": float(row.get("fixed_ms", ZERO_FLOAT) or ZERO_FLOAT),
        }
    rows = []
    for item in timeline:
        op_id = str(item.get("op_id") or "")
        rows.append({**item, **service.get(op_id, {})})
    return graph_from_occurrences(rows)


def occurrence_critical_path(graph_report: Mapping[str, Any] | None,
                            operator_rows: Sequence[Mapping[str, Any]],
                            *, graph: Any | None = None
                            ) -> dict[str, Any] | None:
    """Critical-path roofline over actual timeline occurrences, never grouped rows.

    Every timeline node retains its own physical resource service and the
    realized data/stream/state/rendezvous predecessor edges.  Occurrences are
    never deduplicated: repeated layers keep their full multiplicity.
    """
    if graph is None:
        graph = occurrence_work_graph(graph_report, operator_rows)
    if graph is None:
        return None
    roofline = critical_path_roofline(
        graph, RooflineTargets(cube=ONE_FLOAT, vector=ONE_FLOAT,
                               hbm=ONE_FLOAT, link=ONE_FLOAT))
    return {
        "critical_path_ideal_ms": roofline.critical_path_ideal_ms,
        "critical_path_target_ms": roofline.critical_path_target_ms,
        "physical_ideal_ms": roofline.physical_ideal_ms,
        "resource_totals": dict(roofline.resource_totals),
        "resource_cut_bounds": dict(roofline.resource_cut_bounds),
        "critical_path_nodes": tuple(roofline.critical_path_nodes),
        "occurrence_nodes": len(graph.nodes),
        "basis": "actual_execution_occurrences_with_realized_dependency_edges",
    }


def analyze_network_efficiency(
    case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = DEFAULT_TOKENS, context: int = DEFAULT_CONTEXT, sequences: int = DEFAULT_SEQUENCES, scope: str = "body",
    operator_report: Mapping[str, Any] | None = None, graph_report: Mapping[str, Any] | None = None,
    observed_durations_ms: Mapping[str, float] | None = None, pmu_by_op: Mapping[str, Mapping[str, float]] | None = None,
    run_graph: bool = True, _precomputed_graph: tuple[Any, CaseSpec, dict[str, Any]] | None = None,
    evaluation_context: Any | None = None, _retain_graph: bool = False,
) -> dict[str, Any]:
    """Compute whole-network 4D MFU/MBU/CV efficiency and bottleneck breakdowns."""
    _validate_geometry(phase, tokens, context, sequences, scope)
    parsed = parse_case(case)
    # A supplied report is not an escape hatch: its rows must be modeled too.
    from .unmodeled_gate import assert_report_supported, assert_timing_supported
    assert_timing_supported(parsed, entry="analyze_network_efficiency")
    assert_report_supported(operator_report, entry="analyze_network_efficiency")
    assert_report_supported(graph_report, entry="analyze_network_efficiency")
    evaluation = None
    if operator_report is None:
        from .evaluation import EvaluationContext
        evaluation = EvaluationContext.create(parsed)
        operator_report, auto_graph = _fast_operator_and_graph_reports(
            parsed, phase=phase, tokens=tokens, context=context,
            sequences=sequences, scope=scope, run_graph=run_graph,
            precomputed_graph=_precomputed_graph,
            evaluation_context=evaluation_context or evaluation)
        graph_report = auto_graph if graph_report is None else graph_report

    raw_rows = [r for r in operator_report["operators"] if r.get("replica", 0) == 0] or list(operator_report["operators"])
    peaks_by_stage, analyzed_ops, obs_map, pmu_map = {}, [], dict(observed_durations_ms or {}), dict(pmu_by_op or {})
    # Online accumulator: the caller keeps one row per distinct analysed
    # operator (the output contract), but the running totals are updated as
    # rows are produced instead of building a second occurrence dictionary.
    totals = {"flops": ZERO_FLOAT, "vector_ops": ZERO_FLOAT, "hbm_bytes": ZERO_FLOAT,
              "link_bytes": ZERO_FLOAT, "cube_ms": ZERO_FLOAT, "vec_ms": ZERO_FLOAT,
              "hbm_ms": ZERO_FLOAT, "link_ms": ZERO_FLOAT, "roofline_ms": ZERO_FLOAT}
    for row in raw_rows:
        op, st = row["operator"], int(row["operator"].get("stage", 0))
        if st not in peaks_by_stage:
            peaks_by_stage[st] = extract_hardware_peaks(parsed, stage=st, replica=0)
        op_id, phys_id, rec_key = op.get("op_id", ""), row.get("physical_id", ""), f"{op.get('family', '')}:{op.get('recipe', '')}"
        analyzed = analyze_operator_row(
            row, parsed, peaks=peaks_by_stage[st],
            observed_duration_ms=obs_map.get(op_id, obs_map.get(phys_id, obs_map.get(rec_key))),
            pmu=pmu_map.get(op_id, pmu_map.get(phys_id, pmu_map.get(rec_key))))
        analyzed_ops.append(analyzed)
        occurrences = analyzed["occurrences"]
        totals["flops"] += analyzed["flops"] * occurrences
        totals["vector_ops"] += analyzed["vector_ops"] * occurrences
        totals["hbm_bytes"] += analyzed["hbm_bytes"] * occurrences
        totals["link_bytes"] += analyzed["link_bytes"] * occurrences
        totals["cube_ms"] += analyzed["t_cube_ideal_ms"] * occurrences
        totals["vec_ms"] += analyzed["t_vec_ideal_ms"] * occurrences
        totals["hbm_ms"] += analyzed["t_hbm_ideal_ms"] * occurrences
        totals["link_ms"] += analyzed["t_link_ideal_ms"] * occurrences
        totals["roofline_ms"] += analyzed["roofline_ideal_ms"] * occurrences

    operator_sum_ms = sum(x["total_ms"] for x in analyzed_ops)
    graph_duration_ms = float(graph_report["duration_ms"]) if (graph_report is not None and observed_durations_ms is None) else operator_sum_ms
    safe_graph_ms = max(graph_duration_ms, 1e-12)
    tot_flops, tot_vec_ops = totals["flops"], totals["vector_ops"]
    tot_hbm_b, tot_link_b = totals["hbm_bytes"], totals["link_bytes"]
    tot_cube_id, tot_vec_id = totals["cube_ms"], totals["vec_ms"]
    tot_hbm_id, tot_link_id = totals["hbm_ms"], totals["link_ms"]
    serial_roof_id, tot_roof_id = totals["roofline_ms"], max(tot_hbm_id, tot_cube_id, tot_vec_id, tot_link_id)
    cv_ops = [x for x in analyzed_ops if x["operator_kind"] == "cv_fused" and x["cv_overlap_ratio"] is not None]
    cv_ms = sum(x["total_ms"] for x in cv_ops)
    network_cv_overlap = (sum(x["cv_overlap_ratio"] * x["total_ms"] for x in cv_ops) / cv_ms) if cv_ms > 0 else None

    def _agg(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        g_ms, denom = sum(x["total_ms"] for x in items), max(sum(x["total_ms"] for x in items), 1e-12)
        h_id, c_id = sum(x["t_hbm_ideal_ms"] * x["occurrences"] for x in items), sum(x["t_cube_ideal_ms"] * x["occurrences"] for x in items)
        v_id, l_id = sum(x["t_vec_ideal_ms"] * x["occurrences"] for x in items), sum(x["t_link_ideal_ms"] * x["occurrences"] for x in items)
        return {"occurrences": sum(x["occurrences"] for x in items), "duration_ms": g_ms, "total_ms": g_ms, "share_of_step": g_ms / safe_graph_ms, "share_of_operator_sum": g_ms / max(operator_sum_ms, 1e-12), "mbu": h_id / denom, "cube_mfu": c_id / denom, "vec_mfu": v_id / denom, "link_util": l_id / denom, "roofline_efficiency": max(h_id, c_id, v_id, l_id) / denom}

    fam_g, bound_g, kind_g = defaultdict(list), defaultdict(list), defaultdict(list)
    for x in analyzed_ops:
        fam_g[x["family"], x["recipe"]].append(x)
        bound_g[x["dominant_bound"]].append(x)
        kind_g[x["operator_kind"]].append(x)

    occurrence_graph = occurrence_work_graph(graph_report, analyzed_ops)
    attribution = attribute_program(
        graph_report, analyzed_ops, occurrence_graph=occurrence_graph)
    occurrence_roofline = occurrence_critical_path(
        graph_report, analyzed_ops, graph=occurrence_graph)
    if occurrence_roofline is not None:
        tot_roof_id = max(tot_roof_id,
                          occurrence_roofline["critical_path_ideal_ms"])
        serial_roof_id = max(serial_roof_id,
                             occurrence_roofline["critical_path_ideal_ms"])
    nominal = {"cube_mfu": tot_cube_id / safe_graph_ms, "vec_mfu": tot_vec_id / safe_graph_ms,
               "mbu": tot_hbm_id / safe_graph_ms, "link_util": tot_link_id / safe_graph_ms,
               "roofline_efficiency": tot_roof_id / safe_graph_ms,
               "denominator": "graph_span_ms", "semantics": "nominal_peak_relative_work_over_span"}
    graph_accel = graph_report.get("accel_status") if graph_report is not None else None
    return {
        "schema": "mindie-sim.efficiency/1",
        "engine_backend": (graph_report or {}).get("engine_backend"),
        "accel_status": graph_accel,
        "geometry": {"phase": phase, "tokens": tokens, "context": context, "sequences": sequences, "scope": scope},
        "total_flops": tot_flops, "total_vector_ops": tot_vec_ops, "total_hbm_bytes": tot_hbm_b, "total_link_bytes": tot_link_b,
        "total_cube_ideal_ms": tot_cube_id, "total_vec_ideal_ms": tot_vec_id, "total_hbm_ideal_ms": tot_hbm_id, "total_link_ideal_ms": tot_link_id,
        "total_roofline_ideal_ms": tot_roof_id, "operator_serial_roofline_ideal_ms": serial_roof_id, "operator_sum_ms": operator_sum_ms, "graph_duration_ms": graph_duration_ms,
        "network_cube_mfu": nominal["cube_mfu"], "network_vec_mfu": nominal["vec_mfu"], "network_mbu": nominal["mbu"],
        "network_link_util": nominal["link_util"], "network_roofline_efficiency": nominal["roofline_efficiency"],
        "nominal": nominal, "effective_service_utilization": dict(attribution.get("service_utilization_by_role") or {}),
        "critical_path_roofline_ms": (occurrence_roofline["critical_path_ideal_ms"]
                                      if occurrence_roofline else None),
        "critical_path_roofline_target_ms": (occurrence_roofline["critical_path_target_ms"]
                                             if occurrence_roofline else None),
        "critical_path_nodes": (occurrence_roofline["critical_path_nodes"]
                                if occurrence_roofline else ()),
        "critical_path_occurrence_nodes": (occurrence_roofline["occurrence_nodes"]
                                           if occurrence_roofline else ZERO),
        "physical_resource_cut_ms": (max(occurrence_roofline["resource_cut_bounds"].values(),
                                         default=ZERO_FLOAT)
                                     if occurrence_roofline else None),
        "attribution": attribution,
        **({"legacy_service_regime_diagnostics": diagnostics}
           if (diagnostics := legacy_service_regime_diagnostics(parsed)) is not None else {}),
        "operator_serial_roofline_efficiency": serial_roof_id / max(operator_sum_ms, EPSILON), "network_cv_overlap": network_cv_overlap,
        "breakdown_by_family": sorted([{"family": f, "recipe": r, **_agg(items)} for (f, r), items in fam_g.items()], key=lambda r: -r["total_ms"]),
        "breakdown_by_bound": {b: _agg(bound_g.get(b, ())) for b in _BOUNDS}, "breakdown_by_kind": {k: _agg(kind_g.get(k, ())) for k in _KINDS}, "operators": analyzed_ops,
        **({"_occurrence_graph": occurrence_graph} if _retain_graph else {}),
    }


def _evaluate_target_efficiency_legacy(
    case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = DEFAULT_TOKENS, context: int = DEFAULT_CONTEXT, sequences: int = DEFAULT_SEQUENCES, scope: str = "body",
    target_mbu: float = DEFAULT_TARGET_MBU, target_cube_mfu: float = DEFAULT_TARGET_CUBE_MFU, target_vec_mfu: float = DEFAULT_TARGET_VEC_MFU, target_link_util: float | None = None,
    fixed_launch_us: float | None = None, family_targets: Mapping[str, Mapping[str, float]] | None = None,
    mode: str = "clamp_to_target", baseline_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate counterfactual step latency and bottleneck gaps under target pipe efficiencies."""
    if mode not in {"clamp_to_target", "exact_target"}:
        raise ValueError(f"unsupported_target_efficiency_mode:{mode}")
    _validate_geometry(phase, tokens, context, sequences, scope)
    for k, v in (("target_mbu", target_mbu), ("target_cube_mfu", target_cube_mfu), ("target_vec_mfu", target_vec_mfu), ("target_link_util", target_link_util)):
        _validate_efficiency_param(k, v)
    if fixed_launch_us is not None and (isinstance(fixed_launch_us, bool) or not isinstance(fixed_launch_us, (int, float)) or not math.isfinite(fixed_launch_us) or float(fixed_launch_us) < 0.0):
        raise ValueError(f"fixed_launch_us_must_be_nonnegative_finite:{fixed_launch_us}")

    base = (dict(baseline_analysis) if baseline_analysis is not None else
            analyze_network_efficiency(
                case, phase=phase, tokens=tokens, context=context,
                sequences=sequences, scope=scope, _retain_graph=True))
    fam_ovr, evaluated_ops = dict(family_targets or {}), []
    for op in base["operators"]:
        ovr = fam_ovr.get(f"{op['family']}:{op['recipe']}") or fam_ovr.get(op["recipe"]) or fam_ovr.get(op["family"]) or {}
        eta_hbm, eta_cube, eta_vec, eta_link = float(ovr.get("target_mbu", target_mbu)), float(ovr.get("target_cube_mfu", target_cube_mfu)), float(ovr.get("target_vec_mfu", target_vec_mfu)), ovr.get("target_link_util", target_link_util)
        for k, v in (("family.target_mbu", eta_hbm), ("family.target_cube_mfu", eta_cube), ("family.target_vec_mfu", eta_vec), ("family.target_link_util", eta_link)):
            _validate_efficiency_param(k, v)
        t_fixed = float(ovr["fixed_launch_us"]) / 1000.0 if "fixed_launch_us" in ovr else float(fixed_launch_us) / 1000.0 if fixed_launch_us is not None else float(op["fixed_ms"])
        t_min = float(op.get("minimum_ms", 0.0) or 0.0)
        t_link_req = (op["t_link_ideal_ms"] / float(eta_link)) if eta_link is not None else float(op["t_link_ideal_ms"])
        pipe_req = {"hbm": (op["t_hbm_ideal_ms"] / eta_hbm) if op["hbm_bytes"] > 0 else 0.0,
                    "cube": (op["t_cube_ideal_ms"] / eta_cube) if op["flops"] > 0 else 0.0,
                    "vec": (op["t_vec_ideal_ms"] / eta_vec) if op["vector_ops"] > 0 else 0.0,
                    "link": t_link_req if op["link_bytes"] > 0 else 0.0}
        inclusive_pipe_ms = max(pipe_req.values(), default=0.0)
        raw_t = max(t_min, t_fixed + op["roofline_ideal_ms"], inclusive_pipe_ms) if op["roofline_ideal_ms"] > 0 else max(t_min, op["duration_ms"])
        eval_ms = max(t_min, min(op["duration_ms"], raw_t)) if mode == "clamp_to_target" else raw_t
        cur_tot, tgt_tot, safe_eval, safe_svc = op["total_ms"], eval_ms * op["occurrences"], max(eval_ms, 1e-12), max(eval_ms - t_fixed, 1e-12)
        evaluated_ops.append({
            **op, "actual_mbu": op["mbu"], "actual_cube_mfu": op["cube_mfu"], "actual_vec_mfu": op["vec_mfu"],
            "current_duration_ms": op["duration_ms"], "raw_target_duration_ms": raw_t, "target_duration_ms": eval_ms,
            "current_total_ms": cur_tot, "target_total_ms": tgt_tot, "gap_ms": cur_tot - tgt_tot, "target_pipe_requirement_ms": pipe_req,
            "target_mbu": (op["t_hbm_ideal_ms"] / safe_eval) if op["hbm_bytes"] > 0 else None, "service_only_target_mbu": (op["t_hbm_ideal_ms"] / safe_svc) if op["hbm_bytes"] > 0 else None,
            "target_cube_mfu": (op["t_cube_ideal_ms"] / safe_eval) if op["flops"] > 0 else None, "target_vec_mfu": (op["t_vec_ideal_ms"] / safe_eval) if op["vector_ops"] > 0 else None,
        })

    cur_sum, tgt_sum, cur_step = base["operator_sum_ms"], sum(x["target_total_ms"] for x in evaluated_ops), base["graph_duration_ms"]
    pipe_totals = {pipe: sum(x["target_pipe_requirement_ms"][pipe] * x["occurrences"] for x in evaluated_ops) for pipe in ("hbm", "cube", "vec", "link")}
    pipe_bound = max(pipe_totals.values(), default=0.0)
    tgt_step = min(cur_step, pipe_bound) if pipe_bound > 0.0 else cur_step
    saved_ms, safe_tgt = cur_step - tgt_step, max(tgt_step, 1e-12)
    speedup = cur_step / safe_tgt
    return {
        "mode": mode, "targets": {"target_mbu": target_mbu, "target_cube_mfu": target_cube_mfu, "target_vec_mfu": target_vec_mfu, "target_link_util": target_link_util, "fixed_launch_us": fixed_launch_us},
        "current_step_duration_ms": cur_step, "target_step_duration_ms": tgt_step, "target_step_basis": "per_pipe_work_lower_bound_not_rescheduled_graph",
        "target_step_is_lower_bound": True, "target_pipe_totals_ms": pipe_totals,
        "current_operator_sum_ms": cur_sum, "target_operator_sum_ms": tgt_sum,
        "saved_ms": saved_ms, "speedup_ratio": speedup, "step_speedup": speedup, "step_speedup_is_bound": True, "speedup_pct": (speedup - 1.0) * 100.0,
        "latency_reduction_pct": (saved_ms / max(cur_step, 1e-12)) * 100.0, "current_network_mbu": base["network_mbu"], "target_network_mbu": base["total_hbm_ideal_ms"] / safe_tgt,
        "current_network_cube_mfu": base["network_cube_mfu"], "target_network_cube_mfu": base["total_cube_ideal_ms"] / safe_tgt,
        "current_network_vec_mfu": base["network_vec_mfu"], "target_network_vec_mfu": base["total_vec_ideal_ms"] / safe_tgt,
        "current_network_roofline_efficiency": base["network_roofline_efficiency"], "target_network_roofline_efficiency": base["total_roofline_ideal_ms"] / safe_tgt,
        "top_bottlenecks": sorted(evaluated_ops, key=lambda r: -r["gap_ms"])[:15], "operators": evaluated_ops,
        "_occurrence_graph": base.get("_occurrence_graph"),
    }


def _target_occurrence_nodes(graph, operators):
    """Apply target weights to the realized DAG without flattening it."""
    rows, relative = {}, {}
    for row in operators:
        op = row.get("operator") if isinstance(row.get("operator"), Mapping) else row
        op_id = str(op.get("op_id") or "")
        for key in (row.get("physical_id"), row.get("op_id"), op_id):
            if key:
                rows.setdefault(str(key), row)
        if op_id:
            relative.setdefault(normalize_reference(
                op_id, int(row.get("stage", ZERO)), int(row.get("layer", ZERO))), row)
    nodes = []
    for node in graph.nodes:
        node_id = str(node.node_id)
        candidates = (node_id, node_id.split("#", ONE)[ZERO],
                      node_id.split(":", ONE)[ONE] if ":" in node_id else node_id)
        row = next((rows[key] for key in candidates if key in rows), None)
        if row is None:
            match = _OCCURRENCE_IDENTITY.search(node_id)
            if match is not None:
                identity = (node_id.split(":", ONE)[ONE]
                            if ":" in node_id else node_id)
                row = relative.get(normalize_reference(
                    identity, int(match.group(ONE)), int(match.group(ONE + 1))))
        requirements = (row.get("target_pipe_requirement_ms")
                        if row is not None else None)
        if not requirements:
            nodes.append(node)
            continue
        nodes.append(replace(
            node,
            cube_ms=float(requirements["cube"]),
            vector_ms=float(requirements["vec"]),
            hbm_ms=float(requirements["hbm"]),
            link_ms=float(requirements["link"]),
            launch_ms=float(row.get("raw_target_duration_ms")
                            or node.launch_ms),
        ))
    return tuple(nodes)


def evaluate_target_efficiency(
    case: CaseSpec | Mapping[str, Any], *, phase: str = "decode",
    tokens: int = DEFAULT_TOKENS, context: int = DEFAULT_CONTEXT,
    sequences: int = DEFAULT_SEQUENCES,
    scope: str = "body", target_mbu: float = DEFAULT_TARGET_MBU,
    target_cube_mfu: float = DEFAULT_TARGET_CUBE_MFU,
    target_vec_mfu: float = DEFAULT_TARGET_VEC_MFU,
    target_link_util: float | None = None,
    fixed_launch_us: float | None = None,
    family_targets: Mapping[str, Mapping[str, float]] | None = None,
    mode: str = "clamp_to_target",
    baseline_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Target roofline with DAG critical-path and resource conservation."""
    from .unmodeled_gate import assert_report_supported, assert_timing_supported
    assert_timing_supported(case, entry="evaluate_target_efficiency")
    assert_report_supported(baseline_analysis, entry="evaluate_target_efficiency")
    result = _evaluate_target_efficiency_legacy(
        case, phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope, target_mbu=target_mbu,
        target_cube_mfu=target_cube_mfu, target_vec_mfu=target_vec_mfu,
        target_link_util=target_link_util, fixed_launch_us=fixed_launch_us,
        family_targets=family_targets, mode=mode,
        baseline_analysis=baseline_analysis)
    operators = result["operators"]
    occurrence_graph = result.pop("_occurrence_graph", None)
    nodes = tuple(PhysicalWorkNode(
        node_id=str(row.get("physical_id") or row.get("op_id") or index),
        cube_ms=float(row.get("t_cube_ideal_ms") or ZERO_FLOAT),
        vector_ms=float(row.get("t_vec_ideal_ms") or ZERO_FLOAT),
        hbm_ms=float(row.get("t_hbm_ideal_ms") or ZERO_FLOAT),
        link_ms=float(row.get("t_link_ideal_ms") or ZERO_FLOAT),
        launch_ms=float(row.get("fixed_ms") or ZERO_FLOAT),
        occurrences=max(ONE, int(row.get("occurrences", ONE) or ONE)),
    ) for index, row in enumerate(operators))
    ideal = critical_path_roofline(PhysicalWorkGraph(nodes), RooflineTargets())
    if mode == "exact_target" and occurrence_graph is not None:
        # Reuse the realized occurrence DAG: predecessors and physical
        # resource identities must survive the target-weight substitution.
        target_nodes = _target_occurrence_nodes(occurrence_graph, operators)
        roofline = critical_path_roofline(
            PhysicalWorkGraph(target_nodes, occurrence_graph.edges),
            RooflineTargets())
    else:
        target_nodes = tuple(PhysicalWorkNode(
            node_id=node.node_id,
            cube_ms=float(row["target_pipe_requirement_ms"]["cube"]),
            vector_ms=float(row["target_pipe_requirement_ms"]["vec"]),
            hbm_ms=float(row["target_pipe_requirement_ms"]["hbm"]),
            link_ms=float(row["target_pipe_requirement_ms"]["link"]),
            launch_ms=float(row["raw_target_duration_ms"]),
            occurrences=node.occurrences,
        ) for node, row in zip(nodes, operators))
        roofline = critical_path_roofline(
            PhysicalWorkGraph(target_nodes), RooflineTargets())
    cur_step = float(result["current_step_duration_ms"])
    physical_ideal = max(ideal.physical_ideal_ms, ZERO_FLOAT)
    raw_target = max(roofline.target_duration_ms, physical_ideal)
    target_step = (raw_target if mode == "exact_target"
                   else max(physical_ideal, min(cur_step, raw_target)))
    result.update({
        "physical_ideal_ms": physical_ideal,
        "physical_ideal_1x_step_ms": physical_ideal,
        "raw_target_duration_ms": raw_target,
        "target_step_duration_ms": target_step,
        "target_faster_than_prediction": bool(raw_target < cur_step),
        "critical_path_nodes": roofline.critical_path_nodes,
        "resource_cut_bounds": dict(roofline.resource_cut_bounds),
        "target_step_basis": "dag_critical_path_and_resource_conservation",
    })
    safe_target = max(target_step, EPSILON)
    for suffix in ("mbu", "cube_mfu", "vec_mfu", "roofline_efficiency"):
        result["target_network_" + suffix] = (
            result["current_network_" + suffix] * cur_step / safe_target)
    result["saved_ms"] = cur_step - target_step
    result["speedup_ratio"] = cur_step / safe_target
    result["step_speedup"] = result["speedup_ratio"]
    result["speedup_pct"] = (result["speedup_ratio"] - ONE_FLOAT) * FULL_PERCENT
    result["latency_reduction_pct"] = (
        result["saved_ms"] / max(cur_step, EPSILON)) * FULL_PERCENT
    return result


def _operator_fit_signature(op: Mapping[str, Any]) -> str:
    return json.dumps({"op_id": op.get("op_id", ""), "family": op.get("family", ""), "recipe": op.get("recipe", ""), "dtype": op.get("dtype", ""), "local_shape": list(op.get("local_shape") or ()), "flops": round(float(op.get("flops", 0.0)), 6), "vector_ops": round(float(op.get("vector_ops", 0.0)), 6), "hbm_bytes": round(float(op.get("hbm_bytes", 0.0)), 6), "link_bytes": round(float(op.get("link_bytes", 0.0)), 6)}, sort_keys=True, separators=(",", ":"))


def _decompose_operator_pipes(op: Mapping[str, Any]) -> dict[str, Any]:
    """Export per-pipe work/nominal bounds without identifying a bottleneck from total duration."""
    t_base, t_min = float(op["duration_ms"]), float(op.get("minimum_ms", 0.0) or 0.0)
    t_fixed = min(max(float(op["fixed_ms"]), t_min), t_base)
    service_ms = max(0.0, t_base - t_fixed)
    ideals = {"hbm": float(op["t_hbm_ideal_ms"]), "cube": float(op["t_cube_ideal_ms"]),
              "vec": float(op["t_vec_ideal_ms"]), "link": float(op["t_link_ideal_ms"])}
    active = {pipe: value for pipe, value in ideals.items() if value > 0.0}
    nominal_lower = {pipe: (value / service_ms if service_ms > 0.0 else None) for pipe, value in ideals.items()}
    return {
        "fixed_ms": t_fixed, "minimum_ms": t_min, "service_ms": service_ms,
        "work_ideal_ms": ideals, "roofline_ideal_ms": max(active.values(), default=0.0),
        "nominal_work_lower_bound_utilization": nominal_lower,
        "candidate_pipes": sorted(active),
        "bottleneck_pipe": None,
        "bottleneck_identified": False,
        "bottleneck_evidence": None,
        "identification": "total_duration_does_not_identify_pipe_bottleneck",
        "unidentified_pipes": sorted(active),
        "t_hbm_eff_ms": None, "t_cube_eff_ms": None, "t_vec_eff_ms": None, "t_link_eff_ms": None,
        "effective_mbu": None, "effective_cube_mfu": None,
        "effective_vec_mfu": None, "effective_link_util": None,
    }


def fit_case_roofline_regimes(case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1, context: int = 4096, sequences: int = 1, scope: str = "body", observed_durations_ms: Mapping[str, float] | None = None, baseline_analysis: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit multi-pipe roofline parameters (fixed_us, effective_mbu/cube_mfu/vec_mfu) per operator."""
    base = dict(baseline_analysis) if baseline_analysis is not None else analyze_network_efficiency(case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, observed_durations_ms=observed_durations_ms)
    fitted_ops, by_key, by_recipe = [], {}, defaultdict(list)
    for op in base["operators"]:
        dec, sig_key = _decompose_operator_pipes(op), _operator_fit_signature(op)
        by_key[sig_key] = dec
        by_recipe[f"{op['family']}:{op['recipe']}"].append(dec)
        fitted_ops.append({"signature_key": sig_key, "op_id": op["op_id"], "family": op["family"], "recipe": op["recipe"], "operator_kind": op["operator_kind"], "dominant_bound": op["dominant_bound"], "occurrences": op["occurrences"], "duration_ms": op["duration_ms"], "fixed_us": dec["fixed_ms"] * 1000.0, **dec})
    summaries = []
    for rk, items in sorted(by_recipe.items()):
        summaries.append({
            "family": rk.split(":", 1)[0], "recipe": rk.split(":", 1)[1],
            "fixed_us": sum(x["fixed_ms"] for x in items) / len(items) * 1000.0,
            "candidate_pipes": sorted({pipe for x in items for pipe in x["candidate_pipes"]}),
            "nominal_work_lower_bound_utilization": {
                pipe: max((x["nominal_work_lower_bound_utilization"].get(pipe) or 0.0) for x in items)
                for pipe in ("hbm", "cube", "vec", "link")
            },
            "bottleneck_pipe": None, "bottleneck_identified": False,
            "identification": "total_duration_does_not_identify_pipe_bottleneck",
            "effective_mbu": None, "effective_cube_mfu": None,
            "effective_vec_mfu": None, "effective_link_util": None,
        })
    return {"schema": "mindie-sim.roofline-fit/1", "geometry": base["geometry"], "fitted_regimes": summaries, "operators": fitted_ops, "by_key": by_key,
            "identification_scope": "total_duration_exports_nominal_work_bounds_but_does_not_identify_pipe_bottleneck"}


def fit_efficiency_targets(case: CaseSpec | Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Combined entry point returning baseline analysis, target evaluation, and fitted regimes."""
    base = analyze_network_efficiency(case, phase=kwargs.get("phase", "decode"), tokens=kwargs.get("tokens", 1), context=kwargs.get("context", 4096), sequences=kwargs.get("sequences", 1), scope=kwargs.get("scope", "body"))
    return {**evaluate_target_efficiency(case, baseline_analysis=base, **kwargs), "baseline_analysis": base, "fitted_roofline_regimes": fit_case_roofline_regimes(case, baseline_analysis=base)["fitted_regimes"]}




def scale_case_hardware(case: CaseSpec | Mapping[str, Any], *, hbm_bw_scale: float = 1.0, cube_flops_scale: float = 1.0, vector_ops_scale: float = 1.0, link_bw_scale: float = 1.0, hbm_capacity_scale: float = 1.0, baseline_analysis: Mapping[str, Any] | None = None) -> CaseSpec:
    """Scale hardware peaks, explicit/derived rates, service modes, links, and capacity."""
    for name, val in zip(_SCALE_KEYS, (hbm_bw_scale, cube_flops_scale, vector_ops_scale, link_bw_scale, hbm_capacity_scale)):
        _check_pos_float(f"hardware_scale_{name}", val)
    parsed = parse_case(case)
    if (hbm_bw_scale, cube_flops_scale, vector_ops_scale, link_bw_scale, hbm_capacity_scale) == (1.0, 1.0, 1.0, 1.0, 1.0):
        return parsed
    kw = dict(hbm_bw_scale=hbm_bw_scale, cube_flops_scale=cube_flops_scale, vector_ops_scale=vector_ops_scale, link_bw_scale=link_bw_scale)
    mode_to_role = {m_name: m_role for die in parsed.machine.dies for m_name, (m_role, _) in die.service_modes}
    machine_dict = deepcopy(parsed.machine.to_dict())
    for die, dev_dict in zip(parsed.machine.dies, machine_dict["devices"]):
        ins_map = dev_dict.get("compute_arch", {}).get("instruction_services", {})
        svc_role = {sn: (ss["scale_with"]["resource"] if isinstance(ss.get("scale_with"), Mapping) else str(ss.get("resource") or ss.get("owner_axis") or sn)) for sn, ss in ins_map.items()}
        dev_dict["memory_bytes"] = max(1, int(round(die.memory_bytes * hbm_capacity_scale)))
        dev_dict["compute_peaks"] = {dt: float(val) * cube_flops_scale for dt, val in dev_dict.get("compute_peaks", {}).items()}
        dev_dict["rates"] = {r: float(val) * _role_scale_factor(svc_role.get(r, r), **kw) for r, val in dev_dict.get("rates", {}).items()}
        for m_name, m_spec in dev_dict.get("service_modes", {}).items():
            if m_spec.get("rate") is not None:
                m_spec["rate"] = float(m_spec["rate"]) * _role_scale_factor(str(m_spec.get("resource") or svc_role.get(m_name) or mode_to_role.get(m_name, m_name)), **kw)
        for svc_name, svc_spec in ins_map.items():
            if not svc_spec.get("scale_with"):
                sf = _role_scale_factor(svc_role[svc_name], **kw)
                if svc_spec.get("clock_hz") is not None:
                    svc_spec["clock_hz"] = float(svc_spec["clock_hz"]) * sf
                elif svc_spec.get("per_owner_rate") is not None:
                    svc_spec["per_owner_rate"] = float(svc_spec["per_owner_rate"]) * sf
                elif svc_spec.get("issue_ns") is not None:
                    svc_spec["issue_ns"] = float(svc_spec["issue_ns"]) / sf
    for lnk in machine_dict.get("links", ()):
        for lk in ("capacity_bytes_per_ms", "switch_capacity_bytes_per_ms", "peak_capacity_bytes_per_ms"):
            if lnk.get(lk) is not None:
                lnk[lk] = float(lnk[lk]) * link_bw_scale
    new_kernels = deepcopy(parsed.runtime.kernels)
    legacy_rules = list(new_kernels.get("service_regimes", ()) or ())
    metadata = dict(parsed.runtime.metadata)
    if legacy_rules:
        metadata["legacy_service_regime_diagnostics"] = {
            "production_prediction": False, "scaled": False,
            "source": "runtime.kernels.service_regimes",
            "rules": deepcopy(legacy_rules),
        }
    return replace(parsed, machine=Hardware.from_mapping(machine_dict),
                   runtime=replace(parsed.runtime, kernels=new_kernels,
                                   metadata=tuple(sorted(metadata.items()))))


def _stage_rank_capacity_facts(memory: Memory) -> list[dict[str, Any]]:
    """Read the T08 per-rank supply from ``rank_capacity_plans``/``to_dict``.

    ``headroom_bytes`` is the AllocationPlan supply after resident weights,
    workspace, temporary representation, reserves and any live selected/shared/
    grant allocations.  It is therefore the same bound the T08 admission ledger
    checks, not a second private allocator.
    """
    case, p = memory.case, memory.case.runtime.parallel
    dies, width = case.machine.dies, max(1, int(p.tp) * int(p.pcp))
    rows: list[dict[str, Any]] = []
    for replica in range(int(p.dp)):
        physical = memory.physical_usage(replica)["hbm_bytes_per_stage_rank"]
        for stage in range(int(p.pp)):
            for rank, plan in enumerate(memory.rank_capacity_plans(
                    replica, stage, include_cache=False)):
                report = plan.to_dict()
                allocations = {str(item["name"]): float(item["bytes"])
                               for item in report.get("allocations", ())}
                index = (replica * int(p.pp) + stage) * width + rank
                in_plan = (allocations.get("selected_plan", 0.0)
                           + allocations.get("shared_pages", 0.0)
                           + allocations.get("grants", 0.0))
                occupied = float(physical[stage][rank]) if rank < len(physical[stage]) else 0.0
                external_held = max(0.0, occupied - in_plan)
                rows.append({
                    "replica": replica, "stage": stage, "rank": rank,
                    "basis": str(report.get("basis", "per_rank_stage_physical_allocations")),
                    "capacity_bytes": float(report.get("capacity_bytes") or 0.0),
                    "fixed_bytes": float(report.get("peak_bytes") or 0.0),
                    "external_held_bytes": round(external_held, 3),
                    "supply_bytes": round(float(report.get("headroom_bytes") or 0.0)
                                          - external_held, 3),
                    "workspace_bytes": allocations.get("activation_workspace", 0.0),
                    "fixed_allocations": allocations,
                    "hbm_bytes_per_die": (float(dies[index].hbm_bytes)
                                          if index < len(dies) else 0.0),
                })
    return rows


def capacity_window(
    case: CaseSpec | Mapping[str, Any], *, phase: str = "decode",
    context: int = DEFAULT_CONTEXT,
    tokens_per_seq: int = ONE, sequences: int = ONE,
    output_budget: int = DEFAULT_OUTPUT_BUDGET,
    emitted_tokens_per_seq: float | None = None, async_scheduling: bool | None = None,
    hbm_capacity_scale: float = ONE_FLOAT, memory: Memory | None = None,
) -> dict[str, Any]:
    """Project one candidate B onto T08's per-rank supply and the real KV windows.

    The returned record binds the stage context/query/emitted facts used by the
    latency, SLA, capacity, counterfactual, advise, guardrail, topology and PD
    consumers.  ``supply_bytes`` is already net of weights, workspace and fixed
    reserve; ``active_kv_bytes`` is the executing window, ``retiring_kv_bytes``
    the delayed-release hold, and ``peak_bytes`` the largest simultaneous
    occupancy.  ``active_headroom_bytes`` is retained only as an explicitly
    labelled active-window value, never as the complete-service headroom.
    """
    if phase not in {"decode", "prefill"}:
        raise ValueError(f"capacity_window_phase_invalid:{phase}")
    if isinstance(context, bool) or not isinstance(context, int) or context < 0:
        raise ValueError(f"capacity_window_context_invalid:{context}")
    if isinstance(tokens_per_seq, bool) or not isinstance(tokens_per_seq, int) or tokens_per_seq < 1:
        raise ValueError(f"capacity_window_tokens_per_seq_invalid:{tokens_per_seq}")
    if isinstance(sequences, bool) or not isinstance(sequences, int) or sequences < 1:
        raise ValueError(f"capacity_window_sequences_invalid:{sequences}")
    if isinstance(output_budget, bool) or not isinstance(output_budget, int) or output_budget < 0:
        raise ValueError(f"capacity_window_output_budget_invalid:{output_budget}")
    base_case = (replace(case, runtime=replace(case.runtime, async_scheduling=bool(async_scheduling)))
                 if async_scheduling is not None else case)
    if not isinstance(base_case, CaseSpec):
        base_case = parse_case(base_case)
    scaled = (scale_case_hardware(base_case, hbm_capacity_scale=hbm_capacity_scale)
              if hbm_capacity_scale != 1.0 else base_case)
    q_tokens, b = max(1, int(tokens_per_seq)), int(sequences)
    workspace_tokens = max(1, b * q_tokens)
    ledger = memory if memory is not None else Memory(scaled, workspace_tokens=workspace_tokens)
    if memory is not None and int(memory.case.runtime.parallel.pp) != int(scaled.runtime.parallel.pp):
        raise ValueError("capacity_window_ledger_stage_mismatch")
    prefill = phase == "prefill"
    output_eff = 0 if prefill else max(0, int(output_budget))
    if prefill:
        emit_tokens = float(q_tokens)
        delayed_steps = 0
        lookahead_tokens = 0
        steps_per_wave = 1
        retiring_waves = 0
        staggered_delayed_b = 0
        wave_delayed_b = 0
        active_tokens = context + q_tokens
        retiring_tokens = active_tokens
        wave_tokens = active_tokens
    else:
        emit_tokens = (float(q_tokens) if emitted_tokens_per_seq is None
                       else max(1e-6, float(emitted_tokens_per_seq)))
        delayed_steps = max(0, int(ledger.async_delayed_release_steps))
        lookahead_tokens = max(0, int(ledger.async_in_flight_lookahead_steps)) * q_tokens
        steps_per_wave = max(1, math.ceil(max(1, output_eff) / emit_tokens))
        retiring_waves = (max(1, math.ceil(delayed_steps / steps_per_wave))
                          if delayed_steps > 0 else 0)
        active_tokens = max(1, context + max(q_tokens, output_eff) + lookahead_tokens)
        retiring_tokens = max(1, context + max(q_tokens, output_eff))
        wave_tokens = max(1, context + min(
            max(q_tokens, output_eff), max(1, delayed_steps) * q_tokens) + lookahead_tokens)
        staggered_delayed_b = 0
        wave_delayed_b = 0
        if b >= 2 and delayed_steps > 0 and output_eff > 0:
            staggered_delayed_b = max(1, math.ceil(b * delayed_steps / steps_per_wave))
            wave_delayed_b = retiring_waves * b
    shared_prefix = getattr(
        scaled.runtime, "prefix_cache_mode", "private") == "shared"
    prefix_tokens = (
        int(scaled.workload.requests[FIRST].prefix_tokens)
        if shared_prefix and scaled.workload.requests else ZERO)
    if shared_prefix:
        from .capacity_features import batch_cache_bytes

        def batch_stage_bytes(sequences: int, tokens: int) -> list[float]:
            if sequences <= ZERO:
                return [ZERO_FLOAT] * int(scaled.runtime.parallel.pp)
            if phase == "decode":
                # The KV horizon grows with emitted tokens, but the verified
                # query width stays q_tokens; passing the horizon as q_len
                # would charge one recurrent checkpoint per future token.
                seq_q = max(ONE, int(q_tokens))
                seq_ctx = max(ZERO, int(tokens) - seq_q)
            else:
                seq_q = max(ONE, int(tokens) - context)
                seq_ctx = context
            return [float(value) for value in batch_cache_bytes(
                scaled, sequences=sequences, context=seq_ctx,
                tokens_per_seq=seq_q,
                prefix_tokens=min(prefix_tokens, seq_ctx),
                phase=phase, padded=True).total_bytes]

        kv_active = batch_stage_bytes(b, active_tokens)
        kv_retiring = (batch_stage_bytes(staggered_delayed_b, retiring_tokens)
                       if delayed_steps > ZERO else [ZERO_FLOAT] * len(kv_active))
        kv_wave = (batch_stage_bytes(b, wave_tokens)
                   if delayed_steps > ZERO else list(kv_active))
        kv_wave_retiring = (
            batch_stage_bytes(wave_delayed_b, retiring_tokens)
            if wave_delayed_b > ZERO else [ZERO_FLOAT] * len(kv_active))
    else:
        def ledger_stage_bytes(tokens: int) -> list[float]:
            if phase != "decode" or q_tokens == ledger.query_tokens:
                return [float(value) for value in ledger.cache_bytes(
                    tokens, padded=True)]
            return [float(value) for value in ledger.cache_bytes(
                tokens, padded=True, query_tokens=q_tokens)]

        kv_active = ledger_stage_bytes(active_tokens)
        kv_retiring = (ledger_stage_bytes(retiring_tokens)
                       if delayed_steps > ZERO else [ZERO_FLOAT] * len(kv_active))
        kv_wave = (ledger_stage_bytes(wave_tokens)
                   if delayed_steps > ZERO else list(kv_active))
        kv_wave_retiring = kv_retiring
    rows = _stage_rank_capacity_facts(ledger)
    for row in rows:
        stage = int(row["stage"])
        if shared_prefix:
            active = kv_active[stage]
            retiring = kv_retiring[stage]
            batch_wave = kv_wave[stage]
            wave_retiring = kv_wave_retiring[stage]
        else:
            active = b * kv_active[stage]
            retiring = staggered_delayed_b * kv_retiring[stage]
            batch_wave = b * kv_wave[stage]
            wave_retiring = wave_delayed_b * kv_retiring[stage]
        peak = max(active + retiring, batch_wave + wave_retiring)
        used = float(row["fixed_bytes"]) + float(row["external_held_bytes"]) + peak
        row.update({
            "active_kv_bytes": round(active, 3),
            "retiring_kv_bytes": round(retiring, 3),
            "batch_wave_kv_bytes": round(batch_wave, 3),
            "wave_retiring_kv_bytes": round(wave_retiring, 3),
            "peak_kv_bytes": round(peak, 3),
            "headroom_bytes": round(float(row["supply_bytes"]) - peak, 3),
            "active_headroom_bytes": round(float(row["supply_bytes"]) - active, 3),
            "used_bytes": round(used, 3),
            "hbm_utilization_ratio": round(max(0.0, min(1.0, (
                used) / max(float(row["hbm_bytes_per_die"]), 1e-12))), 9),
        })
    stages: list[dict[str, Any]] = []
    for stage_index in range(int(scaled.runtime.parallel.pp)):
        stage_rows = [row for row in rows if int(row["stage"]) == stage_index]
        if not stage_rows:
            continue
        active = max(float(row["active_kv_bytes"]) for row in stage_rows)
        retiring = max(float(row["retiring_kv_bytes"]) for row in stage_rows)
        batch_wave = max(float(row["batch_wave_kv_bytes"]) for row in stage_rows)
        wave_retiring = max(float(row["wave_retiring_kv_bytes"]) for row in stage_rows)
        peak = max(float(row["peak_kv_bytes"]) for row in stage_rows)
        supply = min(float(row["supply_bytes"]) for row in stage_rows)
        headroom = min(float(row["headroom_bytes"]) for row in stage_rows)
        stages.append({
            "stage": stage_index,
            "supply_bytes": round(supply, 3),
            "capacity_bytes": round(max(float(row["capacity_bytes"]) for row in stage_rows), 3),
            "fixed_bytes": round(max(float(row["fixed_bytes"]) for row in stage_rows), 3),
            "external_held_bytes": round(max(float(row["external_held_bytes"]) for row in stage_rows), 3),
            "workspace_bytes": round(max(float(row["workspace_bytes"]) for row in stage_rows), 3),
            "active_kv_bytes": round(active, 3),
            "retiring_kv_bytes": round(retiring, 3),
            "batch_wave_kv_bytes": round(batch_wave, 3),
            "wave_retiring_kv_bytes": round(wave_retiring, 3),
            "peak_bytes": round(peak, 3),
            "used_bytes": round(max(float(row["used_bytes"]) for row in stage_rows), 3),
            "headroom_bytes": round(headroom, 3),
            "active_headroom_bytes": round(min(float(row["active_headroom_bytes"]) for row in stage_rows), 3),
            "hbm_bytes_per_die": round(min(float(row["hbm_bytes_per_die"]) for row in stage_rows), 3),
            "hbm_utilization_ratio": round(max(float(row["hbm_utilization_ratio"]) for row in stage_rows), 9),
        })
    fits = bool(rows) and all(float(row["peak_kv_bytes"]) <=
                              float(row["supply_bytes"]) + max(1e-9, abs(float(row["supply_bytes"])) * 1e-12)
                              for row in rows)
    return {
        "schema": "mindie-sim.capacity-window/1",
        "phase": phase, "context_tokens": int(context),
        "query_tokens_per_sequence": q_tokens, "emitted_tokens_per_sequence": emit_tokens,
        "output_budget": int(output_budget), "sequences": b,
        "async_scheduling": bool(scaled.runtime.async_scheduling),
        "hbm_capacity_scale": float(hbm_capacity_scale),
        "window": {
            "active_tokens_per_sequence": int(active_tokens),
            "retiring_tokens_per_sequence": int(retiring_tokens),
            "wave_tokens_per_sequence": int(wave_tokens),
            "lookahead_tokens": int(lookahead_tokens),
            "delayed_release_steps": int(delayed_steps),
            "steps_per_wave": int(steps_per_wave),
            "retiring_waves": int(retiring_waves),
            "staggered_delayed_sequences": int(staggered_delayed_b),
            "wave_delayed_sequences": int(wave_delayed_b),
        },
        "supply_bytes": round(min((float(row["supply_bytes"]) for row in rows), default=0.0), 3),
        "fixed_bytes": round(max((float(row["fixed_bytes"]) for row in rows), default=0.0), 3),
        "external_held_bytes": round(max((float(row["external_held_bytes"]) for row in rows), default=0.0), 3),
        "workspace_bytes": round(max((float(row["workspace_bytes"]) for row in rows), default=0.0), 3),
        "weight_bytes_per_die": round(max((
            float(row["fixed_allocations"].get("weights", 0.0)) for row in rows), default=0.0), 3),
        "active_kv_bytes": round(max((float(row["active_kv_bytes"]) for row in rows), default=0.0), 3),
        "retiring_kv_bytes": round(max((float(row["retiring_kv_bytes"]) for row in rows), default=0.0), 3),
        "batch_wave_kv_bytes": round(max((float(row["batch_wave_kv_bytes"]) for row in rows), default=0.0), 3),
        "wave_retiring_kv_bytes": round(max((float(row["wave_retiring_kv_bytes"]) for row in rows), default=0.0), 3),
        "peak_bytes": round(max((float(row["peak_kv_bytes"]) for row in rows), default=0.0), 3),
        "headroom_bytes": round(min((float(row["headroom_bytes"]) for row in rows), default=0.0), 3),
        "active_headroom_bytes": round(min((float(row["active_headroom_bytes"]) for row in rows), default=0.0), 3),
        "stages": stages, "stage_rank": rows, "fits": fits,
    }


def _vram_max_sequences_per_dp(
    case: CaseSpec, *, context: int, tokens_per_seq: int, output_budget: int,
    hbm_capacity_scale: float = 1.0, emitted_tokens_per_seq: float | None = None,
    async_scheduling: bool | None = None, phase: str = "decode",
    memory: Memory | None = None,
) -> int:
    """Solve max concurrent sequences per DP replica with the shared capacity window."""
    base_case = replace(case, runtime=replace(case.runtime, async_scheduling=bool(async_scheduling))) if async_scheduling is not None else case
    scaled = scale_case_hardware(base_case, hbm_capacity_scale=hbm_capacity_scale) if hbm_capacity_scale != 1.0 else base_case
    q_tokens = max(1, int(tokens_per_seq))
    rt_cap = min(int(scaled.runtime.max_num_seqs), max(1, int(scaled.runtime.max_num_batched_tokens) // q_tokens))

    def fits(b: int) -> bool:
        return capacity_window(
            scaled, phase=phase, context=context, tokens_per_seq=q_tokens,
            sequences=b, output_budget=output_budget,
            emitted_tokens_per_seq=emitted_tokens_per_seq,
            async_scheduling=None, memory=memory,
        )["fits"]

    if not fits(1):
        return 0
    lo, hi, best = 1, max(1, rt_cap), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        best, lo, hi = (mid, mid + 1, hi) if fits(mid) else (best, lo, mid - 1)
    return best


def _resolve_speculative_tokens(case: CaseSpec, *, output_budget: int, tokens_per_seq: int | None, emitted_tokens_per_seq: float | None) -> tuple[int, float]:
    """Resolve query tokens per sequence and committed tokens per step aligned with DraftPolicy."""
    meta, req, policy_proposal, policy_mean = dict(case.runtime.metadata), RequestSpec(request_id="sla", prompt_tokens=1, output_tokens=max(1, output_budget), arrival_ms=0.0), 0, 1.0
    try:
        dp_pol = DraftPolicy.for_request(case, req)
        if dp_pol.proposal > 0:
            policy_proposal, policy_mean = dp_pol.proposal, dp_pol.mean
    except ValueError:
        pass
    if policy_proposal == 0:
        raw_spec = meta.get("speculative")
        if raw_spec:
            spec = SpeculativeSpec.from_mapping(json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec)
            policy_proposal, policy_mean = spec.draft_len, spec.expected_committed_tokens()
        elif isinstance(meta.get("speculation"), Mapping):
            s_map, rates = meta["speculation"], list(meta["speculation"].get("synthetic_acceptance_rates") or ())
            policy_proposal = int(s_map.get("num_speculative_tokens", len(rates)) or meta.get("draft_proposal_tokens", 0) or 0)
            policy_mean = (1.0 + sum(float(x) for x in rates[:policy_proposal])) if (rates and policy_proposal > 0) else 1.0
        elif int(meta.get("draft_proposal_tokens", 0) or 0) > 0:
            policy_proposal = int(meta["draft_proposal_tokens"])
            policy_mean = float(dict(case.runtime.mtp_acceptance_by_output_tokens).get(max(1, output_budget), case.runtime.mtp_acceptance_length))
    resolved_q = int(tokens_per_seq) if tokens_per_seq is not None else (1 + policy_proposal if policy_proposal > 0 else 1)
    if isinstance(tokens_per_seq, bool) or resolved_q < 1:
        raise ValueError(f"tokens_per_seq_must_be_positive_int:{tokens_per_seq}")
    if emitted_tokens_per_seq is not None:
        return resolved_q, _check_pos_float("emitted_tokens_per_seq", emitted_tokens_per_seq)
    if resolved_q == 1 and policy_proposal > 0 and tokens_per_seq == 1:
        return resolved_q, 1.0
    return resolved_q, (min(float(resolved_q), float(policy_mean)) if (policy_proposal > 0 and resolved_q > 1) else float(resolved_q))


def select_jointly_feasible_candidate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the highest-B candidate that passed both SLA and capacity in its own evaluation."""
    feasible = [record for record in records if record and bool(record.get("feasible"))]
    return max(feasible, key=lambda record: int(record["b"])) if feasible else None


def apply_sla_baseline_gains(results: Sequence[dict[str, Any]],
                            base_r: Mapping[str, Any]) -> None:
    """Populate baseline-relative throughput gains in place.

    Extracted so a caller that evaluates each counterfactual in its own worker
    (one flat pool over ``(sku, scenario)``) reproduces exactly the fields a
    single-call solve publishes.
    """
    base_tps = (base_r["tps_at_sla"] if base_r["tps_at_sla"] > 0
                else base_r["tps_at_b1"])
    base_feas_tps = (base_r["tps_feasible"] if base_r["tps_feasible"] > 0
                     else base_r["tps_at_b1"])
    base_b = max(1, base_r["b_sla"])
    for row in results:
        eff_tps = (row["tps_at_sla"] if base_r["tps_at_sla"] > 0
                   else row["tps_at_b1"])
        eff_feas = (row["tps_feasible"] if base_r["tps_feasible"] > 0
                    else row["tps_at_b1"])
        row["tps_gain_vs_baseline"] = ((eff_tps / base_tps)
                                       if base_tps > 0 else 1.0)
        row["tps_feasible_gain_vs_baseline"] = (
            (eff_feas / base_feas_tps) if base_feas_tps > 0 else 1.0)
        row["concurrency_gain_vs_baseline"] = (
            (row["b_sla"] / base_b) if base_r["b_sla"] > 0
            else ((base_r["step_ms_at_b1"] / row["step_ms_at_b1"])
                  if row["step_ms_at_b1"] > 0.0 else 0.0))


def solve_sla_sizing(
    case: CaseSpec | Mapping[str, Any], *, sla_ms: float = DEFAULT_SLA_MS,
    context: int = DEFAULT_SLA_CONTEXT, tokens_per_seq: int | None = None,
    emitted_tokens_per_seq: float | None = None,
    max_sequences_per_dp: int = DEFAULT_MAX_SEQUENCES, phase: str = "decode",
    scope: str = "body", output_budget: int = DEFAULT_OUTPUT_BUDGET,
    target_mbu: float = DEFAULT_TARGET_MBU,
    target_cube_mfu: float = DEFAULT_TARGET_CUBE_MFU,
    target_vec_mfu: float = DEFAULT_TARGET_VEC_MFU,
    scenarios: Sequence[Mapping[str, Any]] | None = None, b1_analysis: Mapping[str, Any] | None = None,
    evaluation_context: Any | None = None,
    deadline_monotonic: float | None = None,
    sensitivity: str = SLA_SENSITIVITY_FULL,
    jobs: int | str = 1,
    fast_span_probes: bool = True,
    base_b_hint: int = 1,
) -> dict[str, Any]:
    """Solve SLA-constrained maximum concurrency B*_SLA and throughput across hardware scenarios."""
    from .evaluation import check_active_deadline
    from .unmodeled_gate import assert_timing_supported
    assert_timing_supported(case, entry="solve_sla_sizing")
    check_active_deadline()
    _check_pos_float("sla_ms", sla_ms)
    if isinstance(max_sequences_per_dp, bool) or not isinstance(max_sequences_per_dp, int) or max_sequences_per_dp < 1:
        raise ValueError(f"max_sequences_per_dp_must_be_positive_int:{max_sequences_per_dp}")
    if isinstance(output_budget, bool) or not isinstance(output_budget, int) or output_budget < 0:
        raise ValueError(f"output_budget_must_be_nonnegative_int:{output_budget}")
    for k, v in (("target_mbu", target_mbu), ("target_cube_mfu", target_cube_mfu), ("target_vec_mfu", target_vec_mfu)):
        _validate_efficiency_param(k, v)

    parsed = parse_case(case)
    q_tokens, emit_tokens = _resolve_speculative_tokens(parsed, output_budget=output_budget, tokens_per_seq=tokens_per_seq, emitted_tokens_per_seq=emitted_tokens_per_seq)
    # A prefill step processes its query; the following decode step commits its
    # own accepted tokens.  Keep both quantities: the lifecycle capacity must
    # use the decode commit, never the prompt-processing volume.
    decode_q_tokens, decode_emit_tokens = _resolve_speculative_tokens(
        parsed, output_budget=output_budget, tokens_per_seq=None,
        emitted_tokens_per_seq=None)
    _validate_geometry(phase, q_tokens, context, 1, scope)
    batch_budget = int(parsed.runtime.max_num_batched_tokens)
    unchunkable = (not parsed.runtime.enable_chunked_prefill
                   and q_tokens > batch_budget)
    rt_max_b = min(int(parsed.runtime.max_num_seqs), batch_budget // q_tokens)
    if parsed.runtime.enable_chunked_prefill:
        # A chunked prompt may exceed the per-step budget and still admit one
        # sequence; only the unchunked declaration preserves the zero.
        rt_max_b = max(1, rt_max_b)
    if phase == "decode" and parsed.runtime.graph_capture_mode == "full" and parsed.runtime.graph_capture_keys and dict(parsed.runtime.metadata).get("captured_execution_mode") == "FULL_DECODE_ONLY":
        rt_max_b = min(rt_max_b,
                       max(1, max(parsed.runtime.graph_capture_keys) // q_tokens))
    b_max, dp = min(int(max_sequences_per_dp), rt_max_b), parsed.runtime.parallel.dp

    t_defaults = {"target_mbu": target_mbu, "target_cube_mfu": target_cube_mfu, "target_vec_mfu": target_vec_mfu}
    # ``sensitivity`` only selects the counterfactual *set*; an explicit
    # ``scenarios`` list stays authoritative.  One scenario per call keeps the
    # caller's flat worker pool the only pool (no nested pools).
    if scenarios is not None:
        scenario_source = list(scenarios)
    elif str(sensitivity).strip().lower() == SLA_SENSITIVITY_BASELINE:
        scenario_source = [DEFAULT_SLA_SCENARIOS[0]]
    else:
        scenario_source = list(DEFAULT_SLA_SCENARIOS)
    scenario_list = [({**s, **t_defaults} if (s.get("id") == "target_0.8x_mbu" and scenarios is None) else ({**t_defaults, **s} if s.get("id") == "target_0.8x_mbu" else dict(s))) for s in scenario_source]
    for s in scenario_list:
        for k in _SCALE_KEYS:
            _check_pos_float(f"scenario_scale_{k}", s.get(k, 1.0))
        for k in ("target_mbu", "target_cube_mfu", "target_vec_mfu", "target_link_util"):
            _validate_efficiency_param(k, s.get(k))

    scaled_cases: dict[tuple[float, ...], CaseSpec] = {}
    graphs_by_b: dict[int, Any] = {}
    vram_by_cap: dict[float, tuple[int, int, int]] = {}
    graph_cache: dict[tuple[tuple[float, ...], int, bool], tuple[Any, CaseSpec, dict[str, Any]]] = {}
    fast_spans = bool(fast_span_probes)
    analysis_cache: dict[tuple[tuple[float, ...], int], dict[str, Any]] = {}

    def get_scaled(s_key: tuple[float, ...]) -> CaseSpec:
        if s_key not in scaled_cases:
            scaled_cases[s_key] = scale_case_hardware(parsed, **dict(zip(_SCALE_KEYS, s_key)))
        return scaled_cases[s_key]

    baseline_scales = tuple(1.0 for _ in _SCALE_KEYS)

    def get_bundle(s_key: tuple[float, ...], b: int, *,
                   fast: bool = False) -> tuple[Any, CaseSpec, dict[str, Any]]:
        check_active_deadline()
        if (s_key, b, fast) not in graph_cache:
            if b not in graphs_by_b:
                graphs_by_b[b] = target_graphs(parsed, phase=phase, tokens=q_tokens, context=context, sequences=b, scope=scope)
            context_reuse = evaluation_context if s_key == baseline_scales else None
            g, ec, gr = _lower_and_run_graph(
                get_scaled(s_key), phase=phase, tokens=q_tokens, context=context,
                sequences=b, scope=scope, run_graph=True,
                lowered_graphs=graphs_by_b[b], evaluation_context=context_reuse,
                fast_span_only=bool(fast))
            assert gr is not None
            graph_cache[s_key, b, fast] = (g, ec, gr)
        return graph_cache[s_key, b, fast]

    def get_analysis(s_key: tuple[float, ...], b: int) -> dict[str, Any]:
        check_active_deadline()
        if (s_key, b) not in analysis_cache:
            # ``b1_analysis`` is retained only for API compatibility.  It does not
            # carry the case plan/boundary identity needed to prove the same input,
            # so it may never replace an execution evaluated here.  The graph cache
            # above is the only reuse boundary.
            analysis_cache[s_key, b] = analyze_network_efficiency(
                get_scaled(s_key), phase=phase, tokens=q_tokens, context=context,
                sequences=b, scope=scope, run_graph=True,
                _precomputed_graph=get_bundle(s_key, b, fast=False),
                evaluation_context=(evaluation_context
                                    if s_key == baseline_scales else None))
        return analysis_cache[s_key, b]

    def eval_at(scen: Mapping[str, Any], s_key: tuple[float, ...], b: int, *, full: bool = False) -> dict[str, Any]:
        if unchunkable:
            return {"step_ms": 0.0, "step_basis": "geometry_rejected",
                    "step_is_lower_bound": False, "operator_sum_ms": 0.0,
                    "target_pipe_totals_ms": {}, "network_mbu": None,
                    "network_cube_mfu": None, "network_vec_mfu": None,
                    "network_link_util": None,
                    "network_roofline_efficiency": None,
                    "bottleneck_shares": {}}
        if not full and scen.get("target_mbu") is None:
            return {"step_ms": float(
                get_bundle(s_key, b, fast=fast_spans)[2]["duration_ms"])}
        ans = get_analysis(s_key, b)
        shares = {k: ans["breakdown_by_bound"][k]["share_of_operator_sum"] for k in _BOUNDS}
        if scen.get("target_mbu") is not None:
            ev = evaluate_target_efficiency(
                get_scaled(s_key), phase=phase, tokens=q_tokens, context=context, sequences=b, scope=scope,
                target_mbu=float(scen["target_mbu"]), target_cube_mfu=float(scen.get("target_cube_mfu", target_cube_mfu)),
                target_vec_mfu=float(scen.get("target_vec_mfu", target_vec_mfu)), target_link_util=scen.get("target_link_util"), baseline_analysis=ans,
            )
            step = float(ev["target_step_duration_ms"])
            return {"step_ms": step, "step_basis": "target_pipe_work_lower_bound", "step_is_lower_bound": True, "operator_sum_ms": float(ev["target_operator_sum_ms"]), "target_pipe_totals_ms": dict(ev.get("target_pipe_totals_ms") or {}), "network_mbu": float(ev["target_network_mbu"]), "network_cube_mfu": float(ev["target_network_cube_mfu"]), "network_vec_mfu": float(ev["target_network_vec_mfu"]), "network_link_util": ans["total_link_ideal_ms"] / max(step, 1e-12), "network_roofline_efficiency": float(ev["target_network_roofline_efficiency"]), "bottleneck_shares": shares}
        return {"step_ms": float(ans["graph_duration_ms"]), "step_basis": "graph_evaluation", "step_is_lower_bound": False, "operator_sum_ms": float(ans["operator_sum_ms"]), "network_mbu": float(ans["network_mbu"]), "network_cube_mfu": float(ans["network_cube_mfu"]), "network_vec_mfu": float(ans["network_vec_mfu"]), "network_link_util": float(ans["network_link_util"]), "network_roofline_efficiency": float(ans["network_roofline_efficiency"]), "bottleneck_shares": shares}

    base_b_hint = max(1, int(base_b_hint))
    domain_cap = b_max
    domain_reason = ("prompt_exceeds_batch_budget_without_chunking"
                     if unchunkable else None)

    window_provider: dict[str, Any] = {"fn": None}

    def try_step_ok(scen: Mapping[str, Any], s_key: tuple[float, ...], b: int, *, seen: set[int] | None = None,
                    checks: dict[int, dict[str, Any]] | None = None, capacity_max: int | None = None) -> bool:
        nonlocal domain_cap, domain_reason
        if seen is not None:
            seen.add(int(b))
        if b > domain_cap:
            return False
        try:
            step = float(eval_at(scen, s_key, b, full=False)["step_ms"])
            capacity_pass = True
            if checks is not None:
                window_for = window_provider["fn"]
                if window_for is None:
                    raise ModelingError("sla_capacity_window_provider_not_bound")
                current_window, lifecycle_window = window_for(int(b))
                capacity_pass = bool(
                    (capacity_max is None or b <= capacity_max)
                    and current_window["fits"] and lifecycle_window["fits"])
                checks[int(b)] = {
                    "b": int(b), "step_ms": step,
                    "sla": {"limit_ms": float(sla_ms), "pass": step <= sla_ms},
                    "capacity": {"b_vram_max": int(capacity_max or 0),
                                 "current_fits": bool(current_window["fits"]),
                                 "lifecycle_fits": bool(lifecycle_window["fits"]),
                                 "pass": capacity_pass},
                    "feasible": bool(step <= sla_ms and capacity_pass)}
            return bool(step <= sla_ms)
        except UnmodeledCapabilityError:
            # An unmodeled core is not an infeasible candidate.
            raise
        except ModelingError as exc:
            if b <= 1:
                raise
            domain_cap, domain_reason = min(domain_cap, b - 1), str(exc)
            return False

    memory_cache: dict[tuple[int, bool | None, float], Memory] = {}

    def memory_for(tokens: int, async_scheduling: bool | None, scale: float) -> Memory:
        key = (int(tokens), async_scheduling, float(scale))
        if key not in memory_cache:
            base_case = (scale_case_hardware(parsed, hbm_capacity_scale=scale)
                         if scale != 1.0 else parsed)
            if async_scheduling is not None:
                base_case = replace(base_case, runtime=replace(
                    base_case.runtime, async_scheduling=bool(async_scheduling)))
            memory_cache[key] = Memory(base_case, workspace_tokens=int(tokens))
        return memory_cache[key]

    def solve_one(scen: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal base_b_hint
        check_active_deadline()
        s_key = tuple(float(scen.get(k, 1.0)) for k in _SCALE_KEYS)
        if s_key[4] not in vram_by_cap and unchunkable:
            vram_by_cap[s_key[4]] = (0, 0, 0)
        if s_key[4] not in vram_by_cap:
            b_sync = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=s_key[4], emitted_tokens_per_seq=emit_tokens, async_scheduling=False, phase=phase, memory=memory_for(q_tokens, False, s_key[4]))
            b_async = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=s_key[4], emitted_tokens_per_seq=emit_tokens, async_scheduling=True, phase=phase, memory=memory_for(q_tokens, True, s_key[4]))
            b_cfg = b_async if parsed.runtime.async_scheduling else b_sync
            if parsed.runtime.memory:
                b_cfg = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=s_key[4], emitted_tokens_per_seq=emit_tokens, phase=phase, memory=memory_for(q_tokens, None, s_key[4]))
            vram_by_cap[s_key[4]] = (b_cfg, b_sync, b_async)
        (b_vram, b_vram_sync, b_vram_async) = vram_by_cap[s_key[4]]
        if phase == "prefill":
            life_ctx = context + q_tokens
            verify_tokens = decode_q_tokens
            b_life_sync = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=verify_tokens,
                                                      output_budget=output_budget, hbm_capacity_scale=s_key[4],
                                                      emitted_tokens_per_seq=decode_emit_tokens, async_scheduling=False, phase="decode",
                                                      memory=memory_for(verify_tokens, False, s_key[4]))
            b_life_async = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=verify_tokens,
                                                       output_budget=output_budget, hbm_capacity_scale=s_key[4],
                                                       emitted_tokens_per_seq=decode_emit_tokens, async_scheduling=True, phase="decode",
                                                       memory=memory_for(verify_tokens, True, s_key[4]))
            b_lifecycle = b_life_async if parsed.runtime.async_scheduling else b_life_sync
        else:
            life_ctx = context
            verify_tokens = q_tokens
            b_life_sync, b_life_async, b_lifecycle = b_vram_sync, b_vram_async, b_vram
        b_joint = min(b_vram, b_lifecycle)

        window_cache: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}

        def rejected_window():
            empty = {
                "supply_bytes": 0.0, "active_kv_bytes": 0.0,
                "retiring_kv_bytes": 0.0, "workspace_bytes": 0.0,
                "peak_bytes": 0.0, "headroom_bytes": 0.0, "fits": False,
            }
            return empty

        def candidate_windows(b: int) -> tuple[dict[str, Any], dict[str, Any]]:
            if b in window_cache:
                return window_cache[b]
            if unchunkable:
                window_cache[b] = (rejected_window(), rejected_window())
                return window_cache[b]
            current_window = capacity_window(
                parsed, phase=phase, context=context, tokens_per_seq=q_tokens,
                sequences=b, output_budget=output_budget,
                emitted_tokens_per_seq=emit_tokens, hbm_capacity_scale=s_key[4])
            if phase == "decode":
                lifecycle_window = current_window
            else:
                lifecycle_window = capacity_window(
                    parsed, phase="decode", context=life_ctx, tokens_per_seq=verify_tokens,
                    sequences=b, output_budget=output_budget,
                    emitted_tokens_per_seq=decode_emit_tokens,
                    hbm_capacity_scale=s_key[4])
            window_cache[b] = (current_window, lifecycle_window)
            return current_window, lifecycle_window

        window_provider["fn"] = candidate_windows
        b1_ms = (0.0 if unchunkable else
                 float(get_bundle(s_key, 1, fast=False)[2]["duration_ms"]))
        current_one, lifecycle_one = candidate_windows(1)
        b1_capacity = bool(b_joint >= 1 and current_one["fits"]
                           and lifecycle_one["fits"] and not unchunkable)
        candidate_checks: dict[int, dict[str, Any]] = {
            1: {"b": 1, "step_ms": b1_ms,
                "step_basis": ("geometry_rejected" if unchunkable
                               else "graph_evaluation"),
                "sla": {"limit_ms": float(sla_ms),
                        "pass": bool(b1_ms <= sla_ms and not unchunkable)},
                "capacity": {"b_vram_max": int(b_joint),
                             "current_fits": bool(current_one["fits"]),
                             "lifecycle_fits": bool(lifecycle_one["fits"]),
                             "pass": b1_capacity},
                "feasible": bool(b1_ms <= sla_ms and b1_capacity)}}
        evaluated_b: set[int] = set(candidate_checks)
        eff_max = min(b_max, domain_cap)
        bucket_keys = sorted({int(key) for key in (getattr(parsed.runtime, "graph_capture_keys", ()) or ())}
                             | {int(key) for key in (getattr(parsed.runtime, "piecewise_capture_keys", ()) or ())})

        def candidate_record(b: int, evaluation: Mapping[str, Any]) -> dict[str, Any]:
            sla_pass = float(evaluation["step_ms"]) <= sla_ms
            current_window, lifecycle_window = candidate_windows(int(b))
            capacity_pass = bool(
                b <= b_joint and current_window["fits"]
                and lifecycle_window["fits"] and not unchunkable)
            return {
                "b": int(b), "step_ms": float(evaluation["step_ms"]),
                "step_basis": evaluation.get("step_basis", "graph_evaluation"),
                "input_boundary": {"phase": phase, "tokens": q_tokens, "context": context,
                                   "sequences": int(b), "scope": scope,
                                   "output_budget": output_budget,
                                   "emitted_tokens_per_seq": emit_tokens,
                                   "prefill_tokens_per_seq": q_tokens if phase == "prefill" else None,
                                   "decode_context_tokens": life_ctx,
                                   "decode_query_tokens": verify_tokens,
                                   "decode_emitted_tokens_per_seq": (
                                       decode_emit_tokens if phase == "prefill" else emit_tokens)},
                "hardware": {"scenario_id": scen.get("id", "custom"),
                             "scales": dict(zip(_SCALE_KEYS, s_key))},
                "plan": {"kind": "target_graphs", "selection": "case_lowering"},
                "sla": {"limit_ms": float(sla_ms), "pass": bool(sla_pass)},
                "capacity": {"b_vram_max": int(b_joint), "b_vram_current": int(b_vram),
                             "b_vram_lifecycle": int(b_lifecycle), "pass": bool(capacity_pass),
                             "supply_bytes": current_window["supply_bytes"],
                             "active_kv_bytes": current_window["active_kv_bytes"],
                             "retiring_kv_bytes": current_window["retiring_kv_bytes"],
                             "workspace_bytes": current_window["workspace_bytes"],
                             "peak_bytes": current_window["peak_bytes"],
                             "headroom_bytes": current_window["headroom_bytes"],
                             "lifecycle_headroom_bytes": lifecycle_window["headroom_bytes"]},
                "service_lifecycle": {
                    "phase": "decode", "context_tokens": life_ctx,
                    "query_tokens_per_sequence": verify_tokens,
                    "emitted_tokens_per_sequence": (
                        decode_emit_tokens if phase == "prefill" else emit_tokens),
                    "capacity": {
                        "supply_bytes": lifecycle_window["supply_bytes"],
                        "active_kv_bytes": lifecycle_window["active_kv_bytes"],
                        "retiring_kv_bytes": lifecycle_window["retiring_kv_bytes"],
                        "workspace_bytes": lifecycle_window["workspace_bytes"],
                        "peak_bytes": lifecycle_window["peak_bytes"],
                        "headroom_bytes": lifecycle_window["headroom_bytes"],
                        "fits": lifecycle_window["fits"],
                    },
                },
                "feasible": bool(sla_pass and capacity_pass),
                "source": "full_graph_reevaluation",
                "network_mbu": evaluation.get("network_mbu"), "network_cube_mfu": evaluation.get("network_cube_mfu"),
                "network_vec_mfu": evaluation.get("network_vec_mfu"), "network_link_util": evaluation.get("network_link_util"),
                "network_roofline_efficiency": evaluation.get("network_roofline_efficiency"),
                "bottleneck_shares": dict(evaluation.get("bottleneck_shares") or {}),
            }

        if scen.get("target_mbu") is not None:
            actual_scen = {key: value for key, value in scen.items() if key != "target_mbu"}
            actual = eval_at(actual_scen, s_key, 1, full=True)
            bound = eval_at(scen, s_key, 1, full=True)
            capacity_feasible = b_joint >= 1
            target_current_window, target_lifecycle_window = candidate_windows(1)
            return {
                "scenario_id": scen.get("id", "custom"), "label": scen.get("label", scen.get("id", "custom")), "scales": dict(zip(_SCALE_KEYS, s_key)),
                "b_sla": 0, "b_vram_max": b_vram, "b_vram_max_joint": b_joint,
                "b_vram_max_lifecycle_sync": b_life_sync, "b_vram_max_lifecycle_async": b_life_async,
                "b_vram_max_sync": b_vram_sync, "b_vram_max_async": b_vram_async,
                "prefill_tokens_per_seq": q_tokens if phase == "prefill" else None,
                "decode_emitted_tokens_per_seq": decode_emit_tokens if phase == "prefill" else emit_tokens,
                "emitted_tokens_per_seq": emit_tokens,
                "capacity_window": target_current_window,
                "lifecycle_capacity_window": target_lifecycle_window,
                "delta_b_vram_from_disabling_async": max(0, b_vram_sync - b_vram_async), "hbm_bound": bool(b_joint < 1),
                "b_feasible": 0, "domain_max_b": domain_cap, "domain_limit_reason": domain_reason,
                "sla_feasible": False, "latency_sla_feasible": False, "capacity_feasible": capacity_feasible,
                "jointly_feasible": False, "feasible": False,
                "feasibility_status": "target_bound_only_unverified",
                "b_sla_lower_bound": 0, "b_sla_is_proven_max": False,
                "latency_candidate": None, "feasible_candidate": None,
                "search": {"method": "target_pipe_work_bound_not_a_feasibility_candidate",
                           "monotonicity_assumed": False, "non_monotonic_dispatch_detected": False,
                           "non_monotonic_dispatch_check": "not_applicable_target_bound",
                           "bounded_dispatch_probes": [], "bounded_search_exhaustive": False,
                           "evaluated_candidates": [1], "b_sla_is_proven_max": False,
                           "candidate_step_source": "none", "lower_bound_can_prove_feasibility": False},
                "total_concurrency": 0, "total_feasible_concurrency": 0,
                "step_basis": "graph_evaluation_at_b1_reference", "step_is_lower_bound": False,
                "step_ms_at_b1": actual["step_ms"], "tps_at_b1": (dp * emit_tokens * 1000.0) / actual["step_ms"],
                "step_ms_at_sla": actual["step_ms"], "tps_at_sla": 0.0,
                "target_bound_step_ms_at_b1": bound["step_ms"], "target_bound_is_lower_bound": True,
                "target_bound_pipe_totals_ms": bound.get("target_pipe_totals_ms"),
                "target_bound_operator_sum_ms": bound.get("operator_sum_ms"),
                "target_step_tps_at_sla": 0.0,
                "step_ms_feasible": None, "tps_feasible": 0.0,
                "feasible_network_mbu": None, "feasible_network_cube_mfu": None,
                "feasible_network_vec_mfu": None, "feasible_network_link_util": None,
                "feasible_network_roofline_efficiency": None, "feasible_bottleneck_shares": None,
                "network_source": "latency_reference_not_feasible",
                "network_mbu": actual["network_mbu"], "network_mbu_at_sla": actual["network_mbu"],
                "network_cube_mfu": actual["network_cube_mfu"], "network_cube_mfu_at_sla": actual["network_cube_mfu"],
                "network_vec_mfu": actual["network_vec_mfu"], "network_vec_mfu_at_sla": actual["network_vec_mfu"],
                "network_link_util": actual["network_link_util"], "network_roofline_efficiency": actual["network_roofline_efficiency"],
                "bottleneck_shares": actual["bottleneck_shares"],
            }

        bounded_probes = []
        secant_estimate = None
        # Roofline secant warm-up (Finding 15).  The estimate only *chooses one
        # additional probe* whose result warms the per-call graph cache; it never
        # replaces the existing bucket/non-monotonic, headroom, capacity or
        # lifecycle checks, and ``b_sla_is_proven_max`` stays False.  An
        # infeasible b == 1, ``eff_max < 2`` or a flat/negative initial slope
        # skips it.
        if (0 < base_b_hint <= eff_max and eff_max >= 2 and b1_ms <= sla_ms):
            b2_ms = float(eval_at(scen, s_key, 2, full=False)["step_ms"])
            marginal = b2_ms - b1_ms
            if math.isfinite(marginal) and marginal > SECANT_MIN_SLOPE_MS:
                estimate = 1 + int((sla_ms - b1_ms) / marginal)
                estimate = max(2, min(int(estimate), int(eff_max)))
                if estimate not in evaluated_b:
                    try_step_ok(scen, s_key, estimate, seen=evaluated_b,
                                capacity_max=b_joint)
                    secant_estimate = estimate
        if b1_ms > sla_ms:
            probe_ceiling = min(eff_max, NON_MONOTONIC_PROBE_CAP)
            probe_results = {probe: try_step_ok(scen, s_key, probe, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint) for probe in range(2, probe_ceiling + 1)}
            bounded_probes = list(probe_results)
            feasible_above = [probe for probe, ok in probe_results.items() if ok]
            if feasible_above:
                b_sla = max(feasible_above)
                eval_sla = eval_at(scen, s_key, b_sla, full=True)
            else:
                b_sla, eval_sla = 0, eval_at(scen, s_key, 1, full=True)
        elif eff_max > 1 and (b1_ms * SLA_HEADROOM_MULTIPLIER <= sla_ms or base_b_hint == eff_max) and try_step_ok(scen, s_key, eff_max, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint):
            b_sla, eval_sla = eff_max, eval_at(scen, s_key, eff_max, full=True)
        else:
            eff_max = min(b_max, domain_cap)
            start_b = max(2, min(eff_max, base_b_hint)) if (base_b_hint > 1 and all(x >= 1.0 for x in s_key[:4])) else 2
            low, high, probe = 1, eff_max, max(1, min(eff_max, start_b))
            if probe > 2 and not try_step_ok(scen, s_key, probe, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint):
                high, probe = min(high, domain_cap, probe - 1), 2
            while probe <= high:
                if try_step_ok(scen, s_key, probe, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint):
                    low = probe
                    if probe == high:
                        break
                    probe = min(high, probe * 2)
                else:
                    high = min(high, domain_cap, probe - 1)
                    break
            best_b, lo_i, hi_i = low, low + 1, min(high, domain_cap)
            while lo_i <= hi_i:
                mid = (lo_i + hi_i) // 2
                best_b, lo_i, hi_i = (mid, mid + 1, hi_i) if try_step_ok(scen, s_key, mid, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint) else (best_b, lo_i, min(hi_i, domain_cap, mid - 1))
            while best_b > 1 and not try_step_ok(scen, s_key, best_b, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint):
                best_b -= 1
            b_sla = best_b if try_step_ok(scen, s_key, best_b, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint) else 0
            eval_sla = eval_at(scen, s_key, max(1, b_sla), full=True)

        if b_sla > 0 and b_sla < min(b_max, domain_cap) and bucket_keys:
            ceiling = min(min(b_max, domain_cap), b_sla + SLA_HEADROOM_PROBE_STEPS)
            probe_results = {probe: try_step_ok(scen, s_key, probe, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint) for probe in range(b_sla + 1, ceiling + 1)}
            bounded_probes.extend(probe_results)
            feasible_above = [probe for probe, ok in probe_results.items() if ok]
            if feasible_above:
                b_sla = max(feasible_above)
                eval_sla = eval_at(scen, s_key, b_sla, full=True)
        # A latency-feasible batch still has to pass both capacity windows.
        # Bound the exact window first (memory-only) and price only its
        # capacity-feasible ceiling, so a near-limit answer does not force one
        # graph evaluation per candidate.
        if b_sla > 0 and b_joint > 0:
            ceiling = min(int(b_sla), int(b_joint), int(b_max), int(domain_cap))
            for probe in range(ceiling, 0, -1):
                current_window, lifecycle_window = candidate_windows(probe)
                if current_window["fits"] and lifecycle_window["fits"]:
                    if try_step_ok(scen, s_key, probe, seen=evaluated_b,
                                   checks=candidate_checks,
                                   capacity_max=b_joint):
                        break
        if b_sla > 0:
            evaluated_b.add(b_sla)
        if scen.get("id") == "baseline" and b_sla > 0:
            base_b_hint = b_sla

        # Probe the capacity ceiling and the latency best as actual candidates even
        # when neither was reached by the binary path; all checks share this set.
        for probe in sorted({b_vram, b_sla}):
            if 0 < probe <= min(b_max, domain_cap) and probe not in candidate_checks:
                step = float(eval_at(scen, s_key, probe, full=False)["step_ms"])
                candidate_checks[probe] = {
                    "b": int(probe), "step_ms": step,
                    "sla": {"limit_ms": float(sla_ms), "pass": step <= sla_ms},
                    "capacity": {"b_vram_max": int(b_joint), "b_vram_current": int(b_vram),
                                 "b_vram_lifecycle": int(b_lifecycle), "pass": probe <= b_joint},
                    "feasible": bool(step <= sla_ms and probe <= b_joint)}
                evaluated_b.add(probe)
        latency_candidate = candidate_record(b_sla, eval_sla) if b_sla > 0 else None
        feasible_check = select_jointly_feasible_candidate(list(candidate_checks.values()))
        b_feas = int(feasible_check["b"]) if feasible_check is not None else 0
        feasible_ev = eval_at(scen, s_key, b_feas, full=True) if b_feas > 0 else None
        feasible_candidate = (candidate_record(b_feas, feasible_ev)
                              if feasible_ev is not None else None)
        if feasible_candidate is not None and not feasible_candidate["feasible"]:
            feasible_candidate = None
            b_feas = 0
        capacity_feasible = b_joint >= 1
        latency_sla_feasible = bool(latency_candidate is not None and latency_candidate["sla"]["pass"])
        jointly_feasible = feasible_candidate is not None
        exhaustive = set(range(1, min(b_max, domain_cap) + 1)) <= set(candidate_checks)
        if domain_cap < 1:
            feasibility_status = "domain_infeasible"
        elif not capacity_feasible:
            feasibility_status = "capacity_infeasible"
        elif jointly_feasible:
            feasibility_status = "feasible" if b_sla <= b_feas else "feasible_capacity_limited"
        elif latency_candidate is not None:
            feasibility_status = ("no_jointly_feasible_candidate_in_evaluated_domain" if exhaustive
                                  else "no_feasible_candidate_in_bounded_search")
        else:
            feasibility_status = ("sla_infeasible_within_evaluated_domain" if exhaustive
                                  else "no_feasible_candidate_in_bounded_search")
        tps_sla = (b_sla * dp * emit_tokens * 1000.0) / eval_sla["step_ms"] if b_sla > 0 else 0.0
        tps_feas = (b_feas * dp * emit_tokens * 1000.0) / feasible_ev["step_ms"] if jointly_feasible and feasible_ev is not None else 0.0
        step_basis = eval_sla.get("step_basis", "graph_evaluation") if b_sla > 0 else "graph_evaluation_at_b1_reference"
        output_network = feasible_ev if (jointly_feasible and feasible_ev is not None) else eval_sla
        output_network_source = "feasible_candidate" if jointly_feasible else "latency_candidate_not_feasible"
        selected_b = b_feas if b_feas > 0 else (b_sla if b_sla > 0 else 1)
        selected_current_window, selected_lifecycle_window = candidate_windows(selected_b)
        return {
            "scenario_id": scen.get("id", "custom"), "label": scen.get("label", scen.get("id", "custom")), "scales": dict(zip(_SCALE_KEYS, s_key)),
            "b_sla": b_sla, "b_vram_max": b_vram, "b_vram_max_joint": b_joint,
            "b_vram_max_lifecycle_sync": b_life_sync, "b_vram_max_lifecycle_async": b_life_async,
            "b_vram_max_sync": b_vram_sync, "b_vram_max_async": b_vram_async,
            "prefill_tokens_per_seq": q_tokens if phase == "prefill" else None,
            "decode_emitted_tokens_per_seq": decode_emit_tokens if phase == "prefill" else emit_tokens,
            "emitted_tokens_per_seq": emit_tokens,
            "capacity_window": selected_current_window,
            "lifecycle_capacity_window": selected_lifecycle_window,
            "joint_capacity": {
                "sequences": selected_b,
                "current_supply_bytes": selected_current_window["supply_bytes"],
                "current_active_kv_bytes": selected_current_window["active_kv_bytes"],
                "current_retiring_kv_bytes": selected_current_window["retiring_kv_bytes"],
                "current_workspace_bytes": selected_current_window["workspace_bytes"],
                "current_peak_bytes": selected_current_window["peak_bytes"],
                "current_headroom_bytes": selected_current_window["headroom_bytes"],
                "lifecycle_supply_bytes": selected_lifecycle_window["supply_bytes"],
                "lifecycle_active_kv_bytes": selected_lifecycle_window["active_kv_bytes"],
                "lifecycle_retiring_kv_bytes": selected_lifecycle_window["retiring_kv_bytes"],
                "lifecycle_workspace_bytes": selected_lifecycle_window["workspace_bytes"],
                "lifecycle_peak_bytes": selected_lifecycle_window["peak_bytes"],
                "lifecycle_headroom_bytes": selected_lifecycle_window["headroom_bytes"],
                "fits": bool(selected_current_window["fits"] and selected_lifecycle_window["fits"]),
            },
            "delta_b_vram_from_disabling_async": max(0, b_vram_sync - b_vram_async), "hbm_bound": bool(b_joint < b_sla),
            "b_feasible": b_feas, "domain_max_b": domain_cap, "domain_limit_reason": domain_reason,
            "sla_feasible": jointly_feasible, "latency_sla_feasible": latency_sla_feasible,
            "capacity_feasible": capacity_feasible, "jointly_feasible": jointly_feasible, "feasible": jointly_feasible,
            "feasibility_status": feasibility_status,
            "b_sla_lower_bound": b_sla, "b_sla_is_proven_max": False,
            "latency_candidate": latency_candidate, "feasible_candidate": feasible_candidate,
            "search": {"method": "exponential_probe_plus_binary_under_monotone_assumption",
                       "secant_warmup_probe": secant_estimate,
                       "monotonicity_assumed": True, "non_monotonic_dispatch_detected": bool(
                           latency_candidate is not None and latency_candidate["capacity"]["pass"] is False and b1_ms > sla_ms),
                       "non_monotonic_dispatch_check": "bounded_bucket_probe" if bucket_keys else "not_applicable_no_capture_buckets",
                       "bounded_dispatch_probes": bounded_probes, "bounded_search_exhaustive": bool(exhaustive),
                       "evaluated_candidates": sorted(evaluated_b), "b_sla_is_proven_max": False,
                       "candidate_step_source": "full_graph_reevaluation" if jointly_feasible else "none",
                       "lower_bound_can_prove_feasibility": False},
            "total_concurrency": b_sla * dp, "total_feasible_concurrency": b_feas * dp,
            "step_basis": step_basis, "step_is_lower_bound": False,
            "step_ms_at_b1": b1_ms,
            "tps_at_b1": ((dp * emit_tokens * 1000.0) / b1_ms
                          if b1_ms > 0.0 else 0.0),
            "step_ms_at_sla": eval_sla["step_ms"], "tps_at_sla": tps_sla,
            "target_step_tps_at_sla": (b_sla * dp * q_tokens * 1000.0) / eval_sla["step_ms"] if b_sla > 0 else 0.0,
            "step_ms_feasible": feasible_candidate["step_ms"] if jointly_feasible and feasible_candidate is not None else None,
            "tps_feasible": tps_feas,
            "feasible_network_mbu": feasible_candidate["network_mbu"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_cube_mfu": feasible_candidate["network_cube_mfu"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_vec_mfu": feasible_candidate["network_vec_mfu"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_link_util": feasible_candidate["network_link_util"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_roofline_efficiency": feasible_candidate["network_roofline_efficiency"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_bottleneck_shares": feasible_candidate["bottleneck_shares"] if jointly_feasible and feasible_candidate is not None else None,
            "network_source": output_network_source,
            "network_mbu": output_network["network_mbu"], "network_mbu_at_sla": eval_sla["network_mbu"],
            "network_cube_mfu": output_network["network_cube_mfu"], "network_cube_mfu_at_sla": eval_sla["network_cube_mfu"],
            "network_vec_mfu": output_network["network_vec_mfu"], "network_vec_mfu_at_sla": eval_sla["network_vec_mfu"],
            "network_link_util": output_network["network_link_util"], "network_roofline_efficiency": output_network["network_roofline_efficiency"],
            "bottleneck_shares": output_network["bottleneck_shares"],
        }

    results = [solve_one(scen) for scen in scenario_list]
    base_r = next((r for r in results if r["scenario_id"] == "baseline"), next((r for r in results if tuple(r["scales"].values()) == (1.0, 1.0, 1.0, 1.0, 1.0) and r["scenario_id"] != "target_0.8x_mbu"), None)) or solve_one(DEFAULT_SLA_SCENARIOS[0])
    apply_sla_baseline_gains(results, base_r)
    mem_async_b1 = Memory(replace(parsed, runtime=replace(parsed.runtime, async_scheduling=True)), workspace_tokens=q_tokens)
    prefill = phase == "prefill"
    lifecycle_query = decode_q_tokens if prefill else q_tokens
    lifecycle_emit = decode_emit_tokens if prefill else emit_tokens
    seq_sync_len = max(1, (context + q_tokens) if prefill else (context + max(q_tokens, output_budget)))
    seq_lifecycle_len = max(1, (context + q_tokens + max(lifecycle_query, output_budget)) if prefill else seq_sync_len)
    seq_async_len = max(1, seq_lifecycle_len + mem_async_b1.async_in_flight_lookahead_steps * lifecycle_query)
    if prefill:
        life_ctx = context + q_tokens
        life_sync = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=lifecycle_query,
                                                output_budget=output_budget, emitted_tokens_per_seq=lifecycle_emit,
                                                async_scheduling=False, phase="decode")
        life_async = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=lifecycle_query,
                                                 output_budget=output_budget, emitted_tokens_per_seq=lifecycle_emit,
                                                 async_scheduling=True, phase="decode")
    else:
        life_ctx = context
        life_sync, life_async = base_r["b_vram_max_sync"], base_r["b_vram_max_async"]
    capacity_window_b1 = capacity_window(
        parsed, phase=phase, context=context, tokens_per_seq=q_tokens, sequences=1,
        output_budget=output_budget, emitted_tokens_per_seq=emit_tokens)
    lifecycle_capacity_window_b1 = capacity_window(
        parsed, phase="decode", context=life_ctx, tokens_per_seq=lifecycle_query, sequences=1,
        output_budget=output_budget, emitted_tokens_per_seq=lifecycle_emit)

    async_mem_summary = {
        "configured_async_scheduling": bool(parsed.runtime.async_scheduling),
        "capacity_window": "prefill_prompt_only" if prefill else "decode_step",
        "lifecycle_window": "prefix_query_plus_output_budget" if prefill else "decode_step",
        "prefill_tokens_per_seq": q_tokens if prefill else None,
        "decode_emitted_tokens_per_seq": lifecycle_emit,
        "b_vram_max_sync": base_r["b_vram_max_sync"],
        "b_vram_max_async": base_r["b_vram_max_async"],
        "b_vram_max_lifecycle_sync": int(life_sync),
        "b_vram_max_lifecycle_async": int(life_async),
        "b_vram_max_joint_sync": int(min(base_r["b_vram_max_sync"], life_sync)),
        "b_vram_max_joint_async": int(min(base_r["b_vram_max_async"], life_async)),
        "delta_b_vram_from_disabling_async": base_r["delta_b_vram_from_disabling_async"],
        "delayed_release_steps_async": mem_async_b1.async_delayed_release_steps,
        "in_flight_lookahead_tokens_async": mem_async_b1.async_in_flight_lookahead_steps * lifecycle_query,
        "kv_bytes_per_seq_sync_per_stage": [float(value) for value in mem_async_b1.cache_bytes(seq_sync_len, padded=True)],
        "kv_bytes_per_seq_lifecycle_per_stage": [float(value) for value in mem_async_b1.cache_bytes(seq_lifecycle_len, padded=True)],
        "kv_bytes_per_seq_async_per_stage": [float(value) for value in mem_async_b1.cache_bytes(seq_async_len, padded=True)],
        "async_workspace_overhead_bytes_at_b1_per_stage": [float(value) for value in mem_async_b1.async_workspace],
        "capacity_window_b1": capacity_window_b1,
        "lifecycle_capacity_window_b1": lifecycle_capacity_window_b1,
        "joint_capacity_b1": {
            "supply_bytes": min(capacity_window_b1["supply_bytes"], lifecycle_capacity_window_b1["supply_bytes"]),
            "active_kv_bytes": capacity_window_b1["active_kv_bytes"],
            "retiring_kv_bytes": lifecycle_capacity_window_b1["retiring_kv_bytes"],
            "workspace_bytes": max(capacity_window_b1["workspace_bytes"], lifecycle_capacity_window_b1["workspace_bytes"]),
            "peak_bytes": max(capacity_window_b1["peak_bytes"], lifecycle_capacity_window_b1["peak_bytes"]),
            "headroom_bytes": min(capacity_window_b1["headroom_bytes"], lifecycle_capacity_window_b1["headroom_bytes"]),
            "fits": bool(capacity_window_b1["fits"] and lifecycle_capacity_window_b1["fits"]),
        },
        "hbm_bound_at_baseline": bool(base_r["b_vram_max"] < base_r["b_sla"]),
        "disabling_async_unlocks_concurrency": bool(base_r["b_vram_max_sync"] > base_r["b_vram_max_async"] and base_r["b_vram_max_async"] < base_r["b_sla"]),
    }
    return {"schema": "mindie-sim.hardware-sla/1", "sla_ms": sla_ms, "context": context, "phase": phase, "scope": scope, "timing_boundary": "target_body_only" if scope == "body" else "full_model_step", "tokens_per_seq": q_tokens, "emitted_tokens_per_seq": emit_tokens, "prefill_tokens_per_seq": q_tokens if prefill else None, "decode_emitted_tokens_per_seq": decode_emit_tokens, "dp": dp, "max_sequences_per_dp": b_max,
            "solver": {"monotonicity_assumed": True, "b_sla_is_proven_max": False,
                       "bounded_non_monotonic_dispatch_probe": True,
                       "status_interpretation": "feasible_candidate_found_or_explicit_sla_capacity_domain_infeasibility"},
            "async_scheduling_memory": async_mem_summary, "scenarios": results}


solve_hardware_sla_throughput = solve_sla_sizing
