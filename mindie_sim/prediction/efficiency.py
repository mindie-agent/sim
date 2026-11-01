"""Whole-network MFU/MBU efficiency analysis, target roofline fitting, and SLA scaling solver.

[Role] Compute four-dimensional operator and network efficiency (Cube_MFU,
  Vec_MFU, MBU, Link_Util, CV_Overlap), fit multi-pipe roofline targets, and
  solve SLA-constrained throughput under counterfactual hardware scaling.
[Invariants] Physical FLOPs, vector OPs, HBM bytes, and link bytes are conserved
  from lowered operators and executed resource service. At 1.0x scaling, scaled
  hardware and multi-pipe roofline decomposition reproduce baseline simulator
  latency with 0.00% drift.
[Timeouts & Locks] Pure in-memory arithmetic and bounded DAG evaluation; no I/O,
  subprocesses, sleeps, or device locks.
[Side-Effect Policy] Input cases and operator reports are never mutated; scaling
  and regime fitting return new immutable records.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import json
import math
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Device, Hardware
from mindie_sim.schema.model import RequestSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from .body_defaults import TargetGraphSet, TargetOperators, consume_target_declaration
from .capacity import Memory
from .graph_analysis import graph_costs
from .operator_analysis import operator_costs, target_graphs
from .scheduler import DraftPolicy
from .structure import parse_case

_STRIP_METADATA_KEYS = frozenset({"structure_subgraph_memberships", "structure_subgraph_entries", "structure_subgraphs", "graph_segment", "fusion_group_key", "topk_ids_producer", "payload_producer", "input_conversion_subgraphs", "inverse_index_producer", "topk_weights_producer"})
_HBM_ROLES = frozenset({"hbm", "mte2", "mte2_aic", "mte2_aiv", "mte3", "l2"})
_CUBE_ROLES = frozenset({"cube", "mte1", "fixpipe", "mte1_l0a", "mte1_l0b", "aic"})
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


def _role_scale_factor(role: str, *, hbm_bw_scale: float, cube_flops_scale: float, vector_ops_scale: float, link_bw_scale: float) -> float:
    r = str(role or "").strip()
    if r in _HBM_ROLES or r.startswith(("hbm", "mte2", "mte3", "l2")):
        return hbm_bw_scale
    if r in _CUBE_ROLES or r.startswith(("cube", "mte1", "fixpipe", "aic")):
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


def extract_hardware_peaks(case: CaseSpec | Mapping[str, Any], *, stage: int = 0, replica: int = 0) -> dict[str, Any]:
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
    intra_bw = max((lnk.capacity_bytes_per_ms * 1000.0 for lnk in links if "intra_package" in lnk.channel or len(lnk.members) <= parsed.machine.dies_per_package), default=0.0)
    inter_bw = min((lnk.capacity_bytes_per_ms * 1000.0 for lnk in links), default=0.0)
    link_peak = (intra_bw if max(p.tp, p.ep if p.ep > 1 else 1) <= parsed.machine.dies_per_package and intra_bw > 0 else inter_bw) if links else 0.0
    return {
        "die": dies[0], "dies": dies, "compute_peaks": compute_peaks, "default_dtype": parsed.model.compute_dtype or "bf16",
        "vector_peak_ops_per_s": vec_peak, "hbm_peak_bytes_per_s": hbm_peak, "mte1_peak_bytes_per_s": mte1_peak, "mte2_peak_bytes_per_s": mte2_peak,
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
    reg_fixed_ms = 0.0
    for reg in basis.get("service_regimes", ()):
        queries, resp = reg.get("queries", {}), reg.get("response", {})
        if (queries.get("observable_response") or queries.get("joint_native_service")) and float(resp.get("fixed_us", 0.0) or 0.0) > 0:
            if resp.get("terms") or resp.get("services"):
                reg_fixed_ms = max(reg_fixed_ms, float(resp["fixed_us"]) / 1000.0)
            else:
                minimum_ms = max(minimum_ms, float(resp["fixed_us"]) / 1000.0)
    return min(max(startup_ms, reg_fixed_ms), duration_ms), min(minimum_ms, duration_ms)


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
    comm_wire = (row.get("communication") or {}).get("wire_by_resource") or {}
    _non_lnk = ("memory@", "vector@", "matrix@", "mte", "fixpipe@", "l2@", "ub", "reg_", "sreg_", "vreg_", "scalar_", "vbitsort", "host", "scheduler@", "native_call_service")
    serv_link = (sum(float(v) for v in comm_wire.values()) if comm_wire else sum(float(v) for k, v in serviced.items() if not any(k.startswith(p) for p in _non_lnk))) if not has_jn else 0.0

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
    link_bytes = max(logical_link_bytes, serv_link)
    t_link_ideal_ms = (link_bytes / hw["link_peak_bytes_per_s"]) * 1000.0 if (link_bytes > 0 and hw["link_peak_bytes_per_s"] > 0) else 0.0

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


def _lower_and_run_graph(case: CaseSpec, *, phase: str, tokens: int, context: int, sequences: int, scope: str, run_graph: bool = True, lowered_graphs: Any | None = None) -> tuple[Any, CaseSpec, dict[str, Any] | None]:
    graphs = lowered_graphs if lowered_graphs is not None else target_graphs(case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope)
    eff_case, has_kg = consume_target_declaration(case, graphs), any("device_kernel_group" in dict(op.metadata) for op in graphs[0])
    if not (run_graph or has_kg):
        return graphs, eff_case, None
    ext_in = {"analysis_input": 0} if scope == "body" else None
    sg = (TargetGraphSet({0: graphs[0]}, kernels=graphs.effective_kernels, selection=graphs.default_selection) if isinstance(graphs, TargetGraphSet) else {0: graphs[0]}) if (len(graphs) > 1 and not has_kg and not eff_case.runtime.parallel.ep_spans_dp) else graphs
    sg_map = {r: tuple(replace(op, metadata=tuple((k, v) for k, v in op.metadata if k not in _STRIP_METADATA_KEYS)) for op in ops) for r, ops in sg.items()}
    sg = TargetGraphSet(sg_map, kernels=sg.effective_kernels, selection=sg.default_selection) if isinstance(sg, TargetGraphSet) else sg_map
    return graphs, eff_case, graph_costs(eff_case, sg, external_inputs=ext_in, include_submission=False, isolated=False)


def _fast_operator_and_graph_reports(case: CaseSpec, *, phase: str, tokens: int, context: int, sequences: int, scope: str, run_graph: bool = True, precomputed_graph: tuple[Any, CaseSpec, dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Lower target graphs once and deduplicate layer-lineage metadata for fast pricing."""
    graphs, eff_case, graph_rep = precomputed_graph or _lower_and_run_graph(case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, run_graph=run_graph)
    r0_ops = graphs[0]
    cleaned_r0 = tuple(replace(op, metadata=tuple((k, v) for k, v in op.metadata if k not in _STRIP_METADATA_KEYS)) for op in r0_ops)
    if isinstance(r0_ops, TargetOperators):
        cleaned_r0 = TargetOperators(cleaned_r0, kernels=r0_ops.effective_kernels, selection=r0_ops.default_selection)
    try:
        op_rep = operator_costs(eff_case, cleaned_r0, replica=0)
    except ValueError:
        cleaned_map = {r: tuple(replace(op, metadata=tuple((k, v) for k, v in op.metadata if k not in _STRIP_METADATA_KEYS)) for op in ops) for r, ops in graphs.items()}
        if isinstance(graphs, TargetGraphSet):
            cleaned_map = TargetGraphSet(cleaned_map, kernels=graphs.effective_kernels, selection=graphs.default_selection)
        cohort_rep = operator_costs(eff_case, cleaned_map)
        r0_rows = [row for row in cohort_rep["operators"] if row.get("replica", 0) == 0]
        op_rep = {**cohort_rep, "operator_occurrences": sum(r["occurrences"] for r in r0_rows), "operator_sum_ms": sum(r["total_ms"] for r in r0_rows), "operators": r0_rows}

    if any("device_kernel_group" in dict(op.metadata) for op in r0_ops) and graph_rep is not None:
        group_by_op = {op.op_id: str(dict(op.metadata)["device_kernel_group"]["id"]) for op in r0_ops if isinstance(dict(op.metadata).get("device_kernel_group"), Mapping) and dict(op.metadata)["device_kernel_group"].get("id")}
        extra_by_group: dict[str, float] = defaultdict(float)
        for tl in graph_rep.get("timeline", ()):
            if tl.get("replica", 0) == 0 and tl.get("recipe") == "hccl_exit" and float(tl.get("elapsed_ms", 0.0) or 0.0) > 0 and str(tl.get("op_id", "")) in group_by_op:
                extra_by_group[group_by_op[str(tl["op_id"])]] += float(tl["elapsed_ms"])
        if extra_by_group:
            rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in op_rep["operators"]:
                raw_m = r["operator"].get("metadata") or ()
                dkg = (dict(raw_m) if isinstance(raw_m, Mapping) else {k: v for k, v in raw_m}).get("device_kernel_group")
                if isinstance(dkg, Mapping) and dkg.get("id"):
                    rows_by_group[str(dkg["id"])].append(r)
            for gid, g_extra in extra_by_group.items():
                g_rows = rows_by_group.get(gid, ())
                target_rows = [r for r in g_rows if r["operator"].get("link_bytes", 0.0) > 0] or list(g_rows)
                occ = sum(r["occurrences"] for r in target_rows)
                for r in (target_rows if occ > 0 else ()):
                    r["duration_ms"] += g_extra / occ
                    r["total_ms"] = r["duration_ms"] * r["occurrences"]
            op_rep["operator_sum_ms"] = sum(r["total_ms"] for r in op_rep["operators"])
    return op_rep, graph_rep


def analyze_network_efficiency(
    case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1, context: int = 4096, sequences: int = 1, scope: str = "body",
    operator_report: Mapping[str, Any] | None = None, graph_report: Mapping[str, Any] | None = None,
    observed_durations_ms: Mapping[str, float] | None = None, pmu_by_op: Mapping[str, Mapping[str, float]] | None = None,
    run_graph: bool = True, _precomputed_graph: tuple[Any, CaseSpec, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute whole-network 4D MFU/MBU/CV efficiency and bottleneck breakdowns."""
    _validate_geometry(phase, tokens, context, sequences, scope)
    parsed = parse_case(case)
    if operator_report is None:
        operator_report, auto_graph = _fast_operator_and_graph_reports(parsed, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, run_graph=run_graph, precomputed_graph=_precomputed_graph)
        graph_report = auto_graph if graph_report is None else graph_report

    raw_rows = [r for r in operator_report["operators"] if r.get("replica", 0) == 0] or list(operator_report["operators"])
    peaks_by_stage, analyzed_ops, obs_map, pmu_map = {}, [], dict(observed_durations_ms or {}), dict(pmu_by_op or {})
    for row in raw_rows:
        op, st = row["operator"], int(row["operator"].get("stage", 0))
        if st not in peaks_by_stage:
            peaks_by_stage[st] = extract_hardware_peaks(parsed, stage=st, replica=0)
        op_id, phys_id, rec_key = op.get("op_id", ""), row.get("physical_id", ""), f"{op.get('family', '')}:{op.get('recipe', '')}"
        analyzed_ops.append(analyze_operator_row(row, parsed, peaks=peaks_by_stage[st], observed_duration_ms=obs_map.get(op_id, obs_map.get(phys_id, obs_map.get(rec_key))), pmu=pmu_map.get(op_id, pmu_map.get(phys_id, pmu_map.get(rec_key)))))

    operator_sum_ms = sum(x["total_ms"] for x in analyzed_ops)
    graph_duration_ms = float(graph_report["duration_ms"]) if (graph_report is not None and observed_durations_ms is None) else operator_sum_ms
    safe_graph_ms = max(graph_duration_ms, 1e-12)
    tot_flops, tot_vec_ops = sum(x["flops"] * x["occurrences"] for x in analyzed_ops), sum(x["vector_ops"] * x["occurrences"] for x in analyzed_ops)
    tot_hbm_b, tot_link_b = sum(x["hbm_bytes"] * x["occurrences"] for x in analyzed_ops), sum(x["link_bytes"] * x["occurrences"] for x in analyzed_ops)
    tot_cube_id, tot_vec_id = sum(x["t_cube_ideal_ms"] * x["occurrences"] for x in analyzed_ops), sum(x["t_vec_ideal_ms"] * x["occurrences"] for x in analyzed_ops)
    tot_hbm_id, tot_link_id = sum(x["t_hbm_ideal_ms"] * x["occurrences"] for x in analyzed_ops), sum(x["t_link_ideal_ms"] * x["occurrences"] for x in analyzed_ops)
    serial_roof_id, tot_roof_id = sum(x["roofline_ideal_ms"] * x["occurrences"] for x in analyzed_ops), max(tot_hbm_id, tot_cube_id, tot_vec_id, tot_link_id)
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

    return {
        "schema": "mindie-sim.efficiency/1", "geometry": {"phase": phase, "tokens": tokens, "context": context, "sequences": sequences, "scope": scope},
        "total_flops": tot_flops, "total_vector_ops": tot_vec_ops, "total_hbm_bytes": tot_hbm_b, "total_link_bytes": tot_link_b,
        "total_cube_ideal_ms": tot_cube_id, "total_vec_ideal_ms": tot_vec_id, "total_hbm_ideal_ms": tot_hbm_id, "total_link_ideal_ms": tot_link_id,
        "total_roofline_ideal_ms": tot_roof_id, "operator_serial_roofline_ideal_ms": serial_roof_id, "operator_sum_ms": operator_sum_ms, "graph_duration_ms": graph_duration_ms,
        "network_cube_mfu": tot_cube_id / safe_graph_ms, "network_vec_mfu": tot_vec_id / safe_graph_ms, "network_mbu": tot_hbm_id / safe_graph_ms,
        "network_link_util": tot_link_id / safe_graph_ms, "network_roofline_efficiency": tot_roof_id / safe_graph_ms,
        "operator_serial_roofline_efficiency": serial_roof_id / max(operator_sum_ms, 1e-12), "network_cv_overlap": network_cv_overlap,
        "breakdown_by_family": sorted([{"family": f, "recipe": r, **_agg(items)} for (f, r), items in fam_g.items()], key=lambda r: -r["total_ms"]),
        "breakdown_by_bound": {b: _agg(bound_g.get(b, ())) for b in _BOUNDS}, "breakdown_by_kind": {k: _agg(kind_g.get(k, ())) for k in _KINDS}, "operators": analyzed_ops,
    }


def evaluate_target_efficiency(
    case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1, context: int = 4096, sequences: int = 1, scope: str = "body",
    target_mbu: float = 0.80, target_cube_mfu: float = 0.75, target_vec_mfu: float = 0.65, target_link_util: float | None = None,
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

    base = dict(baseline_analysis) if baseline_analysis is not None else analyze_network_efficiency(case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope)
    fam_ovr, evaluated_ops = dict(family_targets or {}), []
    for op in base["operators"]:
        ovr = fam_ovr.get(f"{op['family']}:{op['recipe']}") or fam_ovr.get(op["recipe"]) or fam_ovr.get(op["family"]) or {}
        eta_hbm, eta_cube, eta_vec, eta_link = float(ovr.get("target_mbu", target_mbu)), float(ovr.get("target_cube_mfu", target_cube_mfu)), float(ovr.get("target_vec_mfu", target_vec_mfu)), ovr.get("target_link_util", target_link_util)
        for k, v in (("family.target_mbu", eta_hbm), ("family.target_cube_mfu", eta_cube), ("family.target_vec_mfu", eta_vec), ("family.target_link_util", eta_link)):
            _validate_efficiency_param(k, v)
        t_fixed = float(ovr["fixed_launch_us"]) / 1000.0 if "fixed_launch_us" in ovr else float(fixed_launch_us) / 1000.0 if fixed_launch_us is not None else float(op["fixed_ms"])
        t_min = float(op.get("minimum_ms", 0.0) or 0.0)
        t_link_req = (op["t_link_ideal_ms"] / float(eta_link)) if eta_link is not None else (op["duration_ms"] if op["link_bytes"] > 0 and op["hbm_bytes"] == 0 and op["flops"] == 0 and op["vector_ops"] == 0 else op["t_link_ideal_ms"] / 0.75)
        inclusive_pipe_ms = max(op["t_hbm_ideal_ms"] / eta_hbm, op["t_cube_ideal_ms"] / eta_cube, op["t_vec_ideal_ms"] / eta_vec, t_link_req)
        raw_t = max(t_min, t_fixed + op["roofline_ideal_ms"], inclusive_pipe_ms) if op["roofline_ideal_ms"] > 0 else max(t_min, op["duration_ms"])
        eval_ms = max(t_min, min(op["duration_ms"], raw_t)) if mode == "clamp_to_target" else raw_t
        cur_tot, tgt_tot, safe_eval, safe_svc = op["total_ms"], eval_ms * op["occurrences"], max(eval_ms, 1e-12), max(eval_ms - t_fixed, 1e-12)
        evaluated_ops.append({
            **op, "actual_mbu": op["mbu"], "actual_cube_mfu": op["cube_mfu"], "actual_vec_mfu": op["vec_mfu"],
            "current_duration_ms": op["duration_ms"], "raw_target_duration_ms": raw_t, "target_duration_ms": eval_ms,
            "current_total_ms": cur_tot, "target_total_ms": tgt_tot, "gap_ms": cur_tot - tgt_tot,
            "target_mbu": (op["t_hbm_ideal_ms"] / safe_eval) if op["hbm_bytes"] > 0 else None, "service_only_target_mbu": (op["t_hbm_ideal_ms"] / safe_svc) if op["hbm_bytes"] > 0 else None,
            "target_cube_mfu": (op["t_cube_ideal_ms"] / safe_eval) if op["flops"] > 0 else None, "target_vec_mfu": (op["t_vec_ideal_ms"] / safe_eval) if op["vector_ops"] > 0 else None,
        })

    cur_sum, tgt_sum, cur_step = base["operator_sum_ms"], sum(x["target_total_ms"] for x in evaluated_ops), base["graph_duration_ms"]
    tgt_step = cur_step * ((tgt_sum / cur_sum) if cur_sum > 0 else 1.0)
    saved_ms, safe_tgt = cur_step - tgt_step, max(tgt_step, 1e-12)
    speedup = cur_step / safe_tgt
    return {
        "mode": mode, "targets": {"target_mbu": target_mbu, "target_cube_mfu": target_cube_mfu, "target_vec_mfu": target_vec_mfu, "target_link_util": target_link_util, "fixed_launch_us": fixed_launch_us},
        "current_step_duration_ms": cur_step, "target_step_duration_ms": tgt_step, "current_operator_sum_ms": cur_sum, "target_operator_sum_ms": tgt_sum,
        "saved_ms": saved_ms, "speedup_ratio": speedup, "step_speedup": speedup, "speedup_pct": (speedup - 1.0) * 100.0,
        "latency_reduction_pct": (saved_ms / max(cur_step, 1e-12)) * 100.0, "current_network_mbu": base["network_mbu"], "target_network_mbu": base["total_hbm_ideal_ms"] / safe_tgt,
        "current_network_cube_mfu": base["network_cube_mfu"], "target_network_cube_mfu": base["total_cube_ideal_ms"] / safe_tgt,
        "current_network_vec_mfu": base["network_vec_mfu"], "target_network_vec_mfu": base["total_vec_ideal_ms"] / safe_tgt,
        "current_network_roofline_efficiency": base["network_roofline_efficiency"], "target_network_roofline_efficiency": base["total_roofline_ideal_ms"] / safe_tgt,
        "top_bottlenecks": sorted(evaluated_ops, key=lambda r: -r["gap_ms"])[:15], "operators": evaluated_ops,
    }


def _operator_fit_signature(op: Mapping[str, Any]) -> str:
    return json.dumps({"op_id": op.get("op_id", ""), "family": op.get("family", ""), "recipe": op.get("recipe", ""), "dtype": op.get("dtype", ""), "local_shape": list(op.get("local_shape") or ()), "flops": round(float(op.get("flops", 0.0)), 6), "vector_ops": round(float(op.get("vector_ops", 0.0)), 6), "hbm_bytes": round(float(op.get("hbm_bytes", 0.0)), 6), "link_bytes": round(float(op.get("link_bytes", 0.0)), 6)}, sort_keys=True, separators=(",", ":"))


def _decompose_operator_pipes(op: Mapping[str, Any]) -> dict[str, float]:
    """Decompose an operator's baseline variable duration across active physical pipes."""
    t_base, t_min = float(op["duration_ms"]), float(op.get("minimum_ms", 0.0) or 0.0)
    t_fixed = min(max(float(op["fixed_ms"]), t_min), t_base)
    delta_t, ideals = max(0.0, t_base - t_fixed), {"hbm": float(op["t_hbm_ideal_ms"]), "cube": float(op["t_cube_ideal_ms"]), "vec": float(op["t_vec_ideal_ms"]), "link": float(op["t_link_ideal_ms"])}
    max_id = max(ideals.values())
    if max_id <= 0.0 or delta_t <= 0.0:
        return {"fixed_ms": t_base, "minimum_ms": t_min, "t_hbm_eff_ms": 0.0, "t_cube_eff_ms": 0.0, "t_vec_eff_ms": 0.0, "t_link_eff_ms": 0.0, "effective_mbu": 0.0, "effective_cube_mfu": 0.0, "effective_vec_mfu": 0.0, "effective_link_util": 0.0}
    dom, eta_dom, def_etas, eff_t, eff_e = max(ideals, key=ideals.__getitem__), _clamp(max_id / delta_t, 1e-4, 1.0), {"hbm": 0.80, "cube": 0.75, "vec": 0.65, "link": 0.75}, {}, {}
    for p, t_id in ideals.items():
        if t_id <= 0.0:
            eff_t[p], eff_e[p] = 0.0, 0.0
        elif p == dom:
            eff_t[p], eff_e[p] = delta_t, eta_dom
        else:
            eff_t[p] = min(delta_t, max(t_id, t_id / _clamp(max(eta_dom, def_etas[p]), 1e-4, 1.0)))
            eff_e[p] = _clamp(t_id / max(eff_t[p], 1e-12), 1e-4, 1.0)
    return {"fixed_ms": t_fixed, "minimum_ms": t_min, "t_hbm_eff_ms": eff_t["hbm"], "t_cube_eff_ms": eff_t["cube"], "t_vec_eff_ms": eff_t["vec"], "t_link_eff_ms": eff_t["link"], "effective_mbu": eff_e["hbm"], "effective_cube_mfu": eff_e["cube"], "effective_vec_mfu": eff_e["vec"], "effective_link_util": eff_e["link"]}


def fit_case_roofline_regimes(case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1, context: int = 4096, sequences: int = 1, scope: str = "body", observed_durations_ms: Mapping[str, float] | None = None, baseline_analysis: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit multi-pipe roofline parameters (fixed_us, effective_mbu/cube_mfu/vec_mfu) per operator."""
    base = dict(baseline_analysis) if baseline_analysis is not None else analyze_network_efficiency(case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, observed_durations_ms=observed_durations_ms)
    fitted_ops, by_key, by_recipe = [], {}, defaultdict(list)
    for op in base["operators"]:
        dec, sig_key = _decompose_operator_pipes(op), _operator_fit_signature(op)
        by_key[sig_key] = dec
        by_recipe[f"{op['family']}:{op['recipe']}"].append(dec)
        fitted_ops.append({"signature_key": sig_key, "op_id": op["op_id"], "family": op["family"], "recipe": op["recipe"], "operator_kind": op["operator_kind"], "dominant_bound": op["dominant_bound"], "occurrences": op["occurrences"], "duration_ms": op["duration_ms"], "fixed_us": dec["fixed_ms"] * 1000.0, **dec})
    summaries = [{"family": rk.split(":", 1)[0], "recipe": rk.split(":", 1)[1], "fixed_us": sum(x["fixed_ms"] for x in items) / len(items) * 1000.0, "effective_mbu": max(x["effective_mbu"] for x in items), "effective_cube_mfu": max(x["effective_cube_mfu"] for x in items), "effective_vec_mfu": max(x["effective_vec_mfu"] for x in items), "effective_link_util": max(x["effective_link_util"] for x in items)} for rk, items in sorted(by_recipe.items())]
    return {"schema": "mindie-sim.roofline-fit/1", "geometry": base["geometry"], "fitted_regimes": summaries, "operators": fitted_ops, "by_key": by_key}


def fit_efficiency_targets(case: CaseSpec | Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Combined entry point returning baseline analysis, target evaluation, and fitted regimes."""
    base = analyze_network_efficiency(case, phase=kwargs.get("phase", "decode"), tokens=kwargs.get("tokens", 1), context=kwargs.get("context", 4096), sequences=kwargs.get("sequences", 1), scope=kwargs.get("scope", "body"))
    return {**evaluate_target_efficiency(case, baseline_analysis=base, **kwargs), "baseline_analysis": base, "fitted_roofline_regimes": fit_case_roofline_regimes(case, baseline_analysis=base)["fitted_regimes"]}


def _scale_service_regimes(regimes: Sequence[Mapping[str, Any]], *, mode_to_role: Mapping[str, str], hbm_bw_scale: float, cube_flops_scale: float, vector_ops_scale: float, link_bw_scale: float) -> list[dict[str, Any]]:
    """Scale regime rate guards and explicitly work-parameterized response terms."""
    kw, scaled = dict(hbm_bw_scale=hbm_bw_scale, cube_flops_scale=cube_flops_scale, vector_ops_scale=vector_ops_scale, link_bw_scale=link_bw_scale), []
    for raw in regimes:
        rule = deepcopy(dict(raw))
        match = rule.get("match", {})
        for k in list(match.keys()):
            if k.startswith("physical_rates.resources."):
                match[k] = float(match[k]) * _role_scale_factor(k.split(".")[-1], **kw)
            elif k.startswith("physical_rates.compute_peaks."):
                match[k] = float(match[k]) * cube_flops_scale
            elif k.startswith("physical_rates.service_modes.") and isinstance(match[k], (list, tuple)) and len(match[k]) == 2:
                m_name = k.split(".")[-1]
                match[k] = [match[k][0], float(match[k][1]) * _role_scale_factor(str(match[k][0] or mode_to_role.get(m_name, m_name)), **kw)]
            elif k == "transport" and isinstance(match[k], Mapping):
                for lnk in match[k].get("links", ()):
                    for lk in ("capacity_bytes_per_ms", "switch_capacity_bytes_per_ms"):
                        if lnk.get(lk) is not None:
                            lnk[lk] = float(lnk[lk]) * link_bw_scale
        resp, coords = rule.get("response", {}), rule.get("coordinates", {})
        if resp.get("kind", "affine") == "affine" and resp.get("terms"):
            for term in resp["terms"]:
                c_path = str((coords.get(term.get("coordinate")) or {}).get("path", "")) if not term.get("scale_with") else ""
                if c_path.startswith("work."):
                    term["us_per_unit"] = float(term["us_per_unit"]) / _role_scale_factor(c_path.split(".", 1)[1], **kw)
        elif resp.get("kind") == "piecewise_linear" and "knots" in resp:
            c_path = str((coords.get(resp.get("coordinate")) or {}).get("path", ""))
            if c_path.startswith("work."):
                sf = _role_scale_factor(c_path.split(".", 1)[1], **kw)
                resp["knots"] = [[x, float(y) / sf] for x, y in resp["knots"]]
        scaled.append(rule)
    return scaled


def scale_case_hardware(case: CaseSpec | Mapping[str, Any], *, hbm_bw_scale: float = 1.0, cube_flops_scale: float = 1.0, vector_ops_scale: float = 1.0, link_bw_scale: float = 1.0, hbm_capacity_scale: float = 1.0, baseline_analysis: Mapping[str, Any] | None = None) -> CaseSpec:
    """Scale hardware peaks, explicit/derived rates, service modes, links, and capacity."""
    for name, val in zip(_SCALE_KEYS, (hbm_bw_scale, cube_flops_scale, vector_ops_scale, link_bw_scale, hbm_capacity_scale)):
        _check_pos_float(f"hardware_scale_{name}", val)
    parsed = parse_case(case)
    if (hbm_bw_scale, cube_flops_scale, vector_ops_scale, link_bw_scale, hbm_capacity_scale) == (1.0, 1.0, 1.0, 1.0, 1.0):
        return parsed
    kw = dict(hbm_bw_scale=hbm_bw_scale, cube_flops_scale=cube_flops_scale, vector_ops_scale=vector_ops_scale, link_bw_scale=link_bw_scale)
    mode_to_role, machine_dict = {m_name: m_role for die in parsed.machine.dies for m_name, (m_role, _) in die.service_modes}, deepcopy(parsed.machine.to_dict())
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
        for lk in ("capacity_bytes_per_ms", "switch_capacity_bytes_per_ms"):
            if lnk.get(lk) is not None:
                lnk[lk] = float(lnk[lk]) * link_bw_scale
    new_kernels = deepcopy(parsed.runtime.kernels)
    if "service_regimes" in new_kernels:
        new_kernels["service_regimes"] = _scale_service_regimes(new_kernels["service_regimes"], mode_to_role=mode_to_role, **kw)
    return replace(parsed, machine=Hardware.from_mapping(machine_dict), runtime=replace(parsed.runtime, kernels=new_kernels))


def _vram_max_sequences_per_dp(
    case: CaseSpec, *, context: int, tokens_per_seq: int, output_budget: int,
    hbm_capacity_scale: float = 1.0, emitted_tokens_per_seq: float | None = None,
    async_scheduling: bool | None = None,
) -> int:
    """Solve max concurrent sequences per DP replica jointly with workspace, KV, and async delayed release."""
    base_case = replace(case, runtime=replace(case.runtime, async_scheduling=bool(async_scheduling))) if async_scheduling is not None else case
    scaled = scale_case_hardware(base_case, hbm_capacity_scale=hbm_capacity_scale) if hbm_capacity_scale != 1.0 else base_case
    q_tokens = max(1, int(tokens_per_seq))
    emit_tokens = float(q_tokens) if emitted_tokens_per_seq is None else max(1e-6, float(emitted_tokens_per_seq))
    rt_cap = min(int(scaled.runtime.max_num_seqs), max(1, int(scaled.runtime.max_num_batched_tokens) // q_tokens))
    probe_mem = Memory(scaled, workspace_tokens=q_tokens)
    lookahead_tokens = probe_mem.async_in_flight_lookahead_steps * q_tokens
    delayed_steps = probe_mem.async_delayed_release_steps
    steps_per_wave = max(1, math.ceil(max(1, output_budget) / emit_tokens))
    retiring_waves = max(1, math.ceil(delayed_steps / steps_per_wave)) if delayed_steps > 0 else 0
    seq_len_active = max(1, context + max(q_tokens, output_budget) + lookahead_tokens)
    seq_len_retiring = max(1, context + max(q_tokens, output_budget))
    seq_len_wave = max(1, context + min(max(q_tokens, output_budget), max(1, delayed_steps) * q_tokens) + lookahead_tokens)
    kv_active = probe_mem.cache_bytes(seq_len_active, padded=True)
    kv_retiring = probe_mem.cache_bytes(seq_len_retiring, padded=True)
    kv_wave = probe_mem.cache_bytes(seq_len_wave, padded=True)

    def fits(b: int) -> bool:
        mem = Memory(scaled, workspace_tokens=b * q_tokens)
        if b >= 2 and delayed_steps > 0 and output_budget > 0:
            staggered_delayed_b = max(1, math.ceil(b * delayed_steps / steps_per_wave))
            wave_delayed_b = retiring_waves * b
            return all(
                min(mem.capacity[r][st] for r in range(len(mem.capacity))) >= max(
                    b * kv_active[st] + staggered_delayed_b * kv_retiring[st],
                    b * kv_wave[st] + wave_delayed_b * kv_retiring[st],
                )
                for st in range(len(kv_active))
            )
        return all(
            min(mem.capacity[r][st] for r in range(len(mem.capacity))) >= b * kv_active[st]
            for st in range(len(kv_active))
        )

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


def solve_hardware_sla_throughput(
    case: CaseSpec | Mapping[str, Any], *, sla_ms: float = 30.0, context: int = 126000, tokens_per_seq: int | None = None,
    emitted_tokens_per_seq: float | None = None, max_sequences_per_dp: int = 256, phase: str = "decode", scope: str = "body",
    output_budget: int = 1024, target_mbu: float = 0.80, target_cube_mfu: float = 0.75, target_vec_mfu: float = 0.65,
    scenarios: Sequence[Mapping[str, Any]] | None = None, b1_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve SLA-constrained maximum concurrency B*_SLA and throughput across hardware scenarios."""
    _check_pos_float("sla_ms", sla_ms)
    if isinstance(max_sequences_per_dp, bool) or not isinstance(max_sequences_per_dp, int) or max_sequences_per_dp < 1:
        raise ValueError(f"max_sequences_per_dp_must_be_positive_int:{max_sequences_per_dp}")
    if isinstance(output_budget, bool) or not isinstance(output_budget, int) or output_budget < 0:
        raise ValueError(f"output_budget_must_be_nonnegative_int:{output_budget}")
    for k, v in (("target_mbu", target_mbu), ("target_cube_mfu", target_cube_mfu), ("target_vec_mfu", target_vec_mfu)):
        _validate_efficiency_param(k, v)

    parsed = parse_case(case)
    q_tokens, emit_tokens = _resolve_speculative_tokens(parsed, output_budget=output_budget, tokens_per_seq=tokens_per_seq, emitted_tokens_per_seq=emitted_tokens_per_seq)
    _validate_geometry(phase, q_tokens, context, 1, scope)
    rt_max_b = min(int(parsed.runtime.max_num_seqs), max(1, int(parsed.runtime.max_num_batched_tokens) // q_tokens))
    if phase == "decode" and parsed.runtime.graph_capture_mode == "full" and parsed.runtime.graph_capture_keys and dict(parsed.runtime.metadata).get("captured_execution_mode") == "FULL_DECODE_ONLY":
        rt_max_b = min(rt_max_b, max(1, max(parsed.runtime.graph_capture_keys) // q_tokens))
    b_max, dp = min(int(max_sequences_per_dp), max(1, rt_max_b)), parsed.runtime.parallel.dp

    t_defaults = {"target_mbu": target_mbu, "target_cube_mfu": target_cube_mfu, "target_vec_mfu": target_vec_mfu}
    scenario_list = [({**s, **t_defaults} if (s.get("id") == "target_0.8x_mbu" and scenarios is None) else ({**t_defaults, **s} if s.get("id") == "target_0.8x_mbu" else dict(s))) for s in (scenarios or DEFAULT_SLA_SCENARIOS)]
    for s in scenario_list:
        for k in _SCALE_KEYS:
            _check_pos_float(f"scenario_scale_{k}", s.get(k, 1.0))
        for k in ("target_mbu", "target_cube_mfu", "target_vec_mfu", "target_link_util"):
            _validate_efficiency_param(k, s.get(k))

    scaled_cases: dict[tuple[float, ...], CaseSpec] = {}
    graphs_by_b: dict[int, Any] = {}
    vram_by_cap: dict[float, tuple[int, int, int]] = {}
    graph_cache: dict[tuple[tuple[float, ...], int], tuple[Any, CaseSpec, dict[str, Any]]] = {}
    analysis_cache: dict[tuple[tuple[float, ...], int], dict[str, Any]] = {}

    def get_scaled(s_key: tuple[float, ...]) -> CaseSpec:
        if s_key not in scaled_cases:
            scaled_cases[s_key] = scale_case_hardware(parsed, **dict(zip(_SCALE_KEYS, s_key)))
        return scaled_cases[s_key]

    def get_bundle(s_key: tuple[float, ...], b: int) -> tuple[Any, CaseSpec, dict[str, Any]]:
        if (s_key, b) not in graph_cache:
            if b not in graphs_by_b:
                graphs_by_b[b] = target_graphs(parsed, phase=phase, tokens=q_tokens, context=context, sequences=b, scope=scope)
            g, ec, gr = _lower_and_run_graph(get_scaled(s_key), phase=phase, tokens=q_tokens, context=context, sequences=b, scope=scope, run_graph=True, lowered_graphs=graphs_by_b[b])
            assert gr is not None
            graph_cache[s_key, b] = (g, ec, gr)
        return graph_cache[s_key, b]

    def get_analysis(s_key: tuple[float, ...], b: int) -> dict[str, Any]:
        if (s_key, b) not in analysis_cache:
            if s_key == (1.0, 1.0, 1.0, 1.0, 1.0) and b == 1 and b1_analysis is not None and b1_analysis.get("geometry", {}).get("tokens") == q_tokens:
                analysis_cache[s_key, b] = dict(b1_analysis)
            else:
                analysis_cache[s_key, b] = analyze_network_efficiency(get_scaled(s_key), phase=phase, tokens=q_tokens, context=context, sequences=b, scope=scope, run_graph=True, _precomputed_graph=get_bundle(s_key, b))
        return analysis_cache[s_key, b]

    def eval_at(scen: Mapping[str, Any], s_key: tuple[float, ...], b: int, *, full: bool = False) -> dict[str, Any]:
        if not full and scen.get("target_mbu") is None:
            return {"step_ms": float(get_bundle(s_key, b)[2]["duration_ms"])}
        ans = get_analysis(s_key, b)
        shares = {k: ans["breakdown_by_bound"][k]["share_of_operator_sum"] for k in _BOUNDS}
        if scen.get("target_mbu") is not None:
            ev = evaluate_target_efficiency(
                get_scaled(s_key), phase=phase, tokens=q_tokens, context=context, sequences=b, scope=scope,
                target_mbu=float(scen["target_mbu"]), target_cube_mfu=float(scen.get("target_cube_mfu", target_cube_mfu)),
                target_vec_mfu=float(scen.get("target_vec_mfu", target_vec_mfu)), target_link_util=scen.get("target_link_util"), baseline_analysis=ans,
            )
            step = float(ev["target_step_duration_ms"])
            return {"step_ms": step, "operator_sum_ms": float(ev["target_operator_sum_ms"]), "network_mbu": float(ev["target_network_mbu"]), "network_cube_mfu": float(ev["target_network_cube_mfu"]), "network_vec_mfu": float(ev["target_network_vec_mfu"]), "network_link_util": ans["total_link_ideal_ms"] / max(step, 1e-12), "network_roofline_efficiency": float(ev["target_network_roofline_efficiency"]), "bottleneck_shares": shares}
        return {"step_ms": float(ans["graph_duration_ms"]), "operator_sum_ms": float(ans["operator_sum_ms"]), "network_mbu": float(ans["network_mbu"]), "network_cube_mfu": float(ans["network_cube_mfu"]), "network_vec_mfu": float(ans["network_vec_mfu"]), "network_link_util": float(ans["network_link_util"]), "network_roofline_efficiency": float(ans["network_roofline_efficiency"]), "bottleneck_shares": shares}

    base_b_hint, domain_cap, domain_reason = 1, b_max, None

    def try_step_ok(scen: Mapping[str, Any], s_key: tuple[float, ...], b: int) -> bool:
        nonlocal domain_cap, domain_reason
        if b > domain_cap:
            return False
        try:
            return eval_at(scen, s_key, b, full=False)["step_ms"] <= sla_ms
        except ModelingError as exc:
            if b <= 1:
                raise
            domain_cap, domain_reason = min(domain_cap, b - 1), str(exc)
            return False

    def solve_one(scen: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal base_b_hint
        s_key = tuple(float(scen.get(k, 1.0)) for k in _SCALE_KEYS)
        if s_key[4] not in vram_by_cap:
            b_sync = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=s_key[4], emitted_tokens_per_seq=emit_tokens, async_scheduling=False)
            b_async = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=s_key[4], emitted_tokens_per_seq=emit_tokens, async_scheduling=True)
            b_cfg = b_async if parsed.runtime.async_scheduling else b_sync
            if parsed.runtime.memory:
                b_cfg = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=s_key[4], emitted_tokens_per_seq=emit_tokens)
            vram_by_cap[s_key[4]] = (b_cfg, b_sync, b_async)
        (b_vram, b_vram_sync, b_vram_async), b1_ms = vram_by_cap[s_key[4]], eval_at(scen, s_key, 1, full=False)["step_ms"]
        eff_max = min(b_max, domain_cap)
        if b1_ms > sla_ms:
            b_sla, eval_sla = 0, eval_at(scen, s_key, 1, full=True)
        elif eff_max > 1 and (b1_ms * 3.0 <= sla_ms or base_b_hint == eff_max) and try_step_ok(scen, s_key, eff_max):
            b_sla, eval_sla = eff_max, eval_at(scen, s_key, eff_max, full=True)
        else:
            eff_max = min(b_max, domain_cap)
            start_b = max(2, min(eff_max, base_b_hint)) if (base_b_hint > 1 and all(x >= 1.0 for x in s_key[:4])) else 2
            low, high, probe = 1, eff_max, min(eff_max, start_b)
            if probe > 2 and not try_step_ok(scen, s_key, probe):
                high, probe = min(high, domain_cap, probe - 1), 2
            while probe <= high:
                if try_step_ok(scen, s_key, probe):
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
                best_b, lo_i, hi_i = (mid, mid + 1, hi_i) if try_step_ok(scen, s_key, mid) else (best_b, lo_i, min(hi_i, domain_cap, mid - 1))
            while best_b > 1 and eval_at(scen, s_key, best_b)["step_ms"] > sla_ms:
                best_b -= 1
            b_sla = best_b if eval_at(scen, s_key, best_b)["step_ms"] <= sla_ms else 0
            eval_sla = eval_at(scen, s_key, max(1, b_sla), full=True)
        if scen.get("id") == "baseline" and b_sla > 0:
            base_b_hint = b_sla
        b_feas = min(b_sla, b_vram)
        ev_feas = eval_at(scen, s_key, b_feas, full=True) if (b_feas > 0 and b_feas != b_sla) else eval_sla
        tps_sla = (b_sla * dp * emit_tokens * 1000.0) / eval_sla["step_ms"] if b_sla > 0 else 0.0
        return {
            "scenario_id": scen.get("id", "custom"), "label": scen.get("label", scen.get("id", "custom")), "scales": dict(zip(_SCALE_KEYS, s_key)),
            "b_sla": b_sla, "b_vram_max": b_vram, "b_vram_max_sync": b_vram_sync, "b_vram_max_async": b_vram_async,
            "delta_b_vram_from_disabling_async": max(0, b_vram_sync - b_vram_async), "hbm_bound": bool(b_vram < b_sla),
            "b_feasible": b_feas, "domain_max_b": domain_cap, "domain_limit_reason": domain_reason,
            "total_concurrency": b_sla * dp, "total_feasible_concurrency": b_feas * dp,
            "step_ms_at_b1": b1_ms, "tps_at_b1": (dp * emit_tokens * 1000.0) / b1_ms, "step_ms_at_sla": eval_sla["step_ms"], "tps_at_sla": tps_sla,
            "target_step_tps_at_sla": (b_sla * dp * q_tokens * 1000.0) / eval_sla["step_ms"] if b_sla > 0 else 0.0,
            "step_ms_feasible": ev_feas["step_ms"], "tps_feasible": (b_feas * dp * emit_tokens * 1000.0) / ev_feas["step_ms"] if b_feas > 0 else 0.0,
            "network_mbu": eval_sla["network_mbu"], "network_mbu_at_sla": eval_sla["network_mbu"], "network_cube_mfu": eval_sla["network_cube_mfu"],
            "network_cube_mfu_at_sla": eval_sla["network_cube_mfu"], "network_vec_mfu": eval_sla["network_vec_mfu"],
            "network_vec_mfu_at_sla": eval_sla["network_vec_mfu"], "network_link_util": eval_sla["network_link_util"],
            "network_roofline_efficiency": eval_sla["network_roofline_efficiency"], "bottleneck_shares": eval_sla["bottleneck_shares"],
        }

    results = [solve_one(scen) for scen in scenario_list]
    base_r = next((r for r in results if r["scenario_id"] == "baseline"), next((r for r in results if tuple(r["scales"].values()) == (1.0, 1.0, 1.0, 1.0, 1.0) and r["scenario_id"] != "target_0.8x_mbu"), None)) or solve_one(DEFAULT_SLA_SCENARIOS[0])
    base_tps, base_feas_tps, base_b = (base_r["tps_at_sla"] if base_r["tps_at_sla"] > 0 else base_r["tps_at_b1"]), (base_r["tps_feasible"] if base_r["tps_feasible"] > 0 else base_r["tps_at_b1"]), max(1, base_r["b_sla"])
    for row in results:
        eff_tps, eff_feas = (row["tps_at_sla"] if base_r["tps_at_sla"] > 0 else row["tps_at_b1"]), (row["tps_feasible"] if base_r["tps_feasible"] > 0 else row["tps_at_b1"])
        row["tps_gain_vs_baseline"], row["tps_feasible_gain_vs_baseline"], row["concurrency_gain_vs_baseline"] = ((eff_tps / base_tps) if base_tps > 0 else 1.0), ((eff_feas / base_feas_tps) if base_feas_tps > 0 else 1.0), ((row["b_sla"] / base_b) if base_r["b_sla"] > 0 else (base_r["step_ms_at_b1"] / row["step_ms_at_b1"]))
    mem_async_b1 = Memory(replace(parsed, runtime=replace(parsed.runtime, async_scheduling=True)), workspace_tokens=q_tokens)
    seq_sync_len = max(1, context + max(q_tokens, output_budget))
    seq_async_len = max(1, seq_sync_len + mem_async_b1.async_in_flight_lookahead_steps * q_tokens)
    async_mem_summary = {
        "configured_async_scheduling": bool(parsed.runtime.async_scheduling),
        "b_vram_max_sync": base_r["b_vram_max_sync"],
        "b_vram_max_async": base_r["b_vram_max_async"],
        "delta_b_vram_from_disabling_async": base_r["delta_b_vram_from_disabling_async"],
        "delayed_release_steps_async": mem_async_b1.async_delayed_release_steps,
        "in_flight_lookahead_tokens_async": mem_async_b1.async_in_flight_lookahead_steps * q_tokens,
        "kv_bytes_per_seq_sync_per_stage": list(mem_async_b1.cache_bytes(seq_sync_len, padded=True)),
        "kv_bytes_per_seq_async_per_stage": list(mem_async_b1.cache_bytes(seq_async_len, padded=True)),
        "async_workspace_overhead_bytes_at_b1_per_stage": list(mem_async_b1.async_workspace),
        "hbm_bound_at_baseline": bool(base_r["b_vram_max"] < base_r["b_sla"]),
        "disabling_async_unlocks_concurrency": bool(base_r["b_vram_max_sync"] > base_r["b_vram_max_async"] and base_r["b_vram_max_async"] < base_r["b_sla"]),
    }
    return {"schema": "mindie-sim.hardware-sla/1", "sla_ms": sla_ms, "context": context, "phase": phase, "scope": scope, "timing_boundary": "target_body_only" if scope == "body" else "full_model_step", "tokens_per_seq": q_tokens, "emitted_tokens_per_seq": emit_tokens, "dp": dp, "max_sequences_per_dp": b_max, "async_scheduling_memory": async_mem_summary, "scenarios": results}
