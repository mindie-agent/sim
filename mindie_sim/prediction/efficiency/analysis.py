"""Network efficiency, occurrence DAGs and hardware scaling.

[Role] Aggregate operator work, report physical critical-path bounds and
  construct hardware-scaled cases.
[Invariants] Occurrence reports preserve multiplicity, realized dependencies
  and resource identities. Scaling preserves the case's modeling boundary.
[Timeouts & Locks] Uses the simulator's evaluation context and active deadline.
[Side-Effect Policy] Analysis reads input records and scaling returns a new
  case when rates or capacity change.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.hardware import Hardware

from ..physical_key import normalize_reference
from ..roofline_graph import (
    PhysicalWorkGraph, PhysicalWorkNode, RooflineTargets, critical_path_roofline,
    graph_from_occurrences,
)
from ..structure import parse_case
from .peaks import (
    DEFAULT_CONTEXT, DEFAULT_SEQUENCES, DEFAULT_TOKENS, _BOUNDS, _KINDS,
    _SCALE_KEYS, _check_pos_float, _role_scale_factor, _validate_geometry,
    analyze_operator_row, extract_hardware_peaks, legacy_service_regime_diagnostics,
)


_OCCURRENCE_IDENTITY = re.compile(r"(?<![\w.])s(\d+)\.l(\d+)")
_WORK_SIGNATURE_PRECISION_DIGITS = 6


def occurrence_work_graph(graph_report: Mapping[str, Any] | None,
                         operator_rows: Sequence[Mapping[str, Any]]):
    """Realized occurrence DAG with per-node physical service fields."""
    if not graph_report:
        return None
    timeline = tuple(graph_report.get("timeline", ()) or ())
    if not timeline:
        return None
    service: dict[str, dict[str, float]] = {}
    relative: dict[str, dict[str, float]] = {}
    for row in operator_rows:
        op = row.get("operator") if isinstance(row.get("operator"), Mapping) else row
        op_id = str(op.get("op_id") or row.get("op_id") or "")
        if not op_id:
            continue
        metrics = {
            "cube_ms": float(row.get("t_cube_ideal_ms", 0.0) or 0.0),
            "vector_ms": float(row.get("t_vec_ideal_ms", 0.0) or 0.0),
            "hbm_ms": float(row.get("t_hbm_ideal_ms", 0.0) or 0.0),
            "link_ms": float(row.get("t_link_ideal_ms", 0.0) or 0.0),
            "fixed_ms": float(row.get("fixed_ms", 0.0) or 0.0),
        }
        service[op_id] = metrics
        relative.setdefault(normalize_reference(
            op_id, int(row.get("stage", 0)), int(row.get("layer", 0))), metrics)
    rows = []
    for item in timeline:
        op_id = str(item.get("op_id") or "")
        matched = service.get(op_id)
        if matched is None and op_id:
            match = _OCCURRENCE_IDENTITY.search(op_id)
            if match is not None:
                matched = relative.get(normalize_reference(
                    op_id, int(match.group(1)), int(match.group(2))))
        rows.append({**item, **(matched or {})})
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
        graph, RooflineTargets(cube=1.0, vector=1.0,
                               hbm=1.0, link=1.0))
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


OCCURRENCE_DAG_SCHEMA = "mindie-sim.occurrence-dag/1"


def _export_occurrence_dag(graph):
    """Serialize the realized occurrence DAG as public node/edge rows.

    The object graph stays available through ``_retain_graph``. The compact
    public form preserves realized predecessors and physical resource identities
    when a report is serialized.
    """
    if graph is None:
        return None
    return {
        "schema": OCCURRENCE_DAG_SCHEMA,
        "nodes": tuple((
            str(node.node_id),
            float(node.cube_ms), float(node.vector_ms),
            float(node.hbm_ms), float(node.link_ms),
            float(node.launch_ms), float(node.fixed_ms),
            tuple(str(before) for before in node.predecessors),
            int(node.occurrences),
            tuple((str(resource), str(identity))
                  for resource, identity in node.resource_ids),
        ) for node in graph.nodes),
        "edges": tuple((str(before), str(after))
                       for before, after in graph.edges),
    }


def _occurrence_dag_graph(payload):
    """Rebuild the realized occurrence DAG from its public node/edge form."""
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema") != OCCURRENCE_DAG_SCHEMA:
        return None
    nodes = tuple(PhysicalWorkNode(
        node_id=str(node_id),
        cube_ms=float(cube_ms), vector_ms=float(vector_ms),
        hbm_ms=float(hbm_ms), link_ms=float(link_ms),
        launch_ms=float(launch_ms), fixed_ms=float(fixed_ms),
        predecessors=tuple(str(before) for before in predecessors),
        occurrences=int(occurrences),
        resource_ids=tuple((str(resource), str(identity))
                           for resource, identity in resource_ids),
    ) for (node_id, cube_ms, vector_ms, hbm_ms, link_ms, launch_ms, fixed_ms,
           predecessors, occurrences, resource_ids, *_) in payload.get("nodes", ()))
    return PhysicalWorkGraph(
        nodes,
        tuple((str(before), str(after))
              for before, after in payload.get("edges", ())))


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
    from ..unmodeled_gate import assert_report_supported, assert_timing_supported
    from .attribution import _fast_operator_and_graph_reports, attribute_program

    pkg = sys.modules.get(__package__ or "")
    fast_reports_fn = (
        getattr(pkg, "_fast_operator_and_graph_reports", _fast_operator_and_graph_reports)
        if pkg is not None else _fast_operator_and_graph_reports
    )
    attribute_fn = (
        getattr(pkg, "attribute_program", attribute_program)
        if pkg is not None else attribute_program
    )

    assert_timing_supported(parsed, entry="analyze_network_efficiency")
    assert_report_supported(operator_report, entry="analyze_network_efficiency")
    assert_report_supported(graph_report, entry="analyze_network_efficiency")
    evaluation = None
    if operator_report is None:
        from ..evaluation import EvaluationContext
        evaluation = EvaluationContext.create(parsed)
        operator_report, auto_graph = fast_reports_fn(
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
    totals = {"flops": 0.0, "vector_ops": 0.0, "hbm_bytes": 0.0,
              "link_bytes": 0.0, "cube_ms": 0.0, "vec_ms": 0.0,
              "hbm_ms": 0.0, "link_ms": 0.0, "roofline_ms": 0.0}
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
    attribution = attribute_fn(
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
    from ..body import describe_included_components
    return {
        "schema": "mindie-sim.efficiency/1",
        "engine_backend": (graph_report or {}).get("engine_backend"),
        "accel_status": graph_accel,
        "geometry": {"phase": phase, "tokens": tokens, "context": context, "sequences": sequences, "scope": scope},
        "included_components": describe_included_components(parsed, scope=scope),
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
                                           if occurrence_roofline else 0),
        "physical_resource_cut_ms": (max(occurrence_roofline["resource_cut_bounds"].values(),
                                         default=0.0)
                                     if occurrence_roofline else None),
        "attribution": attribution,
        **({"legacy_service_regime_diagnostics": diagnostics}
           if (diagnostics := legacy_service_regime_diagnostics(parsed)) is not None else {}),
        "operator_serial_roofline_efficiency": serial_roof_id / max(operator_sum_ms, 1e-12), "network_cv_overlap": network_cv_overlap,
        "breakdown_by_family": sorted([{"family": f, "recipe": r, **_agg(items)} for (f, r), items in fam_g.items()], key=lambda r: -r["total_ms"]),
        "breakdown_by_bound": {b: _agg(bound_g.get(b, ())) for b in _BOUNDS}, "breakdown_by_kind": {k: _agg(kind_g.get(k, ())) for k in _KINDS}, "operators": analyzed_ops,
        # The compact DAG preserves realized edges in serialized reports.
        # A caller retaining the object graph already has those edges.
        **({"occurrence_dag": _export_occurrence_dag(occurrence_graph)}
           if occurrence_graph is not None and not _retain_graph else {}),
        **({"_occurrence_graph": occurrence_graph} if _retain_graph else {}),
    }


def _operator_fit_signature(op: Mapping[str, Any]) -> str:
    return json.dumps({"op_id": op.get("op_id", ""), "family": op.get("family", ""), "recipe": op.get("recipe", ""), "dtype": op.get("dtype", ""), "local_shape": list(op.get("local_shape") or ()), "flops": round(float(op.get("flops", 0.0)), _WORK_SIGNATURE_PRECISION_DIGITS), "vector_ops": round(float(op.get("vector_ops", 0.0)), _WORK_SIGNATURE_PRECISION_DIGITS), "hbm_bytes": round(float(op.get("hbm_bytes", 0.0)), _WORK_SIGNATURE_PRECISION_DIGITS), "link_bytes": round(float(op.get("link_bytes", 0.0)), _WORK_SIGNATURE_PRECISION_DIGITS)}, sort_keys=True, separators=(",", ":"))


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


def fit_case_roofline_regimes(case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1, context: int = DEFAULT_CONTEXT, sequences: int = 1, scope: str = "body", observed_durations_ms: Mapping[str, float] | None = None, baseline_analysis: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit multi-pipe roofline parameters (fixed_us, effective_mbu/cube_mfu/vec_mfu) per operator."""
    pkg = sys.modules.get(__package__ or "")
    analyze_fn = getattr(pkg, "analyze_network_efficiency", analyze_network_efficiency) if pkg is not None else analyze_network_efficiency
    base = dict(baseline_analysis) if baseline_analysis is not None else analyze_fn(case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, observed_durations_ms=observed_durations_ms)
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


def scale_case_hardware(case: CaseSpec | Mapping[str, Any], *, hbm_bw_scale: float = 1.0, cube_flops_scale: float = 1.0, vector_ops_scale: float = 1.0, link_bw_scale: float = 1.0, hbm_capacity_scale: float = 1.0, baseline_analysis: Mapping[str, Any] | None = None) -> CaseSpec:
    """Scale hardware peaks, explicit/derived rates, service modes, links, and capacity."""
    for name, val in zip(_SCALE_KEYS, (hbm_bw_scale, cube_flops_scale, vector_ops_scale, link_bw_scale, hbm_capacity_scale)):
        _check_pos_float(f"hardware_scale_{name}", val)
    parsed = parse_case(case)
    if (hbm_bw_scale, cube_flops_scale, vector_ops_scale, link_bw_scale, hbm_capacity_scale) == (1.0, 1.0, 1.0, 1.0, 1.0):
        return parsed
    kw = dict(hbm_bw_scale=hbm_bw_scale, cube_flops_scale=cube_flops_scale, vector_ops_scale=vector_ops_scale, link_bw_scale=link_bw_scale)
    mode_to_role = {m_name: m_role for die in parsed.machine.dies for m_name, (m_role, _) in die.service_modes}
    # Serialized dies include machine-wide device defaults and per-die overrides.
    machine_dict = deepcopy(parsed.machine.to_dict())
    for die, dev_dict in zip(parsed.machine.dies, machine_dict["devices"]):
        ins_map = dev_dict.get("compute_arch", {}).get("instruction_services", {})
        svc_role = {sn: (ss["scale_with"]["resource"] if isinstance(ss.get("scale_with"), Mapping) else str(ss.get("resource") or ss.get("owner_axis") or sn)) for sn, ss in ins_map.items()}
        dev_dict["memory_bytes"] = max(1, int(round(die.memory_bytes * hbm_capacity_scale)))
        dev_dict["compute_peaks"] = {dt: float(val) * cube_flops_scale for dt, val in dev_dict.get("compute_peaks", {}).items()}
        dev_dict["rates"] = {r: float(val) * _role_scale_factor(svc_role.get(r, r), **kw) for r, val in dev_dict.get("rates", {}).items()}
        for key in ("hbm_read_bytes_per_s", "hbm_write_bytes_per_s"):
            if dev_dict.get(key) is not None:
                dev_dict[key] = float(dev_dict[key]) * hbm_bw_scale
        for m_name, m_spec in dev_dict.get("service_modes", {}).items():
            if m_spec.get("rate") is not None:
                m_role = "hbm" if m_name in {"mte2_aic_dram", "mte2_aiv_dram"} else str(m_spec.get("resource") or svc_role.get(m_name) or mode_to_role.get(m_name, m_name))
                m_spec["rate"] = float(m_spec["rate"]) * _role_scale_factor(m_role, **kw)
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
