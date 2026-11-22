"""Graph execution attribution and fixed-plan counterfactuals.

[Role] Lower and execute target graphs, attribute resource work and compare
  hardware counterfactuals on a shared plan.
[Invariants] Work, service, occupancy and graph span retain distinct boundaries.
  Hardware interventions preserve the prepared plan's declared work.
[Timeouts & Locks] Uses the simulator's evaluation context and active deadline
  with no additional locks.
[Side-Effect Policy] Input cases and plans are read-only. Evaluation state is
  local to a graph run or its explicitly supplied evaluation context.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.model import RequestSpec

from ..body_defaults import TargetGraphSet, consume_target_declaration
from ..graph_analysis import graph_costs
from ..operator_analysis import operator_costs, target_graphs
from ..efficiency.resource_attribution import attribute_stage_dag
from ..scheduler import DraftPolicy
from ..structure import parse_case
from .analysis import analyze_network_efficiency, occurrence_work_graph, scale_case_hardware
from .capacity_window import (
    HBM_BYTES_PRECISION_DIGITS, HBM_UTILIZATION_PRECISION_DIGITS,
    _resolve_speculative_tokens, _vram_max_sequences_per_dp, capacity_window,
)
from .peaks import (
    DEFAULT_CONTEXT, DEFAULT_OUTPUT_BUDGET, _SCALE_KEYS, _check_pos_float,
    _validate_geometry,
)


_CAPACITY_HEADROOM_TOLERANCE_BYTES = 1e-6
_COUNTERFACTUAL_CHANGED_OPERATOR_LIMIT = 12
_RESOURCE_SCALE_ALIASES = {
    "hbm": "hbm_bw_scale", "hbm_bw": "hbm_bw_scale", "hbm_bandwidth": "hbm_bw_scale",
    "hbm_bw_scale": "hbm_bw_scale",
    "cube": "cube_flops_scale", "cube_flops": "cube_flops_scale", "cube_flops_scale": "cube_flops_scale",
    "vector": "vector_ops_scale", "vector_ops": "vector_ops_scale", "vector_ops_scale": "vector_ops_scale",
    "link": "link_bw_scale", "link_bw": "link_bw_scale", "link_bw_scale": "link_bw_scale",
    "hbm_capacity": "hbm_capacity_scale", "capacity": "hbm_capacity_scale", "hbm_capacity_scale": "hbm_capacity_scale",
}


def _pkg_attr(name: str, default: Any) -> Any:
    pkg = sys.modules.get(__package__ or "")
    return getattr(pkg, name, default) if pkg is not None else default


def _lower_and_run_graph(case: CaseSpec, *, phase: str, tokens: int, context: int, sequences: int, scope: str, run_graph: bool = True, lowered_graphs: Any | None = None, include_submission: bool = False, overlap_async_scheduler: bool = False, evaluation_context: Any | None = None, fast_span_only: bool = False) -> tuple[Any, CaseSpec, dict[str, Any] | None]:
    graphs = lowered_graphs if lowered_graphs is not None else _pkg_attr("target_graphs", target_graphs)(case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope)
    eff_case, has_kg = consume_target_declaration(case, graphs), any("device_kernel_group" in dict(op.metadata) for op in graphs[0])
    if not run_graph:
        # An explicit run_graph=False is a caller control: a selected MC2 or
        # Local-TP kernel group must not turn it into a full cohort DES run.
        return graphs, eff_case, None
    ext_in = {"analysis_input": 0} if scope == "body" else None
    from ..sharding.local_tp import rank_zero_graph_reuse
    sg = (TargetGraphSet({0: graphs[0]}, kernels=graphs.effective_kernels,
                         selection=graphs.default_selection)
          if isinstance(graphs, TargetGraphSet) else {0: graphs[0]}) \
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
    from ..evaluation import EvaluationContext
    from ..sharding.local_tp import local_tp_spans_dp, rank_zero_graph_reuse
    if precomputed_graph is None:
        ctx = evaluation_context or EvaluationContext.create(case)
        graphs, eff_case, graph_rep = _pkg_attr("_lower_and_run_graph", _lower_and_run_graph)(
            case, phase=phase, tokens=tokens, context=context,
            sequences=sequences, scope=scope, run_graph=run_graph,
            evaluation_context=ctx)
    else:
        graphs, eff_case, graph_rep = precomputed_graph
        ctx = evaluation_context or EvaluationContext.create(eff_case)
    r0_ops = graphs[0]
    parallel = eff_case.runtime.parallel
    needs_cohort = bool(
        parallel.ep_spans_dp or local_tp_spans_dp(parallel) or parallel.cp_spans_dp
    )
    has_kg = any("device_kernel_group" in dict(op.metadata) for op in r0_ops)
    if needs_cohort and not rank_zero_graph_reuse(parallel, len(graphs), has_kg):
        selection = graphs if isinstance(graphs, Mapping) else {0: graphs}
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
        op_rep = operator_costs(eff_case, r0_ops, replica=0, evaluation=ctx)
    return op_rep, graph_rep


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
        "active_share_of_span": active_ms / span if timeline and span > 0 else None,
        "unaccounted_active_gap_ms": max(0.0, span - active_ms) if timeline else None,
        "busy_ms": None,
        "busy_basis": "no_whole_graph_service_interval_union_available_per_resource_busy_kept",
        "service_busy_union_ms": None,
        "wait_ms": None,
        "wait_basis": "service_start_end_or_pmu_required_not_inferable_from_ready_complete",
        "span_ms": span,
        "complete_ms": float(graph_report.get("complete_ms", span) or 0.0),
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


def prepare_fixed_plan(case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1, context: int = DEFAULT_CONTEXT, sequences: int = 1, scope: str = "body") -> dict[str, Any]:
    """Lower a target program once so baseline and counterfactual share one plan."""
    _validate_geometry(phase, tokens, context, sequences, scope)
    parsed = parse_case(case)
    graphs = _pkg_attr("target_graphs", target_graphs)(parsed, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope)
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


def capacity_feedback(case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1, context: int = DEFAULT_CONTEXT, sequences: int = 1, scope: str = "body", output_budget: int = DEFAULT_OUTPUT_BUDGET, emitted_tokens_per_seq: float | None = None) -> dict[str, Any]:
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
                "available_bytes": round(float(row.get("supply_bytes", 0.0) or 0.0), HBM_BYTES_PRECISION_DIGITS),
                "kv_bytes_per_sequence": round(active / count, HBM_BYTES_PRECISION_DIGITS),
                "headroom_bytes": round(float(row.get("active_headroom_bytes", 0.0) or 0.0), HBM_BYTES_PRECISION_DIGITS),
                "used_bytes": round(float(row.get("fixed_bytes", 0.0) or 0.0) + external + active, HBM_BYTES_PRECISION_DIGITS),
                "hbm_bytes_per_die": round(float(row.get("hbm_bytes_per_die", 0.0) or 0.0), HBM_BYTES_PRECISION_DIGITS),
                "hbm_utilization_ratio": round(float(row.get("hbm_utilization_ratio", 0.0) or 0.0), HBM_UTILIZATION_PRECISION_DIGITS),
                "workspace_bytes": round(float(row.get("workspace_bytes", 0.0) or 0.0), HBM_BYTES_PRECISION_DIGITS),
                "external_held_bytes": round(external, HBM_BYTES_PRECISION_DIGITS),
                "active_kv_bytes": round(active, HBM_BYTES_PRECISION_DIGITS),
                "retiring_kv_bytes": round(float(row.get("retiring_kv_bytes", 0.0) or 0.0), HBM_BYTES_PRECISION_DIGITS),
                "peak_bytes": round(peak, HBM_BYTES_PRECISION_DIGITS),
                "full_headroom_bytes": round(float(row.get("headroom_bytes", 0.0) or 0.0), HBM_BYTES_PRECISION_DIGITS),
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
        "available_kv_capacity_bytes_per_die": round(min((float(row["supply_bytes"]) for row in current["stages"]), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "hbm_bytes_per_die": round(min((float(row["hbm_bytes_per_die"]) for row in current["stages"]), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "hbm_bytes_per_stage": [round(value, HBM_BYTES_PRECISION_DIGITS) for value in hbm_by_stage],
        "weight_bytes_per_die": round(float(current.get("weight_bytes_per_die", 0.0) or 0.0), HBM_BYTES_PRECISION_DIGITS),
        "workspace_bytes": round(float(current.get("workspace_bytes", 0.0) or 0.0), HBM_BYTES_PRECISION_DIGITS),
        "kv_bytes_per_sequence": round(max((float(row["active_kv_bytes"]) / max(1, int(current["sequences"])) for row in current["stages"]), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "capacity_headroom_bytes": round(active_headroom, HBM_BYTES_PRECISION_DIGITS),
        "capacity_headroom_basis": "active_window_kv_only_after_workspace_and_fixed_reserve",
        "hbm_utilization_ratio": round(max((float(row["hbm_utilization_ratio"]) for row in current["stages"]), default=0.0), HBM_UTILIZATION_PRECISION_DIGITS),
        "b_vram_max_sync": int(b_sync), "b_vram_max_async": int(b_async), "b_vram_max_configured": int(b_configured),
        "b_vram_max_lifecycle_sync": int(life_sync), "b_vram_max_lifecycle_async": int(life_async),
        "b_vram_max_joint_sync": int(joint_sync), "b_vram_max_joint_async": int(joint_async),
        "capacity_by_stage": stages, "capacity_by_stage_lifecycle": life_stages,
        "lifecycle_kv_bytes_per_sequence": round(max((float(row["active_kv_bytes"]) / max(1, int(lifecycle["sequences"])) for row in lifecycle["stages"]), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "lifecycle_capacity_headroom_bytes": round(lifecycle_headroom, HBM_BYTES_PRECISION_DIGITS),
        "joint_capacity_headroom_bytes": round(joint_headroom, HBM_BYTES_PRECISION_DIGITS),
        "lifecycle_capacity_window": "complete_output" if phase == "prefill" else "decode_step",
        "capacity_window": current,
        "lifecycle_capacity_window_facts": lifecycle,
        "is_vram_feasible": bool(
            b_configured >= sequences and active_headroom >= -_CAPACITY_HEADROOM_TOLERANCE_BYTES),
        "is_lifecycle_feasible": bool(
            life_configured >= sequences and lifecycle_headroom >= -_CAPACITY_HEADROOM_TOLERANCE_BYTES),
        "is_joint_feasible": bool(
            joint_configured >= sequences and joint_headroom >= -_CAPACITY_HEADROOM_TOLERANCE_BYTES),
    }


def evaluate_fixed_plan(case: CaseSpec | Mapping[str, Any], *, plan: Mapping[str, Any] | None = None, phase: str = "decode", tokens: int = 1, context: int = DEFAULT_CONTEXT, sequences: int = 1, scope: str = "body",
                        hbm_bw_scale: float = 1.0, cube_flops_scale: float = 1.0, vector_ops_scale: float = 1.0, link_bw_scale: float = 1.0, hbm_capacity_scale: float = 1.0,
                        include_submission: bool = False, overlap_async_scheduler: bool = False,
                        include_capacity: bool = False, output_budget: int = DEFAULT_OUTPUT_BUDGET, emitted_tokens_per_seq: float | None = None) -> dict[str, Any]:
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
    graphs, eff_case, report = _pkg_attr("_lower_and_run_graph", _lower_and_run_graph)(scaled, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, run_graph=True, lowered_graphs=plan["graphs"], include_submission=include_submission, overlap_async_scheduler=overlap_async_scheduler)
    analysis = _pkg_attr("analyze_network_efficiency", analyze_network_efficiency)(scaled, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, _precomputed_graph=(graphs, eff_case, report))
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
    totals = {"total_flops": 0.0, "total_vector_ops": 0.0,
              "total_hbm_bytes": 0.0, "total_link_bytes": 0.0}
    for op in items:
        totals["total_flops"] += float(getattr(op, "flops", 0.0) or 0.0)
        totals["total_vector_ops"] += float(getattr(op, "vector_ops", 0.0) or 0.0)
        totals["total_hbm_bytes"] += float(getattr(op, "hbm_bytes", 0.0) or 0.0)
        totals["total_link_bytes"] += float(getattr(op, "link_bytes", 0.0) or 0.0)
    return totals


def evaluate_resource_counterfactual(case: CaseSpec | Mapping[str, Any], *, resource: str, factor: float, phase: str = "decode", tokens: int = 1, context: int = DEFAULT_CONTEXT, sequences: int = 1, scope: str = "body",
                                     plan: Mapping[str, Any] | None = None, include_submission: bool = False, overlap_async_scheduler: bool = False,
                                     include_capacity: bool = True, output_budget: int = DEFAULT_OUTPUT_BUDGET, emitted_tokens_per_seq: float | None = None) -> dict[str, Any]:
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
                    "delta_ratio": 0.0}
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
        "changed_operators": changed_ops[:_COUNTERFACTUAL_CHANGED_OPERATOR_LIMIT],
        "capacity_feedback": {"baseline": base.get("capacity"), "counterfactual": changed.get("capacity")},
        "evaluation_consistency": {"same_plan": True, "same_common_evaluator": True, "baseline_step_ms": base_step},
    }
