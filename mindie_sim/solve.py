"""One-shot high-level deployment solver for AI Agents and production sizing.

[Role] Expose ``solve_deployment``, a single high-level call that returns:
  (1) HBM Capacity Ledger (weights, workspace, KV/seq, max VRAM concurrency);
  (2) Optimal Deployment recommendation (PD Disaggregated vs Colocated, winning
      P/D parallel tuples, P:D split, KV transfer/pool ms, B*, TTFT, TPOT,
      cluster tok/s, tok/s/die, and top runner-ups);
  (3) 4-Track Software Waterfall (eager -> shipping -> proposed ->
      target -> hardware roofline);
  (4) Top Bottleneck Operator Families & 4D Roofline (Cube MFU, Vec MFU, MBU,
      Link Util, CV Overlap) with explicit ``included_components`` provenance.
[Invariants] Strictly enforces Domain-1 fail-fast on omitted scenario/SLA/SSD
  parameters (omitted ``ttft_slo_ms`` stays explicitly unconstrained rather than
  guessing a numeric TTFT budget), Domain-2 optimal production defaults, and
  Domain-3 joint PD-vs-Colocated search.
[Timeouts & Locks] Bounded in-memory graph/SLA solves; no locks or device work.
[Side-Effect Policy] Pure in-memory evaluation; caller mappings are never mutated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import json
from pathlib import Path
from typing import Any

from mindie_sim.compose import compose_case
from mindie_sim.feature_mechanisms import (
    _network,
    compare_feature_tracks,
)
from mindie_sim.prediction.body_defaults import TargetGraphSet, consume_target_declaration
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    _resolve_speculative_tokens,
    analyze_network_efficiency,
)
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.execution.layer_runs import discover_layer_cohorts
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.offload.policy import missing_scenario_error
from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.sharding.local_tp import rank_zero_graph_reuse
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.scenario import validate_deployment_mode
from mindie_sim.sweep.mapreduce import map_ordered, resolve_granted_workers
from mindie_sim.sweep.pd import (
    _prefill_workspace_tokens,
    _reclaim_sweep_memory,
    _single_die_decode_tps,
    _solve_b_vram_max,
    solve_optimal_deployment,
)

SOLVE_SCHEMA = "mindie-sim.deployment-solution/1"
VALID_SOLVE_MODES = frozenset({"simulation", "modeling"})
BYTES_PER_GIB = float(1024 ** 3)
TOP_RUNNER_UPS = 3
TOP_BOTTLENECK_FAMILIES = 8
GIB_DIGITS = 3
RATIO_DIGITS = 4
MS_DIGITS = 4
MIN_RATIO_EPS = 1e-12


def _resolve_solve_mode(mode: str | None) -> str:
    norm = str(mode or "modeling").strip().lower()
    if norm not in VALID_SOLVE_MODES:
        raise ModelingError(f"invalid_solve_mode:{mode}")
    return norm


def _resolve_kv_pooling_mode(kv_pooling: str | bool | None) -> str | None:
    """Translate high-level ``kv_pooling`` selector into a PD transfer mode."""
    if kv_pooling is None:
        return None
    if isinstance(kv_pooling, bool):
        return "pooled_dram_ssd" if kv_pooling else "layerwise_overlap_d2d"
    norm = str(kv_pooling).strip().lower()
    if norm in {"", "auto"}:
        return None
    if norm in {"off", "none", "false", "disabled", "no"}:
        return "layerwise_overlap_d2d"
    if norm in {"on", "true", "pooled", "pooled_dram_ssd", "dram", "ssd"}:
        return "pooled_dram_ssd"
    if norm in {"direct_blocking_d2d", "layerwise_overlap_d2d"}:
        return norm
    raise ModelingError(f"invalid_kv_pooling_mode:{kv_pooling}")


def _instance_capacity_ledger(
    raw_case: Mapping[str, Any],
    *,
    prompt_tokens: int,
    output_tokens: int,
    role: str,
    b_feasible: int = 1,
) -> dict[str, Any]:
    """Compute one instance's HBM ledger from the single physical admission + receipt."""
    del raw_case, prompt_tokens, output_tokens, role, b_feasible
    raise NotImplementedError(
        "LEGACY_PURGED: deleted shadow scalar capacity division int(cap // kv); "
        "consume integer Memory.admissible_tokens / DieResidentReceipt from prediction.capacity.ledger (S1-017)"
    )


def _build_capacity_ledger(
    solved: Mapping[str, Any], *, prompt_tokens: int, output_tokens: int,
) -> dict[str, Any]:
    winner_mode = solved.get("winner_mode") or solved.get("deployment_mode")
    best = solved.get("best_configuration") or {}
    b_feas = max(1, int(best.get("b_feasible", 1) or 1))
    if winner_mode == "pd_disaggregated" and solved.get("prefill_case") and solved.get("decode_case"):
        p_led = _instance_capacity_ledger(
            solved["prefill_case"], prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, role="pd_prefill", b_feasible=1,
        )
        d_led = _instance_capacity_ledger(
            solved["decode_case"], prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, role="pd_decode", b_feasible=b_feas,
        )
        return {
            **d_led,
            "feasible": bool(p_led["feasible"] and d_led["feasible"]),
            "prefill": p_led,
            "decode": d_led,
        }
    rec_case = solved.get("recommended_case")
    if rec_case is not None:
        eff_role = str(winner_mode) if str(winner_mode) in {"pd_decode", "pd_prefill", "colocated"} else "colocated"
        return _instance_capacity_ledger(
            rec_case, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, role=eff_role, b_feasible=b_feas,
        )
    return {
        "role": "none", "workspace_tokens": 0, "hbm_capacity_gib": 0.0, "weights_gib": 0.0,
        "workspace_gib": 0.0, "kv_per_seq_gib": 0.0, "kv_pool_headroom_gib": 0.0,
        "max_seqs_per_dp_vram": 0, "configured_max_num_seqs": 0, "feasible": False,
    }


def _analyze_case_bounded(
    raw_case: Any,
    *,
    phase: str,
    tokens: int,
    context: int,
    sequences: int,
    scope: str,
) -> dict[str, Any]:
    """Run ``analyze_network_efficiency`` with replica-0 timeline bounding."""
    parsed = parse_case(raw_case)
    graphs = target_graphs(
        parsed, phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope,
    )
    eff_case = consume_target_declaration(parsed, graphs)
    ext_in = {"analysis_input": 0} if scope == "body" else None
    r0_ops = graphs[0]
    has_kg = any("device_kernel_group" in dict(op.metadata) for op in r0_ops)
    use_r0 = rank_zero_graph_reuse(eff_case.runtime.parallel, len(graphs), has_kg)
    sg = (
        TargetGraphSet({0: r0_ops}, kernels=graphs.effective_kernels, selection=graphs.default_selection)
        if isinstance(graphs, TargetGraphSet) else {0: r0_ops}
    ) if use_r0 else graphs
    del parsed, raw_case
    if use_r0:
        del graphs
    _reclaim_sweep_memory()
    from mindie_sim.prediction.execution.layer_runs import execute_native_cohort_graph

    ctx = EvaluationContext.create(eff_case)
    graph_rep, _, _ = execute_native_cohort_graph(
        eff_case, sg, external_inputs=ext_in,
        isolated=False, evaluation=ctx, timeline_replicas=(0,),
    )
    _reclaim_sweep_memory()
    try:
        op_rep = operator_costs(eff_case, r0_ops, replica=0, evaluation=ctx)
    except ValueError:
        cohort_rep = operator_costs(eff_case, sg, evaluation=ctx)
        r0_rows = [row for row in cohort_rep["operators"] if row.get("replica", 0) == 0]
        op_rep = {
            **cohort_rep,
            "operator_occurrences": sum(r["occurrences"] for r in r0_rows),
            "operator_sum_ms": sum(r["total_ms"] for r in r0_rows),
            "operators": r0_rows,
        }
        del cohort_rep
    # Release Engine/Arena/cohort graphs before building occurrence_work_graph.
    if not use_r0:
        del graphs
    del sg, r0_ops, ctx
    _reclaim_sweep_memory()
    try:
        return analyze_network_efficiency(
            eff_case, phase=phase, tokens=tokens, context=context,
            sequences=sequences, scope=scope,
            operator_report=op_rep, graph_report=graph_rep,
            _retain_graph=True,
        )
    finally:
        del graph_rep, op_rep
        _reclaim_sweep_memory()


def _case_execution_fingerprint(raw_case: Mapping[str, Any]) -> str:
    """Canonical execution fingerprint ignoring pure provenance metadata."""
    rt = dict(raw_case.get("runtime") or {})
    meta = dict(rt.get("metadata") or {})
    meta.pop("feature_track_report", None)
    meta.pop("feature_track", None)
    meta.pop("optimal_deployment", None)
    meta.pop("operator_selection_summary", None)
    rt["metadata"] = meta
    payload = {
        "model": raw_case.get("model"),
        "machine": raw_case.get("machine"),
        "runtime": rt,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)


def _track_summary_from_eff(eff: Mapping[str, Any]) -> dict[str, Any]:
    roof_1p0 = max(
        float(eff.get("critical_path_roofline_ms") or 0.0),
        float(eff.get("physical_resource_cut_ms") or 0.0),
    ) or float(eff.get("total_roofline_ideal_ms") or 0.0)
    return {
        "step_ms": float(eff["graph_duration_ms"]),
        "operator_sum_ms": float(eff["operator_sum_ms"]),
        "hbm_bytes": float(eff["total_hbm_bytes"]),
        "network": _network(eff),
        "roof_1p0_ms": roof_1p0,
    }


def _bottleneck_payload_from_eff(eff: Mapping[str, Any]) -> dict[str, Any]:
    top_families = [
        {
            "family": row["family"],
            "recipe": row["recipe"],
            "duration_ms": round(float(row["total_ms"]), MS_DIGITS),
            "share_of_step": round(float(row["share_of_step"]), RATIO_DIGITS),
            "mbu": round(float(row["mbu"]), RATIO_DIGITS),
            "cube_mfu": round(float(row["cube_mfu"]), RATIO_DIGITS),
            "vec_mfu": round(float(row["vec_mfu"]), RATIO_DIGITS),
            "link_util": round(float(row["link_util"]), RATIO_DIGITS),
        }
        for row in eff["breakdown_by_family"][:TOP_BOTTLENECK_FAMILIES]
    ]
    cv_overlap = eff.get("network_cv_overlap")
    return {
        "geometry": eff["geometry"],
        "included_components": eff.get("included_components"),
        "step_ms": round(float(eff["graph_duration_ms"]), MS_DIGITS),
        "roofline": {
            "cube_mfu": round(float(eff["network_cube_mfu"]), RATIO_DIGITS),
            "vec_mfu": round(float(eff["network_vec_mfu"]), RATIO_DIGITS),
            "mbu": round(float(eff["network_mbu"]), RATIO_DIGITS),
            "link_util": round(float(eff["network_link_util"]), RATIO_DIGITS),
            "cv_overlap": (round(float(cv_overlap), RATIO_DIGITS)
                           if cv_overlap is not None else None),
            "roofline_efficiency": round(float(eff["network_roofline_efficiency"]), RATIO_DIGITS),
        },
        "top_families": top_families,
    }


def _analyze_track_worker(task: tuple[Any, ...]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Top-level pickle-safe worker evaluating one feature-track case summary."""
    fp, raw_case, phase, tokens, context, sequences, scope, need_bottleneck = task
    eff = _analyze_case_bounded(
        raw_case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope,
    )
    return (
        fp,
        _track_summary_from_eff(eff),
        _bottleneck_payload_from_eff(eff) if need_bottleneck else None,
    )


def _compare_feature_tracks_bounded(
    model: Any, *, hardware: str | Mapping[str, Any], runtime_stack: str | Mapping[str, Any],
    phase: str, tokens: int, context: int, sequences: int, scope: str,
    compose_kwargs: Mapping[str, Any], use_archetype_folding: bool = False,
    precomputed_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    granted_workers: int | None = None,
    bottleneck_case: Mapping[str, Any] | None = None,
    bottleneck_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the 4-track waterfall with bounded concurrency and replica-0 memory caps."""
    del (
        model, hardware, runtime_stack, phase, tokens, context, sequences, scope,
        compose_kwargs, use_archetype_folding, precomputed_summaries, granted_workers,
        bottleneck_case, bottleneck_out,
    )
    raise NotImplementedError(
        "LEGACY_PURGED: deleted parallel duplicate waterfall builder using expected_ideal & "
        "MONOTONIC_TOLERANCE_MS; wire canonical PredictionSpine + EngineStepReceipt "
        "4-track StackTrack (eager/shipping/proposed/target) (S1-015, S1-016)"
    )


def _extract_bottleneck_roofline(
    raw_case: Mapping[str, Any],
    *,
    prompt_tokens: int,
    sequences: int,
    tokens: int,
    scope: str,
    phase: str = "decode",
    summary_sink: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    eff = _analyze_case_bounded(
        raw_case, phase=phase, tokens=tokens,
        context=(0 if phase == "prefill" else prompt_tokens),
        sequences=max(1, int(sequences)), scope=scope,
    )
    if summary_sink is not None:
        summary_sink[_case_execution_fingerprint(raw_case)] = _track_summary_from_eff(eff)
    return _bottleneck_payload_from_eff(eff)


def _summarize_iso_sla_state(
    solved: Mapping[str, Any], *, track: str, mode_label: str, num_devices: int,
) -> dict[str, Any]:
    best = solved.get("best_configuration") or {}
    par = best.get("decode_parallel") or best.get("parallel") or {}
    tps = float(best.get("sustainable_tps", 0.0) or 0.0)
    tps_per_die = float(best.get("sustainable_tps_per_die", 0.0) or 0.0)
    decode_tps_per_die = float(best.get("decode_tps_per_die", tps_per_die) or 0.0)
    e2e_tps_per_die = float(best.get("e2e_tps_per_die", tps_per_die) or 0.0)
    e2e_total_tps = float(best.get("e2e_total_tps", tps) or 0.0)
    return {
        "track": track,
        "mode_label": mode_label,
        "feasible": bool(solved.get("feasible")),
        "winner_mode": solved.get("winner_mode"),
        "parallel": dict(par),
        "prefill_parallel": dict(best.get("prefill_parallel") or {}) if best.get("prefill_parallel") else None,
        "pd_ratio": best.get("pd_ratio"),
        "b_feasible": int(best.get("b_feasible", 0) or 0),
        "b_vram_max": int(best.get("b_vram_max", 0) or 0),
        "configured_max_num_seqs": int(best.get("configured_max_num_seqs", 0) or 0),
        "mtp_proposal_tokens": int(best.get("mtp_proposal_tokens", 0) or 0),
        "effective_tpot_ms": (
            None if best.get("effective_tpot_ms") is None
            else float(best.get("effective_tpot_ms", 0.0) or 0.0)
        ),
        "decode_step_ms": (
            None if best.get("decode_step_ms") is None
            else float(best.get("decode_step_ms", 0.0) or 0.0)
        ),
        "sustainable_tps": tps,
        "sustainable_tps_per_die": tps_per_die,
        "decode_tps_per_die": decode_tps_per_die,
        "e2e_tps_per_die": e2e_tps_per_die,
        "e2e_total_tps": e2e_total_tps,
        "cluster_tps_per_purchased_die": e2e_total_tps / float(max(1, int(num_devices))),
        "n_tok_per_die": float(best.get("n_tok_per_die", 0.0) or 0.0),
        "us_per_tok_die": float(best.get("us_per_tok_die", 0.0) or 0.0),
        "operating_regime": str(best.get("operating_regime") or ""),
        "limiting_constraint": str(best.get("limiting_constraint") or ""),
        "free_hbm_headroom_gib": float(best.get("free_hbm_headroom_gib", 0.0) or 0.0),
    }


def _bundle_inputs_for_solution(
    solved: Mapping[str, Any],
    *,
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any],
    runtime_stack: str | Mapping[str, Any],
    base_kw: Mapping[str, Any],
    feature_track: str,
    prompt_tokens: int,
    output_tokens: int,
    scope: str,
    phase: str | None = None,
) -> tuple[dict[str, Any], StepRequest, dict[str, Any]]:
    del model, hardware, runtime_stack, base_kw, feature_track
    winner_mode = str(solved.get("winner_mode") or solved.get("deployment_mode") or "")
    is_prefill_only = (
        phase == "prefill" if phase is not None
        else (winner_mode == "pd_prefill" or (winner_mode == "colocated" and int(output_tokens) <= 0))
    )
    best = solved.get("best_configuration") or {}
    rec_case = (
        solved.get("prefill_case") or solved.get("recommended_case")
        if is_prefill_only
        else solved.get("decode_case") or solved.get("recommended_case")
    )
    if rec_case is None:
        raise ModelingError(
            "cannot_export_infeasible_deployment:no_feasible_winner_case_available"
        )
    seq = max(1, int(best.get("b_feasible", 1) or 1))
    mtp_n = int(best.get("mtp_proposal_tokens", 0) or 0)
    if is_prefill_only:
        req = StepRequest(
            phase="prefill", tokens=max(1, int(prompt_tokens)), context=0,
            sequences=1, scope=scope,
        )
    else:
        req = StepRequest(
            phase="decode", tokens=1 + mtp_n, context=prompt_tokens,
            sequences=seq, scope=scope,
        )
    bundle_meta: dict[str, Any] = {
        "handoff_tpot_ms": (
            0.0 if is_prefill_only else float(best.get("handoff_tpot_ms", 0.0) or 0.0)
        ),
        "deployment_total_dies": int(best.get("used_dies", 0) or 0),
        "pd_pool_summary": (
            dict(solved["pd_pool_summary"])
            if isinstance(solved.get("pd_pool_summary"), Mapping) else None
        ),
    }
    return dict(rec_case), req, bundle_meta


def _export_solution_artifacts(
    *,
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any],
    runtime_stack: str | Mapping[str, Any],
    wf_kw: Mapping[str, Any],
    before_solved: Mapping[str, Any],
    after_solved: Mapping[str, Any],
    prompt_tokens: int,
    output_tokens: int,
    scope: str,
    xlsx_path: str | Path | None,
    trace_path: str | Path | None,
    compare_track: str,
    target_track: str,
    mode: str,
    rewrite_plan: Any = None,
) -> dict[str, Any]:
    """Lazily delegate ``--xlsx`` / ``--trace`` emission with independent Before/After (Parallel, B)."""
    del (
        model, hardware, runtime_stack, wf_kw, before_solved, after_solved,
        prompt_tokens, output_tokens, scope, xlsx_path, trace_path,
        compare_track, target_track, mode, rewrite_plan,
    )
    raise NotImplementedError(
        "LEGACY_PURGED: deleted inverted bidirectional compare_source calls; wire explicit "
        "(before_bundle, after_bundle, comparison_request) + 3-arg assert_comparison_polarity (S1-014)"
    )


def _evaluate_winner_analysis(
    *,
    solved: Mapping[str, Any],
    before_solved: Mapping[str, Any],
    after_solved: Mapping[str, Any],
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any],
    runtime_stack: str | Mapping[str, Any],
    prompt: int,
    output: int,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float | None,
    layers: int | None,
    quant: str | Mapping[str, Any] | None,
    kv_cache_dtype: str | None,
    indexer_kv_dtype: str | None,
    topology_preset: str | None,
    ranks_per_node: int | None,
    mtp_accept: float | None,
    ablate_prs: bool,
    xlsx_path: str | Path | None,
    trace_path: str | Path | None,
    compare_track: str,
    target_track: str,
    mode: str,
    rewrite_plan: Any = None,
    granted_workers: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Compute bottleneck roofline, 4-track waterfall, and optional exports for winner."""
    rec_case = solved.get("recommended_case")
    if rec_case is None:
        return None, None, None
    best = solved.get("best_configuration")
    parsed_rec = parse_case(rec_case)
    win_mtp = int((best or {}).get("mtp_proposal_tokens", 0) or 0)
    winner_mode = str(solved.get("winner_mode") or solved.get("deployment_mode") or "colocated")
    is_prefill_only = winner_mode == "pd_prefill" or (winner_mode == "colocated" and int(output) <= 0)
    win_phase = "prefill" if is_prefill_only else "decode"
    win_seq = 1 if is_prefill_only else max(1, int((best or {}).get("b_feasible", 1) or 1))
    win_tokens = max(1, int(prompt)) if is_prefill_only else (1 + win_mtp)
    win_context = 0 if is_prefill_only else prompt
    win_scope = "model"
    win_role = (
        ("pd_prefill" if is_prefill_only else "pd_decode")
        if winner_mode == "pd_disaggregated"
        else (winner_mode if winner_mode in {"pd_decode", "pd_prefill"} else "colocated")
    )
    cohort_plan = discover_layer_cohorts(parsed_rec)
    par = parsed_rec.runtime.parallel
    full_model_layers = int(parsed_rec.model.layers)
    max_seqs = int(parsed_rec.runtime.max_num_seqs)
    max_batched = parsed_rec.runtime.max_num_batched_tokens
    rec_accept = float(parsed_rec.runtime.mtp_acceptance_length)
    local_tp_axes = {
        axis: int(getattr(par, axis))
        for axis in (
            "q_proj_tp", "o_proj_tp", "linear_proj_tp", "dcp_linear_tp",
            "mlp_tp", "shared_expert_tp", "lm_head_tp",
        )
        if int(getattr(par, axis, 1) or 1) > 1
    }
    del parsed_rec
    _reclaim_sweep_memory()
    wf_kw: dict[str, Any] = dict(
        tp=par.tp, dp=par.dp, ep=par.ep, pp=par.pp, pcp=par.pcp, dcp=par.cp,
        **local_tp_axes,
        parallel="fixed", layers=layers, quant=quant,
        kv_cache_dtype=kv_cache_dtype, indexer_kv_dtype=indexer_kv_dtype,
        topology_preset=topology_preset, ranks_per_node=ranks_per_node,
        prompt_tokens=prompt, output_tokens=output,
        ttft_slo_ms=(None if win_role == "pd_decode" else ttft_slo_ms),
        tpot_slo_ms=(None if win_role == "pd_prefill" else tpot_slo_ms),
        max_num_seqs=max_seqs, max_num_batched_tokens=max_batched,
        mtp_proposal_tokens=win_mtp, role=win_role,
        homogeneous_graph_reuse="auto",
    )
    if win_mtp > 0:
        wf_kw["mtp_acceptance_length"] = float(mtp_accept) if mtp_accept is not None else rec_accept
    if ablate_prs:
        bottleneck = _extract_bottleneck_roofline(
            rec_case, prompt_tokens=prompt, sequences=win_seq,
            tokens=win_tokens, scope=win_scope, phase=win_phase,
        )
        waterfall = compare_feature_tracks(
            model, hardware=hardware, runtime_stack=runtime_stack,
            phase=win_phase, tokens=win_tokens, context=win_context,
            sequences=win_seq, scope=win_scope, ablate_prs=True,
            **wf_kw,
        )
    else:
        bottleneck = {}
        waterfall = _compare_feature_tracks_bounded(
            model, hardware=hardware, runtime_stack=runtime_stack,
            phase=win_phase, tokens=win_tokens, context=win_context,
            sequences=win_seq, scope=win_scope, compose_kwargs=wf_kw,
            granted_workers=granted_workers,
            bottleneck_case=rec_case, bottleneck_out=bottleneck,
        )
    if isinstance(bottleneck, dict) and bottleneck:
        inc = dict(bottleneck.get("included_components") or {})
        inc["target_layers"] = full_model_layers
        inc["structural_cohorts"] = len(cohort_plan.cohorts)
        bottleneck["included_components"] = inc
    if isinstance(waterfall, dict):
        waterfall["structural_cohorts"] = len(cohort_plan.cohorts)
        waterfall["full_model_layers"] = full_model_layers
    export_kw = {**wf_kw, "layers": layers}
    exports = (
        _export_solution_artifacts(
            model=model, hardware=hardware, runtime_stack=runtime_stack,
            wf_kw=export_kw, before_solved=before_solved, after_solved=after_solved,
            prompt_tokens=prompt, output_tokens=output, scope=win_scope,
            xlsx_path=xlsx_path, trace_path=trace_path,
            compare_track=compare_track, target_track=target_track,
            mode=mode, rewrite_plan=rewrite_plan,
        )
        if (xlsx_path is not None or trace_path is not None) else None
    )
    return waterfall, bottleneck, exports


def _build_iso_sla_comparison(
    before_solved: Mapping[str, Any],
    after_solved: Mapping[str, Any],
    *,
    solve_mode: str,
    before_track: str,
    target_track: str,
    num_devices: int,
    has_target_after: bool = False,
) -> dict[str, Any]:
    """Assemble the Iso-SLA Before vs After comparison summary."""
    before_summary = _summarize_iso_sla_state(
        before_solved, track=before_track,
        mode_label="Before [当前栈仿真基线]", num_devices=num_devices,
    )
    after_mode_label = (
        "After [卡SLA目标建模终态]" if solve_mode == "modeling"
        else "After [改写计划仿真]" if has_target_after
        else "Simulation [仿真基线]"
    )
    after_summary = _summarize_iso_sla_state(
        after_solved, track=target_track if solve_mode == "modeling" else before_track,
        mode_label=after_mode_label,
        num_devices=num_devices,
    )
    b_tps = before_summary["sustainable_tps_per_die"]
    a_tps = after_summary["sustainable_tps_per_die"]
    return {
        "mode": solve_mode,
        "allow_unbound_formula": (solve_mode == "modeling"),
        "before": before_summary,
        "after": after_summary,
        "parallel_changed": before_summary["parallel"] != after_summary["parallel"],
        "batch_changed": before_summary["b_feasible"] != after_summary["b_feasible"],
        "tps_per_die_gain_x": round(a_tps / b_tps, RATIO_DIGITS) if b_tps > MIN_RATIO_EPS else None,
    }


def _infeasible_candidate_summary(solved: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "mode": row.get("deployment_mode"),
            "parallel": row.get("parallel") or row.get("decode_parallel"),
            "reason": row.get("infeasible_reason"),
            "effective_tpot_ms": row.get("effective_tpot_ms"),
            "b_vram_max": row.get("b_vram_max"),
        }
        for row in solved.get("candidates", ())[:TOP_BOTTLENECK_FAMILIES]
    ]


def _validate_solve_deployment_inputs(
    *,
    mode: str,
    deployment: str | None,
    deployment_mode: str | None,
    num_devices: int | None,
    prompt_tokens: int | None,
    output_tokens: int | None,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float | None,
    pool_ssd_ratio: float,
    ssd_bw_gbps: float | None,
) -> tuple[str, str, float | None]:
    """Validate explicit deployment mode, SLA boundaries, and Domain-1 inputs."""
    solve_mode = _resolve_solve_mode(mode)
    if deployment_mode is not None and deployment is not None:
        if str(deployment_mode).strip().lower() != str(deployment).strip().lower():
            raise ModelingError(f"conflicting_deployment_mode:{deployment_mode}!={deployment}")
    dep_norm = validate_deployment_mode(deployment_mode if deployment_mode is not None else deployment)
    if dep_norm == "pd_decode" and ttft_slo_ms is not None:
        raise ModelingError(
            "pd_decode_ttft_unavailable:ttft_slo_ms must be None when deployment_mode='pd_decode'"
        )
    if dep_norm == "pd_prefill" and tpot_slo_ms is not None:
        raise ModelingError(
            "pd_prefill_rejects_decode_sla:tpot_slo_ms must be None when deployment_mode='pd_prefill'"
        )
    tpot_required = dep_norm != "pd_prefill"
    missing = [
        name for name, val in (
            ("num_devices", num_devices), ("prompt_tokens", prompt_tokens),
            ("output_tokens", output_tokens),
            *([] if not tpot_required else [("tpot_slo_ms", tpot_slo_ms)]),
        ) if val is None
    ]
    if float(pool_ssd_ratio) > 0.0 and ssd_bw_gbps is None:
        missing.append("ssd_bw_gbps")
    if missing:
        raise missing_scenario_error(missing)
    eff_tpot = None if dep_norm == "pd_prefill" else float(tpot_slo_ms)
    return solve_mode, dep_norm, eff_tpot


def solve_deployment(
    model: str | Mapping[str, Any], hardware: str | Mapping[str, Any] = "a3", *,
    num_devices: int | None = None, prompt_tokens: int | None = None,
    output_tokens: int | None = None, tpot_slo_ms: float | None = None,
    ttft_slo_ms: float | None = None, mtp_tokens: int | None = None,
    mtp_accept: float | None = None, kv_pooling: str | bool | None = "auto",
    deployment: str | None = None, deployment_mode: str | None = None,
    quant: str | Mapping[str, Any] | None = None,
    kv_cache_dtype: str | None = None, indexer_kv_dtype: str | None = None,
    topology_preset: str | None = None, ranks_per_node: int | None = None,
    layers: int | None = None, max_sequences_per_dp: int | None = None,
    ablate_prs: bool = False, pool_dram_ratio: float | None = None,
    pool_ssd_ratio: float = 0.0, pool_prefix_hit_ratio: float = 0.0,
    ddr_bw_gbps: float | None = None, ssd_bw_gbps: float | None = None,
    kv_transfer_bw_gbps: float | None = None,
    pinned_parallel: Mapping[str, int] | None = None,
    pinned_prefill_parallel: Mapping[str, int] | None = None,
    pinned_decode_parallel: Mapping[str, int] | None = None,
    raw_eager: bool = False, runtime_stack: str | Mapping[str, Any] | None = None,
    optimizations: Sequence[str] = (), exclude_optimizations: Sequence[str] = (),
    xlsx_path: str | Path | None = None, trace_path: str | Path | None = None,
    compare_track: str = "shipping", target_track: str = "target",
    mode: str = "modeling", rewrite_plan: Any = None,
    granted_workers: int | None = None,
) -> dict[str, Any]:
    """One-shot deployment solver requiring explicit ``deployment_mode``."""
    eff_workers = resolve_granted_workers(granted_workers)
    solve_mode, dep_norm, eff_tpot = _validate_solve_deployment_inputs(
        mode=mode, deployment=deployment, deployment_mode=deployment_mode,
        num_devices=num_devices, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
        pool_ssd_ratio=pool_ssd_ratio, ssd_bw_gbps=ssd_bw_gbps,
    )
    pd_mode = _resolve_kv_pooling_mode(kv_pooling) if dep_norm == "pd_disaggregated" else None
    shared_kw: dict[str, Any] = dict(
        num_devices=int(num_devices), prompt_tokens=int(prompt_tokens),
        output_tokens=int(output_tokens), ttft_sla_ms=ttft_slo_ms, tpot_sla_ms=eff_tpot,
        deployment_mode=dep_norm, pd_transfer_mode=pd_mode, kv_transfer_bw_gbps=kv_transfer_bw_gbps,
        pool_dram_ratio=pool_dram_ratio, pool_ssd_ratio=pool_ssd_ratio,
        pool_prefix_hit_ratio=pool_prefix_hit_ratio, ddr_bw_gbps=ddr_bw_gbps,
        ssd_bw_gbps=ssd_bw_gbps, pinned_parallel=pinned_parallel,
        pinned_prefill_parallel=pinned_prefill_parallel, pinned_decode_parallel=pinned_decode_parallel,
        mtp_proposal_tokens=mtp_tokens, mtp_acceptance_length=mtp_accept,
        max_sequences_per_dp=max_sequences_per_dp, quant=quant, layers=layers,
        ranks_per_node=ranks_per_node, topology_preset=topology_preset, raw_eager=raw_eager,
        runtime_stack=runtime_stack, optimizations=optimizations,
        exclude_optimizations=exclude_optimizations, kv_cache_dtype=kv_cache_dtype,
        indexer_kv_dtype=indexer_kv_dtype, allow_unconstrained_ttft=(ttft_slo_ms is None),
        granted_workers=eff_workers,
    )
    before_track = str(compare_track)
    before_solved = solve_optimal_deployment(
        model, hardware, feature_track=before_track,
        apply_target_envelope=False, **shared_kw,
    )
    _reclaim_sweep_memory()
    has_target_after = (solve_mode == "modeling") or (rewrite_plan is not None)
    if has_target_after:
        after_track = str(target_track) if solve_mode == "modeling" else before_track
        after_solved = solve_optimal_deployment(
            model, hardware, feature_track=after_track,
            apply_target_envelope=True,
            rewrite_plan=rewrite_plan,
            allow_unbound_formula=(solve_mode == "modeling"),
            **shared_kw,
        )
        _reclaim_sweep_memory()
    else:
        after_solved = before_solved
    solved = after_solved if has_target_after else before_solved
    prompt, output = int(prompt_tokens), int(output_tokens)
    ledger = _build_capacity_ledger(solved, prompt_tokens=prompt, output_tokens=output)

    best = solved.get("best_configuration")
    if dep_norm == "pd_prefill":
        feasible_sorted = sorted(
            (row for row in solved.get("candidates", ()) if row.get("feasible")),
            key=lambda row: (
                -float(row.get("prefill_rps_per_die", 0.0) or 0.0),
                int(row.get("used_dies", 0)),
                float(row.get("ttft_ms", 0.0) or 0.0),
            ),
        )
    else:
        feasible_sorted = sorted(
            (row for row in solved.get("candidates", ()) if row.get("feasible")),
            key=lambda row: (
                -float(_single_die_decode_tps(row)),
                int(row.get("used_dies", 0)),
                -float(row.get("e2e_tps_per_die", row.get("sustainable_tps_per_die", 0.0)) or 0.0),
                float(row.get("effective_tpot_ms", 0.0) or 0.0),
            ),
        )
    runner_ups = feasible_sorted[1:1 + TOP_RUNNER_UPS]
    waterfall, bottleneck, exports = _evaluate_winner_analysis(
        solved=solved, before_solved=before_solved, after_solved=after_solved,
        model=model, hardware=hardware, runtime_stack=runtime_stack,
        prompt=prompt, output=output, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=eff_tpot,
        layers=layers, quant=quant, kv_cache_dtype=kv_cache_dtype,
        indexer_kv_dtype=indexer_kv_dtype, topology_preset=topology_preset,
        ranks_per_node=ranks_per_node, mtp_accept=mtp_accept, ablate_prs=ablate_prs,
        xlsx_path=xlsx_path, trace_path=trace_path,
        compare_track=before_track, target_track=str(target_track),
        mode=solve_mode, rewrite_plan=rewrite_plan,
        granted_workers=eff_workers,
    )
    iso_sla = _build_iso_sla_comparison(
        before_solved, after_solved, solve_mode=solve_mode,
        before_track=before_track, target_track=str(target_track),
        num_devices=int(num_devices), has_target_after=has_target_after,
    )
    rec_dep: dict[str, Any] = {
        "deployment_mode": dep_norm,
        "winner_mode": solved["winner_mode"], "feasible": bool(solved["feasible"]),
        "deployment_outcome": solved.get("deployment_outcome"),
        "search_coverage": dict(solved.get("search_coverage") or {}),
        "best_configuration": best, "best_pd": solved.get("best_pd"),
        "best_colocated": solved.get("best_colocated"), "runner_ups": runner_ups,
    }
    if not solved["feasible"]:
        rec_dep["infeasible_summary"] = _infeasible_candidate_summary(solved)
    solution: dict[str, Any] = {
        "schema": SOLVE_SCHEMA, "mode": solve_mode, "allow_unbound_formula": (solve_mode == "modeling"),
        "deployment_mode": dep_norm, "deployment_outcome": solved.get("deployment_outcome"),
        "model": solved["model"], "hardware": solved["hardware"], "num_devices": solved["num_devices"],
        "scenario": {
            "prompt_tokens": prompt, "output_tokens": output,
            "ttft_slo_ms": ttft_slo_ms, "tpot_slo_ms": eff_tpot,
            "deployment_mode": dep_norm, "kv_pooling": kv_pooling,
        },
        "feasible": bool(solved["feasible"]), "capacity_ledger": ledger,
        "recommended_deployment": rec_dep, "iso_sla_comparison": iso_sla,
        "four_track_waterfall": waterfall, "bottleneck_roofline": bottleneck,
    }
    if exports is not None:
        solution["exports"] = exports
    return solution


__all__ = ["SOLVE_SCHEMA", "VALID_SOLVE_MODES", "solve_deployment"]
