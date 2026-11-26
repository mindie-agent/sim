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
from dataclasses import replace
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
    _compose_track_arg,
    _prefill_prefix_split,
    _prefill_workspace_tokens,
    _reclaim_sweep_memory,
    _single_die_decode_tps,
    _solve_b_vram_max,
    canonical_deployment_row_key,
    solve_optimal_deployment,
)

SOLVE_SCHEMA = "mindie-sim.deployment-solution/1"
WATERFALL_SCHEMA = "mindie-sim.feature-waterfall/1"
VALID_SOLVE_MODES = frozenset({"simulation", "modeling"})
_UNQUANTIZED_PRESET_NAMES = frozenset({"bf16", "fp16", "fp32", "none"})
BYTES_PER_GIB = float(1024 ** 3)
TOP_RUNNER_UPS = 3
TOP_BOTTLENECK_FAMILIES = 8
GIB_DIGITS = 3
RATIO_DIGITS = 4
PCT_DIGITS = 2
PCT_SCALE = 100.0
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
    """Compute one instance's HBM ledger from canonical integer ``Memory`` + ``DieResidentReceipt``."""
    parsed = parse_case(raw_case)
    role_norm = str(role or "colocated").strip().lower()
    prefill_ws = _prefill_workspace_tokens(parsed, int(prompt_tokens))
    if role_norm == "pd_prefill":
        ws_tokens = max(1, prefill_ws)
        seq_len = int(prompt_tokens) + 1
        _, kv_by_stage, max_seqs_vram = _solve_b_vram_max(
            parsed, base_ws=ws_tokens, decode_q=1, seq_len=seq_len, cap_bound=None,
        )
        mem = Memory(parsed, workspace_tokens=ws_tokens)
        phase_name, op_batch, q_tok = "prefill", 1, ws_tokens
    else:
        decode_q, _ = _resolve_speculative_tokens(
            parsed, output_budget=output_tokens,
            tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        q_tok = max(1, int(decode_q))
        base_ws = max(1, prefill_ws) if role_norm == "colocated" else q_tok
        seq_len = int(prompt_tokens) + int(output_tokens)
        _, kv_by_stage, max_seqs_vram = _solve_b_vram_max(
            parsed, base_ws=base_ws, decode_q=q_tok, seq_len=seq_len, cap_bound=None,
        )
        op_batch = max(1, int(b_feasible))
        ws_tokens = max(base_ws, op_batch * q_tok)
        mem = Memory(parsed, workspace_tokens=ws_tokens)
        phase_name = "decode"
    receipt = mem.resident_memory_receipt(
        phase=phase_name, sequence_tokens=seq_len, batch_per_dp=op_batch, query_tokens=q_tok,
    )
    kv_pool_receipt = mem.resident_memory_receipt(
        phase=phase_name, sequence_tokens=None, batch_per_dp=0, query_tokens=q_tok,
    )
    hbm_bytes = min((int(die.memory_bytes) for die in parsed.machine.dies), default=0)
    weights_bytes = max(
        (int(d.checkpoint_weight_bytes) + int(d.prepared_weight_bytes) + int(d.vision_weight_bytes)
         for d in receipt.dies),
        default=0,
    )
    workspace_bytes = max((int(d.workspace_bytes) for d in receipt.dies), default=0)
    kv_per_seq_bytes = max((int(round(v)) for v in kv_by_stage), default=0)
    kv_pool_headroom_bytes = int(kv_pool_receipt.min_free_hbm_headroom_bytes)
    free_headroom_bytes = int(receipt.min_free_hbm_headroom_bytes)
    return {
        "role": role_norm,
        "parallel": parsed.runtime.parallel.to_dict(),
        "workspace_tokens": int(ws_tokens),
        "hbm_capacity_gib": round(float(hbm_bytes) / BYTES_PER_GIB, GIB_DIGITS),
        "weights_gib": round(float(weights_bytes) / BYTES_PER_GIB, GIB_DIGITS),
        "workspace_gib": round(float(workspace_bytes) / BYTES_PER_GIB, GIB_DIGITS),
        "kv_per_seq_gib": round(float(kv_per_seq_bytes) / BYTES_PER_GIB, GIB_DIGITS),
        "kv_pool_headroom_gib": round(float(max(0, kv_pool_headroom_bytes)) / BYTES_PER_GIB, GIB_DIGITS),
        "free_hbm_headroom_gib": round(float(max(0, free_headroom_bytes)) / BYTES_PER_GIB, GIB_DIGITS),
        "max_seqs_per_dp_vram": int(max_seqs_vram),
        "configured_max_num_seqs": int(parsed.runtime.max_num_seqs),
        "feasible": bool(kv_pool_headroom_bytes >= 0 and max_seqs_vram >= 1),
    }


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
    shared_isolated_caches: dict[Any, Any] | None = None,
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
    if shared_isolated_caches is not None:
        rt = eff_case.runtime
        iso_key = (
            rt.collective_startup_mode,
            rt.matrix_compute_policy,
            rt.graph_mode,
            rt.graph_capture_mode,
            bool(rt.enable_torch_compile),
            bool(rt.enable_npugraph_ex),
            bool(rt.enable_static_kernel),
            bool(rt.enable_super_kernel),
            json.dumps(rt.kernels or {}, sort_keys=True, separators=(",", ":"), default=repr),
            json.dumps(rt.submission or {}, sort_keys=True, separators=(",", ":"), default=repr),
            json.dumps(rt.memory or {}, sort_keys=True, separators=(",", ":"), default=repr),
        )
        shared_iso = shared_isolated_caches.get(iso_key)
        if shared_iso is None:
            shared_isolated_caches[iso_key] = ctx.isolated_results
        else:
            ctx.isolated_results = shared_iso
    graph_rep, _, _ = execute_native_cohort_graph(
        eff_case, sg, external_inputs=ext_in,
        isolated=False, evaluation=ctx, timeline_replicas=(0,),
    )
    _reclaim_sweep_memory()
    try:
        op_rep = operator_costs(eff_case, r0_ops, replica=0, evaluation=ctx)
    except ValueError:
        cohort_rep = operator_costs(eff_case, sg, evaluation=ctx, target_replicas=(0,))
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


def _pct_delta(base_ms: float, cand_ms: float) -> float:
    return ((cand_ms - base_ms) / base_ms) * PCT_SCALE if base_ms > MIN_RATIO_EPS else 0.0


def _speedup_ratio(base_ms: float, cand_ms: float) -> float:
    return (base_ms / cand_ms) if cand_ms > MIN_RATIO_EPS else 0.0


def _compose_canonical_waterfall_cases(
    model: Any,
    *,
    hardware: str | Mapping[str, Any],
    runtime_stack: str | Mapping[str, Any],
    phase: str,
    cleaned_kwargs: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compose canonical 4-track cases (``eager``, ``shipping``, ``proposed``, ``target``)."""
    cases: dict[str, dict[str, Any]] = {}
    for track in ("shipping", "proposed", "target"):
        cases[track] = compose_case(
            model, hardware=hardware, runtime_stack=runtime_stack,
            feature_track=_compose_track_arg(track), _stack_phase=phase, **cleaned_kwargs,
        )
    eager_kw = dict(cleaned_kwargs)
    ship_meta = (cases["shipping"].get("runtime") or {}).get("metadata") or {}
    ship_quant = (cases["shipping"].get("model") or {}).get("quant")
    va_prof = ship_meta.get("va_profile") or {}
    va_flags = dict(va_prof.get("tutorial_flags") or {})
    explicit_quant = eager_kw.get("quant")
    unquantized_override = (
        isinstance(explicit_quant, str)
        and explicit_quant.strip().lower() in _UNQUANTIZED_PRESET_NAMES
    )
    if "quant" not in eager_kw:
        if isinstance(ship_quant, Mapping) and ship_quant:
            eager_kw["quant"] = dict(ship_quant)
        elif va_prof.get("quant") is not None:
            eager_kw["quant"] = va_prof["quant"]
    if not unquantized_override:
        if "kv_cache_dtype" not in eager_kw:
            resolved_kv = (
                (ship_quant.get("kv_dtype") if isinstance(ship_quant, Mapping) else None)
                or va_flags.get("kv_cache_dtype")
            )
            if resolved_kv:
                eager_kw["kv_cache_dtype"] = str(resolved_kv)
        if "indexer_kv_dtype" not in eager_kw:
            resolved_idx = (
                (ship_quant.get("index_dtype") if isinstance(ship_quant, Mapping) else None)
                or va_flags.get("indexer_kv_dtype")
            )
            if resolved_idx:
                eager_kw["indexer_kv_dtype"] = str(resolved_idx)
    cases["eager"] = compose_case(
        model, hardware=hardware, runtime_stack=runtime_stack,
        feature_track=_compose_track_arg("eager"), _stack_phase=phase, **eager_kw,
    )
    return cases


def _compare_feature_tracks_bounded(
    model: Any, *, hardware: str | Mapping[str, Any], runtime_stack: str | Mapping[str, Any],
    phase: str, tokens: int, context: int, sequences: int, scope: str,
    compose_kwargs: Mapping[str, Any], use_archetype_folding: bool = False,
    precomputed_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    granted_workers: int | None = None,
    bottleneck_case: Mapping[str, Any] | None = None,
    bottleneck_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the canonical 4-track waterfall (``eager/shipping/proposed/target``) with bounded memory."""
    del use_archetype_folding
    skip_keys = {"feature_track", "optimizations", "exclude_optimizations", "raw_eager", "_stack_phase"}
    cleaned = {k: v for k, v in compose_kwargs.items() if k not in skip_keys}
    eff_ctx = max(0, int(context))
    eff_seq = max(1, int(sequences))
    eff_workers = resolve_granted_workers(granted_workers)

    summary_cache: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in (precomputed_summaries or {}).items()
    }
    cases = _compose_canonical_waterfall_cases(
        model, hardware=hardware, runtime_stack=runtime_stack, phase=phase, cleaned_kwargs=cleaned,
    )
    track_fps = {track: _case_execution_fingerprint(cases[track]) for track in ("eager", "shipping", "proposed", "target")}
    bn_fp = _case_execution_fingerprint(bottleneck_case) if bottleneck_case is not None else None

    uncached_tasks: list[tuple[Any, ...]] = []
    queued_fps: set[str] = set()
    if bottleneck_case is not None and bn_fp is not None:
        if bn_fp not in summary_cache or bottleneck_out is not None:
            queued_fps.add(bn_fp)
            uncached_tasks.append((
                bn_fp, bottleneck_case, phase, int(tokens), eff_ctx, eff_seq, scope, bottleneck_out is not None,
            ))
    for track in ("eager", "shipping", "proposed", "target"):
        fp = track_fps[track]
        if fp in summary_cache or fp in queued_fps:
            continue
        queued_fps.add(fp)
        uncached_tasks.append((fp, cases[track], phase, int(tokens), eff_ctx, eff_seq, scope, False))

    if len(uncached_tasks) > 1 and eff_workers > 1:
        results = map_ordered(_analyze_track_worker, uncached_tasks, granted_workers=eff_workers)
    else:
        shared_iso_caches: dict[Any, Any] = {}
        try:
            results = []
            for fp, raw_c, ph, tok, ctx_n, seq_n, sc, need_bn in uncached_tasks:
                eff = _analyze_case_bounded(
                    raw_c, phase=ph, tokens=tok, context=ctx_n, sequences=seq_n,
                    scope=sc, shared_isolated_caches=shared_iso_caches,
                )
                results.append((
                    fp,
                    _track_summary_from_eff(eff),
                    _bottleneck_payload_from_eff(eff) if need_bn else None,
                ))
        finally:
            for iso_cache in shared_iso_caches.values():
                iso_cache.clear()
            shared_iso_caches.clear()
    for fp, summary, bn_payload in results:
        summary_cache[fp] = summary
        if bn_payload is not None and bottleneck_out is not None:
            bottleneck_out.update(bn_payload)

    summaries = {track: summary_cache[track_fps[track]] for track in ("eager", "shipping", "proposed", "target")}
    reports = {
        track: ((cases[track].get("runtime") or {}).get("metadata") or {}).get("feature_track_report") or {}
        for track in ("eager", "shipping", "proposed", "target")
    }
    eager_ms = float(summaries["eager"]["step_ms"])
    ship_ms = float(summaries["shipping"]["step_ms"])
    prop_ms = float(summaries["proposed"]["step_ms"])
    target_ms = float(summaries["target"]["step_ms"])
    roof_1p0_ms = float(summaries["target"]["roof_1p0_ms"])
    eager_mono = eager_ms >= ship_ms
    monotonic = eager_mono and ship_ms >= prop_ms and prop_ms >= target_ms and target_ms >= roof_1p0_ms

    return {
        "schema": WATERFALL_SCHEMA,
        "model": cases["shipping"]["model"]["name"],
        "hardware_sku": cases["shipping"]["machine"]["sku"],
        "runtime_stack": reports["shipping"].get("stack"),
        "repository": reports["shipping"].get("repository"),
        "geometry": {
            "phase": phase, "tokens": int(tokens), "context": eff_ctx,
            "sequences": eff_seq, "scope": scope,
        },
        "monotonic_verified": bool(monotonic),
        "eager_vs_shipping_monotonic": bool(eager_mono),
        "waterfall": {
            "eager": {
                "step_ms": round(eager_ms, MS_DIGITS),
                "operator_sum_ms": round(float(summaries["eager"]["operator_sum_ms"]), MS_DIGITS),
                "hbm_bytes": float(summaries["eager"]["hbm_bytes"]),
                "network": dict(summaries["eager"]["network"]),
                "active_mechanisms": [m["id"] for m in reports["eager"].get("active_mechanisms", ())],
            },
            "shipping": {
                "step_ms": round(ship_ms, MS_DIGITS),
                "operator_sum_ms": round(float(summaries["shipping"]["operator_sum_ms"]), MS_DIGITS),
                "hbm_bytes": float(summaries["shipping"]["hbm_bytes"]),
                "network": dict(summaries["shipping"]["network"]),
                "delta_vs_eager_ms": round(ship_ms - eager_ms, MS_DIGITS),
                "delta_vs_eager_pct": round(_pct_delta(eager_ms, ship_ms), PCT_DIGITS),
                "speedup_vs_eager": round(_speedup_ratio(eager_ms, ship_ms), RATIO_DIGITS),
                "active_mechanisms": [m["id"] for m in reports["shipping"].get("active_mechanisms", ())],
            },
            "proposed": {
                "step_ms": round(prop_ms, MS_DIGITS),
                "operator_sum_ms": round(float(summaries["proposed"]["operator_sum_ms"]), MS_DIGITS),
                "hbm_bytes": float(summaries["proposed"]["hbm_bytes"]),
                "network": dict(summaries["proposed"]["network"]),
                "delta_vs_shipping_ms": round(prop_ms - ship_ms, MS_DIGITS),
                "delta_vs_shipping_pct": round(_pct_delta(ship_ms, prop_ms), PCT_DIGITS),
                "speedup_vs_shipping": round(_speedup_ratio(ship_ms, prop_ms), RATIO_DIGITS),
                "active_open_prs": [
                    {"mechanism_id": m["mechanism_id"], "ref": m["ref"], "notes": m["notes"]}
                    for m in reports["proposed"].get("unmerged_pr_mechanisms", ())
                ],
                "single_pr_ablation": [],
                "leave_one_out_ablation": [],
            },
            "target": {
                "step_ms": round(target_ms, MS_DIGITS),
                "operator_sum_ms": round(float(summaries["target"]["operator_sum_ms"]), MS_DIGITS),
                "hbm_bytes": float(summaries["target"]["hbm_bytes"]),
                "network": dict(summaries["target"]["network"]),
                "delta_vs_proposed_ms": round(target_ms - prop_ms, MS_DIGITS),
                "delta_vs_proposed_pct": round(_pct_delta(prop_ms, target_ms), PCT_DIGITS),
                "delta_vs_shipping_ms": round(target_ms - ship_ms, MS_DIGITS),
                "delta_vs_shipping_pct": round(_pct_delta(ship_ms, target_ms), PCT_DIGITS),
                "speedup_vs_shipping": round(_speedup_ratio(ship_ms, target_ms), RATIO_DIGITS),
                "unimplemented_architectural_gaps": list(reports["target"].get("unimplemented_gaps", ())),
                "active_mechanisms": [m["id"] for m in reports["target"].get("active_mechanisms", ())],
            },
            "hardware_roofline": {
                "physical_ideal_1p0x_ms": round(roof_1p0_ms, MS_DIGITS),
                "remaining_kernel_micro_gap_to_1p0x_ms": round(max(0.0, target_ms - roof_1p0_ms), MS_DIGITS),
            },
        },
        "efficiency_reference": {
            "physical_ideal_ms": round(roof_1p0_ms, MS_DIGITS),
            "basis": "physical_1p0x_critical_path_roofline",
        },
    }


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
        else (winner_mode == "pd_prefill" or (winner_mode != "pd_decode" and int(output_tokens) <= 1))
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
    """Lazily emit ``--xlsx`` / ``--trace`` with explicit role-tagged Before/After polarity."""
    try:
        export_mod = importlib.import_module("mindie_sim.export")
    except ModuleNotFoundError as exc:
        raise ModelingError(
            "solve_export_requires_mindie_sim_export_package"
        ) from exc
    has_target_after = (mode == "modeling") or (rewrite_plan is not None)
    after_track = target_track if mode == "modeling" else compare_track
    before_case, before_req, before_meta = _bundle_inputs_for_solution(
        before_solved, model=model, hardware=hardware, runtime_stack=runtime_stack,
        base_kw=wf_kw, feature_track=compare_track, prompt_tokens=prompt_tokens,
        output_tokens=output_tokens, scope=scope,
    )
    before_bundle = export_mod.build_export_bundle(
        before_case, request=before_req, apply_target_envelope=False,
        handoff_tpot_ms=before_meta["handoff_tpot_ms"],
        deployment_total_dies=before_meta["deployment_total_dies"] or None,
        pd_pool_summary=before_meta["pd_pool_summary"],
    )
    if has_target_after:
        after_case, after_req, after_meta = _bundle_inputs_for_solution(
            after_solved, model=model, hardware=hardware, runtime_stack=runtime_stack,
            base_kw=wf_kw, feature_track=after_track, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, scope=scope,
        )
        after_bundle = export_mod.build_export_bundle(
            after_case, request=after_req,
            apply_target_envelope=True,
            rewrite_plan=rewrite_plan,
            allow_unbound_formula=(mode == "modeling"),
            handoff_tpot_ms=after_meta["handoff_tpot_ms"],
            deployment_total_dies=after_meta["deployment_total_dies"] or None,
            pd_pool_summary=after_meta["pd_pool_summary"],
        )
        anchor_fn = getattr(export_mod, "anchor_stage_waterfall", None)
        if callable(anchor_fn) and getattr(after_bundle, "stage_waterfall", None):
            after_bundle = replace(
                after_bundle, stage_waterfall=anchor_fn(before_bundle, after_bundle),
            )
    else:
        after_req, after_meta, after_bundle = before_req, before_meta, before_bundle

    out: dict[str, Any] = {
        "mode": mode,
        "compare_track": compare_track,
        "target_track": after_track,
        "phase": after_req.phase,
    }
    if after_meta["pd_pool_summary"]:
        out["pd_pool_summary"] = dict(after_meta["pd_pool_summary"])
    if xlsx_path is not None:
        cmp_req_cls = getattr(export_mod, "ComparisonRequest", None)
        assert_pol = getattr(export_mod, "assert_comparison_polarity", None)
        if cmp_req_cls is not None and has_target_after:
            cmp_req = cmp_req_cls(
                before_case_digest=str(getattr(before_bundle, "case_digest", "")),
                after_case_digest=str(getattr(after_bundle, "case_digest", "")),
                comparison_policy=(
                    "signed_comparison" if rewrite_plan is not None else "monotonic_optimization"
                ),
                require_model_scope=(scope == "model"),
            )
            if callable(assert_pol):
                assert_pol(cmp_req, before_bundle, after_bundle)
            receipt = export_mod.export_comparison_xlsx(
                before_bundle, after_bundle, cmp_req, output_path=xlsx_path,
            )
        else:
            target_bundle = after_bundle if has_target_after else before_bundle
            receipt = export_mod.export_comparison_xlsx(target_bundle, xlsx_path)
        out["xlsx"] = receipt.to_dict() if hasattr(receipt, "to_dict") else str(xlsx_path)
    if trace_path is not None:
        trace_target = after_bundle if has_target_after else before_bundle
        receipt = export_mod.export_msprof_trace(trace_target, trace_path)
        out["trace"] = receipt.to_dict() if hasattr(receipt, "to_dict") else str(trace_path)
    return out


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
    is_prefill_only = winner_mode == "pd_prefill" or (winner_mode != "pd_decode" and int(output) <= 1)
    win_phase = "prefill" if is_prefill_only else "decode"
    win_seq = 1 if is_prefill_only else max(1, int((best or {}).get("b_feasible", 1) or 1))
    if is_prefill_only:
        win_tokens, win_context = _prefill_prefix_split(parsed_rec, prompt)
    else:
        win_tokens, win_context = 1 + win_mtp, prompt
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
                canonical_deployment_row_key(row),
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
                canonical_deployment_row_key(row),
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
