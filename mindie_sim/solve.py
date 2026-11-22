"""One-shot high-level deployment solver for AI Agents and production sizing.

[Role] Expose ``solve_deployment``, a single high-level call that returns:
  (1) HBM Capacity Ledger (weights, workspace, KV/seq, max VRAM concurrency);
  (2) Optimal Deployment recommendation (PD Disaggregated vs Colocated, winning
      P/D parallel tuples, P:D split, KV transfer/pool ms, B*, TTFT, TPOT,
      cluster tok/s, tok/s/die, and top runner-ups);
  (3) 4-Track Software Waterfall (raw_eager -> merged -> unmerged_pr ->
      expected_ideal -> hardware roofline);
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
from pathlib import Path
from typing import Any

from mindie_sim.compose import compose_case
from mindie_sim.feature_mechanisms import compare_feature_tracks
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    _resolve_speculative_tokens,
    analyze_network_efficiency,
)
from mindie_sim.prediction.offload.policy import missing_scenario_error
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.sweep.pd import (
    ARCHETYPE_SCREEN_MIN_LAYERS,
    build_archetype_representative_case,
    solve_optimal_deployment,
)

SOLVE_SCHEMA = "mindie-sim.deployment-solution/1"
BYTES_PER_GIB = float(1024 ** 3)
TOP_RUNNER_UPS = 3
TOP_BOTTLENECK_FAMILIES = 8
GIB_DIGITS = 3
RATIO_DIGITS = 4
MS_DIGITS = 4


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
) -> dict[str, Any]:
    """Compute one instance's HBM weights, workspace, KV/seq, and VRAM concurrency."""
    parsed = parse_case(raw_case)
    is_prefill = role == "pd_prefill"
    if is_prefill:
        ws_tokens = max(1, int(prompt_tokens))
        seq_len = int(prompt_tokens) + 1
    else:
        decode_q, _ = _resolve_speculative_tokens(
            parsed, output_budget=output_tokens,
            tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        ws_tokens = max(1, int(decode_q))
        seq_len = int(prompt_tokens) + int(output_tokens)
    mem = Memory(parsed, workspace_tokens=ws_tokens)
    hbm_bytes = min(float(die.memory_bytes) for die in parsed.machine.dies) if parsed.machine.dies else 0.0
    weights_bytes = max((float(v) for v in mem.weights), default=0.0)
    workspace_bytes = max((float(v) for v in mem.workspace), default=0.0)
    kv_per_seq_bytes = max((float(v) for v in mem.cache_bytes(seq_len, padded=True)), default=0.0)
    headroom_bytes = min(
        (float(cap) for row in mem.capacity for cap in row),
        default=0.0,
    )
    max_seqs_vram = (
        int(max(0.0, headroom_bytes) // kv_per_seq_bytes)
        if (headroom_bytes >= 0.0 and kv_per_seq_bytes > 0.0)
        else (int(parsed.runtime.max_num_seqs) if headroom_bytes >= 0.0 else 0)
    )
    return {
        "role": role,
        "parallel": parsed.runtime.parallel.to_dict(),
        "hbm_capacity_gib": round(hbm_bytes / BYTES_PER_GIB, GIB_DIGITS),
        "weights_gib": round(weights_bytes / BYTES_PER_GIB, GIB_DIGITS),
        "workspace_gib": round(workspace_bytes / BYTES_PER_GIB, GIB_DIGITS),
        "kv_per_seq_gib": round(kv_per_seq_bytes / BYTES_PER_GIB, GIB_DIGITS),
        "kv_pool_headroom_gib": round(max(0.0, headroom_bytes) / BYTES_PER_GIB, GIB_DIGITS),
        "max_seqs_per_dp_vram": int(max_seqs_vram),
        "feasible": bool(headroom_bytes >= 0.0 and max_seqs_vram >= 1),
    }


def _build_capacity_ledger(
    solved: Mapping[str, Any], *, prompt_tokens: int, output_tokens: int,
) -> dict[str, Any]:
    winner_mode = solved.get("winner_mode")
    if winner_mode == "pd_disaggregated" and solved.get("prefill_case") and solved.get("decode_case"):
        p_led = _instance_capacity_ledger(
            solved["prefill_case"], prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, role="pd_prefill",
        )
        d_led = _instance_capacity_ledger(
            solved["decode_case"], prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, role="pd_decode",
        )
        return {
            **d_led,
            "feasible": bool(p_led["feasible"] and d_led["feasible"]),
            "prefill": p_led,
            "decode": d_led,
        }
    rec_case = solved.get("recommended_case")
    if rec_case is not None:
        return _instance_capacity_ledger(
            rec_case, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, role="colocated",
        )
    return {
        "role": "none", "hbm_capacity_gib": 0.0, "weights_gib": 0.0,
        "workspace_gib": 0.0, "kv_per_seq_gib": 0.0, "kv_pool_headroom_gib": 0.0,
        "max_seqs_per_dp_vram": 0, "feasible": False,
    }


def _extract_bottleneck_roofline(
    raw_case: Mapping[str, Any],
    *,
    prompt_tokens: int,
    sequences: int,
    tokens: int,
    scope: str,
) -> dict[str, Any]:
    eff = analyze_network_efficiency(
        raw_case, phase="decode", tokens=tokens,
        context=prompt_tokens, sequences=max(1, int(sequences)), scope=scope,
    )
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


def _export_solution_artifacts(
    *,
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any],
    runtime_stack: str | Mapping[str, Any],
    wf_kw: Mapping[str, Any],
    prompt_tokens: int,
    sequences: int,
    tokens: int,
    scope: str,
    xlsx_path: str | Path | None,
    trace_path: str | Path | None,
    compare_track: str,
    target_track: str,
) -> dict[str, Any]:
    """Lazily delegate ``--xlsx`` / ``--trace`` emission to ``mindie_sim.export``."""
    try:
        export_mod = importlib.import_module("mindie_sim.export")
    except ModuleNotFoundError as exc:
        raise ModelingError(
            "solve_export_requires_mindie_sim_export_package"
        ) from exc
    compose_kw = dict(wf_kw)
    before_case = compose_case(
        model, hardware, runtime_stack=runtime_stack,
        feature_track=compare_track, **compose_kw,
    )
    after_case = compose_case(
        model, hardware, runtime_stack=runtime_stack,
        feature_track=target_track, **compose_kw,
    )
    step_req = StepRequest(
        phase="decode", tokens=tokens, context=prompt_tokens,
        sequences=max(1, int(sequences)), scope=scope,
    )
    out: dict[str, Any] = {
        "compare_track": compare_track,
        "target_track": target_track,
    }
    if xlsx_path is not None:
        receipt = export_mod.export_comparison_xlsx(
            after_case, xlsx_path, compare_source=before_case, request=step_req,
        )
        out["xlsx"] = receipt.to_dict() if hasattr(receipt, "to_dict") else str(xlsx_path)
    if trace_path is not None:
        receipt = export_mod.export_msprof_trace(
            after_case, trace_path, request=step_req,
        )
        out["trace"] = receipt.to_dict() if hasattr(receipt, "to_dict") else str(trace_path)
    return out


def _evaluate_winner_analysis(
    *,
    solved: Mapping[str, Any],
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any],
    runtime_stack: str | Mapping[str, Any],
    prompt: int,
    output: int,
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
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Compute bottleneck roofline, 4-track waterfall, and optional exports for winner."""
    rec_case = solved.get("recommended_case")
    if rec_case is None:
        return None, None, None
    best = solved.get("best_configuration")
    parsed_rec = parse_case(rec_case)
    win_mtp = int((best or {}).get("mtp_proposal_tokens", 0) or 0)
    win_seq = max(1, int((best or {}).get("b_feasible", 1) or 1))
    win_tokens = 1 + win_mtp
    win_scope = "model"
    win_role = "pd_decode" if solved.get("winner_mode") == "pd_disaggregated" else "colocated"
    use_arch = parsed_rec.model.layers >= ARCHETYPE_SCREEN_MIN_LAYERS
    if use_arch:
        rep_case, layer_weights, _ = build_archetype_representative_case(parsed_rec)
        eval_case: Any = rep_case
        wf_model: Any = rep_case.model.to_dict()
        wf_layers: int | None = len(rep_case.model.layer_plans)
    else:
        eval_case = rec_case
        layer_weights = {}
        wf_model = model
        wf_layers = layers
    bottleneck = _extract_bottleneck_roofline(
        eval_case, prompt_tokens=prompt, sequences=win_seq,
        tokens=win_tokens, scope=win_scope,
    )
    if use_arch and isinstance(bottleneck, dict):
        inc = dict(bottleneck.get("included_components") or {})
        inc["target_layers"] = int(parsed_rec.model.layers)
        inc["archetype_representative_layers"] = len(layer_weights)
        bottleneck["included_components"] = inc
    par = parsed_rec.runtime.parallel
    wf_kw: dict[str, Any] = dict(
        tp=par.tp, dp=par.dp, ep=par.ep, pp=par.pp, pcp=par.pcp, dcp=par.cp,
        num_devices=par.world_size, layers=wf_layers, quant=quant,
        kv_cache_dtype=kv_cache_dtype, indexer_kv_dtype=indexer_kv_dtype,
        topology_preset=topology_preset, ranks_per_node=ranks_per_node,
        prompt_tokens=prompt, output_tokens=output,
        mtp_proposal_tokens=win_mtp, role=win_role,
        homogeneous_graph_reuse="auto",
    )
    if mtp_accept is not None and win_mtp > 0:
        wf_kw["mtp_acceptance_length"] = float(mtp_accept)
    waterfall = compare_feature_tracks(
        wf_model, hardware=hardware, runtime_stack=runtime_stack,
        phase="decode", tokens=win_tokens, context=prompt,
        sequences=win_seq, scope=win_scope, ablate_prs=ablate_prs,
        **wf_kw,
    )
    if use_arch and isinstance(waterfall, dict):
        waterfall["archetype_representative_layers"] = int(wf_layers or 0)
        waterfall["full_model_layers"] = int(parsed_rec.model.layers)
    export_kw = {**wf_kw, "layers": layers}
    exports = (
        _export_solution_artifacts(
            model=model, hardware=hardware, runtime_stack=runtime_stack,
            wf_kw=export_kw, prompt_tokens=prompt, sequences=win_seq,
            tokens=win_tokens, scope=win_scope,
            xlsx_path=xlsx_path, trace_path=trace_path,
            compare_track=compare_track, target_track=target_track,
        )
        if (xlsx_path is not None or trace_path is not None) else None
    )
    return waterfall, bottleneck, exports


def solve_deployment(
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any] = "a3",
    *,
    num_devices: int | None = None,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    tpot_slo_ms: float | None = None,
    ttft_slo_ms: float | None = None,
    mtp_tokens: int | None = None,
    mtp_accept: float | None = None,
    kv_pooling: str | bool | None = "auto",
    deployment: str = "auto",
    quant: str | Mapping[str, Any] | None = None,
    kv_cache_dtype: str | None = None,
    indexer_kv_dtype: str | None = None,
    topology_preset: str | None = None,
    ranks_per_node: int | None = None,
    layers: int | None = None,
    max_sequences_per_dp: int | None = None,
    ablate_prs: bool = False,
    pool_dram_ratio: float | None = None,
    pool_ssd_ratio: float = 0.0,
    pool_prefix_hit_ratio: float = 0.0,
    ddr_bw_gbps: float | None = None,
    ssd_bw_gbps: float | None = None,
    kv_transfer_bw_gbps: float | None = None,
    pinned_parallel: Mapping[str, int] | None = None,
    pinned_prefill_parallel: Mapping[str, int] | None = None,
    pinned_decode_parallel: Mapping[str, int] | None = None,
    raw_eager: bool = False,
    runtime_stack: str | Mapping[str, Any] = "vllm-ascend",
    optimizations: Sequence[str] = (),
    exclude_optimizations: Sequence[str] = (),
    xlsx_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    compare_track: str = "unmerged_pr",
    target_track: str = "expected_ideal",
) -> dict[str, Any]:
    """One-shot deployment solver returning capacity, optimal PD/colocated, waterfall, and roofline."""
    missing = [
        name for name, val in (
            ("num_devices", num_devices),
            ("prompt_tokens", prompt_tokens),
            ("output_tokens", output_tokens),
            ("tpot_slo_ms", tpot_slo_ms),
        ) if val is None
    ]
    if float(pool_ssd_ratio) > 0.0 and ssd_bw_gbps is None:
        missing.append("ssd_bw_gbps")
    if missing:
        raise missing_scenario_error(missing)

    pd_mode = _resolve_kv_pooling_mode(kv_pooling)
    solved = solve_optimal_deployment(
        model, hardware,
        num_devices=int(num_devices),
        prompt_tokens=int(prompt_tokens),
        output_tokens=int(output_tokens),
        ttft_sla_ms=ttft_slo_ms,
        tpot_sla_ms=float(tpot_slo_ms),
        deployment=deployment,
        pd_transfer_mode=pd_mode,
        kv_transfer_bw_gbps=kv_transfer_bw_gbps,
        pool_dram_ratio=pool_dram_ratio,
        pool_ssd_ratio=pool_ssd_ratio,
        pool_prefix_hit_ratio=pool_prefix_hit_ratio,
        ddr_bw_gbps=ddr_bw_gbps,
        ssd_bw_gbps=ssd_bw_gbps,
        pinned_parallel=pinned_parallel,
        pinned_prefill_parallel=pinned_prefill_parallel,
        pinned_decode_parallel=pinned_decode_parallel,
        mtp_proposal_tokens=mtp_tokens,
        mtp_acceptance_length=mtp_accept,
        max_sequences_per_dp=max_sequences_per_dp,
        quant=quant,
        layers=layers,
        ranks_per_node=ranks_per_node,
        topology_preset=topology_preset,
        raw_eager=raw_eager,
        runtime_stack=runtime_stack,
        optimizations=optimizations,
        exclude_optimizations=exclude_optimizations,
        kv_cache_dtype=kv_cache_dtype,
        indexer_kv_dtype=indexer_kv_dtype,
        allow_unconstrained_ttft=(ttft_slo_ms is None),
    )
    prompt = int(prompt_tokens)
    output = int(output_tokens)
    ledger = _build_capacity_ledger(solved, prompt_tokens=prompt, output_tokens=output)

    best = solved.get("best_configuration")
    feasible_sorted = sorted(
        (row for row in solved.get("candidates", ()) if row.get("feasible")),
        key=lambda row: (
            -float(row.get("sustainable_tps", 0.0)),
            int(row.get("used_dies", 0)),
            float(row.get("effective_tpot_ms", 0.0)),
        ),
    )
    runner_ups = feasible_sorted[1:1 + TOP_RUNNER_UPS]

    waterfall, bottleneck, exports = _evaluate_winner_analysis(
        solved=solved, model=model, hardware=hardware, runtime_stack=runtime_stack,
        prompt=prompt, output=output, layers=layers, quant=quant,
        kv_cache_dtype=kv_cache_dtype, indexer_kv_dtype=indexer_kv_dtype,
        topology_preset=topology_preset, ranks_per_node=ranks_per_node,
        mtp_accept=mtp_accept, ablate_prs=ablate_prs,
        xlsx_path=xlsx_path, trace_path=trace_path,
        compare_track=compare_track, target_track=target_track,
    )

    solution: dict[str, Any] = {
        "schema": SOLVE_SCHEMA,
        "model": solved["model"],
        "hardware": solved["hardware"],
        "num_devices": solved["num_devices"],
        "scenario": {
            "prompt_tokens": prompt,
            "output_tokens": output,
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": float(tpot_slo_ms),
            "kv_pooling": kv_pooling,
        },
        "feasible": bool(solved["feasible"]),
        "capacity_ledger": ledger,
        "recommended_deployment": {
            "winner_mode": solved["winner_mode"],
            "feasible": bool(solved["feasible"]),
            "best_configuration": best,
            "best_pd": solved.get("best_pd"),
            "best_colocated": solved.get("best_colocated"),
            "runner_ups": runner_ups,
        },
        "four_track_waterfall": waterfall,
        "bottleneck_roofline": bottleneck,
    }
    if exports is not None:
        solution["exports"] = exports
    return solution


__all__ = ["SOLVE_SCHEMA", "solve_deployment"]
