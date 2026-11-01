"""Causal baseline, four-dimensional attribution and calibration residuals.

[Role]
  Evaluate one validated Case through the simulator's lowered operator/graph
  engine and derive efficiency, skew, capacity and stage attribution plus
  measured-profile residuals.

[Invariants]
  * Every latency comes from ``analyze_network_efficiency`` or the engine's
    memory/SLA primitives; no model-name constants exist here.
  * Observed profile data is used only for residuals, never for prediction.
  * Caller Cases are never mutated; returned records are fresh dictionaries.

[Timeouts & Locks]
  Pure in-memory bounded DAG evaluation; no I/O, network, or locks.

[Side-Effect Policy]
  Read-only over immutable Case records.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from collections.abc import Mapping
from typing import Any

from .advise_util import (
    CANONICAL_BOTTLENECKS,
    _SLA_SOLVER_MAX_CONTEXT,
    _canonical_bottleneck,
    _normalize_shares,
    _number,
    _pct,
)
from .catalog.ops import barrier_op
from .prediction.body_defaults import TargetGraphSet, consume_target_declaration
from .prediction.capacity import Memory
from .prediction.cohort_traffic import bind_cohort_traffic
from .prediction.efficiency import (
    _STRIP_METADATA_KEYS,
    _vram_max_sequences_per_dp,
    analyze_network_efficiency,
    analyze_operator_row,
    solve_hardware_sla_throughput,
)
from .prediction.frontend import Frontend
from .prediction.graph_analysis import graph_costs
from .prediction.implementation_selection import lower_regions
from .prediction.operator_analysis import operator_costs, target_graphs
from .prediction.scheduler import DraftPolicy
from .prediction.skew_analysis import (
    analyze_moe_routing_imbalance,
    analyze_multi_dp_barrier_skew,
)
from .prediction.speculation import draft_program
from .prediction.structure import normalize_model, parse_case
from .schema.case import Case as CaseSpec
from .schema.common import ModelingError
from .schema.model import RequestSpec
from .schema.workload import RequestView


@dataclass(frozen=True)
class _Config:
    phase: str
    scope: str
    context: int
    sequences: int
    tokens: int
    output_tokens: int
    prefix_tokens: int
    sla_tpot_ms: float
    sla_ttft_ms: float


def _should_run_graph(case: CaseSpec, context: int) -> bool:
    return True


def _proposal_tokens(case: CaseSpec, output_tokens: int) -> int:
    req = RequestSpec(request_id="advise", prompt_tokens=1,
                      output_tokens=max(1, int(output_tokens)), arrival_ms=0.0)
    policy = DraftPolicy.for_request(case, req)
    return max(0, int(policy.proposal))


def _config_for(case: CaseSpec, phase: str, scope: str, context: int | None,
                sequences: int | None, tokens: int | None,
                sla_tpot_ms: float | None, sla_ttft_ms: float | None) -> _Config:
    if phase not in {"decode", "prefill"} or scope not in {"body", "model"}:
        raise ValueError(f"advise_invalid_phase_or_scope:{phase}:{scope}")
    requests = case.workload.requests
    prompt = max((int(request.prompt_tokens) for request in requests), default=4096)
    raw_prefix = max((int(request.prefix_tokens) for request in requests), default=0)
    output = max((int(request.output_tokens) for request in requests), default=128)
    ctx = prompt if context is None else int(context)
    prefix = ctx if (phase == "prefill" and tokens is not None and context is not None) else min(raw_prefix, ctx)
    dp = max(1, int(case.runtime.parallel.dp))
    seqs = (max(1, math.ceil(len(requests) / dp)) if requests else 1) if sequences is None else int(sequences)
    tok = (max(1, ctx - prefix) if phase == "prefill" else 1 + _proposal_tokens(case, output)) if tokens is None else int(tokens)
    if ctx < 0 or seqs < 1 or tok < 1:
        raise ValueError(f"advise_invalid_geometry:{ctx}:{seqs}:{tok}")
    slo_tpot = getattr(case.slo, "tpot_ms", None)
    slo_ttft = getattr(case.slo, "ttft_ms", None)
    tpot_sla = float(sla_tpot_ms if sla_tpot_ms is not None
                     else slo_tpot if slo_tpot is not None else 50.0)
    ttft_sla = float(sla_ttft_ms if sla_ttft_ms is not None
                     else slo_ttft if slo_ttft is not None else 5000.0)
    for name, value in (("sla_tpot_ms", tpot_sla), ("sla_ttft_ms", ttft_sla)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name}_must_be_positive_finite:{value}")
    return _Config(phase, scope, ctx, seqs, tok, max(1, output), prefix, tpot_sla, ttft_sla)


def _analyze(case: Any, cfg: _Config, *, run_graph: bool = True, tokens: int | None = None,
             sequences: int | None = None, context: int | None = None,
             phase: str | None = None) -> dict[str, Any]:
    phase_name = cfg.phase if phase is None else phase
    tok = cfg.tokens if tokens is None else int(tokens)
    seqs = cfg.sequences if sequences is None else int(sequences)
    ctx = (cfg.prefix_tokens if phase_name == "prefill" else cfg.context) if context is None else int(context)
    parsed = case if isinstance(case, CaseSpec) else parse_case(case)
    graphs = target_graphs(
        parsed, phase=phase_name, tokens=tok, context=ctx,
        sequences=seqs, scope=cfg.scope,
    )
    eff_case = consume_target_declaration(parsed, graphs)
    has_kg = any("device_kernel_group" in dict(op.metadata) for op in graphs[0])
    ext_in = {"analysis_input": 0} if cfg.scope == "body" else None
    sg = (TargetGraphSet({0: graphs[0]}, kernels=graphs.effective_kernels, selection=graphs.default_selection)
          if isinstance(graphs, TargetGraphSet) else {0: graphs[0]}) if (
        len(graphs) > 1 and not has_kg and not eff_case.runtime.parallel.ep_spans_dp
    ) else graphs
    strip_keys = _STRIP_METADATA_KEYS - {"graph_segment"}
    sg_map = {
        r: tuple(replace(op, metadata=tuple((k, v) for k, v in op.metadata if k not in strip_keys))
                 for op in ops)
        for r, ops in sg.items()
    }
    sg = TargetGraphSet(sg_map, kernels=sg.effective_kernels, selection=sg.default_selection) if isinstance(sg, TargetGraphSet) else sg_map
    sched_ms = float(eff_case.runtime.submission.get("scheduler_ms", 0.0) or 0.0)
    async_decode_sched = bool(phase_name == "decode" and eff_case.runtime.async_scheduling and sched_ms > 0.0)
    if async_decode_sched:
        no_sched_case = replace(
            eff_case, runtime=replace(eff_case.runtime, submission={**eff_case.runtime.submission, "scheduler_ms": 0.0}))
        raw_report = graph_costs(no_sched_case, sg, external_inputs=ext_in,
                                 include_submission=True, isolated=False)
        work_ms = float(raw_report["duration_ms"])
        report = dict(raw_report, duration_ms=max(work_ms, sched_ms))
    else:
        report = graph_costs(eff_case, sg, external_inputs=ext_in,
                             include_submission=True, isolated=False)
        work_ms = float(report["duration_ms"])
    proposal = _proposal_tokens(parsed, cfg.output_tokens)
    if phase_name == "decode" and tok > 1 and proposal > 0:
        draft_tok = min(tok - 1, proposal)
        fe, model, participants = Frontend(), normalize_model(parsed.model), tuple(graphs.keys())
        views_by_r = {
            r: tuple(RequestView(f"advise{r}:{i}", 1, ctx + tok, cfg.scope == "model", "decode",
                                 draft_verify=True, draft_proposal_tokens=draft_tok) for i in range(seqs))
            for r in participants
        }
        global_views = tuple(v for grp in views_by_r.values() for v in grp) if len(views_by_r) > 1 else None
        _, replay_scope = fe.graph.select(views_by_r[participants[0]], parsed.runtime)
        draft_by_r: dict[int, tuple[Any, ...]] = {}
        for r, r_views in views_by_r.items():
            entry = barrier_op(op_id="iteration_input", deps=(), stage=0, layer=-1,
                               requests=tuple(v.request_id for v in r_views), phase="decode",
                               recipe="input_ready", shape=(seqs * tok, model.hidden_size))
            regions, _ = draft_program(fe, model, parsed.runtime, r_views, entry.op_id,
                                       global_views=global_views, replica=r, replay_scope=replay_scope)
            if not regions:
                continue
            raw_ops, _ = lower_regions(regions, graph=fe.graph, replay_scope=replay_scope,
                                       runtime=parsed.runtime, prefix=(entry,), resource_fusion=True)
            ignored = {"barrier", "sampling", "token_commit", "acceptance"} | ({"logits"} if cfg.scope == "body" else set())
            kept_ids = {op.op_id for op in raw_ops if op.family not in ignored and not (cfg.scope == "body" and "logits" in op.op_id)}
            dep_map: dict[str, tuple[str, ...]] = {}
            for op in raw_ops:
                resolved: list[str] = []
                for d in op.deps:
                    for x in ((d,) if d in kept_ids or d == entry.op_id else dep_map.get(d, ())):
                        if x not in resolved:
                            resolved.append(x)
                dep_map[op.op_id] = tuple(resolved)
            draft_by_r[r] = tuple(replace(op, deps=dep_map[op.op_id]) for op in raw_ops if op.op_id in kept_ids)
        if draft_by_r and any(draft_by_r.values()):
            draft_graphs = bind_cohort_traffic(parsed, draft_by_r)
            d_sg = {
                r: tuple(replace(op, metadata=tuple((k, v) for k, v in op.metadata if k not in strip_keys)) for op in ops)
                for r, ops in draft_graphs.items() if r == 0 or parsed.runtime.parallel.ep_spans_dp
            }
            draft_case = replace(
                parsed, runtime=replace(parsed.runtime, submission={
                    **parsed.runtime.submission, "scheduler_ms": 0.0, "completion_ms": 0.0}))
            d_graph = graph_costs(draft_case, d_sg, external_inputs={"iteration_input": 0},
                                  include_submission=True, isolated=False)
            draft_dur = float(d_graph.get("duration_ms", 0.0) or 0.0)
            comb_map = {r: (*graphs[r], *draft_graphs.get(r, ())) for r in participants}
            graphs = (TargetGraphSet(comb_map, kernels=graphs.effective_kernels, selection=graphs.default_selection)
                      if isinstance(graphs, TargetGraphSet) else comb_map)
            report = {
                **report,
                "duration_ms": (max(work_ms + draft_dur, sched_ms) if async_decode_sched
                                else float(report["duration_ms"]) + draft_dur),
                "timeline": (*report.get("timeline", ()), *d_graph.get("timeline", ())),
            }
    return analyze_network_efficiency(
        parsed, phase=phase_name, tokens=tok, context=ctx, sequences=seqs,
        scope=cfg.scope, _precomputed_graph=(graphs, eff_case, report),
    )


def _canonical_shares(analysis: Mapping[str, Any], case: CaseSpec | None = None) -> dict[str, float]:
    totals = {name: 0.0 for name in CANONICAL_BOTTLENECKS}
    for row in analysis.get("operators", ()):
        if int(row.get("replica", 0) or 0) != 0:
            continue
        family = str(row.get("family", "")).lower()
        name = f"{row.get('op_id', '')} {family} {row.get('recipe', '')}".lower()
        if float(row.get("link_bytes", 0.0) or 0.0) > 0.0 or any(
                token in name for token in ("comm", "link", "hccl", "allreduce", "alltoall", "network")):
            canonical = "comm_bound"
        elif row.get("dominant_bound") == "launch_bound":
            canonical = "launch_host_bound"
        elif row.get("dominant_bound") == "memory_bound":
            canonical = "memory_kv_bound"
        else:
            canonical = "compute_bound"
        totals[canonical] += float(row.get("total_ms", 0.0) or 0.0)
    if case is not None and (
        any(float(d.launch_us) > 0.0 for d in case.machine.dies)
        or any(float(v or 0.0) > 0.0 for v in case.runtime.submission.values())
    ):
        exposed_host_ms = max(
            0.0,
            float(analysis.get("graph_duration_ms", 0.0) or 0.0)
            - float(analysis.get("operator_sum_ms", 0.0) or 0.0),
        )
        totals["launch_host_bound"] += exposed_host_ms
    total = sum(totals.values())
    if total <= 0.0:
        return {name: 0.0 for name in CANONICAL_BOTTLENECKS}
    return {name: round(value / total, 9) for name, value in totals.items()}


def _is_attention_row(row: Mapping[str, Any]) -> bool:
    text = f"{row.get('op_id', '')} {row.get('family', '')} {row.get('recipe', '')}".lower()
    return any(token in text for token in ("attention", "attn", "qkv", "q_a", "q_b", "kv_a",
                                           "kv_b", "mla", "rotary", "rope", "indexer"))


def _is_moe_row(row: Mapping[str, Any]) -> bool:
    text = f"{row.get('op_id', '')} {row.get('family', '')} {row.get('recipe', '')}".lower()
    return any(token in text for token in ("moe", "expert", "alltoall", "all_to_all", "router",
                                           "routing", "topk", "grouped"))


def _is_norm_pointwise_row(row: Mapping[str, Any]) -> bool:
    family = str(row.get("family", "")).lower()
    text = f"{row.get('recipe', '')} {row.get('op_id', '')}".lower()
    return family in {"rmsnorm", "norm", "elementwise", "activation", "swiglu", "tensor_copy",
                      "tensor_fill"} or any(token in text for token in (
                          "norm", "residual", "add", "mul", "silu", "swiglu", "copy", "fill", "cast"))


def _stage_and_family_shares(analysis: Mapping[str, Any], case: CaseSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [row for row in analysis.get("operators", ()) if int(row.get("replica", 0) or 0) == 0]
    total = float(analysis.get("operator_sum_ms", 0.0) or 0.0)
    attention = sum(float(row.get("total_ms", 0.0) or 0.0) for row in rows if _is_attention_row(row))
    moe = sum(float(row.get("total_ms", 0.0) or 0.0) for row in rows if _is_moe_row(row))
    norm = sum(float(row.get("total_ms", 0.0) or 0.0) for row in rows if _is_norm_pointwise_row(row))
    comm = float(analysis.get("breakdown_by_kind", {}).get("communication", {}).get("total_ms", 0.0) or 0.0)
    safe = max(total, 1e-12)
    shares = _canonical_shares(analysis, case)
    efficiency = {
        "cube_mfu": round(float(analysis.get("network_cube_mfu", 0.0) or 0.0), 9),
        "vec_mfu": round(float(analysis.get("network_vec_mfu", 0.0) or 0.0), 9),
        "mbu": round(float(analysis.get("network_mbu", 0.0) or 0.0), 9),
        "link_util": round(float(analysis.get("network_link_util", 0.0) or 0.0), 9),
        "cv_overlap": analysis.get("network_cv_overlap"),
        "roofline_efficiency": round(float(analysis.get("network_roofline_efficiency", 0.0) or 0.0), 9),
        "breakdown_by_bound": analysis.get("breakdown_by_bound", {}),
        "breakdown_by_kind": analysis.get("breakdown_by_kind", {}),
        "dominant_bound": max(shares, key=shares.__getitem__) if any(shares.values()) else "compute_bound",
    }
    stage = {
        "pp_stages": int(case.runtime.parallel.pp),
        "stage_layer_counts": list(case.runtime.pp_layer_partition) or [
            int(case.model.layers // case.runtime.parallel.pp
                + (index < case.model.layers % case.runtime.parallel.pp))
            for index in range(case.runtime.parallel.pp)],
        "attention_share": round(attention / safe, 9),
        "moe_share": round(moe / safe, 9),
        "comm_share": round(comm / safe, 9),
        "norm_pointwise_share": round(norm / safe, 9),
    }
    return efficiency, stage


def _emitted_tokens(case: CaseSpec, cfg: _Config) -> float:
    if cfg.phase == "decode" and cfg.tokens <= 1:
        return 1.0
    request = RequestSpec(request_id="advise", prompt_tokens=max(1, cfg.context),
                          output_tokens=max(1, cfg.output_tokens), arrival_ms=0.0)
    policy = DraftPolicy.for_request(case, request)
    if policy.proposal > 0:
        cap = min(int(policy.proposal), cfg.tokens - 1) if cfg.phase == "decode" else int(policy.proposal)
        if cap <= 0:
            return 1.0
        return max(1.0, min(float(cap + 1), float(policy.mean)))
    return 1.0


def _decode_step(case: Any, cfg: _Config, *, sequences: int | None = None) -> float:
    parsed = case if isinstance(case, CaseSpec) else parse_case(case)
    tok = 1 + _proposal_tokens(parsed, cfg.output_tokens)
    analysis = _analyze(parsed, cfg, run_graph=True, phase="decode", tokens=tok,
                        sequences=cfg.sequences if sequences is None else sequences)
    return float(analysis["graph_duration_ms"])


def _sla_metrics(case: CaseSpec, cfg: _Config, *, step_ms: float,
                 ttft_ms: float | None = None, use_solver: bool = False) -> dict[str, Any]:
    p = case.runtime.parallel
    emitted = _emitted_tokens(case, cfg)
    kv_ctx = (cfg.prefix_tokens + cfg.tokens) if cfg.phase == "prefill" else cfg.context
    kv_tok = 1 if cfg.phase == "prefill" else cfg.tokens
    b_sync = _vram_max_sequences_per_dp(case, context=kv_ctx, tokens_per_seq=kv_tok,
                                        output_budget=cfg.output_tokens, emitted_tokens_per_seq=emitted,
                                        async_scheduling=False)
    b_async = _vram_max_sequences_per_dp(case, context=kv_ctx, tokens_per_seq=kv_tok,
                                         output_budget=cfg.output_tokens, emitted_tokens_per_seq=emitted,
                                         async_scheduling=True)
    b_cfg = _vram_max_sequences_per_dp(case, context=kv_ctx, tokens_per_seq=kv_tok,
                                       output_budget=cfg.output_tokens, emitted_tokens_per_seq=emitted,
                                       async_scheduling=None)
    if case.runtime.memory:
        b_sync = min(b_sync, b_cfg)
        b_async = min(b_async, b_cfg)
    configured = b_async if case.runtime.async_scheduling else b_sync
    if configured <= 0 or (ttft_ms is not None and ttft_ms > cfg.sla_ttft_ms):
        return {"sla_goodput_tok_s": 0.0, "b_vram_max_sync": int(b_sync),
                "b_vram_max_async": int(b_async), "b_sla": 0}
    step_budget_ms = cfg.sla_tpot_ms * emitted
    tpot_step = step_ms if cfg.phase == "decode" else _decode_step(case, cfg, sequences=cfg.sequences)
    prefill_tokens = cfg.tokens if cfg.phase == "prefill" else max(1, cfg.context - cfg.prefix_tokens)
    ttft_cache: dict[int, float] = {cfg.sequences: float(ttft_ms)} if ttft_ms is not None else {}

    def _ttft_at(b: int) -> float:
        if b not in ttft_cache:
            try:
                p_res = _analyze(case, cfg, run_graph=True, phase="prefill",
                                 tokens=prefill_tokens, context=cfg.prefix_tokens, sequences=b)
                ttft_cache[b] = float(p_res["graph_duration_ms"])
            except (ValueError, ModelingError):
                ttft_cache[b] = float("inf")
        return ttft_cache[b]

    if tpot_step <= 0.0:
        b_sla = 0
    elif cfg.context > _SLA_SOLVER_MAX_CONTEXT:
        if tpot_step <= step_budget_ms:
            b_sla = min(configured, cfg.sequences)
        elif cfg.sequences > 1:
            try:
                one_step = _decode_step(case, cfg, sequences=1)
            except (ValueError, ModelingError):
                one_step = float("inf")
            if 0.0 < one_step <= step_budget_ms and (ttft_ms is None or 0.0 < _ttft_at(1) <= cfg.sla_ttft_ms):
                b_sla, tpot_step = 1, one_step
            else:
                b_sla = 0
        else:
            b_sla = 0
    else:
        cache: dict[int, float] = {cfg.sequences: tpot_step}

        def _step_at(b: int) -> float:
            if b not in cache:
                try:
                    cache[b] = _decode_step(case, cfg, sequences=b)
                except (ValueError, ModelingError):
                    cache[b] = float("inf")
            return cache[b]

        if tpot_step <= step_budget_ms:
            lo, hi = min(configured, cfg.sequences), configured
        else:
            lo, hi = 0, min(configured, cfg.sequences - 1)
            if hi >= 1 and 0.0 < _step_at(1) <= step_budget_ms and (ttft_ms is None or 0.0 < _ttft_at(1) <= cfg.sla_ttft_ms):
                lo = 1
            else:
                hi = 0
        if hi > lo and 0.0 < _step_at(hi) <= step_budget_ms:
            lo = hi
        elif hi > lo:
            hi_step, lo_step = _step_at(hi), _step_at(lo)
            probe = (
                max(lo + 1, min(hi - 1, int(math.floor(
                    lo + (hi - lo) * max(0.0, step_budget_ms - lo_step)
                    / max(hi_step - lo_step, 1e-9)))))
                if math.isfinite(hi_step) and math.isfinite(lo_step)
                else (lo + hi + 1) // 2
            )
            if 0.0 < _step_at(probe) <= step_budget_ms:
                lo = probe
            else:
                hi = probe - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if 0.0 < _step_at(mid) <= step_budget_ms:
                    lo = mid
                else:
                    hi = mid - 1
        if lo > cfg.sequences and ttft_ms is not None and ttft_ms * (lo / max(1, cfg.sequences)) > cfg.sla_ttft_ms * 0.9:
            if not (0.0 < _ttft_at(lo) <= cfg.sla_ttft_ms):
                t_lo, t_hi = cfg.sequences, lo - 1
                while t_lo < t_hi:
                    mid = (t_lo + t_hi + 1) // 2
                    if 0.0 < _ttft_at(mid) <= cfg.sla_ttft_ms:
                        t_lo = mid
                    else:
                        t_hi = mid - 1
                lo = t_lo
        b_sla = lo
        if b_sla > 0:
            tpot_step = _step_at(b_sla)
    goodput = (b_sla * int(p.dp) * emitted * 1000.0 / tpot_step) if b_sla > 0 and tpot_step > 0.0 else 0.0
    return {"sla_goodput_tok_s": round(goodput, 6), "b_vram_max_sync": int(b_sync),
            "b_vram_max_async": int(b_async), "b_sla": int(b_sla)}


def _memory_facts(case: CaseSpec, cfg: _Config) -> tuple[Memory, float, float, float, float]:
    memory = Memory(case, workspace_tokens=max(1, cfg.sequences * cfg.tokens))
    available = min(min(row) for row in memory.capacity)
    hbm = min(die.hbm_bytes for die in case.machine.dies)
    kv_tokens = (cfg.prefix_tokens + cfg.tokens + max(1, cfg.output_tokens)) if cfg.phase == "prefill" else (
        cfg.context + max(cfg.tokens, cfg.output_tokens))
    kv = max(memory.cache_bytes(kv_tokens, padded=True))
    workspace = max(memory.workspace)
    return memory, float(available), float(hbm), float(kv), float(workspace)


def _capacity_metrics(case: CaseSpec, cfg: _Config, b_sync: int, b_async: int) -> dict[str, Any]:
    memory, available, hbm, kv, workspace = _memory_facts(case, cfg)
    headroom = available - kv * cfg.sequences
    used = max(memory.weights) + workspace + kv * cfg.sequences + memory.policy.reserved
    b_eff = b_async if case.runtime.async_scheduling else b_sync
    return {
        "vram_max_sequences_per_dp_sync": int(b_sync),
        "vram_max_sequences_per_dp_async": int(b_async),
        "weight_bytes_per_die": round(float(max(memory.weights)), 3),
        "kv_bytes_per_seq": round(kv, 3),
        "available_kv_capacity_bytes_per_die": round(available, 3),
        "hbm_utilization_ratio": round(max(0.0, min(1.0, used / max(hbm, 1e-12))), 9),
        "capacity_headroom_bytes": round(headroom, 3),
        "is_vram_feasible": bool(b_eff >= cfg.sequences and headroom >= -1e-6),
    }


def _skew_metrics(case: CaseSpec, cfg: _Config) -> dict[str, Any]:
    dp = int(case.runtime.parallel.dp)
    eff_ctx = cfg.prefix_tokens if cfg.phase == "prefill" else cfg.context
    has_moe = bool(int(getattr(case.model, "num_experts", 0) or 0)) or any(
        int(getattr(layer, "num_experts", 0) or 0) > 0 or getattr(layer, "ffn", "") == "moe"
        for layer in case.model.layer_plans)
    barrier, ep_coupled = 0.0, False
    if dp > 1:
        try:
            skew = analyze_multi_dp_barrier_skew(
                case, dp_contexts=tuple(eff_ctx for _ in range(dp)),
                dp_sequences=tuple(cfg.sequences for _ in range(dp)),
                phase=cfg.phase, tokens=cfg.tokens, scope=cfg.scope,
            )
            barrier = float(skew.get("max_barrier_wait_ms", 0.0) or 0.0)
            ep_coupled = bool(skew.get("ep_coupled", False))
        except (ValueError, ModelingError):
            barrier, ep_coupled = 0.0, False
    ratio = 1.0
    if has_moe:
        try:
            ratio = float(analyze_moe_routing_imbalance(
                case, phase=cfg.phase, tokens=cfg.tokens, context=eff_ctx,
                sequences=cfg.sequences, scope=cfg.scope, expert_imbalance_factor=1.0,
            ).get("imbalance_slowdown_ratio", 1.0) or 1.0)
        except (ValueError, ModelingError):
            ratio = 1.0
    return {
        "ep_coupled": ep_coupled,
        "num_dp_replicas": dp,
        "has_routed_moe": has_moe,
        "multi_dp_barrier_wait_ms": round(barrier, 9),
        "moe_imbalance_slowdown_ratio": round(ratio, 9),
    }


def _baseline(case: CaseSpec, cfg: _Config) -> dict[str, Any]:
    analysis = _analyze(case, cfg, run_graph=True)
    step = float(analysis["graph_duration_ms"])
    emitted = _emitted_tokens(case, cfg)
    tpot = step / emitted if cfg.phase == "decode" else _decode_step(case, cfg, sequences=cfg.sequences) / emitted
    if cfg.phase == "decode":
        prefill_tokens = max(1, cfg.context - cfg.prefix_tokens)
        prefill = _analyze(case, cfg, run_graph=True, phase="prefill", tokens=prefill_tokens,
                           context=cfg.prefix_tokens, sequences=cfg.sequences)
        ttft = float(prefill["graph_duration_ms"])
    else:
        ttft = step
    sla = _sla_metrics(case, cfg, step_ms=step, ttft_ms=ttft)
    efficiency, stage = _stage_and_family_shares(analysis, case)
    shares = _canonical_shares(analysis, case)
    attribution = {
        "efficiency": efficiency,
        "skew": _skew_metrics(case, cfg),
        "capacity": _capacity_metrics(case, cfg, sla["b_vram_max_sync"], sla["b_vram_max_async"]),
        "stage": stage,
    }
    return {
        "analysis": analysis,
        "step_duration_ms": step,
        "operator_sum_ms": float(analysis["operator_sum_ms"]),
        "tpot_p50_ms": tpot,
        "ttft_p50_ms": ttft,
        "sla_goodput_tok_s": sla["sla_goodput_tok_s"],
        "b_vram_max_sync": sla["b_vram_max_sync"],
        "b_vram_max_async": sla["b_vram_max_async"],
        "b_sla": sla["b_sla"],
        "dominant_bottleneck": max(shares, key=shares.__getitem__) if any(shares.values()) else "compute_bound",
        "attribution": attribution,
        "canonical_shares": shares,
    }


def _hw_family(text: str) -> str:
    low = text.strip().lower()
    if any(t in low for t in ("a5", "910_95", "950")):
        return "a5"
    if any(t in low for t in ("a3", "910_93", "910c")):
        return "a3"
    if any(t in low for t in ("a2", "910b")):
        return "a2"
    return low


def _calibration(baseline: Mapping[str, Any], observed: Mapping[str, Any],
                 *, cfg: _Config | None = None, dp: int = 1,
                 case: CaseSpec | None = None) -> dict[str, Any]:
    predicted_step = float(baseline["step_duration_ms"])
    predicted_tpot = float(baseline["tpot_p50_ms"])
    predicted_ttft = float(baseline["ttft_p50_ms"])
    observed_step = next((value for value in (
        _number(observed.get("observed_step_ms")), _number(observed.get("step_time_ms")),
        _number(observed.get("step_ms"))) if value is not None), None)
    observed_tpot = next((value for value in (
        _number(observed.get("observed_tpot_p50_ms")), _number(observed.get("tpot_p50_ms")))
        if value is not None), None)
    observed_ttft = next((value for value in (
        _number(observed.get("observed_ttft_p50_ms")), _number(observed.get("ttft_p50_ms")))
        if value is not None), None)
    observed_shares = _normalize_shares(observed.get("observed_share_by_bound", {}) or {})
    if not observed_shares:
        raw_shares: dict[str, float] = {}
        comm_s = max(_number(observed.get("comm_share")) or 0.0, _number(observed.get("hccl_share")) or 0.0)
        launch_s = (_number(observed.get("aicpu_share")) or 0.0) + (_number(observed.get("host_share")) or 0.0)
        mem_s = _number(observed.get("memory_share")) or 0.0
        comp_core = (_number(observed.get("cube_share")) or 0.0) + (_number(observed.get("vector_share")) or 0.0)
        comp_alt = (_number(observed.get("attention_share")) or 0.0) + (_number(observed.get("moe_share")) or 0.0)
        comp_s = comp_core if comp_core > 0.0 else comp_alt
        for b_key, b_val in (("comm_bound", comm_s), ("launch_host_bound", launch_s),
                             ("memory_kv_bound", mem_s), ("compute_bound", comp_s)):
            if b_val > 0.0:
                raw_shares[b_key] = b_val
        observed_shares = _normalize_shares(raw_shares)
    observed_bottleneck = _canonical_bottleneck(
        observed.get("observed_dominant_bottleneck") or observed.get("dominant_bottleneck")
        or observed.get("dominant_bound"))
    if observed_bottleneck is None and observed_shares and (
        sum(observed_shares.values()) >= 0.5 or max(observed_shares.values()) >= 0.5
    ):
        observed_bottleneck = max(observed_shares, key=observed_shares.__getitem__)
    predicted_bottleneck = baseline["dominant_bottleneck"]
    step_delta = None if observed_step is None else observed_step - predicted_step
    tpot_delta = None if observed_tpot is None else observed_tpot - predicted_tpot
    ttft_delta = None if observed_ttft is None else observed_ttft - predicted_ttft
    predicted_shares = {name: float(baseline["canonical_shares"].get(name, 0.0))
                        for name in CANONICAL_BOTTLENECKS}
    residuals = {}
    if observed_shares:
        bound_keys = CANONICAL_BOTTLENECKS if sum(observed_shares.values()) >= 0.999 else tuple(observed_shares)
        for name in bound_keys:
            pred = predicted_shares.get(name, 0.0)
            obs = float(observed_shares.get(name, 0.0))
            residuals[name] = {"predicted_share": round(pred, 9),
                               "observed_share": round(obs, 9),
                               "residual": round(obs - pred, 9),
                               "abs_residual": round(abs(obs - pred), 9)}
    aligned = bool(observed_bottleneck is not None and observed_bottleneck == predicted_bottleneck)
    obs_phase = str(observed.get("phase") or observed.get("observed_phase") or "").strip().lower()
    if obs_phase.startswith("prefill"):
        obs_phase = "prefill"
    elif obs_phase.startswith("decode"):
        obs_phase = "decode"
    obs_conc = _number(observed.get("concurrency") or observed.get("observed_concurrency"))
    obs_ctx = _number(observed.get("context") or observed.get("observed_context")
                      or observed.get("prompt_tokens") or observed.get("input_len") or observed.get("seq_len"))
    obs_scope = str(observed.get("scope") or observed.get("observed_scope") or "").strip().lower()
    obs_hw = str(observed.get("hardware") or observed.get("observed_hardware") or observed.get("sku") or "").strip()
    obs_impl = str(observed.get("vllm_impl") or observed.get("observed_vllm_impl") or "").strip().lower()
    obs_bound = str(observed.get("measurement_boundary") or observed.get("metric_boundary")
                    or observed.get("boundary") or "").strip().lower()
    geom_mismatch = False
    if cfg is not None:
        if obs_phase in {"decode", "prefill"} and obs_phase != cfg.phase:
            geom_mismatch = True
        if obs_conc is not None and int(round(obs_conc)) != cfg.sequences * max(1, int(dp)):
            geom_mismatch = True
        if obs_ctx is not None and int(round(obs_ctx)) != cfg.context:
            geom_mismatch = True
        if obs_scope in {"body", "model"} and obs_scope != cfg.scope:
            geom_mismatch = True
    if case is not None:
        if obs_hw:
            case_hws = {
                _hw_family(str(v))
                for v in (case.machine.sku, case.machine.name, getattr(case.machine, "hardware_key", None))
                if v and str(v).strip()
            }
            if case_hws and _hw_family(obs_hw) not in case_hws:
                geom_mismatch = True
        if obs_impl and case.runtime.vllm_impl and obs_impl != str(case.runtime.vllm_impl).strip().lower():
            geom_mismatch = True
        if obs_bound in {"client", "client_stream", "client_delivery", "e2e_client", "service", "service_client"}:
            geom_mismatch = True
        if obs_bound in {"device_only", "pure_device", "isolated_operators"} and any(
                float(v or 0.0) > 0.0 for v in case.runtime.submission.values()):
            geom_mismatch = True
        obs_layers = _number(observed.get("layers") or observed.get("observed_layers") or observed.get("num_layers"))
        if obs_layers is not None and int(round(obs_layers)) != int(case.model.layers):
            geom_mismatch = True
        obs_exp = _number(observed.get("experts") or observed.get("num_experts") or observed.get("n_routed_experts"))
        if obs_exp is not None and int(round(obs_exp)) != int(case.model.num_experts or 0):
            geom_mismatch = True
        obs_topk = _number(observed.get("top_k") or observed.get("num_experts_per_tok"))
        if obs_topk is not None and int(round(obs_topk)) != int(case.model.top_k or 0):
            geom_mismatch = True
        for axis in ("tp", "dp", "ep", "pp"):
            ax_val = _number(observed.get(axis))
            if ax_val is not None and int(round(ax_val)) != int(getattr(case.runtime.parallel, axis, 1)):
                geom_mismatch = True
        obs_q = str(observed.get("quantization") or observed.get("quant") or "").strip().lower().replace("-", "").replace("_", "")
        if obs_q and obs_q not in {"none", "auto", "ascend"}:
            case_q = f"{getattr(case.model.quant, 'name', '') or ''} {case.model.weight_dtype or ''}".lower().replace("-", "").replace("_", "")
            q_fam = lambda s: ("w4a8" if ("w4a8" in s or "int4" in s) else "w8a8" if ("w8a8" in s or "int8" in s) else "fp8" if "fp8" in s else "bf16" if ("bf16" in s or "fp16" in s) else s)
            if q_fam(obs_q) != q_fam(case_q):
                geom_mismatch = True
        obs_exec = str(observed.get("execution_mode") or observed.get("graph_mode")
                       or observed.get("graph_capture_mode") or "").strip().lower()
        if obs_exec and obs_exec not in {"auto", "unspecified"}:
            norm_exec = lambda s: ("piecewise" if "piecewise" in s
                                   else "full" if any(t in s for t in ("aclgraph", "torchair", "full"))
                                   else "none" if any(t in s for t in ("eager", "none")) else s)
            case_exec = str(case.runtime.graph_capture_mode or case.runtime.graph_mode
                            or dict(case.runtime.metadata).get("execution_mode") or "none").strip().lower()
            if norm_exec(obs_exec) != norm_exec(case_exec):
                geom_mismatch = True
        obs_mod = str(observed.get("model") or observed.get("model_name") or "").strip()
        if obs_mod:
            from .profile_ingest import _MODEL_PATTERNS, _canonical_model

            def _mid(val: str) -> str:
                canon = _canonical_model(val)
                if canon:
                    return canon
                low = val.strip().lower()
                for needles, target in _MODEL_PATTERNS:
                    if any(n in low for n in needles):
                        return target
                return low.replace("_", "-").replace(".", "")

            obs_id = _mid(obs_mod)
            case_mods = {
                _mid(str(v))
                for v in (getattr(case.model, "catalog_key", None), case.model.name)
                if v and str(v).strip()
            }
            if obs_id and case_mods and obs_id not in case_mods:
                geom_mismatch = True
    phase_name = cfg.phase if cfg is not None else "decode"
    has_latency = (observed_step is not None or observed_tpot is not None) if phase_name == "decode" else (
        observed_step is not None or observed_ttft is not None
    )
    if geom_mismatch:
        status = "geometry_mismatch"
    elif observed_bottleneck is None or not has_latency:
        status = "insufficient_observed_metrics"
    elif not aligned:
        status = "bottleneck_mismatch"
    elif phase_name == "prefill":
        primary_delta = ttft_delta if ttft_delta is not None else step_delta
        primary_pred = predicted_ttft if ttft_delta is not None else predicted_step
        pct_err = abs(_pct(primary_delta, primary_pred) or 0.0)
        status = "calibrated" if pct_err <= 10.0 else ("calibrated_with_bias" if pct_err <= 25.0 else "latency_bias")
    elif (tpot_delta is None or abs(_pct(tpot_delta, predicted_tpot) or 0.0) <= 10.0) and (
            step_delta is None or abs(_pct(step_delta, predicted_step) or 0.0) <= 10.0):
        status = "calibrated"
    elif (tpot_delta is not None and abs(_pct(tpot_delta, predicted_tpot) or 0.0) <= 25.0) or (
            tpot_delta is None and step_delta is not None and abs(_pct(step_delta, predicted_step) or 0.0) <= 25.0):
        status = "calibrated_with_bias"
    else:
        status = "latency_bias"
    return {
        "available": True,
        "observed_step_ms": observed_step,
        "predicted_step_ms": predicted_step,
        "step_delta_ms": step_delta,
        "step_delta_pct": _pct(step_delta, predicted_step),
        "observed_tpot_p50_ms": observed_tpot,
        "predicted_tpot_p50_ms": predicted_tpot,
        "tpot_delta_ms": tpot_delta,
        "tpot_delta_pct": _pct(tpot_delta, predicted_tpot),
        "observed_ttft_p50_ms": observed_ttft,
        "predicted_ttft_p50_ms": predicted_ttft,
        "ttft_delta_pct": _pct(ttft_delta, predicted_ttft),
        "observed_dominant_bottleneck": observed_bottleneck,
        "predicted_dominant_bottleneck": predicted_bottleneck,
        "bottleneck_aligned": aligned,
        "share_residuals": residuals,
        "calibration_status": status,
    }


def _is_a3_or_a5(case: CaseSpec) -> bool:
    text = f"{case.machine.sku} {case.machine.name} {case.machine.hardware_key}".lower()
    return any(token in text for token in ("a3", "910_93", "a5", "910_95", "950"))


__all__ = [name for name in globals() if not name.startswith("__")]
