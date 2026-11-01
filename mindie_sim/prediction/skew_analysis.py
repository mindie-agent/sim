"""Multi-DP KV context skew, MoE expert routing imbalance, and chunked-prefill interference analysis.

[Role] Quantify multi-DP MoE AllToAllV barrier waiting time under context/batch
  skew, hot-rank MoE expert routing imbalance with EPLB redundant-expert
  mitigation, and Decode ITL/TPOT inflation under hybrid chunked prefill.
[Invariants] Physical FLOPs, HBM bytes, link bytes, and token rows are strictly
  conserved. At uniform DP load (equal dp_contexts/dp_sequences), balanced MoE
  routing (expert_imbalance_factor == 1.0), and zero prefill chunk tokens
  (C == 0), all equations reproduce baseline simulator latency with 0.00% drift.
[Timeouts & Locks] Pure in-memory deterministic arithmetic and cached operator
  evaluations; no I/O, subprocesses, sleeps, or device locks.
[Side-Effect Policy] Input cases and baseline efficiency reports are never
  mutated; all analysis functions return new immutable dictionary records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

from mindie_sim.catalog.quant import projection_quant
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.body_defaults import TargetGraphSet, consume_target_declaration, effective_body_declaration
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.cohort_traffic import bind_cohort_traffic
from mindie_sim.prediction.dispatch import cohorts
from mindie_sim.prediction.efficiency import _STRIP_METADATA_KEYS, _fast_operator_and_graph_reports, analyze_network_efficiency
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import _target_ops
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.workload import RequestView

_LOCAL_ATTN_TOKENS: tuple[str, ...] = (
    "attention", "attn", "qkv", "q_a", "q_b", "q_proj", "kv_a", "kv_b", "kv_store",
    "kv_norm", "q_norm", "q_absorb", "v_expand", "rotary", "rope", "index",
    "compress", "norm1", "embedding", "logits",
)
_SHARED_EP_FFN_FAMILIES: frozenset[str] = frozenset({
    "grouped_gemm", "moe_ffn", "moe", "alltoall", "alltoallv", "gate_up_projection",
    "down_projection", "swiglu", "situ", "routing", "router", "topk",
})
_SHARED_EP_FFN_TOKENS: tuple[str, ...] = (
    "dispatch", "combine", "alltoall", "all_to_all", "expert", "grouped", "shared",
    "ffn", "router", "route_", "topk", "pack", "unpack", "gate_up", "dense_gate_up",
    "dense_down", "swiglu", "situ", "mc2", "latent_down", "latent_up", "latent_norm",
    ".join", "norm2",
)
_ALLTOALL_TOKENS: tuple[str, ...] = ("all_to_all", "alltoall", "dispatch", "combine")


def _validate_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"skew_analysis_invalid_{name}:{value}")
    return int(value)


def _validate_float(name: str, value: Any, *, minimum: float = 0.0, strict_gt: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"skew_analysis_invalid_{name}:{value}")
    fval = float(value)
    if (fval <= minimum) if strict_gt else (fval < minimum):
        raise ValueError(f"skew_analysis_invalid_{name}:{value}")
    return fval


def _validate_phase_and_scope(phase: str, scope: str) -> None:
    if phase not in {"prefill", "decode"} or scope not in {"body", "model"}:
        raise ValueError(f"skew_analysis_invalid_phase_or_scope:{phase}:{scope}")


def _is_shared_ep_or_ffn_operator(op: Mapping[str, Any]) -> bool:
    """Return True if operator belongs to the MoE AllToAll + routed/shared FFN path."""
    op_id, family, recipe = str(op.get("op_id", "")).lower(), str(op.get("family", "")).lower(), str(op.get("recipe", "")).lower()
    if any(tok in f"{op_id} {family} {recipe}" for tok in _LOCAL_ATTN_TOKENS):
        return False
    return family in _SHARED_EP_FFN_FAMILIES or any(tok in op_id or tok in recipe for tok in _SHARED_EP_FFN_TOKENS)


def _is_routed_moe_compute_operator(op: Mapping[str, Any]) -> bool:
    """Return True if operator is a per-expert routed MoE compute/quant/activation kernel."""
    if float(op.get("link_bytes", 0.0) or 0.0) > 0.0:
        return False
    op_id, family, recipe = str(op.get("op_id", "")).lower(), str(op.get("family", "")).lower(), str(op.get("recipe", "")).lower()
    if "shared" in op_id or "shared" in recipe or recipe == "expert_add" or op_id.endswith(".join"):
        return False
    return family in {"grouped_gemm", "moe_ffn"} or any(tok in op_id or tok in recipe for tok in ("grouped", "expert"))


def _is_moe_alltoall_operator(op: Mapping[str, Any]) -> bool:
    """Return True if operator is an MoE AllToAll dispatch/combine collective."""
    op_id, family, recipe = str(op.get("op_id", "")).lower(), str(op.get("family", "")).lower(), str(op.get("recipe", "")).lower()
    return family in {"alltoall", "alltoallv"} or any(tok in op_id or tok in recipe for tok in _ALLTOALL_TOKENS)


def _is_attention_core_operator(op: Mapping[str, Any]) -> bool:
    """Return True if operator is the KV-cache/causal attention core kernel."""
    op_id, family, recipe = str(op.get("op_id", "")).lower(), str(op.get("family", "")).lower(), str(op.get("recipe", "")).lower()
    if "output" in family or "residual" in recipe or "residual" in op_id:
        return False
    return family == "attention" or "attention" in recipe


def _has_routed_moe(parsed: CaseSpec) -> bool:
    if int(getattr(parsed.model, "num_experts", 0) or 0) > 0:
        return True
    return any(
        int(getattr(lp, "num_experts", 0) or getattr(lp, "n_routed_experts", 0) or 0) > 0 or getattr(lp, "ffn", "") == "moe"
        for lp in parsed.model.layer_plans
    )


def _eplb_extra_weight_bytes_per_die(parsed: CaseSpec, redundant_experts: int) -> float:
    """Compute exact extra resident weight bytes per Die for replicated routed experts."""
    if redundant_experts <= 0 or not _has_routed_moe(parsed):
        return 0.0
    mem, p = Memory(parsed), parsed.runtime.parallel
    etp, base_quant = max(1, int(p.etp)), parsed.model.quant or QuantSpec("bf16")
    stage_extra = [0.0] * max(1, int(p.pp))
    for lp in parsed.model.layer_plans:
        n_exp = int(getattr(lp, "num_experts", 0) or getattr(lp, "n_routed_experts", 0) or 0)
        moe_inter = int(getattr(lp, "moe_intermediate_size", 0) or 0)
        if getattr(lp, "ffn", "") != "moe" or n_exp <= 0 or moe_inter <= 0:
            continue
        quant = projection_quant(base_quant, lp)
        latent, sharded_inter = int(getattr(lp, "expert_latent_size", 0) or parsed.model.hidden_size), math.ceil(moe_inter / etp)
        one_expert_bytes = 2.0 * float(quant.matrix_bytes(sharded_inter, latent, "routed_expert")) + float(quant.matrix_bytes(latent, sharded_inter, "routed_expert"))
        stage_idx = mem.layer_stage[lp.index] if lp.index < len(mem.layer_stage) else 0
        stage_extra[stage_idx] += float(redundant_experts) * one_expert_bytes
    return float(max(stage_extra, default=0.0))


def _greedy_eplb_rank_loads(
    expert_counts: Sequence[float], ep_ranks: int, redundant_experts: int
) -> tuple[list[float], list[float]]:
    """Partition experts across EP ranks (strided e % ep) and greedily place redundant expert replicas."""
    num_exp, ep = len(expert_counts), max(1, min(len(expert_counts), int(ep_ranks)))
    expert_to_ranks: list[list[int]] = [[e % ep] for e in range(num_exp)]

    def _compute_rank_loads(owners: Sequence[Sequence[int]]) -> list[float]:
        loads = [0.0] * ep
        for e_idx, r_list in enumerate(owners):
            share = float(expert_counts[e_idx]) / len(r_list)
            for rk in r_list:
                loads[rk] += share
        return loads

    initial_loads = _compute_rank_loads(expert_to_ranks)
    if redundant_experts <= 0 or ep <= 1:
        return initial_loads, list(initial_loads)

    owners = [list(r_list) for r_list in expert_to_ranks]
    for _ in range(redundant_experts):
        cur_loads = _compute_rank_loads(owners)
        hottest_exp = max(range(num_exp), key=lambda e: float(expert_counts[e]) / len(owners[e]))
        if float(expert_counts[hottest_exp]) <= 0.0:
            break
        existing = set(owners[hottest_exp])
        candidate_ranks = [r for r in range(ep) if r not in existing] or list(range(ep))
        owners[hottest_exp].append(min(candidate_ranks, key=lambda r: cur_loads[r]))
    return initial_loads, _compute_rank_loads(owners)


def analyze_multi_dp_barrier_skew(
    case: CaseSpec | Mapping[str, Any], *, dp_contexts: Sequence[int],
    dp_sequences: Sequence[int] | int = 1, phase: str = "decode", tokens: int = 1, scope: str = "body",
) -> dict[str, Any]:
    """Analyze multi-DP context/batch skew and MoE AllToAllV barrier waiting mechanics."""
    _validate_phase_and_scope(phase, scope)
    _validate_int("tokens", tokens, minimum=1)
    if not isinstance(dp_contexts, Sequence) or isinstance(dp_contexts, (str, bytes, Mapping)) or len(dp_contexts) == 0:
        raise ValueError("skew_analysis_dp_contexts_must_be_nonempty_sequence")
    contexts = tuple(_validate_int("dp_context", c, minimum=0) for c in dp_contexts)
    num_replicas = len(contexts)

    if isinstance(dp_sequences, int) and not isinstance(dp_sequences, bool):
        sequences = (_validate_int("dp_sequences", dp_sequences, minimum=1),) * num_replicas
    elif isinstance(dp_sequences, Sequence) and not isinstance(dp_sequences, (str, bytes, Mapping)) and len(dp_sequences) == num_replicas:
        sequences = tuple(_validate_int("dp_sequence", s, minimum=1) for s in dp_sequences)
    else:
        raise ValueError("skew_analysis_dp_sequences_must_be_positive_int_or_matching_sequence")

    parsed = parse_case(case)
    p = parsed.runtime.parallel
    if num_replicas != p.dp:
        raise ValueError(f"dp_contexts_length_must_match_parallel_dp:{num_replicas}!={p.dp}")

    has_moe = _has_routed_moe(parsed)
    ep_cohorts = cohorts(p) if has_moe else [tuple([r]) for r in range(num_replicas)]
    ep_coupled = bool(has_moe and any(len(g) > 1 for g in ep_cohorts))

    eval_cache: dict[tuple[int, int], tuple[float, float, float]] = {}

    def _eval_pair(ctx: int, seqs: int) -> tuple[float, float, float]:
        key = (ctx, seqs)
        if key not in eval_cache:
            rep = analyze_network_efficiency(parsed, phase=phase, tokens=tokens, context=ctx, sequences=seqs, scope=scope, run_graph=False)
            step_ms, op_sum_ms = float(rep["graph_duration_ms"]), float(rep["operator_sum_ms"])
            scale = (step_ms / op_sum_ms) if op_sum_ms > 0.0 else 1.0
            ep_ms = sum(float(op["total_ms"]) * scale for op in rep["operators"] if _is_shared_ep_or_ffn_operator(op))
            eval_cache[key] = (step_ms, max(0.0, step_ms - ep_ms), ep_ms)
        return eval_cache[key]

    replica_metrics = [_eval_pair(contexts[r], sequences[r]) for r in range(num_replicas)]
    waits = [0.0] * num_replicas
    cohort_steps: list[float] = []
    for group in ep_cohorts:
        if len(group) > 1 and ep_coupled:
            g_max_local = max(replica_metrics[r][1] for r in group)
            g_ep_sync = max(replica_metrics[r][2] for r in group)
            cohort_steps.append(g_max_local + g_ep_sync)
            for r in group:
                waits[r] = max(0.0, g_max_local - replica_metrics[r][1])
        else:
            for r in group:
                cohort_steps.append(replica_metrics[r][0])

    coupled_step_ms = max(cohort_steps, default=0.0)
    straggler_idx = (
        max(range(num_replicas), key=lambda r: (replica_metrics[r][1], replica_metrics[r][0]))
        if ep_coupled else max(range(num_replicas), key=lambda r: replica_metrics[r][0])
    )

    total_seqs = sum(sequences)
    total_kv_tokens = sum(sequences[r] * contexts[r] for r in range(num_replicas))
    balanced_context = int(round(total_kv_tokens / total_seqs))
    balanced_seqs = sequences[0] if len(set(sequences)) == 1 else max(1, int(round(total_seqs / num_replicas)))
    balanced_step_ms, _, _ = _eval_pair(balanced_context, balanced_seqs)

    max_wait_ms, mean_wait_ms = max(waits, default=0.0), sum(waits) / num_replicas
    slowdown_ratio = coupled_step_ms / max(balanced_step_ms, 1e-12)

    return {
        "schema": "mindie-sim.multi-dp-skew/1", "ep_coupled": ep_coupled,
        "cohorts": [list(g) for g in ep_cohorts], "num_dp_replicas": num_replicas,
        "straggler_replica": int(straggler_idx), "coupled_step_ms": float(coupled_step_ms),
        "kv_balanced_step_ms": float(balanced_step_ms), "max_barrier_wait_ms": float(max_wait_ms),
        "mean_barrier_wait_ms": float(mean_wait_ms), "skew_slowdown_ratio": float(slowdown_ratio),
        "kv_balance_speedup_ratio": float(slowdown_ratio),
        "replicas": [
            {
                "replica": r, "context": int(contexts[r]), "sequences": int(sequences[r]),
                "standalone_step_ms": float(replica_metrics[r][0]),
                "attention_and_local_ms": float(replica_metrics[r][1]),
                "shared_ep_and_ffn_ms": float(replica_metrics[r][2]),
                "barrier_wait_ms": float(waits[r]),
            }
            for r in range(num_replicas)
        ],
    }


def analyze_moe_routing_imbalance(
    case: CaseSpec | Mapping[str, Any], *, phase: str = "decode", tokens: int = 1,
    context: int = 4096, sequences: int = 16, scope: str = "body",
    expert_imbalance_factor: float = 1.0, expert_token_counts: Sequence[int] | None = None,
    eplb_redundant_experts: int = 0,
) -> dict[str, Any]:
    """Analyze MoE hot-rank expert routing imbalance and EPLB redundant-expert mitigation."""
    _validate_phase_and_scope(phase, scope)
    _validate_int("tokens", tokens, minimum=1)
    _validate_int("context", context, minimum=0)
    _validate_int("sequences", sequences, minimum=1)
    imbalance_in = _validate_float("expert_imbalance_factor", expert_imbalance_factor, minimum=1.0)
    redundant_in = _validate_int("eplb_redundant_experts", eplb_redundant_experts, minimum=0)

    parsed = parse_case(case)
    p = parsed.runtime.parallel
    ep = max(1, int(p.ep))
    moe_experts = max(
        [int(getattr(parsed.model, "num_experts", 0) or 0)]
        + [int(getattr(lp, "num_experts", 0) or getattr(lp, "n_routed_experts", 0) or 0) for lp in parsed.model.layer_plans]
    )
    max_top_k = max(
        [int(getattr(parsed.model, "top_k", 1) or 1)]
        + [int(getattr(lp, "top_k", 0) or 0) for lp in parsed.model.layer_plans]
    )

    uniform_case = (
        replace(parsed, model=replace(parsed.model, expert_routing=(), expert_routing_source="uniform"))
        if parsed.model.expert_routing else parsed
    )
    base = analyze_network_efficiency(uniform_case, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, run_graph=False)
    balanced_step_ms, op_sum_ms = float(base["graph_duration_ms"]), float(base["operator_sum_ms"])
    scale = (balanced_step_ms / op_sum_ms) if op_sum_ms > 0.0 else 1.0

    routed_ops = [op for op in base["operators"] if _is_routed_moe_compute_operator(op)]
    alltoall_ops = [op for op in base["operators"] if _is_moe_alltoall_operator(op)]
    routed_total_ms = sum(float(op["total_ms"]) * scale for op in routed_ops)
    routed_var_ms = sum(max(0.0, float(op["duration_ms"]) - float(op["fixed_ms"])) * int(op["occurrences"]) * scale for op in routed_ops)
    alltoall_total_ms = sum(float(op["total_ms"]) * scale for op in alltoall_ops)

    alpha_max = float(ep) if routed_ops else 1.0
    e_local = max(1.0, float(moe_experts) / float(ep)) if moe_experts > 0 else 1.0
    rank_loads: list[float] | None = None
    eplb_rank_loads: list[float] | None = None

    def _eval_routing_ms(routing_weights: Sequence[float], fallback_alpha: float) -> float:
        if not routed_ops or moe_experts <= 0 or len(routing_weights) != moe_experts:
            return balanced_step_ms + (fallback_alpha - 1.0) * routed_var_ms
        if sum(1 for w in routing_weights if w > 0.0) < max_top_k:
            return balanced_step_ms + (fallback_alpha - 1.0) * routed_var_ms
        if all(math.isclose(w, routing_weights[0], rel_tol=1e-12) for w in routing_weights):
            return balanced_step_ms
        rc = replace(parsed, model=replace(parsed.model, expert_routing=tuple(float(w) for w in routing_weights), expert_routing_source="declared"))
        rep = analyze_network_efficiency(rc, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope, run_graph=False)
        return float(rep["graph_duration_ms"])

    eff_counts_in = expert_token_counts if (expert_token_counts is not None or not parsed.model.expert_routing or imbalance_in > 1.0) else parsed.model.expert_routing
    if eff_counts_in is not None:
        if not isinstance(eff_counts_in, Sequence) or isinstance(eff_counts_in, (str, bytes, Mapping)) or len(eff_counts_in) == 0:
            raise ValueError("skew_analysis_expert_token_counts_must_be_nonempty_sequence")
        if moe_experts > 0 and len(eff_counts_in) != moe_experts:
            raise ValueError(f"expert_token_counts_length_must_match_num_experts:{len(eff_counts_in)}!={moe_experts}")
        counts = [_validate_float("expert_token_count", c, minimum=0.0) for c in eff_counts_in]
        total_counts = sum(counts)
        if total_counts <= 0.0:
            raise ValueError("skew_analysis_expert_token_counts_sum_must_be_positive")
        expert_level_alpha = max(counts) / (total_counts / len(counts))
        rank_loads, eplb_rank_loads = _greedy_eplb_rank_loads(counts, ep, redundant_in)
        mean_rank = sum(rank_loads) / len(rank_loads)
        raw_alpha = (max(rank_loads) / mean_rank) if mean_rank > 0.0 else 1.0
        effective_alpha = min(max(1.0, float(raw_alpha)), alpha_max)
        if redundant_in > 0 and routed_ops:
            mean_eplb = sum(eplb_rank_loads) / len(eplb_rank_loads)
            eplb_alpha = min(max(1.0, (max(eplb_rank_loads) / mean_eplb) if mean_eplb > 0.0 else 1.0), effective_alpha)
        else:
            eplb_alpha = effective_alpha
        imbalanced_step_ms = _eval_routing_ms(counts, effective_alpha)
        if redundant_in > 0 and routed_ops and effective_alpha > 1.0 and imbalanced_step_ms > balanced_step_ms:
            frac = (eplb_alpha - 1.0) / (effective_alpha - 1.0)
            eplb_mitigated_step_ms = balanced_step_ms + frac * (imbalanced_step_ms - balanced_step_ms)
        else:
            eplb_mitigated_step_ms = imbalanced_step_ms
    else:
        effective_alpha = min(max(1.0, float(imbalance_in)), alpha_max)
        expert_level_alpha = effective_alpha
        eplb_alpha = (
            1.0 + (effective_alpha - 1.0) * (e_local / (e_local + float(redundant_in)))
            if (redundant_in > 0 and routed_ops) else effective_alpha
        )
        rem_imp = max(0.0, (float(ep) - effective_alpha) / float(max(1, ep - 1)))
        rem_eplb = max(0.0, (float(ep) - eplb_alpha) / float(max(1, ep - 1)))
        imbalanced_step_ms = _eval_routing_ms([effective_alpha if (e % ep == 0) else rem_imp for e in range(moe_experts)], effective_alpha)
        eplb_mitigated_step_ms = (
            _eval_routing_ms([eplb_alpha if (e % ep == 0) else rem_eplb for e in range(moe_experts)], eplb_alpha)
            if redundant_in > 0 else imbalanced_step_ms
        )
    extra_weight_bytes = _eplb_extra_weight_bytes_per_die(parsed, redundant_in)

    return {
        "schema": "mindie-sim.moe-imbalance/1",
        "effective_imbalance_factor": float(effective_alpha),
        "expert_level_imbalance_factor": float(expert_level_alpha),
        "eplb_effective_imbalance_factor": float(eplb_alpha),
        "rank_token_loads": rank_loads, "eplb_rank_token_loads": eplb_rank_loads,
        "balanced_step_ms": float(balanced_step_ms), "imbalanced_step_ms": float(imbalanced_step_ms),
        "eplb_mitigated_step_ms": float(eplb_mitigated_step_ms),
        "imbalance_slowdown_ratio": float(imbalanced_step_ms / max(balanced_step_ms, 1e-12)),
        "eplb_speedup_ratio": float(imbalanced_step_ms / max(eplb_mitigated_step_ms, 1e-12)),
        "eplb_extra_weight_bytes_per_die": float(extra_weight_bytes),
        "routed_moe_share_of_step": float(routed_total_ms / max(balanced_step_ms, 1e-12)),
        "routed_moe_ms": float(routed_total_ms), "alltoall_comm_ms": float(alltoall_total_ms),
        "eplb_redundant_experts": int(redundant_in),
    }


def _eval_mixed_chunk_on_engine(
    parsed: CaseSpec, *, decode_sequences: int, decode_context: int, c_tokens: int,
    prefill_context: int, scope: str, pure_decode_ms: float, dec_attn_ms: float,
) -> dict[str, Any]:
    """Lower and execute the mixed Decode + Prefill-chunk IterationPlan on the shared Engine."""
    total_tokens, total_seqs = decode_sequences + c_tokens, decode_sequences + 1
    eff_case = replace(
        parsed,
        runtime=replace(
            parsed.runtime,
            max_num_batched_tokens=max(parsed.runtime.max_num_batched_tokens, total_tokens),
            max_num_seqs=max(parsed.runtime.max_num_seqs, total_seqs),
        ),
    )
    eff_case, selection = effective_body_declaration(eff_case, "mixed", tokens=total_tokens, scope=scope, enabled=True)
    participants = cohorts(eff_case.runtime.parallel)[0]
    samples = scope == "model"
    views = {
        r: tuple(
            [RequestView(f"analysis{r}:{i}", 1, decode_context, samples, "decode") for i in range(decode_sequences)]
            + [RequestView(f"prefill_chunk{r}", c_tokens, prefill_context, False, "prefill", prompt_tokens=c_tokens)]
        )
        for r in participants
    }
    global_views = tuple(v for group in views.values() for v in group) if len(views) > 1 else None
    graphs = bind_cohort_traffic(
        eff_case,
        {r: _target_ops(eff_case, group, scope, global_views=global_views, phase="mixed", replica=r) for r, group in views.items()},
    )
    if scope == "body":
        graphs = TargetGraphSet(graphs, kernels=eff_case.runtime.kernels, selection=selection)
    run_case = consume_target_declaration(eff_case, graphs)
    ext_in = {"analysis_input": 0} if scope == "body" else None
    sg = {
        r: tuple(replace(op, metadata=tuple((k, v) for k, v in op.metadata if k not in _STRIP_METADATA_KEYS)) for op in ops)
        for r, ops in graphs.items()
    }
    if isinstance(graphs, TargetGraphSet):
        sg = TargetGraphSet(sg, kernels=graphs.effective_kernels, selection=graphs.default_selection)
    g_rep = graph_costs(run_case, sg, external_inputs=ext_in, include_submission=False, isolated=False)
    op_rep, _ = _fast_operator_and_graph_reports(
        run_case, phase="decode", tokens=total_tokens, context=decode_context,
        sequences=total_seqs, scope=scope, precomputed_graph=(graphs, run_case, g_rep),
    )
    mixed_graph_ms, mixed_op_sum = float(g_rep["duration_ms"]), float(op_rep["operator_sum_ms"])
    mixed_scale = (mixed_graph_ms / mixed_op_sum) if mixed_op_sum > 0.0 else 1.0
    mixed_attn_ops = [r for r in op_rep["operators"] if _is_attention_core_operator(r["operator"])]
    mixed_attn_ms = sum(float(r["total_ms"]) * mixed_scale for r in mixed_attn_ops)
    mixed_non_attn_ms = max(0.0, mixed_graph_ms - mixed_attn_ms)

    pre_rep = analyze_network_efficiency(
        parsed, phase="prefill", tokens=c_tokens, context=prefill_context, sequences=1, scope=scope, run_graph=False
    )
    pre_step_ms, pre_op_sum = float(pre_rep["graph_duration_ms"]), float(pre_rep["operator_sum_ms"])
    pre_scale = (pre_step_ms / pre_op_sum) if pre_op_sum > 0.0 else 1.0
    pre_attn_ms = sum(float(op["total_ms"]) * pre_scale for op in pre_rep["operators"] if _is_attention_core_operator(op))

    hybrid_ms = mixed_graph_ms
    return {
        "standalone_prefill_step_ms": float(pre_step_ms), "hybrid_step_ms": float(hybrid_ms),
        "decode_attention_ms": float(dec_attn_ms), "prefill_attention_ms": float(pre_attn_ms),
        "shared_non_attention_ms": float(mixed_non_attn_ms),
    }


def analyze_chunked_prefill_decode_interference(
    case: CaseSpec | Mapping[str, Any], *, decode_sequences: int = 16, decode_context: int = 32768,
    prefill_chunk_tokens: Sequence[int] = (512, 1024, 2048, 4096, 8192),
    prefill_context: int = 0, sla_tpot_ms: float = 30.0, scope: str = "body",
) -> dict[str, Any]:
    """Analyze Decode ITL/TPOT inflation from piggybacked chunked-prefill tokens."""
    _validate_phase_and_scope("decode", scope)
    _validate_int("decode_sequences", decode_sequences, minimum=1)
    _validate_int("decode_context", decode_context, minimum=0)
    _validate_int("prefill_context", prefill_context, minimum=0)
    tpot_sla = _validate_float("sla_tpot_ms", sla_tpot_ms, minimum=0.0, strict_gt=True)
    if not isinstance(prefill_chunk_tokens, Sequence) or isinstance(prefill_chunk_tokens, (str, bytes, Mapping)) or len(prefill_chunk_tokens) == 0:
        raise ValueError("skew_analysis_prefill_chunk_tokens_must_be_nonempty_sequence")
    chunks_in = tuple(_validate_int("prefill_chunk_token", c, minimum=0) for c in prefill_chunk_tokens)

    parsed = parse_case(case)
    dec_rep = analyze_network_efficiency(parsed, phase="decode", tokens=1, context=decode_context, sequences=decode_sequences, scope=scope)
    pure_decode_ms, dec_op_sum = float(dec_rep["graph_duration_ms"]), float(dec_rep["operator_sum_ms"])
    dec_scale = (pure_decode_ms / dec_op_sum) if dec_op_sum > 0.0 else 1.0
    dec_attn_ms = sum(float(op["total_ms"]) * dec_scale for op in dec_rep["operators"] if _is_attention_core_operator(op))

    chunk_cache: dict[int, dict[str, Any]] = {}
    chunk_rows: list[dict[str, Any]] = []

    for c_tokens in chunks_in:
        if c_tokens in chunk_cache:
            chunk_rows.append(dict(chunk_cache[c_tokens]))
            continue
        if c_tokens == 0:
            row = {
                "prefill_chunk_tokens": 0, "standalone_prefill_step_ms": 0.0,
                "hybrid_step_ms": float(pure_decode_ms), "decode_attention_ms": float(dec_attn_ms),
                "prefill_attention_ms": 0.0, "shared_non_attention_ms": float(max(0.0, pure_decode_ms - dec_attn_ms)),
                "decode_itl_inflation_ms": 0.0, "decode_itl_inflation_ratio": 1.0,
                "meets_tpot_sla": bool(pure_decode_ms <= tpot_sla),
            }
            chunk_cache[0] = row
            chunk_rows.append(dict(row))
            continue

        metrics = _eval_mixed_chunk_on_engine(
            parsed, decode_sequences=decode_sequences, decode_context=decode_context,
            c_tokens=c_tokens, prefill_context=prefill_context, scope=scope,
            pure_decode_ms=pure_decode_ms, dec_attn_ms=dec_attn_ms,
        )
        hybrid_ms = metrics["hybrid_step_ms"]
        row = {
            "prefill_chunk_tokens": int(c_tokens), **metrics,
            "decode_itl_inflation_ms": float(hybrid_ms - pure_decode_ms),
            "decode_itl_inflation_ratio": float(hybrid_ms / max(pure_decode_ms, 1e-12)),
            "meets_tpot_sla": bool(hybrid_ms <= tpot_sla),
        }
        chunk_cache[c_tokens] = row
        chunk_rows.append(dict(row))

    compliant = [r["prefill_chunk_tokens"] for r in chunk_rows if r["meets_tpot_sla"]]
    return {
        "schema": "mindie-sim.chunked-prefill-interference/1",
        "pure_decode_step_ms": float(pure_decode_ms),
        "sla_tpot_ms": float(tpot_sla),
        "max_sla_compliant_chunk_tokens": int(max(compliant) if compliant else 0),
        "chunks": chunk_rows,
    }
