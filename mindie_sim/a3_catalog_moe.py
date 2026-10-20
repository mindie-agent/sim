"""MoE lowering with router, latent transform, dispatch, experts and shared branch.

[Role]
  Lower sparse FFNs for DeepSeek/GLM/Qwen and Kimi's latent MoE.  Separate the
  full-width router and shared branch from the routed latent path, and keep
  dispatch/combine payloads at the actually selected activation width.

[Invariants]
  * Kimi routes on H7168, dispatches 3584-wide latent rows, runs experts on
    [3584,3072], combines at 3584, normalizes, then projects back to 7168.
    No H7168 EP payload is charged unless a physical packing declares it.
  * Shared experts consume the same full-width hidden state independently of
    the routed combine.  A final add is the only join.
  * Router weights/scoring stay in their native dtype; expert FP4 scales are
    packed explicitly and experts are counted before EP/ETP sharding.
  * The local rank's expert counts/padding come from the declared routing
    distribution; no learned routing value is invented.
  * Dispatch and combine are separate AllToAllV directions with the same
    logical activation width; scales/count metadata are separate.

[Timeouts & Locks]
  Pure arithmetic bounded by local experts/groups/requests; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .a3_catalog_ops import (
    barrier_op, comm_op, gemm_op, packed_weight_bytes, vector_op,
)
from .a3_catalog_types import QuantSpec
from .a3_model_types import LayerPlan, ModelSpec
from .a3_ops import OpSpec, RoutingStats
from .a3_types import A3InputError, ParallelSpec, align_up, dtype_bytes


def _requests(views: Sequence[Any]) -> tuple[str, ...]:
    return tuple(view.request_id for view in views)


def _experts_on_rank(num_experts: int, ep: int, rank: int, local: int) -> list[int]:
    experts = [expert for expert in range(num_experts) if expert % ep == rank]
    return experts[:local]


def routing_stats(
    model: ModelSpec, plan: LayerPlan, parallel: ParallelSpec,
    padded_tokens: int, *, capacity_factor: float = 1.0,
) -> RoutingStats:
    global_scale = float(parallel.dp) if parallel.ep_spans_dp else 1.0
    routing_tokens = float(padded_tokens) * plan.top_k * global_scale
    local_experts = max(1, plan.num_experts // max(1, parallel.ep))
    distribution = model.expert_routing
    if not distribution:
        distribution = tuple(1.0 / plan.num_experts for _ in range(plan.num_experts))
    capacity = max(1.0, math.ceil(
        routing_tokens / max(1, plan.num_experts) * max(0.1, capacity_factor)
    ))
    worst = 0.0
    local_padded = 0.0
    dropped = 0.0
    counts: tuple[int, ...] = ()
    for rank in range(max(1, parallel.ep)):
        experts = _experts_on_rank(plan.num_experts, parallel.ep, rank, local_experts)
        rank_tokens = 0.0
        rank_padded = 0.0
        rank_counts: list[int] = []
        for expert in experts:
            expected = routing_tokens * distribution[expert]
            kept = min(expected, capacity)
            dropped += max(0.0, expected - kept) / max(1, parallel.ep)
            rank_tokens += kept
            padded = align_up(int(math.ceil(kept)), 32)
            rank_counts.append(padded)
            rank_padded += padded
        if rank_padded >= local_padded:
            counts = tuple(rank_counts)
        worst = max(worst, rank_tokens)
        local_padded = max(local_padded, rank_padded)
    return RoutingStats(
        routing_tokens=routing_tokens, local_experts=local_experts,
        slowest_rank_tokens=worst, local_padded_tokens=local_padded,
        capacity_per_expert=capacity, dropped_tokens=dropped,
        distribution=model.expert_routing_source,
        local_routing_tokens=float(padded_tokens) * plan.top_k,
        global_scale=global_scale, local_expert_counts=counts,
    )


def lower_moe(
    *, prefix: str, plan: LayerPlan, stage: int, deps: str, model: ModelSpec,
    parallel: ParallelSpec, quant: QuantSpec, mode: str, padded_tokens: int,
    views: Sequence[Any], routing: RoutingStats,
) -> tuple[list[OpSpec], str]:
    if plan.num_experts <= 0 or plan.top_k <= 0:
        raise A3InputError(f"catalog_moe_dims_missing:layer{plan.index}")
    ops: list[OpSpec] = []
    requests = _requests(views)
    h = model.hidden_size
    router_in = plan.router_input_size or h
    ep = max(1, parallel.ep)
    etp = max(1, parallel.etp)
    local_experts = max(1, plan.num_experts // ep)
    expert_width = max(1, plan.moe_intermediate_size // etp)
    routed_input = plan.expert_latent_size or h
    routed_output = plan.expert_latent_size or h
    # ---- router ----
    if plan.router_mode == "hash":
        hash_lookup = vector_op(
            op_id=f"{prefix}.hash_router", deps=(deps,), stage=stage,
            layer=plan.index, requests=requests,
            elements=max(1, padded_tokens * plan.top_k), ops_per_element=2.0,
            dtype="fp32", family="hash_router", phase=mode,
            recipe="catalog_hash_router", semantic_family="hash_expert_lookup",
            shape=(padded_tokens, plan.top_k),
            metadata=(("router_mode", "hash"),),
        )
        ops.append(hash_lookup)
        router_ready = hash_lookup.op_id
    else:
        router = gemm_op(
            op_id=f"{prefix}.router", deps=(deps,), stage=stage, layer=plan.index,
            requests=requests, m=padded_tokens, n=plan.num_experts, k=router_in,
            quant=quant, category="router", phase=mode, family="gemm",
            semantic_family="moe_router_full_width",
            metadata=(("router_input_width", router_in),
                      ("routing_distribution", model.expert_routing_source),
                      ("scoring_func", plan.router_mode)),
        )
        ops.append(router)
        score = vector_op(
            op_id=f"{prefix}.router_score", deps=(router.op_id,), stage=stage,
            layer=plan.index, requests=requests,
            elements=max(1, padded_tokens * plan.num_experts),
            ops_per_element=5.0, dtype="fp32", family="router_softmax",
            phase=mode, recipe="catalog_moe_router_score",
            semantic_family="moe_router_topk",
            shape=(padded_tokens, plan.num_experts),
            metadata=(("top_k", plan.top_k),
                      ("scoring_func", plan.router_mode),
                      ("normalize_topk", bool(getattr(plan, "router_norm_topk", False))),
                      ("router_groups", int(getattr(plan, "router_groups", 0))),
                      ("router_group_topk", int(getattr(plan, "router_group_topk", 0))),
                      ("router_contract", (
                          "sigmoid_topk" if plan.router_mode == "sigmoid"
                          else "softmax_topk" if plan.router_mode == "softmax"
                          else f"{plan.router_mode}_topk"
                      )),
                      ("capacity_per_expert", routing.capacity_per_expert),
                      ("dropped_tokens", routing.dropped_tokens),
                      ("local_padded_tokens", routing.local_padded_tokens)),
        )
        ops.append(score)
        router_ready = score.op_id
    if plan.shared_expert_gate:
        shared_gate = gemm_op(
            op_id=f"{prefix}.shared_router", deps=(deps,), stage=stage,
            layer=plan.index, requests=requests, m=padded_tokens, n=1, k=h,
            quant=quant, category="router", phase=mode, family="gemm",
            semantic_family="shared_expert_gate",
        )
        shared_gate_act = vector_op(
            op_id=f"{prefix}.shared_gate", deps=(shared_gate.op_id,), stage=stage,
            layer=plan.index, requests=requests, elements=max(1, padded_tokens),
            ops_per_element=4.0, dtype="fp32", family="router_softmax", phase=mode,
            recipe="catalog_shared_router",
        )
        ops.extend([shared_gate, shared_gate_act])
        shared_gate_dep = shared_gate_act.op_id
    else:
        shared_gate_dep = None
    # ---- routed latent transform ----
    latent_deps = router_ready
    if plan.expert_latent_size:
        down = gemm_op(
            op_id=f"{prefix}.latent_down", deps=(deps,), stage=stage,
            layer=plan.index, requests=requests, m=padded_tokens,
            n=plan.expert_latent_size, k=h, quant=quant,
            category="dense_mlp", phase=mode, family="gemm",
            semantic_family="latent_down_projection",
            metadata=(("routed_latent_width", plan.expert_latent_size),),
        )
        latent_norm = vector_op(
            op_id=f"{prefix}.latent_norm", deps=(down.op_id,), stage=stage,
            layer=plan.index, requests=requests,
            elements=max(1, padded_tokens * plan.expert_latent_size),
            ops_per_element=5.0, dtype=model.activation_dtype, family="rmsnorm",
            phase=mode, recipe="catalog_latent_norm",
            shape=(padded_tokens, plan.expert_latent_size),
            metadata=(("tokens", padded_tokens),
                      ("reduction_width", plan.expert_latent_size),
                      ("norm_type", "rmsnorm"), ("norm_eps", model.norm_eps)),
        )
        control = barrier_op(
            op_id=f"{prefix}.routed_control", deps=(router_ready, latent_norm.op_id),
            stage=stage, layer=plan.index, requests=requests, phase=mode,
            recipe="catalog_routing_control_dependency",
            shape=(max(1, padded_tokens), plan.expert_latent_size),
            metadata=(("routing_dependency", True),
                      ("data_dependency", "latent_norm"),
                      ("control_dependency", "router_score")),
        )
        ops.extend([down, latent_norm, control])
        latent_deps = control.op_id
    dispatch_last = latent_deps
    if ep > 1:
        # Dispatch carries the routed activation after latent down/norm.
        link = routing.routing_tokens * routed_input * dtype_bytes(model.activation_dtype) \
            * (ep - 1) / ep
        dispatch = comm_op(
            op_id=f"{prefix}.ep_dispatch", deps=(latent_deps,), stage=stage,
            layer=plan.index, requests=requests, family="ep_dispatch",
            phase=mode, recipe="catalog_alltoallv_dispatch",
            link_bytes=link,
            shape=(padded_tokens, plan.top_k, routed_input),
            metadata=(("payload_width", routed_input),
                      ("logical_dispatch_width", routed_input),
                      ("payload_semantics", "routed activation, not residual H"),
                      ("scales_exchanged_separately", True)),
        )
        ops.append(dispatch)
        dispatch_last = dispatch.op_id
    # ---- local expert GEMMs ----
    counts = routing.local_expert_counts or tuple(
        max(1, int(math.ceil(routing.local_padded_tokens / local_experts)))
        for _ in range(local_experts)
    )
    if len(counts) < local_experts:
        counts = tuple(counts) + (1,) * (local_experts - len(counts))
    group_size = max(1, math.ceil(local_experts / 2))
    down_ids: list[str] = []
    for start in range(0, local_experts, group_size):
        size = min(group_size, local_experts - start)
        group_counts = counts[start:start + size]
        m = max(1, sum(group_counts))
        gate_weight = 2 * size * packed_weight_bytes(
            quant, expert_width, routed_input, "routed_expert", count=1,
        )
        gate = gemm_op(
            op_id=f"{prefix}.expert_gate_up{start}", deps=(dispatch_last,),
            stage=stage, layer=plan.index, requests=requests, m=m,
            n=2 * expert_width, k=routed_input, quant=quant,
            category="routed_expert", phase=mode, family="moe_gate_up",
            semantic_family="grouped_expert_gate_up",
            weight_bytes_override=gate_weight,
            metadata=(("local_expert_start", start), ("group_size", size),
                      ("expert_tokens", tuple(group_counts)),
                      ("local_experts", local_experts),
                      ("etp", etp), ("routed_width", routed_input)),
        )
        act_elements = max(1, m * expert_width)
        act_bytes = dtype_bytes(model.activation_dtype)
        activation = vector_op(
            op_id=f"{prefix}.expert_act{start}", deps=(gate.op_id,), stage=stage,
            layer=plan.index, requests=requests,
            elements=act_elements, ops_per_element=4.0,
            dtype=model.activation_dtype, family="swiglu", phase=mode,
            recipe="catalog_expert_swiglu",
            semantic_family="expert_gate_up_activation",
            shape=(m, expert_width),
            read_bytes_override=2.0 * act_elements * act_bytes,
            write_bytes_override=1.0 * act_elements * act_bytes,
            metadata=(("read_halves", 2), ("write_outputs", 1),
                      ("packed_gate_input", True)),
        )
        down_weight = size * packed_weight_bytes(
            quant, routed_output, expert_width, "routed_expert",
        )
        down = gemm_op(
            op_id=f"{prefix}.expert_down{start}", deps=(activation.op_id,),
            stage=stage, layer=plan.index, requests=requests, m=m,
            n=routed_output, k=expert_width, quant=quant,
            category="routed_expert", phase=mode, family="moe_down",
            semantic_family="grouped_expert_down",
            weight_bytes_override=down_weight,
            metadata=(("local_expert_start", start), ("group_size", size),
                      ("routed_width", routed_output)),
        )
        ops.extend([gate, activation, down])
        down_ids.append(down.op_id)
    combine_dep = tuple(down_ids)
    if ep > 1:
        link = routing.routing_tokens * routed_output * dtype_bytes(model.activation_dtype) \
            * (ep - 1) / ep
        combine = comm_op(
            op_id=f"{prefix}.ep_combine", deps=tuple(down_ids), stage=stage,
            layer=plan.index, requests=requests, family="ep_combine",
            phase=mode, recipe="catalog_alltoallv_combine",
            link_bytes=link,
            shape=(padded_tokens, routed_output),
            metadata=(("payload_width", routed_output),),
        )
        ops.append(combine)
        combine_dep = (combine.op_id,)
    routed_last = _join(ops, f"{prefix}.routed_join", combine_dep, stage,
                        plan.index, requests, mode)
    if parallel.tp > 1 and (parallel.etp > 1 or parallel.ep <= 1):
        # Expert tensor shards are partial sums.  Reduce at the routed width
        # BEFORE the output RMSNorm: norm(sum(x)) != sum(norm(x)).
        routed_reduce = comm_op(
            op_id=f"{prefix}.routed_reduce", deps=(routed_last,), stage=stage,
            layer=plan.index, requests=requests, family="allreduce",
            phase=mode, recipe="catalog_routed_etp_allreduce",
            link_bytes=2.0 * (parallel.tp - 1) / parallel.tp * padded_tokens
            * routed_output * dtype_bytes(model.activation_dtype),
            shape=(padded_tokens, routed_output),
            metadata=(("collective", "allreduce"),
                      ("reduce_before_norm", True),
                      ("reduce_width", routed_output)),
        )
        ops.append(routed_reduce)
        routed_last = routed_reduce.op_id
    if plan.expert_latent_size:
        out_norm = vector_op(
            op_id=f"{prefix}.latent_out_norm", deps=(routed_last,), stage=stage,
            layer=plan.index, requests=requests,
            elements=max(1, padded_tokens * plan.expert_latent_size),
            ops_per_element=5.0, dtype=model.activation_dtype, family="rmsnorm",
            phase=mode, recipe="catalog_latent_out_norm",
            semantic_family="routed_output_norm_before_up",
            shape=(padded_tokens, plan.expert_latent_size),
            metadata=(("tokens", padded_tokens),
                      ("reduction_width", plan.expert_latent_size),
                      ("norm_type", "rmsnorm"), ("norm_eps", model.norm_eps)),
        )
        up = gemm_op(
            op_id=f"{prefix}.latent_up", deps=(out_norm.op_id,), stage=stage,
            layer=plan.index, requests=requests, m=padded_tokens, n=h,
            k=plan.expert_latent_size, quant=quant, category="dense_mlp",
            phase=mode, family="gemm",
            semantic_family="latent_up_projection",
            metadata=(("routed_latent_width", plan.expert_latent_size),),
        )
        ops.extend([out_norm, up])
        routed_last = up.op_id
    # ---- shared branch remains at full H independently ----
    shared_last = None
    shared_i = plan.shared_intermediate_size or (
        plan.num_shared_experts * plan.moe_intermediate_size
    )
    if plan.num_shared_experts and shared_i > 0:
        shared_width = max(1, shared_i // parallel.tp)
        shared_weight = 2 * packed_weight_bytes(
            quant, shared_i, h, "shared_expert",
        )  # informational full-width weight; shard below
        shared = gemm_op(
            op_id=f"{prefix}.shared_gate_up", deps=(deps,), stage=stage,
            layer=plan.index, requests=requests, m=padded_tokens,
            n=2 * shared_width, k=h, quant=quant, category="shared_expert",
            phase=mode, family="shared_expert_gate_up",
            semantic_family="shared_expert_full_width",
            metadata=(("shared_input_width", h),
                      ("shared_intermediate", shared_i),
                      ("num_shared_experts", plan.num_shared_experts)),
        )
        shared_elements = max(1, padded_tokens * shared_width)
        shared_bytes = dtype_bytes(model.activation_dtype)
        shared_act = vector_op(
            op_id=f"{prefix}.shared_act", deps=(shared.op_id,), stage=stage,
            layer=plan.index, requests=requests,
            elements=shared_elements, ops_per_element=4.0,
            dtype=model.activation_dtype, family="swiglu", phase=mode,
            recipe="catalog_shared_swiglu",
            shape=(padded_tokens, shared_width),
            read_bytes_override=2.0 * shared_elements * shared_bytes,
            write_bytes_override=1.0 * shared_elements * shared_bytes,
            metadata=(("tokens", padded_tokens),
                      ("read_halves", 2), ("write_outputs", 1),
                      ("packed_gate_input", True),
                      ("output_width", shared_width),
                      ("input_width", 2 * shared_width),
                      ("packed_gate_input", True),
                      ("activation", "silu"),
                      ("swiglu_contract", "packed_gate_up")),
        )
        shared_down = gemm_op(
            op_id=f"{prefix}.shared_down", deps=(shared_act.op_id,), stage=stage,
            layer=plan.index, requests=requests, m=padded_tokens, n=h,
            k=shared_width, quant=quant, category="shared_expert", phase=mode,
            family="shared_expert_down", semantic_family="shared_expert_down",
            metadata=(("shared_input_width", h),),
        )
        ops.extend([shared, shared_act, shared_down])
        shared_last = shared_down.op_id
        if parallel.tp > 1:
            shared_reduce = comm_op(
                op_id=f"{prefix}.shared_reduce", deps=(shared_last,), stage=stage,
                layer=plan.index, requests=requests, family="allreduce",
                phase=mode, recipe="catalog_shared_tp_allreduce",
                link_bytes=2.0 * (parallel.tp - 1) / parallel.tp * padded_tokens
                * h * dtype_bytes(model.activation_dtype),
                shape=(padded_tokens, h),
                metadata=(("collective", "allreduce"),
                          ("shared_branch_independent", True)),
            )
            ops.append(shared_reduce)
            shared_last = shared_reduce.op_id
        if shared_gate_dep is not None:
            gated = vector_op(
                op_id=f"{prefix}.shared_gated", deps=(shared_last, shared_gate_dep),
                stage=stage, layer=plan.index, requests=requests,
                elements=max(1, padded_tokens * h), ops_per_element=2.0,
                dtype=model.activation_dtype, family="elementwise", phase=mode,
                recipe="catalog_shared_router_gate",
                semantic_family="shared_expert_output_gate",
            )
            ops.append(gated)
            shared_last = gated.op_id
    # ---- join routed + shared at H ----
    join_deps = (routed_last,) + ((shared_last,) if shared_last else ())
    final = vector_op(
        op_id=f"{prefix}.moe_add", deps=join_deps, stage=stage,
        layer=plan.index, requests=requests,
        elements=max(1, padded_tokens * h), ops_per_element=2.0,
        dtype=model.activation_dtype, family="elementwise", phase=mode,
        recipe="catalog_moe_residual_join",
        semantic_family="shared_plus_routed_add_at_hidden",
        metadata=(("join_width", h),
                  ("shared_and_routed_independent", True)),
    )
    ops.append(final)
    return ops, final.op_id


def _join(
    ops: list[OpSpec], op_id: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], mode: str,
) -> str:
    if len(deps) == 1:
        return deps[0]
    barrier = barrier_op(
        op_id=op_id, deps=deps, stage=stage, layer=layer,
        requests=requests, phase=mode, recipe="catalog_expert_group_join",
        shape=(1,), metadata=(("join_kind", "all_local_expert_groups"),),
    )
    ops.append(barrier)
    return barrier.op_id


__all__ = ["lower_moe", "routing_stats"]
