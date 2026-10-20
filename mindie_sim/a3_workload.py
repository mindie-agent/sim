"""Semantic shape expansion and recipe selection for A3 serving iterations.

[Role]
  Turn an ``IterationPlan`` of request query/context lengths into explicit
  semantic, rank-local, padded and launch shapes; derive FLOPs/bytes/vector
  work and the dependency DAG for Dense/GQA/MLA and Dense/MoE FFNs; select
  graph/KV/MoE recipes through feature guards.  Also expose the per-die memory
  plan used to filter OOM plans.

[Invariants]
  * ``clamp(flops, 0)`` or a fixed efficiency never replaces a missing shape.
    Every op carries its four shapes and the formula that produced its work.
  * Attention work uses query length and KV length independently.  A one-token
    query against a long context is never converted to ``context = 1``.
  * The MoE routing histogram changes local expert padding and the slowest EP
    rank.  EP is a sub-group of TP, so ``tp * ep`` is not a world size.
  * Graph padding adds physical launch work but never increases valid output
    tokens.  Recipe guards are explicit and can be traced.
  * Memory is per die and per PP stage; package totals are not mixed into it.

[Timeouts & Locks]
  Pure arithmetic with loops bounded by layers/experts/requests.  No I/O,
  locks, subprocess, device probe or sleep.

[Side-Effect Policy]
  No filesystem/network/global mutation.  Returns new immutable records.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .a3_types import (
    A3InputError, MachineSpec, ModelSpec, ParallelSpec, RequestSpec, RuntimeSpec,
    WorkloadSpec, align_up, dtype_bytes,
)

DEFAULT_TOKEN_ALIGN = 32
DEFAULT_HEAD_ALIGN = 16


@dataclass(frozen=True)
class RequestView:
    request_id: str
    q_len: int
    context: int
    samples: bool
    phase: str
    prefix_tokens: int = 0
    priority: int = 0
    padding: bool = False
    semantic_phase: str = ""
    execution_mode: str = ""
    media_encoded: bool = False
    draft_verify: bool = True
    draft_proposal_tokens: int = 0
    draft_accepted_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.semantic_phase:
            object.__setattr__(self, "semantic_phase", self.phase)
        if not self.execution_mode:
            object.__setattr__(self, "execution_mode", self.phase)

    @property
    def kv_len(self) -> int:
        return self.context + self.q_len

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "q_len": self.q_len,
            "context": self.context, "kv_len": self.kv_len,
            "samples": self.samples, "phase": self.phase,
            "semantic_phase": self.semantic_phase,
            "execution_mode": self.execution_mode,
            "prefix_tokens": self.prefix_tokens, "padding": self.padding,
            "media_encoded": self.media_encoded,
            "draft_verify": self.draft_verify,
            "draft_proposal_tokens": self.draft_proposal_tokens,
            "draft_accepted_tokens": self.draft_accepted_tokens,
        }


@dataclass(frozen=True)
class IterationPlan:
    requests: tuple[RequestView, ...]
    iteration: int = 0
    reason: str = "scheduled"

    @property
    def valid_tokens(self) -> int:
        return sum(view.q_len for view in self.requests)

    @property
    def modes(self) -> set[str]:
        return {view.phase for view in self.requests}

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "requests": [view.to_dict() for view in self.requests],
            "valid_tokens": self.valid_tokens,
            "mode": selected_mode(self),
        }


def selected_mode(plan: IterationPlan) -> str:
    if not plan.requests:
        return "idle"
    modes = plan.modes
    if modes == {"decode"}:
        return "decode"
    if modes == {"prefill"}:
        return "prefill"
    return "mixed"


from .a3_ops import (  # noqa: E402  (re-exported public records/constructors)
    IterationWork, OpSpec, RoutingStats, _gemm_op, _op, _vec_op,
)


def _swiglu_op(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], tokens: int, width: int, dtype: str,
    phase: str, recipe: str,
    metadata: tuple[tuple[str, Any], ...] = (),
) -> OpSpec:
    """Packed gate/up -> SwiGLU -> down activation work.

    ``width`` is the local intermediate width, so the input is ``2*width`` and
    the shape-changing output is ``tokens x width``.  Work is the same generic
    vector convention catalog uses (output elements, four vector ops each).
    """
    tokens = max(1, int(tokens))
    width = max(1, int(width))
    elements = tokens * width
    bytes_per = dtype_bytes(dtype)
    shape = (tokens, width)
    return _op(
        op_id=op_id, family="swiglu", phase=phase, stage=stage, layer=layer,
        requests=requests, semantic_shape=shape, local_shape=shape,
        padded_shape=(align_up(elements, 256),),
        launch_shape=(align_up(elements, 256),),
        vector_ops=float(elements) * 4.0,
        # Packed gate/up is read (2I), SwiGLU output is written (I).
        hbm_bytes=float(elements) * 3.0 * bytes_per,
        mte2_bytes=float(elements) * 2.0 * bytes_per,
        mte3_bytes=float(elements) * bytes_per,
        deps=deps, recipe=recipe,
        notes=("vector_work=elements*ops_per_element",),
        dtype=dtype,
        metadata=metadata + (
            ("tokens", tokens), ("input_width", 2 * width),
            ("output_width", width), ("packed_gate_input", True),
            ("activation", "silu"), ("swiglu_contract", "packed_gate_up"),
        ),
    )


def _attention_pair(
    *, prefix: str, view: RequestView, stage: int, layer: int, model: ModelSpec,
    parallel: ParallelSpec, runtime: RuntimeSpec, deps: tuple[str, ...], recipe: str,
) -> tuple[list[OpSpec], str]:
    dtype = model.activation_dtype
    local_heads = max(1, model.num_heads // parallel.tp)
    local_kv_heads = model.local_kv_heads(parallel.tp)
    context = max(0, view.context)
    q_len = max(1, view.q_len)
    if view.padding:
        pairs = q_len * (context + 1)
        q_effective = q_len
    elif q_len == 1:
        pairs = context + 1
        q_effective = 1
    else:
        # Causal chunk: prior context plus the triangular part inside this chunk.
        pairs = q_len * context + q_len * (q_len + 1) / 2.0
        q_effective = q_len
    local_pairs = max(1.0, pairs / max(1, parallel.dcp))
    if model.attention == "gqa":
        qk_dim = model.head_dim
        v_dim = model.head_dim
        kv_bytes_per_token = 2 * local_kv_heads * model.head_dim * dtype_bytes(model.kv_dtype)
    else:
        assert model.mla is not None
        qk_dim = model.mla.qk_head_dim
        v_dim = model.mla.v_head_dim
        kv_bytes_per_token = (model.mla.kv_lora_rank + model.mla.qk_rope_head_dim) * dtype_bytes(model.kv_dtype)
    # ``pairs`` already sums over all query rows.  Do not multiply by q again.
    qk_flops = 2.0 * local_heads * local_pairs * qk_dim
    pv_flops = 2.0 * local_heads * local_pairs * v_dim
    q_bytes = q_effective * local_heads * qk_dim * dtype_bytes(dtype)
    kv_read = local_pairs * kv_bytes_per_token
    out_bytes = q_effective * local_heads * v_dim * dtype_bytes(dtype)
    local_kv_len = max(1, math.ceil(view.kv_len / max(1, parallel.dcp)))
    probability_bytes = q_effective * local_heads * local_kv_len * 2.0
    qk_bytes = q_bytes + kv_read + probability_bytes
    pv_bytes = probability_bytes + kv_read + out_bytes
    op_qk = _op(
        op_id=f"{prefix}.qk", family="attention_qk", phase=view.phase, stage=stage,
        layer=layer, requests=(view.request_id,),
        semantic_shape=(q_effective, view.kv_len, local_heads, qk_dim),
        local_shape=(q_effective, max(1, math.ceil(view.kv_len / parallel.dcp)), local_heads, qk_dim),
        padded_shape=(align_up(q_effective, DEFAULT_HEAD_ALIGN),
                      align_up(int(math.ceil(view.kv_len / parallel.dcp)), DEFAULT_HEAD_ALIGN),
                      local_heads, qk_dim),
        launch_shape=(q_effective, max(1, math.ceil(view.kv_len / parallel.dcp)), local_heads, qk_dim),
        flops=qk_flops, hbm_bytes=qk_bytes, mte2_bytes=kv_read + q_bytes,
        mte3_bytes=probability_bytes, deps=deps,
        recipe=f"{recipe}:qk", notes=("attention_pairs=context+q(q+1)/2",),
        dtype=dtype,
    )
    op_softmax = _vec_op(
        op_id=f"{prefix}.softmax", deps=(op_qk.op_id,), stage=stage, layer=layer,
        requests=(view.request_id,),
        elements=max(1, int(q_effective * local_heads * local_kv_len)),
        ops_per_element=5.0, dtype="fp32", family="attention_softmax",
        phase=view.phase, recipe=f"{recipe}:softmax",
        shape=(q_effective, local_heads, local_kv_len),
    )
    op_pv = _op(
        op_id=f"{prefix}.pv", family="attention_pv", phase=view.phase, stage=stage,
        layer=layer, requests=(view.request_id,),
        semantic_shape=(q_effective, view.kv_len, local_heads, v_dim),
        local_shape=(q_effective, max(1, math.ceil(view.kv_len / parallel.dcp)), local_heads, v_dim),
        padded_shape=(align_up(q_effective, DEFAULT_HEAD_ALIGN),
                      align_up(int(math.ceil(view.kv_len / parallel.dcp)), DEFAULT_HEAD_ALIGN),
                      local_heads, v_dim),
        launch_shape=(q_effective, max(1, math.ceil(view.kv_len / parallel.dcp)), local_heads, v_dim),
        flops=pv_flops, hbm_bytes=pv_bytes,
        mte2_bytes=probability_bytes + kv_read,
        mte3_bytes=out_bytes, deps=(op_softmax.op_id,),
        recipe=f"{recipe}:pv", notes=("attention_pv_formula=2*Q*KV*H*Vdim",),
        dtype=dtype,
    )
    return [op_qk, op_softmax, op_pv], op_pv.op_id


def _routing_stats(
    model: ModelSpec, runtime: RuntimeSpec, parallel: ParallelSpec, padded_tokens: int,
    *, ep_global_scale: float = 1.0,
) -> RoutingStats:
    local_routing_tokens = float(padded_tokens) * model.top_k
    routing_tokens = local_routing_tokens * max(1.0, float(ep_global_scale))
    local_experts = max(1, model.num_experts // parallel.ep)
    distribution = model.expert_routing or tuple(1.0 / model.num_experts for _ in range(model.num_experts))
    capacity_per_expert = max(
        1.0, math.ceil((routing_tokens / max(1, model.num_experts)) * runtime.moe_capacity_factor)
    )
    dropped = 0.0
    worst = 0.0
    local_padded = 0.0
    best_counts: tuple[int, ...] = ()
    for ep_rank in range(parallel.ep):
        experts = list(range(ep_rank, model.num_experts, parallel.ep))[:local_experts]
        rank_tokens = 0.0
        rank_padded = 0.0
        counts: list[int] = []
        for expert in experts:
            count = routing_tokens * distribution[expert]
            kept = min(count, capacity_per_expert)
            dropped += max(0.0, count - kept) / max(1, parallel.ep)
            rank_tokens += kept
            padded = align_up(int(math.ceil(kept)), DEFAULT_TOKEN_ALIGN)
            counts.append(padded)
            rank_padded += padded
        if rank_padded >= local_padded:
            best_counts = tuple(counts)
        worst = max(worst, rank_tokens)
        local_padded = max(local_padded, rank_padded)
    return RoutingStats(
        routing_tokens=routing_tokens,
        local_experts=local_experts,
        slowest_rank_tokens=worst,
        local_padded_tokens=local_padded,
        capacity_per_expert=capacity_per_expert,
        dropped_tokens=dropped,
        distribution=model.expert_routing_source,
        local_routing_tokens=local_routing_tokens,
        global_scale=max(1.0, float(ep_global_scale)),
        local_expert_counts=best_counts,
    )


def select_recipe(
    *, model: ModelSpec, plan: IterationPlan, runtime: RuntimeSpec,
    parallel: ParallelSpec,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    mode = selected_mode(plan)
    valid_tokens = plan.valid_tokens
    graph_bucket = None
    graph_allowed = (
        runtime.graph_mode == "force"
        or (runtime.graph_mode == "auto" and mode == "decode")
    )
    graph_overflow = False
    if graph_allowed and runtime.graph_buckets:
        graph_bucket = next((bucket for bucket in runtime.graph_buckets if bucket >= valid_tokens), None)
        if graph_bucket is None:
            # Never shrink the physical work below the business token count.
            # Force mode falls back to eager with an explicit guard instead of
            # silently dropping tokens.
            graph_bucket = None
            graph_overflow = runtime.graph_mode == "force"
    if mode == "decode":
        attention_recipe = "decode" if graph_bucket is None else f"decode_graph_replay@{graph_bucket}"
    elif mode == "prefill":
        attention_recipe = "prefill_causal"
    else:
        attention_recipe = "mixed_prefill_decode"
    if model.ffn == "moe":
        mc2_ok = (
            parallel.ep > 1
            and model.activation_dtype in {"bf16", "bfloat16", "fp16", "float16"}
            and valid_tokens <= runtime.mc2_capacity
        )
        if runtime.moe_recipe == "mc2":
            moe_recipe = "mc2" if mc2_ok else "alltoall"
            guard = {"recipe": "mc2", "requested": "mc2", "selected": moe_recipe,
                     "feasible": mc2_ok, "reason": "capacity_or_dtype_guard"}
        elif runtime.moe_recipe == "alltoall":
            moe_recipe, guard = "alltoall", {"recipe": "alltoall", "requested": "alltoall",
                                             "selected": "alltoall", "feasible": True, "reason": "explicit"}
        else:
            moe_recipe = "mc2" if mc2_ok else "alltoall"
            guard = {"recipe": "auto", "requested": "auto", "selected": moe_recipe,
                     "feasible": True,
                     "reason": "mc2_capacity_dtype_guard" if mc2_ok else "mc2_guard_not_met"}
    else:
        moe_recipe = "dense"
        guard = {"recipe": "dense", "selected": "dense", "feasible": True,
                 "reason": "ffn_is_dense"}
    attention_guard = {
        "recipe": "attention", "selected": attention_recipe,
        "feasible": not graph_overflow, "graph_bucket": graph_bucket,
        "reason": (
            "graph_bucket_exceeds_max_bucket_eager_fallback" if graph_overflow
            else "decode_shape_graph_bucket" if graph_bucket else "eager_shape"
        ),
        "requested_tokens": valid_tokens,
        "max_bucket": runtime.graph_buckets[-1] if runtime.graph_buckets else None,
    }
    return {
        "mode": mode,
        "attention_recipe": attention_recipe,
        "moe_recipe": moe_recipe,
        "graph_bucket": graph_bucket,
        "recipe": f"{attention_recipe}+{moe_recipe}",
    }, (guard, attention_guard)


def infer_iteration(
    plan: IterationPlan, model: ModelSpec, runtime: RuntimeSpec,
    parallel: ParallelSpec, *, ep_global_scale: float | None = None,
) -> IterationWork:
    recipe_payload, guards = select_recipe(model=model, plan=plan, runtime=runtime, parallel=parallel)
    graph_bucket = recipe_payload["graph_bucket"]
    padded_tokens = max(plan.valid_tokens, graph_bucket or plan.valid_tokens)
    if not plan.requests:
        raise A3InputError("iteration_plan_empty")
    mode = recipe_payload["mode"]
    if model.layer_plans:
        from .a3_graph_lowering import infer_catalog_iteration

        return infer_catalog_iteration(
            plan, model, runtime, parallel, ep_global_scale=ep_global_scale,
        )
    routing = None
    if model.ffn == "moe":
        global_scale = (
            max(1.0, float(ep_global_scale))
            if ep_global_scale is not None
            else (float(parallel.dp) if parallel.ep_spans_dp else 1.0)
        )
        routing = _routing_stats(
            model, runtime, parallel, padded_tokens, ep_global_scale=global_scale,
        )
    ops: list[OpSpec] = []
    stage_layers = model.layers_per_stage(parallel.pp)
    embedding_tokens = max(1, plan.valid_tokens)
    embedding = _vec_op(
        op_id="embedding", deps=(), stage=0, layer=0,
        requests=tuple(view.request_id for view in plan.requests),
        elements=embedding_tokens * model.hidden_size, ops_per_element=1.0,
        dtype=model.activation_dtype, family="embedding",
        phase=mode, recipe="embedding_gather",
        shape=(embedding_tokens, model.hidden_size),
    )
    ops.append(embedding)
    previous_end = embedding.op_id
    for stage in range(parallel.pp):
        stage_start_layer = stage * stage_layers
        stage_end_layer = stage_start_layer + stage_layers
        dependency = previous_end
        layer_ids: list[str] = []
        for layer in range(stage_start_layer, stage_end_layer):
            layer_prefix = f"s{stage}.l{layer}"
            requests_all = tuple(view.request_id for view in plan.requests)
            norm = _vec_op(
                op_id=f"{layer_prefix}.rmsnorm1", deps=(dependency,) if dependency else (),
                stage=stage, layer=layer, requests=requests_all,
                elements=padded_tokens * model.hidden_size, ops_per_element=5.0,
                dtype=model.activation_dtype, family="rmsnorm", phase=mode,
                recipe=f"{recipe_payload['attention_recipe']}:norm",
            )
            ops.append(norm)
            if model.attention == "gqa":
                qkv = _gemm_op(
                    op_id=f"{layer_prefix}.qkv", deps=(norm.op_id,), stage=stage, layer=layer,
                    requests=requests_all, m=padded_tokens,
                    n=(model.num_heads // parallel.tp + 2 * model.local_kv_heads(parallel.tp)) * model.head_dim,
                    k=model.hidden_size, dtype=model.activation_dtype,
                    family="qkv_projection", phase=mode,
                    recipe=f"{recipe_payload['attention_recipe']}:qkv",
                    semantic_m=plan.valid_tokens,
                    semantic_n=(model.num_heads + 2 * model.num_kv_heads) * model.head_dim,
                    semantic_k=model.hidden_size,
                )
                ops.append(qkv)
                qkv_id = qkv.op_id
            else:
                assert model.mla is not None
                mla = model.mla
                q_a = _gemm_op(
                    op_id=f"{layer_prefix}.mla.q_a", deps=(norm.op_id,), stage=stage, layer=layer,
                    requests=requests_all, m=padded_tokens, n=mla.q_lora_rank // parallel.tp,
                    k=model.hidden_size, dtype=model.activation_dtype,
                    family="mla_q_a", phase=mode, recipe=f"{recipe_payload['attention_recipe']}:mla",
                )
                kv_a = _gemm_op(
                    op_id=f"{layer_prefix}.mla.kv_a", deps=(norm.op_id,), stage=stage, layer=layer,
                    requests=requests_all, m=padded_tokens,
                    n=(mla.kv_lora_rank + mla.qk_rope_head_dim) // parallel.tp,
                    k=model.hidden_size, dtype=model.activation_dtype,
                    family="mla_kv_a", phase=mode, recipe=f"{recipe_payload['attention_recipe']}:mla",
                )
                q_b = _gemm_op(
                    op_id=f"{layer_prefix}.mla.q_b", deps=(q_a.op_id,), stage=stage, layer=layer,
                    requests=requests_all, m=padded_tokens,
                    n=model.num_heads * mla.qk_head_dim // parallel.tp,
                    k=mla.q_lora_rank, dtype=model.activation_dtype,
                    family="mla_q_b", phase=mode, recipe=f"{recipe_payload['attention_recipe']}:mla",
                )
                kv_b = _gemm_op(
                    op_id=f"{layer_prefix}.mla.kv_b", deps=(kv_a.op_id,), stage=stage, layer=layer,
                    requests=requests_all, m=padded_tokens,
                    n=model.num_heads * (mla.qk_nope_head_dim + mla.v_head_dim) // parallel.tp,
                    k=mla.kv_lora_rank, dtype=model.activation_dtype,
                    family="mla_kv_b", phase=mode, recipe=f"{recipe_payload['attention_recipe']}:mla",
                )
                ops.extend([q_a, kv_a, q_b, kv_b])
                barrier = _op(
                    op_id=f"{layer_prefix}.mla_qkv_barrier", family="barrier",
                    phase=mode, stage=stage, layer=layer, requests=requests_all,
                    semantic_shape=(padded_tokens,), local_shape=(padded_tokens,),
                    padded_shape=(padded_tokens,), launch_shape=(padded_tokens,),
                    deps=(q_b.op_id, kv_b.op_id), recipe="mla_producer_barrier",
                    notes=("q_and_kv_producers_must_both_complete",),
                )
                ops.append(barrier)
                qkv_id = barrier.op_id
            attention_end: list[str] = []
            for view in plan.requests:
                view_recipe = (
                    "decode_graph_replay" if graph_bucket is not None and view.q_len == 1
                    else "prefill_causal" if view.q_len > 1 else "decode"
                )
                attention_ops, end_id = _attention_pair(
                    prefix=f"{layer_prefix}.attn.{view.request_id}", view=view,
                    stage=stage, layer=layer, model=model, parallel=parallel,
                    runtime=runtime, deps=(qkv_id,), recipe=view_recipe,
                )
                ops.extend(attention_ops)
                attention_end.append(end_id)
            if graph_bucket is not None and padded_tokens > plan.valid_tokens:
                padding_view = RequestView(
                    request_id="_graph_padding", q_len=padded_tokens - plan.valid_tokens,
                    context=max((view.context for view in plan.requests), default=0),
                    samples=False, phase="decode", padding=True,
                )
                padding_ops, padding_end = _attention_pair(
                    prefix=f"{layer_prefix}.attn._padding", view=padding_view,
                    stage=stage, layer=layer, model=model, parallel=parallel,
                    runtime=runtime, deps=(qkv_id,), recipe="decode_graph_padding",
                )
                ops.extend(padding_ops)
                attention_end.append(padding_end)
            if parallel.dcp > 1:
                ops.append(_op(
                    op_id=f"{layer_prefix}.dcp_allgather", family="dcp_allgather",
                    phase=mode, stage=stage, layer=layer, requests=requests_all,
                    semantic_shape=(padded_tokens, model.num_heads, model.head_dim),
                    local_shape=(padded_tokens, max(1, model.num_heads // parallel.tp), model.head_dim),
                    padded_shape=(padded_tokens, max(1, model.num_heads // parallel.tp), model.head_dim),
                    launch_shape=(padded_tokens, max(1, model.num_heads // parallel.tp), model.head_dim),
                    link_bytes=float(padded_tokens * model.num_heads * model.head_dim * dtype_bytes(model.activation_dtype) * (parallel.dcp - 1) / parallel.dcp),
                    deps=tuple(attention_end), recipe="dcp_allgather",
                    notes=("dcp_context_shard_recombined",),
                ))
                attention_end.append(f"{layer_prefix}.dcp_allgather")
            o_proj = _gemm_op(
                op_id=f"{layer_prefix}.o_proj", deps=tuple(attention_end), stage=stage,
                layer=layer, requests=requests_all, m=padded_tokens,
                n=model.hidden_size, k=model.num_heads * model.head_dim // parallel.tp,
                dtype=model.activation_dtype, family="o_projection", phase=mode,
                recipe=f"{recipe_payload['attention_recipe']}:o_proj",
            )
            ops.append(o_proj)
            last = o_proj.op_id
            if parallel.tp > 1:
                ar = _op(
                    op_id=f"{layer_prefix}.tp_allreduce_attn", family="allreduce",
                    phase=mode, stage=stage, layer=layer, requests=requests_all,
                    semantic_shape=(padded_tokens, model.hidden_size),
                    local_shape=(padded_tokens, model.hidden_size),
                    padded_shape=(padded_tokens, model.hidden_size),
                    launch_shape=(padded_tokens, model.hidden_size),
                    link_bytes=2.0 * (parallel.tp - 1) / parallel.tp
                    * padded_tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
                    deps=(last,), recipe="tp_allreduce_ring",
                    notes=("allreduce_ring_volume=2*(tp-1)/tp*N",),
                )
                ops.append(ar)
                last = ar.op_id
            norm2 = _vec_op(
                op_id=f"{layer_prefix}.rmsnorm2", deps=(last,), stage=stage, layer=layer,
                requests=requests_all, elements=padded_tokens * model.hidden_size,
                ops_per_element=5.0, dtype=model.activation_dtype, family="rmsnorm",
                phase=mode, recipe=f"{recipe_payload['attention_recipe']}:norm2",
            )
            ops.append(norm2)
            last = norm2.op_id
            if model.ffn == "dense":
                gate_up = _gemm_op(
                    op_id=f"{layer_prefix}.gate_up", deps=(last,), stage=stage, layer=layer,
                    requests=requests_all, m=padded_tokens,
                    n=2 * model.intermediate_size // parallel.tp, k=model.hidden_size,
                    dtype=model.activation_dtype, family="gate_up_projection", phase=mode,
                    recipe=f"{recipe_payload['attention_recipe']}:gate_up",
                )
                act = _swiglu_op(
                    op_id=f"{layer_prefix}.act", deps=(gate_up.op_id,),
                    stage=stage, layer=layer, requests=requests_all,
                    tokens=padded_tokens,
                    width=model.intermediate_size // parallel.tp,
                    dtype=model.activation_dtype, phase=mode,
                    recipe=f"{recipe_payload['attention_recipe']}:swiglu",
                )
                down = _gemm_op(
                    op_id=f"{layer_prefix}.down", deps=(act.op_id,), stage=stage, layer=layer,
                    requests=requests_all, m=padded_tokens, n=model.hidden_size,
                    k=model.intermediate_size // parallel.tp, dtype=model.activation_dtype,
                    family="down_projection", phase=mode,
                    recipe=f"{recipe_payload['attention_recipe']}:down",
                )
                ops.extend([gate_up, act, down])
                last = down.op_id
                if parallel.tp > 1:
                    ar = _op(
                        op_id=f"{layer_prefix}.tp_allreduce_mlp", family="allreduce",
                        phase=mode, stage=stage, layer=layer, requests=requests_all,
                        semantic_shape=(padded_tokens, model.hidden_size),
                        local_shape=(padded_tokens, model.hidden_size),
                        padded_shape=(padded_tokens, model.hidden_size),
                        launch_shape=(padded_tokens, model.hidden_size),
                        link_bytes=2.0 * (parallel.tp - 1) / parallel.tp
                        * padded_tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
                        deps=(last,), recipe="tp_allreduce_ring",
                        notes=("allreduce_ring_volume=2*(tp-1)/tp*N",),
                    )
                    ops.append(ar)
                    last = ar.op_id
            else:
                assert routing is not None
                dispatch_bytes = routing.local_routing_tokens * model.hidden_size * dtype_bytes(model.activation_dtype) * (parallel.ep - 1) / parallel.ep
                moe_recipe = recipe_payload["moe_recipe"]
                expert_width = max(1, model.moe_intermediate_size // max(1, parallel.etp))
                counts = list(routing.local_expert_counts)
                if not counts:
                    counts = [max(1, int(math.ceil(routing.local_padded_tokens)))]
                nonzero = [(index, int(count)) for index, count in enumerate(counts) if count > 0]
                if moe_recipe == "mc2" and parallel.ep > 1:
                    # MC2 preserves both dispatch and combine traffic.  Chunks
                    # carry expert groups and overlap dispatch/gmm/combine while
                    # the final barrier waits for every expert output.
                    chunks = max(1, min(runtime.mc2_chunks, max(1, len(nonzero))))
                    buckets: list[list[tuple[int, int]]] = [[] for _ in range(chunks)]
                    for expert_index, (expert, count) in enumerate(nonzero):
                        buckets[expert_index % chunks].append((expert, count))
                    combine_ids: list[str] = []
                    for chunk, items in enumerate(buckets):
                        if not items:
                            continue
                        this_slots = sum(count for _, count in items)
                        dispatch = _op(
                            op_id=f"{layer_prefix}.mc2_dispatch{chunk}", family="ep_dispatch",
                            phase=mode, stage=stage, layer=layer, requests=requests_all,
                            semantic_shape=(this_slots, model.top_k, model.hidden_size),
                            local_shape=(this_slots, model.top_k, model.hidden_size),
                            padded_shape=(this_slots, model.top_k, model.hidden_size),
                            launch_shape=(this_slots, model.top_k, model.hidden_size),
                            link_bytes=dispatch_bytes / chunks, deps=(last,), recipe="mc2_dispatch",
                            metadata=(("mc2_chunk", chunk), ("mc2_chunks", chunks),
                                      ("experts", tuple(expert for expert, _ in items))),
                        )
                        ops.append(dispatch)
                        chunk_downs: list[str] = []
                        for expert, count in items:
                            gate_up = _gemm_op(
                                op_id=f"{layer_prefix}.moe_gate_up_e{expert}",
                                deps=(dispatch.op_id,), stage=stage, layer=layer,
                                requests=requests_all, m=count, n=2 * expert_width,
                                k=model.hidden_size, dtype=model.activation_dtype,
                                family="moe_gate_up", phase=mode, recipe="moe_mc2",
                                metadata=(("expert", expert), ("expert_tokens", count),
                                          ("etp", parallel.etp), ("mc2_chunk", chunk)),
                            )
                            act = _swiglu_op(
                                op_id=f"{layer_prefix}.moe_act_e{expert}",
                                deps=(gate_up.op_id,), stage=stage, layer=layer,
                                requests=requests_all, tokens=count,
                                width=expert_width,
                                dtype=model.activation_dtype, phase=mode,
                                recipe=f"moe_{moe_recipe}_swiglu",
                                metadata=(
                                    ("expert", expert), ("expert_tokens", count),
                                    ("etp", parallel.etp), ("mc2_chunk", chunk),
                                ),
                            )
                            down = _gemm_op(
                                op_id=f"{layer_prefix}.moe_down_e{expert}",
                                deps=(act.op_id,), stage=stage, layer=layer,
                                requests=requests_all, m=count, n=model.hidden_size,
                                k=expert_width, dtype=model.activation_dtype,
                                family="moe_down", phase=mode, recipe="moe_mc2",
                                metadata=(("expert", expert), ("expert_tokens", count)),
                            )
                            ops.extend([gate_up, act, down])
                            chunk_downs.append(down.op_id)
                        combine = _op(
                            op_id=f"{layer_prefix}.mc2_combine{chunk}", family="ep_combine",
                            phase=mode, stage=stage, layer=layer, requests=requests_all,
                            semantic_shape=(this_slots, model.hidden_size),
                            local_shape=(this_slots, model.hidden_size),
                            padded_shape=(this_slots, model.hidden_size),
                            launch_shape=(this_slots, model.hidden_size),
                            link_bytes=dispatch_bytes / chunks, deps=tuple(chunk_downs),
                            recipe="mc2_combine",
                            metadata=(("mc2_chunk", chunk), ("mc2_chunks", chunks)),
                        )
                        ops.append(combine)
                        combine_ids.append(combine.op_id)
                    barrier = _op(
                        op_id=f"{layer_prefix}.mc2_barrier", family="barrier", phase=mode,
                        stage=stage, layer=layer, requests=requests_all,
                        semantic_shape=(padded_tokens, model.hidden_size),
                        local_shape=(padded_tokens, model.hidden_size),
                        padded_shape=(padded_tokens, model.hidden_size),
                        launch_shape=(padded_tokens, model.hidden_size),
                        deps=tuple(combine_ids) or (last,), recipe="mc2_barrier",
                        notes=("barrier_waits_all_mc2_expert_chunks",),
                    )
                    ops.append(barrier)
                    last = barrier.op_id
                else:
                    if parallel.ep > 1:
                        dispatch = _op(
                            op_id=f"{layer_prefix}.ep_dispatch", family="ep_dispatch",
                            phase=mode, stage=stage, layer=layer, requests=requests_all,
                            semantic_shape=(padded_tokens, model.top_k, model.hidden_size),
                            local_shape=(padded_tokens, model.top_k, model.hidden_size),
                            padded_shape=(padded_tokens, model.top_k, model.hidden_size),
                            launch_shape=(padded_tokens, model.top_k, model.hidden_size),
                            link_bytes=dispatch_bytes, deps=(last,), recipe="alltoall",
                            notes=("ep_dispatch_bytes_are_not_multiplied_by_tp",),
                        )
                        ops.append(dispatch)
                        last = dispatch.op_id
                    down_ids: list[str] = []
                    for expert, count in nonzero:
                        gate_up = _gemm_op(
                            op_id=f"{layer_prefix}.moe_gate_up_e{expert}",
                            deps=(last,), stage=stage, layer=layer,
                            requests=requests_all, m=count, n=2 * expert_width,
                            k=model.hidden_size, dtype=model.activation_dtype,
                            family="moe_gate_up", phase=mode, recipe=f"moe_{moe_recipe}",
                            metadata=(("expert", expert), ("expert_tokens", count),
                                      ("etp", parallel.etp),
                                      ("local_experts", routing.local_experts),
                                      ("dropped_tokens", routing.dropped_tokens)),
                        )
                        act = _swiglu_op(
                            op_id=f"{layer_prefix}.moe_act_e{expert}",
                            deps=(gate_up.op_id,), stage=stage, layer=layer,
                            requests=requests_all, tokens=count,
                            width=expert_width,
                            dtype=model.activation_dtype, phase=mode,
                            recipe=f"moe_{moe_recipe}_swiglu",
                            metadata=(
                                ("expert", expert), ("expert_tokens", count),
                                ("etp", parallel.etp),
                                ("local_experts", routing.local_experts),
                                ("dropped_tokens", routing.dropped_tokens),
                            ),
                        )
                        down = _gemm_op(
                            op_id=f"{layer_prefix}.moe_down_e{expert}",
                            deps=(act.op_id,), stage=stage, layer=layer,
                            requests=requests_all, m=count, n=model.hidden_size,
                            k=expert_width, dtype=model.activation_dtype,
                            family="moe_down", phase=mode, recipe=f"moe_{moe_recipe}",
                            metadata=(("expert", expert), ("expert_tokens", count)),
                        )
                        ops.extend([gate_up, act, down])
                        down_ids.append(down.op_id)
                    barrier = _op(
                        op_id=f"{layer_prefix}.moe_barrier", family="barrier", phase=mode,
                        stage=stage, layer=layer, requests=requests_all,
                        semantic_shape=(padded_tokens, model.hidden_size),
                        local_shape=(padded_tokens, model.hidden_size),
                        padded_shape=(padded_tokens, model.hidden_size),
                        launch_shape=(padded_tokens, model.hidden_size),
                        deps=tuple(down_ids) or (last,), recipe="moe_barrier",
                        notes=("barrier_waits_all_local_experts",),
                    )
                    ops.append(barrier)
                    last = barrier.op_id
                    if parallel.ep > 1:
                        combine = _op(
                            op_id=f"{layer_prefix}.ep_combine", family="ep_combine",
                            phase=mode, stage=stage, layer=layer, requests=requests_all,
                            semantic_shape=(padded_tokens, model.hidden_size),
                            local_shape=(padded_tokens, model.hidden_size),
                            padded_shape=(padded_tokens, model.hidden_size),
                            launch_shape=(padded_tokens, model.hidden_size),
                            link_bytes=dispatch_bytes, deps=(last,), recipe="alltoall",
                            notes=("ep_combine_bytes_are_not_multiplied_by_tp",),
                        )
                        ops.append(combine)
                        last = combine.op_id
                if model.num_shared_experts:
                    shared_width = max(1, model.moe_intermediate_size * model.num_shared_experts // parallel.tp)
                    shared = _gemm_op(
                        op_id=f"{layer_prefix}.shared_experts", deps=(last,), stage=stage,
                        layer=layer, requests=requests_all, m=padded_tokens,
                        n=2 * shared_width, k=model.hidden_size,
                        dtype=model.activation_dtype,
                        family="shared_expert_gate_up", phase=mode, recipe="shared_expert",
                    )
                    shared_act = _swiglu_op(
                        op_id=f"{layer_prefix}.shared_expert_act",
                        deps=(shared.op_id,), stage=stage, layer=layer,
                        requests=requests_all, tokens=padded_tokens,
                        width=shared_width, dtype=model.activation_dtype,
                        phase=mode, recipe="shared_expert_swiglu",
                    )
                    shared_down = _gemm_op(
                        op_id=f"{layer_prefix}.shared_expert_down",
                        deps=(shared_act.op_id,),
                        stage=stage, layer=layer, requests=requests_all, m=padded_tokens,
                        n=model.hidden_size, k=shared_width,
                        dtype=model.activation_dtype,
                        family="shared_expert_down", phase=mode, recipe="shared_expert",
                    )
                    ops.extend([shared, shared_act, shared_down])
                    last = shared_down.op_id
                if parallel.tp > 1:
                    ar = _op(
                        op_id=f"{layer_prefix}.tp_allreduce_moe", family="allreduce",
                        phase=mode, stage=stage, layer=layer, requests=requests_all,
                        semantic_shape=(padded_tokens, model.hidden_size),
                        local_shape=(padded_tokens, model.hidden_size),
                        padded_shape=(padded_tokens, model.hidden_size),
                        launch_shape=(padded_tokens, model.hidden_size),
                        link_bytes=2.0 * (parallel.tp - 1) / parallel.tp
                        * padded_tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
                        deps=(last,), recipe="tp_allreduce_ring",
                    )
                    ops.append(ar)
                    last = ar.op_id
            dependency = last
            layer_ids.append(last)
        if stage + 1 < parallel.pp:
            p2p = _op(
                op_id=f"s{stage}.pp_p2p", family="pp_p2p", phase=mode, stage=stage,
                layer=-1, requests=tuple(view.request_id for view in plan.requests),
                semantic_shape=(padded_tokens, model.hidden_size),
                local_shape=(padded_tokens, model.hidden_size),
                padded_shape=(padded_tokens, model.hidden_size),
                launch_shape=(padded_tokens, model.hidden_size),
                link_bytes=padded_tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
                deps=(dependency,), recipe="pp_p2p",
                notes=("pp_transfer_is_between_adjacent_stages",),
            )
            ops.append(p2p)
            previous_end = p2p.op_id
        else:
            previous_end = dependency
    # KV transfer and visibility: a prefill producer must publish KV before a
    # decoder can consume it.  Chunks carry independent link bytes; the final
    # barrier is the consumer-visible point for the full prompt.
    sample_rows = sum(1 for view in plan.requests if view.samples)
    if runtime.kv_transfer.enabled and sample_rows:
        kv_per_token = runtime.kv_transfer.bytes_per_token or int(round(
            model.kv_bytes_per_token_per_layer() * model.layers_per_stage(parallel.pp)
        ))
        visible_tokens = sum(
            view.q_len + view.context for view in plan.requests if view.samples
        )
        total_kv = float(max(1, visible_tokens) * kv_per_token)
        chunks = max(1, int(runtime.kv_transfer.chunks))
        transfer_deps = previous_end
        transfer_ids: list[str] = []
        for chunk in range(chunks):
            transfer_id = f"kv_transfer.c{chunk}"
            ops.append(_op(
                op_id=transfer_id, family="kv_transfer", phase=mode,
                stage=parallel.pp - 1, layer=-1,
                requests=tuple(view.request_id for view in plan.requests if view.samples),
                semantic_shape=(visible_tokens,),
                local_shape=(visible_tokens,),
                padded_shape=(visible_tokens,),
                launch_shape=(visible_tokens,),
                link_bytes=total_kv / chunks,
                hbm_bytes=2.0 * total_kv / chunks,
                deps=(transfer_deps,),
                recipe=f"kv_transfer_{runtime.kv_transfer.channel}",
                notes=("kv_producer->transfer->visibility->consumer", "chunk_visibility_is_per_chunk"),
            ))
            transfer_ids.append(transfer_id)
            transfer_deps = transfer_id
        barrier = _op(
            op_id="kv_transfer_visible", family="barrier", phase=mode,
            stage=parallel.pp - 1, layer=-1,
            requests=tuple(view.request_id for view in plan.requests if view.samples),
            semantic_shape=(visible_tokens,), local_shape=(visible_tokens,),
            padded_shape=(visible_tokens,), launch_shape=(visible_tokens,),
            deps=tuple(transfer_ids), recipe="kv_visibility_barrier",
            notes=("consumer_waits_full_prompt_kv_visibility",),
        )
        ops.append(barrier)
        previous_end = barrier.op_id
    # Logits/sampling: only valid sampled rows; graph padding rows are not valid.
    if sample_rows:
        last_stage = parallel.pp - 1
        ops.append(_gemm_op(
            op_id="logits", deps=(previous_end,), stage=last_stage, layer=-1,
            requests=tuple(view.request_id for view in plan.requests if view.samples),
            m=sample_rows, n=model.vocab_size // parallel.tp, k=model.hidden_size,
            dtype=model.activation_dtype, family="logits", phase=mode,
            recipe="vocab_parallel_logits",
        ))
        ops.append(_vec_op(
            op_id="sampling", deps=("logits",), stage=last_stage, layer=-1,
            requests=tuple(view.request_id for view in plan.requests if view.samples),
            elements=max(1, sample_rows * model.vocab_size // parallel.tp),
            ops_per_element=1.0, dtype="fp32", family="sampling", phase=mode,
            recipe="argmax_sampling", shape=(sample_rows, model.vocab_size // parallel.tp),
        ))
    if model.weight_dtype in {"int8", "int4", "fp8", "fp8_e4m3", "fp8_e5m2"} and model.weight_dtype != model.compute_dtype:
        # Quantized weights need an explicit dequant/accumulation path.  Do not
        # infer a 2x/4x peak.  The dequant kernel is a real dependency before
        # the consuming GEMM/MoE kernel and carries its weight-read traffic.
        upgraded: list[OpSpec] = []
        quant_families = {
            "qkv_projection", "o_projection", "gate_up_projection", "down_projection",
            "mla_q_a", "mla_kv_a", "mla_q_b", "mla_kv_b", "moe_gate_up",
            "moe_down", "shared_expert_gate_up", "logits",
        }
        for op in ops:
            if op.family in quant_families and len(op.launch_shape) >= 3:
                weight_elements = op.launch_shape[-2] * op.launch_shape[-1]
                weight_bytes = float(weight_elements) * dtype_bytes(model.weight_dtype)
                dequant = _op(
                    op_id=f"{op.op_id}.dequant", family="dequant", phase=op.phase,
                    stage=op.stage, layer=op.layer, requests=op.requests,
                    semantic_shape=(op.launch_shape[-2], op.launch_shape[-1]),
                    local_shape=(op.launch_shape[-2], op.launch_shape[-1]),
                    padded_shape=(align_up(op.launch_shape[-2], 32), align_up(op.launch_shape[-1], 32)),
                    launch_shape=(op.launch_shape[-2], op.launch_shape[-1]),
                    vector_ops=float(weight_elements), hbm_bytes=weight_bytes,
                    mte2_bytes=weight_bytes, mte3_bytes=weight_bytes,
                    deps=op.deps, recipe=f"{op.recipe}:dequant",
                    notes=("quantized_weight_dequant_before_gemm",),
                )
                upgraded.append(dequant)
                upgraded.append(replace(op, deps=(dequant.op_id,)))
            else:
                upgraded.append(op)
        ops = upgraded
    return IterationWork(
        ops=tuple(ops), plan=plan, stage_count=parallel.pp,
        valid_tokens=plan.valid_tokens, padded_tokens=padded_tokens,
        graph_bucket=graph_bucket, mode=mode, recipe=recipe_payload["recipe"],
        routing=routing, recipe_guards=guards,
        notes=(
            "semantic_shape_local_shape_padded_shape_launch_shape_are_distinct",
            "graph_padding_is_physical_work_not_extra_output",
            "attention_uses_query_and_kv_length_independently",
        ),
    )


def derive_iteration_work(
    plan: IterationPlan, model: ModelSpec, runtime: RuntimeSpec,
    machine: MachineSpec, parallel: ParallelSpec | None = None,
    *, ep_global_scale: float | None = None,
) -> IterationWork:
    parallel = parallel or runtime.parallel
    parallel.validate(model)
    machine.validate_placement(parallel)
    encoded_views = sorted(
        str(view.request_id) for view in plan.requests
        if bool(getattr(view, "media_encoded", False))
    )
    if encoded_views:
        import json as _json
        from dataclasses import replace as _replace

        merged = dict(runtime.metadata)
        existing: set[str] = set()
        raw_encoded = merged.get("media_encoded_requests")
        if raw_encoded:
            try:
                existing = set(_json.loads(raw_encoded))
            except (TypeError, ValueError, _json.JSONDecodeError):
                existing = set()
        merged["media_encoded_requests"] = _json.dumps(
            sorted(existing | set(encoded_views))
        )
        runtime = _replace(runtime, metadata=tuple(sorted(merged.items())))
    metadata = dict(runtime.metadata)
    if metadata.get("draft_mode"):
        if metadata.get("draft_accept_tokens") is None:
            # Explicit fallback: absence of caller acceptance evidence is not
            # an all-accepted distribution.  Core state records the same
            # fallback (0 accepted -> one fallback token).
            metadata["draft_accept_tokens"] = "0"
        # Explicit draft mode is preserved for prefill/final-tail scheduling;
        # catalog chooses the initial candidate block and consumes
        # ``draft_next_counts`` to suppress it when no future draft is useful.
        # Absence of a verify view must not silently erase draft semantics.
        _ = tuple(
            view for view in plan.requests
            if bool(getattr(view, "draft_verify", True))
        )
    if tuple(sorted(metadata.items())) != tuple(runtime.metadata):
        from dataclasses import replace as _replace

        runtime = _replace(runtime, metadata=tuple(sorted(metadata.items())))
    return infer_iteration(
        plan, model, runtime, parallel, ep_global_scale=ep_global_scale,
    )


from .a3_memory import MemoryPlan, fits_workload, memory_plan, workload_required_tokens  # noqa: E402


__all__ = [
    "DEFAULT_TOKEN_ALIGN", "DEFAULT_HEAD_ALIGN", "RequestView", "IterationPlan",
    "OpSpec", "RoutingStats", "IterationWork", "MemoryPlan", "selected_mode",
    "select_recipe", "infer_iteration", "derive_iteration_work", "memory_plan",
    "workload_required_tokens", "fits_workload",
]
