"""Optional MTP/DSpark proposal, verification and standalone draft heads.

[Role]
  Lower explicit draft execution: target-feature input projection, draft layers,
  Markov/confidence heads, target verification shapes and accepted-state commit.
  Also lower the standalone Kimi DSpark draft's input and output heads.

[Invariants]
  * A plain forward never calls these functions; body layer plans exclude draft.
  * DSpark uses its declared draft expert count/topK and hidden width, not the
    target MoE configuration.  Target feature input width is
    ``len(target_layer_ids) * hidden_size`` for DSpark.
  * Block size is a proposal width, not an acceptance count.  Rejected
    proposals still incur draft and verify work; only accepted state commits.
  * Markov and confidence widths are explicit (confidence 7168+256=7424 in the
    DSpark fixture).
  * Verification shapes use the actual target layer dimensions and a proposal
    query length separate from committed context.

[Timeouts & Locks]
  Pure arithmetic bounded by draft/verify layers; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from .a3_catalog_moe import lower_moe, routing_stats
from .a3_catalog_ops import (
    barrier_op, cache_store_op, comm_op, gemm_op, packed_weight_bytes, vector_op,
)
from .a3_catalog_types import DraftSpec, QuantSpec
from .a3_model_types import LayerPlan, ModelSpec
from .a3_ops import OpSpec, _op
from .a3_types import A3InputError, ParallelSpec, RuntimeSpec, dtype_bytes


@dataclass
class DraftResult:
    kind: str
    proposal_tokens: int
    verify_tokens: int
    accepted_upper: int
    ops: list[OpSpec] = field(default_factory=list)
    proposal_last: str = ""
    verify_last: str = ""
    commit_id: str = ""
    accepted_tokens: int = 0
    fallback_tokens: int = 0
    configured_proposal_tokens: int = 0
    scheduled_proposal_counts: tuple[int, ...] = ()
    emitted_proposal_counts: tuple[int, ...] = ()
    emitted_proposal_tokens: int = 0
    unverified_proposal_tokens: int = 0
    discarded_proposal_tokens: int = 0
    ordinary_query_rows: int = 0
    target_query_rows: int = 0
    witnesses: tuple[dict[str, Any], ...] = ()
    # Placeholder heads are never inserted; empty means the branch is absent.


def _requests(views: Sequence[Any]) -> tuple[str, ...]:
    return tuple(view.request_id for view in views)


def _accepted_upper(runtime: Any, draft: DraftSpec) -> int:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if "draft_accept_tokens" not in meta:
        return int(draft.verify_tokens)
    raw = int(meta.get("draft_accept_tokens", 0) or 0)
    return max(0, min(raw, int(draft.verify_tokens)))


def _draft_attention(
    *, prefix: str, draft: DraftSpec, model: ModelSpec, deps: str, stage: int,
    layer: int, requests: tuple[str, ...], parallel: ParallelSpec,
    quant: QuantSpec, mode: str, proposal: int, context: int, batch: int = 1,
    proposal_counts: Sequence[int] | None = None,
) -> tuple[list[OpSpec], str]:
    ops: list[OpSpec] = []
    h = draft.hidden_size
    local_heads = max(1, draft.num_heads // max(1, parallel.tp))
    local_kv = max(1, math.ceil(draft.num_kv_heads / max(1, parallel.tp)))
    counts = tuple(
        max(0, int(count)) for count in (
            proposal_counts if proposal_counts is not None else (proposal,)
        )
    )
    active = tuple(
        (request, count) for request, count in zip(requests, counts) if count > 0
    )
    if not active:
        raise A3InputError("catalog_draft_attention_no_active_request")
    seq = max(1, max(context + count for _, count in active))
    if draft.noncausal_query_block:
        pairs = sum(
            count * max(0, context) + count * count for _, count in active
        )
    else:
        pairs = sum(
            count * max(0, context) + count * (count + 1) / 2.0
            for _, count in active
        )
    if draft.attention_kind == "mla" and draft.q_lora_rank:
        qk = draft.qk_nope_head_dim + draft.qk_rope_head_dim
        q_a = gemm_op(
            op_id=f"{prefix}.draft_q_a", deps=(deps,), stage=stage, layer=layer,
            requests=requests, m=proposal, n=draft.q_lora_rank, k=h,
            quant=quant, category="mla", phase=mode, family="mla_q_a",
            semantic_family="draft_q_a",
        )
        q_b = gemm_op(
            op_id=f"{prefix}.draft_q_b", deps=(q_a.op_id,), stage=stage,
            layer=layer, requests=requests, m=proposal,
            n=local_heads * qk, k=draft.q_lora_rank, quant=quant,
            category="mla", phase=mode, family="mla_q_b",
            semantic_family="draft_q_b",
        )
        wkv = gemm_op(
            op_id=f"{prefix}.draft_wkv", deps=(deps,), stage=stage, layer=layer,
            requests=requests, m=proposal, n=draft.kv_lora_rank or draft.head_dim,
            k=h, quant=quant, category="mla", phase=mode, family="mla_kv_a",
            semantic_family="draft_wkv",
        )
        ops.extend([q_a, q_b, wkv])
        qk_dim = draft.head_dim if draft.kv_lora_rank == draft.head_dim else qk
        v_dim = draft.head_dim if draft.kv_lora_rank == draft.head_dim else draft.v_head_dim
        attn_deps = (q_b.op_id, wkv.op_id)
        if draft.kv_lora_rank and draft.kv_lora_rank != draft.head_dim:
            kv_b = gemm_op(
                op_id=f"{prefix}.draft_kv_b", deps=(wkv.op_id,), stage=stage,
                layer=layer, requests=requests, m=proposal,
                n=local_heads * (draft.qk_nope_head_dim + draft.v_head_dim),
                k=draft.kv_lora_rank, quant=quant, category="mla", phase=mode,
                family="mla_kv_b", semantic_family="draft_kv_b",
            )
            ops.append(kv_b)
            attn_deps = (q_b.op_id, kv_b.op_id)
    else:
        q = gemm_op(
            op_id=f"{prefix}.draft_q", deps=(deps,), stage=stage, layer=layer,
            requests=requests, m=proposal, n=draft.q_width, k=h,
            quant=quant, category="attn", phase=mode, family="qkv_projection",
            semantic_family="draft_q_projection",
        )
        k = gemm_op(
            op_id=f"{prefix}.draft_k", deps=(deps,), stage=stage, layer=layer,
            requests=requests, m=proposal, n=draft.kv_width, k=h,
            quant=quant, category="attn", phase=mode, family="qkv_projection",
            semantic_family="draft_k_projection",
        )
        v = gemm_op(
            op_id=f"{prefix}.draft_v", deps=(deps,), stage=stage, layer=layer,
            requests=requests, m=proposal, n=draft.kv_width, k=h,
            quant=quant, category="attn", phase=mode, family="qkv_projection",
            semantic_family="draft_v_projection",
        )
        ops.extend([q, k, v])
        qk_dim, v_dim = draft.head_dim, draft.head_dim
        attn_deps = (q.op_id, k.op_id, v.op_id)
    context_deps: tuple[str, ...] = ()
    if draft.noncausal_query_block and context > 0:
        ctx_n = draft.kv_width if draft.attention_kind != "mla" else (
            draft.kv_lora_rank + draft.qk_rope_head_dim
        )
        ctx_proj = gemm_op(
            op_id=f"{prefix}.draft_context_kv", deps=(deps,), stage=stage,
            layer=layer, requests=requests, m=context, n=ctx_n, k=h,
            quant=quant, category="attn", phase=mode, family="qkv_projection",
            semantic_family="dspark_target_context_kv_projection",
        )
        ctx_norm = vector_op(
            op_id=f"{prefix}.draft_context_norm", deps=(ctx_proj.op_id,),
            stage=stage, layer=layer, requests=requests,
            elements=max(1, context * ctx_n), ops_per_element=5.0,
            dtype="bf16", family="rmsnorm", phase=mode,
            recipe="catalog_dspark_context_norm",
            shape=(max(1, context), ctx_n),
            metadata=(("tokens", max(1, context)), ("reduction_width", ctx_n),
                      ("norm_type", "rmsnorm"), ("norm_eps", model.norm_eps)),
        )
        ctx_rope = vector_op(
            op_id=f"{prefix}.draft_context_rope", deps=(ctx_norm.op_id,),
            stage=stage, layer=layer, requests=requests,
            elements=max(1, context * draft.qk_rope_head_dim),
            ops_per_element=3.0, dtype="bf16", family="rope", phase=mode,
            recipe="catalog_dspark_context_rope",
        )
        ctx_store = cache_store_op(
            op_id=f"{prefix}.draft_context_store", deps=(ctx_rope.op_id,),
            stage=stage, layer=layer, requests=requests, phase=mode,
            recipe="catalog_dspark_context_store", rows=context,
            row_bytes=ctx_n * 2, dtype="bf16", kind="dspark_context_kv",
        )
        ops.extend([ctx_proj, ctx_norm, ctx_rope, ctx_store])
        context_deps = (ctx_store.op_id,)
    qk_ops: list[OpSpec] = []
    soft_ops: list[OpSpec] = []
    pv_ops: list[OpSpec] = []
    pv_ids: list[str] = []
    for index, (request_id, count) in enumerate(active):
        suffix = "" if len(active) == 1 else f".r{request_id}"
        qk = _op(
            op_id=f"{prefix}.draft_qk{suffix}", family="attention_qk", phase=mode,
            stage=stage, layer=layer, requests=(request_id,),
            semantic_shape=(count, context + count, local_heads, qk_dim),
            local_shape=(count, context + count, local_heads, qk_dim),
            padded_shape=(count, context + count, local_heads, qk_dim),
            launch_shape=(count, context + count, local_heads, qk_dim),
            flops=2.0 * local_heads * (
                count * max(0, context) + count * count
                if draft.noncausal_query_block
                else count * max(0, context) + count * (count + 1) / 2.0
            ) * qk_dim,
            hbm_bytes=2.0 * (context + count) * qk_dim * dtype_bytes("bf16")
            + count * local_heads * (context + count) * 2.0,
            deps=tuple(attn_deps) + context_deps,
            recipe="catalog_draft_qk", dtype="bf16",
            metadata=(("noncausal_query_block", draft.noncausal_query_block),
                      ("context_kv_precomputed", bool(context_deps)),
                      ("request_id", request_id), ("query_rows", count)),
        )
        soft = vector_op(
            op_id=f"{prefix}.draft_softmax{suffix}", deps=(qk.op_id,),
            stage=stage, layer=layer, requests=(request_id,),
            elements=max(1, count * local_heads * (context + count)),
            ops_per_element=5.0, dtype="fp32",
            family="attention_softmax", phase=mode,
            recipe="catalog_draft_softmax",
        )
        pv = _op(
            op_id=f"{prefix}.draft_pv{suffix}", family="attention_pv", phase=mode,
            stage=stage, layer=layer, requests=(request_id,),
            semantic_shape=(count, context + count, local_heads, v_dim),
            local_shape=(count, context + count, local_heads, v_dim),
            padded_shape=(count, context + count, local_heads, v_dim),
            launch_shape=(count, context + count, local_heads, v_dim),
            flops=2.0 * local_heads * (
                count * max(0, context) + count * count
                if draft.noncausal_query_block
                else count * max(0, context) + count * (count + 1) / 2.0
            ) * v_dim,
            hbm_bytes=2.0 * (context + count) * v_dim * dtype_bytes("bf16"),
            deps=(soft.op_id,), recipe="catalog_draft_pv", dtype="bf16",
        )
        qk_ops.append(qk)
        soft_ops.append(soft)
        pv_ops.append(pv)
        pv_ids.append(pv.op_id)
    o = gemm_op(
        op_id=f"{prefix}.draft_o", deps=tuple(pv_ids), stage=stage, layer=layer,
        requests=requests, m=proposal, n=h,
        k=local_heads * (v_dim if draft.attention_kind == "mla" else draft.head_dim),
        quant=quant, category="mla" if draft.attention_kind == "mla" else "attn",
        phase=mode, family="o_projection", semantic_family="draft_o_projection",
    )
    ops.extend(qk_ops + soft_ops + pv_ops)
    ops.append(o)
    return ops, o.op_id



def _draft_layer(
    *, prefix: str, draft: DraftSpec, model: ModelSpec, deps: str, stage: int,
    layer: int, requests: tuple[str, ...], parallel: ParallelSpec,
    quant: QuantSpec, mode: str, proposal: int, context: int, batch: int = 1,
    proposal_counts: Sequence[int] | None = None,
) -> tuple[list[OpSpec], str]:
    ops: list[OpSpec] = []
    norm = vector_op(
        op_id=f"{prefix}.draft_norm", deps=(deps,), stage=stage, layer=layer,
        requests=requests, elements=max(1, proposal * draft.hidden_size),
        ops_per_element=5.0, dtype="bf16", family="rmsnorm", phase=mode,
        recipe="catalog_draft_norm",
        shape=(proposal, draft.hidden_size),
        metadata=(("tokens", proposal),
                  ("reduction_width", draft.hidden_size),
                  ("norm_type", "rmsnorm"), ("norm_eps", model.norm_eps)),
    )
    attn_ops, attn_last = _draft_attention(
        prefix=prefix, draft=draft, model=model, deps=norm.op_id,
        stage=stage, layer=layer,
        requests=requests, parallel=parallel, quant=quant, mode=mode,
        proposal=proposal, context=context, batch=batch,
        proposal_counts=proposal_counts,
    )
    ops.append(norm)
    ops.extend(attn_ops)
    add = vector_op(
        op_id=f"{prefix}.draft_attn_add", deps=(attn_last, deps), stage=stage,
        layer=layer, requests=requests,
        elements=max(1, proposal * draft.hidden_size), ops_per_element=2.0,
        dtype="bf16", family="elementwise", phase=mode,
        recipe="catalog_draft_residual",
    )
    norm2 = vector_op(
        op_id=f"{prefix}.draft_ffn_norm", deps=(add.op_id,), stage=stage,
        layer=layer, requests=requests,
        elements=max(1, proposal * draft.hidden_size), ops_per_element=5.0,
        dtype="bf16", family="rmsnorm", phase=mode,
        recipe="catalog_draft_ffn_norm",
        shape=(proposal, draft.hidden_size),
        metadata=(("tokens", proposal),
                  ("reduction_width", draft.hidden_size),
                  ("norm_type", "rmsnorm"), ("norm_eps", model.norm_eps)),
    )
    ops.extend([add, norm2])
    if draft.num_experts > 0:
        plan = LayerPlan(
            index=layer, attention=draft.attention_kind, ffn="moe",
            num_experts=draft.num_experts, top_k=draft.top_k,
            moe_intermediate_size=draft.expert_intermediate_size,
            num_shared_experts=draft.num_shared_experts,
            shared_intermediate_size=draft.expert_intermediate_size * max(1, draft.num_shared_experts),
            router_input_size=draft.hidden_size,
            architecture="gated_gqa" if draft.num_shared_experts else "gqa",
        )
        routing = routing_stats(model, plan, parallel, proposal)
        moe_ops, ffn_last = lower_moe(
            prefix=prefix, plan=plan, stage=stage, deps=norm2.op_id,
            model=model, parallel=parallel, quant=quant, mode=mode,
            padded_tokens=proposal, views=(),
            routing=routing,
        )
        # lower_moe accepts views for request ids; pass a synthetic tuple by
        # rebuilding with requests via metadata is unnecessary for work.
        ops.extend(moe_ops)
    else:
        gate = gemm_op(
            op_id=f"{prefix}.draft_gate", deps=(norm2.op_id,), stage=stage,
            layer=layer, requests=requests, m=proposal,
            n=2 * draft.intermediate_size, k=draft.hidden_size, quant=quant,
            category="dense_mlp", phase=mode, family="gate_up_projection",
            semantic_family="draft_dense_gate_up",
        )
        act_elements = max(1, proposal * draft.intermediate_size)
        act_bytes = dtype_bytes("bf16")
        act = vector_op(
            op_id=f"{prefix}.draft_act", deps=(gate.op_id,), stage=stage,
            layer=layer, requests=requests,
            elements=act_elements,
            ops_per_element=4.0, dtype="bf16", family="swiglu", phase=mode,
            recipe="catalog_draft_swiglu",
            shape=(proposal, draft.intermediate_size),
            read_bytes_override=2.0 * act_elements * act_bytes,
            write_bytes_override=1.0 * act_elements * act_bytes,
            metadata=(("read_halves", 2), ("write_outputs", 1),
                      ("packed_gate_input", True)),
        )
        down = gemm_op(
            op_id=f"{prefix}.draft_down", deps=(act.op_id,), stage=stage,
            layer=layer, requests=requests, m=proposal,
            n=draft.hidden_size, k=draft.intermediate_size, quant=quant,
            category="dense_mlp", phase=mode, family="down_projection",
            semantic_family="draft_dense_down",
        )
        add2 = vector_op(
            op_id=f"{prefix}.draft_ffn_add", deps=(down.op_id, add.op_id),
            stage=stage, layer=layer, requests=requests,
            elements=max(1, proposal * draft.hidden_size), ops_per_element=2.0,
            dtype="bf16", family="elementwise", phase=mode,
            recipe="catalog_draft_residual",
        )
        ops.extend([gate, act, down, add2])
        ffn_last = add2.op_id
    if draft.mhc_mult:
        mix = (2 + draft.mhc_mult) * draft.mhc_mult
        hc = gemm_op(
            op_id=f"{prefix}.draft_hc", deps=(ffn_last,), stage=stage,
            layer=layer, requests=requests, m=proposal, n=mix,
            k=draft.mhc_mult * draft.hidden_size, quant=quant,
            category="mhc", phase=mode, family="gemm",
            semantic_family="draft_hyper_connection_mix",
        )
        hc_out = vector_op(
            op_id=f"{prefix}.draft_hc_out", deps=(hc.op_id,), stage=stage,
            layer=layer, requests=requests, elements=max(1, proposal * mix),
            ops_per_element=4.0, dtype="fp32", family="mhc", phase=mode,
            recipe="catalog_draft_hc_sinkhorn",
        )
        ops.extend([hc, hc_out])
        ffn_last = hc_out.op_id
    return ops, ffn_last


def optional_draft_spec(model: ModelSpec, kind: str) -> DraftSpec:
    aux = model.auxiliary
    if aux is None:
        raise A3InputError("catalog_draft_missing")
    dspark_mode = kind in {"dspark", "verify", "draft_verify"}
    draft = aux.dspark if dspark_mode else aux.mtp
    if draft is None:
        if dspark_mode and aux.mtp is not None and aux.mtp.kind == "mtp_dspark":
            draft = aux.mtp
        elif not dspark_mode and aux.dspark is not None:
            raise A3InputError("catalog_draft_mode_unavailable:mtp")
        elif dspark_mode:
            raise A3InputError("catalog_draft_mode_unavailable:dspark")
    if draft is None:
        raise A3InputError("catalog_draft_mode_missing")
    draft.validate()
    return draft


def scheduled_proposal_counts(views: Sequence[Any]) -> tuple[int, ...]:
    """Return each RequestView's explicit current scheduled draft count.

    The field is numeric state: zero means zero even when ``draft_verify`` is
    true. Missing/foreign inputs are rejected; zero is never reinterpreted as
    the configured capability block.
    """
    counts: list[int] = []
    for view in views:
        raw = getattr(view, "draft_proposal_tokens", 0)
        if raw is None:
            raise A3InputError("catalog_draft_scheduled_count_missing")
        counts.append(max(0, int(raw)))
    return tuple(counts)


def lower_optional_draft(
    *, model: ModelSpec, runtime: Any, parallel: ParallelSpec, quant: QuantSpec,
    views: Sequence[Any], context: int, mode: str = "draft",
    proposal_counts: Sequence[int] | None = None,
) -> DraftResult:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    kind = str(meta.get("draft_mode", "dspark")).lower()
    draft = optional_draft_spec(model, kind)
    requests = _requests(views)
    batch = max(1, len(requests))
    if proposal_counts is None:
        scheduled = scheduled_proposal_counts(views)
    else:
        scheduled = tuple(max(0, int(count)) for count in proposal_counts)
    if len(scheduled) != len(views):
        raise A3InputError("catalog_draft_scheduled_counts_length")
    scheduled_total = sum(scheduled)
    if scheduled_total <= 0:
        raise A3InputError("catalog_draft_scheduled_zero")
    # Draft candidate generation uses the configured block for every request
    # that received a verify slot; the scheduler may truncate only the verify
    # rows.  Keep emitted candidates distinct from scheduled verify rows.
    emitted = tuple(
        max(0, int(draft.proposal_tokens)) if count > 0 else 0
        for count in scheduled
    )
    emitted_total = sum(emitted)
    active_emitted = tuple(count for count in emitted if count > 0)
    ordinary_rows = sum(max(0, int(getattr(view, "q_len", 1))) for view in views)
    accepted = 0
    fallback = 0
    ops: list[OpSpec] = []
    stage = max(0, parallel.pp - 1)
    inp = gemm_op(
        op_id="draft.input_proj", deps=(), stage=stage, layer=-1,
        requests=requests, m=emitted_total, n=draft.hidden_size,
        k=draft.input_width, quant=quant,
        category=draft.input_projection_category, phase=mode, family="gemm",
        semantic_family="dspark_target_feature_input_projection",
        metadata=(("target_layer_ids", draft.target_layer_ids),
                  ("input_width", draft.input_width),
                  ("source_dtype", "fp8_e4m3" if model.quant and model.quant.weight_format != "bf16" else "bf16"),
                  ("runtime_dtype", "bf16"),
                  ("configured_proposal_tokens", draft.proposal_tokens),
                  ("scheduled_proposal_counts", scheduled)),
    )
    ops.append(inp)
    last = inp.op_id
    if draft.target_shared_embedding:
        embed_read = vector_op(
            op_id="draft.target_embedding_read", deps=(inp.op_id,),
            stage=stage, layer=-1, requests=requests,
            elements=max(1, emitted_total * draft.hidden_size),
            ops_per_element=1.0, dtype="bf16", family="embedding",
            phase=mode, recipe="catalog_dspark_target_embedding_read",
            semantic_family="target_shared_embedding_read",
            read_write=1.0, shape=(emitted_total, draft.hidden_size),
            metadata=(("shared_owner", True), ("owner", "target_embed_tokens")),
        )
        ops.append(embed_read)
        last = embed_read.op_id
    for layer in range(draft.layers):
        layer_ops, last = _draft_layer(
            prefix=f"draft.l{layer}", draft=draft, model=model, deps=last,
            stage=stage, layer=layer, requests=requests, parallel=parallel,
            quant=quant, mode=mode, proposal=emitted_total, context=context,
            batch=len(active_emitted), proposal_counts=emitted,
        )
        ops.extend(layer_ops)
    proposal_last = last
    if parallel.pp not in tuple(draft.pp_capability or (1,)):
        raise A3InputError(
            f"catalog_draft_backend_pp_unsupported:{draft.kind}:pp{parallel.pp}"
        )
    base_last = None
    if draft.base_logits and draft.target_shared_lm_head:
        base = gemm_op(
            op_id="draft.base_logits", deps=(last,), stage=stage, layer=-1,
            requests=requests, m=emitted_total, n=draft.vocab_size,
            k=draft.hidden_size, quant=quant, category="target_alias_lm_head",
            phase=mode, family="logits",
            semantic_family="dspark_base_logits_target_shared_head",
            metadata=(("shared_owner", True), ("owner", "target_lm_head"),
                      ("replicated_consumer_read", True),
                      ("scheduled_proposal_counts", scheduled)),
        )
        ops.append(base)
        base_last = base.op_id
    if draft.markov_rank:
        steps = max(1, max(active_emitted))
        lookup_last = last
        sample_ids: list[str] = []
        for step in range(steps):
            active_at_step = tuple(
                request for request, count in zip(requests, emitted)
                if count > step
            )
            active_batch = len(active_at_step)
            lookup = vector_op(
                op_id=f"draft.markov_lookup{step}",
                deps=(lookup_last if step == 0 else sample_ids[-1],),
                stage=stage, layer=-1, requests=active_at_step,
                elements=active_batch, ops_per_element=2.0, dtype="bf16",
                family="embedding", phase=mode,
                recipe="catalog_dspark_markov_lookup",
                semantic_family="dspark_markov_previous_token_lookup",
                read_write=1.0,
                hbm_bytes_override=float(
                    active_batch * draft.markov_rank * 2),
                shape=(active_batch, draft.markov_rank),
                metadata=(("lookup_table", "markov_w1"),
                          ("table_shape", (draft.vocab_size, draft.markov_rank)),
                          ("replicated", draft.markov_replicated),
                          ("step", step),
                          ("depends_on_previous_sample", step > 0)),
            )
            bias = gemm_op(
                op_id=f"draft.markov_bias{step}", deps=(lookup.op_id,),
                stage=stage, layer=-1, requests=active_at_step, m=active_batch,
                n=draft.vocab_size, k=draft.markov_rank, quant=quant,
                category="lm_head", phase=mode, family="logits",
                semantic_family="dspark_markov_transition_bias",
                metadata=(("replicated", draft.markov_replicated),
                          ("weight_shard", "replicated"),
                          ("step", step)),
            )
            add = vector_op(
                op_id=f"draft.markov_add{step}",
                deps=((base_last,) if base_last else ()) + (bias.op_id,),
                stage=stage, layer=-1, requests=active_at_step,
                elements=max(1, active_batch * draft.vocab_size),
                ops_per_element=2.0, dtype="fp32", family="logits_add",
                phase=mode, recipe="catalog_dspark_markov_add",
                semantic_family="dspark_base_plus_markov_bias",
                shape=(active_batch, draft.vocab_size),
                metadata=(("step", step), ("sequential_step_dependency", True),
                          ("active_proposal_requests", active_at_step)),
            )
            sample = vector_op(
                op_id=f"draft.sample{step}", deps=(add.op_id,), stage=stage,
                layer=-1, requests=active_at_step, elements=active_batch,
                ops_per_element=2.0, dtype="fp32", family="sampling",
                phase=mode, recipe="catalog_dspark_sequential_sample",
                shape=(active_batch,),
                metadata=(("step", step),
                          ("next_step_depends_on_this", True)),
            )
            ops.extend([lookup, bias, add, sample])
            sample_ids.append(sample.op_id)
        last = sample_ids[-1]
        if draft.confidence_enabled and draft.confidence_executes:
            confidence = gemm_op(
                op_id="draft.confidence", deps=(proposal_last, sample_ids[-1]),
                stage=stage, layer=-1, requests=requests, m=emitted_total, n=1,
                k=draft.confidence_input_width or (draft.hidden_size + draft.markov_rank),
                quant=quant, category="norm", phase=mode, family="gemm",
                semantic_family="dspark_confidence_head",
                metadata=(("confidence_input_width", draft.confidence_input_width),
                          ("with_markov", True),
                          ("output_scores_per_token", 1)),
            )
            ops.append(confidence)
            last = confidence.op_id
    else:
        if base_last is not None:
            sample = vector_op(
                op_id="draft.sample0", deps=(base_last,), stage=stage, layer=-1,
                requests=requests, elements=len(active_emitted),
                ops_per_element=2.0, dtype="fp32", family="sampling",
                phase=mode, recipe="catalog_draft_sample",
                shape=(len(active_emitted),),
                metadata=(("base_logits", True),),
            )
            ops.append(sample)
            last = sample.op_id
    # Candidate generation only.  Target verification is lowered jointly by
    # graph_lowering, which reuses the target body logits/KV; this function
    # never emits a second target body.
    return DraftResult(
        kind=kind, proposal_tokens=scheduled_total, verify_tokens=0,
        accepted_upper=0, ops=ops, proposal_last=proposal_last,
        verify_last="", commit_id="",
        configured_proposal_tokens=draft.proposal_tokens,
        scheduled_proposal_counts=scheduled,
        emitted_proposal_counts=emitted,
        emitted_proposal_tokens=emitted_total,
        unverified_proposal_tokens=0,
        discarded_proposal_tokens=0,
        ordinary_query_rows=ordinary_rows,
        target_query_rows=ordinary_rows,
    )


def lower_standalone_input(
    *, draft: DraftSpec, model: ModelSpec, parallel: ParallelSpec,
    quant: QuantSpec, mode: str, proposal: int, requests: tuple[str, ...],
    stage: int = 0,
) -> tuple[list[OpSpec], str]:
    """Standalone DSpark target-feature projection + norm before body layers."""
    if draft.kind != "standalone_draft":
        return [], ""
    inp = gemm_op(
        op_id="dspark.input_proj", deps=(), stage=stage, layer=-1,
        requests=requests, m=proposal, n=draft.hidden_size,
        k=draft.input_width, quant=quant, category="dspark_input",
        phase=mode, family="gemm",
        semantic_family="dspark_target_feature_input_projection",
        metadata=(("target_layer_ids", draft.target_layer_ids),
                  ("input_width", draft.input_width)),
    )
    embed_read = vector_op(
        op_id="dspark.target_embedding_read", deps=(inp.op_id,), stage=stage,
        layer=-1, requests=requests,
        elements=max(1, proposal * draft.hidden_size), ops_per_element=1.0,
        dtype="bf16", family="embedding", phase=mode,
        recipe="catalog_dspark_target_embedding_read",
        semantic_family="target_shared_embedding_read", read_write=1.0,
        shape=(proposal, draft.hidden_size),
        metadata=(("shared_owner", True), ("owner", "target_embed_tokens")),
    )
    norm = vector_op(
        op_id="dspark.input_norm", deps=(embed_read.op_id,), stage=stage,
        layer=-1, requests=requests,
        elements=max(1, proposal * draft.hidden_size),
        ops_per_element=5.0, dtype="bf16", family="rmsnorm", phase=mode,
        recipe="catalog_dspark_input_norm",
        shape=(proposal, draft.hidden_size),
        metadata=(("tokens", proposal),
                  ("reduction_width", draft.hidden_size),
                  ("norm_type", "rmsnorm"),
                  ("norm_eps", float(getattr(model, "norm_eps", 1e-6)))),
    )
    return [inp, embed_read, norm], norm.op_id


def lower_standalone_outputs(
    *, draft: DraftSpec, model: ModelSpec, parallel: ParallelSpec,
    quant: QuantSpec, mode: str, proposal: int, requests: tuple[str, ...],
    deps: str, stage: int = 0,
) -> tuple[list[OpSpec], str]:
    """Standalone DSpark base-logit + sequential Markov proposal path."""
    if draft.kind != "standalone_draft":
        return [], deps
    if parallel.pp not in tuple(draft.pp_capability or (1,)):
        raise A3InputError(
            f"catalog_draft_backend_pp_unsupported:{draft.kind}:pp{parallel.pp}"
        )
    ops: list[OpSpec] = []
    batch = max(1, len(requests))
    base = gemm_op(
        op_id="dspark.base_logits", deps=(deps,), stage=stage, layer=-1,
        requests=requests, m=proposal, n=draft.vocab_size,
        k=draft.hidden_size, quant=quant, category="target_alias_lm_head",
        phase=mode, family="logits",
        semantic_family="dspark_base_logits_target_shared_head",
        metadata=(("shared_owner", True), ("owner", "target_lm_head")),
    )
    ops.append(base)
    last = base.op_id
    sample_ids: list[str] = []
    steps = max(1, draft.block_size)
    lookup_last = deps
    for step in range(steps):
        lookup = vector_op(
            op_id=f"dspark.markov_lookup{step}",
            deps=(lookup_last if step == 0 else sample_ids[-1],),
            stage=stage, layer=-1, requests=requests, elements=batch,
            ops_per_element=2.0, dtype="bf16", family="embedding", phase=mode,
            recipe="catalog_dspark_markov_lookup",
            semantic_family="dspark_markov_previous_token_lookup",
            read_write=1.0, shape=(batch, draft.markov_rank),
            hbm_bytes_override=float(batch * draft.markov_rank * 2),
            metadata=(("lookup_table", "markov_w1"),
                      ("table_shape", (draft.vocab_size, draft.markov_rank)),
                      ("replicated", True), ("step", step)),
        )
        bias = gemm_op(
            op_id=f"dspark.markov_bias{step}", deps=(lookup.op_id,),
            stage=stage, layer=-1, requests=requests, m=batch,
            n=draft.vocab_size, k=draft.markov_rank, quant=quant,
            category="lm_head", phase=mode, family="logits",
            semantic_family="dspark_markov_transition_bias",
            metadata=(("replicated", True), ("weight_shard", "replicated"),
                      ("step", step)),
        )
        add = vector_op(
            op_id=f"dspark.markov_add{step}", deps=(base.op_id, bias.op_id),
            stage=stage, layer=-1, requests=requests,
            elements=max(1, batch * draft.vocab_size), ops_per_element=2.0,
            dtype="fp32", family="logits_add", phase=mode,
            recipe="catalog_dspark_markov_add",
            semantic_family="dspark_base_plus_markov_bias",
            shape=(batch, draft.vocab_size),
            metadata=(("step", step), ("sequential_step_dependency", True)),
        )
        sample = vector_op(
            op_id=f"dspark.sample{step}", deps=(add.op_id,), stage=stage,
            layer=-1, requests=requests, elements=batch, ops_per_element=2.0,
            dtype="fp32", family="sampling", phase=mode,
            recipe="catalog_dspark_sequential_sample", shape=(batch,),
            metadata=(("step", step), ("next_step_depends_on_this", True)),
        )
        ops.extend([lookup, bias, add, sample])
        sample_ids.append(sample.op_id)
    last = sample_ids[-1]
    if draft.confidence_enabled and draft.confidence_executes:
        confidence = gemm_op(
            op_id="dspark.confidence", deps=(deps, sample_ids[-1]), stage=stage,
            layer=-1, requests=requests, m=proposal, n=1,
            k=draft.confidence_input_width or (draft.hidden_size + draft.markov_rank),
            quant=quant, category="norm", phase=mode, family="gemm",
            semantic_family="dspark_confidence_head",
            metadata=(("confidence_input_width", draft.confidence_input_width),
                      ("with_markov", True), ("output_scores_per_token", 1)),
        )
        ops.append(confidence)
        last = confidence.op_id
    proposal_out = barrier_op(
        op_id="dspark.proposal", deps=(last,), stage=stage, layer=-1,
        requests=requests, phase=mode, recipe="catalog_dspark_proposal_heads",
        shape=(proposal, draft.block_size),
        metadata=(("block_size", draft.block_size),
                  ("accepted_is_data_dependent", True),
                  ("confidence_executed", bool(draft.confidence_executes))),
    )
    ops.append(proposal_out)
    return ops, proposal_out.op_id
