"""Heterogeneous catalog graph lowering for the nine real model configs.

[Role]
  Orchestrate body layers, optional MTP/DSpark, media/Engram branches, mHC,
  shared cache owners and PP stage transfers into the OpSpec/IterationWork DAG
  consumed by predictive and normative paths.

[Invariants]
  * Body layer count equals the native schedule; optional draft is never
    appended to ``model.layer_plans`` and never executes on a plain forward.
  * Every layer's actual attention/FFN family is lowered; there is no generic
    MLA/GQA substitution for KDA, GDN, latent MoE, DSA or dense MLP.
  * Attention dependencies include Q and KV producers; output projections
    depend on every request's attention output.
  * Shared long-KV/index storage is produced once per declared owner and read
    by each consumer; cross-stage consumers get an explicit transfer event.
  * Media positions expand the real text sequence and the vision tower executes
    every declared block when media is present.  Text-only skips the tower.
  * Engram executes its device-table path; optional host staging is additional
    dependent work, not a replacement.
  * A plain forward excludes draft/MTP work.  Explicit draft mode derives
    proposal/verify shapes and committed accepted state.

[Timeouts & Locks]
  Bounded by actual layer/request/vision/draft counts; no I/O, locks or devices.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .a3_catalog_attention import lower_attention
from .a3_catalog_attention_common import _causal_pairs, _long_selected_pairs
from .a3_catalog_aux import (
    lower_optional_draft, lower_standalone_input, lower_standalone_outputs,
    optional_draft_spec, scheduled_proposal_counts,
)
from .a3_catalog_cache import (
    IndexReadCost, attention_rows, completed_rows, index_selection_cost,
    new_completed_rows, request_cache_breakdown, state_bytes_per_request_per_rank,
)
from .a3_catalog_engram import lower_engram
from .a3_catalog_media import lower_media
from .a3_catalog_moe import lower_moe, routing_stats
from .a3_catalog_ops import (
    barrier_op, cache_store_op, comm_op, gemm_op, vector_op,
)
from .a3_catalog_types import QuantSpec
from .a3_ops import IterationWork, OpSpec, RoutingStats, _op
from .a3_types import A3InputError, ParallelSpec, RuntimeSpec, dtype_bytes
from .a3_workload import RequestView, select_recipe


def _reqs(views: Sequence[Any]) -> tuple[str, ...]:
    return tuple(view.request_id for view in views)


def _effective_views(
    views: Sequence[RequestView], media_positions: int,
) -> tuple[tuple[RequestView, ...], int]:
    if media_positions <= 0:
        return tuple(views), sum(max(1, v.q_len) for v in views)
    attrs: list[RequestView] = []
    placed = False
    for view in views:
        q_len = view.q_len
        context = view.context
        if not placed:
            phase = str(view.semantic_phase or view.phase)
            if phase == "prefill":
                q_len += media_positions
            else:
                context += media_positions
            placed = True
        attrs.append(replace(view, q_len=max(1, q_len), context=max(0, context)))
    total = sum(v.q_len for v in attrs)
    return tuple(attrs), total


def _media_assignments(
    runtime: RuntimeSpec, views: Sequence[RequestView],
) -> tuple[dict[str, int], bool]:
    meta = dict(runtime.metadata or ())
    raw = meta.get("media_tokens_by_request") or meta.get("media_request_tokens")
    assignments: dict[str, int] = {}
    pre_expanded = str(meta.get("media_positions_pre_expanded", "false")).lower() in {"1", "true", "yes"}
    if raw:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise A3InputError("runtime.media_tokens_by_request_invalid") from exc
        if not isinstance(parsed, dict):
            raise A3InputError("runtime.media_tokens_by_request_invalid")
        assignments = {str(k): max(0, int(v)) for k, v in parsed.items()}
    elif runtime.media_tokens:
        if views:
            assignments[views[0].request_id] = int(runtime.media_tokens)
    encoded = meta.get("media_encoded_requests")
    if encoded:
        try:
            encoded_ids = set(json.loads(encoded) if isinstance(encoded, str) else encoded)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise A3InputError("runtime.media_encoded_requests_invalid") from exc
        for request_id in encoded_ids:
            assignments.pop(str(request_id), None)
    return assignments, pre_expanded


def _rename_media_ops(ops: Sequence[OpSpec], suffix: str) -> list[OpSpec]:
    old_ids = {op.op_id for op in ops}
    renamed: list[OpSpec] = []
    for op in ops:
        deps = tuple(f"{dep}{suffix}" if dep in old_ids else dep for dep in op.deps)
        renamed.append(replace(op, op_id=f"{op.op_id}{suffix}", deps=deps))
    return renamed


def _next_proposal_counts(
    views: Sequence[Any], current: Sequence[int], draft: Any, runtime: Any,
) -> tuple[int, ...]:
    """Symbolic next-iteration candidate counts; never actual token IDs."""
    meta = dict(getattr(runtime, "metadata", ()) or ())
    mapping: dict[str, int] = {}
    raw = meta.get("draft_next_counts")
    if raw:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            mapping = {str(k): max(0, int(v)) for k, v in dict(parsed).items()}
        except (TypeError, ValueError, json.JSONDecodeError):
            mapping = {}
    result: list[int] = []
    for index, view in enumerate(views):
        if str(view.request_id) in mapping:
            result.append(mapping[str(view.request_id)])
            continue
        explicit = getattr(view, "draft_next_proposal_tokens", None)
        if explicit is not None:
            result.append(max(0, int(explicit)))
            continue
        if bool(view.samples) and str(view.phase) == "prefill":
            result.append(max(0, int(draft.proposal_tokens)))
        elif int(current[index]) > 0:
            result.append(max(0, int(draft.proposal_tokens)))
        else:
            result.append(0)
    return tuple(result)


def _rename_draft_prefix(ops: Sequence[OpSpec], prefix: str) -> list[OpSpec]:
    ids = {op.op_id for op in ops}
    renamed: list[OpSpec] = []
    for op in ops:
        deps = tuple(
            f"{prefix}{dep}" if dep in ids else dep for dep in op.deps
        )
        new_id = f"{prefix}{op.op_id}" if op.op_id in ids else op.op_id
        renamed.append(replace(op, op_id=new_id, deps=deps))
    return renamed


def _stage_of(parallel: ParallelSpec, layers: int) -> Any:
    if parallel.pp <= 1:
        return lambda index: 0
    per = layers // parallel.pp

    def stage(index: int) -> int:
        return min(parallel.pp - 1, index // max(1, per))

    return stage


def _norm(
    *, op_id: str, deps: str | tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], tokens: int, width: int, mode: str,
    semantic: str = "rmsnorm", norm_eps: float = 1e-6,
) -> OpSpec:
    return vector_op(
        op_id=op_id, deps=(deps,) if isinstance(deps, str) else tuple(deps),
        stage=stage, layer=layer, requests=requests,
        elements=max(1, tokens * width), ops_per_element=5.0, dtype="bf16",
        family="rmsnorm", phase=mode, recipe=f"catalog_{semantic}",
        semantic_family=semantic, shape=(tokens, width),
        metadata=(("tokens", tokens), ("reduction_width", width),
                  ("norm_type", "rmsnorm"), ("norm_eps", float(norm_eps))),
    )


def _residual_add(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], tokens: int, width: int, mode: str,
) -> OpSpec:
    return vector_op(
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
        elements=max(1, tokens * width), ops_per_element=2.0, dtype="bf16",
        family="elementwise", phase=mode, recipe="catalog_residual_add",
        semantic_family="residual_add", shape=(tokens, width),
    )


def _mhc_pre(
    *, prefix: str, plan: Any, deps: str, stage: int, requests: tuple[str, ...],
    tokens: int, model: Any, mode: str,
) -> tuple[list[OpSpec], str, int]:
    h = model.hidden_size
    hc = max(1, plan.hc_mult)
    mix = (2 + hc) * hc
    ops: list[OpSpec] = []
    fn = gemm_op(
        op_id=f"{prefix}.hc_fn", deps=(deps,), stage=stage, layer=plan.index,
        requests=requests, m=tokens, n=mix, k=hc * h, quant=model.quant,
        category="mhc", phase=mode, family="gemm",
        recipe="catalog_mhc_fn",
        semantic_family="mhc_hyper_connection_projection",
        metadata=(("hc_mult", hc), ("mix_width", mix),
                  ("residual_width", hc * h)),
    )
    sink = vector_op(
        op_id=f"{prefix}.hc_sinkhorn", deps=(fn.op_id,), stage=stage,
        layer=plan.index, requests=requests, elements=max(1, tokens * mix),
        ops_per_element=float(max(1, plan.hc_sinkhorn_iters)) * 4.0,
        dtype="fp32", family="mhc", phase=mode, recipe="catalog_mhc_sinkhorn",
        semantic_family="mhc_sinkhorn_normalization",
        metadata=(("sinkhorn_iters", plan.hc_sinkhorn_iters),),
    )
    mixop = vector_op(
        op_id=f"{prefix}.hc_mix", deps=(sink.op_id,), stage=stage,
        layer=plan.index, requests=requests,
        elements=max(1, tokens * hc * h), ops_per_element=3.0,
        dtype="bf16", family="mhc", phase=mode, recipe="catalog_mhc_residual_mix",
        semantic_family="mhc_residual_stream_mix", shape=(tokens, hc, h),
    )
    ops.extend([fn, sink, mixop])
    return ops, mixop.op_id, hc * h


def _mhc_post(
    *, prefix: str, plan: Any, deps: str, stage: int, requests: tuple[str, ...],
    tokens: int, model: Any, mode: str,
) -> tuple[list[OpSpec], str]:
    h = model.hidden_size
    hc = max(1, plan.hc_mult)
    mix = (2 + hc) * hc
    fn = gemm_op(
        op_id=f"{prefix}.hc_post_fn", deps=(deps,), stage=stage,
        layer=plan.index, requests=requests, m=tokens, n=mix, k=hc * h,
        quant=model.quant, category="mhc", phase=mode, family="gemm",
        recipe="catalog_mhc_post_fn",
        semantic_family="mhc_post_mix_projection",
    )
    sink = vector_op(
        op_id=f"{prefix}.hc_post_sinkhorn", deps=(fn.op_id,), stage=stage,
        layer=plan.index, requests=requests, elements=max(1, tokens * mix),
        ops_per_element=float(max(1, plan.hc_sinkhorn_iters)) * 4.0,
        dtype="fp32", family="mhc", phase=mode, recipe="catalog_mhc_post_sinkhorn",
        semantic_family="mhc_post_sinkhorn",
    )
    out = vector_op(
        op_id=f"{prefix}.hc_post_out", deps=(sink.op_id,), stage=stage,
        layer=plan.index, requests=requests,
        elements=max(1, tokens * hc * h), ops_per_element=3.0, dtype="bf16",
        family="mhc", phase=mode, recipe="catalog_mhc_post_mix",
        semantic_family="mhc_post_residual_mix", shape=(tokens, hc, h),
    )
    return [fn, sink, out], out.op_id


def _dense_ffn(
    *, prefix: str, plan: Any, deps: str, stage: int, requests: tuple[str, ...],
    tokens: int, model: Any, parallel: ParallelSpec, mode: str,
) -> tuple[list[OpSpec], str]:
    h = model.hidden_size
    width = max(1, plan.intermediate_size // parallel.tp)
    gate = gemm_op(
        op_id=f"{prefix}.gate_up", deps=(deps,), stage=stage,
        layer=plan.index, requests=requests, m=tokens, n=2 * width, k=h,
        quant=model.quant, category="dense_mlp", phase=mode,
        family="gate_up_projection", recipe="catalog_dense_gate_up",
        semantic_family="dense_gate_up_projection",
    )
    act_elements = max(1, tokens * width)
    act_bytes = dtype_bytes("bf16")
    act = vector_op(
        op_id=f"{prefix}.act", deps=(gate.op_id,), stage=stage,
        layer=plan.index, requests=requests, elements=act_elements,
        ops_per_element=4.0, dtype="bf16", family="swiglu", phase=mode,
        recipe="catalog_dense_swiglu",
        shape=(tokens, width),
        read_bytes_override=2.0 * act_elements * act_bytes,
        write_bytes_override=1.0 * act_elements * act_bytes,
        metadata=(("tokens", tokens), ("output_width", width),
                  ("input_width", 2 * width), ("packed_gate_input", True),
                  ("activation", "silu"),
                  ("swiglu_contract", "packed_gate_up"),
                  ("read_halves", 2), ("write_outputs", 1)),
    )
    down = gemm_op(
        op_id=f"{prefix}.down", deps=(act.op_id,), stage=stage,
        layer=plan.index, requests=requests, m=tokens, n=h, k=width,
        quant=model.quant, category="dense_mlp", phase=mode,
        family="down_projection", recipe="catalog_dense_down",
        semantic_family="dense_down_projection",
    )
    return [gate, act, down], down.op_id


def _p2p(
    *, prefix: str, deps: str, stage: int, layer: int,
    requests: tuple[str, ...], tokens: int, width: int, mode: str,
) -> OpSpec:
    return comm_op(
        op_id=f"{prefix}.pp_p2p", deps=(deps,), stage=stage, layer=layer,
        requests=requests, family="pp_p2p", phase=mode,
        recipe="catalog_pp_p2p", link_bytes=float(tokens * width * 2),
        shape=(tokens, width),
        metadata=(("scope", "adjacent_pp_stages"),),
    )


def infer_catalog_iteration(
    plan: Any, model: Any, runtime: RuntimeSpec, parallel: ParallelSpec,
    *, ep_global_scale: float | None = None,
) -> IterationWork:
    if not model.layer_plans:
        raise A3InputError("catalog_layer_plans_missing")
    if model.quant is None:
        raise A3InputError("catalog_quantization_plan_missing")
    recipe_payload, guards = select_recipe(
        model=model, plan=plan, runtime=runtime, parallel=parallel,
    )
    quant: QuantSpec = model.quant
    mode = recipe_payload["mode"]
    stage_of = _stage_of(parallel, model.layers)
    metadata = dict(runtime.metadata or ())
    draft_mode = str(metadata.get("draft_mode", "") or "").lower()
    # ---- media expansion and effective text views ----
    assignments, pre_expanded = _media_assignments(runtime, plan.requests)
    media_ops: list[OpSpec] = []
    media_lasts: list[str] = []
    media_text_positions = 0
    media_patch_tokens = 0
    if assignments:
        pending = [
            (view, assignments.get(view.request_id, 0)) for view in plan.requests
        ]
        pending = [
            (view, count) for view, count in pending
            if count > 0 and not (
                str(view.semantic_phase or view.phase) == "decode"
                or view.context > 0
            )
        ]
        for index, (media_request, media_count) in enumerate(pending):
            expansion = lower_media(
                model=model, runtime=runtime, parallel=parallel, quant=quant,
                mode=mode, views=(media_request,), stage=0,
                media_tokens_override=media_count,
            )
            if index == 0:
                media_ops.extend(expansion.ops)
            else:
                media_ops.extend(_rename_media_ops(
                    expansion.ops, f".req{media_request.request_id}",
                ))
            media_lasts.append(
                expansion.last if index == 0
                else f"{expansion.last}.req{media_request.request_id}"
            )
            media_text_positions += expansion.text_positions
            media_patch_tokens += expansion.patch_tokens
    views, text_tokens = _effective_views(
        plan.requests,
        0 if pre_expanded or not media_lasts else media_text_positions,
    )
    original_views = views
    standalone = model.is_standalone_draft
    joint_draft = False
    joint_counts: tuple[int, ...] = ()
    draft_spec = None
    if draft_mode and not standalone and model.has_optional_draft:
        draft_spec = optional_draft_spec(model, draft_mode)
        joint_counts = scheduled_proposal_counts(views)
        joint_draft = True
        if any(joint_counts):
            views = tuple(
                replace(view, q_len=max(1, int(view.q_len)) + count)
                for view, count in zip(views, joint_counts)
            )
    text_tokens = sum(max(1, int(view.q_len)) for view in views)
    valid_tokens = text_tokens
    graph_bucket = recipe_payload["graph_bucket"]
    padded_tokens = max(valid_tokens, graph_bucket or valid_tokens)
    requests = _reqs(views)
    h = model.hidden_size
    ops: list[OpSpec] = []
    embedding: OpSpec | None = None
    joint_ready = ""
    if joint_draft and any(joint_counts):
        ready = barrier_op(
            op_id="draft.prev_proposals_ready", deps=(), stage=0, layer=-1,
            requests=_reqs(views), phase=mode,
            recipe="catalog_draft_prev_candidates_ready",
            shape=(sum(joint_counts),),
            metadata=(("provenance", "previous_iteration_draft_candidates"),
                      ("scheduled_verify_counts", joint_counts),
                      ("configured_proposal_tokens", draft_spec.proposal_tokens),
                      ("symbolic_state_no_tensor_contents", True),
                      ("ready_after", "previous_iteration_target_sampling"),
                      ("prior_target_completion_dependency", True),
                      ("readiness_before_target_body", True)),
        )
        ops.append(ready)
        joint_ready = ready.op_id
    if not standalone:
        embedding = vector_op(
            op_id="embedding",
            deps=(joint_ready,) if joint_ready else (), stage=0, layer=0,
            requests=requests, elements=max(1, valid_tokens * h),
            ops_per_element=1.0, dtype="bf16", family="embedding", phase=mode,
            recipe="catalog_embedding_gather", shape=(valid_tokens, h),
        )
        ops.append(embedding)
        previous = embedding.op_id
    else:
        previous = ""
    if media_lasts and media_ops and embedding is not None:
        ops.extend(media_ops)
        join = barrier_op(
            op_id="media_text_join", deps=(embedding.op_id,) + tuple(media_lasts),
            stage=0, layer=0, requests=requests, phase=mode,
            recipe="catalog_media_text_join", shape=(padded_tokens, h),
            metadata=(("media_text_positions", media_text_positions),
                      ("media_requests", len(media_lasts)),
                      ("patch_tokens", media_patch_tokens)),
        )
        ops.append(join)
        previous = join.op_id
    # Standalone DSpark consumes target features before its five body layers.
    if standalone:
        draft_spec = model.auxiliary.dspark
        draft_spec.validate()
        proposal_per_request = draft_spec.proposal_tokens
        source_views = tuple(views)
        draft_views = tuple(
            RequestView(
                request_id=f"draft.{index}", q_len=proposal_per_request,
                context=max(0, int(getattr(source_views[index], "context", 0))),
                samples=False, phase=mode, semantic_phase=mode,
                execution_mode=mode,
            )
            for index in range(max(1, len(source_views)))
        )
        proposal_total = proposal_per_request * len(draft_views)
        input_ops, input_last = lower_standalone_input(
            draft=draft_spec, model=model, parallel=parallel, quant=quant,
            mode=mode, proposal=proposal_total, requests=_reqs(draft_views),
            stage=0,
        )
        ops.extend(input_ops)
        previous = input_ops[-1].op_id if input_ops else previous
        views = draft_views
        valid_tokens = proposal_total
        padded_tokens = max(proposal_total, graph_bucket or proposal_total)
        requests = _reqs(views)
    # ---- body layer traversal ----
    cache_ids: dict[Any, str] = {}
    topk_ids: dict[int, str] = {}
    index_cache_ids: dict[int, str] = {}
    candidate_ids: dict[int, str] = {}
    routing: RoutingStats | None = None
    for layer_index, layer in enumerate(model.layer_plans):
        stage = stage_of(layer_index)
        prefix = f"s{stage}.l{layer_index}"
        layer_requests = _reqs(views)
        hc_ops: list[OpSpec] = []
        hidden_width = h
        norm_deps = previous
        if layer.hyper_connections:
            hc_ops, norm_deps, hidden_width = _mhc_pre(
                prefix=prefix, plan=layer, deps=previous, stage=stage,
                requests=layer_requests, tokens=padded_tokens, model=model,
                mode=mode,
            )
        norm = _norm(
            op_id=f"{prefix}.norm", deps=norm_deps, stage=stage,
            layer=layer_index, requests=layer_requests, tokens=padded_tokens,
            width=h, mode=mode, semantic=f"{layer.attention}_pre_norm",
            norm_eps=model.norm_eps,
        )
        ops.extend(hc_ops)
        ops.append(norm)
        attn = lower_attention(
            prefix=prefix, plan=layer, views=views, stage=stage,
            deps=norm.op_id, model=model, parallel=parallel, quant=quant,
            mode=mode, padded_tokens=padded_tokens, cache_ids=cache_ids,
            topk_ids=topk_ids, index_cache_ids=index_cache_ids,
            candidate_ids=candidate_ids, stage_of=stage_of,
        )
        ops.extend(attn.ops)
        if attn.cache_store_id:
            cache_ids[layer.index] = attn.cache_store_id
        if attn.index_store_id:
            cache_ids[("index", layer.index)] = attn.index_store_id
            index_cache_ids[layer.index] = attn.index_store_id
        if attn.topk_id:
            topk_ids[layer.index] = attn.topk_id
        if attn.candidate_id:
            candidate_ids[layer.index] = attn.candidate_id
        residual = _residual_add(
            op_id=f"{prefix}.attn_residual",
            deps=(norm_deps, attn.last), stage=stage, layer=layer_index,
            requests=layer_requests, tokens=padded_tokens,
            width=hidden_width, mode=mode,
        )
        ops.append(residual)
        last = residual.op_id
        if layer.residual_block_size:
            res_norm = _norm(
                op_id=f"{prefix}.attn_res_norm", deps=last, stage=stage,
                layer=layer_index, requests=layer_requests, tokens=padded_tokens,
                width=h, mode=mode, semantic="attn_residual_norm",
                norm_eps=model.norm_eps,
            )
            res_proj = vector_op(
                op_id=f"{prefix}.attn_res_proj", deps=(res_norm.op_id,),
                stage=stage, layer=layer_index, requests=layer_requests,
                elements=max(1, padded_tokens * h), ops_per_element=2.0,
                dtype="bf16", family="elementwise", phase=mode,
                recipe="catalog_attn_residual_projection",
                semantic_family="attn_residual_projection",
                shape=(padded_tokens, h),
            )
            ops.extend([res_norm, res_proj])
            last = res_proj.op_id
        if layer.engram:
            engram_ops, last = lower_engram(
                prefix=prefix, plan=layer, stage=stage, deps=last,
                model=model, parallel=parallel, quant=quant, mode=mode,
                padded_tokens=padded_tokens, views=views, runtime=runtime,
            )
            ops.extend(engram_ops)
        norm2 = _norm(
            op_id=f"{prefix}.norm2", deps=last, stage=stage, layer=layer_index,
            requests=layer_requests, tokens=padded_tokens, width=h,
            mode=mode, semantic=f"{layer.ffn}_ffn_norm",
            norm_eps=model.norm_eps,
        )
        ops.append(norm2)
        if layer.ffn == "dense":
            ffn_ops, ffn_last = _dense_ffn(
                prefix=prefix, plan=layer, deps=norm2.op_id, stage=stage,
                requests=layer_requests, tokens=padded_tokens, model=model,
                parallel=parallel, mode=mode,
            )
        else:
            routing = routing_stats(
                model, layer, parallel, padded_tokens,
                capacity_factor=runtime.moe_capacity_factor,
            )
            ffn_ops, ffn_last = lower_moe(
                prefix=prefix, plan=layer, stage=stage, deps=norm2.op_id,
                model=model, parallel=parallel, quant=quant, mode=mode,
                padded_tokens=padded_tokens, views=views, routing=routing,
            )
        ops.extend(ffn_ops)
        if parallel.tp > 1 and layer.ffn == "dense":
            ar = comm_op(
                op_id=f"{prefix}.tp_ffn", deps=(ffn_last,), stage=stage,
                layer=layer_index, requests=layer_requests, family="allreduce",
                phase=mode, recipe="catalog_tp_allreduce",
                link_bytes=2.0 * (parallel.tp - 1) / parallel.tp * padded_tokens
                * h * dtype_bytes(model.activation_dtype),
                shape=(padded_tokens, h),
                metadata=(("collective", "allreduce"), ("after", "ffn")),
            )
            ops.append(ar)
            ffn_last = ar.op_id
        ffn_residual = _residual_add(
            op_id=f"{prefix}.ffn_residual", deps=(last, ffn_last), stage=stage,
            layer=layer_index, requests=layer_requests, tokens=padded_tokens,
            width=hidden_width, mode=mode,
        )
        ops.append(ffn_residual)
        last = ffn_residual.op_id
        if layer.residual_block_size:
            mlp_norm = _norm(
                op_id=f"{prefix}.mlp_res_norm", deps=last, stage=stage,
                layer=layer_index, requests=layer_requests, tokens=padded_tokens,
                width=h, mode=mode, semantic="mlp_residual_norm",
                norm_eps=model.norm_eps,
            )
            mlp_proj = vector_op(
                op_id=f"{prefix}.mlp_res_proj", deps=(mlp_norm.op_id,),
                stage=stage, layer=layer_index, requests=layer_requests,
                elements=max(1, padded_tokens * h), ops_per_element=2.0,
                dtype="bf16", family="elementwise", phase=mode,
                recipe="catalog_mlp_residual_projection",
                semantic_family="mlp_residual_projection",
                shape=(padded_tokens, h),
            )
            ops.extend([mlp_norm, mlp_proj])
            last = mlp_proj.op_id
        if layer.hyper_connections:
            post_ops, post_last = _mhc_post(
                prefix=prefix, plan=layer, deps=last, stage=stage,
                requests=layer_requests, tokens=padded_tokens, model=model,
                mode=mode,
            )
            ops.extend(post_ops)
            last = post_last
        previous = last
        if layer_index + 1 < len(model.layer_plans):
            next_stage = stage_of(layer_index + 1)
            if next_stage != stage:
                p2p = _p2p(
                    prefix=prefix, deps=last, stage=stage, layer=layer_index,
                    requests=layer_requests, tokens=padded_tokens, width=h,
                    mode=mode,
                )
                ops.append(p2p)
                previous = p2p.op_id
    # ---- standalone DSpark output heads ----
    if standalone:
        out_ops, out_last = lower_standalone_outputs(
            draft=model.auxiliary.dspark, model=model, parallel=parallel,
            quant=quant, mode=mode,
            proposal=model.auxiliary.dspark.proposal_tokens * len(views),
            requests=_reqs(views), deps=previous,
            stage=max(0, parallel.pp - 1),
        )
        ops.extend(out_ops)
        previous = out_last if out_ops else previous
    # ---- logits/sampling (target/plain path) ----
    target_hidden_last = previous
    sample_rows = sum(1 for view in views if view.samples)
    logit_rows = (
        sum(max(1, int(view.q_len)) for view in views)
        if joint_draft else sample_rows
    )
    logit_requests = (
        tuple(view.request_id for view in views) if joint_draft
        else tuple(view.request_id for view in views if view.samples)
    )
    logits_last = ""
    if sample_rows and not standalone:
        logits = gemm_op(
            op_id="logits", deps=(previous,), stage=max(0, parallel.pp - 1),
            layer=-1, requests=logit_requests,
            m=max(1, logit_rows), n=model.vocab_size // parallel.tp, k=h,
            quant=quant, category="lm_head", phase=mode, family="logits",
            recipe="catalog_logits", semantic_family="vocab_parallel_logits",
        )
        ops.append(logits)
        logits_last = logits.op_id
        if parallel.tp > 1:
            gather = comm_op(
                op_id="logits_allgather", deps=(logits_last,), stage=max(0, parallel.pp - 1),
                layer=-1, requests=logit_requests,
                family="allgather", phase=mode, recipe="catalog_logits_allgather",
                link_bytes=(parallel.tp - 1) / parallel.tp * max(1, logit_rows)
                * model.vocab_size * dtype_bytes(model.activation_dtype),
                shape=(max(1, logit_rows), model.vocab_size),
            )
            ops.append(gather)
            logits_last = gather.op_id
        sampling = vector_op(
            op_id="sampling", deps=(logits_last,), stage=max(0, parallel.pp - 1),
            layer=-1, requests=logit_requests,
            elements=max(1, logit_rows * model.vocab_size),
            ops_per_element=1.0, dtype="fp32", family="sampling", phase=mode,
            recipe="catalog_sampling", shape=(max(1, logit_rows), model.vocab_size),
        )
        ops.append(sampling)
        previous = sampling.op_id
    # ---- joint target verification / next candidate generation ----
    joint_notes: list[str] = []
    if joint_draft and sample_rows:
        scheduled_total = sum(joint_counts)
        accepted_total = sum(
            min(count, max(0, int(getattr(view, "draft_accepted_tokens", 0) or 0)))
            for view, count in zip(original_views, joint_counts)
        )
        if scheduled_total > 0:
            verify_embedding = barrier_op(
                op_id="verify.embedding",
                deps=(embedding.op_id,) + ((joint_ready,) if joint_ready else ()),
                stage=0, layer=0, requests=_reqs(original_views), phase=mode,
                recipe="catalog_joint_verify_embedding_view",
                shape=(max(1, scheduled_total), h),
                metadata=(("view_of_target_embedding", embedding.op_id),
                          ("target_query_rows", sum(
                              max(1, int(v.q_len)) for v in views)),
                          ("scheduled_verify_rows", scheduled_total),
                          ("no_second_gather", True)),
            )
            target_producer = barrier_op(
                op_id=f"verify.{target_hidden_last}", deps=(target_hidden_last,),
                stage=max(0, parallel.pp - 1), layer=-1,
                requests=_reqs(original_views), phase=mode,
                recipe="catalog_joint_target_producer_view",
                shape=(sum(max(1, int(v.q_len)) for v in views), h),
                metadata=(("view_of", target_hidden_last),
                          ("zero_work_alias", True),
                          ("single_target_body", True)),
            )
            ops.append(target_producer)
            verify_logits = barrier_op(
                op_id="verify.target_logits",
                deps=(target_producer.op_id, logits_last),
                stage=max(0, parallel.pp - 1),
                layer=-1, requests=_reqs(original_views), phase=mode,
                recipe="catalog_joint_verify_logits_view",
                shape=(max(1, scheduled_total), model.vocab_size),
                metadata=(("view_of_target_logits", logits_last),
                          ("scheduled_verify_rows", scheduled_total),
                          ("no_second_target_body", True)),
            )
            active_verify = sum(1 for count in joint_counts if count > 0)
            emitted_current = int(draft_spec.proposal_tokens) * active_verify
            rejected_verify = max(0, scheduled_total - accepted_total)
            commit = barrier_op(
                op_id="draft.state_commit", deps=(verify_logits.op_id,),
                stage=max(0, parallel.pp - 1), layer=-1,
                requests=_reqs(original_views), phase=mode,
                recipe="catalog_joint_verify_commit",
                shape=(max(1, scheduled_total),),
                metadata=(("scheduled_proposal_counts", joint_counts),
                          ("configured_proposal_tokens", draft_spec.proposal_tokens),
                          ("accepted_tokens", accepted_total),
                          ("accepted_prefix_only", True),
                          ("rejected_verify_rows", rejected_verify),
                          ("target_body_reused", True),
                          ("emitted_candidate_cap", emitted_current),
                          ("unverified_candidate_tokens", max(
                              0, emitted_current - scheduled_total)),
                          ("discarded_candidate_tokens", max(
                              0, emitted_current - accepted_total))),
            )
            state_bytes = float(model.state_bytes_per_request or 0.0) * active_verify
            kv_row_bytes = sum(
                int(plan.cache_row_elements) for plan in model.layer_plans
            ) * dtype_bytes(quant.kv_dtype)
            rejected_rows = max(0, scheduled_total - accepted_total)
            temp_kv = float(rejected_rows * kv_row_bytes)
            rollback = vector_op(
                op_id="verify.state_rollback", deps=(commit.op_id,),
                stage=max(0, parallel.pp - 1), layer=-1,
                requests=_reqs(original_views), elements=max(1, active_verify),
                ops_per_element=2.0, dtype="fp32", family="state_rollback",
                phase=mode, recipe="catalog_joint_verify_state_rollback",
                hbm_bytes_override=2.0 * state_bytes,
                semantic_family="verify_recurrent_and_conv_state_rollback",
                metadata=(("scheduled_proposal_counts", joint_counts),
                          ("temporary_kv_bytes", temp_kv)),
            )
            discard = vector_op(
                op_id="verify.temporary_kv_discard", deps=(rollback.op_id,),
                stage=max(0, parallel.pp - 1), layer=-1,
                requests=_reqs(original_views), elements=max(1, scheduled_total),
                ops_per_element=1.0, dtype="fp32", family="cache_discard",
                phase=mode, recipe="catalog_joint_verify_kv_discard",
                hbm_bytes_override=temp_kv,
                semantic_family="verify_temporary_kv_discard",
                metadata=(("scheduled_proposal_rows", scheduled_total),
                          ("accepted_prefix_rows", accepted_total)),
            )
            ops.extend([verify_embedding, verify_logits, commit, rollback, discard])
            previous = discard.op_id
        next_counts = _next_proposal_counts(
            original_views, joint_counts, draft_spec, runtime,
        )
        if any(next_counts):
            candidate = lower_optional_draft(
                model=model, runtime=runtime, parallel=parallel, quant=quant,
                views=original_views,
                context=max((v.kv_len for v in original_views), default=0),
                mode=draft_mode, proposal_counts=next_counts,
            )
            candidate_ops = _rename_draft_prefix(candidate.ops, "draft.next.")
            rewritten_candidates: list[OpSpec] = []
            for op in candidate_ops:
                if op.op_id == "draft.next.draft.input_proj":
                    rewritten_candidates.append(replace(op, deps=(previous,)))
                else:
                    rewritten_candidates.append(op)
            ops.extend(rewritten_candidates)
            next_ready = barrier_op(
                op_id="draft.next.proposals_ready",
                deps=(f"draft.next.{candidate.ops[-1].op_id}",),
                stage=max(0, parallel.pp - 1),
                layer=-1, requests=_reqs(original_views), phase=mode,
                recipe="catalog_draft_next_candidates_ready",
                shape=(sum(next_counts),),
                metadata=(("provenance", "current_iteration_candidate_generation"),
                          ("next_scheduled_counts_hint", next_counts),
                          ("configured_proposal_tokens", draft_spec.proposal_tokens),
                          ("for_next_iteration_target_body", True)),
            )
            ops.append(next_ready)
            joint_notes.append(f"draft_next_candidates:{sum(next_counts)}")
        joint_notes.append(f"joint_target_verify_rows:{scheduled_total}")
    # ---- optional KV transfer visibility ----
    if runtime.kv_transfer.enabled and sample_rows:
        kv_per_token = runtime.kv_transfer.bytes_per_token or int(round(
            model.kv_bytes_per_token_per_layer() * model.layers,
        ))
        visible = sum(v.kv_len for v in views if v.samples)
        total = float(max(1, visible) * kv_per_token)
        chunks = max(1, int(runtime.kv_transfer.chunks))
        transfer_deps = previous
        ids = []
        for chunk in range(chunks):
            op = comm_op(
                op_id=f"kv_transfer.c{chunk}", deps=(transfer_deps,),
                stage=max(0, parallel.pp - 1), layer=-1,
                requests=tuple(v.request_id for v in views if v.samples),
                family="kv_transfer", phase=mode, recipe="catalog_kv_transfer",
                link_bytes=total / chunks, shape=(visible,),
            )
            ops.append(op)
            transfer_deps = op.op_id
            ids.append(op.op_id)
        visible_barrier = barrier_op(
            op_id="kv_transfer_visible", deps=tuple(ids), stage=max(0, parallel.pp - 1),
            layer=-1, requests=tuple(v.request_id for v in views if v.samples),
            phase=mode, recipe="catalog_kv_visibility", shape=(visible,),
        )
        ops.append(visible_barrier)
        previous = visible_barrier.op_id
    notes = [
        "catalog_heterogeneous_layers", "body_layer_schedule_preserved",
        "optional_draft_excluded_from_plain_forward",
        "shared_producer_storage_deduplicated_consumer_reads_retained",
        "quantized_weight_bytes_use_declared_packing",
        "media_positions_change_text_sequence",
        "media_assignment_per_request_before_lowering",
        "cross_stage_owner_cache_uses_explicit_transfer",
        *joint_notes,
    ]
    if standalone:
        notes.append("standalone_draft_target_features_are_model_inputs")
    return IterationWork(
        ops=tuple(ops), plan=plan, stage_count=parallel.pp,
        valid_tokens=valid_tokens, padded_tokens=padded_tokens,
        graph_bucket=graph_bucket, mode=mode,
        recipe=recipe_payload["recipe"], routing=routing,
        recipe_guards=guards, notes=tuple(notes),
    )


def _linear_flops_only(plan: Any, model: Any, positions: int) -> float:
    h = model.hidden_size
    flops = 0.0
    if plan.attention == "gqa":
        flops += 2.0 * positions * h * (
            (2 if plan.q_output_gate else 1) * plan.num_heads
            + 2 * plan.num_kv_heads
        ) * plan.head_dim
        flops += 2.0 * positions * plan.num_heads * plan.head_dim * h
    elif plan.attention == "mla":
        heads = plan.num_heads
        qk = plan.qk_nope_head_dim + plan.qk_rope_head_dim
        if plan.mla_variant == "latent_mqa":
            flops += 2.0 * positions * (
                h * plan.q_lora_rank + plan.q_lora_rank * heads * qk
                + h * plan.kv_lora_rank + h * heads * plan.head_dim
            )
        else:
            flops += 2.0 * positions * (
                h * plan.q_lora_rank + plan.q_lora_rank * heads * qk
                + h * (plan.kv_lora_rank + plan.qk_rope_head_dim)
                + plan.kv_lora_rank * heads
                * (plan.qk_nope_head_dim + plan.v_head_dim)
                + h * heads * plan.v_head_dim
            )
    elif plan.attention == "kda":
        d = plan.linear_heads * plan.linear_head_dim
        flops += 2.0 * positions * h * (3 * d + d) + 2.0 * positions * d * h
        flops += 2.0 * positions * (
            plan.linear_head_dim * h + d * plan.linear_head_dim
        )
    elif plan.attention == "gdn":
        key_w = plan.linear_key_heads * plan.linear_key_head_dim
        value_w = plan.linear_value_heads * plan.linear_head_dim
        flops += 2.0 * positions * h * (
            2 * key_w + 2 * value_w + 2 * plan.linear_value_heads
        )
        flops += 2.0 * positions * value_w * h
    if plan.ffn == "dense":
        flops += 6.0 * positions * h * plan.intermediate_size
    else:
        flops += 2.0 * positions * h * plan.num_experts
        flops += 2.0 * positions * h * plan.expert_latent_size * 2.0
        flops += 6.0 * positions * plan.top_k * (
            plan.expert_latent_size or h
        ) * plan.moe_intermediate_size
        flops += 6.0 * positions * h * plan.shared_intermediate_size
    if plan.hyper_connections:
        hc = max(1, plan.hc_mult)
        flops += 4.0 * positions * (2 + hc) * hc * h
    return float(flops)


def _attention_flops_only(
    plan: Any, model: Any, *, prefill_q: int, decode: int, context: int,
) -> float:
    h = model.hidden_size
    facts = dict(model.catalog_facts)
    if plan.attention == "gqa":
        heads = plan.num_heads
        if plan.noncausal_attention:
            pairs = (prefill_q * context + prefill_q * prefill_q
                     if prefill_q > 0 else 0.0)
        else:
            pairs = _causal_pairs(prefill_q, context, 0) if prefill_q > 0 else 0.0
        if decode > 0:
            pairs += _causal_pairs(decode, context + prefill_q, 0)
        return 4.0 * heads * plan.head_dim * pairs
    if plan.attention == "mla":
        heads = plan.num_heads
        use_dim = plan.head_dim if plan.mla_variant == "latent_mqa" else (
            plan.kv_lora_rank + plan.qk_rope_head_dim
        )
        v_dim = plan.head_dim if plan.mla_variant == "latent_mqa" else plan.kv_lora_rank
        native = str(facts.get("profile", ""))
        cap = 0
        if native in {"deepseek_v4", "deepseek_v41"} and plan.compress_ratio > 0:
            cap = int(facts.get("index_topk", 0) or 0)
            if native == "deepseek_v4" and plan.compress_ratio == 128:
                cap = 0
        def pairs_for(q: int, ctx: int) -> float:
            if q <= 0:
                return 0.0
            long_pairs = (
                _long_selected_pairs(q, ctx, plan.compress_ratio, cap)
                if plan.compress_ratio > 0 else 0.0
            )
            return long_pairs + _causal_pairs(q, ctx, plan.window)
        pairs = pairs_for(prefill_q, context)
        if decode > 0:
            pairs += pairs_for(decode, context + prefill_q)
        flops = 2.0 * heads * pairs * use_dim + 2.0 * heads * pairs * v_dim
        if plan.indexer or plan.indexer_type == "shared":
            index_heads = int(facts.get("index_n_heads", 32) or 32)
            index_dim = int(facts.get("index_head_dim", 128) or 128)
            seq = context + prefill_q + decode
            cost = index_selection_cost(model, plan, seq)
            flops += 2.0 * max(1, prefill_q + decode) * cost.scored_rows * index_heads * index_dim
        return float(flops)
    if plan.attention in {"kda", "gdn"}:
        positions = max(0, prefill_q + decode)
        if plan.attention == "kda":
            return float(positions * plan.linear_heads * plan.linear_head_dim ** 2 * 4.0)
        return float(positions * plan.linear_value_heads
                     * plan.linear_key_head_dim * plan.linear_head_dim * 4.0)
    return 0.0


def _mandatory_comm_bytes(
    model: Any, parallel: ParallelSpec, batch_tokens: int, layers: int,
) -> float:
    h = model.hidden_size
    comm = 0.0
    if parallel.tp > 1:
        comm += layers * 2.0 * (parallel.tp - 1) / parallel.tp \
            * batch_tokens * h * dtype_bytes("bf16") * 2.0
    if parallel.pp > 1:
        comm += (parallel.pp - 1) * batch_tokens * h * dtype_bytes("bf16")
    if parallel.ep > 1:
        for plan in model.layer_plans:
            if plan.ffn == "moe" and plan.top_k:
                width = plan.expert_latent_size or h
                comm += 2.0 * batch_tokens * plan.top_k * width \
                    * dtype_bytes("bf16") * (parallel.ep - 1) / parallel.ep
    return float(comm)


def _active_weight_bytes(
    model: Any, workload: Any, parallel: ParallelSpec, *, forward_count: int,
    media_present: bool,
) -> float:
    """Minimum active packed-weight traffic for one legal batched execution."""
    from .a3_catalog_weights import (
        body_head_terms, engram_gate_terms_by_layer,
        external_shared_weight_bytes, layer_weight_terms, term_bytes,
        vision_weight_terms,
    )
    quant = model.quant
    if quant is None:
        return float(model.resident_weight_bytes_override or 0.0)
    total = 0.0
    one_time = 0.0
    for plan in model.layer_plans:
        for term in layer_weight_terms(plan, model):
            if term.category in {"routed_expert", "dspark_routed_expert"}:
                if term.count <= 0:
                    continue
                per_expert = term_bytes(term, quant, ParallelSpec(), 1) / term.count
                selected = max(1, min(plan.top_k or 1, plan.num_experts or 1))
                total += per_expert * selected
            else:
                total += term_bytes(term, quant, ParallelSpec(), 1)
    for term in body_head_terms(model):
        if term.name == "embedding":
            continue  # active-row traffic below
        if not term.source_owned:
            continue
        total += term_bytes(term, quant, ParallelSpec(), 1)
    # A tied LM head aliases the embedding table but still performs one full
    # target-head read.  Untied heads are already in body_head_terms exactly
    # once; never add an unconditional V*H copy here.
    if (model.tie_word_embeddings and not model.is_standalone_draft):
        one_time += float(
            model.vocab_size * model.hidden_size * dtype_bytes("bf16")
        )
    if media_present and model.vision is not None:
        media_requests = sum(
            1 for request in workload.requests
            if int(getattr(request, "media_tokens", 0) or 0) > 0
        )
        for term in vision_weight_terms(model):
            one_time += (
                term_bytes(term, quant, ParallelSpec(), 1)
                * max(1, media_requests)
            )
    if model.engram is not None:
        for layer_terms in engram_gate_terms_by_layer(model).values():
            for term in layer_terms:
                total += term_bytes(term, quant, ParallelSpec(), 1)
        max_tokens = max(
            1,
            max(
                (int(getattr(r, "prompt_tokens", 1)) for r in workload.requests),
                default=1,
            ),
        )
        row_bytes = model.engram.row_bytes
        total += float(max_tokens * model.engram.lookups_per_token * row_bytes)
    if model.is_standalone_draft:
        total += external_shared_weight_bytes(model, ParallelSpec())
    # Exact newly-read embedding rows: uncached prompt rows at prefill, then
    # one newly generated input row per decode forward.  Do not repeat the
    # entire prompt row set on every decode forward.
    prompt_rows = sum(
        max(0, int(getattr(request, "prompt_tokens", 0))
            - int(getattr(request, "prefix_tokens", 0)))
        for request in workload.requests
    )
    decode_rows = sum(
        max(0, int(getattr(request, "output_tokens", 1)) - 1)
        for request in workload.requests
    )
    embedding_rows = float(
        (prompt_rows + decode_rows) * model.hidden_size * dtype_bytes("bf16")
    )
    return (
        float(total) * float(max(1, forward_count))
        + float(one_time)
        + embedding_rows
    )


def catalog_semantic_work(model: Any, workload: Any, parallel: ParallelSpec) -> Any:
    """Mandatory semantics for heterogeneous catalog models.

    Weight bytes are unique selected packed tensor reads with active top-K
    experts, one LM-head read for a tied alias (or the ordinary head term once
    per forward), and exact newly-read embedding rows. They are never logical
    parameter counts multiplied by every token. Prefill and sequential decode
    attention pairs are computed separately per request.
    """
    from .a3_should import SemanticWork

    h = model.hidden_size
    prompt_total = sum(request.prompt_tokens for request in workload.requests)
    output_total = sum(request.output_tokens for request in workload.requests)
    linear_total = 0.0
    attention_total = 0.0
    kv_bytes = 0.0
    linear_positions_total = 0
    decode_total = 0
    max_batch_tokens = 1
    standalone = model.is_standalone_draft
    draft_block = 0
    if standalone and model.auxiliary:
        spec = model.auxiliary.dspark or model.auxiliary.mtp
        draft_block = spec.proposal_tokens if spec else 1
    for request in workload.requests:
        prefix = max(0, request.prefix_tokens)
        q = max(0, request.prompt_tokens - prefix)
        decode = max(0, request.output_tokens - 1)
        if standalone:
            q = max(1, draft_block)
            decode = 0
            prefix = 0
        linear_positions = (q + decode) if q > 0 else (1 + decode)
        linear_positions_total += linear_positions
        decode_total += decode
        max_batch_tokens = max(max_batch_tokens, q + decode)
        seq = max(1, request.prompt_tokens + request.output_tokens)
        breakdown = request_cache_breakdown(model, parallel, seq)
        kv_bytes += breakdown["total_bytes"]
        for plan in model.layer_plans:
            linear_total += _linear_flops_only(plan, model, linear_positions)
            attention_total += _attention_flops_only(
                plan, model, prefill_q=q, decode=decode, context=prefix,
            )
    logits = 0.0 if standalone else 2.0 * output_total * h * model.vocab_size
    if standalone:
        logits = 2.0 * max(1, draft_block) * h * model.vocab_size
    forward_count = max(
        1,
        max((int(request.output_tokens) for request in workload.requests), default=1),
    )
    media_present = any(int(getattr(request, "media_tokens", 0) or 0) > 0
                        for request in workload.requests)
    resident = float(model.resident_weight_bytes_override or 0.0)
    weight_traffic = _active_weight_bytes(
        model, workload, parallel, forward_count=forward_count,
        media_present=media_present,
    )
    activation = float(2.0 * max_batch_tokens * h * dtype_bytes("bf16") * 2.0)
    comm = _mandatory_comm_bytes(model, parallel, max_batch_tokens, model.layers)
    total_bytes = weight_traffic + kv_bytes + activation
    return SemanticWork(
        prompt_tokens=prompt_total, output_tokens=output_total,
        linear_flops=linear_total, attention_flops=attention_total,
        logits_flops=logits, total_flops=linear_total + attention_total + logits,
        weight_bytes=weight_traffic, resident_weight_bytes=resident,
        kv_bytes=kv_bytes, activation_bytes=activation,
        total_bytes=total_bytes, comm_bytes=comm,
        linear_positions=linear_positions_total, decode_positions=decode_total,
        per_request=tuple(
            (request.request_id, float(request.prompt_tokens + request.output_tokens))
            for request in workload.requests
        ),
    )


__all__ = ["infer_catalog_iteration", "catalog_semantic_work"]
