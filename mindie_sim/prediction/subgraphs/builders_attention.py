"""Canonical builders for input, attention, cache and indexer families.

[Role] Construct E0/N0/A0-A4/K0-K1/I0-I1 templates and their static signatures.
[Invariants] Builders consume only ``_Ctx``; no model-name or measurement branch.
  Child macros reference kinds instead of duplicating work.
[Timeouts & Locks] Pure bounded construction; no I/O or locks.
[Side-Effect Policy] Returns immutable ``_Built`` records.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.subgraphs import FusionContract
from .instantiate import (_Ctx, _Built, _HIDDEN_IN, _HIDDEN_OUT, _KV_IN, _KV_OUT, _KV_READY_OUT,
    _RESIDUAL_IN, _RESIDUAL_OUT, _attention_sig, _built, _collective, _copy,
    _elementwise, _in, _matrix, _out, _parallel, _precede, _quant_fields, _sig, _state, _tiles, _variant, )


def _owner_stage(ctx: _Ctx) -> int:
    owner = int(ctx.layer.kv_owner_layer_id)
    if owner < 0:
        return ctx.stage
    if ctx.layers <= 0:
        return 0
    return min(ctx.features.pp - 1, owner * ctx.features.pp // ctx.layers)


def _stage_of_layer(ctx: _Ctx, layer: int) -> int:
    if ctx.features.pp <= 1 or ctx.layers <= 1:
        return 0
    return min(ctx.features.pp - 1, layer * ctx.features.pp // ctx.layers)


def _remote_consumer_stages(ctx: _Ctx) -> tuple[int, ...]:
    """Distinct remote stages that consume this owner's KV (same stage excluded)."""
    if ctx.features.pp <= 1:
        return ()
    return tuple(sorted({_stage_of_layer(ctx, plan.index) for plan in ctx.model.layer_plans
        if int(plan.kv_owner_layer_id) == ctx.layer.index and plan.index > ctx.layer.index
        and _stage_of_layer(ctx, plan.index) != ctx.stage}))


def _group_plan(ctx: _Ctx, groups: int) -> tuple[bool, int]:
    """Mirror the catalog plan: shard head groups by TP or gather first."""
    tp = max(1, ctx.tp)
    if (groups > 0 and groups <= ctx.heads and groups % tp == 0
            and (ctx.heads // groups) * (groups // tp) == ctx.local_heads):
        return True, groups // tp
    return False, groups


def _dcp_merge(ctx: _Ctx, specs: list, signatures: dict, *,
               local_heads: int, d_v: int) -> None:
    """Declare the CP partial-output exchange and its dependent LSE merge."""
    cp = int(ctx.features.cp)
    if cp <= 1:
        return
    act_bytes = dtype_bytes(ctx.features.activation_format)
    specs.append(("dcp_allgather", "collective", "partial_output_allgather"))
    signatures["dcp_allgather"] = _collective(
        "partial_output_allgather", ctx, axis="cp", members=cp,
        bytes_per_token=local_heads * (d_v * act_bytes + 4.0), stages=1)
    specs.append(("dcp_lse_merge", "elementwise", "attention_lse_update"))
    signatures["dcp_lse_merge"] = _elementwise(
        "attention_lse_update", ctx, elements=local_heads * d_v * cp, ops=3,
        read_bytes=cp * local_heads * (d_v * 4 + 4),
        write_bytes=local_heads * d_v * act_bytes)


def _build_e0(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "token")
    specs = [("embedding", "embedding", "embed"), ("input_copy", "copy", "input")]
    if ctx.model.vision is not None or variant == "media":
        specs.append(("media_splice", "copy", "media"))
    signatures = { "embedding": _matrix("embedding", ctx.hidden, ctx.model.vocab_size, ctx, "embedding"),
        "input_copy": _copy("input", ctx, elements=ctx.hidden, bytes_per_element=dtype_bytes(ctx.features.activation_format)),
        "media_splice": _copy("media", ctx, elements=ctx.hidden, bytes_per_element=dtype_bytes(ctx.features.activation_format)),
    }
    facts = _gemma_facts(ctx)
    if facts.get("gemma4_ple"):
        ple_width = int(facts.get("gemma4_ple_width", ctx.hidden) or ctx.hidden)
        layers = max(1, int(ctx.layers or len(ctx.model.layer_plans) or ctx.model.layers))
        width = layers * ple_width
        act_bytes = dtype_bytes(ctx.features.activation_format)
        specs.extend((("per_layer_model_projection", "gemm", "projection"),
                      ("per_layer_projection_norm", "elementwise", "norm"),
                      ("ple_combine", "elementwise", "combine")))
        signatures["per_layer_model_projection"] = _matrix(
            "per_layer_model_projection", ctx.hidden, width, ctx, "ple_projection",
            extra=(("ple_layers", layers), ("ple_width", ple_width)))
        signatures["per_layer_projection_norm"] = _elementwise( "ple_norm", ctx, elements=width, ops=5,
            read_bytes=width * act_bytes, write_bytes=width * act_bytes,
            extra=(("ple_layers", layers), ("ple_width", ple_width)))
        # Contextual projection and the token-table component are combined
        # before the decoder consumes the per-layer input.
        signatures["ple_combine"] = _elementwise(
            "ple_combine", ctx, elements=width, ops=2,
            read_bytes=2 * width * act_bytes, write_bytes=width * act_bytes,
            extra=(("ple_layers", layers), ("ple_width", ple_width), ("table_component", 1)))
    return _built("E0", ctx, variant, "Input Embeddings (输入嵌入)",
                  [(name, family, role) for name, family, role in specs], signatures, state_out=(_HIDDEN_OUT,))


def _gemma_facts(ctx: _Ctx) -> dict:
    return dict(ctx.model.catalog_facts)


def _quant_scale_bytes(ctx: _Ctx, default: str = "fp16") -> float:
    raw = getattr(ctx.quant, "index_scale_dtype", None) if ctx.quant else None
    value = str(raw or "").strip()
    if value.lower() in {"", "none"}:
        value = "fp8_e8m0" if raw is not None else default
    return dtype_bytes(value)


def _gemma_k_eq_v(ctx: _Ctx) -> bool:
    if ctx.layer.k_eq_v:
        return True
    facts = _gemma_facts(ctx)
    layers = facts.get("gemma4_k_eq_v_layers")
    if layers is None:
        return bool(facts.get("gemma4_k_eq_v_all"))
    return ctx.layer.index in {int(value) for value in layers}


def _build_n0(ctx: _Ctx) -> _Built:
    role = ctx.role
    variant = _variant(ctx, {"attn_residual": "residual", "ffn_norm": "rmsnorm",
                             "attn_norm": "rmsnorm"}.get(role, "rmsnorm"))
    specs = []
    signatures = {}
    act_bytes = dtype_bytes(ctx.features.activation_dtype or "bf16")
    if role in {"attn_residual", "attn_norm"}:
        specs.append(("residual_add", "elementwise", "residual"))
        elements = ctx.hidden
        signatures["residual_add"] = _elementwise(
            "residual", ctx, elements=elements, ops=2,
            read_bytes=2 * elements * act_bytes, write_bytes=elements * act_bytes,
        )
    if role != "attn_residual":
        specs.append(("rmsnorm", "elementwise", "norm"))
        elements = ctx.hidden
        facts = _gemma_facts(ctx)
        signatures["rmsnorm"] = _elementwise(
            "norm", ctx, elements=elements, ops=5, read_bytes=elements * act_bytes,
            write_bytes=elements * act_bytes,
            extra=(("norm_eps", ctx.model.norm_eps),
                   ("layernorm", int(variant == "layernorm")),
                   ("norm_weight_offset", float(facts.get("norm_weight_offset", 0.0))),
                   ("norm_arch", "gemma" if facts.get("gemma4") else "rmsnorm")),
        )
    # Q/K normalization belongs to the attention producer after the projected
    # Q/K exist; N0 never emits a pre-projection copy.
    fusions = ()
    if role == "attn_norm":
        members = tuple(name for name, _, _ in specs if name in {"residual_add", "rmsnorm", "activation_quant"})
        if {"residual_add", "rmsnorm"} <= set(members):
            fusions = (FusionContract( "add_rmsnorm_quant", members,
                removes=("residual_add_output", "rmsnorm_output"),
                keeps=("residual", "rstd", "scale", "norm_output")),)
    return _built("N0", ctx, variant, "Normalization and Residual Addition (归一化与残差相加)",
                  specs, signatures, fusions=fusions,
                  state_in=(_HIDDEN_IN, _RESIDUAL_IN), state_out=(_HIDDEN_OUT, _RESIDUAL_OUT))


def _build_a0(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "mqa" if ctx.local_kv_heads == 1 and ctx.group_heads > 1
                       else "mha" if ctx.group_heads == 1 else "gqa")
    shared = ctx.shared_consumer
    specs = []
    signatures = {}
    k_eq_v = _gemma_k_eq_v(ctx)
    act_bytes = dtype_bytes(ctx.features.activation_format)
    q_slice = ctx.local_heads * ctx.width
    q_width = q_slice * (2 if ctx.layer.q_output_gate else 1)
    if shared:
        # A SharedKV consumer still owns its current-hidden query projection
        # (and declared Q gate lanes); only K/V production is omitted.
        specs.append(("q_projection", "gemm", "projection"))
        signatures["q_projection"] = _matrix( "projection", ctx.hidden, q_width, ctx, "attn.q",
            extra=(("q_only", 1),))
        specs.append(("kv_read", "kv_access", "consume"))
        signatures["kv_read"] = _state( "kv_read", ctx, rows=1, row_bytes=0.0,
            extra=(("owner_layer", int(ctx.layer.kv_owner_layer_id)),))
        proj_name = "q_projection"
    else:
        # Gemma K=V stores one key projection and aliases it as value instead
        # of materialising a second V weight matrix / projection output.
        kv_width = (1 if k_eq_v else 2) * ctx.local_kv_heads * ctx.width
        specs.append(("qkv_projection", "gemm", "projection"))
        signatures["qkv_projection"] = _matrix( "projection", ctx.hidden, q_width + kv_width, ctx, "attn.qkv",
            extra=(("k_eq_v", int(k_eq_v)),))
        proj_name = "qkv_projection"
    if ctx.layer.qk_norm:
        heads = ctx.local_heads + (0 if shared else ctx.local_kv_heads)
        specs.append(("qk_norm", "elementwise", "norm"))
        signatures["qk_norm"] = _elementwise( "qk_norm", ctx, elements=heads * ctx.width, ops=5,
            read_bytes=heads * ctx.width * act_bytes, write_bytes=heads * ctx.width * act_bytes)
    specs.append(("rotary", "elementwise", "rope"))
    rotated = ctx.local_heads + (0 if shared else ctx.local_kv_heads)
    signatures["rotary"] = _elementwise( "rope", ctx, elements=rotated * ctx.width, ops=6,
        read_bytes=rotated * ctx.width * act_bytes, write_bytes=rotated * ctx.width * act_bytes)
    specs.append(("attention", "attention", "attention"))
    plans, idx_raw, sel_raw = ctx.model.layer_plans, int(ctx.layer.index_k_owner_layer_id), int(ctx.layer.topk_owner_layer_id)
    if ctx.layer.indexer:
        idx_owner = sel_owner = ctx.layer.index
    else:
        idx_owner = idx_raw if (0 <= idx_raw < ctx.layer.index and idx_raw < len(plans) and plans[idx_raw].indexer) else -1
        sel_owner = sel_raw if (0 <= sel_raw < ctx.layer.index and sel_raw < len(plans) and (plans[sel_raw].indexer or plans[sel_raw].owns_topk)) else idx_owner
    has_indexer = bool(ctx.layer.indexer or ctx.layer.owns_topk or idx_owner >= 0 or sel_owner >= 0)
    gqa_topk = int(ctx.layer.index_topk) if has_indexer else 0
    gqa_kpool = max(1, int(ctx.layer.index_kpool or 1))
    idx_reqs = ((("index_topk", gqa_topk), ("kpool", gqa_kpool),
                 ("require_index", idx_owner), ("require_selection", sel_owner))
                if gqa_topk > 0 and idx_owner >= 0 else ())
    signatures["attention"] = _attention_sig( "attention", ctx, local_heads=ctx.local_heads,
        local_kv_heads=ctx.local_kv_heads, group_heads=ctx.group_heads, width=ctx.width, value_width=ctx.width,
        kv_row_bytes=ctx.kv_row_bytes, extra=(("kv_owner", int(ctx.layer.kv_owner_layer_id)),
               ("shared_consumer", int(shared)), ("split_kv_target", ctx.features.split_target),
               ("max_kv_splits", ctx.features.max_splits), ("q_output_gate", int(ctx.layer.q_output_gate)),
               ("k_eq_v", int(k_eq_v)), ("local_window", int(ctx.layer.window)))
        + idx_reqs
        + ((("require_kv", int(ctx.layer.kv_owner_layer_id)),) if shared else ()))
    _dcp_merge(ctx, specs, signatures, local_heads=ctx.local_heads, d_v=ctx.width)
    if ctx.layer.q_output_gate:
        specs.append(("output_gate", "elementwise", "gate"))
        signatures["output_gate"] = _elementwise( "gate", ctx, elements=ctx.local_heads * ctx.width, ops=5,
            read_bytes=ctx.local_heads * ctx.width * act_bytes, write_bytes=ctx.local_heads * ctx.width * act_bytes)
    specs.append(("o_projection", "gemm", "output"))
    signatures["o_projection"] = _matrix( "output", ctx.local_heads * ctx.width, ctx.hidden, ctx, "attn.o")
    _precede(specs, signatures, ctx, "o_projection", ctx.local_heads * ctx.width, "attn.o")
    if ctx.tp > 1:
        specs.append(("tp_allreduce", "collective", "tp"))
        signatures["tp_allreduce"] = _collective( "tp", ctx, axis="tp", members=ctx.tp,
            bytes_per_token=ctx.hidden * act_bytes)
    fusions = ()
    contract = ()
    if ctx.layer.qk_norm:
        gate_keep = ("gate",) if ctx.layer.q_output_gate else ()
        fusions = (FusionContract( "qkv_norm_rope", (proj_name, "qk_norm", "rotary"),
            removes=(("q_projection_output", "qk_norm_output") if shared
                     else ("qkv_activation", "qk_norm_output")),
            keeps=(("query", *gate_keep, "residual") if shared else
                  ("query", *gate_keep, "key", "value", "residual", "kv"))),)
        k_elems = 0 if shared else ctx.local_kv_heads * ctx.width
        v_elems = 0 if shared or k_eq_v else k_elems
        gate_elems = q_slice if ctx.layer.q_output_gate else 0
        total = q_slice + gate_elems + k_elems + v_elems
        contract = (("qkv_slices", f"{q_slice},{gate_elems},{k_elems},{v_elems},"
                                 f"{total},{act_bytes:g}"),)
    a0_state = ((_in("index_selection", "selection"),) if gqa_topk > 0 else ())
    return _built("A0", ctx, variant, "Multi-Head / Grouped-Query / Multi-Query Attention (多头／分组查询／多查询注意力)",
                  specs, signatures,
                  fusions=fusions, state_in=(_HIDDEN_IN, _KV_IN) + a0_state,
                  state_out=(_HIDDEN_OUT, _KV_OUT, _KV_READY_OUT), contract=contract)


def _build_a1(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "latent_mqa" if ctx.layer.mla_variant == "latent_mqa" else "latent_mla")
    mla = ctx.model.mla
    r_q = int(ctx.layer.q_lora_rank or (mla.q_lora_rank if mla else 0))
    r_kv = int(ctx.layer.kv_lora_rank or (mla.kv_lora_rank if mla else 0))
    d_n = int(ctx.layer.qk_nope_head_dim or (mla.qk_nope_head_dim if mla else ctx.width))
    d_r = int(ctx.layer.qk_rope_head_dim or (mla.qk_rope_head_dim if mla else 0))
    d_v = int(ctx.layer.v_head_dim or (mla.v_head_dim if mla else ctx.width))
    act_bytes = dtype_bytes(ctx.features.activation_dtype or "bf16")
    specs = []
    signatures = {}
    if r_q > 0:
        specs.append(("q_down", "gemm", "projection"))
        signatures["q_down"] = _matrix("q_lora_down", ctx.hidden, r_q, ctx, "mla")
        specs.append(("q_norm", "elementwise", "norm"))
        signatures["q_norm"] = _elementwise( "norm", ctx, elements=r_q, ops=5,
            read_bytes=r_q * act_bytes, write_bytes=r_q * act_bytes)
        specs.append(("q_up", "gemm", "projection"))
        signatures["q_up"] = _matrix("q_lora_up", r_q, ctx.local_heads * (d_n + d_r), ctx, "mla")
    else:
        variant = "direct_q"
        specs.append(("q_direct", "gemm", "projection"))
        signatures["q_direct"] = _matrix("q_direct", ctx.hidden, ctx.local_heads * (d_n + d_r), ctx, "mla")
    specs.append(("kv_down", "gemm", "projection"))
    signatures["kv_down"] = _matrix("kv_lora_down", ctx.hidden, r_kv + d_r, ctx, "mla")
    specs.append(("kv_norm", "elementwise", "norm"))
    signatures["kv_norm"] = _elementwise( "norm", ctx, elements=r_kv, ops=5,
        read_bytes=r_kv * act_bytes, write_bytes=r_kv * act_bytes)
    specs.append(("rotary", "elementwise", "rope"))
    rotated = ctx.local_heads + 1
    signatures["rotary"] = _elementwise( "rope", ctx, elements=rotated * d_r, ops=6,
        read_bytes=rotated * d_r * act_bytes, write_bytes=rotated * d_r * act_bytes)
    specs.append(("kv_store", "kv_access", "produce"))
    signatures["kv_store"] = _state( "kv_store", ctx, rows=1,
        row_bytes=float((r_kv + d_r) * dtype_bytes(ctx.kv_dtype)), extra=(("owner", 1), ("layout", "latent_mla")))
    specs.append(("mla_attention", "mla", "attention"))
    plans, idx_raw, sel_raw = ctx.model.layer_plans, int(ctx.layer.index_k_owner_layer_id), int(ctx.layer.topk_owner_layer_id)
    if ctx.layer.indexer:
        idx_owner = sel_owner = ctx.layer.index
    else:
        idx_owner = idx_raw if (0 <= idx_raw < ctx.layer.index and idx_raw < len(plans) and plans[idx_raw].indexer) else -1
        sel_owner = sel_raw if (0 <= sel_raw < ctx.layer.index and sel_raw < len(plans) and (plans[sel_raw].indexer or plans[sel_raw].owns_topk)) else idx_owner
    has_indexer = bool(ctx.layer.indexer or ctx.layer.owns_topk or idx_owner >= 0 or sel_owner >= 0)
    mla_topk = int(ctx.layer.index_topk or (mla.index_topk if mla else 0)) if has_indexer else 0
    mla_kpool = max(1, int(ctx.layer.index_kpool or 1))
    idx_reqs = ((("require_index", idx_owner), ("require_selection", sel_owner)) if mla_topk > 0 and idx_owner >= 0 else ())
    signatures["mla_attention"] = _sig( "attention", heads=(("q", ctx.local_heads), ("kv", 1), ("dn", d_n),
                            ("dr", d_r), ("dv", d_v), ("rkv", r_kv)),
        quant=(("kv_bytes", float((r_kv + d_r) * dtype_bytes(ctx.kv_dtype))), ("act_bytes", act_bytes)),
        parallel=_parallel(ctx), tiles=_tiles(ctx), extras=(("causal", int(not ctx.layer.noncausal_attention)),
                ("window", int(ctx.layer.window)), ("absorb", 1), ("direct_q", int(r_q == 0)),
                ("index_topk", mla_topk), ("kpool", mla_kpool),
                ("split_kv_target", ctx.features.split_target), ("max_kv_splits", ctx.features.max_splits)) + idx_reqs)
    _dcp_merge(ctx, specs, signatures, local_heads=ctx.local_heads, d_v=d_v)
    groups = int(ctx.layer.o_groups or 0)
    o_rank = max(1, int(ctx.layer.o_lora_rank or 0))
    group_parallel, local_groups = _group_plan(ctx, groups)
    if groups > 0:
        group_width = max(1, (ctx.heads * d_v) // groups)
        if not group_parallel and ctx.tp > 1:
            specs.append(("o_proj_allgather", "collective", "o_proj_input_allgather"))
            signatures["o_proj_allgather"] = _collective(
                "o_proj_input_allgather", ctx, axis="tp", members=ctx.tp,
                bytes_per_token=(ctx.tp - 1) * ctx.local_heads * d_v * act_bytes)
            specs.append(("o_proj_transpose", "copy", "o_proj_input_layout"))
            signatures["o_proj_transpose"] = _copy(
                "o_proj_input_layout", ctx, elements=ctx.heads * d_v,
                bytes_per_element=act_bytes)
        specs.append(("grouped_output_a", "gemm", "output"))
        signatures["grouped_output_a"] = _matrix( "grouped_output_a", group_width, o_rank, ctx, "attn.o_a",
            extra=(("o_groups", groups), ("batch", local_groups)))
        specs.append(("grouped_output_b", "gemm", "output"))
        signatures["grouped_output_b"] = _matrix( "grouped_output_b", local_groups * o_rank, ctx.hidden, ctx, "attn.o_b",
            extra=(("o_groups", groups), ("local_groups", local_groups)))
    else:
        specs.append(("o_projection", "gemm", "output"))
        signatures["o_projection"] = _matrix( "output", ctx.local_heads * d_v, ctx.hidden, ctx, "attn.o")
        _precede(specs, signatures, ctx, "o_projection", ctx.local_heads * d_v, "attn.o")
    if ctx.tp > 1 and (groups == 0 or group_parallel):
        specs.append(("tp_allreduce", "collective", "tp"))
        signatures["tp_allreduce"] = _collective( "tp", ctx, axis="tp", members=ctx.tp,
            bytes_per_token=ctx.hidden * act_bytes)
    prolog_members = tuple( name for name, _, _ in specs
        if name in {"q_down", "q_norm", "q_up", "q_direct", "kv_down", "kv_norm", "rotary", "kv_store"})
    removes = tuple( name for name in ("q_lora_activation", "q_norm_output", "kv_down_output",
                          "kv_norm_output", "rotary_output")
        if r_q > 0 or name not in {"q_lora_activation", "q_norm_output"})
    fusions = ()
    if {"kv_down", "kv_norm", "rotary"} <= set(prolog_members):
        fusions = (FusionContract( "mla_prolog", prolog_members, removes=removes,
            keeps=("query", "key", "value", "kv_cache", "mla_attention_output")),)
    a1_in = (_HIDDEN_IN, _KV_IN) + ((_in("index_selection", "selection"),) if mla_topk > 0 else ())
    return _built("A1", ctx, variant, "Multi-head Latent Attention (MLA 多头潜在注意力)",
                  specs, signatures,
                  fusions=fusions, state_in=a1_in, state_out=(_HIDDEN_OUT, _KV_OUT, _KV_READY_OUT))


def _build_a2(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "window_compressed")
    act_bytes = dtype_bytes(ctx.features.activation_format)
    owner = int(ctx.layer.kv_owner_layer_id)
    owner = owner if 0 <= owner < ctx.layer.index else ctx.layer.index
    index_raw = int(ctx.layer.index_k_owner_layer_id)
    index_owner = ctx.layer.index if ctx.layer.indexer else (index_raw if 0 <= index_raw < ctx.layer.index else ctx.layer.index)
    q_width = ctx.local_heads * ctx.width * (2 if ctx.layer.q_output_gate else 1)
    r_q = int(ctx.layer.q_lora_rank or 0)
    k_eq_v = _gemma_k_eq_v(ctx)
    if ctx.role == "attention_prolog":
        specs, signatures = [], {}
        if r_q > 0:
            specs.extend((("q_down", "gemm", "projection"), ("q_norm", "elementwise", "norm"),
                          ("q_up", "gemm", "projection")))
            signatures["q_down"] = _matrix("q_lora_down", ctx.hidden, r_q, ctx, "attn.q")
            signatures["q_norm"] = _elementwise("norm", ctx, elements=r_q, ops=5, read_bytes=r_q * act_bytes,
                                                write_bytes=r_q * act_bytes)
            signatures["q_up"] = _matrix("q_lora_up", r_q, q_width, ctx, "attn.q")
        else:
            specs.append(("q_direct", "gemm", "projection"))
            signatures["q_direct"] = _matrix("q_direct", ctx.hidden, q_width, ctx, "attn.q")
        kv_width = (1 if k_eq_v else 2) * ctx.local_kv_heads * ctx.width
        specs.extend((("kv_down", "gemm", "projection"), ("kv_norm", "elementwise", "norm"),
                      ("rotary", "elementwise", "rope"), ("kv_store", "kv_access", "produce")))
        signatures["kv_down"] = _matrix("kv_lora_down", ctx.hidden, kv_width, ctx, "attn.kv")
        signatures["kv_norm"] = _elementwise("norm", ctx, elements=kv_width, ops=5, read_bytes=kv_width * act_bytes,
                                             write_bytes=kv_width * act_bytes)
        rotated = ctx.local_heads * ctx.width + ctx.local_kv_heads * ctx.width
        signatures["rotary"] = _elementwise("rope", ctx, elements=rotated, ops=6, read_bytes=rotated * act_bytes,
                                            write_bytes=rotated * act_bytes)
        signatures["kv_store"] = _state( "kv_store", ctx, rows=1, row_bytes=ctx.kv_row_bytes,
            extra=(("owner", 1), ("layout", str(ctx.layer.cache_kind))))
        return _built("A2", ctx, variant,
                      "Sliding-Window Attention with Compressed KV Attention (滑动窗口注意力与压缩 KV 注意力)",
                      specs, signatures, coverage="structural",
                      state_in=(_HIDDEN_IN,), state_out=(_KV_OUT, _KV_READY_OUT))
    specs = [("kv_read", "kv_access", "consume"), ("window_attention", "attention", "window"),
             ("compressed_attention", "attention", "compressed"), ("merge", "elementwise", "merge")]
    signatures = { "kv_read": _state("kv_read", ctx, rows=1, row_bytes=0.0, extra=(("owner_layer", owner),)),
        "window_attention": _attention_sig( "window", ctx, local_heads=ctx.local_heads,
            local_kv_heads=ctx.local_kv_heads, group_heads=ctx.group_heads,
            width=ctx.width, value_width=ctx.width, kv_row_bytes=ctx.kv_row_bytes,
            extra=(("branch", "window"), ("require_kv", owner))), "compressed_attention": _attention_sig(
            "compressed", ctx, local_heads=ctx.local_heads,
            local_kv_heads=ctx.local_kv_heads, group_heads=ctx.group_heads,
            width=ctx.width, value_width=ctx.width, kv_row_bytes=ctx.kv_row_bytes, extra=(("branch", "compressed"),
                   ("compress_ratio", int(ctx.layer.compress_ratio)),
                   ("index_topk", int(ctx.layer.index_topk or (ctx.model.mla.index_topk if ctx.model.mla else 0))),
                   ("require_kv", owner), ("compression_owner", owner),
                   ("require_compression", int(ctx.layer.compress_ratio))) +
                  ((("require_index", index_owner), ("require_selection", index_owner))
                   if ctx.layer.indexer else ())),
        "merge": _elementwise( "merge", ctx, elements=ctx.local_heads * ctx.width, ops=5,
            read_bytes=2 * ctx.local_heads * ctx.width * act_bytes,
            write_bytes=ctx.local_heads * ctx.width * act_bytes), }
    _dcp_merge(ctx, specs, signatures, local_heads=ctx.local_heads, d_v=ctx.width)
    groups = int(ctx.layer.o_groups or 0)
    group_parallel, local_groups = _group_plan(ctx, groups)
    if groups > 0:
        o_rank = max(1, int(ctx.layer.o_lora_rank or 0))
        group_width = max(1, (ctx.heads * ctx.width) // groups)
        if not group_parallel and ctx.tp > 1:
            specs.append(("o_proj_allgather", "collective", "o_proj_input_allgather"))
            signatures["o_proj_allgather"] = _collective(
                "o_proj_input_allgather", ctx, axis="tp", members=ctx.tp,
                bytes_per_token=(ctx.tp - 1) * ctx.local_heads * ctx.width * act_bytes)
            specs.append(("o_proj_transpose", "copy", "o_proj_input_layout"))
            signatures["o_proj_transpose"] = _copy(
                "o_proj_input_layout", ctx, elements=ctx.heads * ctx.width,
                bytes_per_element=act_bytes)
        specs.extend((("grouped_output_a", "gemm", "output"), ("grouped_output_b", "gemm", "output")))
        signatures["grouped_output_a"] = _matrix( "grouped_output_a", group_width, o_rank, ctx, "attn.o_a",
            extra=(("o_groups", groups), ("batch", local_groups)))
        signatures["grouped_output_b"] = _matrix( "grouped_output_b", local_groups * o_rank, ctx.hidden, ctx, "attn.o_b",
            extra=(("o_groups", groups), ("local_groups", local_groups)))
    else:
        specs.append(("grouped_o", "gemm", "output"))
        signatures["grouped_o"] = _matrix("grouped_output", ctx.local_heads * ctx.width, ctx.hidden, ctx, "attn.o",
                                          extra=(("o_groups", 1),))
    if ctx.layer.q_output_gate or ctx.layer.mla_output_gate:
        specs.append(("output_gate", "elementwise", "gate"))
        signatures["output_gate"] = _elementwise( "gate", ctx, elements=ctx.local_heads * ctx.width, ops=5,
            read_bytes=ctx.local_heads * ctx.width * act_bytes, write_bytes=ctx.local_heads * ctx.width * act_bytes)
    if ctx.tp > 1 and (groups == 0 or group_parallel):
        specs.append(("tp_allreduce", "collective", "tp"))
        signatures["tp_allreduce"] = _collective( "tp", ctx, axis="tp", members=ctx.tp,
            bytes_per_token=ctx.hidden * act_bytes)
    return _built("A2", ctx, variant,
                  "Sliding-Window Attention with Compressed KV Attention (滑动窗口注意力与压缩 KV 注意力)",
                  specs, signatures, children=("K1",) + (("I0", "I1") if ctx.layer.indexer else ()),
                  coverage="partial", state_in=(_HIDDEN_IN, _in("compressed_kv", "kv_cache"),
                            _in("index_selection", "selection")), state_out=(_HIDDEN_OUT,),
                  streams={"compressed_attention": "aux"} if ctx.features.multistream_window_compressed else None)

def _build_a3(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "recurrent_decode")
    d_k = int(ctx.layer.linear_key_head_dim or ctx.width)
    d_v = int(ctx.layer.linear_head_dim or ctx.width)
    tp = max(1, ctx.tp)
    key_global = int(ctx.layer.linear_key_heads or ctx.layer.linear_heads or ctx.heads)
    value_global = int(ctx.layer.linear_value_heads or key_global)
    key_heads = max(1, (key_global + tp - 1) // tp)
    value_heads = max(1, (value_global + tp - 1) // tp)
    conv = int(ctx.layer.conv_kernel or 4)
    act_bytes = dtype_bytes(ctx.features.activation_format)
    qkv_width = 2 * key_heads * d_k + value_heads * d_v
    specs = [("qkvz_projection", "gemm", "projection"), ("ba_projection", "gemm", "projection"),
             ("conv", "recurrent", "conv"), ("prepare", "elementwise", "prepare"),
             ("gated_delta", "recurrent", "delta"), ("gated_norm", "elementwise", "norm"),
             ("o_projection", "gemm", "output")]
    act = ("act_bytes", act_bytes)
    signatures = { "qkvz_projection": _matrix("qkvz", ctx.hidden, key_heads * d_k * 2 + value_heads * d_v * 2,
                                   ctx, "gdn", extra=((("key_heads", key_heads), ("value_heads", value_heads), act))),
        "ba_projection": _matrix("ba", ctx.hidden, value_heads * 2 + key_heads, ctx, "gdn", extra=(act,)),
        "conv": _state("conv", ctx, rows=conv, row_bytes=qkv_width * act_bytes,
                       extra=(("kernel", conv), ("history_rows", conv - 1), ("qkv_width", qkv_width), act)),
        "prepare": _elementwise("prepare", ctx, elements=qkv_width, ops=4, read_bytes=qkv_width * act_bytes,
                                write_bytes=qkv_width * act_bytes, extra=(act,)),
        "gated_delta": _state("delta", ctx, rows=1, row_bytes=float(ctx.layer.state_bytes_total) / tp,
                              extra=(("chunk", min(64, max(1, int(ctx.layer.linear_head_dim or 64)))), ("key_heads", key_heads),
                                     ("value_heads", value_heads), ("dk", d_k), ("dv", d_v), act)),
        "gated_norm": _elementwise("norm", ctx, elements=value_heads * d_v, ops=5,
                                   read_bytes=2 * value_heads * d_v * act_bytes,
                                   write_bytes=value_heads * d_v * act_bytes, extra=(act,)),
        "o_projection": _matrix("output", value_heads * d_v, ctx.hidden, ctx, "gdn", extra=(act,)), }
    _precede(specs, signatures, ctx, "o_projection", value_heads * d_v, "gdn")
    if ctx.tp > 1:
        specs.append(("tp_allreduce", "collective", "tp"))
        signatures["tp_allreduce"] = _collective(
            "tp", ctx, axis="tp", members=ctx.tp, bytes_per_token=ctx.hidden * act_bytes, extra=(act,))
    return _built("A3", ctx, variant, "Gated DeltaNet (GDN 门控 DeltaNet)", specs, signatures, coverage="partial",
                  state_in=(_HIDDEN_IN, _in("recurrent_state", "recurrent"), _in("conv_state", "conv")),
                  state_out=(_HIDDEN_OUT, _out("recurrent_state", "recurrent"), _out("conv_state", "conv")),
                  streams={"ba_projection": "aux"} if ctx.features.multistream_kda_proj else None)


def _build_a4(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "recurrent_decode")
    d_k = int(ctx.layer.linear_key_head_dim or ctx.width)
    d_v = int(ctx.layer.linear_head_dim or ctx.width)
    tp = max(1, ctx.tp)
    key_global = int(ctx.layer.linear_key_heads or ctx.layer.linear_heads or ctx.heads)
    value_global = int(ctx.layer.linear_value_heads or key_global)
    key_heads = max(1, (key_global + tp - 1) // tp)
    value_heads = max(1, (value_global + tp - 1) // tp)
    conv = int(ctx.layer.conv_kernel or 4)
    act_bytes = dtype_bytes(ctx.features.activation_format)
    qkv_width = 2 * key_heads * d_k + value_heads * d_v
    specs = [("qkv_projection", "gemm", "projection"), ("gate_projection", "gemm", "projection"),
             ("conv", "recurrent", "conv"), ("prepare", "elementwise", "prepare"),
             ("kda_recurrent", "recurrent", "delta"), ("output_norm", "elementwise", "norm"),
             ("o_projection", "gemm", "output")]
    act = ("act_bytes", act_bytes)
    signatures = { "qkv_projection": _matrix("qkv", ctx.hidden, qkv_width, ctx, "kda",
                                  extra=(("key_heads", key_heads), ("value_heads", value_heads), act)),
        "gate_projection": _matrix("gate", ctx.hidden, value_heads * int(ctx.layer.gate_rank or 1),
                                   ctx, "kda", extra=(act,)),
        "conv": _state("conv", ctx, rows=conv, row_bytes=qkv_width * act_bytes,
                       extra=(("kernel", conv), ("qkv_width", qkv_width), act)),
        "prepare": _elementwise("prepare", ctx, elements=qkv_width, ops=4, read_bytes=qkv_width * act_bytes,
                                write_bytes=qkv_width * act_bytes, extra=(act,)),
        "kda_recurrent": _state("delta", ctx, rows=1, row_bytes=float(ctx.layer.state_bytes_total) / tp,
                                extra=(("key_heads", key_heads), ("value_heads", value_heads),
                                       ("dk", d_k), ("dv", d_v),
                                       ("gate_rank", int(ctx.layer.gate_rank or 1)),
                                       ("accepted_state", 1), act)),
        "output_norm": _elementwise("norm", ctx, elements=value_heads * d_v, ops=5,
                                    read_bytes=2 * value_heads * d_v * act_bytes,
                                    write_bytes=value_heads * d_v * act_bytes, extra=(act,)),
        "o_projection": _matrix("output", value_heads * d_v, ctx.hidden, ctx, "kda", extra=(act,)), }
    _precede(specs, signatures, ctx, "o_projection", value_heads * d_v, "kda")
    if ctx.tp > 1:
        specs.append(("tp_allreduce", "collective", "tp"))
        signatures["tp_allreduce"] = _collective(
            "tp", ctx, axis="tp", members=ctx.tp, bytes_per_token=ctx.hidden * act_bytes, extra=(act,))
    return _built("A4", ctx, variant, "Kimi Delta Attention (KDA Kimi Delta 注意力)", specs, signatures, coverage="partial",
                  state_in=(_HIDDEN_IN, _in("recurrent_state", "recurrent"),
                            _in("conv_state", "conv"), _in("accepted_state", "accept")),
                  state_out=(_HIDDEN_OUT, _out("recurrent_state", "recurrent"), _out("conv_state", "conv")),
                  streams={"gate_projection": "aux"} if ctx.features.multistream_kda_proj else None)


def _build_k0(ctx: _Ctx) -> _Built:
    shared = ctx.shared_consumer
    layout_only = not shared and ctx.attention_kind == "A2"
    variant = _variant(ctx, "read" if shared else "layout" if layout_only else "store")
    specs = []
    signatures = {}
    if shared:
        # The shared consumer's A0 attention already reads the owner's KV at
        # its consuming boundary.  K0 is only a zero-byte readiness port.
        specs.append(("kv_read", "kv_access", "consume"))
        signatures["kv_read"] = _state("kv_read", ctx, rows=1, row_bytes=0.0,
                                       extra=(("owner_layer", int(ctx.layer.kv_owner_layer_id)),
                                              ("require_kv", int(ctx.layer.kv_owner_layer_id)),
                                              ("stage_transfer", int(ctx.stage != _owner_stage(ctx)))))
    elif not layout_only:
        specs.append(("kv_store", "kv_access", "produce"))
        signatures["kv_store"] = _state("kv_store", ctx, rows=1, row_bytes=ctx.kv_row_bytes,
                                        extra=(("owner", 1), ("page", 128), ("layout", str(ctx.layer.cache_kind))))
    specs.append(("layout_convert", "copy", "layout"))
    local_elements = int(round(ctx.kv_row_bytes / dtype_bytes(ctx.kv_dtype)))
    signatures["layout_convert"] = _copy("layout", ctx, elements=local_elements,
                                         bytes_per_element=dtype_bytes(ctx.kv_dtype))
    destinations = _remote_consumer_stages(ctx) if not shared else ()
    if destinations:
        # One declared multicast edge publishes the owner's KV row to every
        # distinct remote consumer stage; same-stage consumers are excluded
        # because they read the local materialisation.
        specs.append(("stage_transfer", "transport", "transfer"))
        signatures["stage_transfer"] = _sig( "transfer", matrix=(("row_bytes", ctx.kv_row_bytes),),
            parallel=_parallel(ctx), tiles=_tiles(ctx), quant=(("wire_bytes", dtype_bytes(ctx.kv_dtype)),),
            extras=(("owner_layer", ctx.layer.index),
                    ("source_stage", ctx.stage), ("destination_stage", destinations[0]),
                    ("destination_stages", destinations),
                    ("transfer", "p2p"), ("collective", "alltoallv"), ("members", 1 + len(destinations)),
                    ("top_k", len(destinations)), ("latent", 1), ("chunks", 1), ("topology", "hccs"),
                    ("ranks_per_node", 1 + len(destinations)), ("mtu_bytes", 1),
                    ("state_effects", (("cross_stage", "transfer"),))))
    return _built("K0", ctx, variant, "KV Cache Read, Write and Transfer (KV 缓存读写与传输)",
                  specs, signatures, state_in=(_KV_IN,), state_out=(_KV_OUT, _KV_READY_OUT))


def _build_k1(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, str(ctx.layer.compressor_kind or "weighted_pool"))
    ratio = max(1, int(ctx.layer.compress_ratio or 1))
    width = int(ctx.layer.compressor_width or 1)
    specs = [("compress", "compression", "compress"), ("kpool", "compression", "pool"),
             ("tail_store", "compression", "tail"), ("state_store", "copy", "state")]
    signatures = { "compress": _state("compress", ctx, rows=1, row_bytes=ctx.kv_row_bytes,
                           extra=(("ratio", ratio), ("width", width),
                                  ("state_rows", int(ctx.layer.compressor_state_rows)))),
        "kpool": _state("kpool", ctx, rows=1, row_bytes=ctx.kv_row_bytes,
                        extra=(("ratio", max(1, int(ctx.layer.index_kpool or 1))),
                               ("always_tail", int(ctx.layer.index_kpool_always_select_tail)))),
        "tail_store": _state("tail", ctx, rows=1, row_bytes=ctx.kv_row_bytes, extra=(("ratio", ratio),)),
        "state_store": _copy("state", ctx, elements=ctx.kv_row_elements, bytes_per_element=dtype_bytes(ctx.kv_dtype)),
    }
    return _built("K1", ctx, variant, "KV Cache Compression and Key Pooling (KV 缓存压缩与键池化)",
                  specs, signatures, coverage="partial",
                  state_in=(_KV_IN, _in("compress_state", "compress")), state_out=(_out("compressed_kv", "kv_cache"),
                             _out("compress_state", "compress")))


def _build_i0(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, str(ctx.layer.indexer_type or "full"))
    heads = int(ctx.layer.index_heads or 1)
    dim = int(ctx.layer.index_dim or 64)
    kpool = max(1, int(ctx.layer.index_kpool or 1))
    specs = [("index_q_projection", "gemm", "projection"), ("index_k_projection", "gemm", "projection"),
             ("index_rope", "elementwise", "rope"), ("index_quant", "elementwise", "quant"),
             ("index_cache_store", "kv_access", "produce")]
    signatures = { "index_q_projection": _matrix("index_q", ctx.hidden, heads * dim, ctx, "indexer"),
        "index_k_projection": _matrix("index_k", ctx.hidden, dim, ctx, "indexer"),
        "index_rope": _elementwise("rope", ctx, elements=heads * dim, ops=6,
                                   read_bytes=heads * dim * dtype_bytes(ctx.features.activation_format),
                                   write_bytes=heads * dim * dtype_bytes(ctx.features.activation_format)),
        "index_quant": _elementwise("quant", ctx, elements=heads * dim, ops=2, read_bytes=heads * dim * 2,
                                    write_bytes=heads * dim), "index_cache_store": _state("index_store", ctx, rows=1,
                                    row_bytes=float(dim * dtype_bytes(ctx.features.index_dtype)),
                                    extra=(("heads", heads), ("dim", dim), ("kpool", kpool),
                                           ("owner_layer", ctx.layer.index),
                                           ("state_effects", (("index", "index_store"),)))), }
    return _built("I0", ctx, variant,
                  "Attention Indexer Projections and Cache Update (注意力索引器投影与缓存更新)",
                  specs, signatures, coverage="partial",
                  state_in=(_HIDDEN_IN, _in("index_cache", "index")),
                  state_out=(_out("index_cache", "index"), _out("index_ready", "readiness")))


def _build_i1(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "candidate" if ctx.layer.candidate_filter else "topk")
    heads = int(ctx.layer.index_heads or 1)
    dim = int(ctx.layer.index_dim or 64)
    topk = int(ctx.layer.index_topk or 0)
    kpool = max(1, int(ctx.layer.index_kpool or 1))
    blocks = int(ctx.layer.candidate_topk_blocks or 0)
    index_raw = int(ctx.layer.index_k_owner_layer_id)
    idx_owner = ctx.layer.index if ctx.layer.indexer else (index_raw if 0 <= index_raw < ctx.layer.index else ctx.layer.index)
    specs = [("index_score", "indexer", "score"), ("topk", "indexer", "select")]
    signatures = { "index_score": _sig("index_score", heads=(("heads", heads), ("dim", dim), ("topk", topk),
                                   ("kpool", kpool)), quant=(("index_bytes", dtype_bytes(ctx.features.index_dtype)),
                                   ("scale_bytes", _quant_scale_bytes(ctx))),
                            parallel=_parallel(ctx), tiles=_tiles(ctx), extras=(("candidate_blocks", blocks),
                                    ("require_index", idx_owner))),
        "topk": _sig("index_select", heads=(("topk", topk), ("heads", heads), ("kpool", kpool)),
                     quant=(("index_bytes", dtype_bytes(ctx.features.index_dtype)),),
                     parallel=_parallel(ctx), tiles=_tiles(ctx), extras=(("candidate_blocks", blocks),
                             ("block_size", int(ctx.layer.candidate_block_size or 1)), ("select_ops_per_row", 4),
                             ("owner_layer", idx_owner),
                             ("state_effects", (("selection", "index_select"),)))), }
    if ctx.layer.candidate_filter:
        specs.append(("candidate_filter", "indexer", "filter"))
        signatures["candidate_filter"] = _sig( "index_filter", heads=(("topk", topk), ("blocks", blocks),
                                   ("heads", heads)), quant=(("index_bytes", dtype_bytes(ctx.features.index_dtype)),),
            parallel=_parallel(ctx), tiles=_tiles(ctx),
            extras=(("block_size", int(ctx.layer.candidate_block_size or 1)), ("select_ops_per_row", 4),
                    ("owner_layer", idx_owner),
                    ("state_effects", (("selection", "index_select"),))))
    if ctx.features.cp > 1:
        specs.append(("cp_gather", "collective", "cp"))
        signatures["cp_gather"] = _collective( "cp_gather", ctx, axis="cp", members=ctx.features.cp,
            bytes_per_token=float(topk * dtype_bytes(ctx.features.index_dtype)), stages=1)
    return _built("I1", ctx, variant,
                  "Attention Index Scoring and Top-K Selection (注意力索引评分与 Top-K 选择)",
                  specs, signatures, coverage="partial",
                  state_in=(_in("index_cache", "index"), _in("index_selection", "selection")),
                  state_out=(_out("index_selection", "selection"),))


def _gemma_config(config: Mapping[str, Any]) -> dict:
    raw = dict(config)
    text = raw.get("text_config")
    if isinstance(text, Mapping):
        raw.update(text)
    return raw
def _gemma_global(index: int, layers: int, pattern: Any) -> bool:
    if isinstance(pattern, Sequence) and not isinstance(pattern, (str, bytes)):
        if len(pattern) != layers:
            raise ModelingError("gemma4_layer_types_length_mismatch")
        return str(pattern[index]).lower() in {"full", "full_attention", "global", "global_attention"}
    stride = int(pattern or 0)
    return index == layers - 1 if stride <= 0 else (index + 1) % stride == 0
def gemma4_model_spec(config: Mapping[str, Any], *, name: str = "gemma4") -> ModelSpec:
    """Map Gemma4 axes to the existing LayerPlan vocabulary.

    Local/global attention resolves per-layer head width, KV-head count and
    K=V use; KV sharing selects the nearest unshared producer of the same
    attention type.  GeGLU is the ordinary ``gelu`` FFN activation.
    """
    c = _gemma_config(config)
    try:
        hidden, layers = int(c["hidden_size"]), int(c["num_hidden_layers"])
        heads, vocab = int(c["num_attention_heads"]), int(c["vocab_size"])
        head_dim = int(c.get("head_dim") or hidden // heads)
        global_head_dim = int(c.get("global_head_dim") or 512)
        kv_heads = int(c.get("num_key_value_heads", heads) or heads)
        global_kv_heads = int(c.get("num_global_key_value_heads") or kv_heads)
        intermediate = int(c.get("intermediate_size") or c.get("shared_expert_intermediate_size")
                           or c.get("ffn_dim") or c.get("moe_intermediate_size") or 1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelingError("gemma4_config_incomplete") from exc
    if min(hidden, layers, heads, head_dim, global_head_dim) < 1:
        raise ModelingError("gemma4_config_invalid")
    # Native configs declare layer_types as nullable: an explicit None means
    # the default six-layer (5, 11, ...) global schedule, not "last only".
    pattern = (c.get("layer_types") or c.get("sliding_window_pattern")
               or c.get("global_attention_every") or 6)
    window = int(c.get("sliding_window", c.get("sliding_window_size", 0)) or 0)
    raw_act = str(c.get("hidden_act", c.get("hidden_activation", "gelu"))).lower()
    activation = ("silu" if "silu" in raw_act or "swish" in raw_act else "gelu")
    experts = int(c.get("num_experts", c.get("n_routed_experts", 0)) or 0)
    top_k = int(c.get("num_experts_per_tok", c.get("top_k", 0)) or 0)
    moe_i = int(c.get("moe_intermediate_size", 0) or 0)
    shared = int(c.get("num_shared_experts", c.get("n_shared_experts", 0)) or 0)
    k_eq_v = bool(c.get("attention_k_eq_v", c.get("k_eq_v", c.get("use_k_eq_v", False))))
    schedule = tuple("global" if _gemma_global(index, layers, pattern) else "local" for index in range(layers))
    explicit_k = c.get("k_eq_v_layers")
    if explicit_k is not None:
        wanted = {int(value) for value in explicit_k}
        k_layers = tuple(index for index, kind in enumerate(schedule) if kind == "global" and index in wanted)
    elif k_eq_v:
        # K=V is a full-attention alternative; sliding layers keep a real V.
        k_layers = tuple(index for index, kind in enumerate(schedule) if kind == "global")
    else:
        k_layers = ()
    ple_raw = c.get("per_layer_embedding", c.get("ple", c.get("hidden_size_per_layer_input", 0)))
    ple = bool(ple_raw)
    ple_width = int(c.get("ple_width", c.get("hidden_size_per_layer_input", hidden)) or hidden)
    ple_layers = c.get("ple_layers")
    ple_layers = (tuple(range(layers)) if ple else ()) if ple_layers is None else tuple(
        int(value) for value in ple_layers)
    sources = c.get("kv_source_layer_ids", c.get("kv_consumer_to_owner"))
    shared_layers = int(c.get("num_kv_shared_layers", 0) or 0)
    if isinstance(sources, Mapping):
        owners = tuple(int(sources.get(str(i), sources.get(i, i))) for i in range(layers))
    elif sources is not None:
        if len(sources) != layers:
            raise ModelingError("gemma4_kv_source_length_mismatch")
        owners = tuple(int(value) for value in sources)
    elif shared_layers > 0:
        unshared = layers - shared_layers
        last_by_type: dict[str, int] = {}
        for index in range(max(0, unshared)):
            last_by_type[schedule[index]] = index
        values = list(range(max(0, unshared))) + [0] * shared_layers
        for index in range(max(0, unshared), layers):
            kind = schedule[index]
            if kind not in last_by_type:
                raise ModelingError(f"gemma4_shared_kv_without_producer:{kind}")
            values[index] = last_by_type[kind]
        owners = tuple(values)
    else:
        owners = tuple(range(layers))
    plans = []
    for index in range(layers):
        global_attn = schedule[index] == "global"
        layer_hd = global_head_dim if global_attn else head_dim
        layer_kv = (global_kv_heads if global_attn and k_eq_v else kv_heads)
        moe = bool(experts and top_k)
        plans.append(LayerPlan( index=index, attention="gqa", architecture="gemma4",
            ffn="moe" if moe else "dense", window=0 if global_attn else window,
            num_heads=heads, num_kv_heads=layer_kv, head_dim=layer_hd,
            intermediate_size=intermediate if not moe else 0,
            num_experts=experts if moe else 0, top_k=top_k if moe else 0, moe_intermediate_size=moe_i if moe else 0,
            num_shared_experts=shared if moe else 0,
            shared_intermediate_size=int(c.get("shared_expert_intermediate_size", 0) or 0),
            qk_norm=bool(c.get("use_qk_norm", c.get("qk_norm", True))),
            ffn_activation=activation, kv_owner_layer_id=owners[index],
            owns_long_kv=owners[index] == index, k_eq_v=(index in set(k_layers)),
            native_layer_id=index))
    facts = tuple(sorted({ "gemma4": 1, "gemma4_ple": int(ple),
        "gemma4_ple_layers": ple_layers, "gemma4_ple_width": ple_width,
        "gemma4_ple_tables": int(c.get("ple_tables", 1) or 1), "gemma4_ple_gated": bool(c.get("ple_gated", True)),
        "gemma4_k_eq_v_all": 0, "gemma4_k_eq_v_layers": k_layers,
        "norm_weight_offset": float(c.get("norm_weight_offset", 1.0) or 0.0),
        "attention_schedule": schedule}.items()))
    dtype = str(c.get("torch_dtype", c.get("dtype", "bf16"))).replace("torch.", "")
    return ModelSpec( name=name, layers=layers, hidden_size=hidden, num_heads=heads,
        head_dim=head_dim, intermediate_size=intermediate or moe_i or 1,
        vocab_size=vocab, attention="gqa", ffn="moe" if experts else "dense",
        num_kv_heads=kv_heads, activation_dtype=dtype, weight_dtype=dtype,
        kv_dtype=str(c.get("kv_dtype", dtype)), num_experts=experts, top_k=top_k,
        moe_intermediate_size=moe_i, num_shared_experts=shared,
        tie_word_embeddings=bool(c.get("tie_word_embeddings", False)),
        layer_plans=tuple(plans), body_layers=layers, catalog_facts=facts,
        metadata=(("family", "gemma4"), ("model_coverage", "structural")))

BUILDERS = { "E0": _build_e0, "N0": _build_n0, "A0": _build_a0, "A1": _build_a1, "A2": _build_a2, "A3": _build_a3,
    "A4": _build_a4, "K0": _build_k0, "K1": _build_k1, "I0": _build_i0, "I1": _build_i1, }
