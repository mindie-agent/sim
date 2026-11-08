"""Canonical builders for FFN, routing, residual, output and draft families.

[Role] Construct F0-F4/R0-R1/E1/O0/D0/V0 templates and static signatures.
[Invariants] Shared/routed branches, quant roles and output accounting are
  explicit; a macro does not recreate a child subgraph's work.
[Timeouts & Locks] Pure bounded construction; no I/O or locks.
[Side-Effect Policy] Returns immutable ``_Built`` records.
"""
from __future__ import annotations

from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.subgraphs import FusionContract
from .instantiate import (
    _Ctx, _Built, _HIDDEN_IN, _HIDDEN_OUT, _attention_sig, _built, _collective,
    _copy, _elementwise, _in, _matrix, _out, _parallel, _precede, _quant_fields,
    _sig, _state, _tiles, _variant,
)


def _build_f0(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, str(ctx.layer.ffn_activation or "silu"))
    act_bytes = dtype_bytes(ctx.features.activation_format)
    ungated = variant == "ungated"
    gate_n = ctx.local_intermediate if ungated else 2 * ctx.local_intermediate
    specs = [("gate_up", "gemm", "gate_up"), ("activation", "elementwise", "activation"),
             ("down", "gemm", "down")]
    signatures = {
        "gate_up": _matrix("gate_up", ctx.hidden, gate_n, ctx, "dense_mlp"),
        "activation": _elementwise(
            "activation", ctx, elements=ctx.local_intermediate, ops=4,
            read_bytes=(ctx.local_intermediate if ungated
                        else 2 * ctx.local_intermediate) * act_bytes,
            write_bytes=ctx.local_intermediate * act_bytes,
            extra=(("activation", variant), ("gated", int(not ungated)))),
        "down": _matrix("down", ctx.local_intermediate, ctx.hidden, ctx, "dense_mlp"),
    }
    if ctx.tp > 1:
        specs.append(("tp_allreduce", "collective", "tp"))
        signatures["tp_allreduce"] = _collective(
            "tp", ctx, axis="tp", members=ctx.tp,
            bytes_per_token=ctx.hidden * act_bytes)
    # An ungated projection has no gate/value pair to fuse, so it must not
    # advertise the gated dequant_swiglu_quant contract.
    fusions = () if ungated else (FusionContract(
        "dequant_swiglu_quant", ("gate_up", "activation"),
        removes=("ffn_activation", "activation_intermediate"),
        keeps=("down_input", "activation_scale", "residual")),)
    return _built("F0", ctx, variant, "Dense Feed-Forward Network (FFN 稠密前馈网络)",
                  specs, signatures, fusions=fusions,
                  state_in=(_HIDDEN_IN,), state_out=(_HIDDEN_OUT,))


def _build_f1(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, str(ctx.layer.router_mode or "softmax"))
    experts = int(ctx.model.num_experts or 1)
    specs = [("router_gemm", "router", "router"), ("router_topk", "router", "topk"),
             ("router_histogram", "router", "histogram"), ("router_sort", "router", "sort")]
    signatures = {
        "router_gemm": _matrix("router", ctx.hidden, experts, ctx, "router",
                               extra=(("groups", int(ctx.layer.router_groups or 1)),
                                      ("group_topk", int(ctx.layer.router_group_topk or 0)))),
        "router_topk": _elementwise("topk", ctx, elements=experts, ops=4,
                             read_bytes=experts * 4, write_bytes=ctx.top_k * 8,
                             extra=(("E", experts), ("k", ctx.top_k))),
        "router_histogram": _elementwise("histogram", ctx, elements=ctx.top_k, ops=4,
                                  read_bytes=ctx.top_k * 8, write_bytes=experts * 8,
                                  extra=(("E", experts), ("k", ctx.top_k))),
        "router_sort": _elementwise("sort", ctx, elements=ctx.top_k, ops=8,
                             read_bytes=ctx.top_k * 8, write_bytes=ctx.top_k * 8,
                             extra=(("E", experts), ("k", ctx.top_k))),
    }
    return _built("F1", ctx, variant, "MoE Routing and Token Permutation (MoE 路由与 Token 重排)",
                  specs, signatures,
                  coverage="partial",
                  state_in=(_HIDDEN_IN,), state_out=(_out("routing_ledger", "routing"),
                                                     _out("expert_rows", "routing")))


def _build_f2(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, ctx.features.moe_recipe)
    hidden = ctx.latent
    top_k = ctx.top_k
    wire = dtype_bytes("bf16")
    dispatch_bytes = hidden * wire
    extra = (("ep", ctx.ep), ("etp", ctx.etp), ("top_k", top_k),
             ("axis", "ep"), ("members", ctx.ep),
             ("chunks", ctx.features.mc2_chunks),
             ("recipe", ctx.features.moe_recipe),
             ("overlap", int(ctx.features.moe_recipe == "mc2")),
             ("latent", hidden), ("wire_bytes", wire))
    combine = _sig("combine", matrix=(("row_bytes", dispatch_bytes),),
                   parallel=_parallel(ctx), tiles=_tiles(ctx),
                   quant=(("wire_bytes", wire),),
                   extras=extra + (("direction", "receive"),))
    if ctx.role == "transport_combine":
        # Combine consumes the routed expert output, so it is a distinct
        # post-expert instance scheduled after F3.
        return _built("F2", ctx, variant,
                      "MoE Token Dispatch and Combine (MoE Token 分发与结果合并)",
                      [("combine", "transport", "combine")],
                      {"combine": combine}, coverage="partial",
                      state_in=(_HIDDEN_IN, _in("expert_rows", "routing")),
                      state_out=(_HIDDEN_OUT, _out("peer_traffic", "traffic")))
    quantize = _elementwise("quant", ctx, elements=hidden, ops=2,
                            read_bytes=hidden * 2, write_bytes=hidden * wire,
                            extra=(("expert_count", ctx.local_experts),))
    dispatch = _sig("dispatch", matrix=(("row_bytes", dispatch_bytes),),
                    parallel=_parallel(ctx), tiles=_tiles(ctx),
                    quant=(("wire_bytes", wire),),
                    extras=extra + (("direction", "send"), ("input_quantized", 1)))
    notify = _collective("notify", ctx, axis="ep", members=ctx.ep,
                         bytes_per_token=8.0, stages=1,
                         extra=(("chunks", ctx.features.mc2_chunks),))
    return _built("F2", ctx, variant,
                  "MoE Token Dispatch and Combine (MoE Token 分发与结果合并)",
                  [("quantize", "elementwise", "quant"),
                   ("dispatch", "transport", "dispatch"),
                   ("notify", "collective", "notify")],
                  {"quantize": quantize, "dispatch": dispatch, "notify": notify},
                  coverage="partial",
                  state_in=(_HIDDEN_IN, _in("expert_rows", "routing")),
                  state_out=(_out("peer_traffic", "traffic"),))


def _build_f3(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "latent" if ctx.layer.expert_latent_size else "packed")
    experts = int(ctx.model.num_experts or 1)
    local = ctx.local_experts
    expert_width = ctx.expert_width
    latent = ctx.latent
    specs = [("expert_gate_up", "grouped_gemm", "gate_up"),
             ("expert_activation", "elementwise", "activation"),
             ("expert_down", "grouped_gemm", "down")]
    extra = (("E", experts), ("E_local", local), ("k", ctx.top_k),
             ("latent", latent), ("expert_width", expert_width),
             ("row_tile", ctx.features.moe_row_tile))
    signatures = {
        "expert_gate_up": _sig("expert_gate_up",
                               matrix=(("K", latent), ("N", 2 * expert_width)),
                               experts=(("E", experts), ("local", local), ("k", ctx.top_k)),
                               quant=_quant_fields(ctx, "routed_expert", 2 * expert_width, latent),
                               parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra),
        "expert_activation": _elementwise("activation", ctx, elements=expert_width, ops=4,
                                          read_bytes=2 * expert_width * 2,
                                          write_bytes=expert_width * 2,
                                          extra=(("expert_width", expert_width),
                                                 ("E", experts), ("local", local),
                                                 ("k", ctx.top_k))),
        "expert_down": _sig("expert_down",
                            matrix=(("K", expert_width), ("N", latent)),
                            experts=(("E", experts), ("local", local), ("k", ctx.top_k)),
                            quant=_quant_fields(ctx, "routed_expert", latent, expert_width),
                            parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra),
    }
    _precede(specs, signatures, ctx, "expert_gate_up", latent, "routed_expert")
    _precede(specs, signatures, ctx, "expert_down", expert_width, "routed_expert")
    if ctx.etp > 1:
        specs.append(("etp_allreduce", "collective", "etp"))
        signatures["etp_allreduce"] = _collective(
            "etp", ctx, axis="etp", members=ctx.etp,
            bytes_per_token=latent * dtype_bytes(ctx.features.activation_format))
    return _built("F3", ctx, variant, "Routed Expert FFN (路由专家前馈网络)", specs, signatures,
                  coverage="partial",
                  state_in=(_HIDDEN_IN, _in("expert_rows", "routing")),
                  state_out=(_HIDDEN_OUT,))


def _build_f4(ctx: _Ctx) -> _Built:
    hidden = ctx.hidden
    act_bytes = dtype_bytes(ctx.features.activation_dtype or "bf16")
    if ctx.role == "join" or ctx.variant == "join":
        join = _elementwise("join", ctx, elements=hidden, ops=2,
                            read_bytes=2 * hidden * act_bytes,
                            write_bytes=hidden * act_bytes)
        return _built("F4", ctx, "join",
                      "Shared Expert FFN and Output Merge (共享专家前馈网络与输出合并)",
                      [("join", "elementwise", "join")], {"join": join},
                      state_in=(_HIDDEN_IN,), state_out=(_HIDDEN_OUT,))
    tp = max(1, ctx.tp)
    width = max(1, (ctx.shared_width + tp - 1) // tp)
    gated = bool(ctx.layer.shared_expert_gate)
    variant = _variant(ctx, "gated" if gated else "plain")
    specs = [("shared_gate_up", "gemm", "gate_up"),
             ("shared_activation", "elementwise", "activation"),
             ("shared_down", "gemm", "down")]
    signatures = {
        "shared_gate_up": _matrix("shared_gate_up", hidden, 2 * width, ctx,
                                  "shared_expert",
                                  extra=(("local_width", width), ("tp", tp))),
        "shared_activation": _elementwise("activation", ctx, elements=width, ops=4,
                                          read_bytes=2 * width * 2,
                                          write_bytes=width * 2,
                                          extra=(("local_width", width),)),
        "shared_down": _matrix("shared_down", width, hidden, ctx, "shared_expert",
                               extra=(("local_width", width), ("tp", tp))),
    }
    _precede(specs, signatures, ctx, "shared_gate_up", hidden, "shared_expert")
    _precede(specs, signatures, ctx, "shared_down", width, "shared_expert")
    if gated:
        specs.extend((("shared_gate", "gemm", "gate"),
                      ("shared_gate_apply", "elementwise", "gate_apply")))
        signatures["shared_gate"] = _matrix("shared_gate", hidden, 1, ctx, "router")
        signatures["shared_gate_apply"] = _elementwise(
            "gate_apply", ctx, elements=hidden, ops=2,
            read_bytes=2 * hidden * act_bytes, write_bytes=hidden * act_bytes)
    if tp > 1:
        specs.append(("tp_allreduce", "collective", "tp"))
        signatures["tp_allreduce"] = _collective(
            "tp", ctx, axis="tp", members=tp,
            bytes_per_token=hidden * act_bytes)
    stream = "aux" if ctx.features.multistream_shared_expert else "main"
    return _built("F4", ctx, variant,
                  "Shared Expert FFN and Output Merge (共享专家前馈网络与输出合并)",
                  specs, signatures,
                  state_in=(_HIDDEN_IN,), state_out=(_HIDDEN_OUT,),
                  contract=(("stream", stream), ("shared_expert_gate", int(gated)),
                            ("local_width", width)))


def _build_r0(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "hc")
    mult = max(1, int(ctx.layer.hc_mult or 1))
    iters = int(ctx.layer.hc_sinkhorn_iters or 0)
    hidden = ctx.hidden
    streams = mult * hidden
    project_n = 2 * mult + mult * mult
    b_x = dtype_bytes(ctx.features.activation_dtype or "bf16")
    coeff = (mult + mult * mult) * 4
    if variant in {"pre", "attn_pre", "ffn_pre", "hc", "hc_arch35"}:
        specs = [("inv_rms", "elementwise", "norm"),
                 ("hc_projection", "gemm", "projection"),
                 ("sinkhorn", "elementwise", "sinkhorn"),
                 ("hc_pre_mix", "elementwise", "hc_pre")]
        signatures = {
            "inv_rms": _elementwise("norm", ctx, elements=streams, ops=5,
                                    read_bytes=streams * b_x, write_bytes=streams * b_x,
                                    extra=(("mult", mult),)),
            "hc_projection": _matrix("hc_projection", streams, project_n, ctx, "mhc",
                                  extra=(("mult", mult), ("static_dims", mult))),
            "sinkhorn": _elementwise("sinkhorn", ctx, elements=mult * mult, ops=6,
                                     read_bytes=mult * mult * 4, write_bytes=mult * mult * 4,
                                     extra=(("mult", mult), ("iters", iters),
                                            ("static_dims", mult))),
            "hc_pre_mix": _elementwise(
                "hc_pre", ctx, elements=hidden, ops=max(1, 2 * mult - 1),
                read_bytes=streams * b_x, write_bytes=hidden * b_x + coeff,
                extra=(("mult", mult), ("static_dims", mult))),
        }
    elif variant in {"post", "attn_post", "ffn_post"}:
        specs = [("hc_post_mix", "hc_post", "hc_post")]
        signatures = {"hc_post_mix": _sig(
            "hc_post",
            matrix=(("elements", streams), ("mult", mult), ("static_dims", mult)),
            quant=(("act_bytes", b_x),), parallel=_parallel(ctx), tiles=_tiles(ctx),
            extras=(("ops_per_element", 2 * mult + 1), ("mult", mult),
                    ("static_dims", mult),
                    ("read_bytes", streams * b_x + hidden * b_x + coeff),
                    ("write_bytes", streams * b_x)))}
    else:
        raise ModelingError(f"hc_variant_unsupported:{variant}")
    return _built("R0", ctx, variant,
                  "Hyper-Connections (HC 超连接; mHC 流形约束超连接)", specs, signatures,
                  state_in=(_HIDDEN_IN,), state_out=(_HIDDEN_OUT,))


def _build_r1(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "block_stack")
    block = int(ctx.layer.residual_block_size or 1)
    # Completed block outputs plus the current residual source.  A block only
    # becomes a source after its last member layer, so layer 0 has one source
    # and layer block sees two; the count is never the model-wide block count.
    slots = 1 + int(ctx.layer.index) // block
    act_bytes = dtype_bytes(ctx.features.activation_dtype or "bf16")
    specs = [("slot_push", "copy", "slot"), ("norm", "elementwise", "norm"),
             ("score", "gemm", "score"),
             ("softmax_mix", "elementwise", "mix"), ("cast", "copy", "cast")]
    signatures = {
        "slot_push": _copy("slot", ctx, elements=ctx.hidden * slots,
                           bytes_per_element=act_bytes,
                           extra=(("slots", slots), ("block", block))),
        "norm": _elementwise("norm", ctx, elements=ctx.hidden * slots, ops=5,
                             read_bytes=ctx.hidden * slots * act_bytes,
                             write_bytes=ctx.hidden * slots * act_bytes,
                             extra=(("slots", slots), ("block", block))),
        # A learned hidden->1 score projection plus its K reduction: one fp32
        # scalar per live source, not a hidden-width elementwise tensor.
        "score": _matrix("score", ctx.hidden, slots, ctx, "residual_score",
                         extra=(("slots", slots), ("block", block), ("score_dtype", "fp32"))
                         ).relocated(output_format="fp32"),
        "softmax_mix": _elementwise("mix", ctx, elements=ctx.hidden * slots, ops=5,
                                    read_bytes=ctx.hidden * slots * act_bytes + slots * 4,
                                    write_bytes=ctx.hidden * act_bytes,
                                    extra=(("slots", slots), ("block", block),
                                           ("score_bytes", slots * 4))),
        "cast": _copy("cast", ctx, elements=ctx.hidden, bytes_per_element=act_bytes),
    }
    return _built("R1", ctx, variant, "Attention Residuals (AttnRes 注意力残差连接)", specs, signatures,
                  state_in=(_HIDDEN_IN, _in("residual_slots", "residual")),
                  state_out=(_HIDDEN_OUT, _out("residual_slots", "residual")))


def _build_e1(ctx: _Ctx) -> _Built:
    variant = "engram" if ctx.layer.engram else "ple"
    extra = ()
    facts = dict(ctx.model.catalog_facts)
    width = ctx.hidden
    if ctx.model.engram is not None:
        width = int(ctx.model.engram.width or ctx.hidden)
        extra = (("tables", len(ctx.model.engram.tables)), ("width", width))
    elif facts.get("gemma4"):
        width = int(facts.get("gemma4_ple_width", ctx.hidden) or ctx.hidden)
        extra = (("tables", int(facts.get("gemma4_ple_tables", 1))),
                 ("width", width),
                 ("vocab", int(facts.get("gemma4_ple_vocab", ctx.model.vocab_size))),
                 ("gated", int(bool(facts.get("gemma4_ple_gated", True)))))
    # PLE is hidden -> ple_width (gate) -> hidden (projection); the table and
    # gate application live at the declared width, not the model hidden size.
    act_bytes = dtype_bytes(ctx.features.activation_dtype or "bf16")
    specs = [("table_lookup", "embedding", "lookup"),
             ("gate", "gemm", "gate"),
             ("gate_activation", "elementwise", "activation"),
             ("gate_apply", "elementwise", "gate_apply"),
             ("projection", "gemm", "projection"),
             ("post_projection_norm", "elementwise", "norm"),
             ("residual_add", "elementwise", "residual")]
    signatures = {
        "table_lookup": _sig("lookup", matrix=(("width", width),),
                             quant=(("act_bytes", 2.0),),
                             parallel=_parallel(ctx), tiles=_tiles(ctx), extras=extra),
        "gate": _matrix("ple_gate", ctx.hidden, width, ctx, "ple_gate"),
        "gate_activation": _elementwise("activation", ctx, elements=width, ops=4,
                                        read_bytes=width * act_bytes, write_bytes=width * act_bytes,
                                        extra=(("activation", "silu"),)),
        "gate_apply": _elementwise("gate_apply", ctx, elements=width, ops=2,
                                   read_bytes=2 * width * act_bytes, write_bytes=width * act_bytes),
        "projection": _matrix("projection", width, ctx.hidden, ctx, "engram_table"),
        "post_projection_norm": _elementwise("norm", ctx, elements=ctx.hidden, ops=5,
                                             read_bytes=ctx.hidden * act_bytes,
                                             write_bytes=ctx.hidden * act_bytes,
                                             extra=(("ple_post_projection", 1),)),
        "residual_add": _elementwise(
            "residual", ctx, elements=ctx.hidden, ops=2,
            read_bytes=2 * ctx.hidden * act_bytes,
            write_bytes=ctx.hidden * act_bytes,
            extra=(("ple_residual", 1),)),
    }
    return _built("E1", ctx, variant,
                  "Engram / Per-Layer Embeddings (PLE Engram 条件记忆／逐层嵌入)",
                  specs, signatures, coverage="structural",
                  state_in=(_HIDDEN_IN, _in("lookup_tokens", "token_index")),
                  state_out=(_HIDDEN_OUT,))


def _build_o0(ctx: _Ctx) -> _Built:
    scope = ctx.features.final_norm_scope
    variant = _variant(ctx, scope)
    act_bytes = dtype_bytes(ctx.features.activation_dtype or "bf16")
    norm = _elementwise("norm", ctx, elements=ctx.hidden, ops=5,
                        read_bytes=ctx.hidden * act_bytes,
                        write_bytes=ctx.hidden * act_bytes)
    select = _copy("select", ctx, elements=ctx.hidden,
                   bytes_per_element=act_bytes)
    if scope == "selected":
        # Select the last valid row first, then normalise only those rows.
        specs = [("hidden_select", "copy", "select"),
                 ("final_norm", "elementwise", "norm")]
        signatures = {"hidden_select": select, "final_norm": norm}
    else:
        # Normalise every row; lm_head/sampling still consume selected rows.
        specs = [("final_norm", "elementwise", "norm")]
        signatures = {"final_norm": norm}
    specs += [("lm_head", "gemm", "lm_head"), ("logits_gather", "copy", "gather"),
              ("sampling", "elementwise", "sample")]
    signatures["lm_head"] = _matrix("lm_head", ctx.hidden, ctx.model.vocab_size,
                                    ctx, "lm_head",
                                    extra=(("selected_rows", 1),))
    signatures["logits_gather"] = _copy("logits", ctx, elements=ctx.model.vocab_size,
                                        bytes_per_element=4.0)
    signatures["sampling"] = _elementwise(
        "sample", ctx, elements=ctx.model.vocab_size, ops=4,
        read_bytes=ctx.model.vocab_size * 4, write_bytes=8.0,
        extra=(("vocab", ctx.model.vocab_size), ("tied", int(ctx.model.tie_word_embeddings))))
    return _built("O0", ctx, variant,
                  "Final Normalization, LM Head and Sampling (最终归一化、语言模型输出头与采样)",
                  specs, signatures,
                  state_in=(_HIDDEN_IN,), state_out=(_out("logits", "logits"),
                                                     _out("sampled_tokens", "token")))


def _append_mla_draft_layer(
    specs, signatures, ctx, draft, layer, *,
    draft_hidden, draft_intermediate, heads, head_dim, kv_row, act_bytes,
) -> None:
    """One latent-MLA draft block: low-rank Q/KV, then the dense FFN."""
    prefix = f"draft_layer{layer}"
    q_lora = max(1, int(getattr(draft, "q_lora_rank", 0) or 0))
    kv_lora = max(1, int(getattr(draft, "kv_lora_rank", 0) or 0))
    nope = int(getattr(draft, "qk_nope_head_dim", 0) or 0)
    rope = int(getattr(draft, "qk_rope_head_dim", 0) or 0)
    v_dim = int(getattr(draft, "v_head_dim", 0) or head_dim)
    q_dim = nope + rope if nope or rope else head_dim
    kv_width = kv_lora + rope
    specs.extend([
        (f"{prefix}_attn_norm", "elementwise", "norm"),
        (f"{prefix}_q_a", "gemm", "projection"),
        (f"{prefix}_q_norm", "elementwise", "norm"),
        (f"{prefix}_q_b", "gemm", "projection"),
        (f"{prefix}_kv_a", "gemm", "projection"),
        (f"{prefix}_kv_norm", "elementwise", "norm"),
        (f"{prefix}_kv_store", "kv_access", "produce"),
        (f"{prefix}_attention", "attention", "attention"),
        (f"{prefix}_o_projection", "gemm", "output"),
        (f"{prefix}_attn_residual", "elementwise", "residual"),
        (f"{prefix}_ffn_norm", "elementwise", "norm"),
        (f"{prefix}_gate_up", "gemm", "gate_up"),
        (f"{prefix}_activation", "elementwise", "activation"),
        (f"{prefix}_down", "gemm", "down"),
        (f"{prefix}_ffn_residual", "elementwise", "residual"),
    ])
    signatures[f"{prefix}_attn_norm"] = _elementwise(
        "norm", ctx, elements=draft_hidden, ops=5,
        read_bytes=draft_hidden * act_bytes, write_bytes=draft_hidden * act_bytes)
    signatures[f"{prefix}_q_a"] = _matrix(
        "draft_q_a", draft_hidden, q_lora, ctx, "mtp")
    signatures[f"{prefix}_q_norm"] = _elementwise(
        "norm", ctx, elements=q_lora, ops=5,
        read_bytes=q_lora * act_bytes, write_bytes=q_lora * act_bytes)
    signatures[f"{prefix}_q_b"] = _matrix(
        "draft_q_b", q_lora, heads * q_dim, ctx, "mtp")
    signatures[f"{prefix}_kv_a"] = _matrix(
        "draft_kv_a", draft_hidden, kv_width, ctx, "mtp")
    signatures[f"{prefix}_kv_norm"] = _elementwise(
        "norm", ctx, elements=kv_lora, ops=5,
        read_bytes=kv_lora * act_bytes, write_bytes=kv_lora * act_bytes)
    signatures[f"{prefix}_kv_store"] = _state(
        "kv_store", ctx, rows=1, row_bytes=float(kv_row),
        extra=(("owner", 1), ("layout", "draft_kv"), ("draft_layer", layer)))
    signatures[f"{prefix}_attention"] = _attention_sig(
        "attention", ctx, local_heads=heads, local_kv_heads=1,
        group_heads=heads, width=q_dim, value_width=v_dim,
        kv_row_bytes=float(kv_row),
        extra=(("branch", "draft"), ("attention", "mla"),
               ("draft_layer", layer), ("require_kv", ctx.layer.index)))
    signatures[f"{prefix}_o_projection"] = _matrix(
        "draft_output", heads * v_dim, draft_hidden, ctx, "mtp")
    signatures[f"{prefix}_attn_residual"] = _elementwise(
        "residual", ctx, elements=draft_hidden, ops=2,
        read_bytes=2 * draft_hidden * act_bytes,
        write_bytes=draft_hidden * act_bytes)
    signatures[f"{prefix}_ffn_norm"] = _elementwise(
        "norm", ctx, elements=draft_hidden, ops=5,
        read_bytes=draft_hidden * act_bytes, write_bytes=draft_hidden * act_bytes)
    signatures[f"{prefix}_gate_up"] = _matrix(
        "draft_gate_up", draft_hidden, 2 * draft_intermediate, ctx, "mtp")
    signatures[f"{prefix}_activation"] = _elementwise(
        "activation", ctx, elements=draft_intermediate, ops=4,
        read_bytes=2 * draft_intermediate * act_bytes,
        write_bytes=draft_intermediate * act_bytes)
    signatures[f"{prefix}_down"] = _matrix(
        "draft_down", draft_intermediate, draft_hidden, ctx, "mtp")
    signatures[f"{prefix}_ffn_residual"] = _elementwise(
        "residual", ctx, elements=draft_hidden, ops=2,
        read_bytes=2 * draft_hidden * act_bytes,
        write_bytes=draft_hidden * act_bytes)
    _precede(specs, signatures, ctx, f"{prefix}_q_b", q_lora, "mtp")
    _precede(specs, signatures, ctx, f"{prefix}_o_projection", heads * v_dim, "mtp")
    _precede(specs, signatures, ctx, f"{prefix}_gate_up", draft_hidden, "mtp")
    _precede(specs, signatures, ctx, f"{prefix}_down", draft_intermediate, "mtp")


def _build_d0(ctx: _Ctx) -> _Built:
    draft = None
    if ctx.model.auxiliary is not None:
        draft = (ctx.model.auxiliary.dspark if ctx.variant == "dspark" and ctx.model.auxiliary.dspark
                 else (ctx.model.auxiliary.mtp or ctx.model.auxiliary.dspark))
    default_var = ("dspark" if getattr(draft, "kind", "") in {"dspark", "mtp_dspark"}
                   else ("mtp" if ctx.model.has_optional_draft else "standalone"))
    variant = _variant(ctx, default_var)
    draft_hidden = int(getattr(draft, "hidden_size", 0) or ctx.hidden)
    proj_in = int(getattr(draft, "input_width", 0) or ctx.hidden)
    draft_layers = max(1, int(getattr(draft, "layers", 0) or 1))
    draft_intermediate = max(
        1, int(getattr(draft, "intermediate_size", 0) or ctx.local_intermediate))
    proposal = max(1, int(getattr(draft, "proposal_tokens", 0) or 1))
    verify = max(proposal + 1, int(getattr(draft, "verify_tokens", 0) or 0))
    heads = max(1, int(getattr(draft, "num_heads", 0) or ctx.heads))
    kv_heads = max(1, int(getattr(draft, "num_kv_heads", 0) or ctx.kv_heads))
    head_dim = max(1, int(getattr(draft, "head_dim", 0) or draft_hidden // heads))
    shared = int(bool(getattr(draft, "target_shared_lm_head", False)
                      or getattr(draft, "target_shared_embedding", False)))
    noncausal = int(bool(getattr(draft, "noncausal_query_block", False)
                         or getattr(draft, "kind", "") in {"dspark", "mtp_dspark"}))
    markov_rank = int(getattr(draft, "markov_rank", 0) or 0)
    act_bytes = dtype_bytes(ctx.features.activation_dtype or "bf16")
    row_elems = ((draft.kv_cache_elements or draft.kv_lora_rank + draft.qk_rope_head_dim)
                 if getattr(draft, "attention_kind", "gqa") == "mla" and draft is not None
                 else 2 * local_kv_heads(kv_heads, ctx.features.tp) * head_dim)
    kv_row = row_elems * dtype_bytes(ctx.features.kv_dtype)
    specs = [("draft_projection", "gemm", "projection"),
             ("draft_heads", "draft", "draft")]
    signatures = {
        "draft_projection": _matrix("draft_projection", proj_in, draft_hidden,
                                    ctx, "mtp"),
        "draft_heads": _sig(
            "draft_heads", matrix=(("K", draft_hidden), ("N", draft_hidden)),
            quant=_quant_fields(ctx, "mtp", draft_hidden, draft_hidden),
            parallel=_parallel(ctx), tiles=_tiles(ctx),
            extras=(("draft_len", proposal), ("layers", draft_layers),
                    ("intermediate", draft_intermediate), ("hidden", draft_hidden),
                    ("heads", heads), ("kv_heads", kv_heads), ("shared", shared),
                    ("noncausal_block", noncausal), ("markov_rank", markov_rank),
                    ("kv_row_bytes", int(kv_row)))),
    }
    for layer in range(draft_layers):
        if getattr(draft, "attention_kind", "gqa") == "mla":
            _append_mla_draft_layer(
                specs, signatures, ctx, draft, layer,
                draft_hidden=draft_hidden, draft_intermediate=draft_intermediate,
                heads=heads, head_dim=head_dim, kv_row=kv_row, act_bytes=act_bytes)
            continue
        prefix = f"draft_layer{layer}"
        specs.extend([
            (f"{prefix}_attn_norm", "elementwise", "norm"),
            (f"{prefix}_qkv", "gemm", "projection"),
            (f"{prefix}_kv_store", "kv_access", "produce"),
            (f"{prefix}_attention", "attention", "attention"),
            (f"{prefix}_o_projection", "gemm", "output"),
            (f"{prefix}_attn_residual", "elementwise", "residual"),
            (f"{prefix}_ffn_norm", "elementwise", "norm"),
            (f"{prefix}_gate_up", "gemm", "gate_up"),
            (f"{prefix}_activation", "elementwise", "activation"),
            (f"{prefix}_down", "gemm", "down"),
            (f"{prefix}_ffn_residual", "elementwise", "residual"),
        ])
        signatures[f"{prefix}_attn_norm"] = _elementwise(
            "norm", ctx, elements=draft_hidden, ops=5,
            read_bytes=draft_hidden * act_bytes, write_bytes=draft_hidden * act_bytes)
        signatures[f"{prefix}_qkv"] = _matrix(
            "draft_qkv", draft_hidden, (heads + 2 * kv_heads) * head_dim,
            ctx, "mtp")
        signatures[f"{prefix}_kv_store"] = _state(
            "kv_store", ctx, rows=1, row_bytes=float(kv_row),
            extra=(("owner", 1), ("layout", "draft_kv"), ("draft_layer", layer)))
        signatures[f"{prefix}_attention"] = _attention_sig(
            "attention", ctx, local_heads=heads, local_kv_heads=kv_heads,
            group_heads=max(1, heads // kv_heads), width=head_dim,
            value_width=head_dim, kv_row_bytes=float(kv_row),
            extra=(("branch", "draft"), ("draft_layer", layer),
                   ("require_kv", ctx.layer.index)))
        signatures[f"{prefix}_o_projection"] = _matrix(
            "draft_output", heads * head_dim, draft_hidden, ctx, "mtp")
        signatures[f"{prefix}_attn_residual"] = _elementwise(
            "residual", ctx, elements=draft_hidden, ops=2,
            read_bytes=2 * draft_hidden * act_bytes,
            write_bytes=draft_hidden * act_bytes)
        signatures[f"{prefix}_ffn_norm"] = _elementwise(
            "norm", ctx, elements=draft_hidden, ops=5,
            read_bytes=draft_hidden * act_bytes, write_bytes=draft_hidden * act_bytes)
        signatures[f"{prefix}_gate_up"] = _matrix(
            "draft_gate_up", draft_hidden, 2 * draft_intermediate, ctx, "mtp")
        signatures[f"{prefix}_activation"] = _elementwise(
            "activation", ctx, elements=draft_intermediate, ops=4,
            read_bytes=2 * draft_intermediate * act_bytes,
            write_bytes=draft_intermediate * act_bytes)
        signatures[f"{prefix}_down"] = _matrix(
            "draft_down", draft_intermediate, draft_hidden, ctx, "mtp")
        signatures[f"{prefix}_ffn_residual"] = _elementwise(
            "residual", ctx, elements=draft_hidden, ops=2,
            read_bytes=2 * draft_hidden * act_bytes,
            write_bytes=draft_hidden * act_bytes)
        _precede(specs, signatures, ctx, f"{prefix}_o_projection", heads * head_dim, "mtp")
        _precede(specs, signatures, ctx, f"{prefix}_gate_up", draft_hidden, "mtp")
        _precede(specs, signatures, ctx, f"{prefix}_down", draft_intermediate, "mtp")
    specs.extend((("draft_lm_head", "gemm", "lm_head"),
                  ("draft_sampling", "elementwise", "sample"),
                  ("target_verify", "draft", "verify"),
                  ("accept_commit", "copy", "commit")))
    signatures["draft_lm_head"] = _matrix(
        "draft_lm_head", draft_hidden, ctx.model.vocab_size, ctx, "lm_head")
    signatures["draft_sampling"] = _elementwise(
        "sample", ctx, elements=ctx.model.vocab_size, ops=4,
        read_bytes=ctx.model.vocab_size * 4, write_bytes=8.0,
        extra=(("vocab", ctx.model.vocab_size), ("draft", 1)))
    signatures["target_verify"] = _sig(
        "target_verify", matrix=(("K", ctx.hidden), ("N", ctx.model.vocab_size)),
        quant=_quant_fields(ctx, "lm_head", ctx.model.vocab_size, ctx.hidden),
        parallel=_parallel(ctx), tiles=_tiles(ctx),
        extras=(("verify_tokens", verify), ("shared_target", shared),
                ("discounted", 0)))
    signatures["accept_commit"] = _copy(
        "commit", ctx, elements=ctx.hidden, bytes_per_element=act_bytes,
        extra=(("draft_len", proposal), ("acceptance_policy", "declared")))
    return _built("D0", ctx, variant,
                  "Speculative Decoding: Draft and Verify (推测解码：起草与验证)",
                  specs, signatures,
                  coverage="partial",
                  state_in=(_HIDDEN_IN, _in("draft_state", "draft"),
                            _in("accept_state", "accept")),
                  state_out=(_HIDDEN_OUT, _out("accept_state", "accept")))

def _build_v0(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "vision" if ctx.model.vision is not None else "unified")
    vision = ctx.model.vision
    width = int(getattr(vision, "hidden_size", ctx.hidden) or ctx.hidden)
    specs = [("vision_encoder", "vision", "encoder"),
             ("projector", "gemm", "projection"), ("splice", "copy", "splice")]
    signatures = {
        "vision_encoder": _sig("vision_encoder", matrix=(("width", width),),
                               parallel=_parallel(ctx), tiles=_tiles(ctx),
                               extras=(("layers", int(getattr(vision, "layers", 0) or 0)),
                                       ("heads", int(getattr(vision, "num_heads", 0) or 0)),
                                       ("patch_width", int(getattr(vision, "patch_input_width",
                                                                  width))))),
        "projector": _matrix("projector", width, ctx.hidden, ctx, "vision"),
        "splice": _copy("splice", ctx, elements=ctx.hidden, bytes_per_element=2.0),
    }
    return _built("V0", ctx, variant,
                  "Multimodal Encoder and Projector (多模态编码器与投影)",
                  specs, signatures,
                  coverage="partial", state_in=(_in("media_tokens", "media"),),
                  state_out=(_HIDDEN_OUT,))


BUILDERS = {
    "F0": _build_f0,
    "F1": _build_f1,
    "F2": _build_f2,
    "F3": _build_f3,
    "F4": _build_f4,
    "R0": _build_r0,
    "R1": _build_r1,
    "E1": _build_e1,
    "O0": _build_o0,
    "D0": _build_d0,
    "V0": _build_v0,
}
