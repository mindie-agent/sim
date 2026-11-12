"""Canonical builders for input, attention, cache and indexer families.

[Role] Construct E0/N0/A0-A4/K0-K1/I0-I1 templates and their static signatures.
[Invariants] Builders consume only ``_Ctx``; no model-name or measurement branch.
  Child macros reference kinds instead of duplicating work.
[Timeouts & Locks] Pure bounded construction; no I/O or locks.
[Side-Effect Policy] Returns immutable ``_Built`` records.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from mindie_sim.prediction.gdn_decode_views import decode_views
from mindie_sim.prediction.sequence_parallel import kimi_sequence_parallel
from mindie_sim.prediction.indexer_source import (
    SOURCE_FAMILY as INDEXER_SOURCE_FAMILY, select_indexer_source)
from mindie_sim.prediction.piecewise.sfa_prolog import (
    port_spec as _sfa_port_spec, mla_prolog_v3_r64_primary,
    sfa_prolog_v3_c8_mx,
)
from mindie_sim.prediction.piecewise.vocab_parallel import (
    embedding_allreduce_world, embedding_table,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.subgraphs import FusionContract
from .instantiate import (_Ctx, _Built, _HIDDEN_IN, _HIDDEN_OUT, _KV_IN, _KV_OUT, _KV_READY_OUT,
    _RESIDUAL_IN, _RESIDUAL_OUT, _attention_sig, _built, _collective, _copy,
    _dense_flashcomm, _dense_sp_block,
    _activation_hidden_bytes, _elementwise, _in, _matrix, _norm_gamma, _out, _parallel,
    _precede, _quant_fields, _result_bytes, _sig, _state, _tiles, _variant,
    source_lowering_selected, )


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
    if cp <= 1 or str(ctx.features.cp_plan) not in {"dcp", "dsa_cp"}:
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


def _pcp_kv_gather(ctx: _Ctx, specs: list, signatures: dict) -> None:
    """Gather the current PCP prefill chunk's K/V across the PCP axis.

    Older context is already replicated; each rank receives the other ranks'
    ``(pcp-1)/pcp`` share.  The collective is prefill-only.
    """
    pcp = int(ctx.features.pcp)
    if pcp <= 1 or str(ctx.features.cp_plan) != "pcp":
        return
    specs.append(("pcp_kv_gather", "collective", "pcp"))
    signatures["pcp_kv_gather"] = _collective(
        "pcp_kv_gather", ctx, axis="pcp", members=pcp,
        bytes_per_token=ctx.kv_row_bytes * (pcp - 1), stages=1,
        extra=(("phase_scope", "prefill"),))


def _dsa_cp_output_all_gather(ctx: _Ctx, specs: list, signatures: dict) -> None:
    """Gather the local query-token output back to the full TP token stream.

    Source: `AscendDSACPImpl._gather_cp_output` in the retained
    `vllm_ascend/attention/context_parallel/dsa_cp.py`.  The message is the
    local padded query rows by the hidden width; the all-gather is prefill
    only.  It is not the DCP partial-output/LSE merge: KV rows stay full.
    """
    if str(ctx.features.cp_plan) != "dsa_cp" or ctx.tp <= 1:
        return
    # The source gathers the o_proj output tensor (`_apply_wo_b` ->
    # `_gather_cp_output`), which is the model activation dtype (BF16 here),
    # not the quantized int8 activation mix used by the GEMM input.
    out_bytes = dtype_bytes(ctx.features.activation_dtype or "bf16")
    specs.append(("dsa_cp_output_all_gather", "collective", "cp_output"))
    signatures["dsa_cp_output_all_gather"] = _collective(
        "dsa_cp_output_all_gather", ctx, axis="tp", members=ctx.tp,
        bytes_per_token=float(ctx.hidden * out_bytes), stages=1,
        extra=(("phase_scope", "prefill"),))


def _embedding_signature(ctx: _Ctx):
    """One vocab lookup. The table is the resident partition, not a GEMM."""
    table = embedding_table(ctx.model, ctx.features)
    return _sig(
        "embed",
        matrix=(("table_rows", int(table.rows)), ("table_cols", int(table.cols))),
        quant=_quant_fields(ctx, "embedding", table.rows, table.cols),
        parallel=_parallel(ctx), tiles=_tiles(ctx),
        extras=(("lookup", 1),
                ("masked", int(bool(table.masked))),
                ("embedding_mode", table.mode),
                ("shard_world", int(table.shard_world)),
                ("live_rows", "real_q")))


def _build_e0(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "token")
    specs = [("embedding", "embedding", "embed")]
    # Vocab-sharded lookup: mask out-of-shard ids, then one TP allreduce of
    # the BF16 hidden. Not a FlashComm edge, and not emitted for TP1,
    # a replicated table, a missing group, or fine-grained token exchange.
    reduce_world = embedding_allreduce_world(ctx.model, ctx.features)
    if reduce_world > 1:
        specs.append(("embedding_allreduce", "collective", "embedding_allreduce"))
    specs.append(("input_copy", "copy", "input"))
    if ctx.model.vision is not None or variant == "media":
        specs.append(("media_splice", "copy", "media"))
    signatures = { "embedding": _embedding_signature(ctx),
        "input_copy": _copy("input", ctx, elements=ctx.hidden, bytes_per_element=dtype_bytes(ctx.features.activation_format)),
        "media_splice": _copy("media", ctx, elements=ctx.hidden, bytes_per_element=dtype_bytes(ctx.features.activation_format)),
    }
    if reduce_world > 1:
        signatures["embedding_allreduce"] = _collective(
            "embedding_allreduce", ctx, axis="tp", members=reduce_world,
            bytes_per_token=_activation_hidden_bytes(ctx),
            extra=(("collective", "allreduce"),
                   ("row_binding", "real_q"),
                   ("payload_dtype", str(ctx.features.activation_dtype or "bf16")),
                   ("source_group", "tp")))
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


def _indexer_source_declaration(ctx: _Ctx) -> tuple[tuple[str, Any], ...] | None:
    """Derive the selected C8 caller's typed inputs from the normal deployment.

    The query representation comes from the model's index quant declaration;
    the index-cache representation comes from the cohort's independent
    ``attention_config.indexer_kv_dtype``.  Query quant is selected only when
    the deployed weight metadata carries ``self_attn.indexer.wq_b.weight_scale``.
    These are genuine source/deployment fields, not candidate count or the main
    attention KV dtype.
    """
    meta = dict(getattr(ctx.features, "metadata", ()) or ())
    q_dtype = str(getattr(ctx.features, "index_dtype", "") or "")
    if q_dtype == "fp8":
        q_dtype = "fp8_e4m3"
    k_dtype = str(meta.get("indexer_kv_dtype") or "")
    if k_dtype == "fp8":
        k_dtype = "fp8_e4m3"
    if q_dtype != "fp8_e4m3" or k_dtype != "fp8_e4m3":
        return None
    if meta.get("indexer_query_scale_present") is not True:
        return None
    scale_dtype = str(meta.get("indexer_query_scale_dtype") or "")
    if scale_dtype != "fp32":
        return None
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    if not image_id:
        return None
    return (
        ("source_family", INDEXER_SOURCE_FAMILY),
        ("image_id", image_id), ("q_dtype", q_dtype), ("k_dtype", k_dtype),
        ("query_scale_present", True), ("query_scale_dtype", "fp32"),
        ("layout_query", "TND"), ("layout_key", "PA_BSND"),
        ("query_quant_mode", 0), ("key_quant_mode", 0),
    )


def _gemma_k_eq_v(ctx: _Ctx) -> bool:
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
    shard_extra = ()
    if _dense_flashcomm(ctx) and int(ctx.tp) > 1:
        shard_extra = (("sequence_rows", "shard"), ("sequence_ranks", int(ctx.tp)))
    if role == "attn_norm" and int(ctx.layer.index) == 0:
        # Decoder entry: one plain RMSNorm before the first attention.
        pass
    elif role in {"attn_residual", "attn_norm", "ffn_norm"}:
        specs.append(("residual_add", "elementwise", "residual"))
        elements = ctx.hidden
        signatures["residual_add"] = _elementwise(
            "residual", ctx, elements=elements, ops=2,
            read_bytes=2 * elements * act_bytes, write_bytes=elements * act_bytes,
            extra=shard_extra,
        )
    if role != "attn_residual":
        specs.append(("rmsnorm", "elementwise", "norm"))
        elements = ctx.hidden
        facts = _gemma_facts(ctx)
        norm_extra = (("norm_eps", ctx.model.norm_eps),
                      ("layernorm", int(variant == "layernorm")),
                      ("norm_weight_offset", float(facts.get("norm_weight_offset", 0.0))),
                      ("norm_arch", "gemma" if facts.get("gemma4") else "rmsnorm")) + \
                     _norm_gamma(ctx, elements, role) + shard_extra
        # Normalization source semantics from the selected model source and
        # real quant description. A model type alone never supplies exact
        # R31 provenance; constructor intent and per-layer trained keys are
        # distinct. No image id or manifest hash selects the class/beta.
        _meta = dict(getattr(ctx.features, "metadata", ()) or ())
        _normal = _meta.get("normal_source") if isinstance(
            _meta.get("normal_source"), dict) else {}
        _class = str(_normal.get("source_class") or "")
        if (role in {"attn_norm", "ffn_norm"}
                and not (role == "attn_norm" and int(ctx.layer.index) == 0)):
            if _class == "AscendGemmaRMSNorm":
                norm_extra = norm_extra + (
                    ("rms_norm_source_class", "AscendGemmaRMSNorm"),
                )
            elif (_class == "AscendRMSNorm"
                  and _normal.get("source_weight_offset") == 0.0
                  and _normal.get("source_role") == "fused_residual"
                  and not _normal.get("source_conflict")):
                # The retained AscendRMSNorm.forward_oot residual branch
                # always calls npu_add_rms_norm_bias when custom ops are on,
                # even when optional beta is None. Constructor intent chooses
                # the beta state, not native eligibility.
                _role = ("input_layernorm" if role == "attn_norm"
                         else "post_attention_layernorm")
                _role_data = dict((_normal.get("role_bias") or {}).get(_role) or {})
                _key = (f"model.layers.{int(ctx.layer.index)}.{_role}.bias")
                _ctor = _normal.get("constructor_wants_bias")
                if _key in set(_role_data.get("loaded") or ()):
                    _bias_state = "present_trained"
                    _bias_value = "loaded"
                elif _ctor is False and _key in set(_role_data.get("absent") or ()):
                    _bias_state = "absent"
                    _bias_value = ""
                elif _ctor is True and _key in set(_role_data.get("absent") or ()):
                    _bias_state = "present_zero"
                    _bias_value = "zero"
                else:
                    _bias_state = "unresolved"
                    _bias_value = ""
                norm_extra = norm_extra + (
                    ("native_norm_kind", "AddRmsNormBias"),
                    ("native_norm_source", "AscendRMSNorm.forward_oot residual branch"),
                    ("norm_bias_state", _bias_state),
                    ("norm_bias_origin", "AscendRMSNorm residual forward_oot; optional beta"),
                    ("norm_bias_value", _bias_value),
                    ("norm_bias_trained", _bias_state == "present_trained"),
                    ("norm_bias_selector", "residual-present custom-op branch; optional beta absent/zero/trained"),
                    ("norm_has_weight", True),
                    ("norm_source_package_sha256", str(_normal.get("vllm_ascend_layernorm_sha256") or "")),
                    ("norm_source_weights_sha256", str(_normal.get("quant_manifest_sha256") or "")),
                )
                if _bias_state in {"present_zero", "present_trained"}:
                    norm_extra = norm_extra + (("norm_has_bias", True),)
                elif _bias_state == "absent":
                    norm_extra = norm_extra + (("norm_has_bias", False),)
        signatures["rmsnorm"] = _elementwise(
            "norm", ctx, elements=elements, ops=5,
            read_bytes=elements * act_bytes,
            write_bytes=elements * act_bytes,
            extra=norm_extra,
        )
    # Q/K normalization belongs to the attention producer after the projected
    # Q/K exist; N0 never emits a pre-projection copy.
    fusions = ()
    if role in {"attn_norm", "ffn_norm"}:
        members = tuple(name for name, _, _ in specs if name in {"residual_add", "rmsnorm", "activation_quant"})
        if {"residual_add", "rmsnorm"} <= set(members):
            source_bound = "source_lowerings" in dict(ctx.features.metadata)
            quant_norm = source_bound and source_lowering_selected(ctx, "add_rmsnorm_quant")
            fusions = (FusionContract(
                "add_rmsnorm_quant" if quant_norm or not source_bound else "add_rmsnorm",
                members, removes=("residual_add_output", "rmsnorm_output"),
                keeps=(("residual", "rstd", "scale", "norm_output")
                       if quant_norm or not source_bound else
                       ("residual", "rstd", "norm_output"))),)
    return _built("N0", ctx, variant, "Normalization and Residual Addition (归一化与残差相加)",
                  specs, signatures, fusions=fusions,
                  state_in=(_HIDDEN_IN, _RESIDUAL_IN), state_out=(_HIDDEN_OUT, _RESIDUAL_OUT))


def _k3_sequence_parallel(ctx: _Ctx) -> bool:
    return kimi_sequence_parallel(ctx.features.metadata, ctx.tp)


def _k3_gather(specs: list, signatures: dict, ctx: _Ctx, *, before: str) -> None:
    if not _k3_sequence_parallel(ctx):
        return
    name = "tp_all_gather"
    specs.insert(0, (name, "collective", "tp"))
    signatures[name] = _collective(
        "tp", ctx, axis="tp", members=ctx.tp,
        bytes_per_token=ctx.hidden * dtype_bytes(ctx.features.activation_dtype or "bf16"),
        extra=(("sequence_parallel_role", "allgather"), ("collective", "allgather")))


def _build_a0(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "mqa" if ctx.local_kv_heads == 1 and ctx.group_heads > 1
                       else "mha" if ctx.group_heads == 1 else "gqa")
    shared = ctx.shared_consumer
    specs = []
    signatures = {}
    k_eq_v = _gemma_k_eq_v(ctx)
    act_bytes = _result_bytes(ctx, "attn.qkv")
    _k3_gather(specs, signatures, ctx, before="qkv_projection")
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
    block = int(dict(ctx.features.metadata).get("kv_block_size", 0) or 0)
    signatures["attention"] = _attention_sig( "attention", ctx, local_heads=ctx.local_heads,
        local_kv_heads=ctx.local_kv_heads, group_heads=ctx.group_heads, width=ctx.width, value_width=ctx.width,
        kv_row_bytes=ctx.kv_row_bytes, extra=(("kv_owner", int(ctx.layer.kv_owner_layer_id)),
               ("shared_consumer", int(shared)), ("split_kv_target", ctx.features.split_target),
               ("max_kv_splits", ctx.features.max_splits), ("q_output_gate", int(ctx.layer.q_output_gate)),
               ("k_eq_v", int(k_eq_v)), ("local_window", int(ctx.layer.window)),
               ("page_attention", 1 if block > 0 else 0),
               ("page", block if block > 0 else 0))
        + ((("require_kv", int(ctx.layer.kv_owner_layer_id)),) if shared else ()))
    _dcp_merge(ctx, specs, signatures, local_heads=ctx.local_heads, d_v=ctx.width)
    _pcp_kv_gather(ctx, specs, signatures)
    if ctx.layer.q_output_gate:
        specs.append(("output_gate", "elementwise", "gate"))
        signatures["output_gate"] = _elementwise( "gate", ctx, elements=ctx.local_heads * ctx.width, ops=5,
            read_bytes=ctx.local_heads * ctx.width * act_bytes, write_bytes=ctx.local_heads * ctx.width * act_bytes)
    specs.append(("o_projection", "gemm", "output"))
    signatures["o_projection"] = _matrix( "output", ctx.local_heads * ctx.width, ctx.hidden, ctx, "attn.o")
    _precede(specs, signatures, ctx, "o_projection", ctx.local_heads * ctx.width, "attn.o")
    if shared:
        _precede(specs, signatures, ctx, "q_projection", ctx.hidden, "attn.q",
                 dynamic_only=True)
    else:
        _precede(specs, signatures, ctx, "qkv_projection", ctx.hidden, "attn.qkv",
                 dynamic_only=True)
    if ctx.tp > 1:
        if _dense_flashcomm(ctx):
            _dense_sp_block(specs, signatures, ctx)
        elif _k3_sequence_parallel(ctx):
            specs.append(("tp_reduce_scatter", "collective", "tp"))
            signatures["tp_reduce_scatter"] = _collective(
                "tp", ctx, axis="tp", members=ctx.tp,
                bytes_per_token=ctx.hidden * act_bytes,
                extra=(("sequence_parallel_role", "reduce_scatter"),
                       ("collective", "reduce_scatter"),
                       ("sp_rows", "padded")))
        else:
            specs.append(("tp_allreduce", "collective", "tp"))
            signatures["tp_allreduce"] = _collective( "tp", ctx, axis="tp", members=ctx.tp,
                bytes_per_token=ctx.hidden * act_bytes)
    fusions = ()
    contract = ()
    if ctx.layer.qk_norm:
        gate_keep = ("gate",) if ctx.layer.q_output_gate else ()
        # The projection is an independent QBMM.  Only qk_norm+rotary form
        # the selected composed split_qkv_rmsnorm_mrope postprocessor.
        fusions = (FusionContract( "qkv_norm_rope", ("qk_norm", "rotary"),
            removes=("qk_norm_output",),
            keeps=(("query", *gate_keep, "residual") if shared else
                  ("query", *gate_keep, "key", "value", "residual", "kv"))),)
        k_elems = 0 if shared else ctx.local_kv_heads * ctx.width
        v_elems = 0 if shared or k_eq_v else k_elems
        gate_elems = q_slice if ctx.layer.q_output_gate else 0
        total = q_slice + gate_elems + k_elems + v_elems
        contract = (("qkv_slices", f"{q_slice},{gate_elems},{k_elems},{v_elems},"
                                 f"{total},{act_bytes:g}"),)
    _dsa_cp_output_all_gather(ctx, specs, signatures)
    return _built("A0", ctx, variant, "Multi-Head / Grouped-Query / Multi-Query Attention (多头／分组查询／多查询注意力)",
                  specs, signatures,
                  fusions=fusions, state_in=(_HIDDEN_IN, _KV_IN),
                  state_out=(_HIDDEN_OUT, _KV_OUT, _KV_READY_OUT), contract=contract)


def _r64_image(metadata) -> bool:
    """The witnessed R64 image only; unqualified metadata stays unselected."""
    from mindie_sim.prediction.mx_source_provenance import (
        source_revision_from_oci)
    meta = dict(metadata or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    return source_revision_from_oci(
        str(meta.get("image") or meta.get("deployment_image")
            or meta.get("oci_image") or ""), image_id) == "r64"


def _r64_value_expansion(ctx: _Ctx, *, heads: int, k: int, n: int) -> Any:
    """The R64 SFA post-attention W_UV head product as one real launch.

    The SFA ``mla_attention`` row keeps its own latent-output charge; this row
    is the separate ``torch_npu.npu_transpose_batchmatmul`` launch that reads
    that latent output and writes the V-width result.  No attention cost is
    removed or duplicated here.
    """
    heads, k, n = int(heads), int(k), int(n)
    view = (
        ("kind", "transpose_batch_matmul"),
        ("batch", heads), ("n", n), ("k", k),
        ("input_shape", (heads, k)),
        ("weight_shape", (heads, k, n)),
        ("output_shape", (heads, n)),
        ("input_permutation", (1, 0, 2)),
        ("output_permutation", (1, 0, 2)),
        ("permutation_scope",
         "logical_view; physical_native_layout_not_selected"),
        ("role", "value_expansion"),
    )
    signature = _matrix(
        "value_expansion", k, n, ctx, "mla.kv_b",
        extra=(("batch", heads), ("matrix_groups", heads),
               ("native_tensor_view", view),
               ("native_kind", "TransposeBatchMatMul"),
               ("source_operation", "torch_npu.npu_transpose_batchmatmul"),
               ("source_lines", (1336, 1350)),
               ("deployment_source_revision", "r64")))
    weight = float(heads * k * n
                   * dtype_bytes(ctx.features.activation_dtype or "bf16"))
    quant = dict(signature.quant)
    quant.update({"weight_total": weight, "weight_payload": weight,
                  "weight_scale": 0.0, "weight_zero": 0.0,
                  "residency": weight})
    return replace(signature, quant=tuple(sorted(quant.items())))


def _build_a1(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "latent_mqa" if ctx.layer.mla_variant == "latent_mqa" else "latent_mla")
    mla = ctx.model.mla
    r_q = int(ctx.layer.q_lora_rank or (mla.q_lora_rank if mla else 0))
    r_kv = int(ctx.layer.kv_lora_rank or (mla.kv_lora_rank if mla else 0))
    d_n = int(ctx.layer.qk_nope_head_dim or (mla.qk_nope_head_dim if mla else ctx.width))
    d_r = int(ctx.layer.qk_rope_head_dim or (mla.qk_rope_head_dim if mla else 0))
    d_v = int(ctx.layer.v_head_dim or (mla.v_head_dim if mla else ctx.width))
    act_bytes = dtype_bytes(ctx.features.activation_dtype or "bf16")
    # The source-selected SFA C8 MX prolog publishes ql_nope/q_pe separately;
    # the A5 caller then concatenates them into the SFA query.  The generic MLA
    # prolog branch keeps one fused `query`, so it must not get a second Cat.
    sfa_selected = sfa_prolog_v3_c8_mx(
        ctx.model, ctx.layer, ctx.features, q_lora=r_q, kv_lora=r_kv,
        rope=d_r)
    r64_primary = mla_prolog_v3_r64_primary(
        ctx.model, ctx.layer, ctx.features, q_lora=r_q, kv_lora=r_kv,
        rope=d_r, nope=d_n)
    specs = []
    signatures = {}
    _k3_gather(specs, signatures, ctx, before="q_down")
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
    if sfa_selected and r64_primary:
        # R64 A5DeviceAdaptor._execute_kv_quant_sparse_flash_attention_vllm
        # L1157: torch.cat([ql_nope, q_pe], dim=-1).  The fused mla_prolog
        # publishes ql_nope/q_pe; this is a separate kernel and its output is
        # the SFA `query` input, so it owns its own copy traffic between them.
        specs.append(("q_concat", "copy", "concat"))
        signatures["q_concat"] = _copy(
            "concat", ctx, elements=ctx.local_heads * (r_kv + d_r),
            bytes_per_element=act_bytes,
            extra=(("q_concat_heads", ctx.local_heads),
                   ("q_concat_latent", r_kv),
                   ("q_concat_rope", d_r),
                   ("concat_role", "mla_q_latent_plus_rope"),
                   ("producer", "fused:mla_prolog"),
                   ("consumer", "mla_attention"),
                   ("readiness", "producer_complete"),
                   ("source_operation", "torch.cat(dim=-1)"),
                   ("source_lines", (1157, 1157)),
                   ("native_kind", "aclnnCat"),
                   ("deployment_source_revision", "r64")))
    specs.append(("mla_attention", "mla", "attention"))
    plans, idx_raw, sel_raw = ctx.model.layer_plans, int(ctx.layer.index_k_owner_layer_id), int(ctx.layer.topk_owner_layer_id)
    # Cache ownership and top-k selection ownership are independent roles.
    # A shared indexer consumes an earlier owner's cache but produces its own
    # selection; a non-indexer sibling reuses both declared owners.
    idx_owner = idx_raw if 0 <= idx_raw < ctx.layer.index else ctx.layer.index
    sel_owner = sel_raw if 0 <= sel_raw < ctx.layer.index else ctx.layer.index
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
    _pcp_kv_gather(ctx, specs, signatures)
    # OPCAT2 owns the pre-attention q_concat.  This is the disjoint
    # post-attention W_UV head product selected by the R64 SFA source; the
    # SFA row keeps its own latent-output charge, so no V projection is
    # duplicated.
    if (variant != "latent_mqa" and sfa_selected
            and _r64_image(ctx.features.metadata)
            and ctx.local_heads > 0 and r_kv > 0 and d_v > 0):
        specs.append(("v_expand", "gemm", "value_expansion"))
        signatures["v_expand"] = _r64_value_expansion(
            ctx, heads=ctx.local_heads, k=r_kv, n=d_v)
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
        if _k3_sequence_parallel(ctx):
            specs.append(("tp_reduce_scatter", "collective", "tp"))
            signatures["tp_reduce_scatter"] = _collective(
                "tp", ctx, axis="tp", members=ctx.tp,
                bytes_per_token=ctx.hidden * act_bytes,
                extra=(("sequence_parallel_role", "reduce_scatter"),
                       ("collective", "reduce_scatter"),
                       ("sp_rows", "padded")))
        else:
            specs.append(("tp_allreduce", "collective", "tp"))
            signatures["tp_allreduce"] = _collective( "tp", ctx, axis="tp", members=ctx.tp,
                bytes_per_token=ctx.hidden * act_bytes)
    prolog_members = tuple( name for name, _, _ in specs
        if name in {"q_down", "q_norm", "q_up", "q_direct", "kv_down", "kv_norm", "rotary", "kv_store"})
    removes = tuple( name for name in ("q_lora_activation", "q_norm_output", "kv_down_output",
                          "kv_norm_output", "rotary_output")
        if r_q > 0 or name not in {"q_lora_activation", "q_norm_output"})
    fusions = ()
    contract = ()
    # Same lora/head shape stays on the generic keeps. Only the pinned
    # SFA C8 MX caller publishes ql_nope, q_pe, q_c, and the index slot.
    if ({"kv_down", "kv_norm", "rotary", "q_up", "q_norm", "kv_store"} <= set(prolog_members)
            and sfa_selected):
        store_row = int((r_kv + d_r) * dtype_bytes(ctx.kv_dtype))
        # q_activation is not consumed by the next member (kv_down). The
        # lowering drops that write only; putting it in removes would also
        # subtract kv_down's own weight and activation read.
        fusions = (FusionContract(
            "mla_prolog", prolog_members, removes=removes,
            keeps=("ql_nope", "q_pe", "q_c", "q_c_scale", "cache_index")),)
        wuk_heads = int(ctx.heads)
        native_contract = (("mla_prolog_v3",
                            f"r64:split_n:W_UK={wuk_heads},{d_n},{r_kv}:query_norm={int(bool(ctx.layer.indexer))}"),)
        primary = r64_primary
        contract = (("sfa_prolog_v3", _sfa_port_spec(
            heads=ctx.local_heads, q_lora=r_q, kv_lora=r_kv, rope=d_r,
            store_row=store_row)),) + (native_contract if primary else ())
    elif {"kv_down", "kv_norm", "rotary"} <= set(prolog_members):
        native_r64 = r64_primary
        fusions = (FusionContract( "mla_prolog", prolog_members, removes=removes,
            keeps=("query", "key", "value", "kv_cache", "mla_attention_output")),)
        if native_r64:
            contract = (("mla_prolog_v3",
                         f"r64:split_n:W_UK={ctx.local_heads},{d_n},{r_kv}:query_norm={int(bool(ctx.layer.indexer))}"),)
    a1_in = (_HIDDEN_IN, _KV_IN) + ((_in("index_selection", "selection"),) if mla_topk > 0 else ())
    _dsa_cp_output_all_gather(ctx, specs, signatures)
    return _built("A1", ctx, variant, "Multi-head Latent Attention (MLA 多头潜在注意力)",
                  specs, signatures,
                  fusions=fusions, state_in=a1_in, state_out=(_HIDDEN_OUT, _KV_OUT, _KV_READY_OUT),
                  contract=contract)


def _layer_compressed_source(ctx: _Ctx) -> bool:
    """Whether the layer plan actually provides a compressed KV branch.

    The layer plan is the source.  A positive ratio alone is not proof of a
    compressor: SWA-only/shared layers declare ``compressor_kind=none`` and no
    long-KV owner, and must keep the window branch only.  A positive ratio with
    an incomplete source is refused explicitly rather than synthesized.
    """
    layer = ctx.layer
    ratio = int(getattr(layer, "compress_ratio", 0) or 0)
    if ratio <= 0:
        return False
    kind = str(getattr(layer, "compressor_kind", "") or "").strip().lower()
    if kind in {"", "none"}:
        raise ModelingError(
            f"a2_compressed_source_missing:layer={layer.index}:ratio={ratio}")
    raw_owner = getattr(layer, "kv_owner_layer_id", -1)
    owner = -1 if raw_owner is None else int(raw_owner)
    if not (0 <= owner < layer.index or bool(getattr(layer, "owns_long_kv", False))):
        raise ModelingError(
            f"a2_compressed_kv_owner_missing:layer={layer.index}:ratio={ratio}")
    return True


def _build_a2(ctx: _Ctx) -> _Built:
    compressed = _layer_compressed_source(ctx)
    variant = _variant(ctx, "window_compressed" if compressed else "window_only")
    title = ("Sliding-Window Attention with Compressed KV Attention (滑动窗口注意力与压缩 KV 注意力)"
             if compressed else "Sliding-Window Attention (滑动窗口注意力)")
    act_bytes = dtype_bytes(ctx.features.activation_format)
    owner = int(ctx.layer.kv_owner_layer_id)
    owner = owner if 0 <= owner < ctx.layer.index else ctx.layer.index
    index_raw = int(ctx.layer.index_k_owner_layer_id)
    index_owner = index_raw if 0 <= index_raw < ctx.layer.index else ctx.layer.index
    sel_raw = int(ctx.layer.topk_owner_layer_id)
    sel_owner = sel_raw if 0 <= sel_raw < ctx.layer.index else ctx.layer.index
    q_width = ctx.local_heads * ctx.width * (2 if ctx.layer.q_output_gate else 1)
    index_topk = int(ctx.layer.index_topk or (ctx.model.mla.index_topk if ctx.model.mla else 0))
    selection_required = (index_topk > 0 and
                          (sel_raw >= 0 or bool(ctx.layer.owns_topk) or bool(ctx.layer.indexer)))
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
        return _built("A2", ctx, variant, title,
                      specs, signatures, coverage="structural",
                      state_in=(_HIDDEN_IN,), state_out=(_KV_OUT, _KV_READY_OUT))
    specs = [("kv_read", "kv_access", "consume"), ("window_attention", "attention", "window")]
    signatures = { "kv_read": _state("kv_read", ctx, rows=1, row_bytes=0.0, extra=(("owner_layer", owner),)),
        "window_attention": _attention_sig( "window", ctx, local_heads=ctx.local_heads,
            local_kv_heads=ctx.local_kv_heads, group_heads=ctx.group_heads,
            width=ctx.width, value_width=ctx.width, kv_row_bytes=ctx.kv_row_bytes,
            extra=(("branch", "window"), ("require_kv", owner))) }
    if compressed:
        specs.extend((("compressed_attention", "attention", "compressed"),
                      ("merge", "elementwise", "merge")))
        signatures["compressed_attention"] = _attention_sig(
            "compressed", ctx, local_heads=ctx.local_heads,
            local_kv_heads=ctx.local_kv_heads, group_heads=ctx.group_heads,
            width=ctx.width, value_width=ctx.width, kv_row_bytes=ctx.kv_row_bytes, extra=(("branch", "compressed"),
                   ("compress_ratio", int(ctx.layer.compress_ratio)),
                   ("index_topk", index_topk),
                   ("require_kv", owner), ("compression_owner", owner),
                   ("require_compression", int(ctx.layer.compress_ratio))) +
                  ((("require_index", index_owner),) if ctx.layer.indexer else ()) +
                  ((("require_selection", sel_owner),) if selection_required else ()))
        signatures["merge"] = _elementwise( "merge", ctx, elements=ctx.local_heads * ctx.width, ops=5,
            read_bytes=2 * ctx.local_heads * ctx.width * act_bytes,
            write_bytes=ctx.local_heads * ctx.width * act_bytes)
    _dcp_merge(ctx, specs, signatures, local_heads=ctx.local_heads, d_v=ctx.width)
    _pcp_kv_gather(ctx, specs, signatures)
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
    _dsa_cp_output_all_gather(ctx, specs, signatures)
    return _built("A2", ctx, variant, title,
                  specs, signatures,
                  children=(("K1",) if compressed else ()) +
                           (("I0", "I1") if ctx.layer.indexer else ()),
                  coverage="partial",
                  state_in=((_HIDDEN_IN, _in("compressed_kv", "kv_cache"),
                             _in("index_selection", "selection"))
                            if compressed else (_HIDDEN_IN, _KV_IN)),
                  state_out=(_HIDDEN_OUT,),
                  streams={"compressed_attention": "aux"}
                  if compressed and ctx.features.multistream_window_compressed else None)

_R31_STATE_IO_IMAGE_PREFIX = "9e869e81484"


def _r31_prefill_state_source(meta: Mapping[str, Any]) -> bool:
    """The exact R31 deployment image, not the older ``source_image`` tag."""
    for key in ("image_id", "deployment_image_id", "oci_image_digest",
                "image", "deployment_image", "oci_image"):
        if _R31_STATE_IO_IMAGE_PREFIX in str(meta.get(key) or "").lower():
            return True
    return False


def _fla_gdn_prefill_hardware(meta: Mapping[str, Any]) -> bool:
    """FLA_GDN_PREFILL hardware contract from the R31 hardware profile.

    ``vllm_ascend/device/hardware_profile.py`` declares FLA_GDN_PREFILL in
    the standard capability set (A2/A3) and in the A5 profile; the 310P
    compatibility profile has no FLA_GDN_PREFILL entry.
    """
    arch = str(meta.get("arch_family") or "").upper()
    if arch in {"A2", "A3", "A5"}:
        return True
    if arch:
        return False
    sku = " ".join(str(meta.get(key) or "") for key in (
        "hardware_sku", "hardware_key", "hardware_name", "device")).lower()
    if not sku or "310" in sku:
        return False
    return any(token in sku for token in ("950", "910", "a2", "a3", "a5"))


def _gdn_fla_state_io(ctx: _Ctx, *, value_heads: int,
                      d_k: int, d_v: int) -> dict[str, Any] | None:
    """The source FLA prefill gather/scatter, when the source condition holds.

    ``gdn.py`` requires PCP 1, an FLA_GDN_PREFILL-capable device and the
    ``_get_fla_gdn_prefill_op()`` operator.  The exact R31 image or a
    declared ``gdn_prefill`` lowering confirms the branch; a capable device
    without either keeps an explicitly conditional structural cost and no
    native claim.  ``gdn_decode`` in the lowering table is a decode grammar
    declaration, not evidence that the prefill FLA branch is absent.
    """
    if str(getattr(ctx.layer, "attention", "")) != "gdn":
        return None
    if int(ctx.features.pcp) != 1:
        return None
    meta = dict(getattr(ctx.features, "metadata", ()) or ())
    if not _fla_gdn_prefill_hardware(meta):
        return None
    declared = meta.get("source_lowerings")
    declared_prefill = declared is not None and "gdn_prefill" in {
        str(value) for value in declared}
    if _r31_prefill_state_source(meta):
        condition, branch = "confirmed_r31_fla", "fla_gdn_prefill"
    elif declared_prefill:
        condition, branch = "declared_gdn_prefill", "fla_gdn_prefill"
    else:
        condition, branch = "conditional_hardware_fla", "conditional"
    state_dtype = str(
        getattr(ctx.layer, "recurrent_state_dtype", "") or "").lower()
    if state_dtype not in {"fp32", "float32"}:
        return None
    if value_heads < 1 or d_k < 1 or d_v < 1:
        return None
    return {
        "state_dtype": "fp32",
        "row_bytes": float(value_heads * d_v * d_k * dtype_bytes("fp32")),
        "condition": condition,
        "branch": branch,
    }


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
    result_bytes = _result_bytes(ctx, "gdn")
    qkv_width = 2 * key_heads * d_k + value_heads * d_v
    views = () if str(variant) == "chunked_prefill" else decode_views(
        key_heads, value_heads, d_k, d_v)
    early = tuple(item for item in views if item.placement == "early")
    late = tuple(item for item in views if item.placement == "late")
    gate = tuple(item for item in views if item.placement == "gate")
    specs = [("qkvz_projection", "gemm", "projection"), ("ba_projection", "gemm", "projection")]
    specs.extend((item.name, "copy", item.role) for item in early)
    state_io = _gdn_fla_state_io(ctx, value_heads=value_heads,
                                 d_k=d_k, d_v=d_v)
    specs.extend((("gdn_gating", "recurrent", "gating"),
                  ("conv", "recurrent", "conv")))
    if state_io is not None:
        specs.append(("gather_recurrent_state", "state_gather", "gather"))
    specs.append(("prepare", "elementwise", "prepare"))
    specs.extend((item.name, "copy", item.role) for item in late)
    if str(variant) != "chunked_prefill":
        specs.extend((("q_l2norm", "elementwise", "norm"),
                      ("k_l2norm", "elementwise", "norm")))
    specs.extend((("gated_delta", "recurrent", "delta"),))
    if state_io is not None:
        specs.append(("scatter_recurrent_state", "state_scatter", "scatter"))
    specs.extend((item.name, "copy", item.role) for item in gate)
    specs.extend((("gated_norm", "elementwise", "norm"),
                  ("o_projection", "gemm", "output")))
    act = ("act_bytes", act_bytes)
    signatures = { "qkvz_projection": _matrix("qkvz", ctx.hidden, key_heads * d_k * 2 + value_heads * d_v * 2,
                                   ctx, "gdn", extra=((("key_heads", key_heads), ("value_heads", value_heads), act))),
        "ba_projection": _matrix("ba", ctx.hidden, value_heads * 2, ctx, "gdn.beta_alpha", extra=(act,)),
        "gdn_gating": _state("gating", ctx, rows=1, row_bytes=value_heads * 6,
                              extra=(("value_heads", value_heads),
                                     ("source_kernel", "fused_gdn_gating_kernel"),
                                     ("gate_outputs", ("fp32_decay", "bf16_beta")),
                                     ("act_bytes", act_bytes))),
        "conv": _state("conv", ctx, rows=conv, row_bytes=qkv_width * result_bytes,
                       extra=(("kernel", conv), ("history_rows", conv - 1), ("qkv_width", qkv_width),
                              ("act_bytes", result_bytes))),
        "prepare": _elementwise("prepare", ctx, elements=qkv_width, ops=4,
                                read_bytes=qkv_width * result_bytes,
                                write_bytes=qkv_width * result_bytes,
                                extra=(("act_bytes", result_bytes),)),
        "gated_delta": _state("delta", ctx, rows=1, row_bytes=float(ctx.layer.state_bytes_total) / tp,
                              extra=(("chunk", min(64, max(1, int(ctx.layer.linear_head_dim or 64)))), ("key_heads", key_heads),
                                     ("value_heads", value_heads), ("dk", d_k), ("dv", d_v),
                                     ("act_bytes", result_bytes))),
        "gated_norm": _elementwise("norm", ctx, elements=value_heads * d_v, ops=5,
                                   read_bytes=2 * value_heads * d_v * result_bytes,
                                   write_bytes=value_heads * d_v * result_bytes,
                                   extra=(("act_bytes", result_bytes),
                                          ("value_heads", value_heads),
                                          ("dv", d_v), ("head_dim", d_v),
                                          ("result_bytes", result_bytes),
                                          ("output_dtype", str(
                                              ctx.features.activation_dtype
                                              or ctx.model.activation_dtype
                                              or "")),
                                          ("compute_dtype", str(
                                              ctx.features.activation_dtype
                                              or ctx.model.activation_dtype
                                              or "")))),
        "o_projection": _matrix("output", value_heads * d_v, ctx.hidden, ctx, "gdn.o", extra=(act,)), }
    if state_io is not None:
        state_extra = (
            ("heads", value_heads), ("value_heads", value_heads),
            ("dk", d_k), ("dv", d_v), ("state_layout", "V,K"),
            ("state_dtype", state_io["state_dtype"]),
            ("index_dtype", "int32"),
            ("state_io_condition", state_io["condition"]),
        )
        gather_extra = state_extra + (
            ("gather_branch", state_io["branch"]),
            ("source_kernel", "_gather_ssm_states_kernel"))
        scatter_extra = state_extra + (
            ("scatter_branch", state_io["branch"]),
            ("source_kernel", "_scatter_ssm_states_kernel"))
        if state_io["branch"] == "fla_gdn_prefill":
            scatter_extra += (("source_dtype", state_io["state_dtype"]),)
        signatures["gather_recurrent_state"] = _state(
            "gather", ctx, rows=1, row_bytes=state_io["row_bytes"],
            extra=gather_extra)
        signatures["scatter_recurrent_state"] = _state(
            "scatter", ctx, rows=1, row_bytes=state_io["row_bytes"],
            extra=scatter_extra)
    if str(variant) != "chunked_prefill":
        signatures["q_l2norm"] = _elementwise(
            "q_l2norm", ctx, elements=key_heads * d_k, ops=5,
            read_bytes=key_heads * d_k * 2,
            write_bytes=key_heads * d_k * 2,
            extra=(("source_kernel", "l2norm_fwd_kernel2"),
                   ("owner_axis", "sequence_key_head"), ("decode_only", 1), act))
        signatures["k_l2norm"] = _elementwise(
            "k_l2norm", ctx, elements=key_heads * d_k, ops=5,
            read_bytes=key_heads * d_k * 2,
            write_bytes=key_heads * d_k * 2,
            extra=(("source_kernel", "l2norm_fwd_kernel2"),
                   ("owner_axis", "sequence_key_head"), ("decode_only", 1), act))
    # Captured Slice and ZerosLike rows are DT_BF16. The W8 activation width
    # belongs to the quantizer and is not the element width of these buffers.
    copy_bytes = dtype_bytes("bf16")
    for item in views:
        # Kept extent only. The parent GEMM keeps the unpacked row.
        # row_source is the v0.30.0 one-row view versus multi-row copy rule.
        signatures[item.name] = _elementwise(
            item.role, ctx, elements=item.elements,
            ops=1 if item.transfer == "fill" else 0,
            read_bytes=item.read_elements * copy_bytes,
            write_bytes=item.write_elements * copy_bytes,
            extra=(("source_kernel", item.source_kernel),
                   ("owner_axis", "sequence"),
                   ("row_source", item.row_source),
                   ("transfer", item.transfer),
                   ("execution", item.execution),
                   ("rank", item.rank),
                   ("source_extent", item.source_extent),
                   ("offset_elements", item.offset_elements),
                   ("head_dim", item.head_dim),
                   ("source_heads", item.source_heads),
                   ("kept_heads", item.kept_heads)))
    _precede(specs, signatures, ctx, "qkvz_projection", ctx.hidden, "gdn",
             dynamic_only=True)
    _precede(specs, signatures, ctx, "o_projection", value_heads * d_v, "gdn.o",
             dynamic_only=True)
    if ctx.tp > 1:
        if _dense_flashcomm(ctx):
            _dense_sp_block(specs, signatures, ctx)
        else:
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
        if _k3_sequence_parallel(ctx):
            specs.insert(0, ("tp_all_gather", "collective", "tp"))
            signatures["tp_all_gather"] = _collective(
                "tp", ctx, axis="tp", members=ctx.tp,
                bytes_per_token=ctx.hidden * act_bytes,
                extra=(("sequence_parallel_role", "allgather"),
                       ("collective", "allgather")))
            specs.append(("tp_reduce_scatter", "collective", "tp"))
            signatures["tp_reduce_scatter"] = _collective(
                "tp", ctx, axis="tp", members=ctx.tp,
                bytes_per_token=ctx.hidden * act_bytes,
                extra=(("sequence_parallel_role", "reduce_scatter"),
                       ("collective", "reduce_scatter"),
                       ("sp_rows", "padded")))
        else:
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
    if not _layer_compressed_source(ctx):
        # SWA-only/shared layer: the K1 slot is a boundary, not four kernel
        # floors.  The canonical ``compress`` name is retained for the model
        # state-advance caller; zero row bytes keeps the marker at exactly
        # zero work and it is never a compressed-attention launch.
        signature = _state("compress", ctx, rows=1, row_bytes=0.0,
                           extra=(("ratio", 0), ("source", "window_only")))
        return _built(
            "K1", ctx, "none",
            "KV Cache Compression and Key Pooling (no source compressor)",
            (("compress", "kv_access", "compress"),),
            {"compress": signature}, coverage="partial",
            state_in=(_KV_IN,),
            state_out=(_out("compressed_kv", "kv_cache"),
                       _out("compress_state", "compress")))
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


def _launch_extra(launch) -> tuple[tuple[str, Any], ...]:
    extra = [("category", launch.category), ("projection_packing", launch.packing)]
    if launch.key_end:
        extra.append(("key_slice_end", int(launch.key_end)))
    if launch.weights_end:
        extra.extend((("weights_slice_begin", int(launch.weights_begin)),
                      ("weights_slice_end", int(launch.weights_end))))
    return tuple(extra)


def _build_i0(ctx: _Ctx) -> _Built:
    from mindie_sim.prediction.piecewise.families_recurrent_indexer import (
        cann_keypool_selected, index_pool_mode,
    )
    variant = _variant(ctx, str(ctx.layer.indexer_type or "full"))
    heads = int(ctx.layer.index_heads or 1)
    dim = int(ctx.layer.index_dim or 64)
    rope_dim = int(getattr(ctx.layer, "qk_rope_head_dim", 0) or 0)
    index_meta = dict(ctx.features.metadata or ())
    r64_sfa_indexer = (
        str(index_meta.get("image_id") or "").split(":")[-1].lower()
        == "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e")
    rope_dtype = str(ctx.features.activation_dtype or "")
    # Actual SFA indexer callable facts: the Triton SISO rope geometry/style
    # and dtype are read from this layer/config, never from the enrolled
    # 32/128/64 prefill sample.  The LayerNorm implementation used by both GLM
    # source variants is vllm.model_executor.layers.layernorm.LayerNorm with
    # explicit FP32 weight/bias and x.float() forward.
    rope_extra = (
        ("rope_implementation", "triton_siso"),
        ("index_dim", dim),
        ("qk_rope_head_dim", rope_dim),
        ("is_neox_style", int(str(index_meta.get("model_identifier") or "")
                              != "glm_moe_dsa")),
        ("rope_dtype", rope_dtype),
        ("index_query_heads", heads),
        ("index_key_heads", 1),
    ) if r64_sfa_indexer and rope_dtype else ()
    model_identifier = str(index_meta.get("model_identifier") or "")
    if model_identifier == "glm_moe_dsa":
        norm_source = {
            "path": "vllm/model_executor/models/deepseek_v2.py",
            "sha256": "02e72b106da8b106b9fb0adf5d22cf1cf9713d7b0f4e002c02eeee6e35ffc305",
            "symbol": "Indexer.__init__",
            "lines": [685, 685],
        }
        norm_call_site = {
            "path": "vllm_ascend/attention/indexer.py",
            "sha256": "99b7e507c2ba6f8f47d04b051e647f51e6b18b23ecadfc839cc351d81b785942",
            "symbol": "AscendSFAIndexerBackend.forward_k",
            "lines": [335, 335],
        }
    elif model_identifier in {"glm5_next", "glm5next"}:
        norm_source = {
            "path": "vllm_ascend/models/glm5next/attention.py",
            "sha256": "f0e1dfa46eea4c6385e10b86b73fa024121cc18e73ae9148dd78d769cfdf8e01",
            "symbol": "Indexer.__init__",
            "lines": [96, 96],
        }
        norm_call_site = {
            "path": "vllm_ascend/attention/indexer_kpool.py",
            "sha256": "786ccb6864ef9299b8b44b61afd446b35b281dc2ca4c4f3d74ae8b7edb3cb57c",
            "symbol": "Glm5NextKPoolIndexerBackend.forward",
            "lines": [763, 770],
        }
    else:
        norm_source = None
        norm_call_site = None
    layer_norm_source = {
        "path": "vllm/model_executor/layers/layernorm.py",
        "sha256": "696615d858292fea7a2942480803cd4c7d2d1e88d818bfe5cd9d695ac464f37c",
        "symbol": "LayerNorm",
        "lines": [305, 320],
        "dtype_source": "weight/bias float32; forward x.float()",
    }
    norm_extra = (
        ("source_kind", "nn.LayerNorm"),
        ("source_path", norm_source["path"]),
        ("source_symbol", norm_source["symbol"]),
        ("source_lines", list(norm_source["lines"])),
        ("source_sha256", norm_source["sha256"]),
        ("call_site_path", norm_call_site["path"]),
        ("call_site_sha256", norm_call_site["sha256"]),
        ("call_site_symbol", norm_call_site["symbol"]),
        ("call_site_lines", list(norm_call_site["lines"])),
        ("source_stage", "I0"),
        ("index_dim", dim),
        ("normalized_shape", [dim]),
        ("norm_eps", 1e-6),
        ("norm_dtype", "fp32"),
        ("layer_norm_source_path", layer_norm_source["path"]),
        ("layer_norm_source_sha256", layer_norm_source["sha256"]),
        ("layer_norm_source_symbol", layer_norm_source["symbol"]),
        ("layer_norm_source_lines", list(layer_norm_source["lines"])),
        ("layer_norm_dtype_source", layer_norm_source["dtype_source"]),
        ("attribution_owner", "I0:indexer:index_k_norm"),
    ) if r64_sfa_indexer and norm_source and norm_call_site else ()
    kpool = max(1, int(ctx.layer.index_kpool or 1))
    mode = index_pool_mode(
        ctx.layer, ctx.quant, hidden=ctx.hidden,
        activation_dtype=ctx.features.activation_dtype or "bf16",
        cann_caller=cann_keypool_selected(ctx.model.metadata, ctx.features.metadata))
    cache_decl = dict((dict(ctx.features.metadata or ()).get("indexer_cache") or {}))
    cache_declared = bool(cache_decl)
    cache_row_bytes = float(cache_decl.get(
        "row_bytes", dim * dtype_bytes(ctx.features.index_dtype)))
    # A declared cache carries an explicit per-execution valid-slot count.
    # A packed declaration without a count stays unresolved (zero chosen
    # work, layout preserved); it is never promoted to all-valid.
    declared_rows = cache_decl.get("rows") if cache_declared else None
    if declared_rows is None:
        cache_rows = 1 if not cache_declared else 0
        valid_rows_mode = "dynamic" if not cache_declared else "unresolved"
    else:
        cache_rows = int(declared_rows)
        valid_rows_mode = "explicit"
    raw_owner = getattr(ctx.layer, "index_k_owner_layer_id", None)
    index_owner = int(ctx.layer.index) if raw_owner is None else int(raw_owner)
    if index_owner < 0:
        index_owner = int(ctx.layer.index)
    store = _state("index_store", ctx, rows=cache_rows,
                   row_bytes=cache_row_bytes,
                   extra=(("heads", heads), ("dim", dim), ("kpool", kpool),
                          ("owner_layer", index_owner),
                          ("cache_format", str(cache_decl.get("format", "index_dtype"))),
                          ("scale_bytes", float(cache_decl.get("scale_bytes", 0))),
                          ("valid_rows_mode", valid_rows_mode),
                          ("state_effects", (("index", "index_store"),))))
    quant_row = _elementwise("quant", ctx, elements=heads * dim, ops=2,
                             read_bytes=heads * dim * 2, write_bytes=heads * dim)
    if mode.kind != "analytical" or mode.declaration is None:
        owns_index_k = bool(getattr(ctx.layer, "owns_index_k", True))
        specs = [("index_q_projection", "gemm", "projection")]
        if owns_index_k:
            specs.append(("index_k_projection", "gemm", "projection"))
        specs.append(("index_weights_projection", "gemm", "projection"))
        if owns_index_k and norm_extra:
            specs.append(("index_k_norm", "elementwise", "norm"))
        if owns_index_k and rope_extra:
            specs.append(("index_k_rope", "elementwise", "rope"))
        if owns_index_k:
            specs.append(("index_cache_store", "kv_access", "produce"))
        specs.append(("index_rope", "elementwise", "rope"))
        specs.append(("index_quant", "elementwise", "quant"))
        # The non-pooled indexer producer applies wq_b to the MLA low-rank
        # query (q_lora_rank), not to the full hidden state.  The ordinary
        # graph previously charged K=hidden and inflated the weight payload
        # by hidden/q_lora.  Architectures without a low-rank query keep the
        # hidden input.
        query_rank = int(getattr(ctx.layer, "q_lora_rank", 0) or 0)
        query_k = query_rank or int(ctx.hidden)
        index_k_width = (int(ctx.layer.head_dim)
                         if str(getattr(ctx.layer, "index_input", "hidden")) == "compressed"
                         else int(ctx.hidden))
        signatures = {
            "index_q_projection": _matrix(
                "index_q", query_k, heads * dim, ctx, "indexer",
                extra=(("query_k_source", "q_lora_rank" if query_rank > 0
                        else "hidden_rank_absent"),)),
            "index_weights_projection": _matrix(
                "index_weights", ctx.hidden, heads, ctx, "indexer.weights",
                extra=(("projection_packing", "source_individual_linear"),
                       ("source_role", "indexer_weights_proj"))),
            "index_rope": _elementwise(
                "rope", ctx, elements=heads * dim, ops=6,
                read_bytes=heads * dim * dtype_bytes(ctx.features.activation_format),
                write_bytes=heads * dim * dtype_bytes(ctx.features.activation_format),
                extra=((("index_rope_role", "query"), ("index_heads", heads),
                        *rope_extra) if rope_extra else ())),
            "index_quant": quant_row,
        }
        if owns_index_k:
            signatures["index_cache_store"] = store
        if owns_index_k and norm_extra:
            signatures["index_k_norm"] = _elementwise(
                "index_k_norm", ctx, elements=dim, ops=5,
                read_bytes=dim * dtype_bytes("fp32"),
                write_bytes=dim * dtype_bytes("fp32") + 8,
                extra=norm_extra)
        if owns_index_k and rope_extra:
            signatures["index_k_rope"] = _elementwise(
                "rope_k", ctx, elements=dim, ops=6,
                read_bytes=dim * dtype_bytes(ctx.features.activation_format),
                write_bytes=dim * dtype_bytes(ctx.features.activation_format),
                extra=(("index_rope_role", "key"), ("index_heads", 1),
                       *rope_extra))
        if owns_index_k:
            # Native compressed-index providers project the compressor's
            # latent (head_dim), not the hidden state; only the owning layer
            # emits the K projection.  Non-owners reuse the owner's cache.
            signatures["index_k_projection"] = _matrix(
                "index_k", index_k_width, dim, ctx, "indexer.pool_projection",
                extra=(("projection_packing", "source_individual_linear"),
                       ("source_role", "indexer_wk"),
                       ("index_k_source", "compressed_latent" if index_k_width != int(ctx.hidden)
                        else "hidden_rank")))
    else:
        decl = mode.declaration
        projected = replace(ctx, features=replace(ctx.features, quant=decl.quant))
        specs = []
        signatures = {}
        if decl.cast_elements:
            specs.append(("index_pool_hidden_cast", "elementwise", "cast"))
            signatures["index_pool_hidden_cast"] = _elementwise(
                "cast", projected, elements=decl.cast_elements, ops=1,
                read_bytes=decl.cast_read_per_token,
                write_bytes=decl.cast_write_per_token,
                extra=(("input_format", decl.input_format), ("output_format", "fp32"),
                       ("category", "indexer.pool_projection")))
        specs.append(("index_q_projection", "gemm", "projection"))
        signatures["index_q_projection"] = _matrix(
            "index_q", decl.query_k, heads * dim, projected, decl.query_category,
            extra=(("category", decl.query_category),
                   ("k_source", decl.query_k_source),
                   ("projection_packing", "query_independent")))
        for launch in decl.launches:
            specs.append((launch.name, "gemm", "projection"))
            signatures[launch.name] = _matrix(
                launch.name, launch.k, launch.n, projected, launch.category,
                extra=_launch_extra(launch))
        specs.extend((("index_quant", "elementwise", "quant"),
                      ("index_cache_store", "kv_access", "produce")))
        signatures["index_quant"] = quant_row
        signatures["index_cache_store"] = store
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
    # The index cache owner and the top-k selection owner are separate source
    # roles.  A shared later indexer (e.g. V4.1 layers 24/28/32/36) consumes the
    # cache written by owner 20 while owning its own top-k selection.  Never
    # substitute the consumer layer for either declared owner.
    idx_owner = index_raw if 0 <= index_raw < ctx.layer.index else ctx.layer.index
    topk_raw = int(ctx.layer.topk_owner_layer_id)
    sel_owner = topk_raw if 0 <= topk_raw < ctx.layer.index else ctx.layer.index
    specs = [("index_score", "indexer", "score"), ("topk", "indexer", "select")]
    source_decl = _indexer_source_declaration(ctx)
    selected = (select_indexer_source(dict(source_decl))
                if source_decl else {"status": "unknown_source"})
    is_c8 = selected.get("status") == "c8_r64"
    source_extra = (("indexer_source", source_decl),) if is_c8 else ()
    signatures = { "index_score": _sig("index_score", heads=(("heads", heads), ("dim", dim), ("topk", topk),
                                   ("kpool", kpool)), quant=(("index_bytes", dtype_bytes(ctx.features.index_dtype)),
                                   ("scale_bytes", _quant_scale_bytes(ctx))),
                            parallel=_parallel(ctx), tiles=_tiles(ctx),
                            extras=(("candidate_blocks", blocks),
                                    ("require_index", idx_owner),
                                    ("cp", int(ctx.features.cp)),
                                    ("cp_plan", str(ctx.features.cp_plan)),
                                    *source_extra)),
        "topk": _sig("index_select", heads=(("topk", topk), ("heads", heads), ("kpool", kpool)),
                     quant=(("index_bytes", dtype_bytes(ctx.features.index_dtype)),),
                     parallel=_parallel(ctx), tiles=_tiles(ctx),
                     extras=(("candidate_blocks", blocks),
                             ("block_size", int(ctx.layer.candidate_block_size or 1)),
                             ("select_ops_per_row", 4),
                             ("owner_layer", sel_owner),
                             ("cp", int(ctx.features.cp)),
                             ("cp_plan", str(ctx.features.cp_plan)),
                             ("state_effects", (("selection", "index_select"),)),
                             *source_extra)), }
    if ctx.layer.candidate_filter:
        specs.append(("candidate_filter", "indexer", "filter"))
        signatures["candidate_filter"] = _sig( "index_filter", heads=(("topk", topk), ("blocks", blocks),
                                   ("heads", heads)), quant=(("index_bytes", dtype_bytes(ctx.features.index_dtype)),),
            parallel=_parallel(ctx), tiles=_tiles(ctx),
            extras=(("block_size", int(ctx.layer.candidate_block_size or 1)),
                    ("select_ops_per_row", 4),
                    ("require_index", idx_owner),
                    ("owner_layer", sel_owner),
                    ("cp", int(ctx.features.cp)),
                    ("cp_plan", str(ctx.features.cp_plan)),
                    ("state_effects", (("selection", "index_select"),))))
    fusions = ()
    if is_c8 and not ctx.layer.candidate_filter:
        # One selected QuantLightningIndexer call: score and topk are physical
        # stages inside the same dispatch, not two standalone launches.  The
        # fusion and its native-only output width exist only when the same
        # source-typed selector says c8_r64; C4/nonquant keep analytic work.
        fusions = (FusionContract(
            "quant_lightning_indexer",
            ("index_score", "topk"),
            removes=("index_scores",),
            keeps=("index_selection",),
            startup=2,
        ),)
    return _built("I1", ctx, variant,
                  "Attention Index Scoring and Top-K Selection (注意力索引评分与 Top-K 选择)",
                  specs, signatures, fusions=fusions, coverage="partial",
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
        intermediate = int(c.get("intermediate_size") or c.get("ffn_dim") or 1)
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
            owns_long_kv=owners[index] == index, native_layer_id=index))
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
