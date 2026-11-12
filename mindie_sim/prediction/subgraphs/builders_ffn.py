"""Canonical builders for FFN, routing, residual, output and draft families.

[Role] Construct F0-F4/R0-R1/E1/O0/D0/V0 templates and static signatures.
[Invariants] Shared/routed branches, quant roles and output accounting are
  explicit; a macro does not recreate a child subgraph's work.
[Timeouts & Locks] Pure bounded construction; no I/O or locks.
[Side-Effect Policy] Returns immutable ``_Built`` records.
"""
from __future__ import annotations
from typing import Any, Mapping

from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.prediction.sequence_parallel import kimi_sequence_parallel
from mindie_sim.prediction.piecewise.vocab_parallel import (
    lm_head_result, logits_collective_kind, vocab_layout,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.subgraphs import FusionContract
from mindie_sim.prediction.moe_comm_selection import (
    ALWAYS_MC2, MoeCommConfig,
)
def _k3_sequence_parallel(ctx: _Ctx) -> bool:
    return kimi_sequence_parallel(ctx.features.metadata, ctx.tp)


from .instantiate import (
    _Ctx, _Built, _HIDDEN_IN, _HIDDEN_OUT, _attention_sig, _built, _collective,
    _copy, _declared_compute_quant, _dense_flashcomm, _dense_sp_block, _dense_sp_exit,
    _elementwise, _in, _matrix, _matrix_act_format, _norm_gamma, _out,
    _parallel, _precede, _quant_fields, _result_bytes, source_lowering_selected,
    _sig, _state, _tiles, _variant,
)


def _build_f0(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, str(ctx.layer.ffn_activation or "silu"))
    act_bytes = _result_bytes(ctx, "dense_mlp")
    ungated = variant == "ungated"
    gate_n = ctx.local_intermediate if ungated else 2 * ctx.local_intermediate
    specs = [("gate_up", "gemm", "gate_up"), ("activation", "elementwise", "activation"),
             ("down", "gemm", "down")]
    if _k3_sequence_parallel(ctx):
        specs.insert(0, ("tp_all_gather", "collective", "tp"))
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
    if _k3_sequence_parallel(ctx):
        signatures["tp_all_gather"] = _collective(
            "tp", ctx, axis="tp", members=ctx.tp,
            bytes_per_token=ctx.hidden * act_bytes,
            extra=(("sequence_parallel_role", "allgather"),
                   ("collective", "allgather")))
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
            signatures["tp_allreduce"] = _collective(
                "tp", ctx, axis="tp", members=ctx.tp,
                bytes_per_token=ctx.hidden * act_bytes)
    # After the collective block so a gather, when present, stays in front.
    # These names are outside dequant_swiglu_quant's member set.
    _precede(specs, signatures, ctx, "gate_up", ctx.hidden, "dense_mlp",
             dynamic_only=True)
    _precede(specs, signatures, ctx, "down", ctx.local_intermediate, "dense_mlp",
             dynamic_only=True)
    # An ungated projection has no gate/value pair to fuse, so it must not
    # advertise the gated dequant_swiglu_quant contract.
    fusions = (() if ungated or not source_lowering_selected(ctx, "dequant_swiglu_quant")
               else (FusionContract(
                   "dequant_swiglu_quant", ("gate_up", "activation"),
                   removes=("ffn_activation", "activation_intermediate"),
                   keeps=("down_input", "activation_scale", "residual")),))
    return _built("F0", ctx, variant, "Dense Feed-Forward Network (FFN 稠密前馈网络)",
                  specs, signatures, fusions=fusions,
                  state_in=(_HIDDEN_IN,), state_out=(_HIDDEN_OUT,))


def _gate_scope(experts: int, groups: int, group_topk: int) -> int:
    """Experts admitted by one grouped gate.

    The score GEMM and the top-k scan still cover every expert.  The grouped
    selector admits ``(E / groups) * group_topk`` of them when the partition
    divides E.  A missing or uneven partition keeps the full expert set.
    """
    if groups > 1 and group_topk > 0 and experts % groups == 0:
        return (experts // groups) * min(group_topk, groups)
    return experts


def _moe_comm_config(ctx: _Ctx, experts: int) -> MoeCommConfig | None:
    """Source-selected method input shared by F1/F2/F3.

    A fused-MC2 deployment declaration activates the dynamic selector.  An
    explicit ``moe_recipe='mc2'`` keeps its historical always-MC2 meaning so
    the existing static caller remains a real physical transport rather than a
    renamed generic AllToAll.
    """
    config = MoeCommConfig.from_features(
        ctx.features, ep=ctx.ep, top_k=ctx.top_k, experts=experts)
    if config is None and ctx.features.moe_recipe == "mc2":
        capture = tuple(ctx.features.graph_buckets or ())
        capacity = max(capture) if capture else 512
        config = MoeCommConfig(
            mode=ALWAYS_MC2, policy="", capacity=capacity,
            world_size=max(1, ctx.ep), tp=max(1, ctx.tp),
            top_k=max(1, ctx.top_k),
            experts_per_device=max(1, int(experts) // max(1, ctx.ep)))
    return config


def _moe_comm_fields(config: MoeCommConfig | None) -> tuple[tuple[str, object], ...]:
    return () if config is None else (("moe_comm", config.to_items()),)


def _mc2_payload(ctx: _Ctx) -> tuple[str, str]:
    """Declared routed payload representation for the native V3 row layout.

    The model quant, when it resolves a routed activation format, owns the
    representation.  A deployment may also declare the source row layout
    explicitly in metadata; that declaration wins because the model header can
    be unreadable while the selected deployment still runs the quantized V3.
    """
    meta = dict(ctx.features.metadata)
    declared = str(meta.get("moe_dispatch_quant", "") or "").strip().lower()
    declared_payload = str(meta.get("moe_dispatch_payload_dtype", "") or "").strip().lower()
    # ``checkpoint_storage_dtypes.routed_expert`` can be the bare packing dtype
    # ``U8``; storage alone does not select FP4.  The signed category/method
    # below is authoritative.
    if declared:
        if declared not in {"none", "mx", "per_token", "per_group"}:
            raise ModelingError(f"moe_dispatch_quant_invalid:{declared}")
        if declared == "none":
            return "none", str(ctx.features.activation_dtype or "bf16")
        payload = declared_payload or {
            "mx": "fp8_e4m3", "per_token": "int8", "per_group": "fp8_e4m3"}[declared]
        return declared, payload
    # The routed expert role, not the model-global activation, owns the
    # dispatch payload representation.  A header-bound W4A4 model can have a
    # global FP8 activation while gate/up/down are MXFP4.
    quant = ctx.features.quant
    resolved = quant.for_category("routed_expert") if quant is not None else None
    act = str((resolved.activation_format if resolved is not None else "") or
              ctx.features.quant_activation_format or
              ctx.features.activation_dtype or "bf16").strip().lower()
    scale = str((resolved.activation_scale_format if resolved is not None else "") or
                getattr(quant, "activation_scale_format", "") or
                meta.get("activation_scale_format", "") or "").strip().lower()
    group = int((resolved.activation_group_size if resolved is not None else 0) or 0)
    if act in {"fp8", "fp8_e4m3", "fp8_e5m2"} and scale in {"fp8_e8m0", "e8m0"}:
        return "mx", act
    if act in {"fp4", "fp4_e2m1", "mxfp4", "mx_fp4"} and (
            group in {0, 32} and scale in {"", "fp8_e8m0", "e8m0"}):
        return "mx", "fp4_e2m1"
    if act in {"int8", "int8_dynamic"}:
        return "per_token", "int8"
    # A model header that does not resolve a routed activation keeps the
    # native source's unquantized BF16 row contract.  FP4 output quantization
    # is not invented here; it remains an explicit unresolved representation.
    return "none", str(ctx.features.activation_dtype or "bf16")


def _mc2_phase_fields(ctx: _Ctx, kind: str) -> tuple[tuple[str, object], ...]:
    quant, payload = _mc2_payload(ctx)
    experts = int(ctx.layer.num_experts or ctx.model.num_experts or 1)
    raw = [("kind", kind), ("width", ctx.latent),
           ("dtype", str(ctx.features.activation_dtype or "bf16")),
           ("experts", experts), ("ep", max(1, ctx.ep)),
           ("top_k", max(1, ctx.top_k))]
    if kind == "dispatch":
        raw += [("payload_dtype", payload), ("dispatch_quant", quant)]
    return (("mc2_phase", tuple(raw)),)


def _build_f1(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, str(ctx.layer.router_mode or "softmax"))
    experts = int(ctx.layer.num_experts or ctx.model.num_experts or 1)
    groups = int(ctx.layer.router_groups or 1)
    group_topk = int(ctx.layer.router_group_topk or 0)
    scope = _gate_scope(experts, groups, group_topk)
    moe_comm = _moe_comm_config(ctx, experts)
    specs = [("router_gemm", "router", "router")]
    # Native MoeGatingTopK inserts the group selector only when the layer
    # declares more than one group.  It is that layer's partition, not the
    # model default and not the neighbouring layer's groups.
    if groups > 1 and group_topk > 0:
        specs.append(("router_group_select", "elementwise", "group_select"))
    specs.extend((("router_topk", "router", "topk"),
                  ("router_histogram", "router", "histogram"),
                  ("router_sort", "router", "sort")))
    router_k = int(ctx.layer.router_input_size or ctx.hidden)
    declared = str(ctx.layer.router_dtype or "")
    if declared:
        router_sig = _sig(
            "router", matrix=(("K", router_k), ("N", experts)),
            quant=_declared_compute_quant(experts, router_k, declared),
            parallel=_parallel(ctx), tiles=_tiles(ctx),
            extras=(("groups", groups), ("group_topk", group_topk),
                    ("scope_experts", scope)))
    else:
        router_sig = _matrix(
            "router", router_k, experts, ctx, "router",
            extra=(("groups", groups), ("group_topk", group_topk),
                   ("scope_experts", scope)))
    signatures = {
        "router_gemm": router_sig,
        "router_topk": _elementwise("topk", ctx, elements=experts, ops=4,
                             read_bytes=experts * 4, write_bytes=ctx.top_k * 8,
                             extra=(("E", experts), ("k", ctx.top_k),
                                    ("groups", groups), ("group_topk", group_topk),
                                    ("scope_experts", scope))),
        "router_histogram": _elementwise("histogram", ctx, elements=ctx.top_k, ops=4,
                                  read_bytes=ctx.top_k * 8, write_bytes=experts * 8,
                                  extra=(("E", experts), ("k", ctx.top_k))),
        "router_sort": _elementwise("sort", ctx, elements=ctx.top_k, ops=8,
                             read_bytes=ctx.top_k * 8, write_bytes=ctx.top_k * 8,
                             extra=(("E", experts), ("k", ctx.top_k))),
    }
    # Outbound local token movement is a separate real launch before dispatch.
    # It is absent on decode/static paths, where only MoeGatingTopK executes.
    specs.append(("router_permute", "router", "permute"))
    signatures["router_permute"] = _elementwise(
        "permute", ctx, elements=ctx.top_k, ops=2,
        read_bytes=ctx.hidden * 2 + ctx.top_k * 4,
        write_bytes=ctx.hidden * ctx.top_k * 2 + ctx.top_k * 4,
        extra=(("E", experts), ("k", ctx.top_k),
               ("router_hidden", ctx.hidden),
               ("router_dtype_bytes", dtype_bytes(
                   ctx.features.activation_dtype or "bf16")),
               ("router_indices_bytes", 4),
               ("router_source_operation",
                "MoeTokenPermute/MoeInitRoutingV2"),
               ("router_source_dtype",
                str(ctx.features.activation_dtype or "bf16")),
               ("router_source_device", "AIV"),
               ("router_selected", "prefill_local_outbound"),
               *_moe_comm_fields(moe_comm)))
    if groups > 1 and group_topk > 0:
        signatures["router_group_select"] = _elementwise(
            "group_select", ctx, elements=experts, ops=2,
            read_bytes=experts * 4, write_bytes=groups * 4,
            extra=(("E", experts), ("groups", groups), ("group_topk", group_topk),
                   ("scope_experts", scope), ("scope", "group_partition")))
    return _built("F1", ctx, variant, "MoE Routing and Token Permutation (MoE 路由与 Token 重排)",
                  specs, signatures,
                  coverage="partial",
                  state_in=(_HIDDEN_IN,), state_out=(_out("routing_ledger", "routing"),
                                                     _out("expert_rows", "routing")))


def _meta_width(meta: dict, name: str) -> int:
    if meta.get(name) is None:
        raise ModelingError(f"moe_exchange_widths_required:{name}")
    raw = meta[name]
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ModelingError(f"moe_exchange_width_invalid:{name}")
    return int(raw)


def _resolve_widths(ctx: _Ctx, meta: dict):
    """Declared row widths, or a native FP8/E8M0/BF16 proof.

    A route that is already active may omit all three widths.  The native
    dispatch spec fills them only when it proves a quantized payload, a
    separate E8M0 scale, and integral byte counts.  One declared width with
    another missing stays an error.  A route-less caller never reaches here.
    """
    names = ("payload_row_bytes", "scale_row_bytes", "return_row_bytes")
    missing = [name for name in names if meta.get(name) is None]
    if missing and len(missing) < 3:
        raise ModelingError(f"moe_exchange_widths_required:{missing[0]}")
    if not missing:
        return tuple(_meta_width(meta, name) for name in names)
    if getattr(ctx.model, "quant", None) is None:
        raise ModelingError("moe_exchange_widths_required:payload_row_bytes")
    from mindie_sim.prediction.moe_transport import default_dispatch_spec
    from mindie_sim.prediction.piecewise.moe_unfused_exchange import (
        proven_exchange_widths,
    )
    spec = default_dispatch_spec(ctx.model, plan=ctx.layer, width=ctx.latent)
    proven = proven_exchange_widths(spec)
    return tuple(proven[name] for name in names)


def _f2_exchange(ctx: _Ctx):
    """Declared unfused route, or None when F2 keeps one transport.

    MC2 stays the fused dispatch/combine pair.  EP1 has no peer exchange.
    A row route, proven equal placement, or ragged source rows become three
    legs.  Ragged rows do not grow a fabricated matrix.  Width lookup happens
    only after a route is declared, so a route-less graph stays one BF16
    dispatch even when the model quant would prove separate widths.
    """
    if ctx.features.moe_recipe == "mc2" or ctx.ep <= 1:
        return None
    meta = dict(ctx.features.metadata)
    placement = str(meta.get("source_placement", "") or "").lower()
    routing = meta.get("routing_matrix")
    source_rows = meta.get("source_rows")
    if routing is None and placement != "equal" and not (
            placement == "ragged" and source_rows is not None):
        return None
    if placement and placement not in {"equal", "ragged", "unknown"}:
        raise ModelingError(f"moe_source_placement_invalid:{placement}")
    payload, scale, ret = _resolve_widths(ctx, meta)
    fields = (("payload_row_bytes", payload), ("scale_row_bytes", scale),
              ("return_row_bytes", ret))
    if meta.get("mtu_bytes") is not None:
        mtu = meta["mtu_bytes"]
        if isinstance(mtu, bool) or not isinstance(mtu, int) or mtu < 1:
            raise ModelingError("moe_unfused_mtu_invalid")
        fields += (("mtu_bytes", int(mtu)),)
    if routing is not None:
        unit = str(meta.get("routing_unit", "") or "")
        if unit != "rows":
            raise ModelingError(f"moe_routing_unit_invalid:{unit or 'missing'}")
        fields += (("routing_matrix", routing), ("routing_unit", "rows"),
                   ("source_placement", placement or "ragged"))
        return "legs", fields, {"payload": payload, "scale": scale, "return": ret}
    if placement == "equal":
        fields += (("source_placement", "equal"),)
        # Unequal rows contradict a proven equal split.  The compiler rejects
        # that pair instead of averaging the rows into the uniform ledger.
        if source_rows is not None:
            fields += (("source_rows", tuple(source_rows)),)
        return "legs", fields, {"payload": payload, "scale": scale, "return": ret}
    fields += (("source_placement", "ragged"), ("source_rows", tuple(source_rows)))
    if meta.get("return_rows") is not None:
        fields += (("return_rows", tuple(meta["return_rows"])),)
    if meta.get("received_rows") is not None:
        fields += (("received_rows", tuple(meta["received_rows"])),)
    return "bound", fields, {"payload": payload, "scale": scale, "return": ret}


def _leg_extra(decl, leg: str):
    _mode, fields, _widths = decl
    return fields + (("exchange_leg", leg),)


def _build_f2(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, ctx.features.moe_recipe)
    hidden = ctx.latent
    top_k = ctx.top_k
    experts = int(ctx.layer.num_experts or ctx.model.num_experts or 1)
    wire = dtype_bytes("bf16")
    dispatch_bytes = hidden * wire
    moe_comm = _moe_comm_config(ctx, experts)
    comm_extra = _moe_comm_fields(moe_comm)
    extra = (("ep", ctx.ep), ("etp", ctx.etp), ("top_k", top_k),
             ("axis", "ep"), ("members", ctx.ep),
             ("chunks", ctx.features.mc2_chunks),
             ("recipe", ctx.features.moe_recipe),
             ("overlap", int(ctx.features.moe_recipe == "mc2")),
             ("latent", hidden), ("wire_bytes", wire))
    decl = _f2_exchange(ctx)
    combine_extra = extra + comm_extra + (("direction", "receive"),)
    combine_row = dispatch_bytes
    if decl is not None:
        combine_extra = (extra + _leg_extra(decl, "return") + comm_extra
                         + (("direction", "receive"),))
        combine_row = decl[2]["return"]
    combine = _sig("combine", matrix=(("row_bytes", combine_row),),
                   parallel=_parallel(ctx), tiles=_tiles(ctx),
                   quant=(("wire_bytes", wire),),
                   extras=combine_extra)
    mc2_combine = _sig("mc2_combine", matrix=(("row_bytes", dispatch_bytes),),
                       parallel=_parallel(ctx), tiles=_tiles(ctx),
                       quant=(("wire_bytes", wire),),
                       extras=extra + _mc2_phase_fields(ctx, "combine") + comm_extra)
    if ctx.role == "transport_combine":
        # Combine consumes the routed expert output, so it is a distinct
        # post-expert instance scheduled after F3.  The weighted return
        # finalize is a separate real launch after this combine; it is not a
        # parallel shared-expert node and is present without shared experts.
        specs = [("combine", "transport", "combine")]
        signatures = {"combine": combine}
        if ctx.features.moe_recipe != "mc2":
            finalize = _elementwise(
                "finalize", ctx, elements=ctx.top_k, ops=2,
                read_bytes=ctx.latent * ctx.top_k * 2 + ctx.top_k * 6,
                write_bytes=ctx.latent * 2,
                extra=(("E", experts),
                       ("k", ctx.top_k),
                       ("router_hidden", ctx.latent),
                       ("router_dtype_bytes", dtype_bytes(
                           ctx.features.activation_dtype or "bf16")),
                       ("router_indices_bytes", 4),
                       ("router_prob_bytes", 2),
                       ("router_source_operation",
                        "MoeTokenUnpermute/MoeFinalizeRoutingV2"),
                       ("router_source_dtype",
                        str(ctx.features.activation_dtype or "bf16")),
                       ("router_source_device", "AIV"),
                       ("router_upstream", "combine"),
                       ("router_input", "combined_routed_return"),
                       ("router_selected", "post_combine_weighted_return"),
                       *comm_extra))
            specs.append(("router_finalize", "router", "finalize"))
            signatures["router_finalize"] = finalize
        if moe_comm is not None:
            # The native V3 return owns framing, transpose and the weighted
            # output publication.  It is the physical fused MC2 combine; the
            # generic AllToAll combine above remains the retained non-MC2 path.
            specs.append(("mc2_combine", "mc2_transport", "mc2_return"))
            signatures["mc2_combine"] = mc2_combine
        return _built("F2", ctx, variant,
                      "MoE Token Dispatch and Combine (MoE Token 分发与结果合并)",
                      specs, signatures, coverage="partial",
                      state_in=(_HIDDEN_IN, _in("expert_rows", "routing")),
                      state_out=(_HIDDEN_OUT, _out("peer_traffic", "traffic")))
    quant_write = (hidden * wire if decl is None
                   else decl[2]["payload"] + decl[2]["scale"])
    quantize = _elementwise("quant", ctx, elements=hidden, ops=2,
                            read_bytes=hidden * 2, write_bytes=quant_write,
                            extra=(("expert_count", ctx.local_experts),
                                   *comm_extra))
    dispatch_extra = extra + comm_extra + (("direction", "send"), ("input_quantized", 1))
    dispatch_row = dispatch_bytes
    specs = [("quantize", "elementwise", "quant"),
             ("dispatch", "transport", "dispatch"),
             ("notify", "collective", "notify")]
    if decl is not None:
        dispatch_extra = extra + _leg_extra(decl, "payload") + comm_extra + (
            ("direction", "send"), ("input_quantized", 1))
        dispatch_row = decl[2]["payload"]
        specs = [("quantize", "elementwise", "quant"),
                 ("dispatch", "transport", "dispatch"),
                 ("dispatch_scale", "transport", "dispatch"),
                 ("notify", "collective", "notify")]
    dispatch = _sig("dispatch", matrix=(("row_bytes", dispatch_row),),
                    parallel=_parallel(ctx), tiles=_tiles(ctx),
                    quant=(("wire_bytes", wire),),
                    extras=dispatch_extra)
    notify = _collective("notify", ctx, axis="ep", members=ctx.ep,
                         bytes_per_token=8.0, stages=1,
                         extra=(("chunks", ctx.features.mc2_chunks), *comm_extra))
    mc2_dispatch = _sig("mc2_dispatch", matrix=(("row_bytes", dispatch_bytes),),
                        parallel=_parallel(ctx), tiles=_tiles(ctx),
                        quant=(("wire_bytes", wire),),
                        extras=extra + _mc2_phase_fields(ctx, "dispatch") + comm_extra)
    allgather_extra = tuple(
        (key, value) for key, value in extra
        if key not in {"members", "group_axis"})
    moe_allgather = _collective(
        "allgather", ctx, axis="ep", members=ctx.ep,
        bytes_per_token=hidden * wire, stages=1,
        extra=allgather_extra + comm_extra + (
            ("moe_comm_active_var", "MOE_COMM_SELECTED_ALLGATHER"),))
    signatures = {"quantize": quantize, "dispatch": dispatch, "notify": notify}
    if decl is not None:
        signatures["dispatch_scale"] = _sig(
            "dispatch", matrix=(("row_bytes", decl[2]["scale"]),),
            parallel=_parallel(ctx), tiles=_tiles(ctx),
            quant=(("wire_bytes", wire),),
            extras=extra + _leg_extra(decl, "scale") + comm_extra + (
                ("direction", "send"), ("input_quantized", 1)))
    if moe_comm is not None:
        # The native fused dispatch owns local init/reorder/quantization and
        # the framed peer send.  It is not a renamed generic AllToAll node.
        specs.insert(0, ("mc2_dispatch", "mc2_transport", "mc2_send"))
        signatures["mc2_dispatch"] = mc2_dispatch
        # A5 can select AllGather instead of either exchange when the group is
        # no wider than top-k.  Keep that as a separate positive owner rather
        # than silently zeroing F2.
        specs.insert(0, ("moe_allgather", "collective", "moe_allgather"))
        signatures["moe_allgather"] = moe_allgather
    return _built("F2", ctx, variant,
                  "MoE Token Dispatch and Combine (MoE Token 分发与结果合并)",
                  specs, signatures,
                  coverage="partial",
                  state_in=(_HIDDEN_IN, _in("expert_rows", "routing")),
                  state_out=(_out("peer_traffic", "traffic"),))

def _build_f3(ctx: _Ctx) -> _Built:
    variant = _variant(ctx, "latent" if ctx.layer.expert_latent_size else "packed")
    experts = int(ctx.layer.num_experts or ctx.model.num_experts or 1)
    local = ctx.local_experts
    expert_width = ctx.expert_width
    latent = ctx.latent
    routed_act = _matrix_act_format(ctx, "routed_expert")
    meta = dict(ctx.features.metadata)
    fusion_roles = tuple(str(role) for role in (meta.get("native_fusion_roles") or ()))
    # The selected routed-expert activation may be produced by the first
    # grouped composite (GroupedMatmulSwigluQuantV2): gate/up + SwiGLU +
    # quantized payload and E8M0 scale, followed by an independent BF16
    # GroupedMatmulV4 return.  A source-selected native_fusion role is itself
    # selection; a high-precision model header still keeps the old separate
    # W8/bf16 activation contract unless the role declares the composite.
    composite = (
        routed_act in {"fp4", "fp4_e2m1", "mxfp4"}
        or ("grouped_gate_up_activation_quant" in fusion_roles
            and routed_act not in {"fp8", "fp8_e4m3", "fp8_e5m2", "int8"}))
    moe_comm = _moe_comm_config(ctx, experts)
    comm_extra = _moe_comm_fields(moe_comm)
    specs = ([("expert_gate_up", "grouped_gemm", "gate_up"),
              ("expert_down", "grouped_gemm", "down")] if composite else
             [("expert_gate_up", "grouped_gemm", "gate_up"),
              ("expert_activation", "elementwise", "activation"),
              ("expert_down", "grouped_gemm", "down")])
    # Inbound MoeInitRoutingV3 is selected only when a real local received-row
    # ledger is declared.  A global marginal or local M*k is not a received
    # row count and leaves this node absent.
    meta = dict(ctx.features.metadata)
    layer_index = int(ctx.layer.index)

    def layer_rows(raw):
        # A full per-layer route ledger is required. A scalar is one layer's
        # value and must never be broadcast over every layer.
        if isinstance(raw, Mapping):
            value = raw.get(layer_index)
            return value if isinstance(value, int) and not isinstance(value, bool) else None
        if (isinstance(raw, (tuple, list))
                and len(raw) == int(ctx.layers or 0)
                and isinstance(raw[layer_index], int)
                and not isinstance(raw[layer_index], bool)):
            return raw[layer_index]
        return None

    received = layer_rows(meta.get("moe_received_rows"))
    if received is None:
        raw = meta.get("received_rows")
        rank = meta.get("moe_rank_index", meta.get("rank_index"))
        if isinstance(raw, Mapping) and isinstance(rank, int) and not isinstance(rank, bool):
            vector = raw.get(layer_index)
            if (isinstance(vector, (tuple, list))
                    and 0 <= rank < len(vector)
                    and isinstance(vector[rank], int)
                    and not isinstance(vector[rank], bool)):
                received = vector[rank]
    # The incoming V3 payload is the routed-expert activation selected by the
    # source caller's dst_type, not the model-global activation format.
    payload_format = str(routed_act or "")
    if payload_format not in {"fp8_e4m3", "fp8", "mxfp8",
                              "fp4", "fp4_e2m1", "mxfp4", "mxfp4_e2m1"}:
        payload_format = ""
    source_image = str(meta.get("image_id") or meta.get("oci_image_digest")
                       or meta.get("deployment_image_id") or "")
    source_r64 = source_image.split(":", 1)[-1].lower().startswith(
        "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e")
    _inbound_signature = None
    if (payload_format and source_r64
            and isinstance(received, int) and not isinstance(received, bool)
            and received > 0):
        payload_bytes = (0.5 if payload_format in {
            "fp4", "fp4_e2m1", "mxfp4", "mxfp4_e2m1"}
            else dtype_bytes(payload_format))
        scale_cols = (ctx.latent + 31) // 32
        read_bytes = received * (ctx.latent * payload_bytes + scale_cols + 4)
        write_bytes = received * (ctx.latent * payload_bytes + scale_cols + 4)
        inbound = _elementwise(
            "inbound", ctx, elements=ctx.latent, ops=2,
            read_bytes=read_bytes, write_bytes=write_bytes,
            extra=(("E", experts), ("k", ctx.top_k),
                   ("router_hidden", ctx.latent),
                   ("router_received_rows", received),
                   ("router_payload_bytes", payload_bytes),
                   ("router_scale_cols", scale_cols),
                   ("router_index_bytes", 4),
                   ("router_source_operation", "MoeInitRoutingV3"),
                   ("router_source_dtype", payload_format),
                   ("router_scale_dtype", "fp8_e8m0"),
                   ("router_source_device", "AIV"),
                   ("router_selected", "post_dispatch_received_rows"),
                   *comm_extra))
        specs.insert(0, ("router_inbound", "router", "inbound"))
        _inbound_signature = inbound
    elif payload_format and source_r64:
        # Source-selected AllToAll V3 exists even when the normal case has no
        # current-rank R ledger. Retain one symbolic owner; do not invent R.
        payload_bytes = (0.5 if payload_format in {
            "fp4", "fp4_e2m1", "mxfp4", "mxfp4_e2m1"}
            else dtype_bytes(payload_format))
        scale_cols = (ctx.latent + 31) // 32
        inbound = _elementwise(
            "inbound", ctx, elements=ctx.latent, ops=2,
            read_bytes=0, write_bytes=0,
            extra=(("E", experts), ("k", ctx.top_k),
                   ("router_hidden", ctx.latent),
                   ("router_received_rows", None),
                   ("router_received_rows_resolved", 0),
                   ("router_symbolic_input", 1),
                   ("router_payload_bytes", payload_bytes),
                   ("router_scale_cols", scale_cols),
                   ("router_index_bytes", 4),
                   ("router_source_operation", "MoeInitRoutingV3"),
                   ("router_source_dtype", payload_format),
                   ("router_scale_dtype", "fp8_e8m0"),
                   ("router_source_device", "AIV"),
                   ("router_selected", "post_dispatch_received_rows"),
                   *comm_extra))
        specs.insert(0, ("router_inbound", "router", "inbound"))
        _inbound_signature = inbound
    extra = (("E", experts), ("E_local", local), ("k", ctx.top_k),
             ("latent", latent), ("expert_width", expert_width),
             ("row_tile", ctx.features.moe_row_tile))
    signatures = {}
    if _inbound_signature is not None:
        signatures["router_inbound"] = _inbound_signature
    gate_extra = extra + comm_extra + (("materialize_selected_rows", 1),)
    down_extra = extra + comm_extra + (("materialize_selected_rows", 1),)
    down_quant = _quant_fields(ctx, "routed_expert", latent, expert_width)
    if composite:
        # The first grouped composite writes the selected activation payload.
        # A source-selected low-precision representation also writes its E8M0
        # scale as a second typed output; an unresolved header keeps the
        # resolved payload representation and does not invent a scale.
        output_format = routed_act
        declared_output = str(meta.get("moe_grouped_output_format", "") or "")
        if declared_output and output_format not in {"fp4", "fp4_e2m1", "mxfp4"}:
            output_format = declared_output
        scale_row = dict(down_quant).get("activation_scale_bytes", 0)
        if output_format in {"fp4", "fp4_e2m1", "mxfp4"} and not scale_row:
            # MXFP4 activation scales are one E8M0 byte per 32 values,
            # packed two per 64-column tail element.
            scale_row = ((expert_width + 31) // 32 + 1) // 2 * 2
        payload_width = (0.5 if output_format in {"fp4", "fp4_e2m1", "mxfp4"}
                         else dtype_bytes(output_format))
        payload_row = expert_width * payload_width
        gate_extra += (
            ("fused_output_format", output_format),
            ("fused_activation_quant", 1),
            ("activation_elements", expert_width),
            ("activation_ops_per_element", 4),
            ("activation_payload_row_bytes", payload_row),
            ("activation_scale_row_bytes", scale_row),
            ("activation_source_operation", "GroupedMatmulSwigluQuantV2"),
            ("activation_typed_outputs", 2 if scale_row else 1),
        )
        down_extra += (
            ("activation_payload_row_bytes", payload_row),
            ("activation_scale_bytes", scale_row),
            ("activation_source", "grouped_gate_up_activation_quant"),
        )
    gate_quant = _quant_fields(ctx, "routed_expert", 2 * expert_width, latent)
    if composite:
        # The typed gate/up theta must expose the selected fused output, not
        # the projection's default BF16 output_format.
        gate_quant = tuple(sorted({
            **dict(gate_quant), "output_format": output_format}.items()))
    signatures["expert_gate_up"] = _sig(
        "expert_gate_up", matrix=(("K", latent), ("N", 2 * expert_width)),
        experts=(("E", experts), ("local", local), ("k", ctx.top_k)),
        quant=gate_quant,
        parallel=_parallel(ctx), tiles=_tiles(ctx), extras=gate_extra)
    if not composite:
        signatures["expert_activation"] = _elementwise(
            "activation", ctx, elements=expert_width, ops=4,
            read_bytes=2 * expert_width * 2,
            write_bytes=expert_width * 2,
            extra=(("expert_width", expert_width),
                   ("E", experts), ("local", local),
                   ("k", ctx.top_k)))
    signatures["expert_down"] = _sig(
        "expert_down", matrix=(("K", expert_width), ("N", latent)),
        experts=(("E", experts), ("local", local), ("k", ctx.top_k)),
        quant=down_quant, parallel=_parallel(ctx), tiles=_tiles(ctx),
        extras=down_extra)
    source_image = str(meta.get("image_id") or meta.get("oci_image_digest")
                       or meta.get("deployment_image_id") or "")
    source_r64 = source_image.split(":", 1)[-1].lower().startswith(
        "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e")
    if _inbound_signature is not None or source_r64:
        # Actual source also has an unweighted local return after GMM and
        # before the return A2A. It is not the weighted final return. If the
        # caller has not declared the local R_rank, retain one physical owner
        # with symbolic [R_rank,H] and zero priced bytes rather than inventing
        # a per-layer row count or dropping the real operator.
        specs.append(("router_unweighted_reorder", "router", "unweighted_return"))
        if received is None:
            signatures["router_unweighted_reorder"] = _elementwise(
                "unweighted_return", ctx, elements=1, ops=2,
                read_bytes=0, write_bytes=0,
                extra=(("E", experts), ("k", ctx.top_k),
                       ("router_hidden", ctx.latent),
                       ("router_received_rows", None),
                       ("router_received_rows_resolved", 0),
                       ("router_symbolic_input", 1),
                       ("router_source_operation",
                        "MoeTokenUnpermute/MoeFinalizeRoutingV2"),
                       ("router_source_dtype", "bf16"),
                       ("router_source_device", "AIV"),
                       ("router_selected", "post_grouped_unweighted_return"),
                       *comm_extra))
        else:
            signatures["router_unweighted_reorder"] = _elementwise(
                "unweighted_return", ctx, elements=ctx.latent, ops=2,
                read_bytes=received * (ctx.latent * 2 + 4),
                write_bytes=received * ctx.latent * 2,
                extra=(("E", experts), ("k", ctx.top_k),
                       ("router_hidden", ctx.latent),
                       ("router_received_rows", received),
                       ("router_received_rows_resolved", 1),
                       ("router_source_operation",
                        "MoeTokenUnpermute/MoeFinalizeRoutingV2"),
                       ("router_source_dtype", "bf16"),
                       ("router_source_device", "AIV"),
                       ("router_selected", "post_grouped_unweighted_return"),
                       *comm_extra))
    if not composite:
        # The first grouped composite consumes the dispatch-owned payload and
        # owns its own SwiGLU/output quantization.  A standalone input quant
        # or a down-input quant in the same route would duplicate that work.
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
        meta = dict(ctx.features.metadata)
        model_meta = dict(getattr(ctx.model, "metadata", ()) or ())
        source_type = str(model_meta.get("source_model_type")
                          or meta.get("source_model_type") or "")
        join_scale = meta.get("routed_scaling_factor")
        # Source paths that apply routed_scaling_factor at the shared/routed
        # combine: glm4_moe.py L218 and kimi_linear.py L167-172.  Other
        # source types own the scale in F1 and stay unresolved (no double).
        join_source_owner = (
            "join" if source_type in {
                "glm_moe_dsa", "glm4_moe",
                "kimi_k3", "kimi_linear"} else "unresolved")
        if (join_source_owner == "join"
                and isinstance(join_scale, (int, float))
                and not isinstance(join_scale, bool) and join_scale > 0):
            join_scale = float(join_scale)
            if join_scale == 1.0:
                # Identity combine scale: the source is a plain add.
                combine_variant = "plain_add"
                scale_owner = "none"
                join_ops = 1
            else:
                combine_variant = "muls_add"
                scale_owner = "join"
                join_ops = 2
        else:
            join_scale = None
            combine_variant = "unresolved"
            scale_owner = "unresolved"
            join_ops = 2
        token_domain = "shared_routed_combine"
        join = _elementwise(
            "join", ctx, elements=hidden, ops=join_ops,
            read_bytes=2 * hidden * act_bytes,
            write_bytes=hidden * act_bytes,
            extra=(("routed_scaling_factor", join_scale),
                   ("combine_variant", combine_variant),
                   ("scale_owner", scale_owner),
                   ("token_domain", token_domain)))
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
    if _k3_sequence_parallel(ctx):
        specs.insert(0, ("shared_tp_allgather", "collective", "tp"))
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
    if _k3_sequence_parallel(ctx):
        signatures["shared_tp_allgather"] = _collective(
            "tp", ctx, axis="tp", members=tp,
            bytes_per_token=hidden * act_bytes,
            extra=(("sequence_parallel_role", "allgather"),
                   ("collective", "allgather"),
                   ("shared_branch", 1)))
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
        if _k3_sequence_parallel(ctx):
            specs.append(("tp_reduce_scatter", "collective", "tp"))
            signatures["tp_reduce_scatter"] = _collective(
                "tp", ctx, axis="tp", members=tp,
                bytes_per_token=hidden * act_bytes,
                extra=(("sequence_parallel_role", "reduce_scatter"),
                       ("collective", "reduce_scatter"),
                       ("sp_rows", "padded")))
        else:
            specs.append(("tp_allreduce", "collective", "tp"))
            signatures["tp_allreduce"] = _collective(
                "tp", ctx, axis="tp", members=tp,
                bytes_per_token=hidden * act_bytes)
    stream = "aux" if ctx.features.multistream_shared_expert else "main"
    return _built("F4", ctx, variant,
                  "Shared Expert FFN and Output Merge (共享专家前馈网络与输出合并)",
                  specs, signatures,
                  state_in=(_HIDDEN_IN,),
                  state_out=(_HIDDEN_OUT,),
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
        # Source HyperConnection.before: FP32 value cast, inverse RMS, the
        # mHC projection, then the scale/bias/gate affine. The inverse RMS
        # stays the existing norm row. The affine is not that norm.
        mixed = (2 + mult) * mult
        specs = [("hc_values_fp32", "elementwise", "cast"),
                 ("inv_rms", "elementwise", "norm"),
                 ("hc_projection", "gemm", "projection"),
                 ("hc_scale_bias_gate", "elementwise", "affine"),
                 ("sinkhorn", "elementwise", "sinkhorn"),
                 ("hc_pre_mix", "elementwise", "hc_pre")]
        signatures = {
            "hc_values_fp32": _elementwise(
                "cast", ctx, elements=streams, ops=1,
                read_bytes=streams * b_x, write_bytes=streams * 4,
                extra=(("input_format", ctx.features.activation_dtype or "bf16"),
                       ("output_format", "fp32"),
                       ("native_recipe", "hc_values_fp32"), ("mult", mult))),
            "inv_rms": _elementwise("norm", ctx, elements=streams, ops=5,
                                    read_bytes=streams * b_x, write_bytes=streams * b_x,
                                    extra=(("mult", mult),)),
            "hc_projection": _matrix("hc_projection", streams, project_n, ctx, "mhc",
                                  extra=(("mult", mult), ("static_dims", mult))),
            "hc_scale_bias_gate": _elementwise(
                "affine", ctx, elements=mixed, ops=5,
                read_bytes=(mixed + 1) * 4, write_bytes=mixed * 4,
                extra=(("mult", mult),
                       ("native_recipe", "hc_normalize_scale_bias_gate"),
                       ("static_parameter_bytes", (mixed + 3) * 4))),
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


_O0_SELECTED_HIDDEN_RUNNER = "NPUModelRunner"
_O0_SELECTED_HIDDEN_INVOCATION = "execute_model"
_O0_SELECTED_HIDDEN_OUTPUT = "hidden_states[logits_indices]"
_O0_SELECTED_HIDDEN_INDEX_CLASS = "advanced_tensor_index"
_O0_SELECTED_HIDDEN_REVISION = "r31/9e869e8148"
_O0_SELECTED_HIDDEN_IMAGE_PREFIX = "9e869e81484"
_O0_SELECTED_HIDDEN_INDICES = "prefill: query_start_loc[1:num_reqs+1]-1"


def _o0_selected_hidden_policy(ctx: _Ctx) -> tuple[tuple[str, Any], ...] | None:
    """The declared source-runner policy for the O0 pre-lm_head logit select.

    The source runner is the retained vLLM-Ascend ``NPUModelRunner``
    ``execute_model`` postprocess, whose output selection is
    ``hidden_states[logits_indices]``.  An explicit ``source_runner`` policy in
    the feature metadata is authoritative (class/invocation/revision/output
    selection/indices).  Without that attachment the exact R31 bound runner is
    still a declared policy model: same class/invocation/output selection and
    the retained indices expression, but ``indices_attached=false`` so the
    native typed occurrence stays partial.  A different image or runner
    returns ``None``; no model-name or trace-kind gate is used.
    """
    if int(ctx.features.pp or 1) != 1:
        # The single-stage non-spec postprocess is the retained source path.
        return None
    facts = dict(ctx.model.catalog_facts or ())
    model_meta = dict(ctx.model.metadata or ())
    if (bool(model_meta.get("is_pooling_model"))
            or bool(model_meta.get("pooling"))
            or "pool" in str(facts.get("architectures") or "").lower()):
        return None
    meta = dict(ctx.features.metadata)
    declared = meta.get("source_runner") or meta.get("source_runner_policy")
    attached = isinstance(declared, Mapping)
    if attached:
        runner = dict(declared)
        if (str(runner.get("class") or "") != _O0_SELECTED_HIDDEN_RUNNER
                or str(runner.get("invocation") or "")
                != _O0_SELECTED_HIDDEN_INVOCATION
                or str(runner.get("output_selection") or "")
                != _O0_SELECTED_HIDDEN_OUTPUT
                or str(runner.get("revision") or "")
                != _O0_SELECTED_HIDDEN_REVISION):
            return None
        indices = str(runner.get("indices") or "").strip()
        if not indices:
            return None
    else:
        image = str(meta.get("image_id") or "").lower()
        if _O0_SELECTED_HIDDEN_IMAGE_PREFIX not in image:
            return None
        indices = _O0_SELECTED_HIDDEN_INDICES
    return (
        ("row_binding", "selected"),
        ("source_runner_class", _O0_SELECTED_HIDDEN_RUNNER),
        ("source_runner_invocation", _O0_SELECTED_HIDDEN_INVOCATION),
        ("source_runner_indices", indices),
        ("source_runner_output_selection", _O0_SELECTED_HIDDEN_OUTPUT),
        ("source_runner_index_class", _O0_SELECTED_HIDDEN_INDEX_CLASS),
        ("source_runner_index_dim", 0),
        ("source_runner_index_dtype", "int64"),
        ("source_runner_revision", _O0_SELECTED_HIDDEN_REVISION),
        ("source_runner_indices_attached", int(attached)),
        ("source_runner_policy_source",
         "declared_metadata" if attached else "retained_r31_runner_policy"),
        ("source_runner_native_acl_api", "aclnnIndex"),
        ("source_runner_observed_kernel", "IndexAiCore"),
        ("source_runner_observed_profiler", "aclnnIndex_IndexAiCore_Index"),
        ("source_runner_observed_core", "aiv"),
        ("source_runner_selected_policy", "logits_indices_terminal_rows"),
        ("native_kind", "aclnnIndex"),
        ("deployment_source_revision", "r31"),
    )


def _build_o0(ctx: _Ctx) -> _Built:
    scope = ctx.features.final_norm_scope
    variant = _variant(ctx, scope)
    act_bytes = dtype_bytes(ctx.features.activation_dtype or "bf16")
    # Scope all on the dense sequence-parallel path normalises this rank's
    # ceil split of the same sequence the exit gather reads. Scope selected
    # normalises the producing rows once and does not split them again.
    # Flash-off and TP 1 keep the unsharded count.
    norm_extra = _norm_gamma(ctx, ctx.hidden, "final")
    _meta = dict(getattr(ctx.features, "metadata", ()) or ())
    _normal = _meta.get("normal_source") if isinstance(
        _meta.get("normal_source"), dict) else {}
    if (str(_normal.get("source_class") or "") == "AscendRMSNorm"
            and _normal.get("source_weight_offset") == 0.0
            and _normal.get("source_role") == "fused_residual"
            and not _normal.get("source_conflict")):
        _final_data = dict((_normal.get("role_bias") or {}).get("final_norm") or {})
        _ctor = _normal.get("constructor_wants_bias")
        if "model.norm.bias" in set(_final_data.get("loaded") or ()):
            _final_state = "present_trained"
            _final_value = "loaded"
        elif _ctor is False and "model.norm.bias" in set(_final_data.get("absent") or ()):
            _final_state = "absent"
            _final_value = ""
        elif _ctor is True and "model.norm.bias" in set(_final_data.get("absent") or ()):
            _final_state = "present_zero"
            _final_value = "zero"
        else:
            _final_state = "unresolved"
            _final_value = ""
        norm_extra = norm_extra + (
            ("norm_eps", ctx.model.norm_eps),
            ("layernorm", 0),
            ("norm_arch", "rmsnorm"),
            ("norm_weight_offset", 0.0),
            ("native_norm_kind", "AddRmsNormBias"),
            ("native_norm_source", "AscendRMSNorm.forward_oot final residual branch"),
            ("norm_bias_state", _final_state),
            ("norm_bias_origin", "AscendRMSNorm final residual forward_oot; optional beta"),
            ("norm_bias_value", _final_value),
            ("norm_bias_trained", _final_state == "present_trained"),
            ("norm_bias_selector", "final residual-present custom-op branch; optional beta absent/zero/trained"),
            ("norm_has_weight", True),
            ("norm_source_package_sha256", str(_normal.get("vllm_ascend_layernorm_sha256") or "")),
            ("norm_source_weights_sha256", str(_normal.get("quant_manifest_sha256") or "")),
            ("returns", ("x", "residual")),
        )
        if _final_state in {"present_zero", "present_trained"}:
            norm_extra = norm_extra + (("norm_has_bias", True),)
        elif _final_state == "absent":
            norm_extra = norm_extra + (("norm_has_bias", False),)
    if scope == "selected":
        norm_extra = norm_extra + (("sequence_rows", "selected"),)
    elif _dense_flashcomm(ctx) and int(ctx.tp) > 1:
        norm_extra = norm_extra + (
            ("sequence_rows", "shard"),
            ("sequence_ranks", int(ctx.tp)),
        )
    # Terminal source role is AddRmsNorm after the final down projection:
    # it consumes residual+down, then writes the 1x5120 hidden for lm_head.
    norm_extra = norm_extra + (("residual_add", 1), ("fusion", "add_rmsnorm"))
    norm = _elementwise("norm", ctx, elements=ctx.hidden, ops=5,
                        read_bytes=2 * ctx.hidden * act_bytes,
                        write_bytes=2 * ctx.hidden * act_bytes + 4.0,
                        extra=norm_extra + (("rstd_bytes", 4.0),
                                            ("residual_out_bytes", ctx.hidden * act_bytes)))
    select = _copy("select", ctx, elements=ctx.hidden,
                   bytes_per_element=act_bytes)
    hidden_select = None
    if scope == "selected":
        # Select the last valid row first, then normalise only those rows.
        specs = [("hidden_select", "copy", "select"),
                 ("final_norm", "elementwise", "norm")]
        signatures = {"hidden_select": select, "final_norm": norm}
    else:
        # Normalise every row, then the bound runner selects its logit rows.
        specs = [("final_norm", "elementwise", "norm")]
        signatures = {"final_norm": norm}
        policy = _o0_selected_hidden_policy(ctx)
        if policy is not None:
            hidden_select = _copy(
                "select", ctx, elements=ctx.hidden,
                bytes_per_element=act_bytes, extra=policy)
    # Source ParallelLMHead/VocabParallelEmbedding pads the vocabulary to a
    # 64-aligned block before dividing it across TP ranks.  TP1 keeps the
    # padded full width.  TP>1 computes one local shard and, unless a
    # declaration says otherwise, all-gathers that shard once before sampling.
    # A tied lm_head reuses the embedding shard; it does not create a second
    # full-vocabulary copy.  The collective moves the matmul result, selected
    # rows by result dtype, not the sampler fp32 width and not the prompt.
    layout = vocab_layout(ctx.model, ctx.features)
    local_vocab = layout.local
    padded_vocab = layout.padded
    full_vocab = layout.full
    if layout.tp > 1 and _dense_flashcomm(ctx):
        _dense_sp_exit(specs, signatures, ctx)
    if _k3_sequence_parallel(ctx):
        specs.insert(0, ("tp_hidden_allgather", "collective", "tp"))
        signatures["tp_hidden_allgather"] = _collective(
            "tp", ctx, axis="tp", members=ctx.tp,
            bytes_per_token=ctx.hidden * act_bytes,
            extra=(("sequence_parallel_role", "allgather"),
                   ("collective", "allgather"), ("terminal", "hidden")))
    if hidden_select is not None:
        specs.append(("hidden_select", "copy", "select"))
        signatures["hidden_select"] = hidden_select
    specs += [("lm_head", "gemm", "lm_head")]
    if _k3_sequence_parallel(ctx):
        specs.append(("tp_vocab_allgather", "collective", "tp"))
        signatures["tp_vocab_allgather"] = _collective(
            "tp", ctx, axis="tp", members=ctx.tp,
            bytes_per_token=local_vocab * 2,
            extra=(("sequence_parallel_role", "allgather"),
                   ("collective", "allgather"), ("terminal", "vocab"),
                   ("result_bytes", 2)))
    signatures["lm_head"] = _matrix("lm_head", ctx.hidden, local_vocab,
                                    ctx, "lm_head",
                                    extra=(("selected_rows", 1),
                                           ("local_vocab", local_vocab),
                                           ("padded_vocab", padded_vocab)))
    logits_kind = (None if _k3_sequence_parallel(ctx)
                   else logits_collective_kind(ctx.model, ctx.features))
    if logits_kind is not None:
        result_fmt, result_bytes = lm_head_result(ctx.model, ctx.features)
        specs.append(("logits_gather", "collective", "gather"))
        signatures["logits_gather"] = _collective(
            "logits", ctx, axis="tp", members=layout.group_world,
            bytes_per_token=local_vocab * result_bytes,
            extra=(("collective", logits_kind),
                   ("row_binding", "lm_head_selected"),
                   ("local_vocab", local_vocab), ("padded_vocab", padded_vocab),
                   ("full_vocab", full_vocab),
                   ("org_vocab", int(ctx.model.vocab_size)),
                   ("result_dtype", result_fmt),
                   ("result_bytes", result_bytes)))
    specs.append(("sampling", "elementwise", "sample"))
    signatures["sampling"] = _elementwise(
        "sample", ctx, elements=ctx.model.vocab_size, ops=4,
        read_bytes=ctx.model.vocab_size * 4, write_bytes=8.0,
        extra=(("vocab", ctx.model.vocab_size), ("tied", int(ctx.model.tie_word_embeddings)),
               ("local_vocab", local_vocab), ("padded_vocab", padded_vocab)))
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
