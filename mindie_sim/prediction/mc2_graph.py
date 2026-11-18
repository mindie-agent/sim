"""Source-selected A5 MC2 dispatch/return phases in the ordinary MoE graph.

[Role] Replace generic packing and all-to-all with framed fullmesh channels.
[Invariants] One native kernel group owns each dispatch/return invocation.
  Count/status and payload have independent peer messages.  Omitted finite
  max_inflight_per_rank resolves to the maximum remote-peer bound, not a
  guessed finite vendor buffer window; an explicit value overrides.  Combine
  reverses
  routed rows before changing frame width. Router weights, expert computation,
  explicit activation quantization and final weighted reduction remain real.
[Timeouts & Locks] O(layer operators + AIV count), no I/O or remote work.
[Side-Effect Policy] Returns immutable operations; no hardware or source reads.
"""
from dataclasses import replace
import math

from mindie_sim.schema.common import ModelingError
from .direct_peer import DirectPeerCollective
from .mechanism import PlannedRecipe, Work
from .mc2_phase import RECIPE, build, contract
from .operators.mc2_fullmesh import SOURCE, dispatch_layout, combine_layout
from .operators.mc2_mx_quant import mx32_work
from .moe_comm_selection import MC2, MoeCommConfig

NETWORK = "arch35_fullmesh_peer_channel"
NO_ASSIGNMENTS = 0
EXPLICIT_ROUTING_LOWERINGS = frozenset(
    {"route_pack", "route_unpack", "moe_init_routing"})


def _absorbed_quant_mode(routing):
    """Dispatch payload mode declared by one absorbed init-routing pack."""
    payload = str(routing.get("payload_format") or "")
    scale_format = str(routing.get("scale_format") or "")
    if payload in {"fp8_e4m3", "fp8_e5m2"} and scale_format in {"fp8_e8m0", "e8m0"}:
        return "mx"
    if payload == "int8":
        return "per_token_int8"
    raise ModelingError("mc2_init_routing_absorption_requires_mx_or_per_token_int8_payload")


def _received_quantizer(pack_quant, gate, dispatch_id, rows, width):
    """Keep receiver conversion when native dispatch publishes float rows."""
    from mindie_sim.catalog.ops import vector_op

    meta = dict(pack_quant.metadata)
    source_rows = float(meta.get("source_rows") or NO_ASSIGNMENTS)
    if source_rows <= NO_ASSIGNMENTS:
        raise ModelingError("mc2_pack_quantizer_requires_source_rows")
    scale = float(rows) / source_rows
    scale_shape = tuple(meta["activation_scale_shape"])
    _, *scale_tail = scale_shape
    meta.update(
        source_rows=rows,
        activation_scale_shape=(math.ceil(rows), *scale_tail),
        activation_scale_bytes=float(meta["activation_scale_bytes"]) * scale,
        quantization_scope="moe_received_expert_input",
    )
    quantizer = vector_op(
        op_id=dispatch_id + ".received_quantize", deps=(dispatch_id,),
        stage=pack_quant.stage, layer=pack_quant.layer,
        requests=pack_quant.requests, phase=pack_quant.phase,
        family=pack_quant.family, recipe=pack_quant.recipe,
        dtype=pack_quant.dtype,
        elements=math.ceil(rows) * width, ops_per_element=NO_ASSIGNMENTS,
        vector_ops_override=pack_quant.vector_ops * scale,
        read_bytes_override=pack_quant.mte2_bytes * scale,
        write_bytes_override=pack_quant.mte3_bytes * scale,
        shape=(math.ceil(rows), width), returns=pack_quant.returns,
        metadata=tuple(meta.items()),
    )
    gate_meta = dict(gate.metadata)
    if gate_meta.get("activation_format") != meta["output_format"]:
        raise ModelingError("mc2_received_quantizer_format_mismatch")
    gate_meta.update(
        activation_producer="explicit_routing_quantization",
        activation_payload_producer=quantizer.op_id,
        activation_scale_producer=quantizer.op_id,
    )
    return quantizer, replace(
        gate, deps=tuple(quantizer.op_id if dep == dispatch_id else dep
                         for dep in gate.deps),
        metadata=tuple(gate_meta.items()))


def select(kernels):
    raw = kernels.get("native_moe_protocol")
    if raw is None:
        return None
    allowed = {"kind", "max_inflight_per_rank", "dispatch_quant", "global_bs",
               "chunk_bytes", "hop_startup_us",
               "receive_capacity_bytes_per_peer"}
    if (not {"kind"} <= set(raw) or set(raw) - allowed
            or raw["kind"] not in {"arch35_fullmesh", "arch22_v2_caller"}):
        raise ModelingError("native_moe_protocol_requires_supported_caller")
    if raw["kind"] == "arch22_v2_caller" and raw.get("dispatch_quant") != "per_token_int8":
        raise ModelingError("arch22_v2_caller_requires_per_token_int8_dispatch")
    if "global_bs" in raw and (raw["kind"] != "arch22_v2_caller"
            or type(raw["global_bs"]) is not int or raw["global_bs"] < 0):
        raise ModelingError("arch22_v2_global_bs_requires_nonnegative_integer")
    if raw.get("dispatch_quant", "none") not in {"none", "mx", "per_token_int8"}:
        raise ModelingError("native_moe_protocol_dispatch_quant_requires_none_mx_or_per_token_int8")
    if raw["kind"] == "arch22_v2_caller" and any(
            raw.get(key) is not None for key in (
                "chunk_bytes", "hop_startup_us",
                "receive_capacity_bytes_per_peer")):
        raise ModelingError(
            "arch22_v2_caller_does_not_accept_chunked_direct_peer_options")
    # A full-mesh dispatch has one destination slot per peer.  When the
    # producer does not declare a finite window, the source-derived physical
    # bound is the actual remote peer count (resolved on the bound group), not
    # a guessed finite setting.  An explicit declaration is preserved.
    # Chunk size, protocol startup and receive credits are likewise explicit
    # T06-style service inputs; their absence remains an unknown.
    DirectPeerCollective(
        max_inflight_per_rank=raw.get("max_inflight_per_rank"),
        chunk_bytes=raw.get("chunk_bytes"),
        hop_startup_us=raw.get("hop_startup_us", 0.0),
        receive_capacity_bytes_per_peer=raw.get(
            "receive_capacity_bytes_per_peer"))
    return raw


def recipes(kernels):
    selected = select(kernels)
    if selected is not None and selected["kind"] == "arch22_v2_caller":
        from .mc2_arch22 import recipes as arch22_recipes
        return arch22_recipes(selected)
    # The native planner is registered independently of whether this runtime
    # declares the explicit kernel entry.  Ordinary selection may come from the
    # common MoeCommConfig, and the recipe must exist for that graph without
    # mutating the caller's runtime.  Registration is not selection.
    result = {RECIPE: PlannedRecipe(build)}
    if selected is not None:
        result[NETWORK] = DirectPeerCollective(
            max_inflight_per_rank=selected.get("max_inflight_per_rank"),
            chunk_bytes=selected.get("chunk_bytes"),
            hop_startup_us=selected.get("hop_startup_us", 0.0),
            receive_capacity_bytes_per_peer=selected.get(
                "receive_capacity_bytes_per_peer"))
    return result


def select_common(runtime, features=None, *, tokens, top_k, experts, ep):
    """Explicit native declaration first, otherwise the common selector.

    The ordinary path and the piecewise families consume the same
    ``MoeCommConfig`` projection of the actual deployment feature record.  No
    model name, measured trace, or local heuristic selects MC2.
    """
    kernels = getattr(runtime, "kernels", {}) or {}
    explicit = select(kernels)
    if explicit is not None:
        return explicit
    if EXPLICIT_ROUTING_LOWERINGS & set(kernels.get("lowerings") or {}):
        # An explicitly lowered routing/init kernel owns the expert input
        # representation; automatic MC2 selection must not replace it.
        return None
    rmeta = dict(getattr(runtime, "metadata", ()) or ())
    radditional = rmeta.get("additional_config")
    radditional = dict(radditional) if isinstance(radditional, dict) else {}
    has_runtime_moe_comm = bool(
        "enable_fused_mc2" in radditional
        or rmeta.get("moe_comm_mode")
        or rmeta.get("image")
        or rmeta.get("image_id")
    )
    if (features is None
            or not has_runtime_moe_comm
            or kernels.get("moe_transport")
            or bool({"expert_dispatch", "expert_combine"} & set(kernels.get("recipes") or ()))
            or getattr(runtime, "moe_recipe", "alltoall") != "alltoall"):
        return None
    config = MoeCommConfig.from_features(
        features, ep=ep, top_k=top_k, experts=experts)
    if config is None or config.select(max(0, int(tokens))) != MC2:
        return None
    return {"kind": "arch35_fullmesh", "max_inflight_per_rank": None,
            "chunk_bytes": None, "hop_startup_us": 0.0,
            "receive_capacity_bytes_per_peer": None,
            "selection": "moe_comm_config", "moe_comm": config.to_items()}


def lower(ops, *, runtime, prefix, tokens, top_k, width, experts, ep,
          source_partitions, received_assignments, dtype, source_routing=None,
          valid_tokens=None, features=None, source_rows_by_replica=None,
          destination_rows_by_rank=None):
    selected = select_common(
        runtime, features, tokens=tokens, top_k=top_k, experts=experts, ep=ep)
    if selected is None:
        return ops
    if selected["kind"] == "arch22_v2_caller":
        from .mc2_arch22 import lower as lower_arch22
        return lower_arch22(
            ops, selected=selected, runtime=runtime, prefix=prefix, tokens=tokens,
            top_k=top_k, width=width, experts=experts, ep=ep,
            source_partitions=source_partitions, received_assignments=received_assignments,
            dtype=dtype, source_routing=source_routing)
    if (source_partitions < 1 or ep < 2 or dtype not in {"bf16", "fp16"}
            or getattr(runtime, "moe_recipe", "alltoall") != "alltoall"):
        raise ModelingError("mc2_fullmesh_requires_unpartitioned_float_rows_and_ep_alltoall")
    if runtime.kernels.get("moe_transport"):
        raise ModelingError("mc2_fullmesh_cannot_duplicate_separate_quantized_transport")
    by_id = {op.op_id: op for op in ops}
    pack, dispatch, combine, unpack = (by_id[prefix + suffix]
                                      for suffix in (".pack", ".dispatch", ".combine", ".unpack"))
    pack_quant = by_id.get(prefix + ".pack.quantize")
    if pack_quant is not None and (
            pack_quant.plan_required
            or pack_quant.deps != (pack.op_id,)
            or dict(pack_quant.metadata).get("quantization_scope")
            != "moe_dispatch_expert_input"
            or any(pack_quant.op_id in item.deps and item.op_id != dispatch.op_id
                   for item in ops)):
        raise ModelingError("mc2_cannot_absorb_external_pack_quantizer")
    source_metadata = source_routing.to_dict() if source_routing is not None else None
    # TPxDP without an explicit slice still publishes one source rank's rows.
    if source_routing is not None:
        valid_source_tokens = source_routing.valid_local_tokens
    elif valid_tokens is not None:
        # The caller's actual logical rows, not a physical padding count.
        valid_source_tokens = valid_tokens
    else:
        valid_source_tokens = tokens
    if source_partitions > 1:
        tokens = math.ceil(tokens / source_partitions)
        if source_routing is None:
            valid_source_tokens = (valid_tokens / source_partitions
                                   if valid_tokens is not None else tokens)
    send = valid_source_tokens * top_k
    quant = selected.get("dispatch_quant", "none")
    # ``aclnnMoeInitRoutingV3`` may own the dispatch-input quantization
    # (``quant_mode=1``) instead of a separate ``pack.quantize`` node.  The
    # sender then publishes quantized rows and the receiver scales come from
    # that pack, so no received quantizer is synthesized here.
    absorbed = dict(pack.metadata).get("moe_init_routing")
    if absorbed is not None:
        if pack_quant is not None:
            raise ModelingError("mc2_init_routing_absorption_conflicts_with_pack_quantizer")
        absorbed_quant = _absorbed_quant_mode(absorbed)
        if quant not in {"none", absorbed_quant}:
            raise ModelingError("mc2_init_routing_absorption_conflicts_with_dispatch_quant")
        quant = absorbed_quant
    received_quant = None
    if pack_quant is not None and quant == "none" and received_assignments:
        gate = by_id.get(prefix + ".expert_up")
        if gate is None or gate.deps != (dispatch.op_id,):
            raise ModelingError("mc2_received_quantizer_requires_direct_expert")
        received_quant, by_id[gate.op_id] = _received_quantizer(
            pack_quant, gate, dispatch.op_id, received_assignments, width)
    payload_dtype = {"mx": "fp8_e4m3", "per_token_int8": "int8"}.get(quant, dtype)
    quant_mode = "per_token" if quant == "per_token_int8" else quant
    layouts = (dispatch_layout(width, input_dtype=dtype, payload_dtype=payload_dtype,
                               quant_mode=quant_mode),
               combine_layout(width, dtype=dtype))
    if quant in {"mx", "per_token_int8"}:
        if quant == "mx":
            mx32_work(width, dtype, payload_dtype)
        gate = by_id.get(prefix + ".expert_up")
        if gate is None:
            if received_assignments:
                raise ModelingError("mc2_quantized_requires_direct_expert_input_consumer")
        else:
            meta = dict(gate.metadata)
            # This dispatch owns exactly this consumer's input representation.
            # A transport selection cannot suppress an unrelated conversion.
            direct = (gate.deps == (dispatch.op_id,) and gate.local_shape[-1] == width
                      and meta.get("activation_format") == payload_dtype)
            if quant == "mx":
                matched = (meta.get("activation_group_size") == 32
                           and meta.get("activation_scale_format") == "fp8_e8m0"
                           and tuple(meta.get("activation_scale_shape", ()))[1:] == (width // 64, 2))
                requirement = "matching_e4m3_e8m0_group32"
            else:
                matched = (meta.get("activation_group_size") == width
                           and meta.get("activation_scale_format") == "fp32"
                           and tuple(meta.get("activation_scale_shape", ())) == (math.ceil(received_assignments),))
                requirement = "matching_int8_fp32_per_row"
            if (not direct or not matched
                    or meta.get("activation_scale_bytes") != received_assignments * layouts[0].scale_bytes):
                raise ModelingError(f"mc2_{quant}_expert_input_requires_{requirement}_producer")
            # The fused dispatch owns the absorbed init-routing quantization,
            # so it publishes both the quantized payload and its scales.
            meta.update(activation_producer="explicit_transport_quantization",
                        activation_payload_producer=dispatch.op_id,
                        activation_scale_producer=dispatch.op_id,
                        moe_transport_kind=("arch35_fullmesh_mx32" if quant == "mx"
                                            else "arch35_fullmesh_per_token_int8"))
            by_id[gate.op_id] = replace(gate, metadata=tuple(meta.items()))
        down = by_id.get(prefix + ".expert_down")
        if down is not None and dict(down.metadata).get("output_format") != dtype:
            raise ModelingError("mc2_quantized_combine_requires_declared_float_expert_return")
    row_format = {"forward_bytes": layouts[0].packet_bytes,
                  "reverse_bytes": layouts[1].packet_bytes}
    # An exact source force-EPLB ledger is a whole routed-row matrix.  It is
    # never synthesized from a statistical destination expectation.
    peer_matrix = None
    if source_routing is not None:
        cells = getattr(source_routing, "peer_assignments", None)
        if cells is not None:
            peer_matrix = tuple(tuple(int(value) for value in row) for row in cells)
    route_rank = (int(getattr(source_routing, "representative_rank", 0))
                  if source_routing is not None else 0)
    if peer_matrix is not None:
        payload_basis = source_routing.basis
    elif source_routing is not None:
        payload_basis = str(getattr(source_routing, "basis",
                                    "declared_peer_expectation"))
    else:
        payload_basis = "uniform_peer_payload"
    source_key = ("source_force_eplb" if source_routing is not None and
                  source_routing.basis == "source_force_eplb_masked_integer_counts"
                  else "source_token_mask")
    # A complete TP1 DP cohort exposes one declared logical source row count
    # per EP rank.  Use that declared policy to build the SAME expected
    # assignment matrix for dispatch and its transpose for combine.  For a
    # wider TP group the vector is DP-scoped, not EP-scoped, and is ignored.
    global_source_rows = None
    if (peer_matrix is None and source_routing is None
            and source_rows_by_replica is not None
            and len(source_rows_by_replica) == ep):
        global_source_rows = tuple(
            max(0, int(value)) for value in source_rows_by_replica)

    def raw_contract(kind, incoming, outgoing):
        raw = {"kind": kind, "width": width, "dtype": dtype,
               "source_tokens": tokens if kind == "dispatch" else valid_source_tokens,
               "top_k": top_k, "source_assignments": incoming,
               "received_assignments": outgoing, "experts": experts, "ep": ep,
               "payload_basis": payload_basis}
        if kind == "dispatch" and quant != "none":
            raw.update(dispatch_quant=quant_mode, payload_dtype=payload_dtype)
        layout = layouts[0] if kind == "dispatch" else layouts[1]
        if peer_matrix is not None:
            # One representation: the shared source->destination forward
            # matrix. route_reverse=True below is the combine transpose.
            raw.update(route_matrix=peer_matrix, route_rank=route_rank)
        elif (source_routing is not None
              and hasattr(source_routing, "payload_matrix")):
            # Declared statistical destination policy: expected directed bytes,
            # never integer frame counts presented as observed peer traffic.
            expected = source_routing.payload_matrix(
                layout.packet_bytes, reverse=kind == "combine")
            raw["route_matrix_bytes"] = tuple(
                tuple(float(value) for value in row) for row in expected)
        elif global_source_rows is not None:
            # A source-blind two-way expectation conditional on the SAME
            # source and destination marginals used by F3:
            #   E[N_ij | s,d] = s_i * d_j / sum(s).
            # The matrix includes the local diagonal; native transport still
            # omits the local wire.  K3's all-one destination marginal keeps
            # the signed 31 off-diagonal assignments; DS's 12 occupied and 4
            # empty F3 ranks give .5 per occupied edge instead of .75*16.
            source_values = tuple(float(rows * top_k)
                                  for rows in global_source_rows)
            total_source = sum(source_values)
            if (destination_rows_by_rank is not None
                    and len(destination_rows_by_rank) == ep):
                destination_values = tuple(
                    float(max(0, int(value)))
                    for value in destination_rows_by_rank)
            else:
                destination_values = (
                    (total_source / float(ep),) * ep
                    if total_source > 0 else (0.0,) * ep)
            matrix = []
            for source in range(ep):
                row = []
                for destination in range(ep):
                    # Dispatch is [original source, expert destination].
                    # Combine is its transpose: [expert source, original
                    # destination], re-framed at the combine packet width.
                    s_value = (source_values[destination]
                               if kind == "combine" else source_values[source])
                    d_value = (destination_values[source]
                               if kind == "combine"
                               else destination_values[destination])
                    expected = (s_value * d_value / total_source
                                if total_source > 0 else 0.0)
                    row.append(expected * float(layout.packet_bytes))
                matrix.append(tuple(row))
            raw["route_matrix_bytes"] = tuple(matrix)
            raw["moe_expected_pairing"] = (
                "conditional_source_destination_marginals"
                if destination_rows_by_rank is not None
                else "conditional_uniform_destination")
        else:
            # Uniform peer policy: one source's useful payload spread across
            # its declared peers.  The expectation is positive unless the
            # source row itself is a known zero.
            raw["uniform_payload_bytes"] = float(
                incoming * layout.packet_bytes)
        raw["route_reverse"] = kind == "combine"
        if incoming <= 0 and source_routing is None:
            raw["payload_basis"] = "actual_zero"
        return raw

    forward_raw = raw_contract("dispatch", send, received_assignments)
    backward_raw = raw_contract("combine", received_assignments, send)
    phase_names = {
        "dispatch": tuple(p.name for p in contract(forward_raw, 2).phases),
        "combine": tuple(p.name for p in contract(backward_raw, 2).phases),
    }
    dispatch_returns = dispatch.returns
    received_output = {}
    if quant != "none":
        # The native kernel completes after receiver materialization and flag
        # retirement. These are receiver rows, even for an idle sender;
        # original.local_shape still describes the source payload.
        rows = math.ceil(received_assignments)
        received_output = {
            "output_format": payload_dtype,
            "activation_payload_shape": (rows, width),
            "activation_group_size": 32 if quant == "mx" else width,
            "activation_scale_format": "fp8_e8m0" if quant == "mx" else "fp32",
            "activation_scale_shape": (rows, width // 64, 2) if quant == "mx" else (rows,),
            "activation_scale_bytes": received_assignments * layouts[0].scale_bytes,
            "activation_expected_rows": received_assignments,
        }
        if absorbed is None:
            dispatch_returns = tuple(dict.fromkeys((*dispatch_returns, "activation_scales")))

    def native_metadata(kind, raw, kernel, group):
        implementation = (
            {"mx": "arch35_fullmesh_mx32_bf16_e4m3",
             "per_token_int8": "arch35_fullmesh_per_token_int8"}.get(
                 quant, "arch35_fullmesh_unquantized_rows")
            if kind == "dispatch" else "arch35_fullmesh_unquantized_rows")
        metadata = {
            **(received_output if kind == "dispatch" else {}),
            **({source_key: source_metadata} if source_routing is not None else {}),
            "mc2_phase": raw,
            "mc2_phases": phase_names[kind],
            "group_axis": "ep", "steps": max(0, ep - 1),
            "device_kernel_group": {"id": group, "entry": True, "exit": True,
                                    "kernel": kernel},
            "implementation": implementation,
            "source": SOURCE, "native_kind": kernel,
            "invocation_class": "vector",
            **({"moe_comm_selection": dict(selected.get("moe_comm") or ())}
               if selected.get("selection") == "moe_comm_config" else {}),
            "peer_traffic_row_format": row_format,
            "source_assignment_partitions": source_partitions,
            "traffic_model": payload_basis,
            "transport_options": {
                "chunk_bytes": selected.get("chunk_bytes"),
                "hop_startup_us": selected.get("hop_startup_us", 0.0),
                "receive_capacity_bytes_per_peer": selected.get(
                    "receive_capacity_bytes_per_peer"),
                # The fused native phase scheduler owns its peer loop.  These
                # are declared for the direct-peer transport recipe and are
                # recorded here instead of being silently dropped.
                "application": "direct_peer_transport_recipe",
                "native_fused_phase_scheduler": (
                    "separate_owner; not_rescheduled_by_mc2_graph"),
            },
            "coverage": "source_protocol_with_declared_row_grid_and_peer_window",
        }
        if kind == "combine":
            metadata["transpose_of_source_assignments"] = True
        if kind == "dispatch" and absorbed is not None:
            # ``MoeDistributeDispatchV3`` owns the permutation *and* the
            # ``aclnnMoeInitRoutingV3`` dynamic quantization (``quant_mode=1``)
            # in one native call: no separate pack quantizer remains.
            metadata["moe_init_routing"] = dict(absorbed)
            metadata["fused_native_kinds"] = ("MoeDistributeDispatchV3",
                                              absorbed["native_kind"])
        return tuple(metadata.items())

    dispatch_op = Work().apply(replace(
        dispatch, op_id=dispatch.op_id, deps=pack.deps,
        family="mc2_protocol", recipe=RECIPE, source_boundary="kernel",
        dtype=dtype, returns=dispatch_returns,
        metadata=native_metadata(
            "dispatch", forward_raw, "MoeDistributeDispatchV3",
            prefix + ".mc2_dispatch")))
    # The source reduction retains the actual weight producer dependency while
    # replacing generic payload reads, arithmetic and final output publication.
    final_deps = (dispatch.op_id,) + tuple(
        d for d in unpack.deps if d not in {pack.op_id, combine.op_id})
    combine_deps = tuple(dict.fromkeys(combine.deps + final_deps))
    combine_op = Work().apply(replace(
        combine, op_id=unpack.op_id, deps=combine_deps,
        family="mc2_protocol", recipe=RECIPE, source_boundary="kernel",
        dtype=dtype,
        metadata=native_metadata(
            "combine", backward_raw, "MoeDistributeCombineV3",
            prefix + ".mc2_combine")))
    # One native call replaces its semantic phase operators; the original
    # pack/dispatch/combine/unpack aliases remain connected through the entry
    # and exit ids, never through duplicated scheduling.
    redirect = {pack.op_id: dispatch.op_id, combine.op_id: unpack.op_id}
    result = []
    for op in ops:
        if op.op_id == pack.op_id:
            result.append(dispatch_op)
            if received_quant is not None:
                result.append(received_quant)
            continue
        if op.op_id == unpack.op_id:
            result.append(combine_op)
            continue
        if (op.op_id in {dispatch.op_id, combine.op_id}
                or (pack_quant is not None and op.op_id == pack_quant.op_id)):
            continue
        current = by_id[op.op_id]
        deps = tuple(redirect.get(dependency, dependency)
                     for dependency in current.deps)
        if deps != current.deps:
            current = replace(current, deps=deps)
        result.append(current)
    return result
