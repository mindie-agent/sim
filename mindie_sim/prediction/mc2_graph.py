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


def select(kernels):
    raw = kernels.get("native_moe_protocol")
    if raw is None:
        return None
    if (not {"kind"} <= set(raw)
            or set(raw) - {"kind", "max_inflight_per_rank", "dispatch_quant", "global_bs"}
            or raw["kind"] not in {"arch35_fullmesh", "arch22_v2_caller"}):
        raise ModelingError("native_moe_protocol_requires_supported_caller")
    if raw["kind"] == "arch22_v2_caller" and raw.get("dispatch_quant") != "per_token_int8":
        raise ModelingError("arch22_v2_caller_requires_per_token_int8_dispatch")
    if "global_bs" in raw and (raw["kind"] != "arch22_v2_caller"
            or type(raw["global_bs"]) is not int or raw["global_bs"] < 0):
        raise ModelingError("arch22_v2_global_bs_requires_nonnegative_integer")
    if raw.get("dispatch_quant", "none") not in {"none", "mx", "per_token_int8"}:
        raise ModelingError("native_moe_protocol_dispatch_quant_requires_none_mx_or_per_token_int8")
    # A full-mesh dispatch has one destination slot per peer.  When the
    # producer does not declare a finite window, the source-derived physical
    # bound is the actual remote peer count (resolved on the bound group), not
    # a guessed finite setting.  An explicit declaration is preserved.
    DirectPeerCollective(max_inflight_per_rank=raw.get("max_inflight_per_rank"))
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
            max_inflight_per_rank=selected.get("max_inflight_per_rank"))
    return result


def select_common(runtime, features=None, *, tokens, top_k, experts, ep):
    """Explicit native declaration first, otherwise the common selector.

    The ordinary path and the piecewise families consume the same
    ``MoeCommConfig`` projection of the actual deployment feature record.  No
    model name, measured trace, or local heuristic selects MC2.
    """
    explicit = select(getattr(runtime, "kernels", {}) or {})
    if explicit is not None:
        return explicit
    if features is None:
        return None
    config = MoeCommConfig.from_features(
        features, ep=ep, top_k=top_k, experts=experts)
    if config is None or config.select(max(0, int(tokens))) != MC2:
        return None
    return {"kind": "arch35_fullmesh", "max_inflight_per_rank": None,
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
            "coverage": "source_protocol_with_declared_row_grid_and_peer_window",
        }
        if kind == "combine":
            metadata["transpose_of_source_assignments"] = True
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
            continue
        if op.op_id == unpack.op_id:
            result.append(combine_op)
            continue
        if op.op_id in {dispatch.op_id, combine.op_id}:
            continue
        current = by_id[op.op_id]
        deps = tuple(redirect.get(dependency, dependency)
                     for dependency in current.deps)
        if deps != current.deps:
            current = replace(current, deps=deps)
        result.append(current)
    return result
