"""Source-selected A5 MC2 dispatch/return phases in the ordinary MoE graph.

[Role] Replace generic packing and all-to-all with framed fullmesh channels.
[Invariants] One native kernel group owns each dispatch/return invocation.
  Count/status and payload have independent peer messages. Combine reverses
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

NETWORK = "arch35_fullmesh_peer_channel"


def select(kernels):
    raw = kernels.get("native_moe_protocol")
    if raw is None:
        return None
    if (not {"kind", "max_inflight_per_rank"} <= set(raw)
            or set(raw) - {"kind", "max_inflight_per_rank", "dispatch_quant", "global_bs"}
            or raw["kind"] not in {"arch35_fullmesh", "arch22_v2_caller"}):
        raise ModelingError("native_moe_protocol_requires_supported_caller_and_explicit_peer_window")
    if raw["kind"] == "arch22_v2_caller" and raw.get("dispatch_quant") != "per_token_int8":
        raise ModelingError("arch22_v2_caller_requires_per_token_int8_dispatch")
    if "global_bs" in raw and (raw["kind"] != "arch22_v2_caller"
            or type(raw["global_bs"]) is not int or raw["global_bs"] < 0):
        raise ModelingError("arch22_v2_global_bs_requires_nonnegative_integer")
    if raw.get("dispatch_quant", "none") not in {"none", "mx", "per_token_int8"}:
        raise ModelingError("native_moe_protocol_dispatch_quant_requires_none_mx_or_per_token_int8")
    DirectPeerCollective(max_inflight_per_rank=raw["max_inflight_per_rank"])
    return raw


def recipes(kernels):
    selected = select(kernels)
    if selected is not None and selected["kind"] == "arch22_v2_caller":
        from .mc2_arch22 import recipes as arch22_recipes
        return arch22_recipes(selected)
    return {} if selected is None else {
        RECIPE: PlannedRecipe(build), NETWORK: DirectPeerCollective(
            max_inflight_per_rank=selected["max_inflight_per_rank"]),
    }


def lower(ops, *, runtime, prefix, tokens, top_k, width, experts, ep,
          source_partitions, received_assignments, dtype, source_routing=None):
    selected = select(getattr(runtime, "kernels", {}) or {})
    if selected is None:
        return ops
    if selected["kind"] == "arch22_v2_caller":
        from .mc2_arch22 import lower as lower_arch22
        return lower_arch22(
            ops, selected=selected, runtime=runtime, prefix=prefix, tokens=tokens,
            top_k=top_k, width=width, experts=experts, ep=ep,
            source_partitions=source_partitions, received_assignments=received_assignments,
            dtype=dtype, source_routing=source_routing)
    if (source_partitions != 1 or ep < 2 or dtype not in {"bf16", "fp16"}
            or getattr(runtime, "moe_recipe", "alltoall") != "alltoall"):
        raise ModelingError("mc2_fullmesh_requires_unpartitioned_float_rows_and_ep_alltoall")
    if runtime.kernels.get("moe_transport"):
        raise ModelingError("mc2_fullmesh_cannot_duplicate_separate_quantized_transport")
    by_id = {op.op_id: op for op in ops}
    pack, dispatch, combine, unpack = (by_id[prefix + suffix]
                                      for suffix in (".pack", ".dispatch", ".combine", ".unpack"))
    source_metadata = source_routing.to_dict() if source_routing is not None else None
    valid_source_tokens = tokens if source_routing is None else source_routing.valid_local_tokens
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
    dispatch_network = dispatch.op_id + ".payload_network"

    def make(kind, original, incoming, outgoing, entry, exit_id, final_deps=()):
        group = prefix + ".mc2_" + kind
        kernel = "MoeDistributeDispatchV3" if kind == "dispatch" else "MoeDistributeCombineV3"
        raw = {"kind": kind, "width": width, "dtype": dtype,
               "source_tokens": tokens if kind == "dispatch" else valid_source_tokens,
               "top_k": top_k, "source_assignments": incoming,
               "received_assignments": outgoing, "experts": experts, "ep": ep}
        if kind == "dispatch" and quant != "none":
            raw.update(dispatch_quant=quant_mode, payload_dtype=payload_dtype)
        # Dependencies/names do not depend on physical AIV count. The actual
        # work planner obtains that count from Geometry at execution/export.
        plan = contract(raw, 2)
        ids = {p.name: group + "." + p.name for p in plan.phases}
        ids[plan.phases[0].name] = entry
        ids[plan.phases[-1].name] = exit_id
        result = []
        network_ids = {"payload_exchange": original.op_id + ".payload_network"}
        if kind == "dispatch":
            network_ids["count_status_exchange"] = original.op_id + ".count_network"

        def info(is_entry=False, is_exit=False):
            return {"id": group, "entry": is_entry, "exit": is_exit, "kernel": kernel}

        for index, phase in enumerate(plan.phases):
            deps = tuple(ids[name] for name in phase.after) or original.deps
            deps += tuple(network_ids[name] for name in phase.after if name in network_ids)
            if index == len(plan.phases) - 1:
                deps = tuple(dict.fromkeys(deps + final_deps))
            received_output = {}
            returns = original.returns
            if kind == "dispatch" and quant != "none" and index == len(plan.phases) - 1:
                # The native kernel completes after receiver materialization and
                # flag retirement. These are receiver rows, even for an idle
                # sender; original.local_shape still describes source payload.
                rows = math.ceil(outgoing)
                received_output = {
                    "output_format": payload_dtype,
                    "activation_payload_shape": (rows, width),
                    "activation_group_size": 32 if quant == "mx" else width,
                    "activation_scale_format": "fp8_e8m0" if quant == "mx" else "fp32",
                    "activation_scale_shape": (rows, width // 64, 2) if quant == "mx" else (rows,),
                    "activation_scale_bytes": outgoing * plan.layout.scale_bytes,
                    "activation_expected_rows": outgoing,
                }
                returns = tuple(dict.fromkeys((*returns, "activation_scales")))
            phase_op = Work().apply(replace(
                original, op_id=ids[phase.name], deps=deps, family="mc2_protocol",
                recipe=RECIPE, source_boundary="kernel", dtype=dtype, returns=returns,
                stream=(group + ".counts" if "count" in phase.name else original.stream),
                metadata=tuple({
                    **received_output,
                    **({("source_force_eplb" if source_routing.basis ==
                         "source_force_eplb_masked_integer_counts" else "source_token_mask"):
                        source_metadata} if source_routing else {}),
                    "mc2_phase": {**raw, "phase": phase.name},
                    "device_kernel_group": info(index == 0, index == len(plan.phases) - 1),
                    "implementation": ({"mx": "arch35_fullmesh_mx32_bf16_e4m3",
                                        "per_token_int8": "arch35_fullmesh_per_token_int8"}.get(
                                            quant, "arch35_fullmesh_unquantized_rows")
                                       if kind == "dispatch" else "arch35_fullmesh_unquantized_rows"),
                    "source": SOURCE, "native_kind": kernel,
                    "invocation_class": "vector",
                    "coverage": "source_protocol_with_declared_row_grid_and_peer_window",
                }.items()),
            ))
            result.append(phase_op)
            if phase.name in network_ids:
                counts = phase.name == "count_status_exchange"
                volume = experts * 32 if counts else incoming * plan.layout.packet_bytes
                channel_meta = {
                    "group_axis": "ep", "steps": max(0, ep - 1),
                    "collective_id": network_ids[phase.name],
                    "device_kernel_group": info(),
                    "native_kind": kernel,
                    "channel": "count_status" if counts else "framed_payload",
                    "source_assignment_partitions": 1,
                    "traffic_model": "uniform_peer_payload",
                }
                if not counts:
                    channel_meta["peer_traffic_row_format"] = row_format
                    if kind == "combine":
                        channel_meta["reverse_of_collective"] = dispatch_network
                link = volume * (ep - 1) / ep
                family = "alltoall"
                if not counts and source_routing is not None:
                    matrix = source_routing.payload_matrix(
                        plan.layout.packet_bytes, reverse=kind == "combine")
                    channel_meta.pop("reverse_of_collective", None)
                    channel_meta.update(
                        traffic_model=("source_masked_integer_assignments" if source_routing.basis ==
                                       "source_force_eplb_masked_integer_counts" else source_routing.basis),
                        peer_volume_basis="group_mean",
                        peer_traffic={"kind": "matrix", "payload_bytes": matrix},
                        transpose_of_source_assignments=(kind == "combine"))
                    channel_meta[("source_force_eplb" if source_routing.basis ==
                                  "source_force_eplb_masked_integer_counts" else
                                  "source_token_mask")] = source_metadata
                    link = sum(map(sum, matrix)) / ep
                    family = "alltoallv"
                network = Work(resources={"link": link}).apply(replace(
                    original, op_id=network_ids[phase.name], deps=tuple(ids[n] for n in phase.after),
                    family=family, recipe=NETWORK, source_boundary="",
                    stream=group + (".counts_net" if counts else ".payload_net"),
                    metadata=tuple(channel_meta.items())))
                result.append(network)
        return result, ids[plan.phases[-1].name], info

    forward, _, _ = make("dispatch", replace(dispatch, deps=pack.deps),
                         send, received_assignments, pack.op_id, dispatch.op_id)
    # The source reduction retains the actual weight producer dependency while
    # replacing generic payload reads, arithmetic and final output publication.
    final_deps = (dispatch.op_id,) + tuple(
        d for d in unpack.deps if d not in {pack.op_id, combine.op_id})
    backward, _, _ = make("combine", combine, received_assignments, send,
                          combine.op_id, unpack.op_id, final_deps)
    replacement = {pack.op_id: forward, dispatch.op_id: (),
                   combine.op_id: backward, unpack.op_id: ()}
    return [item for op in ops for item in replacement.get(op.op_id, (by_id[op.op_id],))]
