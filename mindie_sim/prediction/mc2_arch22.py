"""A3 DispatchV2/CombineV2 caller and typed producer contract.

[Role] Keep INT8 payload, row scales, combine indices and weighted return inside
  their real native calls instead of separate pack/quantize/unpack invocations.
[Invariants] arch22 ABI does not identify fullmesh/basic/hierarchy tiling. Local
  work is an explicit semantic lower bound and peer concurrency is declared;
  no arch35 instruction/phase ledger or inferred polling cycles is reused.
  A complete-call response covers the native group once, including transport.
[Timeouts & Locks] O(local graph + peers) immutable descriptions, no locks/I/O.
[Side-Effect Policy] No device, binary, calibration or external state access.
"""
from dataclasses import replace
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .direct_peer import DirectPeerCollective
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, Stage

LOCAL = "arch22_v2_caller_local"
NETWORK = "arch22_v2_declared_peer_channel"
SOURCE = {
    "repository": "cann/ops-transformer",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "dispatch_entry": "mc2/moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_dispatch_v2_a3.cpp",
    "combine_entry": "mc2/moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2_a3.cpp",
    "dispatch_entry_sha256": "4b0ee2ecdb545444436caca4f4750e837f00fd8e841f96dc3822e11e4c09f4d7",
    "combine_entry_sha256": "d2c94c3be1da9568aaccc81c70fedc88ce398628aa323eca56a3d1470aa8cf59",
    "quant_header_sha256": "46fec97abaec206c9b41bd1b3ab690d7c0ff7fe7a09961599b8c56f304543ed5",
    "dispatch_contract": "BF16/FP16 X and IDs/mask -> INT8 expandX, FP32 dynamicScales, INT32 assistInfo, counts",
    "combine_contract": "BF16/FP16 expandX, assistInfo, counts, source IDs/mask, FP32 weights and optional expandScales -> weighted X",
    "selected_inner_tiling": "unresolved_fullmesh_basic_or_hierarchy",
    "binary_match": "not_established_by_source_reference",
}


def recipes(selected):
    return {LOCAL: PlannedRecipe(build), NETWORK: DirectPeerCollective(
        max_inflight_per_rank=selected["max_inflight_per_rank"])}


def build(op, geometry):
    """Price real caller work without inventing an unidentified core schedule."""
    raw = dict(op.metadata)["mc2_v2_local"]
    work = Work(resources=raw["work"])
    units = geometry.resource_units.get("vector", geometry.compute_units)
    # One row is the smallest independently owned unit in this boundary model.
    # Exact native owner partition/poll counts require the captured tiling key.
    active = min(units, max(1, math.ceil(raw["rows"])))

    def program(context):
        return KernelProgram(tuple(Stage(
            f"owner{i}", context.bind(work.scaled(1 / active)),
            placement=context.owned(i, units, work.resources)) for i in range(active)))

    return OperatorPlan(work, program, {
        "execution_class": "vector", "active_owner_count": active,
        "owner_basis": "declared_semantic_rows; native_tiling_not_identified",
        "coverage": "native_caller_ABI_plus_logical_payload_and_local_work_lower_bound",
        "unmodeled_services": (
            "branch-specific framing, intermediate windows and count replication",
            "scalar addressing, synchronization and failed polling rounds",
            "source owner partition and row-level overlap"),
    })


def _typed_expert(gate, dispatch_id, width, rows, capacity):
    meta = dict(gate.metadata)
    if (gate.deps != (dispatch_id,) or gate.local_shape[-1] != width
            or meta.get("activation_format") != "int8"
            or meta.get("activation_group_size") != width
            or meta.get("activation_scale_format") != "fp32"
            or tuple(meta.get("activation_scale_shape", ())) != (math.ceil(rows),)
            or meta.get("activation_scale_bytes") != rows * 4):
        raise ModelingError("arch22_v2_expert_requires_direct_INT8_FP32_row_scale_consumer")
    meta.update(activation_producer="explicit_transport_quantization",
                activation_payload_producer=dispatch_id,
                activation_scale_producer=dispatch_id,
                moe_transport_kind="arch22_v2_per_token_int8",
                native_input_capacity_shape=(capacity, width),
                native_input_scale_capacity_shape=(capacity,),
                native_output_capacity_shape=(capacity, gate.local_shape[1]))
    return replace(gate, metadata=tuple(meta.items()))


def lower(ops, *, selected, runtime, prefix, tokens, top_k, width, experts, ep,
          source_partitions, received_assignments, dtype, source_routing=None):
    if (source_partitions != 1 or ep < 2 or experts % ep or dtype not in {"bf16", "fp16"}
            or runtime.moe_recipe != "alltoall"):
        raise ModelingError("arch22_v2_requires_float_rows_and_unpartitioned_ep_alltoall")
    if runtime.kernels.get("moe_transport"):
        raise ModelingError("arch22_v2_cannot_duplicate_separate_transport_quantizer")
    by_id = {op.op_id: op for op in ops}
    pack, dispatch, combine, unpack = (by_id[prefix + suffix]
        for suffix in (".pack", ".dispatch", ".combine", ".unpack"))
    # InferShape: a = globalBsReal * min(localExpertNum, topK). Capacity is
    # neither an observed group_list nor permission to load all padded rows.
    global_bs = selected.get("global_bs", 0) or tokens * ep
    capacity = global_bs * min(experts // ep, top_k)
    if received_assignments > capacity:
        raise ModelingError("arch22_v2_received_rows_exceed_capacity_declare_global_bs")
    gate, down = by_id.get(prefix + ".expert_up"), by_id.get(prefix + ".expert_down")
    if gate is not None:
        by_id[gate.op_id] = _typed_expert(gate, dispatch.op_id, width, received_assignments, capacity)
    elif received_assignments:
        raise ModelingError("arch22_v2_requires_expert_input_consumer")
    if down is not None and dict(down.metadata).get("output_format") != dtype:
        raise ModelingError("arch22_v2_requires_float_expert_return")
    for name in (".expert_act", ".expert_down"):
        op = by_id.get(prefix + name)
        if op is not None:
            m = dict(op.metadata)
            m.update(native_output_capacity_shape=(capacity, width if name == ".expert_down"
                                                   else op.local_shape[-1]))
            by_id[op.op_id] = replace(op, metadata=tuple(m.items()))
    valid = tokens if source_routing is None else source_routing.valid_local_tokens
    assignments, received = valid * top_k, received_assignments
    float_bytes = int(width * dtype_bytes(dtype))
    forward_row, reverse_row = width + 4 + 12, float_bytes
    source_meta = source_routing.to_dict() if source_routing is not None else None
    def tensor(shape, dtype, **extra):
        return {"shape": shape, "dtype": dtype, "layout": "ND", **extra}
    abi_outputs = (
        tensor((capacity,width), "int8", quantity="expand_x"),
        tensor((capacity,), "fp32", quantity="dynamic_scales"),
        tensor((capacity*128,), "int32", quantity="assist_info"),
        tensor((experts//ep,), "int64", quantity="expert_token_nums"),
        tensor((experts,), "int32", quantity="ep_recv_counts"),
        tensor((1,), "int32", quantity="tp_recv_counts"),
        tensor((capacity,), "fp32", quantity="expand_scales",
               value_semantics="allocated_optional; valid_content_not_established"))

    def native(kind, original, entry_id, exit_id, deps, final_deps):
        dispatching = kind == "dispatch"
        kernel = "MoeDistributeDispatchV2" if dispatching else "MoeDistributeCombineV2"
        group = prefix + ".mc2_v2_" + kind
        network_id = original.op_id + ".payload_network"
        incoming, outgoing = ((assignments, received) if dispatching else (received, assignments))
        row_bytes = forward_row if dispatching else reverse_row
        # Minimal ABI work: one source-row quantization, then routed row
        # publication. Native fullmesh may quantize again per expert assignment;
        # that difference is not inferred without selecting that branch.
        if dispatching:
            read = valid * float_bytes + tokens * top_k * 4 + tokens
            entry_work = {"hbm": read, "mte2_aiv": read,
                          "vector": valid * (7 * width - 1)}
            published = outgoing * forward_row + experts / ep * 8 + experts * 4
            exit_work = {"hbm": outgoing * forward_row + published,
                         "mte2_aiv": outgoing * forward_row, "mte3": published}
            outputs = {"output_format": "int8",
                "activation_payload_shape": (math.ceil(received), width),
                "activation_group_size": width, "activation_scale_format": "fp32",
                "activation_scale_shape": (math.ceil(received),),
                "activation_scale_bytes": received * 4, "activation_expected_rows": received,
                "assist_info_shape": (math.ceil(received), 3), "assist_info_format": "int32",
                "expert_token_nums_shape": (experts // ep,), "expert_token_nums_format": "int64",
                "ep_send_counts_shape": (experts,), "ep_send_counts_format": "int32"}
            outputs.update(native_output_capacity_rows=capacity, native_abi_outputs=abi_outputs,
                           expand_scales_buffer_shape=(capacity,), expand_scales_buffer_dtype="fp32",
                           expand_scales_value_semantics="allocated_optional; valid_content_not_established")
            returns = ("activation_scales", "assist_info", "expert_token_nums", "ep_send_counts",
                       "tp_recv_counts", "expand_scales")
        else:
            read = received * (float_bytes + 12) + experts * 4
            entry_work = {"hbm": read, "mte2_aiv": read}
            read_return = assignments * (float_bytes + 4)
            output = valid * float_bytes
            exit_work = {"hbm": read_return + output, "mte2_aiv": read_return,
                         "mte3": output, "vector": valid * width * (3 * top_k + 1)}
            outputs, returns = {"output_format": dtype, "output_shape": (tokens, width)}, ()
        contract = {
            "kind": kind, "source_tokens": tokens, "valid_source_tokens": valid,
            "top_k": top_k, "width": width, "experts": experts, "ep": ep,
            "received_assignments": received, "input_dtype": dtype,
            "native_output_capacity_rows": capacity, "global_bs_real": global_bs,
            "dispatch_quant": "per_token_int8", "source_mask": source_meta,
            "transport": "declared_direct_peer_logical_payload",
            "framing": "not_inferred_from_operator_type",
        }

        def metadata(entry=False, exit=False):
            return {"device_kernel_group": {"id": group, "entry": entry, "exit": exit,
                                             "kernel": kernel},
                    "native_kind": kernel, "implementation": "arch22_v2_caller",
                    "source": SOURCE, "mc2_v2_caller": contract,
                    "invocation_class": "vector"}

        def local(op_id, after, work, rows, *, entry=False, exit=False):
            extra = outputs if exit else {}
            shape = ((tokens, width) if dispatching == entry else
                     (math.ceil(received), width))
            md = {**metadata(entry, exit), **extra,
                  "mc2_v2_local": {"work": work, "rows": rows}}
            if dispatching:
                md.update(native_abi_outputs=abi_outputs,
                          native_input_capacity_shape=(tokens,width))
            else:
                md.update(native_input_capacity_shape=(capacity,width),
                    assist_info_producer=dispatch.op_id, ep_counts_producer=dispatch.op_id,
                    expand_scales_buffer_producer=dispatch.op_id,
                    expand_scales_buffer_shape=(capacity,),
                    expand_scales_value_semantics="optional; distinct_from_dynamic_scales",
                    router_weight_producers=tuple(d for d in unpack.deps
                        if d not in {pack.op_id, combine.op_id}))
            return Work().apply(replace(original, op_id=op_id, deps=after,
                family="mc2_protocol", recipe=LOCAL, source_boundary="kernel", dtype=dtype,
                local_shape=shape, semantic_shape=shape,
                padded_shape=shape, launch_shape=shape,
                returns=returns if exit else (), metadata=tuple(md.items())))

        entry = local(entry_id, deps, entry_work, valid if dispatching else received, entry=True)
        channel_meta = {**metadata(), "group_axis": "ep", "steps": ep - 1,
            "collective_id": network_id, "source_assignment_partitions": 1,
            "traffic_model": "uniform_peer_payload", "channel": "logical_payload",
            "peer_traffic_row_format": {"forward_bytes": forward_row, "reverse_bytes": reverse_row}}
        if not dispatching:
            channel_meta["reverse_of_collective"] = dispatch.op_id + ".payload_network"
        link, family = incoming * row_bytes * (ep - 1) / ep, "alltoall"
        if source_routing is not None:
            matrix = source_routing.payload_matrix(row_bytes, reverse=not dispatching)
            channel_meta.pop("reverse_of_collective", None)
            channel_meta.update(peer_volume_basis="group_mean", traffic_model=source_routing.basis,
                                peer_traffic={"kind": "matrix", "payload_bytes": matrix},
                                transpose_of_source_assignments=not dispatching)
            link, family = sum(map(sum, matrix)) / ep, "alltoallv"
        network = Work(resources={"link": link}).apply(replace(original,
            op_id=network_id, deps=(entry_id,), recipe=NETWORK, family=family,
            source_boundary="", stream=group + ".network", metadata=tuple(channel_meta.items())))
        dependencies = (network_id,) + final_deps
        result = [entry, network]
        if dispatching:
            # Count outputs are a separate all-to-all channel in the source ABI.
            # This prices only the logical INT32 counts, not padded status slots.
            count_id = original.op_id + ".count_network"
            counts_meta = {**metadata(), "group_axis": "ep", "steps": ep - 1,
                "collective_id": count_id, "source_assignment_partitions": 1,
                "traffic_model": "uniform_peer_payload", "channel": "logical_counts"}
            count = Work(resources={"link": experts * 4 * (ep - 1) / ep}).apply(replace(
                original, op_id=count_id, deps=(entry_id,), recipe=NETWORK,
                family="alltoall", source_boundary="", stream=group + ".counts",
                metadata=tuple(counts_meta.items())))
            result.append(count)
            dependencies += (count_id,)
        result.append(local(exit_id, tuple(dict.fromkeys(dependencies)), exit_work,
                            outgoing if dispatching else valid, exit=True))
        return result

    forward = native("dispatch", dispatch, pack.op_id, dispatch.op_id, pack.deps, ())
    weights = tuple(d for d in unpack.deps if d not in {pack.op_id, combine.op_id})
    backward = native("combine", combine, combine.op_id, unpack.op_id,
                      tuple(dict.fromkeys(combine.deps + (dispatch.op_id,) + weights)), ())
    replacement = {pack.op_id: forward, dispatch.op_id: (),
                   combine.op_id: backward, unpack.op_id: ()}
    return [item for op in ops for item in replacement.get(op.op_id, (by_id[op.op_id],))]
