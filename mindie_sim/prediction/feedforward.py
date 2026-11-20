"""Dense and routed feed-forward programs.

[Role] Describe router, packing, dispatch, expert matrices, combine and shared work.
[Invariants] Routing conserves top-k assignments; zero-token experts do not read
  matrices. Expected routing is a declared probability model, not a measured
  efficiency. Expert and tensor parallelism shard different dimensions.
[Timeouts & Locks] O(experts) arithmetic per layer, no histories or I/O.
[Side-Effect Policy] Constructs operator records only.
"""
from __future__ import annotations

from dataclasses import replace
import math

from mindie_sim.catalog.ops import gemm_op, vector_op, comm_op, barrier_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from .gated_activation import gated_activation, declare_situ_quant
from .deepep import (deepep_exchange_factor, resolve_exchange_options,
                     scale_exchange_operator, split_deepep_combine_overlap,
                     split_deepep_dispatch_overlap, split_deepep_down_overlap,
                     split_deepep_up_overlap)
from .moe_callers import RoutingPolicy, local_expert_reduce_recipe, tp_join_reduce
from .moe_cann_fusion import (
    absorb_init_routing, build_selection, cann_native_active,
    dispatch_quantizes_internally,
)
from .passes_ffn_fusion import (
    bind_expert_slots, bind_transport_gate, declare_gate_up_fusion_group,
    routed_expert_work, router_gemm,
)
from . import deepep, moe_transport, moe_token_partition

ZERO = 0
ONE = 1
TWO = 2
INVERSE_INDEX_DTYPE = "int32"


def _latent_norm(prefix, source, stage, plan, requests, tokens, latent, model, mode):
    return vector_op(
        op_id=prefix + ".latent_norm", deps=(source,), stage=stage, layer=plan.index,
        requests=requests, elements=tokens * latent, ops_per_element=5,
        dtype=model.activation_dtype, family="rmsnorm", phase=mode, recipe="latent_norm",
        shape=(tokens, latent),
        read_bytes_override=(tokens + 1) * latent * dtype_bytes(model.activation_dtype),
        write_bytes_override=tokens * latent * dtype_bytes(model.activation_dtype),
        metadata=(("reduction_width", latent), ("norm_has_weight", True),
                  ("norm_eps", model.norm_eps),
                  ("latent_norm_position", plan.expert_latent_norm_position)))


def dense(*, prefix, plan, deps, stage, requests, tokens, model, parallel, mode,
          category="dense_mlp", runtime=None, reduce_output=True, **_):
    # A declared selector is configuration even when this FFN is dense. Validate
    # it here so an unsupported declaration cannot sit silently unused.
    moe_transport.select(runtime)
    width = math.ceil(plan.intermediate_size / parallel.tp)
    gate = gemm_op(
        op_id=prefix + ".gate_up", deps=(deps,), stage=stage, layer=plan.index,
        requests=requests, m=tokens, n=2 * width, k=model.hidden_size,
        quant=model.quant, category=category, phase=mode,
        family="gate_up_projection", recipe="dense_gate_up",
        metadata=(("ffn_role", "gate_up"), ("ffn_category", category)),
    )
    activation = gated_activation(
        plan=plan, rows=tokens, width=width,
        op_id=prefix + (".situ" if plan.ffn_activation == "situ" else ".swiglu"),
        deps=(gate.op_id,), stage=stage,
        requests=requests, dtype=model.activation_dtype, phase=mode, recipe="swiglu",
    )
    activation = replace(activation, metadata=activation.metadata + (
        ("ffn_role", "activation"), ("ffn_category", category)))
    down = gemm_op(
        op_id=prefix + ".down", deps=(activation.op_id,), stage=stage, layer=plan.index,
        requests=requests, m=tokens, n=model.hidden_size, k=width,
        quant=model.quant, category=category, phase=mode,
        family="down_projection", recipe="dense_down",
        metadata=(("ffn_role", "down"), ("ffn_category", category)),
    )
    ops, last = [gate, activation, down], down.op_id
    declare_situ_quant(ops, key=prefix + ".situ_quant", activation=activation,
                       consumer=down, runtime=runtime, rows=tokens, width=width)
    if parallel.tp > 1 and reduce_output:
        reduction = comm_op(
            op_id=prefix + ".tp", deps=(last,), stage=stage, layer=plan.index,
            requests=requests, family="allreduce", phase=mode, recipe="ring_allreduce",
            link_bytes=2 * (parallel.tp - 1) / parallel.tp * tokens * model.hidden_size
                       * dtype_bytes(model.activation_dtype), shape=(tokens, model.hidden_size),
            metadata=(("group_axis", "tp"), ("steps", 2 * (parallel.tp - 1))),
        )
        ops.append(reduction)
        last = reduction.op_id
    return ops, last


def alltoall(*, prefix, plan, deps, stage, requests, tokens, model, parallel, mode,
        global_tokens=None, routing=RoutingPolicy(), routing_realization=None,
        overlap_shared=False, runtime=None, reduce_after_join=False, sequence_plan=None,
        caller=None, sp_tokens=None, features=None,
        source_rows_by_replica=None, destination_rows_by_rank=None,
        ep_owner=None, price_owner=None, **_):
    pre_sp_tokens = tokens
    if sp_tokens is not None:
        tokens = sp_tokens
    exchange = caller is None or caller.exchange
    reduce_after_join = reduce_after_join or (caller is not None and caller.reduce_after_join)
    hidden = model.hidden_size
    token_partition = moe_token_partition.select(
        runtime, parallel=parallel, plan=plan, model=model, tokens=tokens,
        sequence_plan=sequence_plan, reduce_after_join=reduce_after_join)
    full_tokens = tokens
    latent = plan.expert_latent_size or hidden
    expert_width = math.ceil(plan.moe_intermediate_size / parallel.etp)
    transport = moe_transport.select(runtime, layer_scope=plan.scope)
    if transport is not None:
        if getattr(runtime, "moe_recipe", "alltoall") not in {"alltoall", "tp_join_reduce"}:
            raise ModelingError("moe_transport_requires_alltoall_recipe")
        # Reject incompatible expert representations before building any graph.
        moe_transport.validate_contract(model, plan, transport)
    overlap_shared = bool(
        overlap_shared or getattr(plan, "overlap_shared", False)
        or getattr(runtime, "multistream_overlap_shared_expert", False)
        or dict(getattr(runtime, "metadata", ()) or ()).get(
            "overlap_shared_expert", False)
        or dict(getattr(runtime, "metadata", ()) or ()).get(
            "enable_shared_expert_overlap", False))
    group_tokens = global_tokens if global_tokens is not None else pre_sp_tokens
    local_experts = max(ONE, (int(plan.num_experts) + int(parallel.ep) - ONE)
                        // int(parallel.ep))
    partitions = (sequence_plan.source_partitions if sequence_plan else
                  getattr(routing_realization, "source_partitions",
                          min(parallel.tp, parallel.ep)))
    if caller is not None:
        # One replicated source owns the complete token selection; the native
        # ledger is projected onto this rank's contiguous expert bank.
        projected = caller.prepare(
            routing, model, plan, parallel, group_tokens, routing_realization)
        routing_realization = projected.routing_realization
        local_rows, active = projected.local_rows, projected.active
        routing_metadata = projected.routing_metadata
        partitions = 1
    else:
        if token_partition:
            partitions = 1
            if routing_realization is not None:
                # Keep the cohort's integer expert counts; only source ownership
                # changes from implicit replicated assignment division to TP views.
                routing_realization = replace(routing_realization, source_partitions=1)
        local_rows, active = (
            routing.local(model, plan, parallel, group_tokens, owner=price_owner)
            if routing_realization is None and price_owner is not None and type(routing) is RoutingPolicy
            else routing.local(model, plan, parallel, group_tokens)
            if routing_realization is None
            else (routing_realization.rows, routing_realization.active))
        routing_metadata = (
            (("routing_basis", "probability_moments"), ("expected_rows", local_rows),
             ("expected_active_experts", active))
            if routing_realization is None else
            (("routing_basis", routing_realization.basis),
             ("routing_realization", routing_realization.to_dict())))
    if token_partition:
        routing_metadata += token_partition.metadata()
    # The local EP owner this graph represents. Base and redundant physical
    # slots are a property of that owner, not of the logical expert ledger.
    local_ep_rank = int(
        getattr(routing_realization, "representative_rank", ZERO) or ZERO)
    if routing_realization is None and ep_owner is not None:
        # Probability routing has no explicit representative rank.  The
        # physical EP owner still decides which redundant slot bank this graph
        # represents; pricing every replica as owner zero is not a copy.
        local_ep_rank = int(ep_owner) % max(ONE, int(parallel.ep))
    destination_rows = (
        tuple(int(value) for value in routing_realization.rank_assignment_rows)
        if routing_realization is not None
        and getattr(routing_realization, "rank_assignment_rows", None) is not None
        else None)
    represented_ranks = (
        tuple(int(value) for value in
              (getattr(routing_realization, "represented_ep_ranks", None)
               or (local_ep_rank,)))
        if destination_rows is not None else ())
    receive_rows = (
        max((destination_rows[rank] for rank in represented_ranks
             if 0 <= rank < len(destination_rows)), default=ZERO)
        if destination_rows is not None else int(math.ceil(local_rows)))
    runtime_routing = dict(getattr(runtime, "routing", {}) or {})
    has_slots = (
        int(getattr(parallel, "redundant_experts_per_rank", ZERO) or ZERO) > ZERO
        or "expert_slot_map" in runtime_routing
        or (int(plan.num_experts) > ZERO and int(parallel.ep) > ZERO
            and int(plan.num_experts) % int(parallel.ep) != ZERO))
    balanced = (routing_realization is not None
                and routing_realization.basis in {"forced_balanced_integer_counts",
                                                  "source_force_eplb_masked_integer_counts"})
    router_prefix, router, weight_dtype, weight_bytes = router_gemm(
        prefix=prefix, plan=plan, deps=deps, stage=stage, requests=requests,
        tokens=tokens, model=model, hidden=hidden, mode=mode)
    routed_input, logits_input = deps, router.op_id
    partition_views = ()
    if token_partition:
        partition_views = token_partition.prepare(
            prefix=prefix, input_id=deps, logits_id=router.op_id, width=hidden,
            experts=plan.num_experts, stage=stage, layer=plan.index,
            requests=requests, phase=mode, dtype=model.activation_dtype,
            logits_dtype=token_partition.logits_dtype or weight_dtype)
        routed_input, logits_input = (op.op_id for op in partition_views[-2:])
        tokens = token_partition.local_rows
    selection = build_selection(
        prefix=prefix, plan=plan, logits_input=logits_input,
        weight_dtype=weight_dtype, weight_bytes=weight_bytes, stage=stage,
        requests=requests, tokens=tokens, mode=mode, runtime=runtime)
    # An idle representative owns no local hidden-state row: its router and
    # selection calls are not scheduled, while the pack still names its
    # selection producer contract.
    ops = ([*router_prefix, router, *partition_views, *selection.ops]
           if tokens else [])
    select = selection.select
    ids_producer = selection.ids_producer if tokens else select.op_id
    weights_producer = selection.weights_producer if tokens else None
    exchange_options = resolve_exchange_options(runtime, transport, layer_scope=plan.scope)
    exchange_factor = deepep_exchange_factor(
        options=exchange_options, ep=parallel.ep, top_k=plan.top_k,
        tokens=tokens, hidden=latent,
        dtype_bytes_val=(transport.element_bytes(latent) if transport is not None
                         else dtype_bytes(model.activation_dtype)))
    source_traffic, payload_tokens = source_issuance(
        routing_realization=routing_realization, token_partition=token_partition,
        sequence_plan=sequence_plan, plan=plan, tokens=tokens,
        full_tokens=full_tokens)
    transport_rows = None
    transport_histogram_gather = None
    if transport is not None:
        # Count/quantize the same valid source rows the pack and explicit peer
        # ledger use.  ``tokens`` may be a graph bucket while an exact route
        # owns fewer valid source rows; reusing the bucket multiplies both
        # histogram and FP32/INT8 scale work without any routed payload.
        transport_rows = moe_transport.source_rows(
            payload_tokens, plan.top_k, partitions)
        transport_ops, transport_histogram_gather = moe_transport.histogram(
            prefix=prefix, plan=plan, stage=stage, requests=requests,
            rows=transport_rows, ep=parallel.ep, deps=deps,
            ids_producer=ids_producer if tokens else None,
            realization=routing_realization, mode=mode, transport=transport)
        ops.extend(transport_ops)
    source = routed_input
    if latent != hidden and tokens:
        projection = gemm_op(
            op_id=prefix + ".latent_down", deps=(deps,), stage=stage, layer=plan.index,
            requests=requests, m=tokens, n=latent, k=hidden, quant=model.quant,
            category="latent_moe", phase=mode, family="projection", recipe="latent_projection",
        )
        ops.append(projection)
        source = projection.op_id
        if plan.expert_latent_norm_position == "pre":
            latent_norm = _latent_norm(
                prefix, source, stage, plan, requests, tokens, latent, model, mode)
            ops.append(latent_norm)
            source = latent_norm.op_id
    pack_read = payload_tokens * (latent * 2 + plan.top_k * 4)
    pack_write = payload_tokens * plan.top_k * (latent * 2 + 4)
    pack_ops = pack_extra = None
    if caller is not None:
        pack_read, pack_write, pack_ops = caller.pack_work(
            rows=tokens, latent=latent, local_rows=local_rows,
            weight_bytes=weight_bytes, act_bytes=dtype_bytes(model.activation_dtype))
        pack_extra = caller.pack_metadata(local_rows)
    packing = vector_op(
        op_id=prefix + ".pack",
        deps=tuple(dict.fromkeys((source, select.op_id, ids_producer))) if tokens else (deps,),
        stage=stage, layer=plan.index, requests=requests,
        elements=payload_tokens * plan.top_k, ops_per_element=1,
        vector_ops_override=pack_ops,
        dtype=model.activation_dtype, family="routing", phase=mode, recipe="route_pack",
        read_bytes_override=pack_read, write_bytes_override=pack_write,
        shape=(tokens, plan.top_k),
        metadata=(
            ("topk_ids_producer", ids_producer),
            ("payload_producer", source),
            ("num_experts", plan.num_experts),
            ("top_k", plan.top_k),
            ("topk_ids_dtype", "int32"),
            ("packed_layout", "expert_grouped"),
            *((pack_extra or ())),
            *routing_metadata,
        ),
    )
    if transport_histogram_gather is not None:
        packing = replace(packing, deps=tuple(dict.fromkeys((
            *packing.deps, transport_histogram_gather))))
    ops.append(packing)
    last = packing.op_id
    transport_scale = transport_payload = transport_quantized = None
    dispatch_overlap_id = None
    dispatch_source = None
    gate_row_cut = None
    if transport is not None:
        transport_quantized = moe_transport.quantize(
            prefix=prefix, plan=plan, stage=stage, requests=requests,
            pack=packing, transport=transport, rows=transport_rows,
            width=latent, partitions=partitions,
            mode=mode)
        ops.append(transport_quantized)
        exchange_ops, transport_scale, transport_payload = moe_transport.dispatch(
            prefix=prefix, plan=plan, stage=stage, requests=requests,
            quantized=transport_quantized, transport=transport,
            rows=transport_rows, width=latent, ep=parallel.ep,
            partitions=partitions, mode=mode,
            hist_gather=transport_histogram_gather)
        for exchange_op in exchange_ops:
            if exchange_op.recipe == "expert_quantized_dispatch":
                exchange_op = scale_exchange_operator(
                    exchange_op, exchange_factor * exchange_options["eplb"],
                    exchange_options, ep=parallel.ep, top_k=plan.top_k)
                transport_cut = deepep.overlap_row_cut(
                    exchange_op.local_shape[0], exchange_options["overlap"])
                split_ops, dispatch_overlap_id = split_deepep_dispatch_overlap(
                    exchange_op, exchange_options, row_cut=transport_cut)
                ops.extend(split_ops)
            else:
                ops.append(exchange_op)
        if transport_payload is not None:
            last = transport_payload
        else:
            last = transport_quantized.op_id
    # One logical source assignment belongs to one local EP rank. Both routing
    # paths use the same token-to-EP-rank ownership, so the declared volume is
    # per physical source rank rather than a replicated TP total. ETP groups
    # still need their own copies; they are separate expert shards.
    source_partitions = partitions
    routing_basis = routing_realization.basis if routing_realization is not None else "probability_moments"
    dispatch_row_bytes, combine_row_bytes, dispatch_quant_meta, combine_quant_meta, pack_quant = (
        moe_transport.default_exchange_contract(
            model, runtime, plan, transport, latent, ep=parallel.ep, prefix=prefix,
            stage=stage, requests=requests, pack=packing, tokens=payload_tokens,
            partitions=partitions, mode=mode))
    absorbed_routing = None
    absorbed_routing_meta = None
    if pack_quant is not None:
        absorbed_pack, retired = (
            absorb_init_routing(pack=packing, quantizer=pack_quant, ep=parallel.ep,
                dispatch_owns_quant=dispatch_quantizes_internally(runtime))
            if cann_native_active(runtime) else (packing, None))
        if retired is not None:
            packing = absorbed_pack
            ops[-1] = packing
            last = packing.op_id
            absorbed_routing = packing
            absorbed_routing_meta = dict(packing.metadata)
        else:
            ops.append(pack_quant)
            last = pack_quant.op_id
    # A declared source assignment is partitioned across its physical source
    # ranks for every routing path.  The old implicit min(tp, ep) special case
    # charged each replica the full assignment, multiplying the per-source
    # egress by the partition count.  The partition is the exact participant
    # count; no coefficient or operator special case.
    exchange_partitions = source_partitions
    ep_scale = tokens * plan.top_k * (parallel.ep - 1) / parallel.ep / exchange_partitions
    dispatch_exchange_bytes, combine_exchange_bytes = ep_scale * dispatch_row_bytes, ep_scale * combine_row_bytes
    collective_meta = (("group_axis", "ep"),
                       ("source_assignment_partitions", source_partitions),
                       ("traffic_model", "uniform_peer_payload"), ("routing_basis", routing_basis))
    def masked_collective(op, reverse=False):
        if source_traffic is None:
            return op
        matrix = source_traffic.payload_matrix(combine_row_bytes if reverse else dispatch_row_bytes, reverse=reverse)
        meta = {k: v for k, v in op.metadata if k != "reverse_of_collective"}
        meta.update(peer_volume_basis="group_mean", peer_traffic={"kind": "matrix", "payload_bytes": matrix},
                    traffic_model=source_traffic.basis, source_token_mask=source_traffic.to_dict())
        return replace(op, family="alltoallv", link_bytes=sum(map(sum, matrix)) / parallel.ep,
                       metadata=tuple(meta.items()))
    if local_rows and exchange_options["overlap"] > 0.0:
        gate_row_cut = deepep.overlap_row_cut(
            max(0, int(math.ceil(local_rows))), exchange_options["overlap"])
    if parallel.ep > 1 and transport is None and exchange:
        dispatch = comm_op(
            op_id=prefix + ".dispatch", deps=(last,), stage=stage, layer=plan.index,
            requests=requests, family="alltoall", phase=mode, recipe="expert_dispatch",
            link_bytes=dispatch_exchange_bytes,
            shape=(int(tokens) * int(plan.top_k), latent),
            metadata=collective_meta + (
                ("collective_id", prefix + ".dispatch"),
                ("moe_num_experts", int(plan.num_experts)),
                ("moe_ep_size", int(parallel.ep)),
            ) + dispatch_quant_meta,
        )
        dispatch = masked_collective(dispatch)
        dispatch = scale_exchange_operator(
            dispatch, exchange_factor * exchange_options["eplb"],
            exchange_options, ep=parallel.ep, top_k=plan.top_k)
        # The receive allocation is a concrete value of this collective: the
        # row views below alias it instead of searching an ancestry for dtype.
        received_dtype = (absorbed_routing_meta or {}).get(
            "activation_payload_format") or model.activation_dtype
        dispatch = deepep.declare_received_value(
            dispatch, dtype=str(received_dtype),
            shape=(receive_rows, latent),
            row_bytes=latent * dtype_bytes(str(received_dtype)),
            storage=f"moe.rx:{prefix}.dispatch",
            binding="static", binding_name=f"{prefix}.dispatch.rx")
        dispatch_source = deepep.received_value_contract(dispatch)
        split_ops, dispatch_overlap_id = split_deepep_dispatch_overlap(
            dispatch, exchange_options, row_cut=gate_row_cut)
        ops.extend(split_ops)
        last = split_ops[0].op_id
    if local_rows:
        gate_weight = model.quant.matrix_bytes(2 * expert_width, latent, "routed_expert") * active
        down_weight = model.quant.matrix_bytes(latent, expert_width, "routed_expert") * active
        common = dict(stage=stage, layer=plan.index, requests=requests,
                      quant=model.quant, category="routed_expert", phase=mode, m=math.ceil(local_rows))
        expert_input_deps = (last,)
        if transport is not None and parallel.ep > 1:
            # Both received quantities are real inputs. The payload already
            # waits for scale completion, but the gate retains the direct edge.
            expert_input_deps = tuple(dict.fromkeys(
                dependency for dependency in
                (transport_scale, transport_payload, transport_quantized.op_id)
                if dependency))
        gate = gemm_op(
            op_id=prefix + ".expert_up", deps=expert_input_deps, n=2 * expert_width, k=latent,
            family="grouped_gemm", recipe="grouped_gate_up", weight_bytes_override=gate_weight, **common,
        )
        down = gemm_op(
            op_id=prefix + ".expert_down", deps=(prefix + ".expert_act",), n=latent, k=expert_width,
            family="grouped_gemm", recipe="grouped_down", weight_bytes_override=down_weight, **common,
        )
        # Probability moments stay fractional; explicit realization counts stay
        # integer. Both paths price active experts without reading empty banks.
        def routed_work(op, n, k, weights, weight_kind):
            return routed_expert_work(
                op, n=n, k=k, weights=weights, weight_kind=weight_kind,
                local_rows=local_rows, active_experts=active,
                local_experts=local_experts, local_ep_rank=local_ep_rank,
                routing_realization=routing_realization,
                routing_metadata=routing_metadata, balanced=balanced)
        gate = routed_work(gate, TWO * expert_width, latent, gate_weight, "gate_up")
        down = routed_work(down, latent, expert_width, down_weight, "down")
        if dispatch_quant_meta and transport is None:
            absorbed_meta = ((("activation_scale_producer", absorbed_routing.op_id),
                              ("activation_quantization_kernel", "aclnnMoeInitRoutingV3"))
                             if absorbed_routing is not None else ())
            gate = replace(gate, metadata=gate.metadata + (
                ("activation_producer", "explicit_transport_quantization"),
                ("activation_payload_producer", last),
                *absorbed_meta,
            ))
        if transport is not None:
            gate = bind_transport_gate(
                gate, down, transport=transport, latent=latent,
                local_rows=local_rows,
                payload=transport_payload or transport_quantized.op_id,
                scale=transport_scale or transport_quantized.op_id)
        allow_up_split = not bool((getattr(runtime, "kernels", None) or {}).get("lowerings"))
        activation_rows = (
            token_partition.activation_tensor_rows
            if token_partition is not None and token_partition.activation_tensor_rows
            else (local_rows if caller is None else caller.activation_rows()))
        gate_ops, gate_last = split_deepep_up_overlap(
            gate, exchange_options, dispatch_overlap_id=dispatch_overlap_id,
            allow_split=allow_up_split, active_rows=activation_rows,
            source_contract=dispatch_source,
            row_cut=gate_row_cut, activation_id=prefix + ".expert_act")
        gate_segments = tuple(op for op in gate_ops if op.family == "grouped_gemm")
        activation = gated_activation(
            plan=plan, rows=activation_rows, width=expert_width,
            op_id=prefix + ".expert_act",
            deps=tuple(op.op_id for op in gate_segments) or (gate_last,),
            stage=stage, requests=requests, dtype=model.activation_dtype,
            phase=mode, recipe="grouped_swiglu",
        )
        if gate_segments:
            activation = replace(activation, metadata=activation.metadata + (
                ("memory_accesses", tuple({
                    "producer": op.op_id,
                    "output": dict(op.metadata).get("primary_output") or "output",
                    "direction": "read", "position": "local",
                    "storage": f"moe.act:{op.op_id}",
                    "bytes": float(int(op.local_shape[0]) * int(op.local_shape[1])
                                   * dtype_bytes(dict(op.metadata).get("output_format") or "bf16")),
                    "span": ((0, int(op.local_shape[0])),),
                    "last_use": prefix + ".expert_down",
                    "purpose": "gate_output_rows",
                } for op in gate_segments)),))
            segment_meta = tuple(
                (op.op_id, dict(op.metadata).get("deepep_physical_row_start"),
                 dict(op.metadata).get("deepep_physical_row_stop"),
                 int(op.local_shape[0]), int(op.local_shape[1]))
                for op in gate_segments)
            activation = replace(activation, metadata=activation.metadata + (
                ("activation_input_segments", segment_meta),
                ("activation_input_bytes", sum(
                    int(op.local_shape[0]) * int(op.local_shape[1])
                    * dtype_bytes(dict(op.metadata).get("output_format") or "bf16")
                    for op in gate_segments)),
            ))
        if token_partition is not None and token_partition.activation_owners:
            activation = replace(activation, metadata=activation.metadata + (
                ("activation_owners", token_partition.activation_owners),
                ("activation_tensor_rows", activation_rows),
                ("activation_capacity_basis", "declared_unmasked_tensor_capacity")))
        if caller is not None:
            activation = caller.tag_activation(activation, local_rows)
        ops.extend((*gate_ops, activation, down))
        declare_gate_up_fusion_group(
            ops=ops, runtime=runtime, prefix=prefix, activation=activation,
            gate_ops=gate_ops, gate=gate, down=ops[-1],
            local_experts=local_experts, latent=latent,
            expert_width=expert_width, routing_metadata=routing_metadata,
            model=model)
        down_ops, down_mid_id, last = split_deepep_down_overlap(
            ops.pop(), exchange_options)
        ops.extend(down_ops)
        if parallel.etp > 1 and not reduce_after_join:
            reduce = comm_op(
                op_id=prefix + ".expert_reduce", deps=(last,), stage=stage, layer=plan.index,
                requests=requests, family="allreduce", phase=mode, recipe="expert_tensor_reduce",
                link_bytes=2 * (parallel.etp - 1) / parallel.etp * local_rows * latent
                           * dtype_bytes(model.activation_dtype),
                shape=(math.ceil(local_rows), latent), metadata=(("group_axis", "etp"),))
            ops.append(reduce)
            last = reduce.op_id
            down_mid_id = reduce.op_id
    else:
        down_mid_id = None
    if parallel.ep > 1 and exchange:
        if transport is not None:
            combine = moe_transport.combine(
                prefix=prefix, plan=plan, stage=stage, requests=requests,
                deps=last, transport=transport, rows=transport_rows, width=latent,
                ep=parallel.ep, partitions=partitions,
                mode=mode, payload_dispatch=transport_payload)
        else:
            combine = comm_op(
                op_id=prefix + ".combine", deps=(last,), stage=stage, layer=plan.index,
                requests=requests, family="alltoall", phase=mode, recipe="expert_combine",
                link_bytes=combine_exchange_bytes, shape=(tokens * plan.top_k, latent),
                metadata=collective_meta + (("collective_id", prefix + ".combine"),
                                           ("reverse_of_collective", prefix + ".dispatch")) + combine_quant_meta,
            )
        if transport is None:
            combine = masked_collective(combine, reverse=True)
        combine = scale_exchange_operator(
            combine, exchange_factor * exchange_options["eplb"],
            exchange_options, ep=parallel.ep, top_k=plan.top_k, is_combine=True)
        overlap_deps = tuple(dict.fromkeys(
            dep for dep in (down_mid_id, dispatch_overlap_id) if dep
        ))
        combine_ops, last = split_deepep_combine_overlap(
            combine, exchange_options, overlap_deps=overlap_deps,
            main_deps=(last,), dispatch_overlap_id=dispatch_overlap_id)
        ops.extend(combine_ops)
    unpack_inputs = tuple(dict.fromkeys(
        (last, packing.op_id) + ((weights_producer,) if weights_producer is not None else ())))
    unpack_metadata = [
        ("payload_producer", last), ("inverse_index_producer", packing.op_id),
        ("topk_weights_producer", weights_producer), ("input_layout", "expert_grouped"),
        ("topk_ids_dtype", "int32"),
        ("router_weight_dtype", weight_dtype if weights_producer is not None else None),
        ("has_expert_map", False), ("apply_router_weight_on_input", False),
        ("ep_degree", parallel.ep), *routing_metadata,
    ]
    if routing_realization is not None:
        unpack_metadata.append(("input_rows", int(local_rows)))
    act_bytes = dtype_bytes(model.activation_dtype)
    unpack_read = tokens * plan.top_k * (latent * act_bytes + dtype_bytes("fp32"))
    if absorbed_routing is not None:
        # The native pack wrote one INT32 inverse slot per assignment.
        inverse_index_bytes = tokens * plan.top_k * dtype_bytes(INVERSE_INDEX_DTYPE)
        unpack_read += inverse_index_bytes
        unpack_metadata.extend((("inverse_index_dtype", INVERSE_INDEX_DTYPE),
                                ("inverse_index_bytes", float(inverse_index_bytes))))
    unpack_write = tokens * latent * act_bytes
    unpack_ops = unpack_extra = None
    if caller is not None:
        unpack_read, unpack_write, unpack_ops = caller.unpack_work(
            tokens=tokens, latent=latent, local_rows=local_rows,
            weight_bytes=weight_bytes, act_bytes=act_bytes)
        unpack_extra = caller.unpack_metadata(local_rows, weight_dtype)
    unpermute = vector_op(
        op_id=prefix + ".unpack", deps=unpack_inputs, stage=stage, layer=plan.index,
        requests=requests, elements=tokens * latent, ops_per_element=2 * plan.top_k,
        vector_ops_override=unpack_ops,
        dtype=model.activation_dtype, family="routing", phase=mode, recipe="route_unpack",
        read_bytes_override=unpack_read, write_bytes_override=unpack_write,
        shape=(tokens, plan.top_k, latent),
        metadata=tuple(unpack_metadata) + (unpack_extra or ()),
    )
    ops.append(unpermute)
    last = unpermute.op_id
    if latent != hidden and tokens:
        if plan.expert_latent_norm_position == "post":
            latent_norm = _latent_norm(
                prefix, last, stage, plan, requests, tokens, latent, model, mode)
            ops.append(latent_norm)
            last = latent_norm.op_id
        projection = gemm_op(
            op_id=prefix + ".latent_up", deps=(last,), stage=stage, layer=plan.index,
            requests=requests, m=tokens, n=hidden, k=latent, quant=model.quant,
            category="latent_moe", phase=mode, family="projection", recipe="latent_projection",
        )
        ops.append(projection)
        last = projection.op_id
    routed_tokens = tokens
    if token_partition:
        gathered = token_partition.finalize(
            prefix=prefix, source=last, width=hidden, stage=stage, layer=plan.index,
            requests=requests, phase=mode, dtype=model.activation_dtype)
        ops.extend(gathered)
        last, tokens = gathered[-1].op_id, full_tokens
    if plan.num_shared_experts and tokens:
        shared_dp = bool(dict(runtime.metadata).get("enable_shared_expert_dp", False)) if runtime is not None else False
        shared_input, shared_tokens, shared_prefix = deps, tokens, []
        if sequence_plan and not shared_dp:
            from .sequence_parallel import allgather, reduce_scatter
            gathered = allgather(
                plan=sequence_plan, op_id=prefix + ".shared.sp_gather", deps=(deps,),
                stage=stage, layer=plan.index, requests=requests, phase=mode,
                width=hidden, dtype=model.activation_dtype)
            shared_prefix.append(gathered)
            shared_input, shared_tokens = gathered.op_id, sequence_plan.tokens_global
        shared, shared_last = dense(
            prefix=prefix + ".shared", plan=replace(
                plan,
                intermediate_size=(plan.shared_intermediate_size
                                   or plan.num_shared_experts * plan.moe_intermediate_size),
                # The shared expert declares its own clamp scalar; an absent
                # shared limit inherits the routed-expert declaration.
                swiglu_limit=(plan.swiglu_shared_limit or plan.swiglu_limit)),
            deps=shared_input, stage=stage, requests=requests, tokens=tokens if shared_dp else shared_tokens, model=model,
            parallel=replace(parallel, tp=1) if shared_dp else parallel, mode=mode, category="shared_expert",
            runtime=runtime, reduce_output=not reduce_after_join and sequence_plan is None,
        )
        shared = shared_prefix + shared
        if sequence_plan and not shared_dp:
            scattered = reduce_scatter(
                plan=sequence_plan, op_id=prefix + ".shared.sp_scatter", deps=(shared_last,),
                stage=stage, layer=plan.index, requests=requests, phase=mode,
                width=hidden, dtype=model.activation_dtype)
            shared.append(scattered)
            shared_last = scattered.op_id
        if plan.shared_expert_gate:
            gate = gemm_op(
                op_id=prefix + ".shared_gate", deps=(deps,), stage=stage, layer=plan.index,
                requests=requests, m=tokens, n=1, k=hidden, quant=model.quant,
                category="router", phase=mode, recipe="shared_gate", family="router")
            applied = vector_op(
                op_id=prefix + ".shared_gate_apply", deps=(shared_last, gate.op_id),
                stage=stage, layer=plan.index, requests=requests, elements=tokens * hidden,
                ops_per_element=2, dtype=model.activation_dtype, family="elementwise",
                phase=mode, recipe="shared_gate_apply")
            shared.extend((gate, applied))
            shared_last = applied.op_id
        if overlap_shared:
            shared = [replace(op, stream="shared_stream") for op in shared]
        ops.extend(shared)
        join = vector_op(
            op_id=prefix + ".join", deps=(last, shared_last), stage=stage, layer=plan.index,
            requests=requests, elements=tokens * hidden, ops_per_element=1,
            dtype=model.activation_dtype, family="elementwise", phase=mode, recipe="expert_add",
            shape=(tokens, hidden),
            read_bytes_override=2 * tokens * hidden * dtype_bytes(model.activation_dtype),
            write_bytes_override=tokens * hidden * dtype_bytes(model.activation_dtype),
            metadata=(("input_tensor_count", 2),),
        )
        ops.append(join)
        last = join.op_id
    if reduce_after_join and parallel.tp > 1 and tokens:
        reduction = comm_op(
            op_id=prefix + ".moe_output_reduce", deps=(last,), stage=stage, layer=plan.index,
            requests=requests, family="allreduce", phase=mode, recipe="moe_output_reduce",
            link_bytes=2 * (parallel.tp - 1) / parallel.tp * tokens * hidden
                       * dtype_bytes(model.activation_dtype),
            shape=(tokens, hidden),
            metadata=(("group_axis", "tp"), ("steps", 2 * (parallel.tp - 1)),
                      ("reduction_boundary", "token_rows_after_routed_shared_join")),
        )
        if caller is not None:
            reduction = caller.tag_reduce(reduction)
        ops.append(reduction)
        last = reduction.op_id
    if (has_slots and runtime is not None and int(plan.num_experts) > ZERO
            and routed_tokens > ZERO):
        # Redundant physical copies change residency and active weight reads,
        # never logical assignments or aggregate FLOPs. Bind the exact
        # per-owner slot bytes after routing and before native graph lowering.
        ops = bind_expert_slots(
            ops, model=model, plan=plan, parallel=parallel, routing=routing,
            runtime_routing=runtime_routing, routing_realization=routing_realization,
            group_tokens=group_tokens, expert_width=expert_width, latent=latent)
    if overlap_shared:
        ops = reorder_overlapped_shared(ops)
    from .mc2_graph import lower as lower_native_mc2
    ops = lower_native_mc2(
        ops, runtime=runtime, prefix=prefix, tokens=routed_tokens, top_k=plan.top_k,
        width=latent, experts=plan.num_experts, ep=parallel.ep,
        source_partitions=partitions, received_assignments=local_rows,
        dtype=model.activation_dtype, source_routing=source_traffic,
        valid_tokens=payload_tokens, features=features,
        source_rows_by_replica=source_rows_by_replica,
        destination_rows_by_rank=destination_rows)
    return ops, last


def source_issuance(*, routing_realization, token_partition, sequence_plan,
                    plan, tokens, full_tokens):
    """Declared source-traffic view and the rows this owner really issues."""
    source_traffic = (
        routing_realization if getattr(routing_realization, "basis", "")
        in {"source_force_eplb_masked_integer_counts",
            "explicit_token_routing_integer"} else
        token_partition.expected_traffic(plan.top_k)
        if token_partition and token_partition.allocated_rows != full_tokens else None)
    if sequence_plan is not None and sum(sequence_plan.valid_per_tp) == 0:
        # This representative owns physical TP/EP participants but no valid
        # source row.  Keep their control/dummy participation; suppress only
        # the payload cell in the ordinary source ledger.
        issued = 0
    elif token_partition is not None:
        # An explicit TP caller owns this slice's actual outgoing rows.
        issued = token_partition.valid_per_rank[0]
    elif source_traffic is not None:
        # A source-mask / exact token ledger owns its valid outgoing rows.
        issued = source_traffic.valid_local_tokens
    else:
        # Ordinary balanced and probability plans route the physical graph
        # rows, including full-capture padding.  ``valid_local_tokens`` is a
        # separate logical-valid statistic and must not shrink that plan.
        issued = tokens
    return source_traffic, issued


def reorder_overlapped_shared(ops):
    """Place the shared-expert stream before the first dispatch collective."""
    shared_ops = [op for op in ops if op.stream == "shared_stream"]
    if not shared_ops:
        return ops
    routed = [op for op in ops if op.stream != "shared_stream"]
    dispatch_index = next(
        (index for index, op in enumerate(routed)
         if op.recipe == "expert_dispatch" or op.family == "alltoall"),
        len(routed))
    return routed[:dispatch_index] + shared_ops + routed[dispatch_index:]


DEFAULT_MC2_CHUNKS = 4


def chunked_mc2(*, prefix, tokens, global_tokens=None, chunks=DEFAULT_MC2_CHUNKS, **kwargs):
    """Finite dispatch/compute/combine chunks with separate ordered streams."""
    group = tokens if global_tokens is None else global_tokens
    count = min(chunks, max(1, group))
    if global_tokens is not None and (tokens % count or group % count):
        # Dividing the cohort total and then dividing each source are not
        # interchangeable at chunk tails. No source partition was supplied.
        raise ModelingError("mc2_cross_dp_tails_require_per_source_chunk_partitions")
    ops, completions = [], []
    for index in range(count):
        local = tokens // count + int(index < tokens % count)
        global_local = group // count + int(index < group % count)
        part, last = alltoall(prefix=f"{prefix}.chunk{index}", tokens=local,
                              global_tokens=global_local, **kwargs)
        phase = "prepare"
        for op in part:
            if op.recipe == "expert_dispatch":
                phase = "dispatch"
            elif op.family == "grouped_gemm":
                phase = "experts"
            elif op.recipe == "expert_combine":
                phase = "combine"
            elif op.recipe == "route_unpack":
                phase = "finish"
            target_stream = (op.stream if op.stream == "shared_stream"
                             else f"mc2:{phase}")
            ops.append(replace(op, stream=target_stream))
        completions.append(last)
    joined = barrier_op(
        op_id=prefix + ".chunks_complete", deps=tuple(completions),
        stage=kwargs["stage"], layer=kwargs["plan"].index,
        requests=kwargs["requests"], phase=kwargs["mode"], recipe="mc2_join",
        shape=(tokens, kwargs["model"].hidden_size))
    ops.append(joined)
    return ops, joined.op_id


MOE_RECIPES = {"alltoall": alltoall, "mc2": chunked_mc2, "tp_join_reduce": tp_join_reduce,
               "local_expert_reduce": local_expert_reduce_recipe}


def moe(*, moe_recipe="alltoall", mc2_chunks=4, **kwargs):
    return MOE_RECIPES[moe_recipe](chunks=mc2_chunks, **kwargs)


FFN_BUILDERS = {"dense": dense, "moe": moe}
