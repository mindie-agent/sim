"""Dense and routed feed-forward programs.

[Role] Describe router, packing, dispatch, expert matrices, combine and shared work.
[Invariants] Routing conserves top-k assignments; zero-token experts do not read
  matrices. Expected routing is a declared probability model, not a measured
  efficiency. Expert and tensor parallelism shard different dimensions.
[Timeouts & Locks] O(experts) arithmetic per layer, no histories or I/O.
[Side-Effect Policy] Constructs operator records only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from mindie_sim.catalog.ops import gemm_op, vector_op, comm_op, barrier_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from .structure_groups import declare_group, source_reference
from .gated_activation import gated_activation, declare_situ_quant, SITU_SOURCE
from . import moe_transport, moe_token_partition


@dataclass(frozen=True)
class RoutingPolicy:
    tile_rows: int = 1

    def local(self, model, layer, parallel, tokens: int) -> tuple[float, float]:
        probabilities = model.expert_routing or (1 / layer.num_experts,) * layer.num_experts
        if len(probabilities) != layer.num_experts:
            raise ValueError("routing_probabilities_do_not_match_layer_experts")
        if (any(value < 0 or not math.isfinite(value) for value in probabilities)
                or sum(value > 0 for value in probabilities) < layer.top_k):
            raise ValueError("routing_support_must_cover_top_k")
        # Weighted sampling without replacement: saturate certain selections,
        # then redistribute remaining probability mass. Sum inclusions = top-k.
        remaining, available = float(layer.top_k), sum(probabilities)
        inclusions = [0.0] * layer.num_experts
        for index in sorted(range(layer.num_experts), key=lambda i: probabilities[i], reverse=True):
            probability = probabilities[index]
            inclusion = min(1.0, remaining * probability / available) if available else 0
            inclusions[index] = inclusion
            remaining -= inclusion
            available -= probability
        rows, active = [], []
        for inclusion in inclusions:
            rows.append(tokens * inclusion)
            active.append(-math.expm1(tokens * math.log1p(-inclusion))
                          if 0 < inclusion < 1 else float(tokens > 0 and inclusion > 0))
        rank_rows = [sum(rows[index::parallel.ep]) for index in range(parallel.ep)]
        slowest = max(range(parallel.ep), key=rank_rows.__getitem__)
        return rank_rows[slowest], sum(active[slowest::parallel.ep])


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
        overlap_shared=False, runtime=None, reduce_after_join=False, sequence_plan=None, **_):
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
        moe_transport.validate_contract(model, plan)
    group_tokens = tokens if global_tokens is None else global_tokens
    partitions = (sequence_plan.source_partitions if sequence_plan else
                  getattr(routing_realization, "source_partitions",
                          min(parallel.tp, parallel.ep)))
    if token_partition:
        partitions = 1
        if routing_realization is not None:
            # Keep the cohort's integer expert counts; only source ownership
            # changes from implicit replicated assignment division to TP views.
            routing_realization = replace(routing_realization, source_partitions=1)
    local_rows, active = (
        routing.local(model, plan, parallel, group_tokens) if routing_realization is None
        else (routing_realization.rows, routing_realization.active))
    routing_metadata = (
        (("routing_basis", "probability_moments"), ("expected_rows", local_rows),
         ("expected_active_experts", active))
        if routing_realization is None else
        (("routing_basis", routing_realization.basis),
         ("routing_realization", routing_realization.to_dict())))
    if token_partition:
        routing_metadata += token_partition.metadata()
    balanced = (routing_realization is not None
                and routing_realization.basis in {"forced_balanced_integer_counts",
                                                  "source_force_eplb_masked_integer_counts"})
    router_input = deps
    router_quant, router_category = model.quant, "router"
    router_prefix = []
    if plan.router_dtype:
        if plan.router_dtype not in {"fp32", "bf16", "fp16"}:
            raise ModelingError(f"unsupported_router_compute_dtype:{plan.router_dtype}")
        router_quant = QuantSpec(
            name="declared_router", weight_format=plan.router_dtype,
            activation_format=plan.router_dtype)
        router_category = "router_runtime"
        if tokens and plan.router_dtype != model.activation_dtype:
            cast = vector_op(
                op_id=prefix + ".router_cast", deps=(deps,), stage=stage, layer=plan.index,
                requests=requests, elements=tokens * (plan.router_input_size or hidden),
                ops_per_element=1, dtype=plan.router_dtype, family="cast",
                phase=mode, recipe="router_input_cast",
                shape=(tokens, plan.router_input_size or hidden),
                read_bytes_override=tokens * (plan.router_input_size or hidden)
                                    * dtype_bytes(model.activation_dtype),
                write_bytes_override=tokens * (plan.router_input_size or hidden)
                                     * dtype_bytes(plan.router_dtype),
                metadata=(("input_format", model.activation_dtype),
                          ("output_format", plan.router_dtype)),
            )
            router_prefix.append(cast)
            router_input = cast.op_id
    router = gemm_op(
        op_id=prefix + ".router", deps=(router_input,), stage=stage, layer=plan.index,
        requests=requests, m=tokens, n=plan.num_experts, k=plan.router_input_size or hidden,
        quant=router_quant, category=router_category, phase=mode, family="router", recipe="router",
        output_dtype=plan.router_output_dtype or None,
    )
    weight_dtype = dict(router.metadata)["output_format"]
    weight_bytes = dtype_bytes(weight_dtype)
    routed_input, logits_input = deps, router.op_id
    partition_views = ()
    if token_partition:
        partition_views = token_partition.prepare(
            prefix=prefix, input_id=deps, logits_id=router.op_id, width=hidden,
            experts=plan.num_experts, stage=stage, layer=plan.index,
            requests=requests, phase=mode, dtype=model.activation_dtype,
            logits_dtype=weight_dtype)
        routed_input, logits_input = (op.op_id for op in partition_views[-2:])
        tokens = token_partition.local_rows
    source_traffic = (routing_realization if getattr(routing_realization, "basis", "")
                      == "source_force_eplb_masked_integer_counts" else
                      token_partition.expected_traffic(plan.top_k)
                      if token_partition and token_partition.allocated_rows != full_tokens else None)
    payload_tokens = tokens if source_traffic is None else source_traffic.valid_local_tokens
    hash_table = plan.router_mode == "hash"
    hash_native = plan.router_mode in {"hash", "sqrtsoftplus"}
    select = vector_op(
        op_id=prefix + ".topk", deps=(logits_input,), stage=stage, layer=plan.index,
        requests=requests, elements=tokens * plan.num_experts,
        ops_per_element=5 + (plan.top_k / plan.num_experts if hash_table
                             else math.log2(max(1, plan.top_k))),
        dtype="fp32", family="topk", phase=mode,
        recipe="hash_routing" if hash_table else "router_topk",
        shape=(tokens, plan.num_experts),
        read_bytes_override=tokens * plan.num_experts * weight_bytes
                            + (tokens * (8 + plan.top_k * 4) if hash_table else 0),
        write_bytes_override=tokens * plan.top_k * (weight_bytes + 4),
        returns=("weights", "ids"),
        metadata=(("input_format", weight_dtype), ("output_format", weight_dtype),
                  ("topk_ids_dtype", "int32"), ("router_weight_dtype", weight_dtype),
                  ("score_function", plan.router_mode),
                  ("hash_table_active_rows", tokens if hash_table else 0)),
    )
    ops = [*router_prefix, router, *partition_views, select] if tokens else []
    selection_members = [select.op_id]
    if tokens and plan.router_groups and not hash_table:
        group_select = vector_op(
            op_id=prefix + ".group_select", deps=(logits_input,), stage=stage, layer=plan.index,
            requests=requests, elements=tokens * plan.num_experts, ops_per_element=2,
            dtype="fp32", family="topk", phase=mode, recipe="router_group_topk",
            shape=(tokens, plan.router_groups, plan.router_group_topk))
        select = replace(select, deps=(group_select.op_id,))
        ops = [*router_prefix, router, *partition_views, group_select, select]
        selection_members.insert(0, group_select.op_id)
    ids_producer = select.op_id
    weights_producer = select.op_id if tokens else None
    if tokens and plan.router_norm_topk:
        normalized = vector_op(
            op_id=prefix + ".route_normalize", deps=(select.op_id,), stage=stage, layer=plan.index,
            requests=requests, elements=tokens * plan.top_k, ops_per_element=3,
            dtype="fp32", family="elementwise", phase=mode, recipe="route_normalize",
            read_bytes_override=tokens * plan.top_k * weight_bytes,
            write_bytes_override=tokens * plan.top_k * weight_bytes,
            metadata=(("input_format", weight_dtype), ("output_format", weight_dtype)))
        ops.append(normalized)
        select = normalized
        weights_producer = normalized.op_id
        if not hash_native:
            selection_members.append(normalized.op_id)
    if tokens:
        # The native selection call consumes logits. Its scope excludes their
        # GEMM, and Hash's unsupported renorm remains a separate consumer.
        native_kind = "MoeGatingTopKHash" if hash_native else "MoeGatingTopK"
        declare_group(
            ops, key=prefix + ".router_selection", role="router_selection",
            family="moe", members=tuple(selection_members),
            entries=(selection_members[0],), anchor=selection_members[-1],
            block="ffn", scope_role="body",
            view={"tokens": tokens, "experts": plan.num_experts, "topk": plan.top_k,
                  "dtype": weight_dtype, "native_kind": native_kind,
                  "selection_mode": "hash_table" if hash_table else "score_topk",
                  "score_function": plan.router_mode,
                  "renorm": bool(plan.router_norm_topk and not hash_native),
                  "external_renorm": bool(plan.router_norm_topk and hash_native),
                  "groups": plan.router_groups, "group_topk": plan.router_group_topk},
            implementation={"native_kind": native_kind,
                            "kind": "declared_native_correspondence",
                            "declared_parameters": {
                                "score_function": plan.router_mode,
                                "renorm": bool(plan.router_norm_topk and not hash_native),
                                "external_renorm": bool(plan.router_norm_topk and hash_native),
                                "groups": plan.router_groups, "group_topk": plan.router_group_topk}},
            source={
                "repository": "vllm-ascend",
                "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
                "path": "vllm_ascend/ops/fused_moe/router/fused_topk_router.py",
                "sha256": "4a4e8cb559bda606457022ad627e787228b24bb6e514f1e162b323e3de6898d4",
                "source_state": "local_worktree", "symbol": "AscendFusedTopKRouter._compute_routing",
                "scope": "independent_source_semantics; not_capture_binary_identity"},
        )
    transport_rows = None
    transport_histogram_gather = None
    if transport is not None:
        transport_rows = moe_transport.source_rows(tokens, plan.top_k, partitions)
        transport_ops, transport_histogram_gather = moe_transport.histogram(
            prefix=prefix, plan=plan, stage=stage, requests=requests,
            rows=transport_rows, ep=parallel.ep, deps=deps,
            ids_producer=ids_producer if tokens else None,
            realization=routing_realization, mode=mode)
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
    packing = vector_op(
        op_id=prefix + ".pack",
        deps=tuple(dict.fromkeys((source, select.op_id, ids_producer))) if tokens else (deps,),
        stage=stage, layer=plan.index, requests=requests,
        elements=payload_tokens * plan.top_k, ops_per_element=1,
        dtype=model.activation_dtype, family="routing", phase=mode, recipe="route_pack",
        read_bytes_override=payload_tokens * (latent * 2 + plan.top_k * 4),
        write_bytes_override=payload_tokens * plan.top_k * (latent * 2 + 4),
        shape=(tokens, plan.top_k),
        metadata=(
            ("topk_ids_producer", ids_producer),
            ("payload_producer", source),
            ("num_experts", plan.num_experts),
            ("top_k", plan.top_k),
            ("topk_ids_dtype", "int32"),
            ("packed_layout", "expert_grouped"),
            *routing_metadata,
        ),
    )
    if transport_histogram_gather is not None:
        packing = replace(packing, deps=tuple(dict.fromkeys((
            *packing.deps, transport_histogram_gather))))
    ops.append(packing)
    last = packing.op_id
    transport_scale = transport_payload = transport_quantized = None
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
        ops.extend(exchange_ops)
        if transport_payload is not None:
            last = transport_payload
        else:
            last = transport_quantized.op_id
    # One logical source assignment belongs to one local EP rank. Both routing
    # paths use the same token-to-EP-rank ownership, so the declared volume is
    # per physical source rank rather than a replicated TP total. ETP groups
    # still need their own copies; they are separate expert shards.
    source_partitions = partitions
    routing_basis = (routing_realization.basis
                     if routing_realization is not None else "probability_moments")
    exchange_bytes = tokens * plan.top_k * latent * dtype_bytes(model.activation_dtype) \
        * (parallel.ep - 1) / parallel.ep / source_partitions
    collective_meta = (
        ("group_axis", "ep"), ("steps", max(0, parallel.ep - 1)),
        ("source_assignment_partitions", source_partitions),
        ("traffic_model", "uniform_peer_payload"),
        ("routing_basis", routing_basis),
    )
    def masked_collective(op, reverse=False):
        if source_traffic is None:
            return op
        matrix = source_traffic.payload_matrix(latent * dtype_bytes(model.activation_dtype),
                                               reverse=reverse)
        meta = dict(op.metadata)
        meta.pop("reverse_of_collective", None)
        meta.update(peer_volume_basis="group_mean",
                    peer_traffic={"kind": "matrix", "payload_bytes": matrix},
                    traffic_model=source_traffic.basis,
                    source_token_mask=source_traffic.to_dict())
        return replace(op, family="alltoallv", link_bytes=sum(map(sum, matrix)) / parallel.ep,
                       metadata=tuple(meta.items()))
    if parallel.ep > 1 and transport is None:
        dispatch = comm_op(
            op_id=prefix + ".dispatch", deps=(last,), stage=stage, layer=plan.index,
            requests=requests, family="alltoall", phase=mode, recipe="expert_dispatch",
            link_bytes=exchange_bytes, shape=(tokens * plan.top_k, latent),
            metadata=collective_meta + (("collective_id", prefix + ".dispatch"),),
        )
        dispatch = masked_collective(dispatch)
        ops.append(dispatch)
        last = dispatch.op_id
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
                (transport_scale, transport_payload, transport_quantized.op_id)))
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
        def routed_work(op, n, k, weights):
            fraction = local_rows / op.local_shape[0]
            read = (op.mte2_bytes - weights) * fraction + weights
            write = op.mte3_bytes * fraction
            metadata = dict(op.metadata)
            # Aggregate expert counts do not declare per-expert tile geometry.
            # Keep this analytical program distinct from dense/batched mapping;
            # an explicit source grouped planner still owns its actual rows.
            metadata["matrix_mapping_domain"] = "aggregate_experts"
            metadata["matrix_weight_bank_groups"] = math.ceil(plan.num_experts / parallel.ep)
            if routing_realization is not None:
                metadata["matrix_mapping_domain"] = "equal_expert_groups"
                metadata["matrix_groups"] = active
                metadata["matrix_group_rows"] = routing_realization.local_expert_rows
                if len(set(routing_realization.local_expert_rows) - {0}) > 1:
                    metadata["matrix_mapping_domain"] = "integer_expert_groups"
            if "activation_scale_bytes" in metadata:
                metadata["activation_scale_bytes"] *= fraction
            if "matrix_epilogue" in metadata:
                epilogue = dict(metadata["matrix_epilogue"])
                epilogue["row_scale_bytes"] *= fraction
                epilogue["operations_per_output"] *= fraction
                metadata["matrix_epilogue"] = epilogue
            return replace(op, flops=2 * local_rows * n * k, hbm_bytes=read + write,
                           vector_ops=op.vector_ops * fraction,
                           mte2_bytes=read, mte3_bytes=write,
                           recipe=("balanced_" + op.recipe) if balanced else op.recipe,
                           metadata=tuple(metadata.items()) + routing_metadata)
        gate = routed_work(gate, 2 * expert_width, latent, gate_weight)
        down = routed_work(down, latent, expert_width, down_weight)
        if transport is not None:
            gate_meta = dict(gate.metadata)
            expected_rows = math.ceil(local_rows)
            if (gate_meta.get("activation_format") != transport.payload_dtype
                    or gate_meta.get("activation_scale_format", "fp32") not in {"fp32", "float32"}
                    or gate_meta.get("activation_group_size", latent) < latent
                    or tuple(gate_meta.get("activation_scale_shape", ())) != (expected_rows,)):
                raise ModelingError(
                    "moe_transport_gate_requires_int8_row_payload_and_fp32_row_scales")
            if dict(down.metadata).get("output_format") != "bf16":
                raise ModelingError("moe_transport_initial_branch_requires_bf16_expert_output")
            gate = replace(gate, metadata=gate.metadata + (
                ("activation_producer", "explicit_transport_quantization"),
                ("activation_payload_producer", transport_payload or transport_quantized.op_id),
                ("activation_scale_producer", transport_scale or transport_quantized.op_id),
                ("moe_transport_kind", transport.kind),
            ))
        activation = gated_activation(
            plan=plan, rows=local_rows, width=expert_width,
            op_id=prefix + ".expert_act", deps=(gate.op_id,), stage=stage,
            requests=requests, dtype=model.activation_dtype, phase=mode, recipe="grouped_swiglu",
        )
        ops.extend((gate, activation, down))
        if (dict(down.metadata).get("activation_scale_bytes") and runtime
                and "activation_quantization" in runtime.lowering_passes):
            group_role = ("grouped_gate_up_situ_quant" if activation.family == "situ"
                          else "grouped_gate_up_activation_quant")
            group_key = prefix + "." + group_role
            # The output quantizer is inserted by the ordinary lowering pass,
            # and belongs to the fused gate-up interface rather than down GEMM.
            ops[-1] = replace(down, metadata=down.metadata + (
                ("input_conversion_subgraphs", (group_key,)),))
            gate_meta, down_meta = dict(gate.metadata), dict(down.metadata)
            situ_mx_domain = (
                activation.family == "situ" and activation.dtype == gate_meta["output_format"] == "bf16"
                and gate_meta["activation_format"] == down_meta["activation_format"] == "fp8_e4m3"
                and gate_meta["weight_format"] == "mxfp4" and model.quant.expert_block == 32
                and gate_meta.get("activation_group_size") == down_meta.get("activation_group_size") == 32
                and gate_meta.get("activation_scale_format") ==
                down_meta.get("activation_scale_format") == "fp8_e8m0"
                and tuple(gate_meta.get("activation_scale_shape", ()))[1:] == (latent // 64, 2)
                and tuple(down_meta.get("activation_scale_shape", ()))[1:] == (expert_width // 64, 2)
                and latent % 64 == expert_width % 64 == 0
                and bool(dict(activation.metadata).get("activation_linear_beta")))
            situ_native_source = {
                "repository": "vllm-ascend",
                "base_revision": "75f52c1d7df8df8548d854a01b64935727a70e88",
                "source_state": "independent_local_delivery_overlay",
                "archive": "gmmsituquant-latest-0916/gmmsituquant-latest-overlay.zip",
                "archive_sha256": "fd5ded62862cabd7b686feceadcd8f39b692bdbb1989eb93c72eea29f4fc11da",
                "path": "csrc/moe/grouped_matmul_situ_quant/grouped_matmul_situ_quant_torch_adpt.h",
                "sha256": "d1bd2db4fedebcdd075cce5a0ce0022d145719bfdad7d5c2729308471a8771d3",
                "lines": [148, 184], "symbol": "RunV2Core",
                "scope": "independent_source_semantics; not_capture_binary_identity",
            }
            declare_group(
                ops, key=group_key, role=group_role, family="moe",
                members=(gate.op_id, activation.op_id), entries=(gate.op_id,),
                anchor=activation.op_id, block="ffn", scope_role="body",
                view={"input_width": latent, "intermediate_width": expert_width,
                      "local_experts": math.ceil(plan.num_experts / parallel.ep),
                      "routing_basis": dict(routing_metadata)["routing_basis"],
                      "input_dtype": dict(gate.metadata)["activation_format"],
                      "output_dtype": dict(down.metadata)["activation_format"],
                      "weight_dtype": dict(gate.metadata)["weight_format"],
                      "intermediate_dtype": gate_meta["output_format"],
                      "weight_scale_group_size": model.quant.expert_block,
                      "input_scale_dtype": gate_meta.get("activation_scale_format"),
                      "input_scale_group_size": gate_meta.get("activation_group_size"),
                      "input_scale_tail": tuple(gate_meta.get("activation_scale_shape", ()))[1:],
                      "output_scale_dtype": down_meta.get("activation_scale_format"),
                      "output_scale_group_size": down_meta.get("activation_group_size"),
                      "output_scale_tail": tuple(down_meta.get("activation_scale_shape", ()))[1:],
                      "activation": dict(activation.metadata)},
                implementation={"native_kinds": (
                    ["GroupedMatmulSituQuant"] if activation.family == "situ" else
                    ["GroupedMatmulSwigluQuant", "GroupedMatmulSwigluQuantV2"]),
                    "conditions": {"scope": "expert_gate_up_activation_output_quantization",
                                   "down_projection_external": True}},
                source=((situ_native_source if situ_mx_domain else SITU_SOURCE)
                        if activation.family == "situ" else source_reference(
                    "vllm_ascend/ops/fused_moe/moe_mlp.py", "grouped_matmul_swiglu_quant_v2")),
                coverage={"kind": ("native_source_domain_mismatch"
                                   if activation.family == "situ" and not situ_mx_domain
                                   else "complete_subgraph"),
                          "limitations": ["native_binary_and_scalar_flags_not_established",
                                          "expected_expert_counts_are_not_observed_routing"]},
            )
        last = down.op_id
        if parallel.etp > 1 and not reduce_after_join:
            reduce = comm_op(
                op_id=prefix + ".expert_reduce", deps=(last,), stage=stage, layer=plan.index,
                requests=requests, family="allreduce", phase=mode, recipe="expert_tensor_reduce",
                link_bytes=2 * (parallel.etp - 1) / parallel.etp * local_rows * latent * 2,
                shape=(math.ceil(local_rows), latent), metadata=(("group_axis", "etp"),))
            ops.append(reduce)
            last = reduce.op_id
    if parallel.ep > 1:
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
                link_bytes=exchange_bytes, shape=(tokens * plan.top_k, latent),
                metadata=collective_meta + (("collective_id", prefix + ".combine"),
                                           ("reverse_of_collective", prefix + ".dispatch")),
            )
        if transport is None:
            combine = masked_collective(combine, reverse=True)
        ops.append(combine)
        last = combine.op_id
    unpack_inputs = tuple(dict.fromkeys(
        (last, packing.op_id) + ((weights_producer,) if weights_producer is not None else ())))
    unpack_metadata = [
        ("payload_producer", last),
        ("inverse_index_producer", packing.op_id),
        ("topk_weights_producer", weights_producer),
        ("input_layout", "expert_grouped"),
        ("topk_ids_dtype", "int32"),
        ("router_weight_dtype", weight_dtype if weights_producer is not None else None),
        ("has_expert_map", False),
        ("apply_router_weight_on_input", False),
        ("ep_degree", parallel.ep),
        *routing_metadata,
    ]
    if routing_realization is not None:
        unpack_metadata.append(("input_rows", int(local_rows)))
    unpermute = vector_op(
        op_id=prefix + ".unpack", deps=unpack_inputs, stage=stage, layer=plan.index,
        requests=requests, elements=tokens * latent, ops_per_element=2 * plan.top_k,
        dtype=model.activation_dtype, family="routing", phase=mode, recipe="route_unpack",
        read_bytes_override=tokens * plan.top_k * (latent * 2 + 4),
        write_bytes_override=tokens * latent * 2,
        shape=(tokens, plan.top_k, latent),
        metadata=tuple(unpack_metadata),
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
        shared_input, shared_tokens = deps, tokens
        shared_prefix = []
        if sequence_plan:
            from .sequence_parallel import allgather, reduce_scatter
            gathered = allgather(
                plan=sequence_plan, op_id=prefix + ".shared.sp_gather", deps=(deps,),
                stage=stage, layer=plan.index, requests=requests, phase=mode,
                width=hidden, dtype=model.activation_dtype)
            shared_prefix.append(gathered)
            shared_input, shared_tokens = gathered.op_id, sequence_plan.tokens_global
        shared, shared_last = dense(
            prefix=prefix + ".shared", plan=replace(plan, intermediate_size=(
                plan.shared_intermediate_size or plan.num_shared_experts * plan.moe_intermediate_size)),
            deps=shared_input, stage=stage, requests=requests, tokens=shared_tokens, model=model,
            parallel=parallel, mode=mode, category="shared_expert",
            runtime=runtime, reduce_output=not reduce_after_join and sequence_plan is None,
        )
        shared = shared_prefix + shared
        if sequence_plan:
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
            shared = [replace(op, stream="shared") for op in shared]
        ops.extend(shared)
        join = vector_op(
            op_id=prefix + ".join", deps=(last, shared_last), stage=stage, layer=plan.index,
            requests=requests, elements=tokens * hidden, ops_per_element=1,
            dtype=model.activation_dtype, family="elementwise", phase=mode, recipe="expert_add",
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
        ops.append(reduction)
        last = reduction.op_id
    from .mc2_graph import lower as lower_native_mc2
    ops = lower_native_mc2(
        ops, runtime=runtime, prefix=prefix, tokens=routed_tokens, top_k=plan.top_k,
        width=latent, experts=plan.num_experts, ep=parallel.ep,
        source_partitions=partitions, received_assignments=local_rows,
        dtype=model.activation_dtype, source_routing=source_traffic)
    return ops, last


def tp_join_reduce(*, plan, parallel, model, **kwargs):
    """TP-sharded experts and shared output reduce once after linear combine."""
    from mindie_sim.schema.common import ModelingError

    if parallel.ep != 1 or parallel.dp != 1 or parallel.cp != 1 or parallel.pcp != 1:
        raise ModelingError("tp_join_reduce_requires_tp_only_non_sequence_parallel")
    if plan.expert_latent_size not in (0, None, model.hidden_size):
        raise ModelingError("tp_join_reduce_cannot_defer_across_routed_output_transform")
    # This explicit recipe declares TP-sharded shared weights and replicated
    # router/shared gates. Their linear combine commutes with the final sum.
    return alltoall(plan=plan, parallel=parallel, model=model,
                   reduce_after_join=True, **kwargs)


def mc2(*, prefix, tokens, global_tokens=None, chunks=4, **kwargs):
    """Finite dispatch/compute/combine chunks with separate ordered streams."""
    group = tokens if global_tokens is None else global_tokens
    count = min(chunks, max(1, group))
    if global_tokens is not None and (tokens % count or group % count):
        # Dividing the cohort total and then dividing each source are not
        # interchangeable at chunk tails. No source partition was supplied.
        from mindie_sim.schema.common import ModelingError
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
            ops.append(replace(op, stream=f"mc2:{phase}"))
        completions.append(last)
    joined = barrier_op(
        op_id=prefix + ".chunks_complete", deps=tuple(completions), stage=kwargs["stage"],
        layer=kwargs["plan"].index, requests=kwargs["requests"], phase=kwargs["mode"],
        recipe="mc2_join", shape=(tokens, kwargs["model"].hidden_size))
    ops.append(joined)
    return ops, joined.op_id


MOE_RECIPES = {"alltoall": alltoall, "mc2": mc2, "tp_join_reduce": tp_join_reduce}


def moe(*, moe_recipe="alltoall", mc2_chunks=4, **kwargs):
    return MOE_RECIPES[moe_recipe](chunks=mc2_chunks, **kwargs)


FFN_BUILDERS = {"dense": dense, "moe": moe}
