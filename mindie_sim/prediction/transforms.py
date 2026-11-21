"""Representation conversions and structural kernel fusion.

[Role] Lower semantic tensor producers into software-selected device operators.
[Invariants] A low-precision input has a producer for payload and scales.
  Fusion removes intermediate traffic but preserves live residual outputs.
  Source planners own their conversions and kernel boundaries; generic passes
  do not rewrite those declared implementations.
  Passes match tensor contracts and dependencies, never model names.
[Timeouts & Locks] Linear passes over an active graph, no I/O or locks.
[Side-Effect Policy] Returns new immutable operators.
"""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
import math

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from .fusion_regions import fold_region_metadata_dict
from .implementations import (
    add_input_ports, append_audit,
    capture_fusion_overlap, field_disposition, input_ports,
    set_input_ports,
    merge_typed_roles, producer_input_values, produced_values,
    typed_outputs,
)
from .rewrite_ffn_rules import (
    activation_quant_fusion,
    gmm_swiglu_quant_fusion,
    moe_tail_residual_fusion,
)
from .rewrite_indexer_rules import online_indexer_topk
from .rewrite_mhc_rules import (
    MHC_DEFAULT_CUTOFF_ROWS,
    gdn_projection_overlap,
    kda_in_kernel_l2norm,
    mhc_boundary_fusion_overlap,
    qkv_norm_rope_fusion,
)
from mindie_sim.prediction.operators.vector.transform_norm_quant import (
    _norm_quant_domain,
    norm_quant,
    residual_norm,
)

TORCH_COMPILE_PASSES = frozenset({
    "norm_quant",
    "activation_quant",
    "gmm_swiglu_quant",
    "moe_tail_residual",
})
NPUGRAPH_EX_PASSES = frozenset({
    "gmm_swiglu_quant",
    "moe_tail_residual",
})


# A selected source subgraph owns its own kernel boundaries.  A generic rewrite
# of the same operators is refused instead of being applied before that source.
PROTECTED_PASS_ROLES = {
    "indexer_online_topk": frozenset({"lightning_indexer",
                                      "quant_lightning_indexer"}),
}
#: Generic passes whose fusion domain overlaps an explicitly selected native
#: group.  Members of those groups are excluded from the generic rewrite; other
#: independent groups in the same graph remain eligible.
NATIVE_EXCLUSION_ROLES = {
    "activation_quant": frozenset({
        "situ_quant", "grouped_gate_up_situ_quant",
        "grouped_gate_up_activation_quant"}),
    "gmm_swiglu_quant": frozenset({
        "situ_quant", "grouped_gate_up_situ_quant",
        "grouped_gate_up_activation_quant"}),
    "mhc_boundary_pr": frozenset({"hc_pre", "hc_post"}),
    "mhc_boundary_ideal": frozenset({"hc_pre", "hc_post"}),
}


def _native_group_members(ops, roles):
    """Members of declared native groups whose role is explicitly selected."""
    if not roles:
        return frozenset()
    from mindie_sim.prediction.operators.hyper_connections.native_fusion import _groups
    members = set()
    for declaration, group in _groups(ops):
        if declaration.get("role") in roles:
            members.update(op.op_id for op in group)
    return frozenset(members)


def _rewrite_outside_native_groups(fn, ops, source_dtype, excluded):
    """Run one generic pass with selected native-group members left intact."""
    kept = [op for op in ops if op.op_id not in excluded]
    transformed = list(fn(kept, source_dtype))
    transformed_by_id = {op.op_id: op for op in transformed}
    replacement_of = {}
    for op in transformed:
        for origin in dict(op.metadata).get("fused_origin_scopes", ()):
            source = origin.get("op_id") if isinstance(origin, dict) else None
            if source and source not in transformed_by_id:
                replacement_of.setdefault(source, op.op_id)
    rebuilt = []
    for op in ops:
        if op.op_id in transformed_by_id:
            rebuilt.append(transformed_by_id[op.op_id])
        elif op.op_id in excluded:
            rebuilt.append(op)
    seen = {op.op_id for op in rebuilt}
    rebuilt.extend(op for op in transformed if op.op_id not in seen)
    aliased = []
    for op in rebuilt:
        deps = tuple(dict.fromkeys(
            replacement_of.get(dep, dep) for dep in op.deps))
        aliased.append(replace(op, deps=deps) if deps != op.deps else op)
    # If a reader still names a rewritten-away member and the pass declared no
    # replacement for it, keep the original declared graph.
    removed = {op.op_id for op in kept} - set(transformed_by_id)
    if any(dep in removed for op in aliased for dep in op.deps):
        return list(ops)
    return aliased


def _fusion_metadata(*ops):
    return fold_region_metadata_dict(ops)


def activation_quantization(ops, source_dtype):
    result, conversions, positions = [], {}, {}
    outputs_by_id = {}
    for op in ops:
        try:
            outputs_by_id[op.op_id] = produced_values(op)
        except ModelingError:
            outputs_by_id[op.op_id] = ()
    for op in ops:
        meta = dict(op.metadata)
        if (op.plan_required or meta.get("program_kind") != "matmul"
                or not meta.get("activation_scale_bytes")
                or meta.get("activation_producer") in {
                    "explicit_transport_quantization", "explicit_routing_quantization",
                    "explicit_quantization"}):
            # Selected callers or an earlier pass already produced this
            # low-precision payload and scales. Preserve that producer;
            # other matmuls still receive their ordinary conversion.
            result.append(op)
            continue
        rows, _, cols = op.local_shape
        expected_rows = meta.get("expected_rows", rows)
        shape_rows = int(math.ceil(expected_rows)) if expected_rows > 0 else 0
        output_dtype = meta["activation_format"]
        inputs = producer_input_values(op, outputs_by_id, source_dtype)
        if not inputs:
            raise ModelingError("activation_quantization_requires_typed_operand:" + op.op_id)
        input_value = inputs[0]
        read_dtype = str(input_value["dtype"])
        # A conversion has its own access plan.  It inherits representation,
        # not the matrix's repeats; the matrix repeats move to the payload.
        storage_shape = tuple(input_value.get("storage_shape") or ())
        logical_shape = tuple(input_value.get("logical_shape") or ())
        if (storage_shape and logical_shape and storage_shape != logical_shape
                and expected_rows == storage_shape[0]):
            read_bytes = float(expected_rows * cols * dtype_bytes(read_dtype))
        else:
            read_bytes = input_value.get("base_read_bytes")
            if read_bytes is None:
                read_bytes = expected_rows * cols * dtype_bytes(read_dtype)
        conversion_input = {**input_value, "read_bytes": read_bytes,
                            "repeats": 1, "access_shape": list(
                                input_value.get("access_shape")
                                or input_value.get("logical_shape") or [])}
        # The cache key carries the complete conversion contract.  A new
        # batch/route/layer/phase must not reuse a conversion bound to another
        # live producer merely because its element count is equal.
        layout = input_value.get("layout") or {}
        signature = (
            op.deps, op.stage, op.phase, op.layer, op.requests, op.stream,
            rows, cols, op.dtype, source_dtype, read_dtype, output_dtype,
            meta["activation_group_size"], meta["activation_scale_shape"],
            meta["activation_scale_format"], expected_rows,
            input_value.get("producer"), input_value.get("output"),
            tuple(layout.get("strides", ())) if isinstance(layout, dict) else (),
            layout.get("offset", 0) if isinstance(layout, dict) else 0,
            tuple(layout.get("storage_shape", ())) if isinstance(layout, dict) else (),
            input_value.get("state_version"), input_value.get("binding"),
            input_value.get("binding_name"), read_bytes,
        )
        if signature not in conversions:
            output = expected_rows * cols * dtype_bytes(output_dtype) + meta["activation_scale_bytes"]
            groups = expected_rows * math.ceil(cols / meta["activation_group_size"])
            # Per-group absolute maximum, scale construction, rescale and cast.
            vector = 4 * expected_rows * cols - groups + 3 * groups
            quant_meta = {
                "output_format": output_dtype,
                "primary_output": "payload",
                **{key: meta[key] for key in (
                    "activation_group_size", "activation_scale_shape",
                    "activation_scale_format", "activation_scale_bytes")},
                **{key: meta[key] for key in ("structure_block", "structure_role",
                    "graph_active_query_rows", "graph_physical_query_rows") if key in meta},
            }
            if meta.get("input_conversion_subgraphs"):
                quant_meta["structure_subgraph_memberships"] = meta["input_conversion_subgraphs"]
            quantized = vector_op(
                op_id=op.op_id + ".quantize", deps=op.deps, stage=op.stage, layer=op.layer,
                requests=op.requests, elements=rows * cols, ops_per_element=0,
                vector_ops_override=vector, dtype=read_dtype, family="quantize",
                phase=op.phase, recipe="dynamic_quantize", shape=(shape_rows, cols),
                read_bytes_override=read_bytes,
                write_bytes_override=output, returns=("activation_scales",),
                metadata=tuple(quant_meta.items()),
            )
            quantized = replace(quantized, stream=op.stream)
            outputs = typed_outputs(quantized)
            quant_meta = dict(quantized.metadata)
            audit = {
                "kind": "activation_materialization",
                "replaces_work": True, "replaces_allocation": True,
                "replaced_occurrences": (),
                "replacement_occurrence": quantized.op_id,
                "work": {"mte2_bytes": quantized.mte2_bytes,
                         "mte3_bytes": quantized.mte3_bytes,
                         "hbm_bytes": quantized.hbm_bytes,
                         "vector_ops": quantized.vector_ops,
                         "flops": quantized.flops},
                "allocations": {"inserted": (quantized.op_id,),
                                "materialized": tuple(value["output"] for value in outputs)},
                "external_outputs": (),
                "external_consumers": (),
                "events": {"deps": list(quantized.deps),
                           "stream": quantized.stream or "main",
                           "outputs": list(quantized.returns)},
                "capture": capture_fusion_overlap((quantized,), kind="activation")["capture"],
                "overlap": capture_fusion_overlap((quantized,), kind="activation")["overlap"],
                "capture_fusion_overlap_separate": True,
                "typed_inputs": (conversion_input,),
                "typed_outputs": outputs,
                "typed_roles": merge_typed_roles(op, quantized),
                "output_ports": tuple(
                    {"output": value["output"], "dtype": value["dtype"],
                     "role": value["role"]} for value in outputs),
                "field_dispositions": (
                    field_disposition("activation.input", "consume", op.op_id,
                                      "read once by grouped dynamic quantization"),
                    field_disposition("activation.payload", "rewrite", quantized.op_id,
                                      "low-precision payload materialized"),
                    field_disposition("activation.scales", "rewrite", quantized.op_id,
                                      "scale output is an independent value"),
                    field_disposition("activation.group_size", "consume", op.op_id,
                                      "grouped amax and scale construction"),
                    field_disposition("matrix.activation_operand", "rewrite", quantized.op_id,
                                      "consumer edge is rebound to the payload port"),
                ),
                "materialization": {
                    "kind": "implicit_activation_conversion",
                    "inserted_occurrence": quantized.op_id,
                    "rewired_consumer": op.op_id,
                    "replaces_implicit_conversion": True,
                    "outputs": tuple(value["output"] for value in outputs),
                },
            }
            append_audit(quant_meta, kind="activation_materialization",
                         op_id=quantized.op_id, boundary=audit,
                         layer=quantized.layer, stage=quantized.stage,
                         block=quant_meta.get("structure_block"),
                         role=quant_meta.get("structure_role"))
            quantized = replace(quantized, metadata=tuple(quant_meta.items()))
            old_producer = input_value.get("producer") or (op.deps[0] if op.deps else "")
            old_output = input_value.get("output") or "output"
            quantized = add_input_ports(quantized, ({
                "op": old_producer, "output": old_output,
                "source_op": old_producer},))
            result.append(quantized)
            conversions[signature] = quantized.op_id
            positions[quantized.op_id] = len(result) - 1
        elif meta.get("input_conversion_subgraphs"):
            index = positions[conversions[signature]]
            quantized = result[index]
            quant_meta = dict(quantized.metadata)
            quant_meta["structure_subgraph_memberships"] = tuple(dict.fromkeys((
                *quant_meta.get("structure_subgraph_memberships", ()),
                *meta["input_conversion_subgraphs"],
            )))
            result[index] = replace(quantized, metadata=tuple(quant_meta.items()))
        old_producer = input_value.get("producer") or (op.deps[0] if op.deps else "")
        old_output = input_value.get("output") or "output"
        consumer_meta = dict(op.metadata)
        for key in ("input_repeats", "operand_read_bytes"):
            mapping = consumer_meta.get(key)
            if isinstance(mapping, dict):
                moved = dict(mapping)
                for old_key in (old_producer, old_output):
                    if old_key in moved:
                        moved["payload"] = moved.pop(old_key)
                consumer_meta[key] = moved
        consumer_meta["activation_producer"] = "explicit_quantization"
        consumer = replace(op, deps=(conversions[signature],),
                           metadata=tuple(consumer_meta.items()))
        existing = [entry for entry in input_ports(op)
                    if entry.get("op") != old_producer]
        new_ports = [{"op": conversions[signature], "output": "payload",
                      "source_op": old_producer}]
        if meta.get("activation_scale_bytes"):
            new_ports.append({"op": conversions[signature],
                              "output": "activation_scales",
                              "source_op": old_producer})
        consumer = set_input_ports(
            consumer, tuple(existing + new_ports))
        result.append(consumer)
    return result


LOWERING_PASSES = {
    "activation_quantization": activation_quantization,
    "norm_quant": norm_quant,
    "residual_norm": residual_norm,
    "indexer_online_topk": online_indexer_topk,
    "mhc_boundary_pr": partial(mhc_boundary_fusion_overlap,
                               max_rows=MHC_DEFAULT_CUTOFF_ROWS,
                               require_captured=True),
    "mhc_boundary_ideal": partial(mhc_boundary_fusion_overlap,
                                  max_rows=0, require_captured=False),
    "activation_quant": activation_quant_fusion,
    "gmm_swiglu_quant": gmm_swiglu_quant_fusion,
    "qkv_norm_rope": qkv_norm_rope_fusion,
    "kda_in_kernel_l2norm": kda_in_kernel_l2norm,
    "gdn_projection_overlap": gdn_projection_overlap,
    "moe_tail_residual": moe_tail_residual_fusion,
}


def insert_representations(ops, runtime, source_dtype):
    """Insert representation/source nodes before the one actual partition."""
    inserted = False
    result = tuple(ops)
    for name in runtime.lowering_passes:
        if name == "activation_quantization":
            result = tuple(activation_quantization(result, source_dtype))
            inserted = True
    return result, inserted


def lower_implementations(ops, runtime, source_dtype, *, passes=None, resource_fusion=True,
                          local_tokens=None, replay_scope=None, capture_state=None,
                          skip_passes=(), preactivated=False):
    from .implementations import capture_partition, lower_operators, select

    bindings = runtime.kernels.get("lowerings", {})
    from mindie_sim.prediction.operators.mx_quant_matrix.family_shared_quant import lower_shared_experts
    # Bind explicitly selected kernels before generic fusion can erase them.
    # A fused-family selection binds after that family is actually produced.
    ops = lower_operators(ops, bindings)
    from mindie_sim.prediction.operators.moe_routing.family_moe_init import apply as apply_moe_init
    ops = apply_moe_init(ops, bindings, runtime)
    consumers, external_output = _norm_quant_domain(runtime.kernels)
    registry = {**LOWERING_PASSES, "norm_quant": partial(
        norm_quant, consumers=consumers, external_output=external_output),
        **(passes or {})}
    activation_quantized = bool(preactivated)
    native = select(bindings, "native_fusion")
    native_roles = set(native.options.get("roles", ())) if native else set()
    enable_compile = bool(getattr(runtime, "enable_torch_compile", True))
    enable_npugraph_ex = bool(
        enable_compile and getattr(runtime, "enable_npugraph_ex", True)
    )
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    qkv_fx_only = rt_meta.get("qkv_norm_rope_mode") == "fx"
    for name in runtime.lowering_passes:
        if name in skip_passes:
            continue
        if name in TORCH_COMPILE_PASSES and not enable_compile:
            continue
        if name in NPUGRAPH_EX_PASSES and not enable_npugraph_ex:
            continue
        if name == "qkv_norm_rope" and qkv_fx_only and not enable_npugraph_ex:
            continue
        protected = PROTECTED_PASS_ROLES.get(name)
        if protected and protected.intersection(native_roles):
            continue
        excluded_roles = NATIVE_EXCLUSION_ROLES.get(name)
        excluded = (_native_group_members(ops, excluded_roles & native_roles)
                    if excluded_roles else frozenset())
        activation_quantized |= name == "activation_quantization"
        if name == "norm_quant":
            state = (capture_state if capture_state is not None
                     else capture_partition(ops, runtime, replay_scope))
            transformed = norm_quant(
                ops, source_dtype, consumers=consumers,
                external_output=external_output, capture_state=state)
        elif name == "residual_norm":
            state = (capture_state if capture_state is not None
                     else capture_partition(ops, runtime, replay_scope))
            transformed = residual_norm(ops, source_dtype, capture_state=state)
        elif excluded:
            transformed = _rewrite_outside_native_groups(
                registry[name], ops, source_dtype, excluded)
        else:
            transformed = registry[name](ops, source_dtype)
        ops = lower_operators(transformed, bindings,
                              activation_quantized=activation_quantized)
        if capture_state is not None:
            generated = capture_partition(ops, runtime, replay_scope)
            if generated:
                for identifier, value in generated.items():
                    capture_state.setdefault(identifier, value)
    ops = lower_operators(ops, bindings, activation_quantized=activation_quantized, final=True)
    ops = lower_shared_experts(ops, bindings)
    grouped_int8 = select(bindings, "grouped_int8_caller")
    if grouped_int8:
        ops = grouped_int8.module.lower(ops, bindings)
    from mindie_sim.prediction.operators.moe_routing.moe_group_list import lower as lower_group_list
    ops = lower_group_list(ops, bindings)
    wrapper = select(bindings, "matrix_wrapper")
    if wrapper:
        ops = wrapper.module.apply_matrix_wrapper(
            ops, wrapper, runtime, local_tokens=local_tokens)
    mx_matrix = select(bindings, "mx_matrix")
    if mx_matrix:
        ops = mx_matrix.module.apply(ops, mx_matrix, local_tokens=local_tokens)
    from mindie_sim.prediction.operators.collectives.hccl_graph import lower as lower_hccl
    ops = lower_hccl(ops, runtime, source_dtype)
    if resource_fusion:
        from mindie_sim.prediction.operators.hyper_connections.native_fusion import lower_native_fusions
        ops = lower_native_fusions(ops, runtime, local_tokens=local_tokens)
    return tuple(ops)


def lower_implementations_with_capture(ops, runtime, source_dtype, *,
                                        replay_scope=None, capture_state=None,
                                        **kwargs):
    """Return the lowered ops and the complete capture state.

    The state includes identifiers created by representation insertion
    (for example a quantizer), so a public body entry must consume this return
    instead of copying an older pre-insertion mapping.
    """
    state = dict(capture_state) if capture_state is not None else {}
    lowered = lower_implementations(
        ops, runtime, source_dtype, replay_scope=replay_scope,
        capture_state=state, **kwargs)
    return lowered, state
