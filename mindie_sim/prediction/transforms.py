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


def _users(ops):
    users = defaultdict(list)
    for op in ops:
        for dep in op.deps:
            users[dep].append(op.op_id)
    return users


def _fusion_metadata(*ops):
    merged, origins = {}, {}
    for op in ops:
        meta = dict(op.metadata)
        merged.update(meta)
        for origin in meta.get("fused_origin_scopes", ()):
            origins[origin["op_id"]] = origin
        origins[op.op_id] = {
            "op_id": op.op_id, "layer": op.layer, "stage": op.stage,
            "block": meta.get("structure_block"), "role": meta.get("structure_role"),
        }
    for key in ("structure_subgraph_memberships", "structure_subgraph_entries",
                "structure_subgraphs"):
        values = []
        for op in ops:
            for value in dict(op.metadata).get(key, ()):
                if value not in values:
                    values.append(value)
        if values:
            merged[key] = tuple(values)
    merged["fused_origin_scopes"] = tuple(origins.values())
    return merged


def activation_quantization(ops, source_dtype):
    result, conversions, positions = [], {}, {}
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
        output_dtype = meta["activation_format"]
        signature = (op.deps, op.stage, op.stream, rows, cols, source_dtype, output_dtype,
                     meta["activation_group_size"], meta["activation_scale_shape"],
                     meta["activation_scale_format"], expected_rows)
        if signature not in conversions:
            output = expected_rows * cols * dtype_bytes(output_dtype) + meta["activation_scale_bytes"]
            groups = expected_rows * math.ceil(cols / meta["activation_group_size"])
            # Per-group absolute maximum, scale construction, rescale and cast.
            vector = 4 * expected_rows * cols - groups + 3 * groups
            quant_meta = {
                "output_format": output_dtype,
                **{key: meta[key] for key in (
                    "activation_group_size", "activation_scale_shape",
                    "activation_scale_format", "activation_scale_bytes")},
                **{key: meta[key] for key in ("structure_block", "structure_role") if key in meta},
            }
            if meta.get("input_conversion_subgraphs"):
                quant_meta["structure_subgraph_memberships"] = meta["input_conversion_subgraphs"]
            quantized = vector_op(
                op_id=op.op_id + ".quantize", deps=op.deps, stage=op.stage, layer=op.layer,
                requests=op.requests, elements=rows * cols, ops_per_element=0,
                vector_ops_override=vector, dtype=source_dtype, family="quantize", phase=op.phase,
                recipe="dynamic_quantize", shape=(rows, cols),
                read_bytes_override=expected_rows * cols * dtype_bytes(source_dtype),
                write_bytes_override=output, returns=("activation_scales",),
                metadata=tuple(quant_meta.items()),
            )
            quantized = replace(quantized, stream=op.stream)
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
        result.append(replace(op, deps=(conversions[signature],),
                              metadata=op.metadata + (("activation_producer", "explicit_quantization"),)))
    return result


def _norm_quant_domain(kernels):
    domains = kernels.get("fusion_domains", {})
    if not isinstance(domains, Mapping) or set(domains) - {"norm_quant"}:
        raise ModelingError("fusion_domains_require_declared_norm_quant_consumer_domain")
    raw = domains.get("norm_quant")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {"consumers"}:
        raise ModelingError("norm_quant_domain_requires_consumers")
    selectors = raw["consumers"]
    fields = {"recipe", "family", "structure_block", "structure_role"}
    if (not isinstance(selectors, (list, tuple)) or any(
            not isinstance(item, Mapping) or not item or set(item) - fields
            or any(not isinstance(value, str) or not value for value in item.values())
            for item in selectors)):
        raise ModelingError("norm_quant_consumers_require_semantic_recipe_family_or_structure_selectors")
    return tuple(dict(item) for item in selectors)


def _consumers_match(consumer_ids, by_id, selectors):
    if selectors is None:
        return True
    # A shared payload/scales conversion is one producer. Fusion can only be
    # selected if every consumer permits the complete fused call boundary.
    def matches(op, rule):
        metadata = dict(op.metadata)
        return all((getattr(op, key) if key in {"recipe", "family"} else
                    metadata.get(key)) == value for key, value in rule.items())
    return bool(consumer_ids) and all(any(matches(by_id[node], rule) for rule in selectors)
                                      for node in consumer_ids)


def norm_quant(ops, _, *, consumers=None):
    by_id, users = {op.op_id: op for op in ops}, _users(ops)
    replacements, removed = {}, set()
    for op in ops:
        if op.plan_required or op.family != "quantize" or len(op.deps) != 1:
            continue
        norm = by_id.get(op.deps[0])
        if (norm is None or norm.plan_required or norm.family != "rmsnorm" or norm.local_shape != op.local_shape
                or norm.stage != op.stage or norm.stream != op.stream
                or users[norm.op_id] != [op.op_id]):
            continue
        if not _consumers_match(users[op.op_id], by_id, consumers):
            continue
        read, write = norm.mte2_bytes, op.mte3_bytes
        meta = {**_fusion_metadata(norm, op), "fused_from": ("rmsnorm", "quantize")}
        if consumers is not None:
            meta["fusion_consumer_domain"] = {"pass": "norm_quant", "consumers": consumers}
        replacements[op.op_id] = replace(
            op, family="rmsnorm_quant", recipe="rmsnorm_quantize", deps=norm.deps,
            vector_ops=norm.vector_ops + op.vector_ops,
            mte2_bytes=read, hbm_bytes=read + write, metadata=tuple(meta.items()),
        )
        removed.add(norm.op_id)
    return [replacements.get(op.op_id, op) for op in ops if op.op_id not in removed]


def residual_norm(ops, _):
    by_id, users = {op.op_id: op for op in ops}, _users(ops)
    positions = {op.op_id: index for index, op in enumerate(ops)}
    replacements, aliases, removed = {}, {}, set()
    for norm in ops:
        if (norm.plan_required or "implementation_binding" in dict(norm.metadata)
                or norm.family not in {"rmsnorm", "rmsnorm_quant"}
                or len(norm.deps) != 1):
            continue
        residual = by_id.get(norm.deps[0])
        if (residual is None or residual.plan_required or residual.recipe != "residual_add"
                or residual.op_id in removed):
            continue
        if (norm.stage != residual.stage or norm.stream != residual.stream
                or math.prod(norm.local_shape) != math.prod(residual.local_shape)
                or any(positions[user] < positions[norm.op_id] for user in users[residual.op_id])):
            continue
        # The residual is still consumed by subsequent sublayers. It remains an
        # explicit output of the fused operator, unlike the eliminated norm input.
        read = residual.mte2_bytes + norm.mte2_bytes - residual.mte3_bytes
        write = norm.mte3_bytes + residual.mte3_bytes
        quantized = norm.family == "rmsnorm_quant"
        meta = {**_fusion_metadata(residual, norm), "fused_from": (
            "residual_add", "rmsnorm", *(("quantize",) if quantized else ()))}
        replacements[norm.op_id] = replace(
            norm, deps=residual.deps, family="add_rmsnorm_quant" if quantized else "add_rmsnorm",
            recipe="add_rmsnorm_quantize" if quantized else "add_rmsnorm",
            vector_ops=residual.vector_ops + norm.vector_ops,
            mte2_bytes=read, mte3_bytes=write, hbm_bytes=read + write,
            metadata=tuple(meta.items()), returns=(*norm.returns, "residual"),
        )
        aliases[residual.op_id] = norm.op_id
        removed.add(residual.op_id)
    result = []
    for op in ops:
        if op.op_id in removed:
            continue
        op = replacements.get(op.op_id, op)
        result.append(replace(op, deps=tuple(dict.fromkeys(aliases.get(dep, dep) for dep in op.deps))))
    return result


LOWERING_PASSES = {
    "activation_quantization": activation_quantization,
    "norm_quant": norm_quant,
    "residual_norm": residual_norm,
}


def lower_implementations(ops, runtime, source_dtype, *, passes=None, resource_fusion=True,
                          local_tokens=None):
    from .implementations import lower_operators, select

    bindings = runtime.kernels.get("lowerings", {})
    from .family_shared_quant import lower_shared_experts
    # Bind explicitly selected kernels before generic fusion can erase them.
    # A fused-family selection binds after that family is actually produced.
    ops = lower_operators(ops, bindings)
    from .family_moe_init import apply as apply_moe_init
    ops = apply_moe_init(ops, bindings, runtime)
    registry = {**LOWERING_PASSES, "norm_quant": partial(
        norm_quant, consumers=_norm_quant_domain(runtime.kernels)), **(passes or {})}
    activation_quantized = False
    for name in runtime.lowering_passes:
        activation_quantized |= name == "activation_quantization"
        ops = lower_operators(registry[name](ops, source_dtype), bindings,
                              activation_quantized=activation_quantized)
    ops = lower_operators(ops, bindings, activation_quantized=activation_quantized, final=True)
    ops = lower_shared_experts(ops, bindings)
    from .family_grouped_quant import lower as lower_grouped_int8
    ops = lower_grouped_int8(ops, bindings)
    from .moe_group_list import lower as lower_group_list
    ops = lower_group_list(ops, bindings)
    wrapper = select(bindings, "matrix_wrapper")
    if wrapper:
        ops = wrapper.module.apply_matrix_wrapper(
            ops, wrapper, runtime, local_tokens=local_tokens)
    mx_matrix = select(bindings, "mx_matrix")
    if mx_matrix:
        ops = mx_matrix.module.apply(ops, mx_matrix, local_tokens=local_tokens)
    from .hccl_graph import lower as lower_hccl
    ops = lower_hccl(ops, runtime, source_dtype)
    if resource_fusion:
        from .native_fusion import lower_native_fusions
        ops = lower_native_fusions(ops, runtime, local_tokens=local_tokens)
    return tuple(ops)
