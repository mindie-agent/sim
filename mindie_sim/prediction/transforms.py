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
from .implementations import (
    add_input_ports, append_audit, attach_quantized_ports,
    capture_fusion_overlap, field_disposition, input_ports, member_contracts,
    set_input_ports,
    merge_typed_roles, nested_boundary, producer_input_values, produced_values,
    replaced_occurrences, rewire_replacement, rewire_replacements,
    semantic_attrs, typed_outputs,
    typed_roles, typed_value,
)
from .passes_ffn_fusion import (
    activation_quant_fusion,
    gmm_swiglu_quant_fusion,
    moe_tail_residual_fusion,
)
from .passes_mhc_recurrent import (
    MHC_DEFAULT_CUTOFF_ROWS,
    gdn_projection_overlap,
    kda_in_kernel_l2norm,
    mhc_boundary_fusion_overlap,
    qkv_norm_rope_fusion,
)


def _value_map(ops):
    result = {}
    for op in ops:
        try:
            result[op.op_id] = produced_values(op)
        except ModelingError:
            result[op.op_id] = ()
    return result


def _first_value(values):
    return values[0] if values else None


def _declared_input_bytes(op):
    metadata = dict(op.metadata)
    dtype = str(metadata.get("input_format") or metadata.get("activation_format")
                or op.dtype or "")
    try:
        return float(math.prod(op.local_shape) * dtype_bytes(dtype))
    except (KeyError, ValueError, TypeError):
        return 0.0


def _same_representation(op, values):
    if not values:
        return True
    metadata = dict(op.metadata)
    if metadata.get("input_repeats") or metadata.get("operand_read_bytes"):
        return False
    declared = str(metadata.get("input_format") or metadata.get("activation_format")
                   or op.dtype or "")
    elements = math.prod(op.local_shape)
    for value in values:
        if str(value.get("dtype")) != declared:
            return False
        logical_elems = math.prod(value.get("logical_shape") or ())
        storage_elems = math.prod(value.get("storage_shape") or ())
        written_bytes = value.get("write_bytes")
        try:
            expected_bytes = elements * dtype_bytes(declared)
        except (KeyError, ValueError, TypeError):
            expected_bytes = None
        if (logical_elems != elements and storage_elems != elements
                and (written_bytes is None or expected_bytes is None
                     or not math.isclose(float(written_bytes), float(expected_bytes)))):
            return False
    return True


def _parameter_read_bytes(op):
    metadata = dict(op.metadata)
    declared = (metadata.get("parameter_read_bytes")
                or metadata.get("parameter_reads") or {})
    if isinstance(declared, dict):
        total = sum(float(value) for value in declared.values())
    elif isinstance(declared, (int, float)) and not isinstance(declared, bool):
        total = float(declared)
    else:
        total = 0.0
    for key in ("norm_gamma_runtime_bytes", "norm_gamma_bytes",
                "gamma_read_bytes", "parameter_bytes"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total = max(total, float(value))
    return total


def _access_read(values, fallback):
    reads = [value.get("read_bytes") for value in values
             if value.get("read_bytes") is not None]
    return sum(reads) if reads else fallback


def _read_bytes(value, op, *, fallback=0.0):
    if value is None:
        return fallback
    actual = value.get("write_bytes")
    if actual is None:
        actual = value.get("storage_bytes") or value.get("bytes")
    if actual is None:
        return fallback
    metadata = dict(op.metadata)
    declared_dtype = str(metadata.get("input_format") or metadata.get("activation_format")
                         or op.dtype or "")
    declared = fallback
    try:
        declared = math.prod(op.local_shape) * dtype_bytes(declared_dtype)
    except (KeyError, ValueError, TypeError):
        declared = fallback
    return float(actual) + max(0.0, float(op.mte2_bytes) - float(declared))


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


def _norm_quant_domain(kernels):
    domains = kernels.get("fusion_domains", {})
    if not isinstance(domains, Mapping) or set(domains) - {"norm_quant"}:
        raise ModelingError("fusion_domains_require_declared_norm_quant_consumer_domain")
    raw = domains.get("norm_quant")
    if raw is None:
        return None, "eliminate"
    if not isinstance(raw, Mapping) or set(raw) - {"consumers", "external_outputs"}:
        raise ModelingError("norm_quant_domain_requires_consumers")
    if "consumers" not in raw:
        raise ModelingError("norm_quant_domain_requires_consumers")
    selectors = raw["consumers"]
    fields = {"recipe", "family", "structure_block", "structure_role"}
    if (not isinstance(selectors, (list, tuple)) or any(
            not isinstance(item, Mapping) or not item or set(item) - fields
            or any(not isinstance(value, str) or not value for value in item.values())
            for item in selectors)):
        raise ModelingError("norm_quant_consumers_require_semantic_recipe_family_or_structure_selectors")
    external = raw.get("external_outputs", "eliminate")
    if external not in {"eliminate", "materialize"}:
        raise ModelingError("norm_quant_external_output_mode_unknown")
    return tuple(dict(item) for item in selectors), external


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


def _capture_state(op, partition):
    metadata = dict(op.metadata)
    fallback = (metadata.get("graph_captured"), metadata.get("graph_segment"))
    if partition is not None:
        return partition.get(op.op_id, fallback)
    return fallback


def norm_quant(ops, _, *, consumers=None, external_output="eliminate",
               capture_state=None):
    if external_output not in {"eliminate", "materialize"}:
        raise ModelingError("norm_quant_external_output_mode_unknown")
    by_id, users = {op.op_id: op for op in ops}, _users(ops)
    positions = {op.op_id: index for index, op in enumerate(ops)}
    values_by_id = _value_map(ops)
    replacements, removed, ports, removed_map = {}, set(), {}, {}
    for op in ops:
        if op.plan_required or op.family != "quantize" or len(op.deps) != 1:
            continue
        norm = by_id.get(op.deps[0])
        if (norm is None or norm.plan_required or norm.family != "rmsnorm"
                or norm.local_shape != op.local_shape or norm.stage != op.stage
                or norm.stream != op.stream):
            continue
        if op.op_id not in users[norm.op_id]:
            continue
        external = [user for user in users[norm.op_id] if user != op.op_id]
        norm_values = produced_values(norm)
        norm_primary = (norm_values[0].get("output") if norm_values else "norm")
        external_norm_outputs = set()
        external_records = []
        for user in external:
            entries = [entry for entry in input_ports(by_id[user])
                       if entry.get("op") == norm.op_id]
            if entries:
                names = {value.get("output") for value in norm_values}
                for entry in entries:
                    output = entry.get("output")
                    if output not in names:
                        output = norm_primary
                    external_norm_outputs.add(output)
                    external_records.append({
                        "consumer": user, "source_op": norm.op_id,
                        "output": output})
            else:
                external_norm_outputs.add(norm_primary)
                external_records.append({
                    "consumer": user, "source_op": norm.op_id,
                    "output": norm_primary})
        external_primary = norm_primary in external_norm_outputs
        if external_primary and external_output != "materialize":
            # Keep the old producer/consumer pair when no caller selected the
            # legal "fuse and still materialize the norm output" plan.
            continue
        if [user for user in users[norm.op_id]
                if by_id[user].family == "quantize"] != [op.op_id]:
            # Two quantization consumers mean two overlapping fused plans;
            # leave the shared producer standalone instead of double-charging.
            continue
        # A fused plan may still materialize the norm output for a later
        # consumer, but it cannot move the output before the fused occurrence.
        if any(positions[user] < positions[op.op_id] for user in external):
            continue
        if _capture_state(norm, capture_state) != _capture_state(op, capture_state):
            # Capture boundaries are not fusion boundaries.  Do not fold a
            # graph-captured producer into an eager consumer (or vice versa).
            continue
        if not _consumers_match(users[op.op_id], by_id, consumers):
            continue
        norm_meta = dict(norm.metadata)
        quant_meta = dict(op.metadata)
        norm_inputs = producer_input_values(norm, values_by_id)
        norm_input_value = _first_value(norm_inputs)
        norm_dtype = str(norm_meta.get("output_format") or norm.dtype)
        outputs = list(typed_outputs(op))
        if outputs:
            outputs[0] = {**outputs[0], "output": "payload", "role": "activation",
                          "source": f"{op.op_id}:output_format"}
        for norm_value in norm_values:
            value_output = norm_value.get("output")
            if value_output == norm_primary:
                if not external_primary:
                    continue
                port = "norm"
            elif value_output in external_norm_outputs:
                port = value_output
            else:
                continue
            outputs.append({**norm_value, "producer": op.op_id,
                            "output": port, "role": "external_intermediate",
                            "materialized": True})
        outputs = tuple(outputs)
        if _same_representation(norm, norm_inputs):
            read = norm.mte2_bytes
        else:
            read = (_access_read(norm_inputs, norm.mte2_bytes)
                    + _parameter_read_bytes(norm))
        external_norm_writes = sum(
            float(value.get("write_bytes") or 0.0)
            for value in norm_values
            if ((value.get("output") == norm_primary and external_primary)
                or value.get("output") in external_norm_outputs))
        write = op.mte3_bytes + external_norm_writes
        exposed_norm_names = tuple(
            "norm" if value.get("output") == norm_primary else value.get("output")
            for value in norm_values
            if ((value.get("output") == norm_primary and external_primary)
                or value.get("output") in external_norm_outputs))
        returned = tuple(dict.fromkeys((*op.returns, *exposed_norm_names)))
        capture = capture_fusion_overlap((norm, op), kind="norm_quant")
        dispositions = (
            field_disposition("normalization.input", "consume", norm.op_id,
                              "read once by the fused normalization"),
            field_disposition("normalization.weight", "preserve", norm.op_id,
                              "gamma runtime/storage representation is retained"),
            field_disposition("normalization.epsilon", "preserve", norm.op_id,
                              "normalization attribute is retained"),
            field_disposition("normalization.output", "rewrite", op.op_id,
                              ("materialized as the named norm port" if external else
                               "eliminated as a private intermediate")),
            field_disposition("quantization.payload", "rewrite", op.op_id,
                              "payload is a fused output value"),
            field_disposition("quantization.scales", "rewrite", op.op_id,
                              "scale shape/format are separate from the payload"),
            field_disposition("quantization.group_size", "consume", op.op_id,
                              "grouped amax/scale reduction"),
        )
        boundary = {
            "kind": "norm_quant",
            "replaces_work": True, "replaces_allocation": True,
            "replaced_occurrences": (norm.op_id, op.op_id),
            "replacement_occurrence": op.op_id,
            "work": {"mte2_bytes": read, "mte3_bytes": write,
                     "hbm_bytes": read + write,
                     "vector_ops": norm.vector_ops + op.vector_ops,
                     "flops": op.flops},
            "allocations": {
                "replaced": (norm.op_id, op.op_id),
                "materialized": tuple(value["output"] for value in outputs),
                "eliminated": (() if external else ("normalization.output",)),
            },
            "external_outputs": exposed_norm_names,
            "external_consumers": tuple(external_records),
            "events": {"deps": list(norm.deps), "stream": op.stream or "main",
                       "outputs": list(returned)},
            "capture": capture["capture"],
            "overlap": capture["overlap"],
            "capture_fusion_overlap_separate": True,
            "field_dispositions": dispositions,
            "inputs": tuple(norm_inputs),
        }
        meta = _fusion_metadata(norm, op)
        meta.update({
            "fused_from": ("rmsnorm", "quantize"),
            "primary_output": "payload",
            "norm_output_materialized": bool(external),
        })
        audit = {
            **boundary,
            "transform_members": member_contracts((norm, op)),
            "typed_outputs": outputs,
            "typed_roles": merge_typed_roles(norm, op),
            "semantic_attrs": semantic_attrs(
                (norm, op), ("reduction_width", "norm_eps", "epsilon",
                             "activation_group_size", "activation_scale_shape",
                             "activation_scale_format")),
            "output_ports": tuple(
                {"output": value["output"], "dtype": value["dtype"],
                 "role": value["role"], "materialized": value["materialized"]}
                for value in outputs),
            "selection": {"basis": "selected_software_plan",
                          "kind": "norm_quant"},
        }
        append_audit(meta, kind="transform_boundary", op_id=op.op_id,
                     boundary=audit, layer=op.layer, stage=op.stage,
                     block=meta.get("structure_block"),
                     role=meta.get("structure_role"))
        if consumers is not None:
            meta["fusion_consumer_domain"] = {"pass": "norm_quant", "consumers": consumers}
        fused = replace(
            op, family="rmsnorm_quant", recipe="rmsnorm_quantize", deps=norm.deps,
            vector_ops=norm.vector_ops + op.vector_ops,
            mte2_bytes=read, mte3_bytes=write, hbm_bytes=read + write,
            returns=returned, metadata=tuple(meta.items()),
        )
        replacements[op.op_id] = fused
        removed.add(norm.op_id)
        removed_map[norm.op_id] = op.op_id
        if exposed_norm_names:
            ports[norm.op_id] = {
                ("norm" if value.get("output") == norm_primary else value.get("output")):
                ("norm" if value.get("output") == norm_primary else value.get("output"))
                for value in norm_values
                if ((value.get("output") == norm_primary and external_primary)
                    or value.get("output") in external_norm_outputs)}
    if not replacements:
        return list(ops)
    aliases = {old: old for old in replacements}
    # Each removed norm points at its own quant replacement; multiple fusions
    # in one graph must not alias across layers.
    aliases.update(removed_map)
    result = rewire_replacements(
        ops, replacements=replacements, aliases=aliases,
        removed=removed, ports=ports)
    return attach_quantized_ports(result, replacements)


def residual_norm(ops, _, *, capture_state=None):
    by_id, users = {op.op_id: op for op in ops}, _users(ops)
    positions = {op.op_id: index for index, op in enumerate(ops)}
    values_by_id = _value_map(ops)
    replacements, removed, residual_map, ports = {}, set(), {}, {}
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
                or any(positions[user] < positions[norm.op_id]
                       for user in users[residual.op_id])):
            continue
        if _capture_state(norm, capture_state) != _capture_state(residual, capture_state):
            continue
        # The residual is still consumed by subsequent sublayers. It remains an
        # explicit output of the fused operator, unlike the eliminated norm input.
        residual_inputs = producer_input_values(residual, values_by_id)
        residual_input_value = _first_value(residual_inputs)
        residual_meta = dict(residual.metadata)
        residual_same = _same_representation(residual, residual_inputs)
        norm_inputs_for_read = producer_input_values(norm, values_by_id)
        norm_same = _same_representation(norm, norm_inputs_for_read)
        if residual_same and norm_same:
            read = residual.mte2_bytes + norm.mte2_bytes - residual.mte3_bytes
        else:
            residual_read = (_access_read(residual_inputs, residual.mte2_bytes)
                             + _parameter_read_bytes(residual))
            norm_read = (_access_read(norm_inputs_for_read, norm.mte2_bytes)
                         + _parameter_read_bytes(norm))
            read = residual_read + norm_read - residual.mte3_bytes
        write = norm.mte3_bytes + residual.mte3_bytes
        quantized = norm.family == "rmsnorm_quant"
        norm_meta = dict(norm.metadata)
        residual_meta = dict(residual.metadata)
        nested = nested_boundary(norm)
        outputs = list(norm_meta.get("typed_outputs", ()) or
                       nested.get("typed_outputs", ()))
        if outputs:
            # A previously selected norm/quant plan already names its payload,
            # scales and any externally materialized norm output.
            if not quantized:
                outputs[0] = {**outputs[0], "output": "norm", "role": "activation"}
        else:
            outputs = list(typed_outputs(norm))
            if not quantized and outputs:
                outputs[0] = {**outputs[0], "output": "norm", "role": "activation"}
        residual_value = produced_values(residual)[0] if produced_values(residual) else None
        if residual_value is not None:
            outputs.append({**residual_value, "producer": norm.op_id,
                            "output": "residual", "role": "residual",
                            "materialized": True})
        outputs = tuple(outputs)
        returned = tuple(dict.fromkeys((*norm.returns, "residual")))
        external_residual = [user for user in users[residual.op_id] if user != norm.op_id]
        dispositions = (
            field_disposition("residual.input", "consume", residual.op_id,
                              "residual inputs feed the fused add"),
            field_disposition("residual.output", "preserve", norm.op_id,
                              "residual is still consumed by later sublayers"),
            field_disposition("normalization.output", "preserve", norm.op_id,
                              "norm payload remains the primary fused output"),
            field_disposition("normalization.epsilon", "preserve", norm.op_id,
                              "normalization attribute is retained"),
        )
        if quantized:
            dispositions += (
                field_disposition("quantization.scales", "rewrite", norm.op_id,
                                  "scale values remain a separate output"),
                field_disposition("quantization.group_size", "consume", norm.op_id,
                                  "grouped quantization is unchanged"),
            )
        capture = capture_fusion_overlap((residual, norm), kind="residual_norm")
        external_outputs = ("residual",) + (
            ("norm",) if (not quantized
                          or norm_meta.get("norm_output_materialized")) else ())
        boundary = {
            "kind": "residual_norm",
            "replaces_work": True, "replaces_allocation": True,
            "replaced_occurrences": replaced_occurrences(residual, norm),
            "replacement_occurrence": norm.op_id,
            "work": {"mte2_bytes": read, "mte3_bytes": write,
                     "hbm_bytes": read + write,
                     "vector_ops": residual.vector_ops + norm.vector_ops,
                     "flops": norm.flops},
            "allocations": {
                "replaced": replaced_occurrences(residual, norm),
                "materialized": tuple(value["output"] for value in outputs),
                "eliminated": ("residual.intermediate_write",
                               *nested.get("allocations", {}).get("eliminated", ())),
            },
            "external_outputs": external_outputs,
            "external_consumers": (
                *nested.get("external_consumers", ()),
                *(  {"consumer": user, "source_op": residual.op_id,
                     "output": "residual"} for user in external_residual)),
            "events": {"deps": list(residual.deps), "stream": norm.stream or "main",
                       "outputs": list(returned)},
            "capture": capture["capture"],
            "overlap": capture["overlap"],
            "capture_fusion_overlap_separate": True,
            "field_dispositions": dispositions,
            "inputs": tuple(residual_inputs),
        }
        meta = _fusion_metadata(residual, norm)
        meta.update({
            "fused_from": ("residual_add", "rmsnorm",
                           *(("quantize",) if quantized else ())),
            "primary_output": "payload" if quantized else "norm",
            "residual_output_materialized": True,
        })
        audit = {
            **boundary,
            "transform_members": member_contracts((residual, norm)),
            "typed_outputs": outputs,
            "typed_roles": merge_typed_roles(residual, norm),
            "semantic_attrs": semantic_attrs(
                (norm,), ("reduction_width", "norm_eps", "epsilon",
                          "activation_group_size", "activation_scale_shape",
                          "activation_scale_format")),
            "output_ports": tuple(
                {"output": value["output"], "dtype": value["dtype"],
                 "role": value["role"], "materialized": value["materialized"]}
                for value in outputs),
            "selection": {"basis": "selected_software_plan",
                          "kind": "residual_norm"},
        }
        append_audit(meta, kind="transform_boundary", op_id=norm.op_id,
                     boundary=audit, layer=norm.layer, stage=norm.stage,
                     block=meta.get("structure_block"),
                     role=meta.get("structure_role"))
        replacements[norm.op_id] = replace(
            norm, deps=residual.deps,
            family="add_rmsnorm_quant" if quantized else "add_rmsnorm",
            recipe="add_rmsnorm_quantize" if quantized else "add_rmsnorm",
            vector_ops=residual.vector_ops + norm.vector_ops,
            mte2_bytes=read, mte3_bytes=write, hbm_bytes=read + write,
            metadata=tuple(meta.items()), returns=returned,
        )
        removed.add(residual.op_id)
        residual_map[residual.op_id] = norm.op_id
        ports[residual.op_id] = "residual"
    if not replacements:
        return list(ops)
    aliases = {old: old for old in replacements}
    aliases.update(residual_map)
    result = rewire_replacements(
        ops, replacements=replacements, aliases=aliases,
        removed=removed, ports=ports)
    return attach_quantized_ports(result, replacements)


LOWERING_PASSES = {
    "activation_quantization": activation_quantization,
    "norm_quant": norm_quant,
    "residual_norm": residual_norm,
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
    from .family_shared_quant import lower_shared_experts
    # Bind explicitly selected kernels before generic fusion can erase them.
    # A fused-family selection binds after that family is actually produced.
    ops = lower_operators(ops, bindings)
    from .family_moe_init import apply as apply_moe_init
    ops = apply_moe_init(ops, bindings, runtime)
    consumers, external_output = _norm_quant_domain(runtime.kernels)
    registry = {**LOWERING_PASSES, "norm_quant": partial(
        norm_quant, consumers=consumers, external_output=external_output),
        **(passes or {})}
    activation_quantized = bool(preactivated)
    for name in runtime.lowering_passes:
        if name in skip_passes:
            continue
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
