"""RMSNorm + dynamic quantization and residual + RMSNorm fusion passes.

[Role]
  Rewrite adjacent ``rmsnorm -> quantize`` and ``residual_add -> rmsnorm``
  occurrences into single fused operators when capture boundaries and consumer
  domains agree, preserving live external norm and residual outputs.

[Invariants]
  * Capture boundaries are respected: producers and consumers in distinct
    graph-capture segments never fuse across segment boundaries.
  * Declared native prolog/compressor groups own their own normalization and
    quantization boundaries and are never rewritten here.
  * Subgraph memberships and declarations are preserved via
    ``fold_region_metadata_dict``.

[Timeouts & Locks]
  Linear O(|V| + |E|) passes over one active layer graph; no locks or I/O.

[Side-Effect Policy]
  Pure functional transforms returning new immutable ``OpSpec`` sequences.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes

from .fusion_regions import fold_region_metadata_dict
from .implementations import (
    append_audit, attach_quantized_ports, capture_fusion_overlap,
    field_disposition, input_ports, member_contracts, merge_typed_roles,
    nested_boundary, produced_values, producer_input_values,
    replaced_occurrences, rewire_replacements, semantic_attrs, typed_outputs,
)

_OWNED_FUSION_ROLES = ("mla_prolog", "compressor")
_NORM_QUANT_SEMANTIC_KEYS = (
    "reduction_width", "norm_eps", "epsilon",
    "activation_group_size", "activation_scale_shape", "activation_scale_format",
)


def _value_map(ops):
    result = {}
    for op in ops:
        try:
            result[op.op_id] = produced_values(op)
        except ModelingError:
            result[op.op_id] = ()
    return result


def _same_representation(op, values):
    if not values:
        return True
    metadata = dict(op.metadata)
    if metadata.get("input_repeats") or metadata.get("operand_read_bytes"):
        return False
    declared = str(metadata.get("input_format") or metadata.get("activation_format") or op.dtype or "")
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
    declared = metadata.get("parameter_read_bytes") or metadata.get("parameter_reads") or {}
    if isinstance(declared, dict):
        total = sum(float(value) for value in declared.values())
    elif isinstance(declared, (int, float)) and not isinstance(declared, bool):
        total = float(declared)
    else:
        total = 0.0
    for key in ("norm_gamma_runtime_bytes", "norm_gamma_bytes", "gamma_read_bytes", "parameter_bytes"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total = max(total, float(value))
    return total


def _access_read(values, fallback):
    reads = [value.get("read_bytes") for value in values if value.get("read_bytes") is not None]
    return sum(reads) if reads else fallback


def _users(ops):
    users = defaultdict(list)
    for op in ops:
        for dep in op.deps:
            users[dep].append(op.op_id)
    return users


def _declared_fusion_owns(op):
    memberships = dict(op.metadata).get("structure_subgraph_memberships", ())
    return any(str(key).rsplit(".", 1)[-1] in _OWNED_FUSION_ROLES for key in memberships or ())


def _norm_quant_domain(kernels):
    domains = kernels.get("fusion_domains", {})
    if not isinstance(domains, Mapping) or set(domains) - {"norm_quant"}:
        raise ModelingError("fusion_domains_require_declared_norm_quant_consumer_domain")
    raw = domains.get("norm_quant")
    if raw is None:
        return None, "eliminate"
    if not isinstance(raw, Mapping) or set(raw) - {"consumers", "external_outputs"} or "consumers" not in raw:
        raise ModelingError("norm_quant_domain_requires_consumers")
    selectors = raw["consumers"]
    fields = {"recipe", "family", "structure_block", "structure_role"}
    if not isinstance(selectors, (list, tuple)) or any(
        not isinstance(item, Mapping) or not item or set(item) - fields
        or any(not isinstance(value, str) or not value for value in item.values())
        for item in selectors
    ):
        raise ModelingError("norm_quant_consumers_require_semantic_recipe_family_or_structure_selectors")
    external = raw.get("external_outputs", "eliminate")
    if external not in {"eliminate", "materialize"}:
        raise ModelingError("norm_quant_external_output_mode_unknown")
    return tuple(dict(item) for item in selectors), external


def _consumers_match(consumer_ids, by_id, selectors):
    if selectors is None:
        return True

    def matches(op, rule):
        metadata = dict(op.metadata)
        return all(
            (getattr(op, k) if k in {"recipe", "family"} else metadata.get(k)) == v
            for k, v in rule.items()
        )

    return bool(consumer_ids) and all(
        any(matches(by_id[node], rule) for rule in selectors) for node in consumer_ids
    )


def _capture_state(op, partition):
    metadata = dict(op.metadata)
    fallback = (metadata.get("graph_captured"), metadata.get("graph_segment"))
    return partition.get(op.op_id, fallback) if partition is not None else fallback


def norm_quant(ops, _, *, consumers=None, external_output="eliminate", capture_state=None):
    if external_output not in {"eliminate", "materialize"}:
        raise ModelingError("norm_quant_external_output_mode_unknown")
    by_id, users = {op.op_id: op for op in ops}, _users(ops)
    positions = {op.op_id: idx for idx, op in enumerate(ops)}
    values_by_id = _value_map(ops)
    replacements, removed, ports, removed_map = {}, set(), {}, {}
    for op in ops:
        if op.plan_required or op.family != "quantize" or len(op.deps) != 1:
            continue
        norm = by_id.get(op.deps[0])
        if (norm is None or norm.plan_required or norm.family != "rmsnorm"
                or norm.local_shape != op.local_shape
                or norm.stage != op.stage or norm.stream != op.stream
                or _declared_fusion_owns(norm) or _declared_fusion_owns(op)
                or op.op_id not in users[norm.op_id]):
            continue
        external = [user for user in users[norm.op_id] if user != op.op_id]
        norm_values = produced_values(norm)
        norm_primary = norm_values[0].get("output") if norm_values else "norm"
        external_norm_outputs, external_records = set(), []
        for user in external:
            entries = [e for e in input_ports(by_id[user]) if e.get("op") == norm.op_id]
            if entries:
                names = {v.get("output") for v in norm_values}
                for entry in entries:
                    out = entry.get("output") if entry.get("output") in names else norm_primary
                    external_norm_outputs.add(out)
                    external_records.append({"consumer": user, "source_op": norm.op_id, "output": out})
            else:
                external_norm_outputs.add(norm_primary)
                external_records.append({"consumer": user, "source_op": norm.op_id, "output": norm_primary})
        external_primary = norm_primary in external_norm_outputs
        if ((external_primary and external_output != "materialize")
                or [u for u in users[norm.op_id] if by_id[u].family == "quantize"] != [op.op_id]
                or any(positions[u] < positions[op.op_id] for u in external)
                or _capture_state(norm, capture_state) != _capture_state(op, capture_state)
                or not _consumers_match(users[op.op_id], by_id, consumers)):
            continue
        norm_inputs = producer_input_values(norm, values_by_id)
        outputs = list(typed_outputs(op))
        if outputs:
            outputs[0] = {**outputs[0], "output": "payload", "role": "activation",
                          "source": f"{op.op_id}:output_format"}
        for nv in norm_values:
            vo = nv.get("output")
            if vo == norm_primary and external_primary:
                port = "norm"
            elif vo != norm_primary and vo in external_norm_outputs:
                port = vo
            else:
                continue
            outputs.append({**nv, "producer": op.op_id, "output": port,
                            "role": "external_intermediate", "materialized": True})
        outputs = tuple(outputs)
        read = (norm.mte2_bytes if _same_representation(norm, norm_inputs)
                else (_access_read(norm_inputs, norm.mte2_bytes) + _parameter_read_bytes(norm)))
        exposed = tuple(
            "norm" if v.get("output") == norm_primary else v.get("output")
            for v in norm_values
            if (v.get("output") == norm_primary and external_primary) or v.get("output") in external_norm_outputs
        )
        ext_writes = sum(
            float(v.get("write_bytes") or 0.0) for v in norm_values
            if (v.get("output") == norm_primary and external_primary) or v.get("output") in external_norm_outputs
        )
        write = op.mte3_bytes + ext_writes
        returned = tuple(dict.fromkeys((*op.returns, *exposed)))
        capture = capture_fusion_overlap((norm, op), kind="norm_quant")
        dispositions = (
            field_disposition("normalization.input", "consume", norm.op_id, "read once by the fused normalization"),
            field_disposition("normalization.weight", "preserve", norm.op_id, "gamma runtime/storage representation is retained"),
            field_disposition("normalization.epsilon", "preserve", norm.op_id, "normalization attribute is retained"),
            field_disposition("normalization.output", "rewrite", op.op_id, "materialized as the named norm port" if external else "eliminated as a private intermediate"),
            field_disposition("quantization.payload", "rewrite", op.op_id, "payload is a fused output value"),
            field_disposition("quantization.scales", "rewrite", op.op_id, "scale shape/format are separate from the payload"),
            field_disposition("quantization.group_size", "consume", op.op_id, "grouped amax/scale reduction"),
        )
        boundary = {
            "kind": "norm_quant", "replaces_work": True, "replaces_allocation": True,
            "replaced_occurrences": (norm.op_id, op.op_id), "replacement_occurrence": op.op_id,
            "work": {"mte2_bytes": read, "mte3_bytes": write, "hbm_bytes": read + write,
                     "vector_ops": norm.vector_ops + op.vector_ops, "flops": op.flops},
            "allocations": {"replaced": (norm.op_id, op.op_id),
                            "materialized": tuple(v["output"] for v in outputs),
                            "eliminated": () if external else ("normalization.output",)},
            "external_outputs": exposed, "external_consumers": tuple(external_records),
            "events": {"deps": list(norm.deps), "stream": op.stream or "main", "outputs": list(returned)},
            "capture": capture["capture"], "overlap": capture["overlap"],
            "capture_fusion_overlap_separate": True, "field_dispositions": dispositions,
            "inputs": tuple(norm_inputs),
        }
        meta = fold_region_metadata_dict((norm, op), primary_anchor_id=op.op_id, extra=(
            ("fused_from", ("rmsnorm", "quantize")),
            ("primary_output", "payload"), ("norm_output_materialized", bool(external)),
        ))
        audit = {
            **boundary, "transform_members": member_contracts((norm, op)),
            "typed_outputs": outputs, "typed_roles": merge_typed_roles(norm, op),
            "semantic_attrs": semantic_attrs((norm, op), _NORM_QUANT_SEMANTIC_KEYS),
            "output_ports": tuple({"output": v["output"], "dtype": v["dtype"], "role": v["role"],
                                   "materialized": v["materialized"]} for v in outputs),
            "selection": {"basis": "selected_software_plan", "kind": "norm_quant"},
        }
        append_audit(meta, kind="transform_boundary", op_id=op.op_id, boundary=audit,
                     layer=op.layer, stage=op.stage, block=meta.get("structure_block"), role=meta.get("structure_role"))
        if consumers is not None:
            meta["fusion_consumer_domain"] = {"pass": "norm_quant", "consumers": consumers}
        replacements[op.op_id] = replace(
            op, family="rmsnorm_quant", recipe="rmsnorm_quantize", deps=norm.deps,
            vector_ops=norm.vector_ops + op.vector_ops, mte2_bytes=read, mte3_bytes=write,
            hbm_bytes=read + write, returns=returned, metadata=tuple(meta.items()),
        )
        removed.add(norm.op_id)
        removed_map[norm.op_id] = op.op_id
        if exposed:
            ports[norm.op_id] = {name: name for name in exposed}
    if not replacements:
        return list(ops)
    aliases = {**{old: old for old in replacements}, **removed_map}
    result = rewire_replacements(ops, replacements=replacements, aliases=aliases, removed=removed, ports=ports)
    return attach_quantized_ports(result, replacements)


def residual_norm(ops, _, *, capture_state=None):
    by_id, users = {op.op_id: op for op in ops}, _users(ops)
    positions = {op.op_id: idx for idx, op in enumerate(ops)}
    values_by_id = _value_map(ops)
    replacements, removed, residual_map, ports = {}, set(), {}, {}
    for norm in ops:
        if (norm.plan_required or "implementation_binding" in dict(norm.metadata)
                or norm.family not in {"rmsnorm", "rmsnorm_quant"} or len(norm.deps) != 1):
            continue
        residual = by_id.get(norm.deps[0])
        if (residual is None or residual.plan_required or residual.recipe != "residual_add"
                or residual.op_id in removed or norm.stage != residual.stage
                or norm.stream != residual.stream
                or math.prod(norm.local_shape) != math.prod(residual.local_shape)
                or any(positions[u] < positions[norm.op_id] for u in users[residual.op_id])
                or _capture_state(norm, capture_state) != _capture_state(residual, capture_state)):
            continue
        res_in = producer_input_values(residual, values_by_id)
        norm_in = producer_input_values(norm, values_by_id)
        if _same_representation(residual, res_in) and _same_representation(norm, norm_in):
            read = residual.mte2_bytes + norm.mte2_bytes - residual.mte3_bytes
        else:
            read = (_access_read(res_in, residual.mte2_bytes) + _parameter_read_bytes(residual)
                    + _access_read(norm_in, norm.mte2_bytes) + _parameter_read_bytes(norm)
                    - residual.mte3_bytes)
        write = norm.mte3_bytes + residual.mte3_bytes
        quantized = norm.family == "rmsnorm_quant"
        norm_meta, nested = dict(norm.metadata), nested_boundary(norm)
        outputs = list(norm_meta.get("typed_outputs", ()) or nested.get("typed_outputs", ()) or typed_outputs(norm))
        if not quantized and outputs:
            outputs[0] = {**outputs[0], "output": "norm", "role": "activation"}
        res_val = produced_values(residual)[0] if produced_values(residual) else None
        if res_val is not None:
            outputs.append({**res_val, "producer": norm.op_id, "output": "residual",
                            "role": "residual", "materialized": True})
        outputs = tuple(outputs)
        returned = tuple(dict.fromkeys((*norm.returns, "residual")))
        ext_res = [u for u in users[residual.op_id] if u != norm.op_id]
        dispositions = (
            field_disposition("residual.input", "consume", residual.op_id, "residual inputs feed the fused add"),
            field_disposition("residual.output", "preserve", norm.op_id, "residual is still consumed by later sublayers"),
            field_disposition("normalization.output", "preserve", norm.op_id, "norm payload remains the primary fused output"),
            field_disposition("normalization.epsilon", "preserve", norm.op_id, "normalization attribute is retained"),
            *((field_disposition("quantization.scales", "rewrite", norm.op_id, "scale values remain a separate output"),
               field_disposition("quantization.group_size", "consume", norm.op_id, "grouped quantization is unchanged"))
              if quantized else ()),
        )
        capture = capture_fusion_overlap((residual, norm), kind="residual_norm")
        ext_outs = ("residual",) + (("norm",) if (not quantized or norm_meta.get("norm_output_materialized")) else ())
        boundary = {
            "kind": "residual_norm", "replaces_work": True, "replaces_allocation": True,
            "replaced_occurrences": replaced_occurrences(residual, norm), "replacement_occurrence": norm.op_id,
            "work": {"mte2_bytes": read, "mte3_bytes": write, "hbm_bytes": read + write,
                     "vector_ops": residual.vector_ops + norm.vector_ops, "flops": norm.flops},
            "allocations": {"replaced": replaced_occurrences(residual, norm),
                            "materialized": tuple(v["output"] for v in outputs),
                            "eliminated": ("residual.intermediate_write", *nested.get("allocations", {}).get("eliminated", ()))},
            "external_outputs": ext_outs,
            "external_consumers": (*nested.get("external_consumers", ()),
                                   *({"consumer": u, "source_op": residual.op_id, "output": "residual"} for u in ext_res)),
            "events": {"deps": list(residual.deps), "stream": norm.stream or "main", "outputs": list(returned)},
            "capture": capture["capture"], "overlap": capture["overlap"],
            "capture_fusion_overlap_separate": True, "field_dispositions": dispositions,
            "inputs": tuple(res_in),
        }
        meta = fold_region_metadata_dict((residual, norm), primary_anchor_id=norm.op_id, extra=(
            ("fused_from", ("residual_add", "rmsnorm", *(("quantize",) if quantized else ()))),
            ("primary_output", "payload" if quantized else "norm"),
            ("residual_output_materialized", True),
        ))
        audit = {
            **boundary, "transform_members": member_contracts((residual, norm)),
            "typed_outputs": outputs, "typed_roles": merge_typed_roles(residual, norm),
            "semantic_attrs": semantic_attrs((norm,), _NORM_QUANT_SEMANTIC_KEYS),
            "output_ports": tuple({"output": v["output"], "dtype": v["dtype"], "role": v["role"],
                                   "materialized": v["materialized"]} for v in outputs),
            "selection": {"basis": "selected_software_plan", "kind": "residual_norm"},
        }
        append_audit(meta, kind="transform_boundary", op_id=norm.op_id, boundary=audit,
                     layer=norm.layer, stage=norm.stage, block=meta.get("structure_block"), role=meta.get("structure_role"))
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
    aliases = {**{old: old for old in replacements}, **residual_map}
    result = rewire_replacements(ops, replacements=replacements, aliases=aliases, removed=removed, ports=ports)
    return attach_quantized_ports(result, replacements)


__all__ = ["_norm_quant_domain", "norm_quant", "residual_norm"]
