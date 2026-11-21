"""Typed graph and port rewrites for physical replacements.

[Role] Preserve graph placement, value identities and consumer access plans
  when a lowering or transform replaces operators.
[Invariants] Replacement mappings cover dependencies and named ports together.
  Unrelated consumer edges and source provenance are preserved.
[Timeouts & Locks] No blocking operations or locks.
[Side-Effect Policy] Build replacement operators and edit local metadata copies.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError

from .implementation_audit import _audit_outputs, append_audit, input_ports
from .typed_values import _json_identity, _normalize_output, produced_values
from .value_uses import _PRODUCER_FIELDS, consumer_value_uses

_SINGLE_DEP = 1


def preserve_graph(source, semantic):
    """A physical replacement keeps callers, outputs and graph placement.

    The selected source owns its dispatch/ABI metadata, but a semantic field
    that only the replaced record declared (dtype/format/scale/state/attr) is
    not dropped merely because the source planner does not repeat it.  Source
    keys take precedence; the complete semantic record remains available under
    ``semantic_metadata`` for exact provenance.
    """
    metadata = dict(source.metadata)
    for key, value in semantic.metadata:
        if key in {"graph_captured", "graph_segment"}:
            metadata.setdefault(key, value)
        else:
            metadata.setdefault(key, value)
    metadata["semantic_recipe"] = semantic.recipe
    metadata["semantic_metadata"] = dict(semantic.metadata)
    metadata["source_binding"] = {
        "implementation_recipe": source.recipe,
        "semantic_recipe": semantic.recipe,
        "semantic_input_dtype": semantic.dtype,
        "semantic_input_format": dict(semantic.metadata).get("input_format", semantic.dtype),
        "semantic_output_format": dict(semantic.metadata).get("output_format", semantic.dtype),
        "semantic_returns": list(semantic.returns),
        "source_boundary": source.source_boundary,
    }
    explicit_semantic = _audit_outputs(semantic)
    if explicit_semantic and not _audit_outputs(source):
        append_audit(
            metadata, kind="source_value_contract", op_id=semantic.op_id,
            boundary={"typed_outputs": tuple(
                _normalize_output(value, semantic, None, native=native)
                for value, native in explicit_semantic)},
            layer=semantic.layer, stage=semantic.stage)
    return replace(
        source, op_id=semantic.op_id, family=semantic.family, phase=semantic.phase,
        stage=semantic.stage, layer=semantic.layer, requests=semantic.requests,
        deps=semantic.deps, stream=semantic.stream, semantic_shape=semantic.semantic_shape,
        metadata=tuple(metadata.items()),
        notes=semantic.notes + source.notes,
    )


def set_input_ports(op, entries):
    """Replace named consumer ports, including removal of revoked producers."""
    entries = tuple(entry for entry in entries if entry)
    metadata = dict(op.metadata)
    origins = [record for record in metadata.get("fused_origin_scopes", ())
               if not (record.get("kind") == "input_ports"
                       and record.get("op_id") == op.op_id)]
    metadata["fused_origin_scopes"] = tuple(origins)
    op = replace(op, metadata=tuple(metadata.items()))
    return add_input_ports(op, entries)


def add_input_ports(op, entries):
    """Attach named output ports to an existing consumer edge."""
    entries = tuple(entry for entry in entries if entry)
    if not entries:
        return op
    metadata = dict(op.metadata)
    existing = list(input_ports(op))
    merged = list(existing)
    for entry in entries:
        if entry not in merged:
            merged.append(entry)
    if merged == existing:
        return op
    append_audit(metadata, kind="input_ports", op_id=op.op_id, ports=merged,
                 layer=op.layer, stage=op.stage,
                 block=dict(op.metadata).get("structure_block"),
                 role=dict(op.metadata).get("structure_role"))
    return replace(op, metadata=tuple(metadata.items()))


def attach_quantized_ports(ops, replacements):
    """Name payload and scale ports for matrix consumers of a fused quantizer.

    A matrix consumer reads both the low-precision payload and its scale.  A
    producer of scales (``add_rmsnorm_quant``) is not a consumer of them, and a
    reader that already owns a named ``residual`` port keeps that edge.
    """
    result = list(ops)
    for index, op in enumerate(result):
        metadata = dict(op.metadata)
        if metadata.get("program_kind") != "matmul":
            continue
        if not (metadata.get("activation_scale_bytes")
                or metadata.get("activation_scale_in_mte2")):
            continue
        existing = {entry.get("op"): entry.get("output")
                    for entry in input_ports(op)}
        entries = []
        for dep in op.deps:
            fused = replacements.get(dep)
            if fused is None or existing.get(dep) == "residual":
                continue
            if not any(value["output"] == "payload"
                       for value in produced_values(fused)):
                continue
            entries.extend((
                {"op": dep, "output": "payload", "source_op": dep},
                {"op": dep, "output": "activation_scales", "source_op": dep},
            ))
        if entries:
            result[index] = add_input_ports(op, entries)
    return result


def _move_access_slots(metadata, uses, port_map, aliases):
    """Move consumer-slot repeats/issued bytes when the producer port changes."""
    changed = False
    for key in ("input_repeats", "operand_read_bytes"):
        mapping = metadata.get(key)
        if not isinstance(mapping, dict):
            continue
        moved = dict(mapping)
        for member, output in uses:
            target = port_map.get((member, output))
            if target is None:
                continue
            for old_key in (member, output):
                if old_key in moved:
                    moved.setdefault(target[1], moved.pop(old_key))
                    changed = True
        if moved != mapping:
            metadata[key] = moved
    return changed


def rewire_value_graph(ops, *, replacements, aliases, removed, port_map,
                      values_by_id):
    """One replacement result for ordinary nodes and replacement nodes.

    ``port_map`` maps ``(old producer, output)`` to ``(new producer, port,
    value)``.  The same pass rewrites deps, named ports, producer references
    and unrelated ports; a bare primary edge is implicit only when it is the
    sole old-primary edge folded into one replacement.
    """
    member_ids = set(aliases)
    values_by_id = values_by_id or {}
    replacement_ports = {}
    for (_, _), target in port_map.items():
        replacement_ports.setdefault(target[0], set()).add(target[1])

    def rewire_node(op):
        has_dep_alias = any(dep in aliases for dep in op.deps)
        old_ports = input_ports(op)
        has_port_alias = any(entry.get("op") in aliases for entry in old_ports)
        has_prod_alias = any(
            key in _PRODUCER_FIELDS and isinstance(value, str) and value in aliases
            for key, value in op.metadata
        )
        if not has_dep_alias and not has_port_alias and not has_prod_alias:
            if len(op.deps) > _SINGLE_DEP:
                dedup = tuple(dict.fromkeys(op.deps))
                if dedup != op.deps:
                    return replace(op, deps=dedup)
            return op
        deps = tuple(dict.fromkeys(aliases.get(dep, dep) for dep in op.deps))
        uses = consumer_value_uses(op, member_ids, values_by_id)
        existing = [entry for entry in old_ports
                    if entry.get("op") not in aliases]
        explicit_old = {entry.get("op") for entry in old_ports}
        producer_fields = {value for value in dict(op.metadata).values()
                           if isinstance(value, str)}
        target_counts = {}
        for member, output in uses:
            target = port_map.get((member, output))
            if target is not None:
                target_counts[target[0]] = target_counts.get(target[0], 0) + 1
        mapped = []
        for member, output in uses:
            target = port_map.get((member, output))
            if target is None:
                continue
            primary = (values_by_id.get(member) or [{}])[0].get("output")
            folded = (target_counts.get(target[0], 0) > 1
                      or len(replacement_ports.get(target[0], ())) > 1)
            if (output == primary and member not in explicit_old
                    and member not in producer_fields and not folded):
                continue
            value = target[2] if len(target) > 2 else {}
            entry = {"op": target[0], "output": target[1], "source_op": member}
            for key in ("state_version", "binding", "binding_name", "quant",
                        "role", "access_shape", "repeats", "read_bytes",
                        "base_read_bytes", "storage", "alias_of"):
                if value.get(key) not in (None, ""):
                    entry[key] = (_json_identity(value[key], "")
                                  if key in {"state_version", "binding",
                                             "binding_name"} else value[key])
            if entry not in existing and entry not in mapped:
                mapped.append(entry)
        metadata = dict(op.metadata)
        changed = _move_access_slots(metadata, uses, port_map, aliases)
        for field, requested in _PRODUCER_FIELDS.items():
            producer = metadata.get(field)
            if producer not in aliases:
                continue
            primary = (values_by_id.get(producer) or [{}])[0].get("output")
            target = port_map.get((producer, requested or primary))
            if target is None:
                continue
            metadata[field] = target[0]
            metadata[f"{field}_output"] = target[1]
            changed = True
        rewired = replace(op, deps=deps,
                          metadata=tuple(metadata.items()) if changed else op.metadata)
        ports = tuple(existing + mapped)
        if existing != list(old_ports) or mapped:
            return set_input_ports(rewired, ports)
        return rewired

    result = []
    for op in ops:
        if op.op_id in replacements:
            result.append(rewire_node(replacements[op.op_id]))
            continue
        if op.op_id in removed:
            continue
        result.append(rewire_node(op))
    return result


def rewire_replacements(ops, *, replacements, aliases, removed, ports):
    """Apply one shared value map to any number of generic replacements."""
    values_by_id = {}
    for candidate in ops:
        try:
            values_by_id[candidate.op_id] = produced_values(candidate)
        except ModelingError:
            values_by_id[candidate.op_id] = ()
    port_map = {}
    targets = {}
    for old_id in set(replacements) | set(removed):
        fused = replacements.get(old_id)
        if fused is None:
            fused = replacements.get(aliases.get(old_id))
        if fused is None:
            continue
        targets[old_id] = fused
        fused_values = {value["output"]: value for value in produced_values(fused)}
        requested = ports.get(old_id)
        for value in values_by_id.get(old_id, ()):
            if isinstance(requested, dict):
                target_name = requested.get(value["output"], value["output"])
            else:
                target_name = requested or value["output"]
            target = fused_values.get(target_name)
            if target is not None:
                port_map[(old_id, value["output"])] = (
                    fused.op_id, target["output"], target)
    return rewire_value_graph(
        ops, replacements=replacements, aliases=aliases,
        removed=set(removed), port_map=port_map, values_by_id=values_by_id)


def rewire_replacement(ops, *, replacement_key, fused, removed, ports):
    """Generic-transform adapter for the shared replacement mapping."""
    removed = set(removed)
    aliases = {key: replacement_key for key in removed}
    aliases[replacement_key] = replacement_key
    values_by_id = {}
    for candidate in ops:
        try:
            values_by_id[candidate.op_id] = produced_values(candidate)
        except ModelingError:
            values_by_id[candidate.op_id] = ()
    fused_values = {value["output"]: value for value in produced_values(fused)}
    port_map = {}
    for old in ops:
        if old.op_id != replacement_key and old.op_id not in removed:
            continue
        requested = ports.get(old.op_id)
        for value in values_by_id.get(old.op_id, ()):
            target_name = requested or value["output"]
            target = fused_values.get(target_name)
            if target is None and requested is None:
                target = next(iter(fused_values.values()), None)
            if target is not None:
                port_map[(old.op_id, value["output"])] = (
                    fused.op_id, target["output"], target)
    return rewire_value_graph(
        ops, replacements={replacement_key: fused}, aliases=aliases,
        removed=removed, port_map=port_map, values_by_id=values_by_id)
