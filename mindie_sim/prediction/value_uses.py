"""Producer and consumer value references and operand accounting.

[Role] Resolve graph edges to named values and derive each consumer's read work.
[Invariants] Explicit ports take precedence over bare producer dependencies.
  Access geometry, repeats and issued bytes belong to the consumer.
[Timeouts & Locks] No blocking operations or locks.
[Side-Effect Policy] Read operators and descriptors and return new records.
"""
import math

from mindie_sim.schema.common import dtype_bytes

from .implementation_audit import audit_records, input_ports
from .typed_values import _row_shape, typed_value


def _access_bytes(value, *, shape=None, repeats=1, explicit=None,
                  expected_elements=None,
                  contract="declared_operand_geometry"):
    shape = tuple(shape or value.get("logical_shape") or value.get("shape") or ())
    dtype = value.get("dtype")
    elements = int(math.prod(shape)) if shape else 0
    effective_elements = elements if expected_elements is None else float(expected_elements)
    try:
        base_read = float(effective_elements * dtype_bytes(dtype))
    except (KeyError, ValueError):
        base_read = None
    if explicit is not None:
        # Convention: operand_read_bytes is total issued bytes for this
        # operand and already includes repeats.
        read_bytes = float(explicit)
        contract = "explicit_operand_access"
    else:
        read_bytes = base_read
    result = dict(value)
    result.update({
        "access_shape": list(shape),
        "read_elements": elements,
        "repeats": int(repeats),
        "base_read_bytes": base_read,
        "contract": contract,
    })
    if read_bytes is not None:
        result["read_bytes"] = (read_bytes if explicit is not None
                                else read_bytes * int(repeats))
    return result


def _select_named_value(op, dependency, requested, outputs_by_id):
    available = list((outputs_by_id or {}).get(dependency, ()))
    if not available:
        return None
    if requested:
        exact = next((value for value in available
                      if value.get("output") == requested), None)
        if exact is not None:
            return exact
        if requested in {"payload", "output", "activation"}:
            return available[0]
        return None
    return available[0]


def producer_input_values(op, outputs_by_id=None, source_dtype=None):
    """Resolve every consumer operand by value identity, not by first output.

    Named ``input_ports`` select the exact producer output (for example a BF16
    residual port of an INT8-payload fusion).  A bare dependency is only the
    explicit primary-output compatibility rule.  Each returned value carries a
    per-operand access shape, repeat count and issued read bytes; a producer's
    write bytes are never substituted for the consumer's access plan.
    """
    metadata = dict(op.metadata)
    is_matmul = metadata.get("program_kind") == "matmul" and len(op.local_shape) >= 3
    if not op.deps and not (is_matmul and (source_dtype or metadata.get("input_format"))):
        return ()
    produced_by_dep = outputs_by_id or {}
    ports_by_dep = {}
    for entry in input_ports(op):
        producer = entry.get("op")
        if isinstance(producer, str):
            ports_by_dep.setdefault(producer, []).append(entry)

    repeats = metadata.get("input_repeats") or {}
    explicit = metadata.get("operand_read_bytes") or {}
    if not isinstance(repeats, dict):
        repeats = {}
    if not isinstance(explicit, dict):
        explicit = {}

    def requested(dependency):
        entries = ports_by_dep.get(dependency, ())
        for entry in entries:
            output = entry.get("output")
            if output and output not in {"activation_scales", "scale"}:
                return output
        return None

    def selected(dependency):
        available = list(produced_by_dep.get(dependency, ()))
        requested_value = requested(dependency)
        if requested_value:
            value = _select_named_value(op, dependency, requested_value, produced_by_dep)
            if value is not None:
                return value, requested_value
        for entry in ports_by_dep.get(dependency, ()):
            value = _select_named_value(op, dependency, entry.get("output"),
                                        produced_by_dep)
            if value is not None:
                return value, entry.get("output")
        return (available[0], available[0].get("output")) if available else (None, None)

    def add(value, *, role, field, shape=None, dependency="", repeats_value=1,
            explicit_bytes=None, expected_elements=None):
        value = _access_bytes(value, shape=shape, repeats=repeats_value,
                              explicit=explicit_bytes,
                              expected_elements=expected_elements)
        value.update({"role": role, "consumer": op.op_id,
                      "consumer_field": field, "producer": dependency or
                      value.get("producer", "")})
        return value

    if is_matmul:
        m, _, k = op.local_shape
        expected_rows = float(metadata.get("expected_rows", m))
        shape_rows = int(math.ceil(expected_rows)) if expected_rows > 0 else 0
        dependency = op.deps[0] if op.deps else ""
        primary, primary_output = selected(dependency) if dependency else (None, None)
        logical, storage = _row_shape(metadata, (shape_rows, k))
        expected_elements = (
            expected_rows * k
            if "expected_rows" in metadata and expected_rows != shape_rows
            else None
        )
        if primary is None:
            activation_dtype = str(
                (source_dtype if not produced_by_dep.get(dependency) and source_dtype else "")
                or metadata.get("input_format")
                or metadata.get("activation_format") or "")
            if not activation_dtype:
                return ()
            primary = typed_value(
                producer=dependency, dtype=activation_dtype, shape=logical,
                storage_shape=storage, output="activation", role="activation",
                source=f"{op.op_id}:activation_format",
                materialized=True, contract="declared_operand_value")
            primary_output = "activation"
        primary_repeats = repeats.get(primary_output, repeats.get(dependency, 1))
        explicit_primary = explicit.get(primary_output, explicit.get(dependency))
        values = [add(primary, role="activation", field=primary_output or "activation",
                      shape=logical, dependency=dependency,
                      repeats_value=primary_repeats, explicit_bytes=explicit_primary,
                      expected_elements=expected_elements)]
        seen_outputs = {primary_output}
        for entry in ports_by_dep.get(dependency, ()):
            if entry.get("output") in {"payload", "output", "activation", None}:
                continue
            if entry.get("output") in seen_outputs:
                continue
            seen_outputs.add(entry.get("output"))
            value = _select_named_value(op, dependency, entry.get("output"),
                                        produced_by_dep)
            if value is not None:
                values.append(add(value, role="input",
                                  field=entry.get("output") or "input",
                                  shape=tuple(value.get("logical_shape") or ()),
                                  dependency=dependency,
                                  repeats_value=repeats.get(entry.get("output"), 1),
                                  explicit_bytes=explicit.get(entry.get("output"))))
        scale = next((value for value in produced_by_dep.get(dependency, ())
                      if value.get("output") == "activation_scales"), None)
        if scale is None and metadata.get("activation_scale_format"):
            shape = tuple(metadata.get("activation_scale_shape") or ())
            scale = typed_value(
                producer=dependency,
                dtype=str(metadata["activation_scale_format"]), shape=shape,
                storage_shape=_row_shape(metadata, shape)[1],
                output="activation_scales", role="scale",
                source=f"{op.op_id}:activation_scale_format",
                materialized=True,
                write_bytes=metadata.get("activation_scale_bytes"),
                contract="declared_scale_operand")
        if scale is not None and "activation_scales" not in seen_outputs:
            values.append(add(scale, role="scale", field="activation_scales",
                              shape=tuple(scale.get("logical_shape") or ()),
                              dependency=dependency,
                              repeats_value=repeats.get("activation_scales", 1),
                              explicit_bytes=explicit.get("activation_scales")))
        weight_dtype = metadata.get("weight_format") or metadata.get("weight_dtype")
        if weight_dtype:
            shape = (op.local_shape[1], op.local_shape[2])
            weight = typed_value(
                producer="", dtype=str(weight_dtype), shape=shape,
                storage_shape=shape, output="weight", role="weight",
                source=f"{op.op_id}:weight_format", materialized=True,
                write_bytes=metadata.get("weight_bytes")
                or metadata.get("packed_weight_bytes"),
                contract="parameter_value")
            values.append(add(weight, role="weight", field="weight", shape=shape,
                              dependency="", repeats_value=repeats.get("weight", 1),
                              explicit_bytes=explicit.get("weight")))
        return tuple(values)

    values = []
    seen = set()
    for dependency in op.deps:
        entries = ports_by_dep.get(dependency, ())
        selected_entries = []
        for entry in entries:
            output = entry.get("output")
            value = _select_named_value(op, dependency, output, produced_by_dep)
            if value is not None:
                selected_entries.append((value, output))
        if not selected_entries:
            value, output = selected(dependency)
            if value is not None:
                selected_entries.append((value, output))
        for value, output in selected_entries:
            key = (dependency, output, value.get("value_id"))
            if key in seen:
                continue
            seen.add(key)
            values.append(add(
                value, role="input", field=output or "input",
                shape=tuple(value.get("logical_shape") or ()),
                dependency=dependency,
                repeats_value=repeats.get(output, repeats.get(dependency, 1)),
                explicit_bytes=explicit.get(output, explicit.get(dependency))))
        if selected_entries:
            continue
        activation_dtype = str(
            (source_dtype if dependency not in produced_by_dep and source_dtype else "")
            or metadata.get("input_format") or op.dtype or "")
        if not activation_dtype:
            continue
        logical, storage = _row_shape(metadata, tuple(op.local_shape))
        fallback = typed_value(
            producer=dependency, dtype=activation_dtype, shape=logical,
            storage_shape=storage, output="activation", role="activation",
            source=f"{op.op_id}:input_format", materialized=True,
            contract="declared_operand_value")
        values.append(add(fallback, role="input", field="activation",
                          shape=logical, dependency=dependency,
                          repeats_value=repeats.get(dependency, 1),
                          explicit_bytes=explicit.get(dependency)))
    return tuple(values)


_PRODUCER_FIELDS = {
    "activation_scale_producer": "activation_scales",
    "input_scale_producer": "activation_scales",
    "activation_payload_producer": None,
    "payload_producer": None,
    "topk_ids_producer": "topk_ids",
    "topk_weights_producer": "topk_weights",
    "inverse_index_producer": "inverse_index",
}


def _op_producer_refs(op):
    refs = set(op.deps)
    for record in audit_records(op):
        if record.get("kind") == "input_ports":
            for entry in record.get("ports", ()):
                target = entry.get("op")
                if target:
                    refs.add(target)
    for key, value in op.metadata:
        if key in _PRODUCER_FIELDS and isinstance(value, str):
            refs.add(value)
    return refs


_LAST_ENTRY = -1
_FIRST_SLOT = 0


def consumer_index(ops):
    """Index candidate consumers by producer op_id while preserving ``ops`` order."""
    by_producer = {}
    by_id = {}
    for index, op in enumerate(ops):
        by_id[op.op_id] = op
        item = (index, op)
        for producer in _op_producer_refs(op):
            bucket = by_producer.get(producer)
            if bucket is None:
                by_producer[producer] = [item]
            elif bucket[_LAST_ENTRY][_FIRST_SLOT] != index:
                bucket.append(item)
    return by_id, by_producer


def consumer_value_uses(op, member_ids, values_by_id):
    """Resolve the actual ``(member, output)`` values an external reader uses.

    Explicit ports are authoritative.  Producer-reference metadata is used
    next, and a bare dependency falls back to that producer's primary output.
    A reader of one member output never implicitly marks the member's other
    returns external.
    """
    if not isinstance(member_ids, (set, frozenset)):
        member_ids = set(member_ids)
    if not member_ids:
        return ()
    ports = input_ports(op)
    has_port_hit = any(entry.get("op") in member_ids for entry in ports)
    has_prod_hit = any(
        key in _PRODUCER_FIELDS and isinstance(value, str) and value in member_ids
        for key, value in op.metadata
    )
    has_dep_hit = any(dependency in member_ids for dependency in op.deps)
    if not has_port_hit and not has_prod_hit and not has_dep_hit:
        return ()
    metadata = dict(op.metadata) if has_prod_hit else None
    uses = []

    def names(member):
        return {value.get("output") for value in values_by_id.get(member, ())}

    def add(member, requested, *, fallback=True):
        if member not in member_ids:
            return
        available = names(member)
        if requested and requested in available:
            output = requested
        elif (requested in {None, "payload", "output", "activation"}
              and values_by_id.get(member)):
            output = values_by_id[member][0].get("output")
        elif fallback and values_by_id.get(member):
            output = values_by_id[member][0].get("output")
        else:
            return
        if not output:
            return
        pair = (member, output)
        if pair not in uses:
            uses.append(pair)

    if has_port_hit:
        for entry in ports:
            add(entry.get("op"), entry.get("output"), fallback=False)
    if metadata is not None:
        for field, requested in _PRODUCER_FIELDS.items():
            producer = metadata.get(field)
            if isinstance(producer, str):
                add(producer, requested, fallback=requested is None)
    if has_dep_hit:
        for dependency in op.deps:
            if dependency in member_ids and not any(
                    member == dependency for member, _ in uses):
                add(dependency, None)
    return tuple(uses)


def external_value_plan(ops, members, values_by_id, public_outputs=(), *,
                        consumer_map=None):
    """Exact externally-live values for one selected fusion group.

    ``ops`` includes the region around the group.  A member with no consumer
    at all is a call/region public root: all of its produced values remain
    public even when no node follows it (independent native/subgraph calls).
    Consumers inside the group are internal edges, not public readers.
    """
    member_ids = {member.op_id for member in members}
    specs, consumers = [], []
    consumed_members = set()
    if consumer_map is not None:
        by_id, by_producer = consumer_map
        cand_by_pos = {}
        for m_id in member_ids:
            for pos, op in by_producer.get(m_id, ()):
                if op.op_id not in member_ids:
                    cand_by_pos[pos] = op
        outside = [cand_by_pos[pos] for pos in sorted(cand_by_pos)]
        group_ops = [by_id[m_id] for m_id in member_ids if m_id in by_id]
    else:
        outside = [op for op in ops if op.op_id not in member_ids]
        group_ops = [op for op in ops if op.op_id in member_ids]
    for op in outside:
        for member_id, output in consumer_value_uses(op, member_ids, values_by_id):
            consumed_members.add(member_id)
            pair = (member_id, output)
            if pair not in specs:
                specs.append(pair)
            entry = {"consumer": op.op_id, "member": member_id,
                     "output": output}
            if entry not in consumers:
                consumers.append(entry)
    public_refs = set()
    for entry in public_outputs or ():
        if isinstance(entry, str):
            member_id, output = entry.rsplit(".", 1) if "." in entry else (entry, "output")
        else:
            member_id, output = entry[:2]
        if member_id in member_ids:
            public_refs.add((member_id, output))
            if (member_id, output) not in specs:
                specs.append((member_id, output))
    internal_uses = [
        (op.op_id, {m_id for m_id, _ in consumer_value_uses(op, member_ids, values_by_id)})
        for op in group_ops
    ]
    for member in members:
        member_public = {pair for pair in public_refs if pair[0] == member.op_id}
        if member_public:
            for pair in member_public:
                if pair not in specs:
                    specs.append(pair)
            continue
        if (member.op_id in consumed_members
                or any(member.op_id in used for op_id, used in internal_uses
                       if op_id != member.op_id)):
            continue
        for value in values_by_id.get(member.op_id, ()):
            pair = (member.op_id, value["output"])
            if pair not in specs:
                specs.append(pair)
    counts = {}
    for _, output in specs:
        counts[output] = counts.get(output, 0) + 1
    seen = {}
    port_for = {}
    values = []
    for member_id, output in specs:
        value = next((item for item in values_by_id.get(member_id, ())
                      if item.get("output") == output), None)
        if value is None:
            continue
        duplicate = counts[output]
        if duplicate == 1:
            port = output
        else:
            ordinal = seen.get(output, 0)
            seen[output] = ordinal + 1
            port = f"{output}__{ordinal}"
        port_for[(member_id, output)] = port
        descriptor = {**value, "member": member_id, "output": output,
                      "port": port,
                      "materialized": bool(value.get("materialized", True))}
        values.append(descriptor)
    member_ids_out = tuple(dict.fromkeys(member_id for member_id, _ in specs))
    anchor = members[-1].op_id if members else None
    anchor_primary = (values_by_id.get(anchor) or [{}])[0].get("output")
    explicit_public = {(value["member"], value["output"])
                       for value in values
                       if (value["member"], value["output"]) in public_refs}
    returns = tuple(
        value["port"] for value in values
        if (value["member"], value["output"]) in explicit_public
        or not (value["member"] == anchor and value["output"] == anchor_primary)
    )
    return {
        "member_ids": member_ids_out,
        "values": tuple(values),
        "references": tuple((value["member"], value["output"]) for value in values),
        "ports": port_for,
        "returns": tuple(dict.fromkeys(returns)),
        "consumers": tuple(consumers),
    }
