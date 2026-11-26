"""Producer and consumer value references and operand accounting.

[Role] Resolve graph edges to named values and derive each consumer's read work.
[Invariants] Explicit ports take precedence over bare producer dependencies.
  Access geometry, repeats and issued bytes belong to the consumer.
[Timeouts & Locks] No blocking operations or locks.
[Side-Effect Policy] Read operators and descriptors and return new records.
"""
import math

from mindie_sim.schema.common import dtype_bytes

from mindie_sim.prediction.program.implementation_audit import audit_records, input_ports
from mindie_sim.prediction.values.typed import _row_shape, typed_value


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
        return _matmul_input_values(
            op, metadata, source_dtype, produced_by_dep, ports_by_dep,
            repeats, explicit, selected, add)

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


_GROUP_TYPE_K = 2


def _group_list_input_value(
    op, metadata, group_list_producer, produced_by_dep, repeats, explicit, selected, add,
):
    group_list_val = _select_named_value(
        op, group_list_producer, "group_list", produced_by_dep,
    )
    if group_list_val is None:
        group_list_val, _ = selected(group_list_producer)
    if group_list_val is None:
        wrapper_meta = (
            metadata.get("matrix_wrapper")
            if isinstance(metadata.get("matrix_wrapper"), dict)
            else {}
        )
        gl_dtype = str(
            metadata.get("group_list_dtype")
            or wrapper_meta.get("group_list_dtype")
            or "int64"
        )
        raw_gl_shape = (
            metadata.get("group_list_shape")
            or wrapper_meta.get("group_list_shape")
            or (
                (int(metadata["matrix_weight_bank_groups"]),)
                if metadata.get("matrix_weight_bank_groups")
                else (1,)
            )
        )
        gl_shape = tuple(int(dim) for dim in raw_gl_shape)
        group_list_val = typed_value(
            producer=group_list_producer, dtype=gl_dtype, shape=gl_shape,
            storage_shape=gl_shape, output="group_list", role="index",
            source=f"{op.op_id}:group_list_producer", materialized=True,
            contract="declared_group_list_operand",
        )
    return add(
        group_list_val, role="index", field="group_list",
        shape=tuple(group_list_val.get("logical_shape") or ()),
        dependency=group_list_producer,
        repeats_value=repeats.get("group_list", repeats.get(group_list_producer, 1)),
        explicit_bytes=explicit.get("group_list", explicit.get(group_list_producer)),
    )


def _active_bank_runs(active_indices):
    """Partition sorted active bank indices into exact affine ``(start, count, step)`` runs."""
    if not active_indices:
        return ()
    if len(active_indices) == 1:
        return ((active_indices[0], 1, 1),)
    steps = {b - a for a, b in zip(active_indices, active_indices[1:])}
    if len(steps) == 1:
        return ((active_indices[0], len(active_indices), next(iter(steps))),)
    runs = []
    start = prev = active_indices[0]
    count = 1
    for idx in active_indices[1:]:
        if idx == prev + 1:
            count += 1
        else:
            runs.append((start, count, 1))
            start = idx
            count = 1
        prev = idx
    runs.append((start, count, 1))
    return tuple(runs)


def _weight_input_values(op, metadata, weight_dtype, repeats, explicit, add):
    weight_n, weight_k = op.local_shape[1], op.local_shape[2]
    col_span = metadata.get("deepep_column_span")
    col_cut = metadata.get("deepep_column_cut")
    has_col_split = (
        isinstance(col_span, (tuple, list)) and len(col_span) == 2
        and isinstance(col_span[0], int) and isinstance(col_span[1], int)
        and col_span[0] >= 0 and col_span[1] >= col_span[0]
    )
    col_start = int(col_span[0]) if has_col_split else 0
    if has_col_split and isinstance(col_cut, (tuple, list)) and col_cut:
        full_n = max(int(sum(int(c) for c in col_cut)), int(col_span[1]), weight_n)
    elif has_col_split:
        full_n = max(int(col_span[1]), weight_n)
    else:
        full_n = weight_n
    bank_elems = weight_n * weight_k
    full_bank_elems = full_n * weight_k
    total_write_bytes = metadata.get("weight_bytes") or metadata.get("packed_weight_bytes")
    total_explicit = explicit.get("weight")
    weight_repeats = repeats.get("weight", 1)
    expected_weight_elements = None
    storage_shape = (full_n, weight_k)
    if op.family == "grouped_gemm" and metadata.get("group_type") != _GROUP_TYPE_K:
        seg_rows = metadata.get("deepep_segment_expert_bank_rows")
        if isinstance(seg_rows, (tuple, list)):
            active_groups: float | None = float(sum(1 for rows in seg_rows if int(rows) > 0))
        else:
            raw_groups = (
                metadata["expected_active_experts"]
                if "expected_active_experts" in metadata
                else metadata.get("matrix_groups")
            )
            if raw_groups is not None:
                active_groups = float(raw_groups)
                if "deepep_weight_scale" in metadata and not has_col_split:
                    active_groups *= float(metadata["deepep_weight_scale"])
            else:
                active_groups = None
        if active_groups is not None:
            group_dim = int(math.ceil(active_groups)) if active_groups > 0 else 0
            shape = (group_dim, weight_n, weight_k)
            if active_groups != group_dim:
                expected_weight_elements = active_groups * bank_elems
            bank_groups = metadata.get("matrix_weight_bank_groups")
            group_rows = seg_rows if isinstance(seg_rows, (tuple, list)) else metadata.get("matrix_group_rows")
            active_indices = (
                [idx for idx, rows in enumerate(group_rows) if int(rows) > 0]
                if isinstance(group_rows, (tuple, list)) else []
            )
            min_banks = max(
                group_dim,
                (active_indices[-1] + 1) if active_indices else 0,
                len(group_rows) if isinstance(group_rows, (tuple, list)) else 0,
            )
            resident_banks = (
                max(int(bank_groups), min_banks)
                if bank_groups is not None and int(bank_groups) > 0
                else min_banks
            )
            storage_shape = (resident_banks, full_n, weight_k)
            if active_indices and len(active_indices) == group_dim:
                runs = _active_bank_runs(active_indices)
                results = []
                for run_idx, (start, count, step) in enumerate(runs):
                    run_shape = (count, weight_n, weight_k)
                    run_strides = (step * full_bank_elems, weight_k, 1)
                    run_offset = start * full_bank_elems + col_start * weight_k
                    share = float(count) / float(group_dim)
                    out_name = "weight" if len(runs) == 1 else f"weight_slice{run_idx}"
                    run_write = (
                        float(total_write_bytes) * share
                        if total_write_bytes is not None else None
                    )
                    run_explicit = (
                        float(total_explicit) * share
                        if total_explicit is not None else None
                    )
                    weight = typed_value(
                        producer="", dtype=str(weight_dtype), shape=run_shape,
                        storage_shape=storage_shape, strides=run_strides,
                        offset=run_offset, output=out_name, role="weight",
                        source=f"{op.op_id}:weight_format", materialized=True,
                        write_bytes=run_write, contract="parameter_value",
                    )
                    run_expected = (
                        expected_weight_elements * share
                        if expected_weight_elements is not None else None
                    )
                    results.append(add(
                        weight, role="weight", field=out_name, shape=run_shape,
                        dependency="", repeats_value=weight_repeats,
                        explicit_bytes=run_explicit,
                        expected_elements=run_expected,
                    ))
                return tuple(results)
        else:
            shape = (weight_n, weight_k)
    else:
        shape = (weight_n, weight_k)
    default_offset = col_start * weight_k
    default_strides = (
        (full_bank_elems, weight_k, 1) if len(shape) == 3 and has_col_split else None
    )
    weight = typed_value(
        producer="", dtype=str(weight_dtype), shape=shape,
        storage_shape=storage_shape, strides=default_strides,
        offset=default_offset, output="weight", role="weight",
        source=f"{op.op_id}:weight_format", materialized=True,
        write_bytes=total_write_bytes, contract="parameter_value")
    return (add(
        weight, role="weight", field="weight", shape=shape,
        dependency="", repeats_value=weight_repeats,
        explicit_bytes=total_explicit,
        expected_elements=expected_weight_elements,
    ),)


def _declared_producer(metadata, keys, op_deps):
    for key in keys:
        raw = metadata.get(key)
        if isinstance(raw, str) and raw and (not op_deps or raw in op_deps):
            raw_out = metadata.get(f"{key}_output")
            out_name = raw_out if isinstance(raw_out, str) and raw_out else None
            return raw, out_name
    return "", None


def _matmul_input_values(
    op, metadata, source_dtype, produced_by_dep, ports_by_dep,
    repeats, explicit, selected, add,
):
    m, _, k = op.local_shape
    expected_rows = float(metadata.get("expected_rows", m))
    shape_rows = int(math.ceil(expected_rows)) if expected_rows > 0 else 0
    raw_gl_producer = metadata.get("group_list_producer")
    group_list_producer = (
        raw_gl_producer if isinstance(raw_gl_producer, str) and raw_gl_producer else ""
    )
    activation_deps = tuple(dep for dep in op.deps if dep != group_list_producer)
    payload_producer, payload_output = _declared_producer(
        metadata, ("activation_payload_producer", "payload_producer"),
        op.deps,
    )
    scale_producer, scale_output = _declared_producer(
        metadata, ("activation_scale_producer", "input_scale_producer"),
        op.deps,
    )
    if not scale_producer:
        for dep in activation_deps:
            for entry in ports_by_dep.get(dep, ()):
                port_out = entry.get("output")
                if port_out in {"activation_scales", "scale"}:
                    scale_producer = dep
                    scale_output = port_out
                    break
            if scale_producer:
                break
    if payload_producer:
        dependency = payload_producer
    else:
        non_scale_deps = (
            tuple(dep for dep in activation_deps if dep != scale_producer)
            if scale_producer else activation_deps
        )
        candidates = non_scale_deps or activation_deps
        port_dep = next(
            (
                dep for dep in candidates
                if any(
                    entry.get("output")
                    and entry.get("output") not in {"activation_scales", "scale", "group_list"}
                    for entry in ports_by_dep.get(dep, ())
                )
            ),
            "",
        )
        dependency = port_dep or (candidates[0] if candidates else "")
    if dependency and payload_output:
        primary = _select_named_value(op, dependency, payload_output, produced_by_dep)
        primary_output = payload_output if primary is not None else None
        if primary is None:
            primary, primary_output = selected(dependency)
    elif dependency:
        primary, primary_output = selected(dependency)
    else:
        primary, primary_output = (None, None)
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
    scale_dep = scale_producer or dependency
    if scale_output:
        scale = _select_named_value(op, scale_dep, scale_output, produced_by_dep)
    else:
        scale = next((value for value in produced_by_dep.get(scale_dep, ())
                      if value.get("output") == "activation_scales"), None)
        if scale is None and scale_producer and scale_producer != dependency:
            scale, _ = selected(scale_producer)
    if scale is None and metadata.get("activation_scale_format"):
        shape = tuple(metadata.get("activation_scale_shape") or ())
        scale = typed_value(
            producer=scale_dep,
            dtype=str(metadata["activation_scale_format"]), shape=shape,
            storage_shape=_row_shape(metadata, shape)[1],
            output="activation_scales", role="scale",
            source=f"{op.op_id}:activation_scale_format",
            materialized=True,
            write_bytes=metadata.get("activation_scale_bytes"),
            contract="declared_scale_operand")
    scale_already_seen = (
        scale_dep == dependency
        and (scale_output or "activation_scales") in seen_outputs
    )
    if scale is not None and not scale_already_seen:
        values.append(add(scale, role="scale", field="activation_scales",
                          shape=tuple(scale.get("logical_shape") or ()),
                          dependency=scale_dep,
                          repeats_value=repeats.get("activation_scales", repeats.get(scale_dep, 1)),
                          explicit_bytes=explicit.get("activation_scales", explicit.get(scale_dep))))
    if group_list_producer:
        values.append(_group_list_input_value(
            op, metadata, group_list_producer, produced_by_dep,
            repeats, explicit, selected, add,
        ))
    weight_dtype = metadata.get("weight_format") or metadata.get("weight_dtype")
    if weight_dtype:
        values.extend(_weight_input_values(
            op, metadata, weight_dtype, repeats, explicit, add,
        ))
    return tuple(values)


_PRODUCER_FIELDS = {
    "activation_scale_producer": "activation_scales",
    "input_scale_producer": "activation_scales",
    "activation_payload_producer": None,
    "payload_producer": None,
    "topk_ids_producer": "topk_ids",
    "topk_weights_producer": "topk_weights",
    "inverse_index_producer": "inverse_index",
    "group_list_producer": "group_list",
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
