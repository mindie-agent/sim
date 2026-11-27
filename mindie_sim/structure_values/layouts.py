"""Declared operator layouts, views and state values.

[Role] Project native layouts, source views and operator-specific value facts.
[Invariants] Preserve producer ports, state and representation attributes from declared metadata.
[Timeouts & Locks] Bounded by operator metadata and input ports. No I/O, waits or locks.
[Side-Effect Policy] Builds structural facts from the operator and producer records.
"""
import math

from .records import _nbytes, _shape, _typed_record_value, _value


_EXPLICIT_LAYOUT_KINDS = frozenset({
    "native_tensor_abi", "layernorm_active", "attention_active_qkv",
    "sparse_attention_active", "quant_lightning_indexer_active",
    "lightning_indexer_active", "attention_paged_active_qkv",
    "attention_paged_mla_qk_rope", "mla_prolog_active", "dynamic_quant_active",
    "add_rmsnorm_dynamic_quant_active", "rmsnorm_mxquant_active",
})
_SEMANTIC_ATTR_KEYS = (
    "norm_eps", "epsilon", "norm_has_weight", "norm_has_bias",
    "norm_weight_offset", "norm_gamma_runtime_format", "norm_gamma_storage_format",
    "reduction_width", "activation_kind", "activation_beta",
    "activation_linear_beta", "activation_compute_dtype", "activation_group_size",
    "causal", "window", "scale", "state_dtype", "conv_state_dtype",
    "conv_kernel", "state_layout", "slots", "commit_policy",
)


def _layout_hint(operator):
    return "strided" if operator.get("layout") in {"contiguous", "strided"} else None


def _input_ports(metadata):
    ports = []
    for record in metadata.get("fused_origin_scopes", ()):
        if record.get("kind") == "input_ports":
            for port in record.get("ports", ()):
                if port not in ports:
                    ports.append(port)
    return ports


def _metadata_boundary(metadata):
    for record in reversed(metadata.get("fused_origin_scopes", ())):
        if record.get("kind") in {"transform_boundary", "native_fusion_boundary",
                                  "activation_materialization"}:
            boundary = record.get("boundary")
            if isinstance(boundary, dict):
                return boundary
    boundary = metadata.get("transform_boundary")
    return boundary if isinstance(boundary, dict) else None


def _audit_inputs(operator, boundary, typed):
    if boundary.get("inputs"):
        return list(boundary["inputs"]), []
    if typed.get("operands"):
        return list(typed["operands"]), []
    candidates = list(boundary.get("typed_inputs")
                      or typed.get("inputs") or ())
    deps = set(operator.get("deps") or ())
    actual = [record for record in candidates
              if record.get("producer") in deps
              and record.get("role") in {"activation", "input"}]
    if not actual:
        actual = [record for record in candidates
                  if record.get("producer")
                  and record.get("role") in {"activation", "input"}]
    declarations = [record for record in candidates if record not in actual]
    return actual, declarations


def _audit_projection(operator, boundary):
    op_id = operator.get("op_id", "")
    typed = boundary.get("typed") if isinstance(boundary.get("typed"), dict) else {}
    raw_inputs, declaration_inputs = _audit_inputs(operator, boundary, typed)
    raw_outputs = boundary.get("typed_outputs") or typed.get("external_outputs") or ()
    inputs = [_typed_record_value(record, "input", name=f"input{index}", op_id=op_id)
              for index, record in enumerate(raw_inputs)]
    for value, record in zip(inputs, raw_inputs):
        if record.get("consumer") and record.get("consumer") != op_id:
            value["attrs"]["declared_consumer"] = record["consumer"]
            value["attrs"]["consumer"] = op_id
            value["attrs"]["consumer_basis"] = "producer_dependency"
    outputs = [_typed_record_value(record, "output", name=record.get("output"),
                                   op_id=op_id)
               for record in raw_outputs]
    declared = []
    seen = {(value["producer"], value["output"]) for value in (*inputs, *outputs)}
    for record in declaration_inputs:
        value = _typed_record_value(record, record.get("role") or "parameter",
                                    name=record.get("output") or record.get("field"),
                                    op_id=op_id)
        value["attrs"]["declaration_only"] = True
        if value["output"] in {item["output"] for item in outputs} and (
                value["producer"] != op_id):
            value["attrs"]["conflict_with_output"] = True
        key = (value["producer"], value["output"])
        if key not in seen:
            declared.append(value)
            seen.add(key)
    for record in typed.get("internal", ()):
        key = (record.get("producer"), record.get("output"))
        if key in seen:
            continue
        value = _typed_record_value(record, "internal", name=record.get("output"),
                                    op_id=op_id)
        value["attrs"]["internal"] = True
        declared.append(value)
        seen.add(key)
    for record in (typed.get("inputs") or ()):
        value = _typed_record_value(record, record.get("role") or "parameter",
                                    name=record.get("field"), op_id=op_id)
        value["attrs"]["declaration_only"] = True
        declared.append(value)
    roles = dict(boundary.get("typed_roles") or {})
    roles.update(typed.get("roles") or {})
    for role, records in roles.items():
        if role in {"input", "output"}:
            continue
        for record in records:
            field = record.get("field") or role
            key = (op_id, field)
            if key in seen:
                continue
            declared.append(_value(
                f"{role}:{field}", role, producer=op_id, output=field,
                dtype=record.get("dtype"), shape=record.get("shape"),
                layout_kind=record.get("layout_kind") or "strided",
                attrs={"source": f"{op_id}:typed_roles.{role}.{field}",
                       "declared_contract": "typed_role"},
                contract="typed_role"))
            seen.add(key)
    external_outputs = (boundary.get("external_outputs")
                        or typed.get("external_ports")
                        or [value["output"] for value in outputs])
    return {
        "inputs": inputs, "outputs": outputs, "declared": declared,
        "semantic_attrs": list(boundary.get("semantic_attrs") or ()),
        "external_outputs": list(external_outputs),
        "external_ports": list(typed.get("external_ports") or ()),
        "external_consumers": list(boundary.get("external_consumers") or ()),
        "field_dispositions": list(boundary.get("field_dispositions") or ()),
        "selection": boundary.get("selection"),
        "basis": "typed_records",
    }


def _layout_value(name, role, *, producer, output, dtype, layout, attrs,
                  materialized=True, contract="declared_view"):
    layout = layout if isinstance(layout, dict) else {}
    return _value(
        name, role, producer=producer, output=output, dtype=dtype,
        shape=layout.get("shape"), storage_shape=layout.get("storage_shape"),
        strides=layout.get("strides"), offset=layout.get("offset", 0),
        layout_kind="strided", attrs=attrs, materialized=materialized,
        contract=contract)


def _view_values(operator, context=None):
    meta = operator.get("metadata", {})
    source = meta.get("source_layout")
    target = meta.get("target_layout")
    if not source and not target:
        return None
    op_id = operator.get("op_id", "")
    deps = operator.get("deps", ())
    native = meta.get("native_source") if isinstance(meta.get("native_source"), dict) else {}
    common = {
        "materialization_reason": meta.get("materialization_reason"),
        "source_output": meta.get("source_output"),
        "native_source": {key: native.get(key) for key in
                          ("repository", "revision", "path", "sha256", "symbol",
                           "source_state") if native.get(key)},
    }
    inputs, outputs, gaps = [], [], []
    producer, producer_output, producer_basis = "", meta.get("source_output") or "source", "unresolved"
    if source and context and len(deps) == 1:
        resolved = _producer_inputs(operator, context)
        if len(resolved) == 1 and resolved[0]["producer"]:
            producer = resolved[0]["producer"]
            producer_output = resolved[0]["output"] or producer_output
            producer_basis = "single_declared_dependency"
    if source:
        inputs.append(_layout_value(
            "source_view", "input", producer=producer,
            output=producer_output,
            dtype=meta.get("input_format") or operator.get("dtype"),
            layout=source,
            attrs={**common, "source": "operator.metadata.source_layout",
                   "layout_source": "declared_source_layout",
                   "producer_basis": producer_basis,
                   "producer_unresolved": not bool(producer),
                   "view": "source"}))
        if not producer:
            gaps.append({"code": "source_view_producer_unresolved",
                         "detail": {"op_id": op_id, "deps": list(deps),
                                    "source_output": meta.get("source_output")}})
    else:
        gaps.append({"code": "view_source_layout_missing",
                     "detail": {"op_id": op_id}})
    if target:
        outputs.append(_layout_value(
            "target_view", "output", producer=op_id, output="output",
            dtype=meta.get("output_format") or operator.get("dtype"),
            layout=target,
            attrs={**common, "source": "operator.metadata.target_layout",
                   "layout_source": "declared_target_layout",
                   "view": "target"}))
    else:
        gaps.append({"code": "view_target_layout_missing",
                     "detail": {"op_id": op_id}})
    return {"inputs": inputs, "outputs": outputs, "declared": [],
            "basis": "declared_view", "gaps": gaps}


def _producer_inputs(operator, context):
    context = context or {}
    declared_op = context.get("op")
    outputs_by_id = context.get("outputs_by_id") or {}
    if declared_op is None or not outputs_by_id:
        return []
    from mindie_sim.prediction.implementations import producer_input_values

    try:
        records = producer_input_values(declared_op, outputs_by_id=outputs_by_id)
    except (KeyError, ValueError, TypeError):
        return []
    return [_typed_record_value(record, "input", name=record.get("role") or "input",
                                op_id=operator.get("op_id", ""))
            for record in records]


def _swiglu_values(operator, contract, context):
    meta = operator.get("metadata", {})
    op_id = operator.get("op_id", "")
    shape, dtype = _shape(contract[1]), contract[2]
    output = _value("output", "output", producer=op_id, dtype=dtype, shape=shape,
                    storage_shape=shape, layout_kind=_layout_hint(operator),
                    attrs={"source": "contract[1:3]", "shape_basis": "swiglu_output"},
                    contract="swiglu_semantic_output")
    inputs = _producer_inputs(operator, context)
    gaps = []
    if not inputs:
        rows = shape[0] if shape else None
        width = shape[-1] if shape else None
        read = meta.get("read_bytes_override", meta.get("read_bytes"))
        dtype_bytes = _nbytes(dtype, [1])
        input_width = None
        if rows and read and dtype_bytes:
            unit = read / (rows * dtype_bytes)
            if float(unit).is_integer():
                input_width = int(unit)
        basis = "declared_read_bytes"
        if input_width is None and width and meta.get("activation_kind") == "silu":
            input_width = 2 * width
            basis = "swiglu_gate_up_width_2x"
        if rows and input_width:
            inputs = [_value(
                "input", "input", producer="",
                output="output", dtype=meta.get("input_format") or operator.get("dtype"),
                shape=[rows, input_width], storage_shape=[rows, input_width],
                layout_kind=_layout_hint(operator),
                attrs={"source": "swiglu_declared_geometry", "shape_basis": basis},
                contract="declared_operand_value")]
        else:
            gaps.append({"code": "swiglu_input_unresolved",
                         "detail": {"rows": rows, "output_width": width,
                                    "read_bytes": read}})
    if inputs and not inputs[0].get("producer"):
        gaps.append({"code": "swiglu_input_producer_unresolved",
                     "detail": {"deps": list(operator.get("deps") or ())}})
    return {"inputs": inputs, "outputs": [output], "declared": [],
            "basis": "contract+producer" if inputs else "contract",
            "gaps": gaps}


def _source_layout_producer(operator, layout, context):
    metadata_by_id = (context or {}).get("metadata_by_id") or {}
    target = tuple(layout.get("storage_shape") or ())
    target_elements = math.prod(target) if target else 0
    matches = []
    for dep in operator.get("deps") or ():
        meta = metadata_by_id.get(dep) or {}
        for candidate in meta.get("source_layouts") or ():
            storage = tuple(candidate.get("storage_shape") or ())
            if (target_elements and storage
                    and math.prod(storage) == target_elements):
                if dep not in matches:
                    matches.append(dep)
                break
    return matches[0] if len(matches) == 1 else ""


def _source_view_value(name, role, *, producer, output, dtype, layout,
                      contract_shape, source_index, producer_basis="",
                      output_unresolved=False, root_version=None):
    layout = layout if isinstance(layout, dict) else {}
    shape = _shape(layout.get("shape"))
    storage = _shape(layout.get("storage_shape"))
    attrs = {
        "source": f"operator.metadata.source_layouts[{source_index}]",
        "shape_basis": "declared_source_layout",
        "source_layout_index": source_index,
        "contract_shape": list(contract_shape) if contract_shape else None,
        "shared_storage": True,
        "separate_materialization": False,
        "producer_basis": producer_basis,
        "producer_unresolved": not bool(producer),
        "producer_output_unresolved": output_unresolved,
        "root_storage_shape": list(storage) if storage else None,
        "root_storage_bytes": _nbytes(dtype, storage),
        "root_producer": producer,
        "root_output": output,
        "root_state_version": root_version,
    }
    return _value(
        name, role, producer=producer, output=output, dtype=dtype,
        shape=shape, storage_shape=storage, strides=layout.get("strides"),
        offset=layout.get("offset", 0), layout_kind="strided", attrs=attrs,
        materialized=True, contract="declared_source_view", write_bytes=0.0)


def _recurrent_values(operator, contract, context=None):
    meta = operator.get("metadata", {})
    op_id = operator.get("op_id", "")
    tokens, key_heads, heads, dk, dv = contract[1]
    state_dtype = contract[3]
    dtype = operator.get("dtype") or meta.get("input_format")
    deps = operator.get("deps") or ()
    layouts = list(meta.get("source_layouts") or ())
    q_shape = [tokens, key_heads, dk]
    k_shape = [tokens, key_heads, dk]
    v_shape = [tokens, heads, dv]
    operand_unresolved = []

    def operand(name, role, shape, layout=None, source_index=0):
        if layout is not None:
            producer = _source_layout_producer(operator, layout, context)
            root_version = None
            for root in ((context or {}).get("outputs_by_id", {}).get(producer) or ()):
                if root.get("output") == "output":
                    root_version = root.get("state_version")
                    break
            return _source_view_value(
                name, role, producer=producer, output=("output" if producer else ""),
                dtype=dtype, layout=layout, contract_shape=shape,
                source_index=source_index,
                producer_basis=("shared_source_layout" if producer else
                                "unresolved"),
                output_unresolved=not bool(producer),
                root_version=root_version)
        producer = ""
        if context and (context.get("metadata_by_id") or {}):
            operand_unresolved.append(name)
        return _value(
            name, role, producer=producer, output="output", dtype=dtype,
            shape=shape, layout_kind=_layout_hint(operator),
            attrs={"source": "contract[1]",
                   "shape_basis": "recurrent_declared_geometry",
                   "producer_unresolved": True},
            contract="declared_operand_value")

    q = operand("query", "input", q_shape,
                layouts[0] if len(layouts) >= 1 else None, 0)
    k = operand("key", "input", k_shape,
                layouts[1] if len(layouts) >= 2 else None, 1)
    v = operand("value", "input", v_shape,
                layouts[2] if len(layouts) >= 3 else None, 2)
    state = meta.get("gdn_state_contract") or (
        (meta.get("program_execution") or {}).get("state_contract")
        if isinstance(meta.get("program_execution"), dict) else None) or {}
    layout = state.get("state_layout")
    raw_active_slots = state.get("active_slots")
    if raw_active_slots is not None:
        active_count = len(raw_active_slots)
    elif state.get("slots") is not None:
        active_count = int(state["slots"])
    else:
        active_count = None
    capacity = state.get("state_capacity")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        capacity = None
    active_shape = None
    if active_count is not None and layout == "V,K":
        active_shape = [active_count, heads, dv, dk]
    elif active_count is not None and layout == "K,V":
        active_shape = [active_count, heads, dk, dv]
    capacity_shape = None
    if active_shape is not None and capacity is not None:
        capacity_shape = [capacity, *active_shape[1:]]
    state_attrs = {
        "source": "operator.metadata.gdn_state_contract",
        "shape_basis": "declared_state_layout",
        "state_layout": layout, "active_slots": state.get("active_slots"),
        "state_capacity": capacity,
        "state_capacity_lower_bound": state.get("state_capacity_lower_bound"),
        "targets": state.get("targets"),
        "slot_indices": state.get("slot_indices"),
        "slot_source": state.get("slot_source"),
        "publish_point": state.get("publish_point"),
        "read_versions": state.get("read_versions"),
        "capacity_basis": ("declared_state_capacity" if capacity is not None
                           else "capacity_not_declared"),
    }
    output = _value("output", "output", producer=op_id, output="output", dtype=dtype,
                    shape=v_shape, layout_kind=_layout_hint(operator),
                    attrs={"source": "contract[1]",
                           "shape_basis": "recurrent_declared_geometry"},
                    contract="recurrent_semantic_output")
    state_value = _value(
        "state", "input", producer="", output="state", dtype=state_dtype,
        shape=active_shape, storage_shape=capacity_shape,
        storage_unknown=capacity_shape is None,
        layout_kind=layout or None, attrs=state_attrs,
        contract="declared_state_value")
    state_out = _value(
        "state_out", "output", producer=op_id, output="state", dtype=state_dtype,
        shape=active_shape, storage_shape=capacity_shape,
        storage_unknown=capacity_shape is None,
        layout_kind=layout or None,
        attrs={**state_attrs, "commit_versions": state.get("commit_versions")},
        contract="declared_state_value")
    gaps = []
    if active_shape is None:
        gaps.append({"code": "state_shape_not_declared",
                     "detail": {"state_layout": layout,
                                "active_slots": state.get("active_slots"),
                                "heads": heads, "dk": dk, "dv": dv}})
    if operand_unresolved:
        gaps.append({"code": "operand_producer_unresolved",
                     "detail": {"operands": operand_unresolved,
                                "effect": "no unique source_layout producer"}})
    if any(not value["producer"] for value in (q, k, v)):
        gaps.append({"code": "source_view_producer_unresolved",
                     "detail": {"operands": [value["name"] for value in (q, k, v)
                                             if not value["producer"]]}})
    if capacity_shape is None:
        gaps.append({"code": "state_capacity_not_declared",
                     "detail": {"state_layout": layout,
                                "active_shape": active_shape,
                                "active_slots": state.get("active_slots"),
                                "state_capacity": state.get("state_capacity"),
                                "effect": "storage_shape/storage_bytes unknown; active bytes retained"}})
    return {"inputs": [q, k, v, state_value], "outputs": [output, state_out],
            "declared": [], "basis": "contract+state", "gaps": gaps}


def _semantic_attrs(operator, boundary_attrs):
    meta = operator.get("metadata", {})
    attrs = list(boundary_attrs or ())
    seen = {(item.get("name"), item.get("value")) for item in attrs
            if isinstance(item, dict)}
    for key in _SEMANTIC_ATTR_KEYS:
        if key not in meta:
            continue
        item = {"name": key, "value": meta[key], "unit": "",
                "source": f"{operator.get('op_id', '')}.metadata"}
        if (key, meta[key]) not in seen:
            attrs.append(item)
            seen.add((key, meta[key]))
    return attrs


def _state_facts(operator):
    meta = operator.get("metadata", {})
    execution = meta.get("program_execution")
    execution = execution if isinstance(execution, dict) else {}
    contract = meta.get("gdn_state_contract") or execution.get("state_contract")
    effects = meta.get("gdn_state_effects") or execution.get("state_effects")
    return contract, effects


def _postprocess_compact_layout(values, operator, contract):
    if contract is None or contract[0] in _EXPLICIT_LAYOUT_KINDS:
        return
    hint = _layout_hint(operator)
    for value in (*values["inputs"], *values["outputs"], *values["declared"]):
        if value["layout_kind"] == "ND" and str(value["attrs"].get("source", "")).startswith("contract"):
            value["layout_kind"] = hint
            value["layout"] = hint
            value["attrs"]["layout_basis"] = "compact_contract_does_not_declare_layout"


def _tensor_copy_values(operator, contract, context=None):
    """Public values for a copy/concatenate without a target view.

    A concatenate's source_layouts describe slices of one root allocation.  The
    public output is that root (shape from its declared semantic size when it
    agrees with the root storage); consumers keep their own view shapes.
    """
    meta = operator.get("metadata", {})
    op_id = operator.get("op_id", "")
    layouts = [item for item in (meta.get("source_layouts") or ())
               if isinstance(item, dict)]
    if not layouts:
        return None
    dtype = meta.get("output_format") or operator.get("dtype")
    storage = _shape(layouts[0].get("storage_shape"))
    storage_elements = math.prod(storage) if storage else 0
    semantic = _shape(operator.get("semantic_shape") or operator.get("local_shape"))
    shape = semantic if semantic and (not storage_elements
                                      or math.prod(semantic) == storage_elements) else storage
    if shape is None:
        return None
    output = _value(
        "output", "output", producer=op_id, output="output", dtype=dtype,
        shape=shape, storage_shape=storage, layout_kind="strided",
        attrs={"source": "operator.metadata.source_layouts",
               "shape_basis": "declared_semantic_shape" if semantic and
               math.prod(shape) == math.prod(semantic) else "root_storage",
               "materialization_reason": meta.get("materialization_reason"),
               "root_storage_shape": list(storage) if storage else None,
               "root_elements": storage_elements,
               "source_views": [{"shape": list(_shape(item.get("shape")) or ()),
                                 "offset": item.get("offset", 0),
                                 "strides": list(item.get("strides") or ())}
                                for item in layouts]},
        contract="declared_copy_output")
    gaps = []
    if len(layouts) > 1:
        gaps.append({"code": "copy_source_views_preserved",
                     "detail": {"op_id": op_id, "views": len(layouts),
                                "root_storage_shape": list(storage) if storage else None}})
    return {"inputs": [], "outputs": [output], "declared": [], "gaps": gaps,
            "basis": "source_layouts"}


def _attention_values(operator):
    meta = operator.get("metadata", {})
    gqa = meta.get("gqa_decode") if isinstance(meta.get("gqa_decode"), dict) else {}
    execution = meta.get("program_execution")
    kv = (execution or {}).get("kv_cache_access") if isinstance(execution, dict) else None
    op_id = operator.get("op_id", "")
    deps = operator.get("deps") or ()
    query_dtype = gqa.get("query_dtype") or operator.get("dtype")
    kv_dtype = gqa.get("kv_dtype") or query_dtype
    requests = gqa.get("requests")
    if isinstance(requests, (list, tuple)) and requests:
        total_query_tokens = sum(int(req[0]) for req in requests)
    else:
        total_query_tokens = gqa.get("uniform_query_tokens")
    out_head_dim = gqa.get("v_head_dim")
    if out_head_dim is None:
        mte3 = operator.get("mte3_bytes")
        elem_bytes = _nbytes(query_dtype, [1])
        q_heads = gqa.get("query_heads")
        if (isinstance(mte3, (int, float)) and mte3 > 0
                and total_query_tokens and q_heads and elem_bytes):
            inferred = float(mte3) / (total_query_tokens * q_heads * elem_bytes)
            rounded = int(round(inferred))
            if rounded >= 1 and abs(inferred - rounded) < 1e-6:
                out_head_dim = rounded
    if out_head_dim is None:
        out_head_dim = gqa.get("head_dim")
    query_attrs = {
        "source": "operator.metadata.gqa_decode",
        "shape_basis": "semantic_geometry_only",
        "query_tokens": total_query_tokens,
        "per_request_query_tokens": gqa.get("uniform_query_tokens"),
        "query_heads": gqa.get("query_heads"),
        "kv_heads": gqa.get("kv_heads"),
        "head_dim": gqa.get("head_dim"),
        "causal": gqa.get("causal"),
        "window": gqa.get("window"),
        "cp": gqa.get("cp"),
    }
    cache_attrs = {
        "source": "operator.metadata.program_execution.kv_cache_access",
        "shape_basis": "native_cache_storage_not_declared",
        "kv_heads": gqa.get("kv_heads"),
        "head_dim": gqa.get("head_dim"),
        "context_tokens": gqa.get("uniform_context_tokens"),
        "storage": (kv or {}).get("storage"),
        "page_size": (kv or {}).get("page_size"),
        "key_format": (kv or {}).get("key_format"),
        "value_format": (kv or {}).get("value_format"),
        "key_payload_row_bytes": (kv or {}).get("key_payload_row_bytes"),
        "value_payload_row_bytes": (kv or {}).get("value_payload_row_bytes"),
        "payload_scope": (kv or {}).get("payload_scope"),
    }
    inputs = [
        _value("query", "input", output="query", dtype=query_dtype, shape=None,
               attrs={**query_attrs, "producer_unresolved": True},
               contract="declared_semantic_geometry"),
        _value("key_cache", "input", output="key_cache", dtype=kv_dtype,
               shape=None, attrs={**cache_attrs, "external": True},
               contract="declared_cache_access"),
        _value("value_cache", "input", output="value_cache", dtype=kv_dtype,
               shape=None, attrs={**cache_attrs, "external": True},
               contract="declared_cache_access"),
    ]
    output_shape = None
    if (total_query_tokens is not None
            and gqa.get("query_heads") is not None
            and out_head_dim is not None):
        output_shape = [total_query_tokens, gqa["query_heads"], out_head_dim]
    outputs = [_value("output", "output", producer=op_id, output="output",
                      dtype=query_dtype, shape=output_shape,
                      attrs={"source": "operator.metadata.gqa_decode",
                             "shape_basis": "semantic_geometry_only",
                             "query_tokens": total_query_tokens,
                             "per_request_query_tokens": gqa.get("uniform_query_tokens"),
                             "query_heads": gqa.get("query_heads"),
                             "head_dim": out_head_dim},
                      contract="declared_semantic_geometry")]
    attrs = [
        {"name": "query_tokens", "value": total_query_tokens,
         "unit": "tokens", "source": "operator.metadata.gqa_decode"},
        {"name": "context_tokens", "value": gqa.get("uniform_context_tokens"),
         "unit": "tokens", "source": "operator.metadata.gqa_decode"},
        {"name": "query_heads", "value": gqa.get("query_heads"),
         "unit": "heads", "source": "operator.metadata.gqa_decode"},
        {"name": "kv_heads", "value": gqa.get("kv_heads"),
         "unit": "heads", "source": "operator.metadata.gqa_decode"},
        {"name": "head_dim", "value": gqa.get("head_dim"),
         "unit": "elements", "source": "operator.metadata.gqa_decode"},
        {"name": "query_dtype", "value": query_dtype, "unit": "",
         "source": "operator.metadata.gqa_decode"},
        {"name": "kv_dtype", "value": kv_dtype, "unit": "",
         "source": "operator.metadata.gqa_decode"},
        {"name": "causal", "value": gqa.get("causal"), "unit": "",
         "source": "operator.metadata.gqa_decode"},
        {"name": "window", "value": gqa.get("window"), "unit": "tokens",
         "source": "operator.metadata.gqa_decode"},
        {"name": "cp", "value": gqa.get("cp"), "unit": "",
         "source": "operator.metadata.gqa_decode"},
    ]
    gaps = [{"code": "attention_native_slots_unresolved",
             "detail": {"recipe": operator.get("recipe"),
                        "missing": ["query_layout", "key_cache_layout",
                                    "value_cache_layout", "mask", "scale",
                                    "block_table", "sequence_lengths",
                                    "output_layout", "cache_capacity"],
                        "known": ["query_tokens", "context_tokens", "query_heads",
                                  "kv_heads", "head_dim", "query_dtype", "kv_dtype",
                                  "causal", "window", "kv_cache_access"]}}]
    return {"inputs": inputs, "outputs": outputs, "declared": [],
            "basis": "semantic_geometry", "gaps": gaps,
            "semantic_attrs": attrs, "kv_cache_access": kv}
