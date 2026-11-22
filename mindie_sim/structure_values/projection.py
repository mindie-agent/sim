"""Assemble typed operator values for structural export.

[Role] Reconcile value records, producer bindings, scales and representation gaps.
[Invariants] Preserve declared ports and record missing geometry as explicit gaps.
[Timeouts & Locks] Bounded by the supplied operator and its value records. No waits or locks.
[Side-Effect Policy] Returns per-value structure facts without running the cost engine.
"""
import re

from .records import (
    _QUANT_DTYPES,
    _common_values,
    _matrix_values,
    _scan_tensors,
    _shape,
    _typed_record_value,
    _value,
)
from .layouts import (
    _attention_values,
    _audit_projection,
    _input_ports,
    _layout_hint,
    _metadata_boundary,
    _postprocess_compact_layout,
    _recurrent_values,
    _semantic_attrs,
    _state_facts,
    _swiglu_values,
    _tensor_copy_values,
    _view_values,
)


def _quantization(operator):
    meta = operator.get("metadata", {})
    fields = {}
    for key in ("quantization", "quant_category", "activation_format", "weight_format",
                "output_format", "input_format", "compute_format",
                "activation_scale_format", "weight_scale_format"):
        if meta.get(key) not in (None, ""):
            fields[key] = meta[key]
    if not fields:
        return None
    activation = fields.get("activation_format") or fields.get("input_format")
    weight = fields.get("weight_format")
    output = fields.get("output_format")
    scheme = None
    if activation and weight:
        if activation == weight == "int8":
            scheme = "w8a8"
        elif activation == "int8" and weight == "int4":
            scheme = "w4a8"
        elif weight == "mxfp4":
            scheme = "mxfp4"
        elif activation.startswith("fp8") and weight.startswith("fp8"):
            scheme = "fp8"
        elif activation == weight == "bf16":
            scheme = "bf16"
        else:
            scheme = f"a_{activation}_w_{weight}"
    elif output in _QUANT_DTYPES:
        # A standalone quantize/cast value, not a matrix weight scheme.
        scheme = f"dynamic_{output}"
    return {"scheme": scheme, **fields}


def _declared_metadata_values(operator):
    """Explicit operator shapes, used only when they name their own value."""
    meta, dtype = operator.get("metadata", {}), operator.get("dtype")
    layout = _layout_hint(operator)
    dtypes = {
        "input": meta.get("input_dtype") or meta.get("activation_format") or dtype,
        "weight": meta.get("weight_dtype") or meta.get("weight_format") or dtype,
        "output": meta.get("output_dtype") or meta.get("output_format") or dtype,
    }
    inputs, outputs = [], []
    for name, key, role in (("input", "input_shape", "input"),
                            ("weight", "weight_shape", "input"),
                            ("output", "output_shape", "output")):
        shape = _shape(meta.get(key))
        if shape is None:
            continue
        value = _value(
            name, role, producer=operator.get("op_id", "") if role == "output" else "",
            output=name, dtype=dtypes[name], shape=shape, storage_shape=shape,
            layout_kind=layout,
            attrs={"source": f"operator.metadata.{key}",
                   "layout_source": "operator.layout" if layout else "not_declared"},
            contract="declared_operator_value")
        (outputs if role == "output" else inputs).append(value)
    bias_shape = _shape(meta.get("bias_shape"))
    if bias_shape is not None and meta.get("bias_dtype"):
        inputs.append(_value(
            "bias", "input", output="bias", dtype=meta["bias_dtype"],
            shape=bias_shape, storage_shape=bias_shape, layout_kind=layout,
            attrs={"source": "operator.metadata.bias_shape",
                   "layout_source": "operator.layout" if layout else "not_declared"},
            contract="declared_operator_value"))
    return inputs, outputs


def _scale_details(meta):
    details = {}
    if meta.get("activation_scale_format"):
        details["activation"] = {
            "format": meta.get("activation_scale_format"),
            "shape": list(meta.get("activation_scale_shape") or ()) or None,
            "group_size": meta.get("activation_group_size"),
            "bytes": meta.get("activation_scale_bytes"),
            "in_mte2": meta.get("activation_scale_in_mte2"),
            "producer": meta.get("activation_producer"),
            "source": "operator.metadata.activation_scale_format",
        }
    if meta.get("weight_scale_format"):
        details["weight"] = {
            "format": meta.get("weight_scale_format"),
            "storage_format": meta.get("weight_scale_storage_format"),
            "bytes": meta.get("weight_scale_bytes"),
            "storage_bytes": meta.get("weight_scale_storage_bytes"),
            "group_size": meta.get("weight_scale_group_size"),
            "shape": list(meta.get("weight_scale_shape") or ()) or None,
            "source": "operator.metadata.weight_scale_format",
        }
    if meta.get("output_scale_format"):
        details["output"] = {
            "format": meta.get("output_scale_format"),
            "shape": list(meta.get("output_scale_shape") or ()) or None,
            "bytes": meta.get("output_scale_bytes"),
            "source": "operator.metadata.output_scale_format",
        }
    return details


def _scale_values(operator, values, quantization):
    meta = operator.get("metadata", {})
    details = _scale_details(meta)
    if not details:
        return
    op_id = operator.get("op_id", "")
    returns = set(operator.get("returns") or ())
    output_names = {value["output"] for value in values["outputs"]}
    scale_output = bool({"activation_scales", "activation_scale"} &
                        (returns | output_names))
    inputs_by_name = {value["name"]: value for value in values["inputs"]}
    inputs_by_output = {value["output"]: value for value in values["inputs"]}
    outputs_by_output = {value["output"]: value for value in values["outputs"]}
    for role, detail in details.items():
        name = f"{role}_scale"
        attrs = {
            "source": detail["source"], "association": role,
            "format": detail.get("format"),
            "storage_format": detail.get("storage_format") or detail.get("format"),
            "group_size": detail.get("group_size"),
            "bytes": detail.get("bytes"),
            "storage_bytes": detail.get("storage_bytes"),
            "shape_basis": ("declared_scale_shape" if detail.get("shape")
                            else "not_declared"),
        }
        if role == "activation":
            attrs.update(in_mte2=detail.get("in_mte2"),
                         producer=detail.get("producer"))
        if role == "activation" and scale_output:
            value = (next((item for item in values["outputs"]
                           if item["output"] in {"activation_scales", "activation_scale"}
                           and item["producer"] == op_id), None)
                     or outputs_by_output.get("activation_scales")
                     or outputs_by_output.get("activation_scale"))
            if value is None:
                value = _value("activation_scales", "scale", producer=op_id,
                               output="activation_scales",
                               dtype=detail.get("format"), shape=detail.get("shape"),
                               storage_shape=detail.get("shape"),
                               layout_kind=_layout_hint(operator), attrs=attrs,
                               contract="declared_scale_output")
                values["outputs"].append(value)
                outputs_by_output[value["output"]] = value
            value["dtype"] = detail.get("format") or value["dtype"]
            value["shape"] = value["shape"] or detail.get("shape")
            value["logical_shape"] = value["shape"]
            value["storage_shape"] = value["storage_shape"] or value["shape"]
            attrs.update(payload="payload", payload_producer=op_id)
            value["attrs"].update(attrs)
        else:
            value = inputs_by_name.get(name) or inputs_by_output.get(
                "activation_scales" if role == "activation" else name)
            if role == "activation":
                payload = (inputs_by_name.get("activation")
                           or inputs_by_output.get("payload"))
                if payload is not None:
                    attrs.update(payload_producer=payload.get("producer"),
                                 payload_output=payload.get("output"))
            if value is None:
                value = _value(name, "scale" if role != "output" else "auxiliary",
                               producer=op_id, output=f"{role}_scales",
                               dtype=detail.get("format"), shape=detail.get("shape"),
                               storage_shape=detail.get("shape"),
                               layout_kind=_layout_hint(operator),
                               attrs=attrs,
                               contract="declared_scale_association")
                target = values["inputs"] if role == "activation" else values["declared"]
                target.append(value)
            else:
                value["dtype"] = detail.get("format") or value["dtype"]
                value["shape"] = value["shape"] or detail.get("shape")
                value["logical_shape"] = value["shape"]
                value["storage_shape"] = value["storage_shape"] or value["shape"]
                value["attrs"].update(attrs)
            inputs_by_name[value["name"]] = value
            inputs_by_output[value["output"]] = value
    if quantization is not None:
        quantization["scales"] = {
            role: {key: detail.get(key) for key in
                   ("format", "storage_format", "shape", "group_size", "bytes",
                    "storage_bytes", "in_mte2", "producer") if detail.get(key) is not None}
            for role, detail in details.items()}


def _merge_values(values, extra):
    if extra is None:
        return
    for key in ("inputs", "outputs", "declared"):
        existing = {(value["name"], value["producer"], value["output"]): value
                    for value in values[key]}
        for value in extra.get(key, ()):
            current = existing.get((value["name"], value["producer"], value["output"]))
            if current is None:
                values[key].append(value)
                existing[(value["name"], value["producer"], value["output"])] = value
            else:
                for field, item in value.items():
                    if current.get(field) in (None, "", []) and item not in (None, "", []):
                        current[field] = item
                current["attrs"].update({name: item for name, item in value["attrs"].items()
                                         if item is not None})


def _contract_gap(operator, contract, declared):
    meta = operator.get("metadata", {})
    family, recipe = operator.get("family"), operator.get("recipe")
    if contract is None:
        if operator.get("source_boundary") == "call":
            return {"code": "source_call_boundary",
                    "detail": {"recipe": recipe,
                               "effect": "native tensors belong to expanded children"}}
        if meta.get("program_kind") == "matmul":
            return {"code": "matrix_tensor_abi_unavailable",
                    "detail": {"recipe": recipe,
                               "matrix_groups": meta.get("matrix_groups"),
                               "activation_format": meta.get("activation_format"),
                               "weight_format": meta.get("weight_format"),
                               "output_format": meta.get("output_format"),
                               "bias": bool(meta.get("bias")),
                               "activation_scale_shape": meta.get("activation_scale_shape")}}
        if family == "attention":
            return {"code": "attention_tensor_abi_unavailable",
                    "detail": {"recipe": recipe,
                               "attention_tensor_view": meta.get("attention_tensor_view")}}
        return {"code": "native_tensor_abi_unavailable",
                "detail": {"family": family, "recipe": recipe}}
    kind = contract[0]
    if not declared:
        return {"code": "contract_value_layout_unmodeled",
                "detail": {"contract_kind": kind,
                           "effect": "contract tuple retained; no per-value projection"}}
    return None


def _apply_input_ports(operator, values, context, gaps):
    ports = _input_ports(operator.get("metadata", {}))
    if not ports:
        return
    outputs_by_id = (context or {}).get("outputs_by_id") or {}
    op_id = operator.get("op_id", "")
    used = set()
    for port in ports:
        producer = port.get("op")
        name = port.get("output")
        if not producer or not name:
            continue
        candidates = [value for value in values["inputs"]
                      if id(value) not in used]
        target = None
        if name in {"payload", "activation", "output"}:
            target = next((value for value in candidates
                           if value["name"] in {"activation", "input", "source_view"}),
                          None)
        elif name in {"activation_scales", "activation_scale"}:
            target = next((value for value in candidates
                           if value["name"] in {"activation_scale", "activation_scales"}
                           or value["role"] == "scale"), None)
        if target is None:
            target = next((value for value in candidates
                           if value["name"] == name or value["output"] == name), None)
        if target is None:
            continue
        used.add(id(target))
        produced = list(outputs_by_id.get(producer) or ())
        source = next((value for value in produced
                       if value.get("output") == name), None)
        target["producer"] = producer
        target["output"] = name
        target["attrs"]["input_port"] = dict(port)
        target["attrs"]["input_port_source"] = "operator.metadata.input_ports"
        if source is None:
            target["attrs"]["producer_output_unresolved"] = True
            gaps.append({"code": "input_port_producer_unresolved",
                         "detail": {"op_id": op_id, "producer": producer,
                                    "output": name}})
            continue
        for key in ("dtype", "shape", "logical_shape", "storage_shape", "strides",
                    "offset", "layout", "layout_kind", "quant", "bytes",
                    "storage_bytes", "write_bytes", "declared_bytes", "storage",
                    "alias_of", "state_version", "binding", "binding_name"):
            if key in source and source[key] is not None:
                target[key] = source[key]
        target["attrs"]["producer_output"] = name
        target["attrs"]["producer_contract"] = source.get("contract")


def _fill_from_produced(existing, produced, *, replace=False, contract=None):
    keep_attrs = dict(existing.get("attrs") or {})
    ordinal = keep_attrs.get("abi_ordinal")
    matrix = contract is not None and str(contract[0]).startswith(
        ("matrix", "grouped_matrix"))
    for key in ("dtype", "shape", "logical_shape", "storage_shape", "strides",
                "offset", "layout", "layout_kind", "quant", "bytes",
                "storage_bytes", "write_bytes", "declared_bytes", "storage",
                "alias_of", "state_version", "binding", "binding_name",
                "materialized", "producer", "output"):
        if produced.get(key) is None:
            continue
        if matrix and key in {"layout", "layout_kind"} and not replace:
            continue
        if replace or existing.get(key) in (None, "", []):
            existing[key] = produced[key]
    if not replace and (existing.get("shape") != produced.get("shape")):
        keep_attrs["produced_shape_conflict"] = {
            "existing": existing.get("shape"), "produced": produced.get("shape")}
    if replace:
        existing["name"] = produced.get("output") or produced.get("name")
    keep_attrs["identity_source"] = "produced_values"
    if ordinal is not None:
        keep_attrs["abi_ordinal"] = ordinal
    existing["attrs"] = {**keep_attrs, **{
        key: value for key, value in produced.get("attrs", {}).items()
        if value is not None}}


def _merge_produced_outputs(operator, contract, values, produced):
    if not produced:
        return
    outputs = values["outputs"]
    op_id = operator.get("op_id", "")
    for index, record in enumerate(produced):
        actual = _typed_record_value(record, "output",
                                     name=record.get("output") or "output",
                                     op_id=op_id)
        match = next((value for value in outputs
                      if value.get("output") == actual["output"]
                      or value.get("name") == actual["name"]), None)
        if match is not None:
            _fill_from_produced(match, actual, contract=contract)
            continue
        generic = next((value for value in outputs
                        if re.fullmatch(r"output\d+", str(value.get("name")))),
                       None)
        if generic is not None:
            generic["attrs"]["abi_ordinal"] = outputs.index(generic)
            _fill_from_produced(generic, actual, replace=True, contract=contract)
            continue
        outputs.append(actual)
    # A native norm's auxiliary value is independent even when the producer
    # helper exposes only the primary output.  Keep the ABI ordinal mapping.
    if contract is not None and contract[0] == "native_tensor_abi":
        native = str(contract[1])
        if "RmsNorm" in native or "LayerNorm" in native:
            for ordinal, value in enumerate(outputs):
                if value.get("name") in {"output0", "output1"}:
                    value["attrs"]["abi_ordinal"] = ordinal
                    if ordinal == 0 and value["dtype"] not in {None, "fp32"}:
                        value["name"] = value["output"] = "norm"
                    elif ordinal == 1 and value["dtype"] == "fp32":
                        value["name"] = value["output"] = "rstd"


def _bind_value_ports(operator, contract, values, context, gaps, op_id):
    for value in values["outputs"]:
        if not value["producer"]:
            value["producer"] = op_id
        if value["output"] is None:
            value["output"] = value["name"]

    _apply_input_ports(operator, values, context, gaps)
    native_norm = (contract is not None and contract[0] == "native_tensor_abi"
                   and ("RmsNorm" in str(contract[1])
                        or "LayerNorm" in str(contract[1])))
    for index, value in enumerate(values["inputs"]):
        if (value["producer"] or value["role"] not in {"input", "activation"}
                or value["name"] in {"weight", "bias", "state", "key_cache",
                                     "value_cache"}
                or (native_norm and index > 0)):
            if native_norm and index > 0:
                value["attrs"]["parameter"] = True
            continue
        value["attrs"]["producer_unresolved"] = True
        gaps.append({"code": "operand_producer_unresolved",
                     "detail": {"value": value["name"],
                                "deps": list(operator.get("deps") or ()),
                                "effect": "no actual input port; not inferred by position"}})
    if contract is not None and str(contract[0]).startswith(
            ("matrix", "grouped_matrix")):
        for value in values["inputs"]:
            if value["name"] == "weight" and not value["producer"]:
                value["attrs"]["weight_source_unresolved"] = True
                gaps.append({"code": "weight_source_unresolved",
                             "detail": {"matrix_kind": contract[0],
                                        "weight_shape": value.get("shape"),
                                        "effect": "declared weight retained; producer not named"}})


def _project_non_boundary_values(operator, contract, context, meta, gaps):
    """Project values for operators that do not declare an audit boundary."""
    attention = (_attention_values(operator)
                 if operator.get("family") == "attention"
                 and meta.get("gqa_decode") and contract is None else None)
    if attention is not None:
        gaps.extend(attention["gaps"])
        return attention, attention, attention["semantic_attrs"], ["output"]
    view = (_view_values(operator, context)
            if meta.get("source_layout") or meta.get("target_layout") else None)
    if view is not None:
        gaps.extend(view["gaps"])
        return view, None, [], []
    family = operator.get("family")
    if family == "swiglu" and contract is not None:
        values = _swiglu_values(operator, contract, context)
        gaps.extend(values["gaps"])
        return values, None, [], []
    if family == "recurrent_vector" and contract is not None:
        values = _recurrent_values(operator, contract, context)
        gaps.extend(values["gaps"])
        return values, None, [], []
    if family == "tensor_copy":
        copied = _tensor_copy_values(operator, contract, context)
        if copied is not None:
            gaps.extend(copied["gaps"])
            return copied, None, [], []
        inputs, outputs = _declared_metadata_values(operator)
        basis = "operator_metadata" if inputs or outputs else "unavailable"
        return {"inputs": inputs, "outputs": outputs, "declared": [], "basis": basis}, None, [], []
    projected = (_matrix_values(contract) or _common_values(contract)
                 if contract is not None else None)
    if projected is not None:
        inputs, outputs, declared, _extra = projected
        return {"inputs": inputs, "outputs": outputs, "declared": declared, "basis": "contract"}, None, [], []
    if contract is not None:
        scanned = _scan_tensors(contract, "contract")
        declared = [
            _value(
                f"declared{index}", "declared", dtype=spec[1],
                shape=spec[0], storage_shape=spec[0],
                layout_kind=spec[2], attrs={"source": path},
                contract="native_tensor_contract")
            for index, (path, spec) in enumerate(scanned)
        ]
        if declared:
            gaps.append({"code": "tensor_roles_not_distinguished",
                         "detail": {"contract_kind": contract[0],
                                    "value_count": len(declared)}})
        return {"inputs": [], "outputs": [], "declared": declared, "basis": "contract_scan"}, None, [], []
    inputs, outputs = _declared_metadata_values(operator)
    basis = "operator_metadata" if inputs or outputs else "unavailable"
    return {"inputs": inputs, "outputs": outputs, "declared": [], "basis": basis}, None, [], []


def typed_values(operator, contract, *, context=None):
    """Project one operator's declared values and representation facts.

    Existing typed records (transform/native-fusion audits), declared source
    views and producer outputs take precedence over the compact comparison
    tuple. The tuple fills only fields it actually declares. Recognition is
    metadata-only: no measurement table, target duration or hardware rate is
    consulted, and an unmeasured shape is accepted.
    """
    meta = operator.get("metadata", {})
    context = context or {}
    op_id = operator.get("op_id", "")
    gaps = []
    quantization = _quantization(operator)
    state_contract, state_effects = _state_facts(operator)
    semantic_attrs, external_outputs, external_consumers = [], [], []
    external_ports = []
    field_dispositions, selection = [], None
    attention_data = None
    boundary = _metadata_boundary(meta)

    if boundary is not None:
        audit = _audit_projection(operator, boundary)
        values = {
            "inputs": audit["inputs"], "outputs": audit["outputs"],
            "declared": audit["declared"], "basis": "typed_records",
        }
        semantic_attrs = audit["semantic_attrs"]
        external_outputs = audit["external_outputs"]
        external_ports = audit["external_ports"]
        external_consumers = audit["external_consumers"]
        field_dispositions = audit["field_dispositions"]
        selection = audit["selection"]
        if contract is not None and (not values["inputs"] or not values["outputs"]):
            projected = _matrix_values(contract) or _common_values(contract)
            if projected is not None:
                extra = {"inputs": projected[0], "outputs": projected[1],
                         "declared": projected[2]}
                _merge_values(values, extra)
                if projected[0] or projected[1]:
                    values["basis"] = "typed_records+contract"
    else:
        values, attention_data, semantic_attrs, external_outputs = (
            _project_non_boundary_values(operator, contract, context, meta, gaps)
        )

    if contract is not None:
        _postprocess_compact_layout(values, operator, contract)

    produced = (context or {}).get("produced") or ()
    if produced and operator.get("family") not in {
            "recurrent_vector", "tensor_copy"}:
        _merge_produced_outputs(operator, contract, values, produced)

    _bind_value_ports(operator, contract, values, context, gaps, op_id)
    _scale_values(operator, values, quantization)
    semantic_attrs = _semantic_attrs(operator, semantic_attrs)
    if not external_outputs:
        external_outputs = [
            value["output"] for value in values["outputs"]
            if value["materialized"] is not False
            and value["role"] in {"output", "activation", "residual", "auxiliary"}]

    gap = _contract_gap(operator, contract,
                        values["inputs"] or values["outputs"] or values["declared"])
    if gap is not None:
        gaps.append(gap)
    for value in (*values["inputs"], *values["outputs"], *values["declared"]):
        missing = [key for key in ("dtype", "layout") if value.get(key) is None]
        if missing:
            gaps.append({"code": "value_representation_partial",
                         "detail": {"value": value["name"], "missing": missing}})
        if value.get("shape") is None:
            gaps.append({"code": "value_shape_not_declared",
                         "detail": {"value": value["name"],
                                    "shape_basis": value["attrs"].get("shape_basis")}})
    unique = []
    for gap in gaps:
        if gap not in unique:
            unique.append(gap)
    return {
        "basis": values["basis"],
        "contract_kind": contract[0] if contract is not None else None,
        "kv_cache_access": (attention_data or {}).get("kv_cache_access"),
        "inputs": values["inputs"],
        "outputs": values["outputs"],
        "declared": values["declared"],
        "quantization": quantization,
        "semantic_attrs": semantic_attrs,
        "state_contract": state_contract,
        "state_effects": state_effects,
        "external_outputs": external_outputs,
        "external_ports": external_ports,
        "external_consumers": external_consumers,
        "field_dispositions": field_dispositions,
        "input_ports": _input_ports(meta),
        "selection": selection,
        "gaps": unique,
    }
