"""Core tensor records for structural export.

[Role] Project compact and typed tensor declarations into per-value records.
[Invariants] Logical and storage representations retain their declared units and provenance.
[Timeouts & Locks] Bounded metadata arithmetic. No I/O, waits or locks.
[Side-Effect Policy] Returns JSON-ready records.
"""
import math

from mindie_sim.schema.common import dtype_bytes as _dtype_bytes


# --- Typed value projection -------------------------------------------------
# A native contract is a compact comparison tuple.  Consumers that bind an
# actual call also need one record per value with its own representation.  The
# projection preserves existing typed records/views/state; it never invents a
# tensor that neither the contract nor the declared metadata contains.

_TENSOR_LAYOUTS = frozenset({
    "ND", "NCL", "NZ", "NDCHW", "NHWC", "TND", "BNSD", "NBSD", "NDHWC",
})
_QUANT_DTYPES = frozenset({
    "int8", "int4", "int2", "fp8_e4m3", "fp8_e5m2", "mxfp4", "fp4", "fp4_e2m1",
})


def _shape(value):
    if (isinstance(value, (list, tuple))
            and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in value)):
        return [int(dim) for dim in value]
    return None


def _is_tensor(value):
    """A declared ``(shape, dtype, layout)`` triple, never a matrix tuple."""
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and _shape(value[0]) is not None
            and (value[1] is None or isinstance(value[1], str))
            and (value[2] is None
                 or (isinstance(value[2], str) and value[2].upper() in _TENSOR_LAYOUTS)))


def _is_tensor_seq(value):
    return (isinstance(value, (list, tuple)) and bool(value)
            and all(_is_tensor(item) for item in value))


def _nbytes(dtype, shape):
    if not dtype or shape is None:
        return None
    try:
        return float(math.prod(shape) * _dtype_bytes(dtype))
    except (KeyError, ValueError, TypeError):
        return None


def _value(name, role, *, producer="", output=None, dtype=None, shape=None,
           storage_shape=None, strides=None, offset=0, layout_kind=None,
           quant="", attrs=None, materialized=None, contract="", logical_bytes=None,
           storage_bytes=None, write_bytes=None, declared_bytes=None,
           storage="", alias_of="", state_version=None, binding=None,
           binding_name=None, storage_unknown=False):
    """One per-value record with logical/storage representation and provenance."""
    logical = _shape(shape) if shape is not None else None
    physical = (None if storage_unknown else
                _shape(storage_shape) if storage_shape is not None else logical)
    if physical is None and not storage_unknown:
        physical = logical
    record_attrs = {"source": attrs.get("source", "")} if attrs else {"source": ""}
    if attrs:
        record_attrs.update(attrs)
    if logical_bytes is None:
        logical_bytes = _nbytes(dtype, logical)
    if storage_bytes is None and not storage_unknown:
        storage_bytes = _nbytes(dtype, physical)
    return {
        "name": name,
        "output": output if output is not None else name,
        "role": role,
        "producer": producer,
        "shape": logical,
        "logical_shape": logical,
        "storage_shape": physical,
        "strides": _shape(strides) if strides is not None else None,
        "offset": int(offset or 0),
        "dtype": dtype,
        "layout": layout_kind,
        "layout_kind": layout_kind,
        "quant": quant if quant else (dtype if dtype in _QUANT_DTYPES else ""),
        "attrs": record_attrs,
        "materialized": materialized,
        "contract": contract,
        "bytes": logical_bytes,
        "storage_bytes": storage_bytes,
        "write_bytes": write_bytes,
        "declared_bytes": declared_bytes,
        "storage": storage,
        "alias_of": alias_of,
        "state_version": state_version,
        "binding": binding,
        "binding_name": binding_name,
    }


def _tensor_record(name, role, spec, path, *, attrs=None, producer="",
                   output=None, layout_kind=None, storage_unknown=False):
    shape, dtype, layout = spec
    record_attrs = {"source": path}
    if attrs:
        record_attrs.update(attrs)
    return _value(
        name, role, producer=producer, output=output, dtype=dtype, shape=shape,
        storage_shape=None if storage_unknown else shape,
        layout_kind=layout if layout_kind is None else layout_kind,
        attrs=record_attrs, contract="native_tensor_contract",
        storage_unknown=storage_unknown)


def _named_tensors(prefix, role, specs, path):
    return [_tensor_record(f"{prefix}{index}", role, spec, f"{path}[{index}]")
            for index, spec in enumerate(specs)
            if not (spec[1] is None and spec[2] is None)]


def _cache_head_tensors(prefix, role, specs, path, *, paged_page=False):
    """Project partial cache descriptors without claiming total backing storage."""
    records = []
    for index, spec in enumerate(specs):
        if spec[1] is None and spec[2] is None:
            continue
        desc_shape = _shape(spec[0])
        if paged_page and desc_shape and len(desc_shape) >= 2:
            row_shape = [1, *desc_shape[1:]]
            attrs = {
                "page_shape": desc_shape,
                "page_bytes": _nbytes(spec[1], desc_shape),
                "page_tokens": desc_shape[0],
                "row_shape": row_shape,
                "row_bytes": _nbytes(spec[1], row_shape),
                "shape_basis": "cache_page_descriptor_only",
                "capacity_basis": "capacity_not_declared",
            }
        else:
            attrs = {
                "row_shape": desc_shape,
                "row_bytes": _nbytes(spec[1], desc_shape),
                "shape_basis": "cache_head_row_descriptor_only",
                "capacity_basis": "capacity_not_declared",
            }
        records.append(_tensor_record(
            f"{prefix}{index}", role, spec, f"{path}[{index}]",
            storage_unknown=True,
            attrs=attrs,
        ))
    return records


def _matrix_values(contract):
    kind, geometry = contract[0], contract[1]
    if kind == "matrix":
        m, n, k = geometry
        activation, weight, output = contract[2], contract[3], contract[4]
        scale_shape, scale_format = contract[5], contract[6]
        inputs = [
            _tensor_record("activation", "input", ([m, k], activation, "ND"),
                           "contract[2]", attrs={"matrix_geometry": [m, n, k],
                                                 "shape_basis": "matrix_m_n_k_convention"}),
            _tensor_record("weight", "input", ([n, k], weight, "ND"),
                           "contract[3]", attrs={"matrix_geometry": [m, n, k],
                                                 "shape_basis": "matrix_m_n_k_convention"}),
        ]
        if scale_shape:
            inputs.append(_tensor_record(
                "activation_scale", "input", (scale_shape, scale_format, "ND"),
                "contract[5]", attrs={"not_a_payload_value": True}))
        outputs = [_tensor_record("output", "output", ([m, n], output, "ND"),
                                  "contract[4]", attrs={"matrix_geometry": [m, n, k]})]
        return inputs, outputs, [], {"matrix_geometry": [m, n, k]}
    if kind == "matrix_bias":
        m, n, k = geometry
        activation, weight, output = contract[2], contract[3], contract[4]
        bias_shape, bias_dtype = contract[5], contract[6]
        inputs = [
            _tensor_record("activation", "input", ([m, k], activation, "ND"),
                           "contract[2]", attrs={"shape_basis": "matrix_m_n_k_convention"}),
            _tensor_record("weight", "input", ([n, k], weight, "ND"),
                           "contract[3]", attrs={"shape_basis": "matrix_m_n_k_convention"}),
            _tensor_record("bias", "input", (bias_shape, bias_dtype, "ND"), "contract[5]"),
        ]
        outputs = [_tensor_record("output", "output", ([m, n], output, "ND"), "contract[4]")]
        return inputs, outputs, [], {"matrix_geometry": [m, n, k]}
    if kind == "grouped_matrix":
        bank, m, n, k = geometry
        activation, weight, output = contract[2], contract[3], contract[4]
        scale_shape, scale_format = contract[5], contract[6]
        inputs = [
            _tensor_record("activation", "input", ([m, k], activation, "ND"),
                           "contract[2]", attrs={"shape_basis": "matrix_m_n_k_convention"}),
            _tensor_record("weight", "input", ([bank, n, k], weight, "ND"),
                           "contract[3]", attrs={"expert_bank": bank,
                                                 "shape_basis": "grouped_matrix_m_n_k_convention"}),
        ]
        if scale_shape:
            inputs.append(_tensor_record(
                "activation_scale", "input", (scale_shape, scale_format, "ND"),
                "contract[5]", attrs={"not_a_payload_value": True}))
        outputs = [_tensor_record("output", "output", ([m, n], output, "ND"), "contract[4]")]
        return inputs, outputs, [], {"expert_bank": bank, "matrix_geometry": [m, n, k]}
    return None


def _common_attention_values(contract):
    """Project attention and MLA-prolog contract variants."""
    kind = contract[0]
    if kind == "attention_active_qkv":
        if (len(contract) == 4 and len(contract[2]) == 3
                and len(contract[3]) == 1
                and _is_tensor_seq(contract[2]) and _is_tensor_seq(contract[3])):
            inputs = [_tensor_record(name, "input", spec, f"contract[2][{index}]")
                      for index, (name, spec) in enumerate(zip(("query", "key", "value"),
                                                               contract[2]))]
            outputs = [_tensor_record("output", "output", contract[3][0], "contract[3][0]")]
            return inputs, outputs, [], {}
        return None
    if kind in {"sparse_attention_active", "quant_lightning_indexer_active",
                "lightning_indexer_active"}:
        # (query/weights..., cache heads..., primary output); the cache may be
        # repeated for key and value, so each entry keeps its own index.
        if len(contract) == 5 and _is_tensor_seq(contract[2]) \
                and _is_tensor_seq(contract[3]) and _is_tensor(contract[4]):
            inputs = (_named_tensors("input", "input", contract[2], "contract[2]")
                      + _cache_head_tensors("cache", "input", contract[3], "contract[3]"))
            outputs = [_tensor_record("output", "output", contract[4], "contract[4]")]
            return inputs, outputs, [], {}
        return None
    if kind == "attention_paged_active_qkv":
        if (len(contract) == 6 and _is_tensor(contract[2])
                and _is_tensor_seq(contract[3]) and _is_tensor(contract[4])):
            inputs = [_tensor_record("query", "input", contract[2], "contract[2]")]
            inputs += _cache_head_tensors(
                "cache", "input", contract[3], "contract[3]", paged_page=True,
            )
            outputs = [_tensor_record("output", "output", contract[4], "contract[4]")]
            return inputs, outputs, [], {"batch_count": contract[5]}
        return None
    if kind == "attention_paged_mla_qk_rope":
        if (len(contract) == 6 and _is_tensor_seq(contract[2])
                and _is_tensor_seq(contract[3]) and _is_tensor(contract[4])):
            inputs = (_named_tensors("query", "input", contract[2], "contract[2]")
                      + _cache_head_tensors("cache", "input", contract[3], "contract[3]"))
            outputs = [_tensor_record("output", "output", contract[4], "contract[4]")]
            return inputs, outputs, [], {"batch_count": contract[5]}
        return None
    if kind == "mla_prolog_active":
        if (len(contract) == 6 and _is_tensor(contract[2])
                and _is_tensor_seq(contract[3]) and _is_tensor_seq(contract[4])
                and _is_tensor_seq(contract[5])):
            inputs = ([_tensor_record("input", "input", contract[2], "contract[2]")]
                      + _named_tensors("parameter", "input", contract[3], "contract[3]")
                      + _cache_head_tensors("cache", "input", contract[4], "contract[4]"))
            outputs = _named_tensors("output", "output", contract[5], "contract[5]")
            return inputs, outputs, [], {}
        return None
    return None


_NATIVE_NORM_SLOTS = {
    "AddRmsNormBias": (
        (("x", False), ("residual", False), ("gamma", True), ("beta", True)),
        ("norm", "rstd", "residual"),
    ),
    "AddRmsNorm": (
        (("x", False), ("residual", False), ("gamma", True), ("beta", True)),
        ("norm", "rstd", "residual"),
    ),
    "GemmaRmsNorm": (
        (("x", False), ("gamma", True)),
        ("norm", "rstd"),
    ),
    "_layer_norm_fwd_1pass_kernel_npu": (
        (("x", False), ("gamma", True), ("gate", False)),
        ("output", "rstd"),
    ),
}


def _native_tensor_abi_values(contract):
    native = str(contract[1])
    slot_spec = _NATIVE_NORM_SLOTS.get(native)
    if slot_spec is None:
        return (_named_tensors("input", "input", contract[2], "contract[2]"),
                _named_tensors("output", "output", contract[3], "contract[3]"), [], {})
    in_specs, out_names = slot_spec
    inputs = []
    for index, spec in enumerate(contract[2]):
        if spec[1] is None and spec[2] is None:
            continue
        if index < len(in_specs):
            slot_name, is_param = in_specs[index]
        else:
            slot_name, is_param = f"input{index}", False
        inputs.append(_tensor_record(
            slot_name, "parameter" if is_param else "input", spec, f"contract[2][{index}]",
            attrs={"abi_ordinal": index, "abi_role": slot_name, "parameter": is_param},
        ))
    outputs = []
    for index, spec in enumerate(contract[3]):
        if spec[1] is None and spec[2] is None:
            continue
        slot_name = out_names[index] if index < len(out_names) else f"output{index}"
        outputs.append(_tensor_record(
            slot_name, "output", spec, f"contract[3][{index}]",
            output=slot_name,
            attrs={"abi_ordinal": index, "abi_role": slot_name},
        ))
    return inputs, outputs, [], {}


def _common_values(contract, operator=None):
    """Project the contract shapes that share an unambiguous tensor order."""
    kind = contract[0]
    if kind == "native_tensor_abi":
        if len(contract) == 4 and _is_tensor_seq(contract[2]) and _is_tensor_seq(contract[3]):
            return _native_tensor_abi_values(contract)
        return None
    if kind == "layernorm_active":
        # x, weight, bias, output; the native boundary is one row-wise call.
        if len(contract) == 6 and all(_is_tensor(item) for item in contract[2:]):
            inputs = [_tensor_record("x", "input", contract[2], "contract[2]"),
                      _tensor_record("weight", "input", contract[3], "contract[3]"),
                      _tensor_record("bias", "input", contract[4], "contract[4]")]
            outputs = [_tensor_record("output", "output", contract[5], "contract[5]")]
            return inputs, outputs, [], {}
        return None
    if kind in {
        "attention_active_qkv", "sparse_attention_active",
        "quant_lightning_indexer_active", "lightning_indexer_active",
        "attention_paged_active_qkv", "attention_paged_mla_qk_rope",
        "mla_prolog_active",
    }:
        return _common_attention_values(contract)
    if kind == "dynamic_quant_active":
        if len(contract) == 5 and all(_is_tensor(item) for item in contract[2:]):
            return ([_tensor_record("x", "input", contract[2], "contract[2]")],
                    [_tensor_record("output", "output", contract[3], "contract[3]"),
                     _tensor_record("scale", "output", contract[4], "contract[4]",
                                    attrs={"not_a_payload_value": True})], [], {})
        return None
    if kind == "add_rmsnorm_dynamic_quant_active":
        if (len(contract) == 5 and _is_tensor_seq(contract[2])
                and _is_tensor_seq(contract[3])):
            names = ("x", "residual", "gamma", "beta")
            inputs = [_tensor_record(names[index], "input", spec, f"contract[2][{index}]",
                                     attrs={"not_a_payload_value": names[index] == "beta"})
                      for index, spec in enumerate(contract[2])
                      if not (spec[1] is None and spec[2] is None)]
            outputs = _named_tensors("output", "output", contract[3], "contract[3]")
            return inputs, outputs, [], {}
        return None
    if kind == "rmsnorm_mxquant_active":
        if len(contract) == 6 and all(_is_tensor(item) for item in contract[2:]):
            return ([_tensor_record("x", "input", contract[2], "contract[2]"),
                     _tensor_record("gamma", "input", contract[3], "contract[3]")],
                    [_tensor_record("output", "output", contract[4], "contract[4]"),
                     _tensor_record("scale", "output", contract[5], "contract[5]",
                                    attrs={"not_a_payload_value": True})], [], {})
        return None
    if kind in {"swiglu", "relu"}:
        if len(contract) == 3:
            spec = (contract[1], contract[2], "ND")
            return ([_tensor_record("input", "input", spec, "contract[1:3]")],
                    [_tensor_record("output", "output", spec, "contract[1:3]")], [], {})
        return None
    if kind == "slice_copy":
        if len(contract) == 5:
            source = (contract[1], contract[3], "ND")
            target = (contract[2], contract[4], "ND")
            return ([_tensor_record("input", "input", source, "contract[1,3]")],
                    [_tensor_record("output", "output", target, "contract[2,4]")], [], {})
        return None
    if kind == "transpose_batchmatmul_active":
        if len(contract) == 6:
            batch, rows, n, k = contract[2]
            view = ((operator or {}).get("metadata") or {}).get("native_tensor_view") or {}
            if view.get("kind") == "transpose_batchmatmul":
                in_shape = _shape(view.get("input_shape"))
                wt_shape = _shape(view.get("weight_shape"))
                out_shape = _shape(view.get("output_shape"))
            else:
                in_shape = wt_shape = out_shape = None
            if in_shape is not None and wt_shape is not None and out_shape is not None:
                in_attrs = {"shape_basis": "declared_native_tensor_view"}
                if view.get("input_permutation"):
                    in_attrs["permutation"] = list(view["input_permutation"])
                out_attrs = {"shape_basis": "declared_native_tensor_view"}
                if view.get("output_permutation"):
                    out_attrs["permutation"] = list(view["output_permutation"])
                return (
                    [_tensor_record("input", "input", (in_shape, contract[3], "ND"),
                                    "contract[2:5]", attrs=in_attrs),
                     _tensor_record("weight", "input", (wt_shape, contract[4], "ND"),
                                    "contract[2:5]", attrs={"shape_basis": "declared_native_tensor_view"})],
                    [_tensor_record("output", "output", (out_shape, contract[5], "ND"),
                                    "contract[2:5]", attrs=out_attrs)], [], {},
                )
            activation = ([batch, rows, k], contract[3], "ND")
            weight = ([batch, k, n], contract[4], "ND")
            output = ([batch, rows, n], contract[5], "ND")
            return ([_tensor_record("input", "input", activation, "contract[2:5]",
                                    attrs={"shape_basis": "batch_rows_n_k_convention"}),
                     _tensor_record("weight", "input", weight, "contract[2:5]",
                                    attrs={"shape_basis": "batch_rows_n_k_convention"})],
                    [_tensor_record("output", "output", output, "contract[2:5]")], [], {})
        return None
    if kind == "causal_conv1d_active":
        if len(contract) == 7 and all(_is_tensor(item) for item in contract[2:]):
            inputs = [
                _tensor_record("x", "input", contract[2], "contract[2]"),
                _tensor_record("weight", "input", contract[3], "contract[3]"),
                _tensor_record("history", "input", contract[5], "contract[5]",
                               attrs={"state": True}),
            ]
            if not (contract[4][1] is None and contract[4][2] is None):
                inputs.append(_tensor_record("bias", "input", contract[4], "contract[4]"))
            outputs = [_tensor_record("output", "output", contract[6], "contract[6]")]
            return inputs, outputs, [], {}
        return None
    if kind == "kv_cache_store_active_elements":
        if len(contract) == 4 and len(contract[2]) == 2:
            values = []
            for name, (elements, dtype) in zip(("key", "value"), contract[2]):
                values.append(_value(
                    name, "input", dtype=dtype, shape=None,
                    attrs={"source": "contract[2]", "elements": elements,
                           "shape_basis": "active_element_count_only"},
                    contract="active_element_count"))
            return values, [], [], {"slot_shape": list(contract[3])}
        return None
    if kind in {"rmsnorm", "add_rmsnorm"}:
        if len(contract) == 4:
            spec = (contract[1], contract[2], "ND")
            inputs = [_tensor_record("x", "input", spec, "contract[1:3]")]
            if contract[3]:
                inputs.append(_tensor_record("weight", "input",
                                             ((contract[1][-1],), contract[2], "ND"),
                                             "contract[1,3]"))
            outputs = [_tensor_record("output", "output", spec, "contract[1:3]")]
            return inputs, outputs, [], {}
        return None
    return None


def _scan_tensors(value, path):
    """Collect declared tensor triples with their structural path."""
    found = []
    if _is_tensor(value):
        if not (value[1] is None and value[2] is None):
            found.append((path, value))
        return found
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_scan_tensors(item, f"{path}[{index}]"))
    return found


def _typed_record_value(record, default_role, *, name=None, op_id=""):
    raw_layout = record.get("layout") if isinstance(record.get("layout"), dict) else {}
    logical = record.get("logical_shape") or record.get("shape") or raw_layout.get("shape")
    storage = (record.get("storage_shape") or raw_layout.get("storage_shape")
               or logical)
    strides = record.get("strides") or raw_layout.get("strides")
    offset = record.get("offset", raw_layout.get("offset", 0))
    output = record.get("output") or name or default_role
    attrs = {
        "source": record.get("source") or f"{op_id}:{output}",
        "layout_source": "typed_record" if raw_layout else "not_declared",
        "declared_contract": record.get("contract", ""),
    }
    for key in ("consumer", "consumer_field", "value_id", "semantic_op",
                "semantic_producer", "semantic_output", "access_shape",
                "read_elements", "repeats", "read_bytes", "member", "port"):
        if record.get(key) is not None:
            attrs[key] = record[key]
    if record.get("hbm_materialized") is not None:
        attrs["hbm_materialized"] = record["hbm_materialized"]
    layout_kind = record.get("layout_kind")
    if not (isinstance(layout_kind, str)
            and (layout_kind.upper() in _TENSOR_LAYOUTS or layout_kind == "strided")):
        layout_kind = "strided" if raw_layout else None
    return _value(
        name or output, record.get("role") or default_role,
        producer=record.get("producer") or op_id, output=output,
        dtype=record.get("dtype"), shape=logical, storage_shape=storage,
        strides=strides, offset=offset, layout_kind=layout_kind,
        quant=record.get("quant", ""), attrs=attrs,
        materialized=record.get("materialized"),
        contract=record.get("contract", ""), logical_bytes=record.get("bytes"),
        storage_bytes=record.get("storage_bytes"), write_bytes=record.get("write_bytes"),
        declared_bytes=record.get("declared_bytes"), storage=record.get("storage", ""),
        alias_of=record.get("alias_of", ""), state_version=record.get("state_version"),
        binding=record.get("binding"), binding_name=record.get("binding_name"))
