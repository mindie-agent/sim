"""Typed tensor descriptors and operator output contracts.

[Role] Derive named output values, shapes, dtypes and layout from operator data.
[Invariants] Logical shapes, storage shapes and materialized writes retain their
  own meanings. Declared output and audit contracts take precedence over defaults.
[Timeouts & Locks] No blocking operations or locks.
[Side-Effect Policy] Build descriptors without mutating operators or metadata.
"""
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.tensors import TensorLayout, TensorValue

from .implementation_audit import _audit_outputs


def _contiguous_strides(shape):
    stride, strides = 1, []
    for size in reversed(shape):
        strides.append(stride)
        stride *= max(1, size)
    return tuple(reversed(strides))


def typed_value(*, producer, dtype, shape, output, role, source, quant="",
                materialized=True, layout_kind="strided", storage="", alias_of="",
                state_version=None, binding="static", binding_name="",
                storage_shape=None, write_bytes=None, declared_bytes=None,
                contract="", layout=None, strides=None, offset=0,
                semantic_producer="", semantic_output="", layout_name="",
                source_format=""):
    """One explicit value descriptor, independent of the op-level ``dtype``.

    ``shape`` is the logical value shape.  ``storage_shape`` is the actual
    allocation view (for example four captured rows for three valid rows).
    A matmul output is the (M,N) value, not the operator's (M,N,K) geometry.
    """
    if not isinstance(dtype, str) or not dtype:
        raise ModelingError(f"typed_value_dtype_required:{producer}:{output}")
    logical = tuple(int(size) for size in shape)
    physical = tuple(int(size) for size in (storage_shape or logical))
    if any(size < 0 for size in logical) or any(size < 0 for size in physical):
        raise ModelingError(f"typed_value_shape_negative:{producer}:{output}")
    if len(physical) < len(logical):
        physical = logical
    if isinstance(layout, str):
        layout_name = layout_name or layout
        layout = None
    if isinstance(layout, dict):
        for key in ("layout_name", "semantic_layout", "name", "layout"):
            if not layout_name and isinstance(layout.get(key), str) and layout[key]:
                layout_name = str(layout[key])
    if layout is not None:
        if isinstance(layout, dict):
            layout_shape = tuple(int(size) for size in
                                 (layout.get("shape") or logical))
            strides = tuple(int(size) for size in
                            (layout.get("strides") or _contiguous_strides(layout_shape)))
            offset = int(layout.get("offset", 0) or 0)
            physical = tuple(int(size) for size in
                             (layout.get("storage_shape") or physical))
            logical = layout_shape
        else:
            logical = tuple(int(size) for size in layout.shape)
            strides = tuple(int(size) for size in layout.strides)
            offset = int(layout.offset)
            physical = tuple(int(size) for size in (layout.storage_shape or physical))
    if strides is None:
        strides = _contiguous_strides(logical)
    if len(strides) != len(logical):
        raise ModelingError(f"typed_value_strides_rank_mismatch:{producer}:{output}")
    value = TensorValue(
        producer=producer, dtype=dtype,
        layout=TensorLayout(logical, tuple(strides), int(offset), physical),
        output=output, quant=quant, layout_kind=layout_kind, storage=storage,
        alias_of=alias_of, state_version=state_version, binding=binding,
        binding_name=binding_name)
    descriptor = value.to_dict()
    logical_bytes = None
    storage_bytes = None
    try:
        from mindie_sim.schema.common import dtype_bytes
        logical_bytes = float(math.prod(logical) * dtype_bytes(dtype))
        storage_bytes = float(math.prod(physical) * dtype_bytes(dtype))
    except (KeyError, ValueError):
        pass
    descriptor.update({
        "shape": list(logical),
        "logical_shape": list(logical),
        "storage_shape": list(physical),
        "role": role,
        "source": source,
        "materialized": bool(materialized),
        "contract": contract,
        "value_id": f"{producer}.{output}",
        "semantic_producer": semantic_producer or producer,
        "semantic_output": semantic_output or output,
    })
    if logical_bytes is not None:
        descriptor["bytes"] = logical_bytes
        descriptor["storage_bytes"] = storage_bytes
    if write_bytes is not None:
        descriptor["write_bytes"] = float(write_bytes)
    elif descriptor["materialized"] and logical_bytes is not None:
        descriptor["write_bytes"] = logical_bytes
    if declared_bytes is not None:
        descriptor["declared_bytes"] = float(declared_bytes)
    if layout_name:
        descriptor["layout_name"] = str(layout_name)
        descriptor["semantic_layout"] = str(layout_name)
    if source_format:
        descriptor["source_format"] = str(source_format)
    return descriptor


def typed_roles(source):
    """Separate declared input/scale/state/index/accumulator/output types.

    ``source`` is an OpSpec or a mapping (a fusion tensor view).  Only declared
    dtype/format fields are reported; a missing field stays missing rather
    than being inferred from the model's activation dtype.
    """
    metadata = dict(getattr(source, "metadata", source) or {})
    roles = {}

    def add(role, field, dtype, shape=None):
        if not isinstance(dtype, str) or not dtype:
            return
        record = {"field": field, "dtype": dtype}
        if shape:
            record["shape"] = list(shape)
        bucket = roles.setdefault(role, [])
        if record not in bucket:
            bucket.append(record)

    for key, value in metadata.items():
        if not isinstance(value, str) or not value or key == "dtype":
            continue
        lower = key.lower()
        if not (lower.endswith("dtype") or lower.endswith("format")):
            continue
        if "scale" in lower or "amax" in lower:
            role = "scale"
        elif "index" in lower or "topk" in lower:
            role = "index"
        elif "accum" in lower or "compute" in lower:
            role = "accumulator"
        elif "state" in lower or lower in {"cache_dtype", "kv_dtype", "kv_cache_dtype"}:
            role = "state"
        elif "gamma" in lower or "parameter" in lower or "coefficient" in lower or "bias" in lower:
            role = "parameter"
        elif "weight" in lower or "router" in lower:
            role = "weight"
        elif "input" in lower or "activation" in lower:
            role = "input"
        elif "output" in lower:
            role = "output"
        else:
            continue
        shape = None
        if key == "activation_scale_format":
            shape = metadata.get("activation_scale_shape")
        elif key in {"output_format", "dtype"} and hasattr(source, "local_shape"):
            if hasattr(source, "op_id"):
                produced = produced_values(source)
                if produced:
                    value = produced[0]["dtype"]
                    shape = produced[0]["logical_shape"]
                else:
                    shape = getattr(source, "local_shape", None)
            else:
                shape = getattr(source, "local_shape", None)
        add(role, key, value, shape)
    if hasattr(source, "dtype") and source.dtype:
        add("output", "output_format", metadata.get("output_format") or source.dtype,
            getattr(source, "local_shape", None))
    for name in getattr(source, "returns", ()) or ():
        if name == "activation_scales":
            add("scale", "returns.activation_scales",
                metadata.get("activation_scale_format") or "fp32",
                metadata.get("activation_scale_shape") or
                ((source.local_shape[0], 1) if getattr(source, "local_shape", ()) else None))
        elif name == "rstd":
            add("auxiliary", "returns.rstd", "fp32")
        elif name:
            produced = next((value for value in produced_values(source)
                             if value.get("output") == name), None)
            add("output", f"returns.{name}",
                (produced or {}).get("dtype") or metadata.get(f"{name}_dtype") or
                (metadata.get("output_format") or getattr(source, "dtype", "")),
                (produced or {}).get("logical_shape"))
    return {role: tuple(records) for role, records in roles.items()}


def merge_typed_roles(*sources):
    merged = {}
    for source in sources:
        for role, records in typed_roles(source).items():
            bucket = merged.setdefault(role, [])
            for record in records:
                if record not in bucket:
                    bucket.append(record)
    return {role: tuple(records) for role, records in merged.items()}


def _row_shape(metadata, shape):
    logical, storage = tuple(shape), tuple(shape)
    if logical:
        active = metadata.get("graph_active_query_rows")
        physical = metadata.get("graph_physical_query_rows")
        if (isinstance(active, int) and not isinstance(active, bool)
                and isinstance(physical, int) and not isinstance(physical, bool)
                and 0 <= active <= physical
                and shape[0] == physical
                and "expected_rows" not in metadata
                and "routing_basis" not in metadata
                and "packed_layout" not in metadata
                and not metadata.get("activation_input_segments")):
            logical, storage = (active,) + logical[1:], (physical,) + storage[1:]
    return logical, storage


def _primary_output_name(op, metadata):
    declared = metadata.get("primary_output")
    if isinstance(declared, str) and declared:
        return declared
    if op.recipe in {"dynamic_quantize", "rmsnorm_quantize",
                     "add_rmsnorm_quantize"} or op.family in {
            "quantize", "rmsnorm_quant", "add_rmsnorm_quant"}:
        return "payload"
    if op.family in {"rmsnorm", "add_rmsnorm", "layernorm", "normalization"}:
        return "norm"
    return "output"


def _row_geometry(op, metadata):
    shape = tuple(op.local_shape)
    if metadata.get("program_kind") == "matmul" and len(shape) >= 2:
        return shape[:2]
    return shape


def _json_identity(value, default="static"):
    if value is None:
        return default
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _declared_output_geometry(op, metadata):
    """Prefer actual declared output views over operator input/KV geometry."""
    for key in ("output_shape", "combine_output_shape", "public_output_shape"):
        shape = metadata.get(key)
        if isinstance(shape, (list, tuple)) and shape:
            return tuple(int(value) for value in shape), None, ""
    for key in ("output_view", "result_view", "public_view"):
        view = metadata.get(key)
        if isinstance(view, dict):
            layout = view.get("layout") if isinstance(view.get("layout"), dict) else None
            shape = view.get("shape") or (layout or {}).get("shape")
            if isinstance(shape, (list, tuple)) and shape:
                return (tuple(int(value) for value in shape), layout,
                        str(view.get("layout_name") or view.get("semantic_layout") or ""))
    for key in ("output_ports", "typed_ports"):
        ports = metadata.get(key)
        if isinstance(ports, (list, tuple)):
            for port in ports:
                if not isinstance(port, dict):
                    continue
                shape = port.get("shape") or port.get("logical_shape")
                if port.get("output") in (None, "output", "q") and shape:
                    layout = port.get("layout") if isinstance(port.get("layout"), dict) else None
                    return (tuple(int(value) for value in shape), layout,
                            str(port.get("layout_name")
                                or port.get("semantic_layout") or ""))
    attention = metadata.get("attention_tensor_view")
    if isinstance(attention, dict):
        shape = attention.get("output_shape") or attention.get("query_shape")
        if isinstance(shape, (list, tuple)) and shape:
            layout_name = str(attention.get("output_layout")
                              or attention.get("logical_layout") or "")
            return tuple(int(value) for value in shape), None, layout_name
    target = metadata.get("target_layout")
    if isinstance(target, dict) and target.get("shape"):
        return (tuple(int(value) for value in target["shape"]), target,
                str(metadata.get("target_layout_name") or ""))
    return None, None, ""


def _normalize_output(value, op, materialized, *, native=False):
    raw_layout = value.get("layout")
    layout_name = (value.get("layout_name") or value.get("semantic_layout")
                   or (raw_layout if isinstance(raw_layout, str) else ""))
    if not layout_name and isinstance(raw_layout, dict):
        for key in ("layout_name", "semantic_layout", "name", "layout"):
            if isinstance(raw_layout.get(key), str) and raw_layout[key]:
                layout_name = str(raw_layout[key])
                break
    layout = raw_layout if isinstance(raw_layout, (dict, TensorLayout)) else None
    logical = tuple(value.get("logical_shape") or value.get("shape") or ())
    storage = tuple(value.get("storage_shape") or logical)
    if isinstance(layout, dict):
        logical = tuple(layout.get("shape") or logical)
        storage = tuple(layout.get("storage_shape") or storage)
    dtype = value.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise ModelingError(f"typed_output_dtype_unresolved:{op.op_id}:{value.get('output')}")
    if native:
        producer = op.op_id
        output = value.get("port") or value.get("output") or "output"
        semantic_producer = value.get("member") or value.get("producer") or producer
        semantic_output = value.get("output") or output
    else:
        producer = value.get("producer") or op.op_id
        output = value.get("output") or "output"
        semantic_producer = value.get("semantic_producer") or producer
        semantic_output = value.get("semantic_output") or output
    return typed_value(
        producer=producer, dtype=dtype, shape=logical,
        storage_shape=storage, output=output,
        role=value.get("role") or "declared_output",
        source=value.get("source") or op.op_id, quant=value.get("quant") or "",
        materialized=(value.get("materialized", True) if materialized is None
                      else materialized),
        layout_kind=value.get("layout_kind") or "strided",
        storage=value.get("storage") or "", alias_of=value.get("alias_of") or "",
        state_version=_json_identity(value.get("state_version"), None),
        binding=_json_identity(value.get("binding"), "static"),
        binding_name=_json_identity(value.get("binding_name"), ""),
        write_bytes=value.get("write_bytes"),
        declared_bytes=value.get("declared_bytes"),
        contract=value.get("contract") or "",
        layout=layout,
        semantic_producer=semantic_producer,
        semantic_output=semantic_output,
        layout_name=layout_name,
        source_format=value.get("source_format") or "",
    )


def produced_values(op, *, materialized=None):
    """All actual values written by ``op``, with logical and storage shapes.

    Produced values are derived from the operator's real output contract:
    a matmul writes ``(M,N)`` even though its operand geometry is ``(M,N,K)``;
    quantized values carry the declared payload format, not the consumer's
    activation format; scales are separate values.
    """
    explicit = _audit_outputs(op)
    if explicit:
        return tuple(_normalize_output(value, op, materialized, native=native)
                     for value, native in explicit)
    metadata = dict(op.metadata)
    declared_shape, declared_layout, declared_layout_name = _declared_output_geometry(
        op, metadata)
    geometry = declared_shape or _row_geometry(op, metadata)
    if not geometry:
        return ()
    logical, storage = _row_shape(metadata, geometry)
    dtype = str(metadata.get("output_format") or metadata.get("output_dtype")
                or op.dtype)
    if not dtype:
        raise ModelingError(f"typed_output_primary_dtype_unresolved:{op.op_id}")
    name = _primary_output_name(op, metadata)
    quant = str(metadata.get("output_quant") or metadata.get("quantization") or "")
    layout_kind = str(metadata.get("output_layout_kind") or
                      ("strided" if op.layout in {"", "contiguous", "strided"} else op.layout))
    layout_name = str(declared_layout_name
                      or metadata.get("output_layout_name")
                      or metadata.get("semantic_layout")
                      or (metadata.get("output_layout")
                          if isinstance(metadata.get("output_layout"), str) else ""))
    try:
        payload_bytes = float(math.prod(logical) * dtype_bytes(dtype))
    except (KeyError, ValueError):
        payload_bytes = None
    declared_materialized = metadata.get("materialized")
    is_materialized = (bool(declared_materialized)
                       if declared_materialized is not None else True)
    if materialized is not None:
        is_materialized = materialized
    alias_of = str(metadata.get("alias_of") or "")
    write_bytes = metadata.get("write_bytes")
    if write_bytes is None and is_materialized:
        write_bytes = payload_bytes
    primary = typed_value(
        producer=op.op_id, dtype=dtype, shape=logical, storage_shape=storage,
        output=name, role="activation", source=f"{op.op_id}:output_format",
        quant=quant, materialized=is_materialized,
        layout_kind=layout_kind, write_bytes=write_bytes,
        declared_bytes=op.mte3_bytes, alias_of=alias_of,
        storage=str(metadata.get("storage") or ""),
        layout=(declared_layout or (metadata.get("output_layout")
                                    if isinstance(metadata.get("output_layout"), dict) else None)),
        layout_name=layout_name,
        source_format=str(metadata.get("source_format") or ""),
        contract="operator_output_contract")
    if metadata.get("source_outputs"):
        primary["source_outputs"] = tuple(metadata["source_outputs"])
    if metadata.get("source_layouts"):
        primary["source_layouts"] = tuple(metadata["source_layouts"])
    if op.family == "tensor_copy" and not (
            metadata.get("source_layout") or metadata.get("source_layouts")
            or metadata.get("target_layout") or metadata.get("output_layout")):
        primary["layout_gap"] = "missing_source_layout"
    outputs = [primary]
    for returned in op.returns:
        if returned == "activation_scales":
            dtype = str(metadata.get("activation_scale_format") or "fp32")
            shape = tuple(metadata.get("activation_scale_shape") or
                          ((logical[0], 1) if logical else ()))
            logical_scale, storage_scale = _row_shape(metadata, shape)
            outputs.append(typed_value(
                producer=op.op_id, dtype=dtype, shape=logical_scale,
                storage_shape=storage_scale, output=returned, role="scale",
                source=f"{op.op_id}:returns.{returned}",
                materialized=(True if materialized is None else materialized),
                write_bytes=metadata.get("activation_scale_bytes"),
                contract="declared_scale_value"))
        elif returned == "rstd":
            rows = logical[0] if logical else 0
            outputs.append(typed_value(
                producer=op.op_id, dtype="fp32", shape=(rows,),
                storage_shape=(storage[0],) if storage else (rows,),
                output=returned, role="auxiliary",
                source=f"{op.op_id}:returns.{returned}",
                materialized=True, contract="declared_auxiliary_value"))
        else:
            declared_shape = metadata.get(f"{returned}_shape")
            shape = tuple(declared_shape) if declared_shape else logical
            if returned == "residual":
                dtype = str(metadata.get("residual_dtype")
                            or metadata.get("input_format")
                            or metadata.get("activation_format") or op.dtype)
            else:
                dtype = str(metadata.get(f"{returned}_dtype")
                            or metadata.get("output_format") or op.dtype)
            outputs.append(typed_value(
                producer=op.op_id, dtype=dtype, shape=shape, storage_shape=shape,
                output=returned, role="declared_output",
                source=f"{op.op_id}:returns.{returned}",
                materialized=True,
                write_bytes=None,
                contract="declared_return_value"))
    return tuple(outputs)


def typed_outputs(op, *, materialized=True):
    """Compatibility wrapper for the produced value contract."""
    return produced_values(op, materialized=materialized)


def first_value(values):
    return values[0] if values else None
