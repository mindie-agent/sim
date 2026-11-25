"""Tensor views and explicit consumer materialization.

[Role] Translate layout requirements into real transfers in the operator DAG.
[Invariants] A view is free only when its address map can be retained.
  A copy reads the selected payload, not the full backing allocation.
  Consumer contracts and dtype determine work, never measured durations.
[Timeouts & Locks] O(active tensor descriptors), no locks or device I/O.
[Side-Effect Policy] Appends operators to the caller's Builder only.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.tensors import TensorLayout, TensorValue

_DTYPE_CANONICAL = {
    "float32": "fp32",
    "bfloat16": "bf16",
    "float16": "fp16",
    "hif8": "hifloat8",
    "e8m0": "fp8_e8m0",
    "mx_fp8": "mxfp8",
    "mxfp8_e4m3": "mxfp8",
    "mx_fp4": "mxfp4",
    "mxfp4_e2m1": "mxfp4",
}
_INT_RANK = {
    "int4": (True, 4),
    "w4a8": (True, 4),
    "int8": (True, 8),
    "w8a8": (True, 8),
    "uint8": (False, 8),
    "int32": (True, 32),
    "int64": (True, 64),
}
_INT_BY_SIGNED_WIDTH = {4: "int4", 8: "int8", 32: "int32", 64: "int64"}
_NON_PROMOTING_DTYPES = frozenset({
    "fp4", "fp4_e2m1", "mxfp4",
    "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_e8m0", "hifloat8", "mxfp8",
})
_FLOAT_BITS = {
    "fp16": (5, 10),
    "bf16": (8, 7),
    "fp32": (8, 23),
}
_STANDARD_FLOATS = (
    ("fp16", _FLOAT_BITS["fp16"]),
    ("bf16", _FLOAT_BITS["bf16"]),
    ("fp32", _FLOAT_BITS["fp32"]),
)
_SIGNED_INT_FOR_MIXED_UINT8 = 32


def _canonical_dtype(dtype: str) -> str:
    low = str(dtype).strip().lower()
    return _DTYPE_CANONICAL.get(low, low)


def _promote_dtypes(dtypes: tuple[str, ...]) -> str:
    """Symmetric PyTorch/CANN ``result_type`` promotion across operand dtypes."""
    if not dtypes:
        raise ModelingError("concatenate_flat_requires_values")
    for raw in dtypes:
        dtype_bytes(raw)
    canonical = tuple(_canonical_dtype(raw) for raw in dtypes)
    unique = set(canonical)
    if len(unique) == 1:
        return dtypes[0] if len(set(dtypes)) == 1 else canonical[0]
    if unique & _NON_PROMOTING_DTYPES:
        raise ModelingError(
            "concatenate_flat_unsupported_dtype_promotion:"
            + ",".join(sorted(unique))
        )
    floats = [dt for dt in unique if dt in _FLOAT_BITS]
    if floats:
        if len(floats) == 1:
            return floats[0]
        req_exp = max(_FLOAT_BITS[dt][0] for dt in floats)
        req_man = max(_FLOAT_BITS[dt][1] for dt in floats)
        for candidate, (exp_b, man_b) in _STANDARD_FLOATS:
            if exp_b >= req_exp and man_b >= req_man:
                return candidate
        return "fp32"
    ints = [dt for dt in unique if dt in _INT_RANK]
    if ints:
        if len(ints) == 1:
            return ints[0]
        has_signed = any(_INT_RANK[dt][0] for dt in ints)
        has_unsigned = any(not _INT_RANK[dt][0] for dt in ints)
        max_width = max(_INT_RANK[dt][1] for dt in ints)
        if has_signed and has_unsigned and max_width < _SIGNED_INT_FOR_MIXED_UINT8:
            # PyTorch promotes int8 + uint8 -> int16, which is outside
            # schema/common.py:_DTYPE_BYTES; reject rather than fabricate int32.
            raise ModelingError(
                "concatenate_flat_unsupported_dtype_promotion:"
                + ",".join(sorted(unique))
            )
        return _INT_BY_SIGNED_WIDTH[max_width]
    return "bool"


class TensorLowering:
    def __init__(self, builder, contracts=None):
        self.builder = builder
        self.contracts = contracts or {}
        self.copies = {}

    def result(self, producer, shape, dtype=None, *, output=0):
        return TensorValue(producer, dtype or self.builder.model.activation_dtype,
                           TensorLayout.contiguous(shape), output)

    def contiguous(self, value, name, *, reason, dtype=None):
        dtype = dtype or value.dtype
        if value.layout.is_contiguous and dtype == value.dtype:
            return value
        return self.materialize(value, name, reason=reason, dtype=dtype)

    def materialize(self, value, name, *, reason, dtype=None):
        dtype = dtype or value.dtype
        signature = value, dtype
        if signature in self.copies:
            return self.copies[signature]
        layout = value.layout
        read = layout.elements * dtype_bytes(value.dtype)
        write = layout.elements * dtype_bytes(dtype)
        producer = self.builder.work(
            name, (value.producer,), layout.shape, read=read, write=write,
            vector=layout.elements if dtype != value.dtype else 0,
            family="tensor_copy", dtype=dtype,
            metadata=(("source_layout", layout.to_dict()),
                      ("target_layout", TensorLayout.contiguous(layout.shape).to_dict()),
                      ("input_format", value.dtype), ("output_format", dtype),
                      ("source_output", value.output),
                      ("materialization_reason", reason)),
        )
        result = self.result(producer, layout.shape, dtype)
        self.copies[signature] = result
        return result

    def require(self, value, consumer, argument, *, name):
        contract = self.contracts.get(consumer, {}).get(argument, "strided")
        if contract == "strided" or contract == "contiguous" and value.layout.is_contiguous:
            return value
        if contract != "contiguous":
            raise ValueError(f"unknown_tensor_layout_contract:{consumer}:{argument}:{contract}")
        return self.materialize(value, name, reason=f"{consumer}.{argument}:contiguous")

    def copy_into(self, value, destination, name):
        """A caller-owned output is a real write even if the input is contiguous."""
        read = value.layout.elements * dtype_bytes(value.dtype)
        write = value.layout.elements * dtype_bytes(destination.dtype)
        producer = self.builder.work(
            name, tuple(dict.fromkeys(producer for producer in
                                     (value.producer, destination.producer) if producer)),
            value.layout.shape, read=read, write=write,
            vector=value.layout.elements if value.dtype != destination.dtype else 0,
            family="tensor_copy", dtype=destination.dtype,
            metadata=(("source_layout", value.layout.to_dict()),
                      ("target_layout", destination.layout.to_dict()),
                      ("input_format", value.dtype), ("output_format", destination.dtype),
                      ("source_output", value.output), ("target_output", destination.output),
                      ("materialization_reason", "caller_owned_output")),
        )
        return replace(destination, producer=producer)

    def concatenate_flat(self, values, name):
        """Materialized one-dimensional concatenation; input views retain offsets."""
        items = tuple(values)
        dtype = _promote_dtypes(tuple(value.dtype for value in items))
        target_canon = _canonical_dtype(dtype)
        elements = sum(value.layout.elements for value in items)
        read = sum(value.layout.elements * dtype_bytes(value.dtype) for value in items)
        write = elements * dtype_bytes(dtype)
        cast_elems = sum(
            value.layout.elements for value in items
            if _canonical_dtype(value.dtype) != target_canon
        )
        producer = self.builder.work(
            name, tuple(dict.fromkeys(value.producer for value in items)), (elements,),
            read=read, write=write, vector=cast_elems, family="tensor_copy", dtype=dtype,
            metadata=(("materialization_reason", "concatenate"),
                      ("source_outputs", tuple(value.output for value in items)),
                      ("source_layouts", tuple(value.layout.to_dict() for value in items))),
        )
        return self.result(producer, (elements,), dtype)

    def reshape(self, value, shape, *, name):
        view = value.layout.reshape_view(shape)
        if view is None:
            value = self.materialize(value, name, reason="reshape_crosses_storage_gap")
            view = value.layout.reshape_view(shape)
        return replace(value, layout=view)
