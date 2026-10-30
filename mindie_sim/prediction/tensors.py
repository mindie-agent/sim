"""Tensor views and explicit consumer materialization.

[Role] Translate layout requirements into real transfers in the operator DAG.
[Invariants] A view is free only when its address map can be retained.
  A copy reads the selected payload, not the full backing allocation.
  Consumer contracts and dtype determine work, never measured durations.
[Timeouts & Locks] O(active tensor descriptors), no locks or device I/O.
[Side-Effect Policy] Appends operators to the caller's Builder only.
"""
from dataclasses import replace

from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.tensors import TensorLayout, TensorValue


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
        elements = sum(value.layout.elements for value in values)
        dtype = values[0].dtype
        volume = elements * dtype_bytes(dtype)
        producer = self.builder.work(
            name, tuple(dict.fromkeys(value.producer for value in values)), (elements,),
            read=volume, write=volume, family="tensor_copy", dtype=dtype,
            metadata=(("materialization_reason", "concatenate"),
                      ("source_outputs", tuple(value.output for value in values)),
                      ("source_layouts", tuple(value.layout.to_dict() for value in values))),
        )
        return self.result(producer, (elements,), dtype)

    def reshape(self, value, shape, *, name):
        view = value.layout.reshape_view(shape)
        if view is None:
            value = self.materialize(value, name, reason="reshape_crosses_storage_gap")
            view = value.layout.reshape_view(shape)
        return replace(value, layout=view)
