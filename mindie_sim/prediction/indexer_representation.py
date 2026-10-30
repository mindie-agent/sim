"""Explicit indexer basis conversion and low-precision tensor producers.

[Role] Materialize query/key values and scales consumed by index selection.
[Invariants] A low-precision cache or query has an actual conversion producer.
  The compressed-indexer source uses a dense Hadamard multiplication; source
  storage and native tensor packing are not inferred from captured allocations.
[Timeouts & Locks] Pure finite operator construction; no I/O or locks.
[Side-Effect Policy] Adds operators to a caller-owned builder.
"""
from mindie_sim.schema.common import dtype_bytes


def index_representation(b, name, value, rows, width, *, quant, model, rotate_basis):
    if rotate_basis:
        rotated = b.matrix(
            name + "_hadamard", (value,), rows, width, width, category="norm")
        value = b.vector(
            name + "_basis_scale", (rotated,), rows * width, 1,
            family="elementwise", shape=(rows, width),
            metadata=(("basis", "hadamard"), ("normalization", "inverse_sqrt_width")),
        )
    if quant.index_dtype == model.activation_dtype:
        return value
    elements = rows * width
    return b.vector(
        name + "_quantize", (value,), elements, 4,
        family="quantize", shape=(rows, width),
        read_bytes_override=elements * dtype_bytes(model.activation_dtype),
        write_bytes_override=elements * dtype_bytes(quant.index_dtype)
                             + rows * dtype_bytes(quant.index_scale_dtype),
        returns=("index_scales",),
        metadata=(("output_format", quant.index_dtype),
                  ("scale_format", quant.index_scale_dtype),
                  ("scale_shape", (rows,)), ("quantization_axis", width)),
    )
