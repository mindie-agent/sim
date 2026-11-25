"""Explicit indexer basis conversion and low-precision tensor producers.

[Role] Materialize query/key values and scales consumed by index selection.
[Invariants] A low-precision cache or query has an actual conversion producer.
  The compressed-indexer source uses a dense Hadamard multiplication; source
  storage and native tensor packing are not inferred from captured allocations.
[Timeouts & Locks] Pure finite operator construction; no I/O or locks.
[Side-Effect Policy] Adds operators to a caller-owned builder.
"""
from math import ceil
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.catalog.cache_format import CacheFormat, cache_format


FP4_SCALE_VALUES_PER_BLOCK = 4
MXFP4_GROUP_SIZE = 32
INDEX_QUANTIZATION_OPS_PER_VALUE = 4


def _plan_format(quant, width):
    """Query/plain fallback: declared plan quant, not KV cache metadata."""
    dtype = str(getattr(quant, "index_dtype", "") or
                getattr(quant, "kv_dtype", "") or "bf16").strip().lower()
    scale = str(getattr(quant, "index_scale_dtype", "") or "none").strip().lower()
    if scale in {"", "none"}:
        return CacheFormat(dtype)
    return CacheFormat(dtype, int(width), scale)


def index_representation(b, name, value, rows, width, *, quant, model, rotate_basis):
    if rotate_basis:
        rotated = b.matrix(
            name + "_hadamard", (value,), rows, width, width, category="norm")
        value = b.vector(
            name + "_basis_scale", (rotated,), rows * width, 1,
            family="elementwise", shape=(rows, width),
            metadata=(("basis", "hadamard"), ("normalization", "inverse_sqrt_width")),
        )
    elements = rows * width
    declarations = dict(getattr(model, "metadata", ())).get(
        "cache_representations", {})
    if "index_k" in declarations:
        fmt = cache_format(model, "index_k")
    else:
        # Missing explicit source region restores the original plan logical
        # query/key representation; cache_format's kv fallback is not storage.
        fmt = _plan_format(quant, width)
    producer = next((op for op in reversed(getattr(b, "ops", ())) if op.op_id == value), None)
    producer_meta = dict(getattr(producer, "metadata", ()) or ()) if producer is not None else {}
    input_dtype = str(producer_meta.get("output_format") or model.activation_dtype)
    if (fmt.dtype == input_dtype and fmt.packed_elements == 1
            and not fmt.group_size):
        return value
    # Compute may remain FP8/BF16/MXFP4 while the backing cache row is packed
    # in a byte container (e.g. uint8). Keep logical compute dtype distinct from
    # storage_payload_dtype while preserving physical row bytes.
    if fmt.packed_elements > 1:
        compute_dtype = (
            "mxfp4" if fmt.packed_elements == 2 and fmt.group_size == MXFP4_GROUP_SIZE
            else _plan_format(quant, width).dtype
        )
    else:
        compute_dtype = fmt.dtype
    scale_count = ceil(width / fmt.group_size) if fmt.group_size else 0
    native_scale_shape = ()
    if scale_count:
        native_scale_shape = (rows, scale_count)
        if fmt.packed_elements == 2 and scale_count == FP4_SCALE_VALUES_PER_BLOCK:
            # MXFP4 indexer ABI: scales are [.., 2, 2], not [.., 32].
            native_scale_shape = (rows, 2, 2)
    return b.vector(
        name + "_quantize", (value,), elements, INDEX_QUANTIZATION_OPS_PER_VALUE,
        family="quantize", dtype=compute_dtype, shape=(rows, width),
        read_bytes_override=elements * dtype_bytes(input_dtype),
        write_bytes_override=rows * fmt.row_bytes(width),
        returns=("index_scales",) if fmt.group_size else (),
        metadata=(("input_format", input_dtype),
                  ("output_format", compute_dtype),
                  ("storage_payload_dtype", fmt.dtype),
                  ("scale_format", fmt.scale_dtype if fmt.group_size else "none"),
                  ("scale_count", scale_count),
                  ("scale_shape", (rows, scale_count) if scale_count else ()),
                  ("native_scale_shape", native_scale_shape),
                  ("packed_elements", fmt.packed_elements),
                  ("physical_row_bytes", fmt.row_bytes(width)),
                  ("quantization_axis", width)),
    )
