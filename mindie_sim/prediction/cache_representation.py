"""Analytic cache conversion producers with explicit packed output formats.

[Role] Make cache allocation, updates and attention consume the same row ABI.
[Invariants] BF16 input bytes are not relabelled as packed output bytes. Scale
  reductions and casts are explicit work; this is not an exact vendor kernel.
[Timeouts & Locks] Constant row arithmetic, no I/O or locks.
[Side-Effect Policy] Adds immutable graph operators to a local builder.
"""
from math import ceil

from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.schema.common import dtype_bytes


def store(b, name, deps, rows, width, region):
    fmt = cache_format(b.model, region)
    if not fmt.group_size:
        return b.store(name, deps, rows, fmt.row_bytes(width), fmt.dtype)
    groups = ceil(width / fmt.group_size)
    # Abs/reduce max, normalize and cast; scale reciprocal is once per group.
    operations = rows * (4 * width + groups)
    return b.work(
        name, deps, (rows, width), family="cache_store", dtype=b.model.activation_dtype,
        vector=operations, read=rows * width * dtype_bytes(b.model.activation_dtype),
        write=rows * fmt.row_bytes(width), metadata=(("cache_kind", name), ("cache_visible", True),
        ("shared_owner", False), ("cache_representation", {
            "region": region, "dtype": fmt.dtype, "group_size": fmt.group_size,
            "scale_dtype": fmt.scale_dtype, "row_bytes": fmt.row_bytes(width),
            "producer": "analytic_group_quantization_and_cache_write",
            "source_kernel_identity": "not_established",
        }),))
