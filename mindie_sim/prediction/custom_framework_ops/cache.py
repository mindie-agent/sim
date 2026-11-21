"""xLLM paged KV cache writes.

[Role] Plan xLLM paged KV cache writes.
[Invariants] The slot index read precedes masked writes and publication.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

from math import prod
from typing import Any

from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.operators import OpSpec

from ..mechanism import OperatorPlan, Work
from ..program import KernelProgram, Stage
from .contracts import _shape

_DEFAULT_KV_PAGE_SIZE = 128
_SLOT_INDEX_BYTES = int(dtype_bytes("int32"))


def _cache_row_bytes(op: OpSpec, meta: dict[str, Any], dtype: str) -> tuple[int, str]:
    """Typed paged-cache row size.

    ``cache_store`` declares a byte row; the vendor semantic shape declares
    elements.  When the source view is present, K/V shapes and their own dtypes
    are authoritative and mixed K/V dtypes are not collapsed.
    """
    view = dict(meta.get("cache_tensor_view") or {})
    if view:
        total = 0
        for shape_key, dtype_key in (("key_shape", "key_dtype"),
                                     ("value_shape", "value_dtype")):
            shape = view.get(shape_key)
            if isinstance(shape, (tuple, list)) and len(shape) >= 2:
                total += prod(int(value) for value in shape[1:]) * dtype_bytes(
                    view.get(dtype_key) or dtype)
        if total > 0:
            return total, "cache_tensor_view_key_value_bytes"
    shape = _shape(op)
    if op.family == "cache_store" or meta.get("cache_kind"):
        return shape[1], "cache_store_declared_row_bytes"
    return prod(shape[1:]) * dtype_bytes(dtype), "vendor_shape_elements_times_dtype"


def _build_cache_a5(op: OpSpec, geometry):
    meta = dict(op.metadata)
    index_bytes = float(meta.get("cache_slot_index_read_bytes", 0))
    read_bytes = float(op.mte2_bytes)
    write_bytes = float(op.mte3_bytes)
    ready = bool(meta.get("cache_update_ready", False))
    total = Work(resources={"hbm": read_bytes + write_bytes,
                            "mte2": index_bytes + (read_bytes - index_bytes),
                            "mte3": write_bytes})
    def program(context):
        stages = [Stage("slot_external_ready", {}),
                  Stage("slot_index_read", context.bind(Work(resources={
                      "hbm": index_bytes, "mte2": index_bytes})),
                        after=("slot_external_ready",))]
        if write_bytes > 0:
            stages.append(Stage("masked_cache_write", context.bind(Work(
                resources={"hbm": write_bytes, "mte3": write_bytes})),
                after=("slot_index_read",)))
            stages.append(Stage("cache_publish", {}, after=("masked_cache_write",)))
        return KernelProgram(tuple(stages))
    return OperatorPlan(total, program, metadata={
        "cache_slot_mapping_known": ready,
        "cache_index_read_bytes": int(index_bytes),
        "cache_write_bytes": int(write_bytes),
        "cache_conditional_write_upper_bound": meta.get(
            "cache_conditional_write_upper_bound"),
        "cache_execution": "slot_index_read_then_masked_write_and_publish",
    })
