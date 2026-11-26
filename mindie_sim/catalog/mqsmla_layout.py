"""Physical KV record fields for arch35 MQSMLA.
[Role] Decode the simulator's window/compressed representation pair.
[Invariants] Data, numerical scales and alignment exhaust the native row.
[Timeouts & Locks] Constant work, no locks.
[Side-Effect Policy] Read-only.
"""
from .cache_format import cache_format

HEAD_WIDTH = 512
ROPE_WIDTH = 64
NATIVE_GROUP = 64
V0919_MODE_TAG = 32
MODE_PADDING = {"bf16": 18, "uint8": 1}
EXPONENT_SCALE_TYPES = frozenset({"uint8", "fp8_e8m0", "e8m0"})


def packed_scale_type(window, compressed):
    if (window != compressed or window.packed_elements != 1 or window.dtype != "fp8_e4m3"
            or window.group_size not in (V0919_MODE_TAG, NATIVE_GROUP)
            or window.suffix_elements or window.alignment != 1):
        return None
    return "uint8" if window.scale_dtype in EXPONENT_SCALE_TYPES else (
        "bf16" if window.scale_dtype == "bf16" else None)


def packed_row_parts(model, region, width, *, plan=None, phase=None):
    if region not in {"window", "compressed_kv"} or width != HEAD_WIDTH:
        return None
    fmt = cache_format(model, "window", plan=plan, phase=phase)
    kind = packed_scale_type(fmt, cache_format(model, "compressed_kv", plan=plan, phase=phase))
    if kind is None:
        return None
    from mindie_sim.schema.common import dtype_bytes
    nope = HEAD_WIDTH - ROPE_WIDTH
    data = nope + ROPE_WIDTH * int(dtype_bytes("bf16"))
    scales = nope // NATIVE_GROUP * int(dtype_bytes(fmt.scale_dtype))
    return data, scales, MODE_PADDING[kind]
