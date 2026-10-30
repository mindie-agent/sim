"""Register inventories for the native arch35 convolution rolling branch.

[Role] Count RestoreFnLocalPartialsRegbase and ComputeFnRollingTokenRegbase.
[Invariants] Unmasked loads retain full register payload on channel tails.
  Calls, active lanes and UB bytes are separate quantities. Exp/Div use the
  declared CANN default FTZ intrinsic path. Capacity projection is not a
  reconstruction of the packaged ELF's issue bundles or optimized schedule.
[Timeouts & Locks] Bounded source-block inventories; no token-sized history.
[Side-Effect Policy] Pure arithmetic; no compiler, observations or device I/O.
"""
from functools import lru_cache

from ..instructions import InstructionClass, InstructionCount, InstructionInventory


CLASSES = {
    "UpdateMask": InstructionClass(1),
    "LoadB16": InstructionClass(1, 1, ub_read=True),
    "LoadB32": InstructionClass(1, 1, ub_read=True),
    "StoreB16": InstructionClass(2, ub_write=True),
    "StoreB32": InstructionClass(2, ub_write=True),
    "CastIn": InstructionClass(1, 1, "reg_alu_lanes"),
    "CastOut": InstructionClass(1, 1, "reg_alu_lanes"),
    "Mul": InstructionClass(1, 1, "reg_alu_lanes"),
    "Add": InstructionClass(1, 1, "reg_alu_lanes"),
    "Muls": InstructionClass(1, 1, "reg_alu_lanes"),
    "Adds": InstructionClass(1, 1, "reg_alu_lanes"),
    "Exp": InstructionClass(1, 1, "reg_special_lanes"),
    "Div": InstructionClass(1, 1, "reg_special_lanes"),
}


def _count(calls_per_vector, channels, *, load_bytes=0, store_bytes=0):
    vectors = (channels + 63) // 64
    calls = calls_per_vector * vectors
    # LoadAlign has no mask. StoreAlign and arithmetic use pregLoop.
    lanes = calls * 64 if load_bytes else calls_per_vector * channels
    return InstructionCount(calls, lanes, calls * load_bytes, lanes * store_bytes)


@lru_cache(maxsize=128)
def restore(width: int, channels: int) -> InstructionInventory:
    history = width - 1
    products = history * (history + 1) // 2
    additions = products - history
    return InstructionInventory({
        "UpdateMask": InstructionCount((channels + 63) // 64),
        "LoadB32": _count(history, channels, load_bytes=256),
        "LoadB16": _count(history, channels, load_bytes=128),
        "CastIn": _count(history, channels),
        "Mul": _count(products, channels),
        "Add": _count(additions, channels),
        "StoreB32": _count(history, channels, store_bytes=4),
    })


@lru_cache(maxsize=256)
def rolling(width: int, channels: int, activation: str) -> InstructionInventory:
    counts = {
        "UpdateMask": InstructionCount((channels + 63) // 64),
        "LoadB16": _count(1, channels, load_bytes=128),
        "LoadB32": _count(2 * width - 1, channels, load_bytes=256),
        "CastIn": _count(1, channels),
        "Mul": _count(width, channels),
        "Add": _count(width - 1, channels),
        "StoreB32": _count(width - 1, channels, store_bytes=4),
        "CastOut": _count(1, channels),
        "StoreB16": _count(1, channels, store_bytes=2),
    }
    if activation == "silu":
        counts.update({name: _count(1, channels) for name in ("Muls", "Exp", "Adds", "Div")})
    return InstructionInventory(counts, barriers=1)
