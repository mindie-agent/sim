"""Register work in the pinned arch35 Prepare vector functions.

[Role] Count source instructions in CumsumScaleVF and L2NormK128VF.
[Invariants] A dual-output shuffle is one call. Predicate writes do not consume
  unverified data-register ports. Scalar masks preserve their instruction calls.
[Timeouts & Locks] O(log(chunk)) construction, independent of total model tokens.
[Side-Effect Policy] Pure source inventories and physical work.
"""
from functools import lru_cache

from ..instructions import InstructionClass, InstructionCount, InstructionInventory


# Pinned Reg ISA tables distinguish two issue slots, two data Reg writes and
# two UB accesses (at most one store) per cycle. These are capacity constraints.
CLASSES = {
    "CreateMask": InstructionClass(1),
    "UpdateMask": InstructionClass(1),
    "XorMask": InstructionClass(1),
    "Duplicate": InstructionClass(1, 1, "reg_alu_lanes"),
    "Add": InstructionClass(1, 1, "reg_alu_lanes"),
    "Adds": InstructionClass(1, 1, "reg_alu_lanes"),
    "Mul": InstructionClass(1, 1, "reg_alu_lanes"),
    "Muls": InstructionClass(1, 1, "reg_alu_lanes"),
    "Sqrt": InstructionClass(1, 1, "reg_special_lanes"),
    "Div": InstructionClass(1, 1, "reg_special_lanes"),
    "CastIn": InstructionClass(1, 1, "reg_alu_lanes"),
    "CastOut": InstructionClass(1, 1, "reg_alu_lanes"),
    "ReduceSUM": InstructionClass(2, 1, "reg_reduce_lanes"),
    "Interleave": InstructionClass(2, 2, "reg_shuffle_lanes"),
    "DeInterleave": InstructionClass(2, 2, "reg_shuffle_lanes"),
    "Load": InstructionClass(1, 1, ub_read=True),
    "LoadBRC": InstructionClass(1, 1, ub_read=True),
    "Store": InstructionClass(2, ub_write=True),
}
@lru_cache(maxsize=65)
def cumsum(rows: int) -> InstructionInventory:
    if not 0 <= rows <= 64:
        raise ValueError("prepare_cumsum_requires_at_most_64_rows")
    if not rows:
        return InstructionInventory({})
    align = 1 << (rows - 1).bit_length()
    counts = {
        "CreateMask": InstructionCount(2),
        "Duplicate": InstructionCount(1, 64),
        "Load": InstructionCount(1, 64, read_bytes=256),
    }

    def add(opcode, calls=1, lanes=0, read=0, write=0):
        counts[opcode] = counts.get(opcode, InstructionCount()) + InstructionCount(
            calls, lanes, read, write)

    # Each deeper tree expands the full 64-lane register again, irrespective
    # of the useful tail. Its final VL32 adds are not mathematical prefix FLOPs.
    for step in (1, 2, 4, 8):
        if align >= 2 * step:
            shuffles = 2 * step - 1
            add("DeInterleave", shuffles, shuffles * 64)
            add("Add", step, step * 32)
            add("Interleave", shuffles, shuffles * 64)
    barriers = 0
    if align >= 32:
        add("Store", lanes=64, write=256)
        barriers += 1
        groups = align // 32
        for group in range(groups):
            add("LoadBRC", lanes=64, read=4)
            add("UpdateMask", 1 if align == 64 and group == groups - 1 else 2)
            add("XorMask")
            add("Add", lanes=16)
    if align == 64:
        add("Store", lanes=64, write=256)
        barriers += 1
        add("LoadBRC", lanes=64, read=4)
        add("XorMask")
        add("Add", lanes=32)
    add("UpdateMask")
    add("Muls", lanes=rows)
    add("Store", lanes=rows, write=4 * rows)
    return InstructionInventory(counts, barriers)


@lru_cache(maxsize=195)
def l2norm(rows: int, dtype="bf16") -> InstructionInventory:
    if not 0 <= rows <= 64 or dtype not in ("bf16", "fp16", "fp32"):
        raise ValueError("prepare_l2norm_requires_64_rows_and_supported_dtype")
    width = 4 if dtype == "fp32" else 2
    per_row = {
        "Load": InstructionCount(2, 128, read_bytes=128 * width),
        "Mul": InstructionCount(4, 256),
        "Add": InstructionCount(1, 64),
        "ReduceSUM": InstructionCount(1, 64),
        "Adds": InstructionCount(1, 1),
        "Sqrt": InstructionCount(1, 1),
        "Div": InstructionCount(1, 1),
        "Duplicate": InstructionCount(1, 64),
        "Store": InstructionCount(3, 129, write_bytes=128 * width + 4),
    }
    if dtype != "fp32":
        per_row.update(CastIn=InstructionCount(2, 128),
                       CastOut=InstructionCount(2, 128))
    setup = InstructionInventory({
        "CreateMask": InstructionCount(2), "Duplicate": InstructionCount(1, 1)})
    return setup + InstructionInventory(per_row).scaled(rows)
