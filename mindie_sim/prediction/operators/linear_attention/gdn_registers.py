"""Register work in the pinned arch35 Prepare vector functions.

[Role] Count source instructions in CumsumScaleVF and L2NormK128VF.
[Invariants] A dual-output shuffle is one call. Predicate writes do not consume
  unverified data-register ports. Scalar masks preserve their instruction calls.
[Timeouts & Locks] O(log(chunk)) construction, independent of total model tokens.
[Side-Effect Policy] Pure source inventories and physical work.
"""
from functools import lru_cache

from mindie_sim.schema.common import dtype_bytes
from .gdn_domain import GDN_CHUNK64_TOKENS, GDN_HEAD_DIM_K

from mindie_sim.prediction.instructions import InstructionClass, InstructionCount, InstructionInventory

REGISTER_FP32_LANES = 64
FP32_BYTES = 4
REGISTER_BYTES = REGISTER_FP32_LANES * FP32_BYTES
CUMSUM_TREE_STEPS = (1, 2, 4, 8)
CUMSUM_GROUP_LANES = REGISTER_FP32_LANES // 2
CUMSUM_MASK_LANES = CUMSUM_GROUP_LANES // 2
L2NORM_DTYPES = ("bf16", "fp16", "fp32")
CUMSUM_INVENTORY_CACHE_SIZE = GDN_CHUNK64_TOKENS + 1
L2NORM_INVENTORY_CACHE_SIZE = CUMSUM_INVENTORY_CACHE_SIZE * len(L2NORM_DTYPES)
L2NORM_MUL_CALLS = 4
L2NORM_STORE_CALLS = 3


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
@lru_cache(maxsize=CUMSUM_INVENTORY_CACHE_SIZE)
def cumsum(rows: int) -> InstructionInventory:
    if not 0 <= rows <= GDN_CHUNK64_TOKENS:
        raise ValueError("prepare_cumsum_requires_at_most_64_rows")
    if not rows:
        return InstructionInventory({})
    align = 1 << (rows - 1).bit_length()
    counts = {
        "CreateMask": InstructionCount(2),
        "Duplicate": InstructionCount(1, REGISTER_FP32_LANES),
        "Load": InstructionCount(1, REGISTER_FP32_LANES, read_bytes=REGISTER_BYTES),
    }

    def add(opcode, calls=1, lanes=0, read=0, write=0):
        counts[opcode] = counts.get(opcode, InstructionCount()) + InstructionCount(
            calls, lanes, read, write)

    # Each deeper tree expands the full 64-lane register again, irrespective
    # of the useful tail. Its final VL32 adds are not mathematical prefix FLOPs.
    for step in CUMSUM_TREE_STEPS:
        if align >= 2 * step:
            shuffles = 2 * step - 1
            add("DeInterleave", shuffles, shuffles * REGISTER_FP32_LANES)
            add("Add", step, step * CUMSUM_GROUP_LANES)
            add("Interleave", shuffles, shuffles * REGISTER_FP32_LANES)
    barriers = 0
    if align >= CUMSUM_GROUP_LANES:
        add("Store", lanes=REGISTER_FP32_LANES, write=REGISTER_BYTES)
        barriers += 1
        groups = align // CUMSUM_GROUP_LANES
        for group in range(groups):
            add("LoadBRC", lanes=REGISTER_FP32_LANES, read=FP32_BYTES)
            add("UpdateMask", 1 if align == REGISTER_FP32_LANES and group == groups - 1 else 2)
            add("XorMask")
            add("Add", lanes=CUMSUM_MASK_LANES)
    if align == REGISTER_FP32_LANES:
        add("Store", lanes=REGISTER_FP32_LANES, write=REGISTER_BYTES)
        barriers += 1
        add("LoadBRC", lanes=REGISTER_FP32_LANES, read=FP32_BYTES)
        add("XorMask")
        add("Add", lanes=CUMSUM_GROUP_LANES)
    add("UpdateMask")
    add("Muls", lanes=rows)
    add("Store", lanes=rows, write=FP32_BYTES * rows)
    return InstructionInventory(counts, barriers)


@lru_cache(maxsize=L2NORM_INVENTORY_CACHE_SIZE)
def l2norm(rows: int, dtype="bf16") -> InstructionInventory:
    if not 0 <= rows <= GDN_CHUNK64_TOKENS or dtype not in L2NORM_DTYPES:
        raise ValueError("prepare_l2norm_requires_64_rows_and_supported_dtype")
    width = int(dtype_bytes(dtype))
    per_row = {
        "Load": InstructionCount(2, GDN_HEAD_DIM_K, read_bytes=GDN_HEAD_DIM_K * width),
        "Mul": InstructionCount(L2NORM_MUL_CALLS, 2 * GDN_HEAD_DIM_K),
        "Add": InstructionCount(1, REGISTER_FP32_LANES),
        "ReduceSUM": InstructionCount(1, REGISTER_FP32_LANES),
        "Adds": InstructionCount(1, 1),
        "Sqrt": InstructionCount(1, 1),
        "Div": InstructionCount(1, 1),
        "Duplicate": InstructionCount(1, REGISTER_FP32_LANES),
        "Store": InstructionCount(L2NORM_STORE_CALLS, GDN_HEAD_DIM_K + 1,
                                  write_bytes=GDN_HEAD_DIM_K * width + FP32_BYTES),
    }
    if dtype != "fp32":
        per_row.update(CastIn=InstructionCount(2, GDN_HEAD_DIM_K),
                       CastOut=InstructionCount(2, GDN_HEAD_DIM_K))
    setup = InstructionInventory({
        "CreateMask": InstructionCount(2), "Duplicate": InstructionCount(1, 1)})
    return setup + InstructionInventory(per_row).scaled(rows)
