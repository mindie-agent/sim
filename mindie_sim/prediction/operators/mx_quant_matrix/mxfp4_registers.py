"""Typed source register inventories for MXA8W4 weight expansion and scales.

[Role] Project retained 950 Reg instruction counts onto existing capacities.
[Invariants] B8 arithmetic is retained as native source counts and per-opcode
  throughput bounds; no unsupported single-port aggregation is charged. Issue,
  data writes and UB ports are independent bounds; UB stores are not GM/MTE3
  writes.
[Timeouts & Locks] Constant-sized inventories for a finite source panel.
[Side-Effect Policy] Pure inventories; no fitting or compiler/device access.
"""
from mindie_sim.prediction.instructions import InstructionClass, InstructionCount, InstructionInventory

B8_WIDEN_BLOCK_ELEMENTS = 256
B8_BITWISE_OPS_PER_ELEMENT = 5
WIDEN_PROLOGUE_DUPLICATE_CALLS = 3
B16_TRANSPOSE_BLOCK_LANES = 128
TRANSPOSE_INDEX_BYTES = B16_TRANSPOSE_BLOCK_LANES * 2

TABLE_ROOT = "data/observations/sources/arch3510-physical-services/register-evidence/platform-units/"
TABLES = (TABLE_ROOT + "reg_vector_compute_interface_theoretical_perf_summary.md",
          TABLE_ROOT + "reg_vector_compute_interface_instruction_single_dual_issue_summary.md")

CLASSES = {
    "Mask": InstructionClass(1),
    "DuplicateB8": InstructionClass(1, 1),
    "LoadUS_B8": InstructionClass(1, 1, ub_read=True),
    "ShiftRightB8": InstructionClass(1, 1),
    "ShiftLeftB8": InstructionClass(1, 1),
    "SelectB8": InstructionClass(1, 1),
    "AndB8": InstructionClass(1, 1),
    "StoreB8": InstructionClass(2, ub_write=True),
    "LoadIndexB16": InstructionClass(1, 1, ub_read=True),
    # Table 1 UB Gather B16 runs 64 elements/cycle; table 13 register gather
    # is a different operation. Existing shuffle capacity has this exact rate.
    "GatherUB_B16": InstructionClass(2, 1, "reg_shuffle_lanes", ub_read=True),
    "StoreB16": InstructionClass(2, ub_write=True),
}


def widen(elements):
    blocks = elements // B8_WIDEN_BLOCK_ELEMENTS
    assert elements % B8_WIDEN_BLOCK_ELEMENTS == 0
    # Exact WeightNz MX vector block: one-time per invocation 2 CreateMask
    # and 3 Duplicate (L373-381), then for each 256 expanded bytes one
    # DIST_US_B8 packed read128B, 2x ShiftRight.I8, 1x ShiftLeft.I8,
    # 1x Select, 1x And, 1x Store256B.
    return InstructionInventory({
        "Mask": InstructionCount(2),
        "DuplicateB8": InstructionCount(
            WIDEN_PROLOGUE_DUPLICATE_CALLS,
            WIDEN_PROLOGUE_DUPLICATE_CALLS * B8_WIDEN_BLOCK_ELEMENTS,
        ),
        "LoadUS_B8": InstructionCount(blocks, elements, read_bytes=elements // 2),
        "ShiftRightB8": InstructionCount(2 * blocks, 2 * elements),
        "ShiftLeftB8": InstructionCount(blocks, elements),
        "SelectB8": InstructionCount(blocks, elements),
        "AndB8": InstructionCount(blocks, elements),
        "StoreB8": InstructionCount(blocks, elements, write_bytes=elements),
    })


def transpose(lanes):
    if not lanes:
        return InstructionInventory({})
    blocks = lanes // B16_TRANSPOSE_BLOCK_LANES
    assert lanes % B16_TRANSPOSE_BLOCK_LANES == 0
    return InstructionInventory({
        "Mask": InstructionCount(1),
        "LoadIndexB16": InstructionCount(
            1, B16_TRANSPOSE_BLOCK_LANES, read_bytes=TRANSPOSE_INDEX_BYTES,
        ),
        "GatherUB_B16": InstructionCount(blocks, lanes, read_bytes=2 * lanes),
        "StoreB16": InstructionCount(blocks, lanes, write_bytes=2 * lanes),
    })
