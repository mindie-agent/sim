"""Typed source register inventories for MXA8W4 weight expansion and scales.

[Role] Project retained 950 Reg instruction counts onto existing capacities.
[Invariants] B8 lanes are not charged at a FP32 arithmetic rate. Issue, data
  writes and UB ports are independent bounds; UB stores are not GM/MTE3 writes.
[Timeouts & Locks] Constant-sized inventories for a finite source panel.
[Side-Effect Policy] Pure inventories; no fitting or compiler/device access.
"""
from .instructions import InstructionClass, InstructionCount, InstructionInventory

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
    blocks = elements // 256
    assert elements % 256 == 0
    return InstructionInventory({
        "Mask": InstructionCount(2),
        "DuplicateB8": InstructionCount(3, 3 * 256),
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
    blocks = lanes // 128
    assert lanes % 128 == 0
    return InstructionInventory({
        "Mask": InstructionCount(1), "LoadIndexB16": InstructionCount(1, 128, read_bytes=256),
        "GatherUB_B16": InstructionCount(blocks, lanes, read_bytes=2 * lanes),
        "StoreB16": InstructionCount(blocks, lanes, write_bytes=2 * lanes),
    })
