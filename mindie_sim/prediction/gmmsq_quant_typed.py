"""Typed U8 register inventory for the shipping GMMSQ MxA8W4 widen loop.

[Role] Exact gmmsq_quant_mx_vf.h 344-348 + 363-379 body.
[Invariants] One CreateMask and three Duplicates per 64-group VF callback;
  per packed 256B group one normalized VLD, 2 SHR + 1 SHL + 2 AND, one
  two-destination VINTLV, two VST.  U8 opcodes have no shared FP16/FP32 lane
  resource and no separate element-shuffle lane ceiling in this contract.
[Side-Effect Policy] Pure immutable inventory; physical Work comes from the
  common InstructionInventory.capacity_work.
"""
from dataclasses import dataclass
from .instructions import InstructionClass, InstructionCount, InstructionInventory


# Source calls.  Issue/data-write/UB units only; no generic lane resources.
CLASSES = {
    "CreateMask": InstructionClass(1, 0),
    "DuplicateU8": InstructionClass(1, 1),
    "LoadAlignU8": InstructionClass(1, 1, ub_read=True),
    "ShiftRightU8": InstructionClass(1, 1),
    "ShiftLeftU8": InstructionClass(1, 1),
    "AndU8": InstructionClass(1, 1),
    "InterleaveU8": InstructionClass(2, 2),
    "StoreAlignU8": InstructionClass(2, 0, ub_write=True),
}


@dataclass(frozen=True)
class Counts:
    groups: int
    vf_groups: int

    @property
    def loads(self):
        return self.groups

    @property
    def shift_right(self):
        return 2 * self.groups

    @property
    def shift_left(self):
        return self.groups

    @property
    def and_ops(self):
        return 2 * self.groups

    @property
    def simple_i8(self):
        return 5 * self.groups

    @property
    def interleave(self):
        return self.groups

    @property
    def stores(self):
        return 2 * self.groups

    @property
    def create_mask(self):
        return self.vf_groups

    @property
    def duplicate(self):
        return 3 * self.vf_groups

    @property
    def packed_bytes(self):
        return 256 * self.groups

    @property
    def expanded_bytes(self):
        return 512 * self.groups

    @property
    def raw_alu_lanes(self):
        return 5 * 256 * self.groups

    @property
    def raw_shuffle_lanes(self):
        return 256 * self.groups


def inventory(groups: int, vf_groups: int) -> tuple[InstructionInventory, Counts]:
    if type(groups) is not int or groups < 0:
        raise ValueError("gmmsq_groups_must_be_nonnegative_integer")
    if type(vf_groups) is not int or vf_groups < 0:
        raise ValueError("gmmsq_vf_groups_must_be_nonnegative_integer")
    c = Counts(groups, vf_groups)
    inv = InstructionInventory({
        "CreateMask": InstructionCount(c.create_mask),
        "DuplicateU8": InstructionCount(c.duplicate, c.duplicate * 256),
        "LoadAlignU8": InstructionCount(c.loads, c.loads * 256, read_bytes=c.packed_bytes),
        "ShiftRightU8": InstructionCount(c.shift_right, c.shift_right * 256),
        "ShiftLeftU8": InstructionCount(c.shift_left, c.shift_left * 256),
        "AndU8": InstructionCount(c.and_ops, c.and_ops * 256),
        "InterleaveU8": InstructionCount(c.interleave, c.interleave * 256),
        "StoreAlignU8": InstructionCount(c.stores, c.stores * 256,
                                         write_bytes=c.expanded_bytes),
    })
    return inv, c


def work(groups: int):
    # One scalar callback invocation per 64 packed groups in this source split.
    vf_groups = (groups + 63) // 64
    inv, c = inventory(groups, vf_groups)
    return inv.capacity_work(CLASSES), c, inv
