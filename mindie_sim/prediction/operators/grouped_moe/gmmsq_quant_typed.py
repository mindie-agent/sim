"""Typed U8 register inventory for the shipping GMMSQ MxA8W4 widen loop.

[Role] Exact gmmsq_quant_mx_vf.h 344-348 + 363-379 body.
[Invariants] One CreateMask and three Duplicates per 64-group VF callback;
  per packed 256B group one normalized VLD, 2 SHR + 1 SHL + 2 AND, one
  two-destination VINTLV, two VST.  U8 opcodes have no shared FP16/FP32 lane
  resource and no separate element-shuffle lane ceiling in this contract.
[Timeouts & Locks] Constant-time inventory construction; no I/O or locks.
[Side-Effect Policy] Pure immutable inventory; physical Work comes from the
  common InstructionInventory.capacity_work.
"""
from dataclasses import dataclass
from mindie_sim.prediction.instructions import InstructionClass, InstructionCount, InstructionInventory

_PACKED_BYTES_PER_GROUP = 256
_U8_REGISTER_LANES = _PACKED_BYTES_PER_GROUP
_DUPLICATES_PER_CALLBACK = 3
_GROUPS_PER_CALLBACK = 64

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
        return self.shift_right + self.shift_left + self.and_ops

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
        return _DUPLICATES_PER_CALLBACK * self.vf_groups

    @property
    def packed_bytes(self):
        return _PACKED_BYTES_PER_GROUP * self.groups

    @property
    def expanded_bytes(self):
        return 2 * self.packed_bytes

    @property
    def raw_alu_lanes(self):
        return self.simple_i8 * _U8_REGISTER_LANES

    @property
    def raw_shuffle_lanes(self):
        return self.interleave * _U8_REGISTER_LANES


def inventory(groups: int, vf_groups: int) -> tuple[InstructionInventory, Counts]:
    if type(groups) is not int or groups < 0:
        raise ValueError("gmmsq_groups_must_be_nonnegative_integer")
    if type(vf_groups) is not int or vf_groups < 0:
        raise ValueError("gmmsq_vf_groups_must_be_nonnegative_integer")
    c = Counts(groups, vf_groups)
    inv = InstructionInventory({
        "CreateMask": InstructionCount(c.create_mask),
        "DuplicateU8": InstructionCount(c.duplicate, c.duplicate * _U8_REGISTER_LANES),
        "LoadAlignU8": InstructionCount(c.loads, c.loads * _U8_REGISTER_LANES, read_bytes=c.packed_bytes),
        "ShiftRightU8": InstructionCount(c.shift_right, c.shift_right * _U8_REGISTER_LANES),
        "ShiftLeftU8": InstructionCount(c.shift_left, c.shift_left * _U8_REGISTER_LANES),
        "AndU8": InstructionCount(c.and_ops, c.and_ops * _U8_REGISTER_LANES),
        "InterleaveU8": InstructionCount(c.interleave, c.raw_shuffle_lanes),
        "StoreAlignU8": InstructionCount(c.stores, c.stores * _U8_REGISTER_LANES,
                                         write_bytes=c.expanded_bytes),
    })
    return inv, c


def work(groups: int):
    vf_groups = (groups + _GROUPS_PER_CALLBACK - 1) // _GROUPS_PER_CALLBACK
    inv, c = inventory(groups, vf_groups)
    return inv.capacity_work(CLASSES), c, inv
