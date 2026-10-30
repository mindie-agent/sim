"""Typed register instruction inventories and independent capacity constraints.

[Role] Preserve instruction calls, active lanes and local payload separately.
[Invariants] Issue slots are capacity demand, never instruction latency. Mask
  lanes do not erase calls. UB payload is not a bank transaction count.
[Timeouts & Locks] Pure, bounded inventories of a source basic block.
[Side-Effect Policy] No compiler, device, calibration or observation access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .mechanism import Work

# Units are operations of the declared source/compiled boundary. These names
# do not assert one API call lowers to one hardware instruction.
PRIMITIVE_UNITS = {
    **dict.fromkeys(("scalar_ub_access", "scalar_gm_access", "scalar_alu_issue",
                    "pipe_barrier", "cross_core_sync"), "calls"),
    "mte2_descriptor_issue": "descriptors", "mte3_descriptor_issue": "descriptors",
    "simt_warp_issue": "warps", "simt_atomic_rmw": "transactions",
    "vbitsort32": "blocks32", "vmrgsort4v2": "merges4",
}


@dataclass(frozen=True)
class PrimitiveInventory:
    counts: Mapping[str, float]

    def partition(self, declared_roles):
        """Keep unknown nonzero service visible instead of inventing a rate."""
        work = Work(resources=self.counts)
        priced = {role: n for role, n in work.resources.items()
                  if n and role in declared_roles}
        unpriced = {role: n for role, n in work.resources.items()
                    if n and role not in declared_roles}
        return Work(resources=priced), unpriced

    def to_dict(self, declared_roles):
        work, unknown = self.partition(declared_roles)
        return {"counts": dict(self.counts), "priced_service_work": dict(work.resources),
                "unpriced_service_work": unknown,
                "boundary": "declared_operation_counts_not_universal_isa_latency"}

REGISTER_RESOURCES = (
    "reg_data_writes", "reg_ub_accesses", "reg_ub_stores", "reg_alu_lanes",
    "reg_special_lanes", "reg_reduce_lanes", "reg_shuffle_lanes",
)
REGISTER_WORK_ROLES = ("reg_issue_slots", *REGISTER_RESOURCES)


@dataclass(frozen=True)
class InstructionClass:
    issue_slots: int
    data_writes: int = 0
    lane_resource: str | None = None
    ub_read: bool = False
    ub_write: bool = False
    call_resource: str | None = None


@dataclass(frozen=True)
class InstructionCount:
    calls: int = 0
    active_lanes: int = 0
    read_bytes: int = 0
    write_bytes: int = 0

    def __add__(self, other):
        return InstructionCount(
            self.calls + other.calls, self.active_lanes + other.active_lanes,
            self.read_bytes + other.read_bytes, self.write_bytes + other.write_bytes)

    def scaled(self, count):
        return InstructionCount(*(value * count for value in (
            self.calls, self.active_lanes, self.read_bytes, self.write_bytes)))


@dataclass(frozen=True)
class InstructionInventory:
    counts: Mapping[str, InstructionCount]
    barriers: int = 0

    def __add__(self, other):
        zero = InstructionCount()
        return InstructionInventory(
            {key: self.counts.get(key, zero) + other.counts.get(key, zero)
             for key in self.counts.keys() | other.counts.keys()},
            self.barriers + other.barriers)

    def scaled(self, count):
        return InstructionInventory(
            {key: value.scaled(count) for key, value in self.counts.items()},
            self.barriers * count)

    def capacity_work(self, classes: Mapping[str, InstructionClass]) -> Work:
        """Independent necessary capacity bounds, with no compiler schedule.

        A single-issue call consumes both slots of a two-slot issue cycle.
        Combining inventories can admit fractional cycles and ideal pairing;
        dependencies and compiler bundles require a finer executable recipe.
        """
        resources = {}

        def add(role, amount):
            if amount:
                resources[role] = resources.get(role, 0) + amount

        for opcode, count in self.counts.items():
            rule = classes[opcode]
            if rule.call_resource:
                add(rule.call_resource, count.calls)
            add("reg_issue_slots", rule.issue_slots * count.calls)
            add("reg_data_writes", rule.data_writes * count.calls)
            if rule.lane_resource:
                add(rule.lane_resource, count.active_lanes)
            if rule.ub_read:
                add("reg_ub_accesses", count.calls)
            if rule.ub_write:
                add("reg_ub_accesses", count.calls)
                add("reg_ub_stores", count.calls)
        return Work(resources=resources)

    def to_dict(self):
        return {
            "instructions": {
                key: {"calls": value.calls, "active_lanes": value.active_lanes,
                      "read_bytes": value.read_bytes, "write_bytes": value.write_bytes}
                for key, value in sorted(self.counts.items())},
            "barriers": self.barriers,
            "boundary": "source_calls_and_capacity_bounds_not_instruction_latency",
        }


# These classes bind declared *operation* capacity only. No undocumented
# Reg issue-slot count, compiler lowering or sorting latency is assigned.
CONTROL_INSTRUCTIONS = {
    "GetValueUB": InstructionClass(0, call_resource="scalar_ub_access"),
    "GetValueGM": InstructionClass(0, call_resource="scalar_gm_access"),
    "ScalarALU": InstructionClass(0, call_resource="scalar_alu_issue"),
    "DataCopyRead": InstructionClass(0, call_resource="mte2_descriptor_issue"),
    "DataCopyWrite": InstructionClass(0, call_resource="mte3_descriptor_issue"),
    "SIMTWarpIssue": InstructionClass(0, call_resource="simt_warp_issue"),
    "SIMTAtomicRMW": InstructionClass(0, call_resource="simt_atomic_rmw"),
    "VBitSort32": InstructionClass(0, call_resource="vbitsort32"),
    "VMrgSort4V2": InstructionClass(0, call_resource="vmrgsort4v2"),
    "PipeBarrier": InstructionClass(0, call_resource="pipe_barrier"),
    "CrossCoreSync": InstructionClass(0, call_resource="cross_core_sync"),
}
