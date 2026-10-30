"""Conditional joint busy service for one declared source-program schedule.

[Role] Project source-program counts onto overlapping owner-local pipe capacity.
[Invariants] Correlated program/metadata/byte features identify one joint
  response, never independent instruction prices. Busy pipes overlap; their
  sum is not wall time. Native payload service and dependencies remain intact.
[Timeouts & Locks] Constant work per physical owner; no locks, I/O or waits.
[Side-Effect Policy] Creates work and stages on the existing Engine only.
"""
from dataclasses import dataclass
from typing import Mapping

from .mechanism import Work
from .program import Stage


@dataclass(frozen=True)
class SourceProgramService:
    """Affine busy demand in ns; resource supply is busy ns per second.

    Each pipe's intercept is inclusive busy work on that owner, not startup
    latency or an inferred barrier. All pipes run concurrently with the source
    program. An enclosing whole-call minimum remains inclusive through its join.
    """

    pipes: Mapping

    def work(self, programs, owners=1):
        return Work(resources={item["resource"]:
            owners * item.get("owner_busy_ns", 0) + programs * item["program_busy_ns"]
            for item in self.pipes.values()})

    def stage(self, context, programs, owner, units):
        work = self.work(programs)
        return Stage("joint_pipe_busy", context.bind(work), after=("reserve",),
                     placement=context.owned(owner, units, work.resources))

    @classmethod
    def select(cls, declaration, *, direction, payload_bytes, metadata_bytes,
               all_active, full_programs, state_shape, dtype, owners):
        if not declaration:
            return None, "not_declared"
        contract = declaration["contract"]
        matches = (payload_bytes == contract["payload_bytes_per_program"]
                   and metadata_bytes == contract["metadata_bytes_per_program"]
                   and tuple(state_shape) == tuple(contract["state_tail_shape"])
                   and dtype == contract["dtype"] and owners == contract["owners"]
                   and all_active and full_programs)
        if not matches:
            return None, "analytic_outside_joint_program_contract"
        return cls(declaration["directions"][direction]), "conditional_joint_busy"
