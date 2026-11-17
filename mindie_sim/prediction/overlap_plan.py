"""Shared-expert and microbatch overlap contracts.

[Role] Bind a shared-expert branch to its own stream, build independent
  microbatch regions and validate that joins wait for every branch.
[Invariants] Shared work depends on the common input, never on routed
  completion. Streams retain the same HBM/AIV/Cube/link resource identities.
  A microbatch never divides a full-batch duration; it rebuilds real operators.
[Timeouts & Locks] Bounded request/operator traversal; no I/O or locks.
[Side-Effect Policy] Returns immutable operator tuples and region records.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec



@dataclass(frozen=True)
class SharedExpertRegion:
    prefix: str
    input_id: str
    shared_members: tuple[str, ...]
    routed_members: tuple[str, ...] = ()
    join_id: str = ""
    stream: str = "shared_stream"


@dataclass(frozen=True)
class MicrobatchPolicy:
    count: int = 2
    min_tokens: int = 1


@dataclass(frozen=True)
class MicrobatchRegion:
    index: int
    views: tuple[object, ...]
    stream: str
    comm_stream: str


def _is_shared(op: OpSpec) -> bool:
    return (".shared" in str(getattr(op, "op_id", ""))
            or str(getattr(op, "family", "")).startswith("shared_"))


def bind_shared_expert_streams(ops: Sequence[OpSpec], *,
                               region: SharedExpertRegion) -> tuple[OpSpec, ...]:
    """Place shared operations on one stream without changing their work."""
    members = set(region.shared_members)
    result = []
    for op in ops:
        if op.op_id in members or _is_shared(op):
            if region.input_id and region.input_id not in tuple(op.deps):
                op = replace(op, deps=tuple(dict.fromkeys((*op.deps, region.input_id))))
            op = replace(op, stream=region.stream)
        result.append(op)
    return tuple(result)


def build_microbatch_regions(views: Sequence[object], *,
                             policy: MicrobatchPolicy) -> tuple[MicrobatchRegion, ...]:
    """Partition independent request views into real microbatch cohorts."""
    views = tuple(views)
    if policy.count < 1:
        raise ModelingError("microbatch_policy_requires_positive_count")
    if not views:
        return ()
    count = min(int(policy.count), len(views))
    regions = []
    for index in range(count):
        cohort = tuple(view for position, view in enumerate(views)
                       if position % count == index)
        if not cohort:
            continue
        regions.append(MicrobatchRegion(
            index=index, views=cohort,
            stream=f"dbo{index}:compute", comm_stream=f"dbo{index}:comm"))
    return tuple(regions)


def validate_overlap_dependencies(ops: Sequence[OpSpec]) -> None:
    """Require explicit branch inputs and leave join dependencies intact."""
    ids = {op.op_id for op in ops}
    for op in ops:
        if _is_shared(op) and op.op_id not in ids:
            raise ModelingError(f"shared_expert_member_missing:{op.op_id}")
        if op.op_id.endswith(".join"):
            if len(op.deps) < 2:
                raise ModelingError(f"overlap_join_requires_both_branches:{op.op_id}")
            missing = set(op.deps) - ids
            if missing:
                raise ModelingError(
                    f"overlap_join_dependency_missing:{op.op_id}:"
                    + ",".join(sorted(missing)))
