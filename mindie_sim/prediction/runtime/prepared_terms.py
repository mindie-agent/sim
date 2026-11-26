"""Vector sub-pipe lowering and Engine residual snapping constant.

[Role] Lower typed AIV vector instruction groups onto ``vec_issue`` plus
  declared ``vec_{alu,sfu,reduce}`` sub-pipes without a competing aggregate
  ``vector`` row, and expose ``RESIDUAL_SNAP_ULPS`` for Engine solo-step
  residual snapping.
[Invariants] ``target_eff`` scales issue and sub-pipe throughput while leaving
  hardware result latency cycles invariant.
[Timeouts & Locks] Linear O(groups) DAG lowering; no I/O, threads or locks.
[Side-Effect Policy] Pure immutable value builders with zero shared mutation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from mindie_sim.schema.common import ModelingError

if TYPE_CHECKING:
    from ..program.stages import Stage

RESIDUAL_SNAP_ULPS = 4.0
MS_PER_SECOND = 1000.0
VALID_VECTOR_PIPES = frozenset({"vec_alu", "vec_sfu", "vec_reduce"})


@dataclass(frozen=True, slots=True)
class VectorInstructionGroup:
    """One typed AIV instruction group targeting ``vec_issue`` and a sub-pipe."""

    group_id: str
    pipe: Literal["vec_alu", "vec_sfu", "vec_reduce"]
    count: int
    issue_slots: int = 1
    initiation_cycles: int = 1
    result_latency_cycles: int = 0
    dependencies: tuple[str, ...] = ()
    owner: int = 0
    last_issue_mark: bool = True

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ModelingError("vector_group_id_required")
        if self.pipe not in VALID_VECTOR_PIPES:
            raise ModelingError(f"vector_group_pipe_invalid:{self.pipe!r}")
        if (
            self.count < 0 or self.issue_slots < 1
            or self.initiation_cycles < 1 or self.result_latency_cycles < 0
            or self.owner < 0
        ):
            raise ModelingError(f"vector_group_counts_invalid:{self.group_id}")


@dataclass(frozen=True, slots=True)
class BoundVectorOp:
    """Lowered ``vec_issue + vec_{alu,sfu,reduce}`` work without a competing total vector row."""

    owner: int
    issue_work: float
    pipe_work: Mapping[str, float]
    vector_projection: float
    raw_latency_ms: float
    span_ms: float
    stages: tuple[Any, ...] = ()


def _append_vector_group_stages(stages, grp, issue_res, pipe_res, deps_after,
                                issue_w, exec_w, raw_lat_ms, target_eff, pl):
    from ..program.stages import Stage

    if grp.count > 1 and raw_lat_ms > 0.0 and grp.last_issue_mark:
        prefix_count = grp.count - 1
        stages.append(Stage(
            f"{grp.group_id}:prefix",
            work={
                issue_res: float(prefix_count * grp.issue_slots) / target_eff,
                pipe_res: float(prefix_count * grp.initiation_cycles) / target_eff,
            },
            after=deps_after,
            placement=pl,
            coupled=True,
        ))
        stages.append(Stage(
            grp.group_id,
            work={
                issue_res: float(grp.issue_slots) / target_eff,
                pipe_res: float(grp.initiation_cycles) / target_eff,
            },
            after=(f"{grp.group_id}:prefix",),
            latency=raw_lat_ms,
            placement=pl,
            coupled=True,
            latency_from_first_service=True,
        ))
        stages.append(Stage(
            f"{grp.group_id}:raw",
            after=(grp.group_id,),
            keep_empty=True,
        ))
    elif grp.count == 1 and raw_lat_ms > 0.0 and grp.last_issue_mark:
        stages.append(Stage(
            grp.group_id,
            work={
                issue_res: issue_w / target_eff,
                pipe_res: exec_w / target_eff,
            },
            after=deps_after,
            latency=raw_lat_ms,
            placement=pl,
            coupled=True,
            latency_from_first_service=True,
        ))
        stages.append(Stage(
            f"{grp.group_id}:raw",
            after=(grp.group_id,),
            keep_empty=True,
        ))
    else:
        stages.append(Stage(
            grp.group_id,
            work={
                issue_res: issue_w / target_eff,
                pipe_res: exec_w / target_eff,
            } if grp.count > 0 else {},
            after=deps_after,
            placement=pl if grp.count > 0 else {},
            coupled=grp.count > 0,
        ))
        if raw_lat_ms > 0.0 and grp.count > 0:
            stages.append(Stage(
                f"{grp.group_id}:raw",
                latency=raw_lat_ms,
                after=(grp.group_id,),
            ))


def _causal_vector_lower_bound(
    ordered_ids: Sequence[str],
    *,
    by_id: Mapping[str, VectorInstructionGroup],
    start_by_id: Mapping[str, float],
    children_by_id: Mapping[str, Sequence[str]],
    ancestors_by_id: Mapping[str, set[str]],
    issue_by_id: Mapping[str, float],
    exec_by_id: Mapping[str, tuple[str, float]],
    pre_issue_by_id: Mapping[str, float],
    pre_exec_by_id: Mapping[str, tuple[str, float]],
    pre_thru_by_id: Mapping[str, float],
    self_tail_by_id: Mapping[str, float],
    eff_issue: float,
    eff_pipes: Mapping[str, float],
) -> float:
    """Carlier-Jackson causal lower bound over release times, pre-anchor work, and RAW tails."""
    if not ordered_ids:
        return 0.0
    node_set = set(ordered_ids)
    tail_by_id: dict[str, float] = {}
    for gid in reversed(ordered_ids):
        child_tail = max(
            (pre_thru_by_id[c] + tail_by_id[c] for c in children_by_id.get(gid, ()) if c in node_set),
            default=0.0,
        )
        tail_by_id[gid] = self_tail_by_id[gid] + child_tail
    pos_ids = [gid for gid in ordered_ids if by_id[gid].count > 0]
    if not pos_ids:
        return 0.0
    topo_pos = {gid: idx for idx, gid in enumerate(ordered_ids)}
    best = 0.0
    for r_thresh in sorted({start_by_id[gid] for gid in pos_ids}):
        eligible = sorted(
            (gid for gid in pos_ids if start_by_id[gid] >= r_thresh),
            key=lambda gid: (-tail_by_id[gid], topo_pos[gid]),
        )
        in_sub: set[str] = set()
        upgraded: set[str] = set()
        sum_issue = 0.0
        sum_pipes: dict[str, float] = {}
        n_el = len(eligible)
        for idx, gid in enumerate(eligible):
            sum_issue += pre_issue_by_id[gid]
            g_pipe, g_pre_exec = pre_exec_by_id[gid]
            sum_pipes[g_pipe] = sum_pipes.get(g_pipe, 0.0) + g_pre_exec
            in_sub.add(gid)
            for anc in ancestors_by_id[gid]:
                if anc in in_sub and anc not in upgraded:
                    upgraded.add(anc)
                    sum_issue += issue_by_id[anc] - pre_issue_by_id[anc]
                    a_pipe, a_exec = exec_by_id[anc]
                    _, a_pre_exec = pre_exec_by_id[anc]
                    sum_pipes[a_pipe] = sum_pipes.get(a_pipe, 0.0) + (a_exec - a_pre_exec)
            if idx == n_el - 1 or tail_by_id[eligible[idx + 1]] < tail_by_id[gid]:
                pipe_lb = max((w / eff_pipes[p] for p, w in sum_pipes.items()), default=0.0)
                cand = r_thresh + max(sum_issue / eff_issue, pipe_lb) + tail_by_id[gid]
                if cand > best:
                    best = cand
    return best


def bind_vector_groups(
    groups: Sequence[VectorInstructionGroup],
    *,
    issue_rate_per_ms: float,
    pipe_rates_per_ms: Mapping[str, float],
    clock_hz: float,
    target_eff: float = 1.0,
    resource_names: Mapping[str, str] | None = None,
) -> BoundVectorOp:
    """Lower AIV instruction groups onto ``vec_issue`` and declared sub-pipes.

    ``target_eff`` scales issue and sub-pipe throughput while leaving hardware
    result latency cycles invariant. The legacy ``vector`` total is computed as
    a pure reporting projection and never occupies a competing resource row.
    """
    if issue_rate_per_ms <= 0.0 or clock_hz <= 0.0 or not (0.0 < target_eff <= 1.0):
        raise ModelingError("bind_vector_rates_invalid")
    eff_issue = issue_rate_per_ms * target_eff
    eff_pipes = {
        pipe: float(rate) * target_eff
        for pipe, rate in pipe_rates_per_ms.items()
    }
    res_map = dict(resource_names or {})
    if not groups:
        return BoundVectorOp(
            owner=0, issue_work=0.0, pipe_work={}, vector_projection=0.0,
            raw_latency_ms=0.0, span_ms=0.0, stages=(),
        )
    owner = groups[0].owner
    by_id: dict[str, VectorInstructionGroup] = {}
    for grp in groups:
        if grp.owner != owner:
            raise ModelingError(f"vector_group_owner_mismatch:{grp.group_id}")
        if grp.group_id in by_id:
            raise ModelingError(f"vector_group_duplicate:{grp.group_id}")
        if grp.pipe not in eff_pipes or eff_pipes[grp.pipe] <= 0.0:
            raise ModelingError(f"vector_subpipe_missing:{grp.pipe}")
        by_id[grp.group_id] = grp

    cycle_ms = MS_PER_SECOND / float(clock_hz)
    total_issue = 0.0
    pipe_totals: dict[str, float] = {}
    stages: list[Stage] = []
    ordered_ids: list[str] = []
    start_by_id: dict[str, float] = {}
    raw_ready_by_id: dict[str, float] = {}
    ancestors_by_id: dict[str, set[str]] = {}
    children_by_id: dict[str, list[str]] = {grp.group_id: [] for grp in groups}
    issue_by_id: dict[str, float] = {}
    exec_by_id: dict[str, tuple[str, float]] = {}
    pre_issue_by_id: dict[str, float] = {}
    pre_exec_by_id: dict[str, tuple[str, float]] = {}
    pre_thru_by_id: dict[str, float] = {}
    self_tail_by_id: dict[str, float] = {}
    max_raw_lat = 0.0

    for grp in groups:
        ancestors: set[str] = set()
        for dep in grp.dependencies:
            if dep not in by_id:
                raise ModelingError(f"vector_group_unknown_dep:{grp.group_id}:{dep}")
            ancestors.add(dep)
            ancestors.update(ancestors_by_id[dep])
        ancestors_by_id[grp.group_id] = ancestors
        start_ms = max((raw_ready_by_id[d] for d in grp.dependencies), default=0.0)
        if ancestors:
            anc_issue = sum(issue_by_id[u] for u in ancestors) / eff_issue
            anc_pipes: dict[str, float] = {}
            for u in ancestors:
                u_pipe, u_exec = exec_by_id[u]
                anc_pipes[u_pipe] = anc_pipes.get(u_pipe, 0.0) + u_exec
            anc_pipe_lb = max((w / eff_pipes[p] for p, w in anc_pipes.items()), default=0.0)
            anc_ordered = [uid for uid in ordered_ids if uid in ancestors]
            anc_causal_lb = _causal_vector_lower_bound(
                anc_ordered,
                by_id=by_id,
                start_by_id=start_by_id,
                children_by_id=children_by_id,
                ancestors_by_id=ancestors_by_id,
                issue_by_id=issue_by_id,
                exec_by_id=exec_by_id,
                pre_issue_by_id=pre_issue_by_id,
                pre_exec_by_id=pre_exec_by_id,
                pre_thru_by_id=pre_thru_by_id,
                self_tail_by_id=self_tail_by_id,
                eff_issue=eff_issue,
                eff_pipes=eff_pipes,
            )
            start_ms = max(start_ms, anc_issue, anc_pipe_lb, anc_causal_lb)
        start_by_id[grp.group_id] = start_ms
        for dep in grp.dependencies:
            children_by_id[dep].append(grp.group_id)
        issue_w = float(grp.count * grp.issue_slots)
        exec_w = float(grp.count * grp.initiation_cycles)
        issue_by_id[grp.group_id] = issue_w
        exec_by_id[grp.group_id] = (grp.pipe, exec_w)
        total_issue += issue_w
        pipe_totals[grp.pipe] = pipe_totals.get(grp.pipe, 0.0) + exec_w
        thru_ms = max(issue_w / eff_issue, exec_w / eff_pipes[grp.pipe]) if grp.count > 0 else 0.0
        raw_lat_ms = float(grp.result_latency_cycles) * cycle_ms if grp.count > 0 else 0.0
        if raw_lat_ms > max_raw_lat:
            max_raw_lat = raw_lat_ms
        if raw_lat_ms > 0.0 and grp.count > 0:
            if grp.last_issue_mark:
                pre_i = float((grp.count - 1) * grp.issue_slots)
                pre_e = float((grp.count - 1) * grp.initiation_cycles)
                tail_thru = max(
                    float(grp.issue_slots) / eff_issue,
                    float(grp.initiation_cycles) / eff_pipes[grp.pipe],
                )
                self_tail = max(tail_thru, raw_lat_ms)
                per_inst_ms = thru_ms / float(grp.count)
                last_issue_ms = start_ms + float(grp.count - 1) * per_inst_ms
                raw_ready = max(start_ms + thru_ms, last_issue_ms + raw_lat_ms)
            else:
                pre_i = issue_w
                pre_e = exec_w
                self_tail = raw_lat_ms
                raw_ready = start_ms + thru_ms + raw_lat_ms
        else:
            pre_i = issue_w
            pre_e = exec_w
            self_tail = 0.0
            raw_ready = start_ms + thru_ms if grp.count > 0 else start_ms
        pre_issue_by_id[grp.group_id] = pre_i
        pre_exec_by_id[grp.group_id] = (grp.pipe, pre_e)
        pre_thru_by_id[grp.group_id] = (
            max(pre_i / eff_issue, pre_e / eff_pipes[grp.pipe]) if grp.count > 0 else 0.0
        )
        self_tail_by_id[grp.group_id] = self_tail
        raw_ready_by_id[grp.group_id] = raw_ready
        ordered_ids.append(grp.group_id)

        issue_res = res_map.get("vec_issue", "vec_issue")
        pipe_res = res_map.get(grp.pipe, grp.pipe)
        deps_after = tuple(
            f"{d}:raw" if (by_id[d].count > 0 and by_id[d].result_latency_cycles > 0) else d
            for d in grp.dependencies
        )
        pl = {issue_res: (owner,), pipe_res: (owner,)}
        _append_vector_group_stages(
            stages, grp, issue_res, pipe_res, deps_after, issue_w, exec_w,
            raw_lat_ms, target_eff, pl,
        )

    causal_lb = _causal_vector_lower_bound(
        ordered_ids,
        by_id=by_id,
        start_by_id=start_by_id,
        children_by_id=children_by_id,
        ancestors_by_id=ancestors_by_id,
        issue_by_id=issue_by_id,
        exec_by_id=exec_by_id,
        pre_issue_by_id=pre_issue_by_id,
        pre_exec_by_id=pre_exec_by_id,
        pre_thru_by_id=pre_thru_by_id,
        self_tail_by_id=self_tail_by_id,
        eff_issue=eff_issue,
        eff_pipes=eff_pipes,
    )
    span_ms = max(
        total_issue / eff_issue,
        max((w / eff_pipes[p] for p, w in pipe_totals.items()), default=0.0),
        max(raw_ready_by_id.values(), default=0.0),
        causal_lb,
    )

    return BoundVectorOp(
        owner=owner,
        issue_work=total_issue,
        pipe_work=pipe_totals,
        vector_projection=sum(pipe_totals.values()),
        raw_latency_ms=max_raw_lat,
        span_ms=span_ms,
        stages=tuple(stages),
    )


__all__ = [
    "BoundVectorOp",
    "RESIDUAL_SNAP_ULPS",
    "VALID_VECTOR_PIPES",
    "VectorInstructionGroup",
    "bind_vector_groups",
]
