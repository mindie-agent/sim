"""Per-owner UB reservations, tiled liveness and single-operator demand.

[Role] Build the exact per-owner UB worksheet (allocation liveness, padding,
  scratch and cube operands) and the irreducible single-operator reservation
  used by baseline plan admission.
[Invariants] Effective capacity is ``min(physical, usable)``; every physical
  tile owns a reservation and the aligned tail is never undercounted.  An
  oversized demand is retained, never erased, so plan/Engine admission rejects
  it explicitly.
[Timeouts & Locks] Closed-form tile accounting; no I/O, clocks or locks.
[Side-Effect Policy] Pure immutable records over the supplied typed graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from mindie_sim.prediction.fusion.contract import (
    CubeResources, ReductionMerge, _cube_operands, max_tile_bytes, tile_bytes,
    tile_partition_padding,
)
from mindie_sim.prediction.fusion.graph import (
    FeasibilityError, PARAM_ROLES, REDUCTION_KINDS, TypedGraph, _align,
    _is_public, external_readers,
)

_MIN_REDUCTION_SCRATCH_BYTES = 32


@dataclass(frozen=True)
class LivenessPoint:
    phase: int
    op_id: str
    live_values: tuple[str, ...]
    live_bytes: int


@dataclass(frozen=True)
class TileLiveness:
    points: tuple[LivenessPoint, ...]
    peak_live_bytes: int


def tile_liveness(graph: TypedGraph, members: Iterable[str], *, tile_rows: int,
                  channel_width: int | None = None, rows: int | None = None,
                  start_row: int = 0, worst: bool = False) -> TileLiveness:
    """Single-slot activation liveness at each fused-kernel phase."""
    chosen = tuple(dict.fromkeys(members))
    total_rows = int(rows or graph.rows)
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    for index, op_id in enumerate(chosen):
        op = graph.ops[op_id]
        for value_id in op.inputs:
            first.setdefault(value_id, index)
            last[value_id] = max(last.get(value_id, index), index)
        for value_id in op.outputs:
            first[value_id] = index
            last.setdefault(value_id, index)
    points = []
    for index, op_id in enumerate(chosen):
        live = tuple(value_id for value_id in first
                     if first[value_id] <= index <= last[value_id]
                     and graph.values[value_id].role not in PARAM_ROLES
                     and not (graph.values[value_id].role == "scale"
                              and graph.values[value_id].producer not in chosen))
        live_bytes = sum(_align(
            _tile_size(graph.values[v], tile_rows, total_rows, channel_width,
                       start_row, worst),
            graph.values[v].alignment_bytes) for v in live)
        points.append(LivenessPoint(index, op_id, live, live_bytes))
    return TileLiveness(tuple(points), max((p.live_bytes for p in points), default=0))


@dataclass(frozen=True)
class Allocation:
    name: str
    bytes_per_instance: int
    instances: int
    lifetime: str
    pool: str = "ub"
    units: tuple[int, ...] = (0,)

    @property
    def total_bytes(self) -> int:
        return self.bytes_per_instance * self.instances

    def record(self) -> dict:
        return {"name": self.name, "pool": self.pool,
                "bytes_per_instance": self.bytes_per_instance,
                "total_bytes": self.total_bytes, "instances": self.instances,
                "units": list(self.units), "lifetime": self.lifetime}


@dataclass(frozen=True)
class UbWorksheet:
    capacity_per_owner_bytes: int
    token_tile_rows: int
    channel_tile_width: int
    allocations: tuple[Allocation, ...]
    ub_tile_bytes: int
    live_tile_bytes: int
    persistent_bytes: int
    reduction_scratch_bytes: int
    buffer_factor: int
    tail_tile_bytes: int
    fits: bool
    usable_bytes: int = 0
    padding_overhead_bytes: int = 0
    cube_allocations: tuple[Allocation, ...] = ()

    def record(self) -> dict:
        return {"owner_axis": "AIV", "owner_units": [0],
                "capacity_per_owner_bytes": self.capacity_per_owner_bytes,
                "usable_bytes": self.usable_bytes or self.capacity_per_owner_bytes,
                "token_tile_rows": self.token_tile_rows,
                "channel_tile_width": self.channel_tile_width,
                "allocations": [a.record() for a in self.allocations],
                "cube_allocations": [a.record() for a in self.cube_allocations],
                "ub_tile_bytes": self.ub_tile_bytes,
                "tail_tile_bytes": self.tail_tile_bytes,
                "padding_overhead_bytes": self.padding_overhead_bytes,
                "fits": self.fits}


def _tile_size(value, rows: int, total_rows: int, channel_width: int | None,
               start_row: int, worst: bool) -> int:
    if worst:
        return max_tile_bytes(value, rows, total_rows, channel_width)
    return tile_bytes(value, rows, total_rows, channel_width, start_row)


def _ub_allocations(graph: TypedGraph, chosen: tuple[str, ...], rows: int,
                    total_rows: int, channel_width: int, working: bool,
                    external: Mapping[str, tuple[str, ...]],
                    allow_cube: bool = False,
                    reduction_merge: ReductionMerge | None = None,
                    owner: int = 0, start_row: int = 0,
                    worst: bool = False) -> tuple[list[Allocation], int]:
    boundary = tuple(dict.fromkeys(
        v for o in chosen for v in graph.ops[o].inputs
        if graph.values[v].producer is None or graph.values[v].producer not in chosen))
    inside = tuple(dict.fromkeys(v for o in chosen for v in graph.ops[o].outputs))
    allocations: list[Allocation] = []
    params: set[str] = set()
    units = (owner,)

    def size_of(value_id: str) -> int:
        value = graph.values[value_id]
        return _tile_size(value, rows, total_rows, channel_width, start_row,
                          worst)

    def add_param(value_id: str):
        if value_id not in params and not (allow_cube and graph.values[value_id].role == "weight"):
            params.add(value_id)
            value = graph.values[value_id]
            allocations.append(Allocation(
                f"param:{value_id}", _align(size_of(value_id),
                                            value.alignment_bytes),
                1, "persistent", units=units))

    for value_id in boundary:
        value = graph.values[value_id]
        size = size_of(value_id)
        if allow_cube and value.role == "weight":
            continue
        if value.role in PARAM_ROLES or value.role == "scale":
            add_param(value_id)
        elif size:
            allocations.append(Allocation(f"in:{value_id}", _align(size, value.alignment_bytes),
                                          2, "input_queue", units=units))
    for value_id in inside:
        value = graph.values[value_id]
        if allow_cube and value.role == "weight":
            continue
        if value.role in PARAM_ROLES:
            add_param(value_id)
        elif value_id in external or _is_public(graph, value_id):
            allocations.append(Allocation(
                f"out:{value_id}", _align(size_of(value_id),
                                          value.alignment_bytes), 2, "output_queue", units=units))
        elif value.role == "scale":
            allocations.append(Allocation(
                f"scale:{value_id}", _align(size_of(value_id),
                                            value.alignment_bytes), 1, "persistent", units=units))
        elif not working:
            allocations.append(Allocation(
                f"work:{value_id}", _align(size_of(value_id),
                                           value.alignment_bytes), 2, "working_tile", units=units))
    if working:
        reduction_widths = [
            graph.values[v].shape[-1]
            for op_id in chosen if graph.ops[op_id].kind in REDUCTION_KINDS
            for v in graph.ops[op_id].inputs
            if graph.values[v].role not in {"weight", "parameter", "state"}]
        if reduction_widths:
            # The fp32 accumulator spans the actual reduced width, not an
            # unrelated wider boundary operand of the same region.
            width = min(channel_width, max(reduction_widths))
        else:
            width = min(channel_width, max(
                (graph.values[v].shape[-1] for v in boundary
                 if graph.values[v].role not in {"weight", "parameter", "state"}),
                default=channel_width))
        allocations.append(Allocation("fp32_working", _align(rows * width * 4),
                                      1, "working_tile", units=units))
    for op_id in chosen:
        for value_id in graph.ops[op_id].inputs:
            if graph.values[value_id].role in {"weight", "parameter", "scale"}:
                add_param(value_id)
    reduction_widths = [
        graph.values[v].shape[-1]
        for op_id in chosen if graph.ops[op_id].kind in REDUCTION_KINDS
        for v in graph.ops[op_id].inputs
        if graph.values[v].role not in PARAM_ROLES]
    scratch_width = (min(channel_width, max(reduction_widths))
                     if reduction_widths else channel_width)
    scratch = _align(scratch_width) if any(
        graph.ops[o].kind in REDUCTION_KINDS for o in chosen) else 0
    if reduction_merge is not None:
        scratch += _align(reduction_merge.merge_bytes)
    if scratch:
        allocations.append(Allocation("reduction_scratch", max(scratch, _MIN_REDUCTION_SCRATCH_BYTES), 1,
                                      "reduction", units=units))
    return allocations, scratch


def _cube_allocations(graph: TypedGraph, chosen: tuple[str, ...], rows: int,
                      resources: CubeResources | None,
                      owner: int = 0) -> tuple[Allocation, ...]:
    """Declared L1/L0A/L0B/L0C operands held while the Cube phase runs."""
    if resources is None or not resources.complete:
        return ()
    peak: dict[str, int] = {}
    for op_id in chosen:
        op = graph.ops[op_id]
        if op.resource != "cube":
            continue
        for pool, amount in _cube_operands(graph, op, rows, resources).items():
            peak[pool] = max(peak.get(pool, 0), int(amount))
    return tuple(Allocation(f"cube:{pool}", amount, 1, "cube_operand", pool=pool,
                            units=(owner,))
                 for pool, amount in sorted(peak.items()) if amount > 0)


def ub_worksheet(graph: TypedGraph, members: Iterable[str], *, tile_rows: int,
                 channel_width: int, capacity_per_owner: int,
                 buffer_factor: int = 2, rows: int | None = None,
                 allow_cube: bool = False, usable_bytes: int | None = None,
                 owner: int = 0,
                 reduction_merge: ReductionMerge | None = None,
                 cube_resources: CubeResources | None = None) -> UbWorksheet:
    chosen = tuple(dict.fromkeys(members))
    total_rows = int(rows or graph.rows)
    tile_rows = max(1, min(int(tile_rows), total_rows))
    channel_width = max(1, int(channel_width))
    external = external_readers(graph, chosen)
    working = any(graph.ops[o].kind in {"add", "rms_norm", "reduce", "softmax"}
                  for o in chosen)
    physical = int(capacity_per_owner)
    if physical <= 0:
        raise FeasibilityError("ub_capacity_invalid")
    usable = (physical if usable_bytes is None
              else min(physical, int(usable_bytes)))
    if usable <= 0:
        raise FeasibilityError("ub_usable_capacity_invalid")
    # Every physical tile owns a reservation: the aligned remainder tile can
    # be larger than an interior tile (e.g. 33 B -> 64 B per allocation), so
    # admission is the maximum over all starts, never tile 0 alone.  The
    # balanced partition makes the worst full block and the tail sufficient;
    # both are computed in closed form rather than expanding ``row_count``.
    full_rows = min(tile_rows, total_rows)
    remainder = total_rows % tile_rows
    tail_rows = remainder or (tile_rows if total_rows > tile_rows else 0)
    cases: list[tuple[int, bool]] = [(full_rows, False)]
    if tail_rows and (tail_rows != full_rows or total_rows > tile_rows):
        cases.append((tail_rows, True))
    scratch = 0
    worst_allocations: tuple[Allocation, ...] = ()
    best_total = -1
    cube_allocations: tuple[Allocation, ...] = ()
    totals: list[int] = []
    lives: list[int] = []
    persistents: list[int] = []
    tail_total = 0
    for rows_here, is_tail in cases:
        allocations, scratch = _ub_allocations(
            graph, chosen, rows_here, total_rows, channel_width, working, external,
            allow_cube, reduction_merge, owner, worst=True)
        tile_total = sum(a.total_bytes for a in allocations)
        if tile_total > best_total:
            best_total, worst_allocations = tile_total, tuple(allocations)
        totals.append(tile_total)
        if is_tail:
            tail_total = tile_total
        persistents.append(sum(a.total_bytes for a in allocations
                               if a.lifetime == "persistent"))
        lives.append(tile_liveness(graph, chosen, tile_rows=rows_here,
                                   channel_width=channel_width, rows=total_rows,
                                   worst=True).peak_live_bytes)
    padding = sum(tile_partition_padding(graph.values[v], tile_rows, total_rows)
                  for o in chosen
                  for v in (*graph.ops[o].inputs, *graph.ops[o].outputs))
    if allow_cube:
        cube_allocations = _cube_allocations(graph, chosen, tile_rows,
                                             cube_resources, owner)
    total = max(totals, default=0)
    live = max(lives, default=0)
    persistent = max(persistents, default=0)
    tail = tail_total if len(cases) > 1 else 0
    fits = (total <= usable and
            live * buffer_factor + scratch + persistent <= usable)
    return UbWorksheet(capacity_per_owner, tile_rows, channel_width,
                       worst_allocations, total, live, persistent, scratch,
                       buffer_factor, tail, fits, usable, padding, cube_allocations)




def single_op_ub(graph: TypedGraph, op_id: str, *, tile_rows: int,
                 capacity_per_owner: int, owner: int = 0,
                 allow_cube: bool = False,
                 cube_resources: CubeResources | None = None,
                 usable_ub_bytes: int | None = None) -> tuple[Allocation, ...]:
    """Per-owner reservation of one unfused operator, max over its tiles.

    The effective budget is the declared physical UB capacity capped by the
    declared usable capacity.  An allocation that does not fit is *kept*: the
    Engine/plan admission then rejects the infeasible plan explicitly.  It is
    never silently dropped, which would erase real UB demand.
    """
    if not isinstance(capacity_per_owner, (int, float)) \
            or not math.isfinite(float(capacity_per_owner)) \
            or capacity_per_owner <= 0:
        raise FeasibilityError("single_op_ub_capacity_invalid")
    _effective = int(capacity_per_owner)
    if usable_ub_bytes is not None:
        if not isinstance(usable_ub_bytes, (int, float)) \
                or not math.isfinite(float(usable_ub_bytes)) \
                or usable_ub_bytes <= 0:
            raise FeasibilityError("single_op_ub_usable_capacity_invalid")
        # The effective budget is the physical capacity capped by the declared
        # usable per-owner budget.  Plan admission compares the returned sum
        # against this same bound and rejects an oversized single operator.
        _effective = min(_effective, int(usable_ub_bytes))
    op = graph.ops[op_id]
    width = max((graph.values[v].shape[-1] for v in (*op.inputs, *op.outputs)
                 if graph.values[v].role not in {"weight", "parameter"}), default=1)
    tile = max(1, min(tile_rows, graph.rows))
    full_rows = min(tile, graph.rows)
    tail_rows = graph.rows % tile
    cases = [(full_rows, False)]
    if tail_rows and tail_rows != full_rows:
        cases.append((tail_rows, True))
    best: tuple[Allocation, ...] = ()
    best_total = -1
    for rows_here, _is_tail in cases:
        allocations, _ = _ub_allocations(
            graph, (op_id,), rows_here, graph.rows, max(1, width), False, {},
            allow_cube, None, owner, worst=True)
        total = sum(allocation.total_bytes for allocation in allocations)
        if total > best_total:
            best, best_total = tuple(allocations), total
    return best
