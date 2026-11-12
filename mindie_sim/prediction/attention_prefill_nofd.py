"""No-FD host core assignment for causal FIA batches.

[Role]
  Reproduce the admitted ``split_core_v2`` search: candidate core counts,
  per-row cost, and the stream-K versus whole-row counterfactual.  Masking
  is right-down causal (sparse mode 3).  Equal S1 and S2 make that shift
  zero.  Flash-decode stays off only when the host cost check says so, including
  shapes whose S2 tile count is larger than ``cores * fdTolerance``.  The
  published pieces are those right-open cursors.  The terminal sentinel keeps
  the tail tiles.  Equal chunks remain only when this search does not apply.

[Invariants]
  Ownership is a serialized right-open cursor over M then S2.  Real M rows are
  contiguous and disjoint.  Phantom rows past ``mBaseNum`` affect the cursor
  the way the host loop does, and are not emitted as tasks.  Costs use the
  host 16/64 alignment, not elapsed time.

[Side-Effect Policy]
  Pure arithmetic.  No device runtime, file, or measurement.
"""
from __future__ import annotations

import math

_FA_TOLERANCE_RATIO = 2
_PRETOKEN = 2147483647
_M_BASE = 128
_S2_BASE = 128


def _within(limit: int, toler: int, value: int) -> bool:
    return limit + toler >= value


def _calc_cost(basic_m: int, basic_s2: int) -> int:
    align_m = (basic_m + 16 - 1) >> 4
    align_s2 = (basic_s2 + 64 - 1) >> 6
    return 6 * align_m + 10 * align_s2


class _RowCost:
    __slots__ = ("s2_start", "s2_end", "cost", "last_cost", "normal_cost", "blocks")

    def __init__(self, s2_start, s2_end, cost, last_cost, normal_cost, blocks):
        self.s2_start = s2_start
        self.s2_end = s2_end
        self.cost = cost
        self.last_cost = last_cost
        self.normal_cost = normal_cost
        self.blocks = blocks


class _Shape:
    """One batch's M rows under right-down causal S1G flattening."""

    def __init__(self, s1: int, s2: int, group: int, *,
                 m_base: int = _M_BASE, s2_base: int = _S2_BASE):
        self.s1 = int(s1)
        self.s2 = int(s2)
        self.group = int(group)
        self.m_base = int(m_base)
        self.s2_base = int(s2_base)
        covered = self.s1 * self.group
        self.m_base_num = (covered + self.m_base - 1) // self.m_base if covered else 0
        self.m_tail = covered % self.m_base
        self.s2_base_num = (self.s2 + self.s2_base - 1) // self.s2_base if self.s2 else 0
        self.s2_tail = self.s2 % self.s2_base
        normal_m = self.m_base
        normal_s2 = self.s2_base
        self.cost_nn = _calc_cost(normal_m, normal_s2)
        self.cost_tn = 0 if self.m_tail == 0 else _calc_cost(self.m_tail, normal_s2)
        self.cost_nt = 0 if self.s2_tail == 0 else _calc_cost(normal_m, self.s2_tail)
        self.cost_tt = (0 if self.m_tail == 0 or self.s2_tail == 0
                        else _calc_cost(self.m_tail, self.s2_tail))
        self.rows = tuple(self._row(m) for m in range(self.m_base_num))
        self.total_cost = sum(row.cost for row in self.rows)
        self.total_blocks = sum(row.blocks for row in self.rows)
        self.max_m_cost = max((row.cost for row in self.rows), default=0)
        last = 0
        for row in self.rows:
            if row.blocks:
                last = row.last_cost
        self.last_block_cost = last

    def _s2_range(self, m_idx: int) -> tuple[int, int]:
        if self.s1 == 0 or self.s2 == 0:
            return 0, 0
        m_first = m_idx * self.m_base
        covered = self.s1 * self.group
        m_last = min(m_first + self.m_base, covered) - 1
        s1_first = m_first // self.group
        s1_last = m_last // self.group
        # sparseMode 3 is right-down causal.  When S1 == S2 the shift is zero,
        # so the triangle matches a left-up nextToken of 0.
        s2_first = s1_first - _PRETOKEN
        s2_last = s1_last + (self.s2 - self.s1)
        if s2_last < 0 or s2_first >= self.s2 or s2_last < s2_first:
            return 0, 0
        s2_first = min(max(s2_first, 0), self.s2 - 1)
        s2_last = min(max(s2_last, 0), self.s2 - 1)
        return s2_first // self.s2_base, s2_last // self.s2_base + 1

    def _row(self, m_idx: int) -> _RowCost:
        s2_start, s2_end = self._s2_range(m_idx)
        if s2_start >= s2_end:
            return _RowCost(s2_start, s2_end, 0, 0, 0, 0)
        blocks = s2_end - s2_start
        tail_blocks = 1 if self.s2_tail and s2_end == self.s2_base_num else 0
        normal_blocks = blocks - tail_blocks
        tail_m = m_idx == self.m_base_num - 1 and self.m_tail != 0
        if tail_m:
            normal_cost = self.cost_tn
            tail_cost = self.cost_tt
        else:
            normal_cost = self.cost_nn
            tail_cost = self.cost_nt
        cost = normal_cost * normal_blocks + tail_cost * tail_blocks
        last = tail_cost if tail_blocks else normal_cost
        return _RowCost(s2_start, s2_end, cost, last, normal_cost, blocks)

    def row_at(self, m_idx: int) -> _RowCost:
        if 0 <= m_idx < self.m_base_num:
            return self.rows[m_idx]
        return self._row(m_idx)

    def copy_row(self, m_idx: int) -> _RowCost:
        row = self.row_at(m_idx)
        return _RowCost(row.s2_start, row.s2_end, row.cost, row.last_cost,
                        row.normal_cost, row.blocks)


class _Work:
    """Batch-major (batch, KV) units.  KV heads of one batch share a shape."""

    def __init__(self, shapes: tuple[_Shape, ...], n2: int):
        self.shapes = shapes
        self.n2 = int(n2)
        self.n_units = len(shapes) * self.n2
        self.m_base = shapes[0].m_base if shapes else _M_BASE
        self.s2_base = shapes[0].s2_base if shapes else _S2_BASE
        self.max_m_cost = max((shape.max_m_cost for shape in shapes), default=0)
        self.total_cost = sum(shape.total_cost for shape in shapes) * self.n2
        self.total_blocks = sum(shape.total_blocks for shape in shapes) * self.n2
        bases = []
        running = 0
        for shape in shapes:
            bases.append(running)
            running += self.n2 * shape.m_base_num
        self._base = tuple(bases)
        self.task_count = running

    def shape_of(self, bn2: int) -> _Shape:
        return self.shapes[bn2 // self.n2]

    def task_index(self, bn2: int, m_idx: int) -> int:
        batch = bn2 // self.n2
        kv = bn2 % self.n2
        shape = self.shapes[batch]
        return self._base[batch] + kv * shape.m_base_num + m_idx


class _Cursor:
    def __init__(self, work: _Work, stream_k: bool):
        self.work = work
        self.stream_k = stream_k
        self.bn2 = 0
        self.finished = False
        self.core_cost = 0
        self.core_block = 0
        self.cost_limit = 0
        self.unassigned = work.total_cost
        self.kv_parts = 1
        self.taken: list[tuple[int, int, int]] = []
        self.shape = work.shape_of(0)
        self.b_cost = self.shape.total_cost
        self.b_block = self.shape.total_blocks
        self.m = 0
        self.row = self.shape.copy_row(0)
        self.s2 = self.row.s2_start

    def _enter_unit(self) -> bool:
        """Move to ``bn2``.  A past-the-end index is the host sentinel."""
        if self.bn2 >= self.work.n_units:
            self.m = 0
            self.s2 = 0
            self.finished = True
            self.b_cost = 0
            self.b_block = 0
            return False
        self.shape = self.work.shape_of(self.bn2)
        self.b_cost = self.shape.total_cost
        self.b_block = self.shape.total_blocks
        self.m = 0
        self.row = self.shape.copy_row(0)
        self.s2 = self.row.s2_start
        return True

    def _note_real(self, bn2: int, m_idx: int, s2_lo: int, s2_hi: int) -> None:
        shape = self.work.shape_of(bn2)
        if 0 <= m_idx < shape.m_base_num and s2_hi > s2_lo:
            self.taken.append((self.work.task_index(bn2, m_idx), s2_lo, s2_hi))

    def assign_by_batch(self) -> None:
        if self.finished:
            return
        while self.b_cost == 0 or _within(
                self.cost_limit, self.shape.last_block_cost // _FA_TOLERANCE_RATIO,
                self.core_cost + self.b_cost):
            if self.m < self.shape.m_base_num:
                for m_idx in range(self.m, self.shape.m_base_num):
                    row = self.shape.rows[m_idx]
                    self._note_real(self.bn2, m_idx, row.s2_start, row.s2_end)
            self.core_cost += self.b_cost
            self.core_block += self.b_block
            self.bn2 += 1
            if not self._enter_unit():
                return

    def assign_by_row(self) -> None:
        if self.finished:
            return
        shape = self.shape
        while _within(self.cost_limit, self.row.last_cost // _FA_TOLERANCE_RATIO,
                      self.core_cost + self.row.cost):
            self._note_real(self.bn2, self.m, self.row.s2_start, self.row.s2_end)
            self.core_cost += self.row.cost
            self.core_block += self.row.blocks
            self.b_cost = self.b_cost - self.row.cost if self.b_cost > self.row.cost else 0
            self.b_block = (self.b_block - self.row.blocks
                            if self.b_block > self.row.blocks else 0)
            guard = 0
            while True:
                self.m += 1
                guard += 1
                self.row = shape.copy_row(self.m)
                if self.row.blocks != 0 or guard > shape.m_base_num + shape.s2_base_num + 2:
                    break
            self.s2 = self.row.s2_start

    def _block_cost(self) -> int:
        if self.s2 == self.row.s2_end - 1:
            return self.row.last_cost
        return self.row.normal_cost

    def assign_by_block(self) -> None:
        if self.finished or not self.stream_k or self.row.blocks == 0:
            return
        cur = self._block_cost()
        while (cur > 0 and self.row.blocks > 0 and self.s2 < self.row.s2_end
               and _within(self.cost_limit, cur // _FA_TOLERANCE_RATIO,
                           self.core_cost + cur)):
            self._note_real(self.bn2, self.m, self.s2, self.s2 + 1)
            self.core_cost += cur
            self.core_block += 1
            self.s2 += 1
            self.b_cost -= cur
            self.row.cost -= cur
            self.b_block -= 1
            self.row.blocks -= 1

    def force_assign(self) -> None:
        if self.finished or self.row.blocks == 0:
            return
        cur = self._block_cost()
        self._note_real(self.bn2, self.m, self.s2, self.s2 + 1)
        self.core_cost += cur
        self.core_block += 1
        self.s2 += 1
        self.b_cost -= cur
        self.b_block -= 1
        self.row.cost -= cur
        self.row.blocks -= 1
        if self.s2 >= self.row.s2_end:
            self.m += 1
            if self.m >= self.shape.m_base_num and self.b_cost <= 0:
                self.bn2 += 1
                self._enter_unit()
                return
            self.row = self.shape.copy_row(self.m)
            self.s2 = self.row.s2_start


def _fd_decision(parts_by_core: list[int], ends: list[tuple[int, int, int]],
                 work: _Work) -> tuple[int, list[int]]:
    """Return (fd_num, m_sizes) using the host's lagged split-row rule."""
    fd_num = 0
    m_sizes: list[int] = []
    kv_parts = 1
    for core, (bn2, m_end, s2_end) in enumerate(ends):
        if core > 0 and kv_parts > 1:
            prev_bn2, prev_m, _prev_s2 = ends[core - 1]
            if not (bn2 == prev_bn2 and m_end == prev_m):
                if prev_bn2 < work.n_units:
                    shape = work.shape_of(prev_bn2)
                    split_m = prev_m
                    if split_m == shape.m_base_num - 1:
                        m_size = shape.s1 * shape.group - split_m * shape.m_base
                    else:
                        m_size = shape.m_base
                    m_sizes.append(m_size)
                    fd_num += 1
                kv_parts = 1
        row = work.shape_of(bn2).row_at(m_end) if bn2 < work.n_units else None
        if row is not None and row.s2_start < s2_end <= row.s2_end:
            kv_parts += 1
        parts_by_core.append(kv_parts)
    return fd_num, m_sizes


def _plan_for_cores(work: _Work, core_num: int, cost_limit: int, *,
                    stream_k: bool) -> dict | None:
    if core_num <= 0 or work.total_blocks <= 0:
        return None
    cursor = _Cursor(work, stream_k)
    ends: list[tuple[int, int, int]] = []
    taken_at: list[list[tuple[int, int, int]]] = []
    max_cost = 0
    for core in range(core_num):
        if max_cost > cost_limit or cursor.finished or cursor.unassigned <= 0:
            break
        cursor.core_cost = 0
        cursor.core_block = 0
        cursor.taken = []
        remaining = core_num - core
        avg = cursor.unassigned // remaining
        if stream_k:
            cursor.cost_limit = avg
        else:
            cursor.cost_limit = max(avg, work.max_m_cost)
        cursor.assign_by_batch()
        cursor.assign_by_row()
        cursor.assign_by_block()
        if cursor.core_block == 0 and stream_k:
            cursor.force_assign()
        ends.append((cursor.bn2, cursor.m, cursor.s2))
        taken_at.append(list(cursor.taken))
        max_cost = max(max_cost, cursor.core_cost)
        cursor.unassigned -= cursor.core_cost
    if not ends:
        return None
    _fd_scratch: list[int] = []
    fd_num, m_sizes = _fd_decision(_fd_scratch, ends, work)
    return {
        "used": len(ends),
        "max_cost": max_cost,
        "ends": ends,
        "taken": taken_at,
        "fd_num": fd_num,
        "m_sizes": m_sizes,
    }


def _choose(work: _Work, cores: int, fd_tolerance: int) -> dict | None:
    max_core = min(cores, work.total_blocks)
    if max_core <= 0:
        return None
    min_core = int(math.sqrt(float(work.total_blocks) + 0.25) + 0.5)
    min_core = min(max(min_core, 1), max_core)
    best = None
    best_cost = 2 ** 62
    for count in range(min_core, max_core + 1):
        trial = _plan_for_cores(work, count, best_cost, stream_k=True)
        if trial is None:
            continue
        if trial["max_cost"] > best_cost:
            continue
        if trial["max_cost"] < best_cost:
            best = trial
            best_cost = trial["max_cost"]
    if best is None:
        return None
    if best["fd_num"] > 0:
        whole = _plan_for_cores(work, best["used"], 2 ** 62, stream_k=False)
        if whole is None:
            return best
        max_fd = max(best["m_sizes"]) if best["m_sizes"] else 0
        full = _calc_cost(min(max_fd, work.m_base), work.s2_base)
        no_fd_cost = whole["max_cost"]
        with_fd_cost = best["max_cost"]
        keep_fd = True
        if no_fd_cost <= 0 * full:
            keep_fd = False
        else:
            keep_fd = no_fd_cost - fd_tolerance * full > with_fd_cost
        if not keep_fd:
            best = whole
    return best


def _work_from_spec(spec) -> _Work | None:
    """Native 128-row causal tasks, or None when this search does not apply."""
    if not spec.native_fia or spec.key_tile != _S2_BASE:
        return None
    if spec.kv_heads < 1 or spec.group < 1 or not spec.requests:
        return None
    shapes = tuple(
        _Shape(query, context + query, spec.group)
        for query, context in spec.requests)
    return _Work(shapes, spec.kv_heads)


def _cursor_ranges(work: _Work, ends: list[tuple[int, int, int]], cores: int):
    """Owner slices between host cursors, including the terminal sentinel.

    ``ends`` are right-open ``(bn2, m, s2)`` cursors.  A backwards cursor is
    refused.  The sentinel ``(n_units, 0, 0)`` closes the tile stream, so the
    last used core keeps the tail instead of dropping it.
    """
    rows = []
    task = 0
    for batch, shape in enumerate(work.shapes):
        for kv in range(work.n2):
            bn2 = batch * work.n2 + kv
            for m_idx, row in enumerate(shape.rows):
                rows.append((bn2, m_idx, row.s2_start, row.s2_end, task))
                task += 1
    origin = {}
    tiles: list[tuple[int, int]] = []
    flat = 0
    for bn2, m_idx, s2_start, s2_end, task in rows:
        origin[bn2, m_idx] = (flat, s2_start)
        for offset in range(max(0, s2_end - s2_start)):
            tiles.append((task, offset))
        flat += max(0, s2_end - s2_start)

    def position(bn2: int, m_idx: int, s2: int):
        if bn2 >= work.n_units and m_idx == 0 and s2 == 0:
            return flat
        found = origin.get((bn2, m_idx))
        if found is None or s2 < found[1]:
            return None
        return found[0] + (s2 - found[1])

    cuts = [0]
    for bn2, m_idx, s2 in ends:
        pos = position(bn2, m_idx, s2)
        if pos is None or pos < cuts[-1] or pos > flat:
            return None
        cuts.append(pos)
    if not cuts or cuts[-1] != flat:
        return None
    grouped: list[list[tuple[int, int, int]]] = []
    for left, right in zip(cuts, cuts[1:]):
        merged: list[tuple[int, int, int]] = []
        for task, offset in tiles[left:right]:
            if merged and merged[-1][0] == task and merged[-1][2] == offset:
                prev = merged[-1]
                merged[-1] = (task, prev[1], offset + 1)
            else:
                merged.append((task, offset, offset + 1))
        grouped.append(merged)
    while len(grouped) < cores:
        grouped.append([])
    return grouped[:cores]


def _host_owner(spec, cores, indices, sizes, fd_tolerance):
    """Host owner plan, or ``(None, "skip")`` when the search does not apply.

    No-FD and FD both come from the same cursors.  ``kind`` is ``nofd`` or
    ``fd``.  A cursor walk that would drop or repeat a tile is not published.
    """
    work = _work_from_spec(spec)
    if work is None or work.task_count != len(sizes) or work.n_units <= 0:
        return None, "skip"
    chosen = _choose(work, cores, max(0, int(fd_tolerance)))
    if chosen is None:
        return None, "skip"
    grouped = _cursor_ranges(work, chosen["ends"], cores)
    if grouped is None:
        return None, "skip"
    owned: dict[tuple[int, int], int] = {}
    for core_rows in grouped:
        for task, lo, hi in core_rows:
            if task < 0 or task >= len(sizes) or hi < lo:
                return None, "skip"
            for offset in range(lo, hi):
                key = (task, offset)
                owned[key] = owned.get(key, 0) + 1
    for task, size in enumerate(sizes):
        if any(owned.get((task, offset), 0) != 1 for offset in range(size)):
            return None, "skip"
        if any(offset >= size for (seen, offset) in owned if seen == task):
            return None, "skip"
    kind = "fd" if chosen["fd_num"] else "nofd"
    plan = _plan_from_ranges(tuple(grouped), indices, sizes, kind=kind)
    if plan is None or (kind == "fd" and plan["fd_splits"] != chosen["fd_num"]):
        return None, "skip"
    if kind == "nofd" and plan["fd_splits"] != 0:
        return None, "skip"
    return plan, kind


def _plan_from_ranges(ranges, indices, sizes, *, kind: str = "nofd"):
    loads = []
    mapped = []
    task_owners: dict = {}
    for core, core_rows in enumerate(ranges):
        load = 0
        row_out = []
        for task, lo, hi in core_rows:
            real = indices[task]
            row_out.append((real, lo, hi))
            load += hi - lo
            task_owners.setdefault(real, set()).add(core)
        if load < 0:
            return None
        loads.append(load)
        mapped.append(tuple(row_out))
    split_tasks = sum(1 for owners in task_owners.values() if len(owners) > 1)
    pieces = sum(len(row) for row in mapped)
    fragments = 0 if kind == "nofd" else pieces - len(sizes)
    return {
        "loads": tuple(loads),
        "tasks": tuple(tuple(item[0] for item in row) for row in mapped),
        "ranges": tuple(mapped),
        "fd_splits": split_tasks,
        "fd_fragments": fragments,
        "total": sum(sizes),
    }


def owner_s2_plan(spec, cores: int, *, fd_tolerance: int = 9):
    """Contiguous S2 ownership.

    Native causal FIA uses the host core search for both the no-FD whole-row
    plan and the FD mid-row plan.  The switch is that search's cost check,
    not ``S2 tiles > cores * fd_tolerance``.  Other geometries use whole rows
    only while the tile count is within that tolerance, then equal chunks.
    """
    from mindie_sim.prediction.attention_prefill import _ceil_div, _time_major_tasks

    cores = max(1, int(cores))
    entries = tuple(_time_major_tasks(spec))
    tasks = tuple(task for _index, task in entries)
    indices = tuple(index for index, _task in entries)
    sizes = tuple(task.key_tiles for task in tasks)
    total = sum(sizes)
    sourced, kind = _host_owner(spec, cores, indices, sizes, fd_tolerance)
    if kind in ("nofd", "fd") and sourced is not None:
        return sourced
    if total <= cores * max(1, int(fd_tolerance)):
        loads = [0] * cores
        ranges = [[] for _ in range(cores)]
        cursor = 0
        target = total / cores
        for core in range(cores):
            while cursor < len(tasks) and (
                    loads[core] == 0 or loads[core] + sizes[cursor] <= target):
                loads[core] += sizes[cursor]
                ranges[core].append((indices[cursor], 0, sizes[cursor]))
                cursor += 1
        while cursor < len(tasks):
            core = min(range(cores), key=lambda value: loads[value])
            loads[core] += sizes[cursor]
            ranges[core].append((indices[cursor], 0, sizes[cursor]))
            cursor += 1
        return {
            "loads": tuple(loads),
            "tasks": tuple(tuple(item[0] for item in row) for row in ranges),
            "ranges": tuple(tuple(row) for row in ranges),
            "fd_splits": 0, "fd_fragments": 0, "total": total,
        }
    chunk = _ceil_div(total, cores) if total else 0
    loads, ranges = [], []
    task_owners = {}
    for core in range(cores):
        start = core * chunk
        if start >= total:
            loads.append(0)
            ranges.append(())
            continue
        end = min(total, start + chunk)
        begin = start
        entries = []
        while start < end:
            ti, k = 0, start
            while k >= sizes[ti]:
                k -= sizes[ti]
                ti += 1
            take = min(sizes[ti] - k, end - start)
            entries.append((indices[ti], k, k + take))
            task_owners.setdefault(indices[ti], set()).add(core)
            start += take
        loads.append(end - begin)
        ranges.append(tuple(entries))
    split_tasks = sum(1 for owners in task_owners.values() if len(owners) > 1)
    fragments = sum(len(row) for row in ranges) - len(tasks)
    return {
        "loads": tuple(loads),
        "tasks": tuple(tuple(item[0] for item in row) for row in ranges),
        "ranges": tuple(ranges),
        "fd_splits": split_tasks, "fd_fragments": fragments, "total": total,
    }


class _PrefillProgram:
    """Owner pipelines for one prefill op.  FD reduction stays on split rows."""

    def __init__(self, context, recipe, spec):
        self.context = context
        self.recipe = recipe
        self.spec = spec

    def start(self, engine, name, ready, *, owner=""):
        from dataclasses import replace

        from mindie_sim.prediction.attention_prefill import (
            _FP32_BYTES, _ceil_div, _read_role, _task_at_for, _task_program,
        )
        from mindie_sim.prediction.mechanism import Work
        from mindie_sim.prediction.program import PipelineProgram, Stage

        total, cores = self.spec.task_count, self.context.core_count
        if not total:
            return ready
        if not self.spec.native_fia:
            # Generic CVDiff/A2/A3 callers keep the historical cyclic owner
            # assignment; the arch35 S2 split is only for the bound FIA path.
            events = []
            for core in range(min(total, cores)):
                count = (total - core + cores - 1) // cores

                def task(index: int, core: int = core):
                    item = _task_at_for(self.spec, core + index * cores)
                    return (Stage("task", program=_task_program(
                        self.context, self.recipe, self.spec, item, core,
                    )),)

                events.append(PipelineProgram(count, min(2, count), task).start(
                    engine, f"{name}:owner{core}:tasks", ready,
                    owner=(f"{owner}:prefill{core}" if owner
                           else f"prefill{core}"),
                ))
            return engine.submit(name + ":complete", after=tuple(events),
                                 owner=owner)

        plan = owner_s2_plan(self.spec, cores)
        events = []
        owner_event = {}
        # The writer owner set is immutable source-plan data.  It must exist
        # before any body callback runs, otherwise the reduction graph is
        # scheduled empty (the previous bug).
        task_owners = {}
        for core, ranges in enumerate(plan["ranges"]):
            for ti, _key_start, _key_end in ranges:
                task_owners.setdefault(ti, []).append(core)
        for core, ranges in enumerate(plan["ranges"]):
            if not ranges:
                continue
            owner_name = (f"{owner}:prefill{core}" if owner else f"prefill{core}")
            if self.spec.issue_kind:
                # One taskId per core across these M ranges.  The owner list
                # itself is the existing plan; this does not search again.
                from mindie_sim.prediction.attention_prefill_issue import (
                    IssueCursor, owner_segments,
                )
                segments = owner_segments(self.spec, ranges, _task_at_for)
                event = IssueCursor(self.context, self.spec, segments, core).start(
                    engine, f"{name}:owner{core}:tasks", ready, owner=owner_name)
            else:
                count = len(ranges)

                def task(index: int, core: int = core, ranges=ranges):
                    ti, key_start, key_end = ranges[index]
                    base = _task_at_for(self.spec, ti)
                    item = replace(base, key_start=key_start,
                                   key_tiles=key_end - key_start)
                    partial = key_start > 0 or key_end < base.key_tiles
                    return (Stage("task", program=_task_program(
                        self.context, self.recipe, self.spec, item, core,
                        partial=partial,
                    )),)

                event = PipelineProgram(count, min(2, count), task).start(
                    engine, f"{name}:owner{core}:tasks", ready, owner=owner_name)
            events.append(event)
            owner_event[core] = event

        fd_events = []
        read = _read_role(self.context, vector=True)
        vector_name = self.context.resource_names.get("vector")
        vector_units = self.context.resource_units.get("vector", 1)
        read_name = self.context.resource_names.get(read)
        read_units = self.context.resource_units.get(read, vector_units)
        lane = 0
        for ti, owners in sorted(task_owners.items()):
            if len(owners) < 2:
                continue
            base = _task_at_for(self.spec, ti)
            rows, dim = base.rows, self.spec.head_dim
            # Arch35 emits two 64-row FD fragments per split-M task, on
            # distinct AIV lanes.  Each fragment reads the matching partial
            # rows from every S2 writer (numerator + LSE denominator).
            fragments = len(owners)
            frag_rows = _ceil_div(rows, 2)
            for fragment in range(2):
                this_rows = min(frag_rows, rows - fragment * frag_rows)
                if this_rows <= 0:
                    continue
                part = fragments * this_rows * (dim * _FP32_BYTES
                                                + 2 * _FP32_BYTES)
                out = this_rows * dim * self.spec.query_bytes
                placement = {}
                if vector_name:
                    placement[vector_name] = (lane % vector_units,)
                if read_name:
                    placement[read_name] = (lane % read_units,)
                fd_events.append(engine.submit(
                    f"{name}:fd{ti}:{fragment}",
                    work=self.context.bind(Work(resources={
                        "hbm": part + out, read: part, "mte3": out,
                        "vector": fragments * this_rows * (dim + 2),
                    })),
                    after=tuple(owner_event[core] for core in owners),
                    placement=placement,
                    owner=(f"{owner}:fd{ti}:{fragment}" if owner
                           else f"fd{ti}:{fragment}"),
                ))
                lane += 1
        return engine.submit(name + ":complete",
                             after=tuple(events) + tuple(fd_events),
                             owner=owner)
