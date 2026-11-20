"""Five-point WaveRecorder and exact 4-bucket idle decomposition for Engine runs.

[Role] Observe Engine execution across the 5 canonical points (submit/coalesce,
  admission, positive-rate service, allocation grant/release, proved periodic
  skip) and emit ``FusedPipelineDescriptor`` with exact integer-ns 4-bucket idle
  accounting (``fill``, ``steady_bubble``, ``drain``, ``handoff``).
[Invariants] For every physical ``PipeIdleRow`` on window ``I = [t0, t1)``,
  ``busy_ns + fill_ns + steady_bubble_ns + drain_ns + handoff_ns == span_ns``
  holds as an exact integer identity. Observer attachment never alters Engine
  scheduling, rates, or completion timestamps. Periodic skips translate pattern
  endpoints from original ``(anchor_ms, period_ms)`` without cumulative rounding
  drift and never paint skipped bubbles as busy. Zero Excel/OOXML/Perfetto deps.
[Timeouts & Locks] Bounded interval union arithmetic; no I/O or locks.
[Side-Effect Policy] Mutates only this recorder's bounded observation buffers.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError
from .wave_types import (
    CUBE_PIPE_ROLES,
    DEFAULT_HEAD_TAIL_LIMIT,
    NS_PER_MS,
    SCHEMA_VERSION,
    SINGLE_CAUSE_HANDOFF_KINDS,
    VECTOR_PIPE_ROLES,
    BottleneckDiagnosis,
    BufferResidency,
    FusedPipelineDescriptor,
    IntervalRun,
    MemberBinding,
    OwnerPoolPeak,
    PeriodicRun,
    PipeIdleRow,
    RecordedPipelineSpec,
    ResourceRowRef,
    TileWaveTimeline,
    WaitWitness,
    _clamp_intervals,
    _intersect_intervals,
    _measure,
    _ms_to_ns,
    _pipe_base_role,
    _subtract_intervals,
    _union_intervals,
)


class WaveRecorder:
    """Five-point bounded observer producing ``FusedPipelineDescriptor``."""

    def __init__(
        self,
        *,
        declared_pipes: Sequence[str] = (),
        scope_id: int = 0,
        max_head_tail: int = DEFAULT_HEAD_TAIL_LIMIT,
    ) -> None:
        self.scope_id = int(scope_id)
        self.max_head_tail = max(1, int(max_head_tail))
        self._pipe_ids: dict[str, int] = {}
        self._pipe_names: dict[int, str] = {}
        for name in declared_pipes:
            self.register_pipe(name)
        self._member_tasks: dict[str, list[int]] = {}
        self._task_occurrences: dict[int, Any] = {}
        self._submitted_tasks: set[int] = set()
        self._admitted: dict[int, tuple[float, tuple[int, ...], str, int, tuple[str, ...]]] = {}
        self._service_by_pipe: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._owner_time_by_pipe: dict[tuple[int, int], int] = {}
        self._handoff_by_pipe: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._wait_witnesses: list[WaitWitness] = []
        self._head: list[IntervalRun] = []
        self._tail: list[IntervalRun] = []
        self._steady: list[PeriodicRun] = []
        self._live_allocs: dict[tuple[str, int, int, int], BufferResidency] = {}
        self._closed_allocs: list[BufferResidency] = []
        self._live_pool_bytes: dict[tuple[int, str], int] = {}
        self._peak_pool_bytes: dict[tuple[int, str], int] = {}
        self._alloc_name_slot_gen: dict[tuple[str, int, str], tuple[int, int]] = {}
        self._alloc_next_gen: dict[tuple[str, int, int], int] = {}
        self._critical_events: list[str] = []
        self._min_ns: int | None = None
        self._max_ns: int | None = None

    def register_pipe(self, pipe_name: str) -> int:
        pid = self._pipe_ids.get(pipe_name)
        if pid is None:
            pid = len(self._pipe_ids)
            self._pipe_ids[pipe_name] = pid
            self._pipe_names[pid] = pipe_name
        return pid

    def allocate_slot_generation(
        self, pool: str, owner: int, alloc_name: str,
    ) -> tuple[int, int]:
        """Return a distinct ``(slot, generation)`` pair for an allocation grant."""
        key = (str(pool), int(owner), str(alloc_name))
        existing = self._alloc_name_slot_gen.get(key)
        if existing is not None:
            return existing
        slot = 0
        if ":slot" in alloc_name:
            tail = alloc_name.rsplit(":slot", 1)[1]
            digits = ""
            for ch in tail:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                slot = int(digits)
        else:
            live_slots = {
                s for (p, o, s, _) in self._live_allocs
                if p == str(pool) and o == int(owner)
            }
            slot = 0
            while slot in live_slots:
                slot += 1
        gkey = (str(pool), int(owner), slot)
        gen = self._alloc_next_gen.get(gkey, 0)
        self._alloc_next_gen[gkey] = gen + 1
        pair = (slot, gen)
        self._alloc_name_slot_gen[key] = pair
        return pair

    def release_slot_generation(
        self, pool: str, owner: int, alloc_name: str,
    ) -> tuple[int, int]:
        """Look up and release the ``(slot, generation)`` assigned at grant."""
        return self._alloc_name_slot_gen.pop((str(pool), int(owner), str(alloc_name)), (0, 0))

    def extract_cycle_pattern(
        self,
        cycle_start_ms: float,
        cycle_end_ms: float,
        *,
        scope_id: int | None = None,
    ) -> tuple[tuple[tuple[str, float, float], ...], tuple[tuple[str, float, float], ...]]:
        """Extract exact relative per-pipe intervals observed in one proved cycle."""
        lo_ns = _ms_to_ns(cycle_start_ms)
        hi_ns = _ms_to_ns(cycle_end_ms)
        if hi_ns <= lo_ns:
            return (), ()
        sid = self.scope_id if scope_id is None else int(scope_id)
        svc_pat: list[tuple[str, float, float]] = []
        ho_pat: list[tuple[str, float, float]] = []
        for pid, pname in sorted(self._pipe_names.items()):
            for a, b in _clamp_intervals(self._service_by_pipe.get((sid, pid), ()), lo_ns, hi_ns):
                svc_pat.append((pname, (a - lo_ns) / NS_PER_MS, (b - lo_ns) / NS_PER_MS))
            for a, b in _clamp_intervals(self._handoff_by_pipe.get((sid, pid), ()), lo_ns, hi_ns):
                ho_pat.append((pname, (a - lo_ns) / NS_PER_MS, (b - lo_ns) / NS_PER_MS))
        return tuple(svc_pat), tuple(ho_pat)

    def _note_bounds(self, lo_ns: int, hi_ns: int) -> None:
        if self._min_ns is None or lo_ns < self._min_ns:
            self._min_ns = lo_ns
        if self._max_ns is None or hi_ns > self._max_ns:
            self._max_ns = hi_ns

    # Point 1: submit / coalesce
    def on_submit(
        self,
        task_id: int,
        stage_id: str,
        aliases: Sequence[str] = (),
        predecessors: Sequence[int] = (),
        signals: Sequence[str] = (),
        occurrence: Any = None,
    ) -> None:
        del predecessors
        tid = int(task_id)
        self._submitted_tasks.add(tid)
        if occurrence is not None:
            self._task_occurrences[tid] = occurrence
        members = tuple(aliases) if aliases else (str(stage_id),)
        for member in members:
            self._member_tasks.setdefault(str(member), []).append(int(task_id))
        for sig in signals:
            if sig and sig not in self._critical_events:
                self._critical_events.append(str(sig))

    # Point 2: admission / blocker transition
    def on_admit(
        self,
        task_id: int,
        t_ms: float,
        owners: Sequence[int] = (),
        stream: str = "main",
        generation: int = 0,
        blockers: Sequence[tuple[str, str, float, str]] = (),
    ) -> None:
        """Record task admission and optional exposed single-cause handoff waits.

        Each ``blockers`` entry is ``(pipe_name, blocker_event, ready_ms, reason)``.
        Only when ``len(blockers) == 1`` and ``reason`` is a genuine cross-pipe
        handoff wait does ``[ready_ms, t_ms)`` enter ``H_0``.
        """
        admit_ns = _ms_to_ns(t_ms)
        self._admitted[int(task_id)] = (
            float(t_ms), tuple(int(o) for o in owners), str(stream), int(generation),
            tuple(b[1] for b in blockers),
        )
        if len(blockers) == 1:
            pipe_name, blocker_ev, ready_ms, reason = blockers[0]
            if reason in SINGLE_CAUSE_HANDOFF_KINDS:
                start_ns = _ms_to_ns(ready_ms)
                if admit_ns > start_ns:
                    pid = self.register_pipe(str(pipe_name))
                    key = (self.scope_id, pid)
                    self._handoff_by_pipe.setdefault(key, []).append((start_ns, admit_ns))
                    self._note_bounds(start_ns, admit_ns)
                    self._wait_witnesses.append(WaitWitness(
                        task_id=int(task_id),
                        blocker_event=str(blocker_ev),
                        pipe_id=pid,
                        start_ns=start_ns,
                        end_ns=admit_ns,
                        reason=str(reason),
                    ))

    # Point 3: positive-rate service
    def on_service(
        self,
        task_id: int,
        physical_row: str,
        owner: int,
        t0_ms: float,
        t1_ms: float,
        delivered: float,
        *,
        scope_id: int | None = None,
    ) -> None:
        if delivered <= 0.0 or t1_ms <= t0_ms:
            return
        lo_ns = _ms_to_ns(t0_ms)
        hi_ns = _ms_to_ns(t1_ms)
        if hi_ns <= lo_ns:
            return
        sid = self.scope_id if scope_id is None else int(scope_id)
        pid = self.register_pipe(str(physical_row))
        key = (sid, pid)
        self._service_by_pipe.setdefault(key, []).append((lo_ns, hi_ns))
        self._owner_time_by_pipe[key] = self._owner_time_by_pipe.get(key, 0) + (hi_ns - lo_ns)
        self._note_bounds(lo_ns, hi_ns)
        run = IntervalRun(
            start_ns=lo_ns,
            end_ns=hi_ns,
            task_id=int(task_id),
            pipe_id=pid,
            scope_id=sid,
            owner=int(owner),
            delivered=float(delivered),
        )
        if len(self._head) < self.max_head_tail:
            self._head.append(run)
        else:
            if len(self._tail) >= self.max_head_tail:
                self._tail.pop(0)
            self._tail.append(run)

    # Point 4: allocation grant / release
    def on_allocation(
        self,
        pool: str,
        owner: int,
        slot: int,
        generation: int,
        size_bytes: int,
        t_ms: float,
        action: Literal["grant", "release"],
        *,
        alias_of: str | None = None,
    ) -> None:
        t_ns = _ms_to_ns(t_ms)
        key = (str(pool), int(owner), int(slot), int(generation))
        pool_key = (int(owner), str(pool))
        if action == "grant":
            self._live_allocs[key] = BufferResidency(
                pool=str(pool),
                owner=int(owner),
                slot=int(slot),
                generation=int(generation),
                size_bytes=int(size_bytes),
                grant_ns=t_ns,
                release_ns=None,
                alias_of=alias_of,
            )
            if alias_of is None:
                cur = self._live_pool_bytes.get(pool_key, 0) + int(size_bytes)
                self._live_pool_bytes[pool_key] = cur
                if cur > self._peak_pool_bytes.get(pool_key, 0):
                    self._peak_pool_bytes[pool_key] = cur
        elif action == "release":
            live = self._live_allocs.pop(key, None)
            if live is not None:
                self._closed_allocs.append(BufferResidency(
                    pool=live.pool,
                    owner=live.owner,
                    slot=live.slot,
                    generation=live.generation,
                    size_bytes=live.size_bytes,
                    grant_ns=live.grant_ns,
                    release_ns=t_ns,
                    alias_of=live.alias_of,
                ))
                if live.alias_of is None:
                    cur = max(0, self._live_pool_bytes.get(pool_key, 0) - live.size_bytes)
                    self._live_pool_bytes[pool_key] = cur

    # Point 5: proved periodic skip (compressed O(1) storage and integration)
    def on_period(
        self,
        anchor_ms: float,
        period_ms: float,
        repeats: int,
        template_id: str,
        state_digest: str,
        *,
        service_pattern: Sequence[tuple[str, float, float]] = (),
        handoff_pattern: Sequence[tuple[str, float, float]] = (),
        scope_id: int | None = None,
    ) -> None:
        """Record a proved periodic skip in O(1) space preserving relative phase offsets."""
        if repeats < 1 or period_ms <= 0.0:
            raise ModelingError("wave_recorder_periodic_skip_invalid")
        sid = self.scope_id if scope_id is None else int(scope_id)
        svc_norm = tuple(
            (str(pn), float(lo), float(hi)) for pn, lo, hi in service_pattern if hi > lo
        )
        ho_norm = tuple(
            (str(pn), float(lo), float(hi)) for pn, lo, hi in handoff_pattern if hi > lo
        )
        for pn, lo, hi in svc_norm:
            pid = self.register_pipe(pn)
            key = (sid, pid)
            self._owner_time_by_pipe[key] = (
                self._owner_time_by_pipe.get(key, 0)
                + int(repeats) * max(0, _ms_to_ns(hi) - _ms_to_ns(lo))
            )
        for pn, _lo, _hi in ho_norm:
            self.register_pipe(pn)
        self._steady.append(PeriodicRun(
            anchor_ms=float(anchor_ms),
            period_ms=float(period_ms),
            repeats=int(repeats),
            template_id=str(template_id),
            state_digest=str(state_digest),
            service_pattern=svc_norm,
            handoff_pattern=ho_norm,
            scope_id=sid,
        ))
        self._note_bounds(
            _ms_to_ns(anchor_ms),
            _ms_to_ns(anchor_ms + float(repeats) * period_ms),
        )

    def _periodic_pipe_contrib(
        self, sid: int, pname: str, start_ns: int, end_ns: int,
    ) -> tuple[int, int, int | None, int | None, list[tuple[int, int]], list[tuple[int, int]]]:
        """Integrate compressed periodic skips for one pipe in O(#runs)."""
        extra_busy = 0
        extra_ho = 0
        first_a: int | None = None
        last_b: int | None = None
        edge_busy: list[tuple[int, int]] = []
        edge_ho: list[tuple[int, int]] = []
        for pr in self._steady:
            if pr.scope_id != sid:
                continue
            period_ns = _ms_to_ns(pr.period_ms)
            if period_ns <= 0:
                continue
            anchor_ns = _ms_to_ns(pr.anchor_ms)
            run_end_ns = _ms_to_ns(pr.anchor_ms + float(pr.repeats) * pr.period_ms)
            if run_end_ns <= start_ns or anchor_ns >= end_ns:
                continue
            rel_b = _union_intervals(
                (_ms_to_ns(lo), _ms_to_ns(hi))
                for pn, lo, hi in pr.service_pattern if pn == pname
            )
            rel_h0 = _union_intervals(
                (_ms_to_ns(lo), _ms_to_ns(hi))
                for pn, lo, hi in pr.handoff_pattern if pn == pname
            )
            rel_h = _subtract_intervals(rel_h0, rel_b)
            if not rel_b and not rel_h:
                continue
            r_lo = max(0, (start_ns - anchor_ns) // period_ns)
            r_hi = min(pr.repeats, (end_ns - anchor_ns + period_ns - 1) // period_ns)
            if r_hi <= r_lo:
                continue
            boundary_reps = {r_lo, r_hi - 1}
            for rep in sorted(boundary_reps):
                base_ns = _ms_to_ns(pr.anchor_ms + float(rep) * pr.period_ms)
                edge_busy.extend((base_ns + a, base_ns + b) for a, b in rel_b)
                edge_ho.extend((base_ns + a, base_ns + b) for a, b in rel_h)
            n_mid = max(0, (r_hi - r_lo) - len(boundary_reps))
            if n_mid > 0:
                extra_busy += n_mid * _measure(rel_b)
                extra_ho += n_mid * _measure(rel_h)
                if rel_b:
                    mid_lo_ns = _ms_to_ns(pr.anchor_ms + float(r_lo + 1) * pr.period_ms) + rel_b[0][0]
                    mid_hi_ns = _ms_to_ns(pr.anchor_ms + float(r_hi - 2) * pr.period_ms) + rel_b[-1][1]
                    if first_a is None or mid_lo_ns < first_a:
                        first_a = max(start_ns, mid_lo_ns)
                    if last_b is None or mid_hi_ns > last_b:
                        last_b = min(end_ns, mid_hi_ns)
        return extra_busy, extra_ho, first_a, last_b, edge_busy, edge_ho

    def _periodic_cv_overlap(self, start_ns: int, end_ns: int) -> int:
        """Compute O(#runs) compressed cube-vector overlap honoring phase offsets."""
        total = 0
        for pr in self._steady:
            period_ns = _ms_to_ns(pr.period_ms)
            if period_ns <= 0:
                continue
            anchor_ns = _ms_to_ns(pr.anchor_ms)
            run_end_ns = _ms_to_ns(pr.anchor_ms + float(pr.repeats) * pr.period_ms)
            if run_end_ns <= start_ns or anchor_ns >= end_ns:
                continue
            rel_c = _union_intervals(
                (_ms_to_ns(lo), _ms_to_ns(hi))
                for pn, lo, hi in pr.service_pattern if _pipe_base_role(pn) in CUBE_PIPE_ROLES
            )
            rel_v = _union_intervals(
                (_ms_to_ns(lo), _ms_to_ns(hi))
                for pn, lo, hi in pr.service_pattern if _pipe_base_role(pn) in VECTOR_PIPE_ROLES
            )
            ov_rel = _intersect_intervals(rel_c, rel_v)
            if not ov_rel:
                continue
            r_lo = max(0, (start_ns - anchor_ns) // period_ns)
            r_hi = min(pr.repeats, (end_ns - anchor_ns + period_ns - 1) // period_ns)
            if r_hi <= r_lo:
                continue
            boundary_reps = {r_lo, r_hi - 1}
            for rep in sorted(boundary_reps):
                base_ns = _ms_to_ns(pr.anchor_ms + float(rep) * pr.period_ms)
                total += _measure(_clamp_intervals([(base_ns + a, base_ns + b) for a, b in ov_rel], start_ns, end_ns))
            n_mid = max(0, (r_hi - r_lo) - len(boundary_reps))
            if n_mid > 0:
                total += n_mid * _measure(ov_rel)
        return total

    def finish(
        self,
        spec: Any,
        occurrence_key: Any,
        *,
        window_start_ms: float | None = None,
        window_end_ms: float | None = None,
        semantic_group_id: str = "",
        timing_scope: Literal["graph_occurrence", "isolated_occurrence"] = "graph_occurrence",
        submitted_kernel_count: int | None = None,
        uncovered_stars_ns: int = 0,
    ) -> FusedPipelineDescriptor:
        """Compute exact 4-bucket idle rows and return ``FusedPipelineDescriptor``."""
        start_ns = (
            _ms_to_ns(window_start_ms)
            if window_start_ms is not None else (self._min_ns or 0)
        )
        end_ns = (
            _ms_to_ns(window_end_ms)
            if window_end_ms is not None else (self._max_ns if self._max_ns is not None else start_ns)
        )
        if end_ns < start_ns:
            raise ModelingError(f"wave_recorder_window_reversed:{start_ns}>{end_ns}")
        span_ns = end_ns - start_ns
        window_iv = ((start_ns, end_ns),) if span_ns > 0 else ()

        all_keys = sorted(
            {(self.scope_id, pid) for pid in self._pipe_names}
            | set(self._service_by_pipe)
            | set(self._handoff_by_pipe)
        )
        pipe_rows: list[PipeIdleRow] = []
        cube_intervals: list[tuple[int, int]] = []
        vec_intervals: list[tuple[int, int]] = []
        limiting: list[ResourceRowRef] = []
        max_busy = 0

        for sid, pid in all_keys:
            pname = self._pipe_names.get(pid, f"pipe_{pid}")
            p_busy, p_ho, p_fa, p_lb, eb, eh = self._periodic_pipe_contrib(
                sid, pname, start_ns, end_ns,
            )
            busy_set = _clamp_intervals((*self._service_by_pipe.get((sid, pid), ()), *eb), start_ns, end_ns)
            h0_set = _clamp_intervals((*self._handoff_by_pipe.get((sid, pid), ()), *eh), start_ns, end_ns)
            handoff_set = _subtract_intervals(h0_set, busy_set)
            covered = _union_intervals((*busy_set, *handoff_set))
            idle_j = _subtract_intervals(window_iv, covered)

            busy_ns = _measure(busy_set) + p_busy
            handoff_ns = _measure(handoff_set) + p_ho
            first_a = busy_set[0][0] if busy_set else p_fa
            if p_fa is not None and (first_a is None or p_fa < first_a):
                first_a = p_fa
            last_b = busy_set[-1][1] if busy_set else p_lb
            if p_lb is not None and (last_b is None or p_lb > last_b):
                last_b = p_lb
            if first_a is not None and last_b is not None:
                fill_ns = _measure(_clamp_intervals(idle_j, start_ns, first_a))
                drain_ns = _measure(_clamp_intervals(idle_j, last_b, end_ns))
                steady_bubble_ns = max(0, span_ns - busy_ns - handoff_ns - fill_ns - drain_ns)
                inactive = False
            else:
                fill_ns = max(0, span_ns - handoff_ns)
                steady_bubble_ns = 0
                drain_ns = 0
                inactive = True

            pipe_rows.append(PipeIdleRow(
                scope_id=sid,
                pipe_id=pid,
                span_ns=span_ns,
                busy_ns=busy_ns,
                fill_ns=fill_ns,
                steady_bubble_ns=steady_bubble_ns,
                drain_ns=drain_ns,
                handoff_ns=handoff_ns,
                inactive_in_window=inactive,
                pipe_name=pname,
                owner_time_ns=self._owner_time_by_pipe.get((sid, pid), 0),
            ))
            role = _pipe_base_role(pname)
            if role in CUBE_PIPE_ROLES:
                cube_intervals.extend(busy_set)
            elif role in VECTOR_PIPE_ROLES:
                vec_intervals.extend(busy_set)
            if busy_ns > max_busy:
                max_busy = busy_ns

        for row in pipe_rows:
            if max_busy > 0 and row.busy_ns == max_busy:
                limiting.append(ResourceRowRef(row.pipe_id, row.pipe_name, row.scope_id))

        cube_union = _union_intervals(cube_intervals)
        vec_union = _union_intervals(vec_intervals)
        overlap_ns = _measure(_intersect_intervals(cube_union, vec_union)) + self._periodic_cv_overlap(start_ns, end_ns)

        timeline = TileWaveTimeline(
            start_ns=start_ns,
            end_ns=end_ns,
            head=tuple(self._head),
            steady=tuple(self._steady),
            tail=tuple(self._tail),
            pipe_rows=tuple(pipe_rows),
            cube_vector_overlap_ns=overlap_ns,
        )
        peaks = tuple(
            OwnerPoolPeak(owner=owner, pool=pool, peak_bytes=peak)
            for (owner, pool), peak in sorted(self._peak_pool_bytes.items())
        )
        diagnosis = BottleneckDiagnosis(
            limiting_rows=tuple(limiting),
            exposed_waits=tuple(self._wait_witnesses),
            peak_allocations=peaks,
            critical_events=tuple(self._critical_events),
            uncovered_stars_ns=int(uncovered_stars_ns),
        )
        bindings = tuple(
            MemberBinding(member_id=mid, task_ids=tuple(tids))
            for mid, tids in sorted(self._member_tasks.items())
        )
        buffers = tuple((*self._closed_allocs, *self._live_allocs.values()))
        kernel_cnt = (
            len(self._submitted_tasks)
            if submitted_kernel_count is None else int(submitted_kernel_count)
        )
        return FusedPipelineDescriptor(
            spec=spec,
            occurrence_key=occurrence_key,
            semantic_group_id=semantic_group_id or getattr(spec, "spec_id", ""),
            member_bindings=bindings,
            timing_scope=timing_scope,
            submitted_kernel_count=kernel_cnt,
            buffers=buffers,
            timeline=timeline,
            diagnosis=diagnosis,
        )


__all__ = [
    "BottleneckDiagnosis",
    "BufferResidency",
    "FusedPipelineDescriptor",
    "IntervalRun",
    "MemberBinding",
    "NS_PER_MS",
    "OwnerPoolPeak",
    "PeriodicRun",
    "PipeIdleRow",
    "RecordedPipelineSpec",
    "ResourceRowRef",
    "SCHEMA_VERSION",
    "TileWaveTimeline",
    "WaitWitness",
    "WaveRecorder",
]
