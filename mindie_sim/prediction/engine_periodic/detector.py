"""Bounded cycle detection over settled Engine frontiers.

[Role] Find consecutive equivalent periods and return a bounded skip.
[Invariants] History retains detached evidence. Measured time, service and
  strides must agree before any skip can be proposed.
[Timeouts & Locks] History has observation and byte limits. No locks or I/O.
[Side-Effect Policy] Mutates only detector history and diagnostics. The caller applies skips.
"""
from __future__ import annotations

import math
import time
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

from .frontier import (
    CycleCorrespondence,
    CycleSkip,
    IntervalDelta,
    RunCorrespondence,
    SettledFrontier,
    WindowTile,
    _ADDITIVE_COUNTERS,
    _advance_periodic_clock,
    _state_difference,
    _state_equal,
)
from .interner import CanonicalState, _DEFAULT_INTERNER, metrics
from .snapshots import (
    PeriodicContractError,
    _ULP_SLOP,
    _clock_rel_tol,
    _measure_retained,
    _numeric_close,
    _require_float,
    _require_int,
    _structural_equal,
)


_DEFAULT_HISTORY_BYTES = 8 * 1024 * 1024
_FRAME_OVERHEAD = 128
_CURSOR_OVERHEAD = 96
_INTERVAL_OVERHEAD = 128
_OBSERVATION_OVERHEAD = 64
_DEFAULT_MAX_LAG = 24


@dataclass(frozen=True)
class _HistoryFrame:
    """Detached history evidence: never retains the public mutable input."""

    now: float
    epoch: Hashable
    cursors: tuple
    sequence: int
    _canonical: CanonicalState | None
    _raw_snapshot: tuple | None
    _state_form: str
    retained_bytes: int


def _history_frame(
    frame: SettledFrontier, *, seen: set[int] | None = None,
) -> _HistoryFrame:
    if seen is None:
        seen = set()
    canonical = frame._canonical
    if canonical is not None:
        # The template is shared with the interner and accounted once there;
        # history pays for the per-frontier dynamic snapshot, identifiers and
        # cursors.  An engine-origin snapshot already charged its own nodes
        # while it was frozen, so no second recursive walk is needed.
        if canonical.origin == "engine":
            owned = canonical.retained_bytes
        else:
            owned = _measure_retained(canonical.dynamic, seen=seen)
    else:
        owned = _measure_retained(frame._raw_snapshot, seen=seen)
    owned += _measure_retained(frame.epoch, seen=seen)
    for cursor in frame.cursors:
        owned += _CURSOR_OVERHEAD
        owned += _measure_retained(cursor.run_id, seen=seen)
        owned += _measure_retained(cursor.slots, seen=seen)
    owned += _FRAME_OVERHEAD
    return _HistoryFrame(
        now=frame.now,
        epoch=frame.epoch,
        cursors=frame.cursors,
        sequence=frame.sequence,
        _canonical=canonical,
        _raw_snapshot=frame._raw_snapshot,
        _state_form=frame._state_form,
        retained_bytes=owned,
    )


def _interval_retained_bytes(
    interval: IntervalDelta | None, *, seen: set[int] | None = None,
) -> int:
    """Charge the whole frozen interval, including every mapping key/value."""
    if interval is None:
        return 0
    if seen is None:
        seen = set()
    owned = _INTERVAL_OVERHEAD
    for name in (
        "issued_retired_by_run",
        *_ADDITIVE_COUNTERS,
        "per_unit_peaks",
        "clock_steps",
    ):
        owned += _measure_retained(getattr(interval, name), seen=seen)
    return owned


@dataclass(frozen=True)
class _Observation:
    frame: _HistoryFrame
    interval: IntervalDelta | None
    retained_bytes: int


@dataclass(frozen=True)
class _CycleEvidence:
    period: float
    tile_strides: Mapping
    retired_strides: Mapping
    seq_stride: int
    delta: IntervalDelta
    clock_scale: float = 0.0


class CycleDetector:
    """Pure detector for the frozen ``CycleDetector.observe`` interface.

    One tile retirement normally spans several settled micro-steps (load,
    compute, store, retire, allocation grant/release), so the full structural
    state repeats at a *lag*, not necessarily on adjacent observations.  The
    detector keeps a bounded rolling history, compares the current frontier
    with older frontiers, and aggregates every real interval across the whole
    candidate period.  It then requires the same lag/period/strides/delta to
    occur for ``evidence_periods`` consecutive cycles.  Phase remaining is
    used only to bound the skip; it is never a structural key.
    """

    def __init__(
        self,
        *,
        evidence_periods: int = 2,
        max_lag: int = _DEFAULT_MAX_LAG,
        history_limit: int | None = None,
        history_bytes_limit: int = _DEFAULT_HISTORY_BYTES,
    ) -> None:
        self.evidence_periods = _require_int("evidence_periods", evidence_periods, minimum=1)
        self.max_lag = _require_int("max_lag", max_lag, minimum=1)
        minimum_history = (self.evidence_periods + 1) * self.max_lag
        if history_limit is None:
            history_limit = minimum_history
        self.history_limit = _require_int("history_limit", history_limit, minimum=minimum_history)
        self.history_bytes_limit = _require_int(
            "history_bytes_limit", history_bytes_limit, minimum=1)
        self._observations: list[_Observation] = []
        self._history_bytes = 0
        self._history_templates: dict[int, list] = {}
        self._last_rejection = "initial"
        self._last_difference: str | None = None
        self._last_difference_pair: tuple[_HistoryFrame, _HistoryFrame] | None = None
        self._last_skip: CycleSkip | None = None
        self.observations_seen = 0
        self.skips_returned = 0
        self._stats = {
            "observations": 0,
            "skips": 0,
            "skipped_cycles": 0,
            "candidate_lags": 0,
            "fast_state_compares": 0,
            "exact_state_compares": 0,
            "self_s": 0.0,
            "rejections": {},
        }

    @property
    def last_rejection(self) -> str:
        """Why the most recent observation did not return a skip."""
        return self._last_rejection

    def _materialize_last_difference(self) -> None:
        pair = self._last_difference_pair
        if pair is not None:
            self._last_difference = _state_difference(pair[0], pair[1])
            self._last_difference_pair = None

    @property
    def last_difference(self) -> str | None:
        """First unequal structural component of the latest adjacent frontier."""
        self._materialize_last_difference()
        return self._last_difference

    @property
    def last_skip(self) -> CycleSkip | None:
        return self._last_skip

    @property
    def observations(self) -> int:
        return len(self._observations)

    def _set_rejection(self, reason: str) -> None:
        self._last_rejection = reason
        counts = self._stats["rejections"]
        counts[reason] = counts.get(reason, 0) + 1

    def stats(self) -> dict:
        """Return bounded detector counters and structural-cache evidence."""
        result = dict(self._stats)
        result["rejections"] = dict(self._stats["rejections"])
        result["observations_in_history"] = len(self._observations)
        result["history_bytes"] = self._history_bytes
        result["history_bytes_limit"] = self.history_bytes_limit
        result["history_templates"] = len(self._history_templates)
        result["last_rejection"] = self._last_rejection
        result["interner"] = _DEFAULT_INTERNER.stats()
        result["state_metrics"] = metrics()
        result["helper_self_s"] = result["self_s"]
        result["frames"] = result["observations"]
        result["candidate_count"] = result["candidate_lags"]
        result["rejection_reasons"] = dict(self._stats["rejections"])
        return result

    def reset_stats(self) -> None:
        for key in (
            "observations",
            "skips",
            "skipped_cycles",
            "candidate_lags",
            "fast_state_compares",
            "exact_state_compares",
        ):
            self._stats[key] = 0
        self._stats["self_s"] = 0.0
        self._stats["rejections"] = {}

    def reset(self) -> None:
        """Forget the private history; the next observe becomes a baseline."""
        self._materialize_last_difference()
        self._clear_history()
        self._last_rejection = "reset"

    def _clear_history(self) -> None:
        self._observations.clear()
        self._history_bytes = 0
        self._history_templates.clear()

    def _retain_template(self, frame: _HistoryFrame) -> None:
        canonical = frame._canonical
        if canonical is None:
            return
        template = canonical.template
        key = id(template)
        entry = self._history_templates.get(key)
        if entry is None:
            self._history_templates[key] = [template, 1, template.retained_bytes]
            self._history_bytes += template.retained_bytes
        else:
            entry[1] += 1

    def _release_template(self, frame: _HistoryFrame) -> None:
        canonical = frame._canonical
        if canonical is None:
            return
        template = canonical.template
        key = id(template)
        entry = self._history_templates.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            self._history_bytes -= entry[2]
            del self._history_templates[key]

    def _store_observation(
        self, frame: SettledFrontier, interval: IntervalDelta | None,
    ) -> bool:
        seen: set[int] = set()
        history = _history_frame(frame, seen=seen)
        owned = (
            history.retained_bytes
            + _interval_retained_bytes(interval, seen=seen)
            + _OBSERVATION_OVERHEAD
        )
        self._observations.append(_Observation(history, interval, owned))
        self._history_bytes += owned
        self._retain_template(history)
        return self._trim()

    def _trim(self) -> bool:
        """Trim by item count and retained bytes; ``False`` means no baseline."""
        while self._observations and (
            len(self._observations) > self.history_limit
            or self._history_bytes > self.history_bytes_limit
        ):
            if len(self._observations) <= 1:
                # The current exact representation alone does not fit the
                # caller's remaining budget.  Drop the baseline rather than
                # retaining an unaccounted tree or approximating it.
                self._clear_history()
                return False
            removed = self._observations.pop(0)
            pair = self._last_difference_pair
            if pair is not None and (removed.frame is pair[0] or removed.frame is pair[1]):
                self._materialize_last_difference()
            self._history_bytes -= removed.retained_bytes
            self._release_template(removed.frame)
        return True

    def observe(
        self,
        frame: SettledFrontier,
        interval: IntervalDelta | None,
        *,
        stop_at: float,
    ) -> CycleSkip | None:
        """Frozen call boundary; records self time and delegates."""
        started = time.perf_counter()
        self._stats["observations"] += 1
        try:
            return self._observe(frame, interval, stop_at=stop_at)
        finally:
            self._stats["self_s"] += time.perf_counter() - started

    def _observe(
        self,
        frame: SettledFrontier,
        interval: IntervalDelta | None,
        *,
        stop_at: float,
    ) -> CycleSkip | None:
        """Observe one settled landmark; return a skip only when proven.

        ``stop_at`` is the adapter's actual safe boundary (horizon, pending
        external release or another true external event).  It is not a target
        the detector may invent; ``math.inf`` is accepted when the only bound
        is phase remaining.  A ``None`` interval is allowed for the initial
        landmark or a caller that cannot measure a zero-time step, but no
        candidate period may contain one.
        """
        if not isinstance(frame, SettledFrontier):
            raise PeriodicContractError("periodic_observe_requires_frontier")
        if interval is not None and not isinstance(interval, IntervalDelta):
            raise PeriodicContractError("periodic_observe_interval_invalid")
        stop = _require_float("stop_at", stop_at, allow_infinite=True)
        self.observations_seen += 1
        self._last_skip = None

        reset_reason = None
        if self._observations and self._observations[-1].frame.epoch != frame.epoch:
            self._clear_history()
            reset_reason = "epoch_changed"
        elif self._observations and not self._same_phase_class(self._observations[-1].frame, frame):
            # A phase/wave boundary is a real boundary.  Drop old candidates
            # immediately instead of holding complete frontiers from a prior
            # phase class in the bounded cache.
            self._clear_history()
            reset_reason = "phase_changed"
        prev_frame = (
            self._observations[-1].frame
            if self._observations and interval is not None
            else None
        )
        self._last_difference = None
        self._last_difference_pair = None
        if interval is not None and interval.crossed_external_or_opaque:
            self._clear_history()
            stored = self._store_observation(frame, interval)
            if prev_frame is not None and stored and self._observations:
                self._last_difference = _state_difference(prev_frame, self._observations[-1].frame)
            elif prev_frame is not None:
                self._last_difference = _state_difference(prev_frame, frame)
            self._set_rejection("external_or_opaque")
            return None
        if not self._store_observation(frame, interval):
            self._last_difference = None
            self._last_difference_pair = None
            self._set_rejection("history_budget")
            return None
        if prev_frame is not None:
            if len(self._observations) >= 2 and self._observations[-2].frame is prev_frame:
                self._last_difference_pair = (prev_frame, self._observations[-1].frame)
            else:
                self._last_difference = _state_difference(prev_frame, self._observations[-1].frame)

        evidence = self._find_evidence()
        if evidence is None:
            if reset_reason is not None:
                self._set_rejection(reset_reason)
            elif len(self._observations) < self.evidence_periods + 1:
                self._set_rejection("baseline")
            else:
                self._set_rejection("no_repeated_span")
            return None

        cycles = self._bounded_cycles(frame, evidence, stop)
        if cycles < 1:
            self._set_rejection("boundary")
            return None

        skip = self._build_skip(frame, evidence, cycles)
        self.reset()
        self._last_skip = skip
        self._set_rejection("skip")
        self.skips_returned += 1
        self._stats["skips"] += 1
        self._stats["skipped_cycles"] += cycles
        return skip

    def _find_evidence(self) -> _CycleEvidence | None:
        count = len(self._observations)
        max_lag = min(self.max_lag, (count - 1) // self.evidence_periods)
        for lag in range(1, max_lag + 1):
            indices = [count - 1 - step * lag for step in range(self.evidence_periods, -1, -1)]
            if indices[0] < 0:
                continue
            frames = [self._observations[index].frame for index in indices]
            if not all(self._same_class(frames[0], other) for other in frames[1:]):
                continue
            spans = [self._aggregate(left, right) for left, right in zip(indices, indices[1:])]
            if any(span is None for span in spans):
                continue
            first = spans[0]
            if not all(self._evidence_equal(first, span) for span in spans[1:]):
                continue
            self._stats["candidate_lags"] += 1
            return spans[-1]
        return None

    @staticmethod
    def _same_phase_class(
        first: SettledFrontier,
        second: SettledFrontier,
    ) -> bool:
        left = {item.run_id: item for item in first.cursors}
        right = {item.run_id: item for item in second.cursors}
        if left.keys() != right.keys():
            return False
        return all(
            left[run_id].phase_ordinal == right[run_id].phase_ordinal
            and left[run_id].phase_end_exclusive == right[run_id].phase_end_exclusive
            for run_id in left
        )

    def _same_class(self, first: SettledFrontier, second: SettledFrontier) -> bool:
        if first.epoch != second.epoch:
            return False
        left_canonical = first._canonical
        right_canonical = second._canonical
        if (
            left_canonical is not None
            and right_canonical is not None
            and left_canonical.origin != "external"
            and right_canonical.origin != "external"
            and left_canonical.template is right_canonical.template
        ):
            self._stats["fast_state_compares"] += 1
        else:
            self._stats["exact_state_compares"] += 1
        if not _state_equal(first, second):
            return False
        left = {item.run_id: item for item in first.cursors}
        right = {item.run_id: item for item in second.cursors}
        if left.keys() != right.keys():
            return False
        for run_id, cursor in left.items():
            other = right[run_id]
            if cursor.phase_ordinal != other.phase_ordinal or cursor.phase_end_exclusive != other.phase_end_exclusive:
                return False
            if not _structural_equal(cursor.slots, other.slots):
                return False
        return True

    def _aggregate(
        self,
        start_index: int,
        end_index: int,
    ) -> _CycleEvidence | None:
        start = self._observations[start_index].frame
        end = self._observations[end_index].frame
        parts = [self._observations[index].interval for index in range(start_index + 1, end_index + 1)]
        if any(part is None for part in parts):
            return None
        if any(part.crossed_external_or_opaque for part in parts):
            return None
        period = end.now - start.now
        if not math.isfinite(period) or period <= 0:
            return None
        clock_scale = max(abs(start.now), abs(end.now))
        step_slop = max(_ULP_SLOP, len(parts) * 2)
        measured_dt = math.fsum(part.dt for part in parts)
        if not _numeric_close(measured_dt, period, clock_scale=clock_scale, slop=step_slop):
            return None

        start_cursors = {item.run_id: item for item in start.cursors}
        end_cursors = {item.run_id: item for item in end.cursors}
        if start_cursors.keys() != end_cursors.keys():
            return None
        issued: dict[Hashable, int] = {run_id: 0 for run_id in start_cursors}
        retired: dict[Hashable, int] = {run_id: 0 for run_id in start_cursors}
        for part in parts:
            if set(part.issued_retired_by_run) != set(issued):
                return None
            for run_id, (
                part_issued,
                part_retired,
            ) in part.issued_retired_by_run.items():
                issued[run_id] += part_issued
                retired[run_id] += part_retired
        for run_id, cursor in end_cursors.items():
            before = start_cursors[run_id]
            stride = cursor.next_issue - before.next_issue
            retire_stride = cursor.retired - before.retired
            if stride < 0 or retire_stride < 0:
                return None
            if (issued[run_id], retired[run_id]) != (stride, retire_stride):
                return None

        seq_stride = end.sequence - start.sequence
        sequence_count = sum(part.sequence_count for part in parts)
        if seq_stride < 0 or sequence_count != seq_stride:
            return None
        if not any(issued.values()) or seq_stride == 0:
            return None

        counters = {}
        for field in _ADDITIVE_COUNTERS:
            amounts: dict[str, list[float]] = {}
            for part in parts:
                for name, amount in getattr(part, field).items():
                    amounts.setdefault(name, []).append(amount)
            counters[field] = {name: math.fsum(values) for name, values in amounts.items()}
        clock_steps = tuple(dt for part in parts for dt in part.clock_steps)
        if clock_steps and not _numeric_close(
            _advance_periodic_clock(start.now, clock_steps, 1),
            end.now,
            clock_scale=clock_scale,
            slop=max(_ULP_SLOP, len(clock_steps) * 2),
        ):
            return None
        peaks: dict[Any, float] = {}
        for part in parts:
            for key, value in part.per_unit_peaks.items():
                peaks[key] = max(peaks.get(key, 0.0), value)
        aggregate_delta = IntervalDelta(
            dt=period,
            issued_retired_by_run={run_id: (issued[run_id], retired[run_id]) for run_id in issued},
            sequence_count=seq_stride,
            allocation_event_count=sum(part.allocation_event_count for part in parts),
            per_unit_peaks=peaks,
            logical_completions=sum(part.logical_completions for part in parts),
            crossed_external_or_opaque=False,
            allocation_grants=sum(part.allocation_grants for part in parts),
            allocation_last_uses=sum(part.allocation_last_uses for part in parts),
            clock_steps=clock_steps,
            **counters,
        )
        return _CycleEvidence(
            period=period,
            tile_strides=dict(issued),
            retired_strides=dict(retired),
            seq_stride=seq_stride,
            delta=aggregate_delta,
            clock_scale=clock_scale,
        )

    @staticmethod
    def _evidence_equal(
        first: _CycleEvidence,
        second: _CycleEvidence,
    ) -> bool:
        clock_scale = max(first.clock_scale, second.clock_scale)
        step_slop = max(
            _ULP_SLOP,
            max(len(first.delta.clock_steps), len(second.delta.clock_steps)) * 2,
        )
        if not _numeric_close(
            first.period, second.period, clock_scale=clock_scale, slop=step_slop,
        ):
            return False
        if dict(first.tile_strides) != dict(second.tile_strides):
            return False
        if dict(first.retired_strides) != dict(second.retired_strides):
            return False
        if first.seq_stride != second.seq_stride:
            return False
        left, right = first.delta, second.delta
        if not _numeric_close(
            left.dt, right.dt, clock_scale=clock_scale, slop=step_slop,
        ):
            return False
        if dict(left.issued_retired_by_run) != dict(right.issued_retired_by_run):
            return False
        if left.sequence_count != right.sequence_count:
            return False
        if left.allocation_event_count != right.allocation_event_count:
            return False
        if left.logical_completions != right.logical_completions:
            return False
        if left.allocation_grants != right.allocation_grants:
            return False
        if left.allocation_last_uses != right.allocation_last_uses:
            return False
        if left.crossed_external_or_opaque or right.crossed_external_or_opaque:
            return False
        if len(left.clock_steps) != len(right.clock_steps):
            return False
        if any(
            not _numeric_close(a, b, clock_scale=clock_scale, slop=step_slop)
            for a, b in zip(left.clock_steps, right.clock_steps)
        ):
            return False
        rel_tol = _clock_rel_tol(
            clock_scale, min(first.period, second.period), slop=step_slop,
        )
        for field in (*_ADDITIVE_COUNTERS, "per_unit_peaks"):
            a, b = getattr(left, field), getattr(right, field)
            use_clock = clock_scale if field == "busy_by_resource" else 0.0
            use_rel = 0.0 if field == "busy_by_resource" else rel_tol
            if a.keys() != b.keys() or any(
                not _numeric_close(
                    a[key], b[key],
                    clock_scale=use_clock, rel_tol=use_rel, slop=step_slop,
                )
                for key in a
            ):
                return False
        return True

    @staticmethod
    def _bounded_cycles(
        frame: SettledFrontier,
        evidence: _CycleEvidence,
        stop_at: float,
    ) -> int:
        if stop_at < frame.now:
            return 0
        if math.isinf(stop_at):
            cycles = math.inf
        else:
            # The report requires ``now + n*period < stop_at``: an exact
            # boundary is left for the original advance so it can process the
            # tie/release/completion in real order.  Floor first, then remove
            # an endpoint that floating addition placed at/after the boundary.
            quotient = (stop_at - frame.now) / evidence.period
            cycles = math.floor(quotient) if math.isfinite(quotient) else 0
            if cycles > 0 and frame.now + cycles * evidence.period >= stop_at:
                cycles -= 1
        for cursor in frame.cursors:
            stride = evidence.tile_strides[cursor.run_id]
            if stride <= 0:
                continue
            remaining = cursor.phase_end_exclusive - cursor.next_issue
            if remaining < 0:
                return 0
            cycles = min(cycles, remaining // stride)
        if not math.isfinite(cycles):
            return 0
        steps = evidence.delta.clock_steps
        if steps and math.isfinite(stop_at) and _advance_periodic_clock(frame.now, steps, cycles) >= stop_at:
            low, high = 0, cycles
            while high - low > 1:
                mid = (low + high) // 2
                if _advance_periodic_clock(frame.now, steps, mid) < stop_at:
                    low = mid
                else:
                    high = mid
            cycles = low
        return int(cycles)

    @staticmethod
    def _build_skip(
        frame: SettledFrontier,
        evidence: _CycleEvidence,
        cycles: int,
    ) -> CycleSkip:
        runs: list[RunCorrespondence] = []
        window: list[WindowTile] = []
        offsets: dict[Hashable, int] = {}
        for cursor in sorted(frame.cursors, key=lambda item: item.run_id):
            stride = evidence.tile_strides[cursor.run_id]
            offset = cycles * stride
            offsets[cursor.run_id] = offset
            runs.append(
                RunCorrespondence(
                    run_id=cursor.run_id,
                    phase_ordinal=cursor.phase_ordinal,
                    phase_end_exclusive=cursor.phase_end_exclusive,
                    next_issue_before=cursor.next_issue,
                    next_issue_after=cursor.next_issue + offset,
                    retired_before=cursor.retired,
                    retired_after=cursor.retired + offset,
                    tile_offset=offset,
                    finished_offset=cycles * evidence.retired_strides[cursor.run_id],
                )
            )
            for old_tile in range(cursor.retired, cursor.next_issue):
                window.append(
                    WindowTile(
                        run_id=cursor.run_id,
                        old_tile=old_tile,
                        new_tile=old_tile + offset,
                    )
                )
        correspondence = CycleCorrespondence(
            runs=tuple(runs),
            window=tuple(window),
            event_sequence_offset=cycles * evidence.seq_stride,
            tile_offset_by_run=offsets,
        )
        return CycleSkip(
            cycles=cycles,
            period=evidence.period,
            tile_strides_by_run=dict(evidence.tile_strides),
            seq_stride=evidence.seq_stride,
            delta=evidence.delta,
            correspondence=correspondence,
        )
