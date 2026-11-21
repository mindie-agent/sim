"""Settled frontiers, measured intervals and skip contracts.

[Role] Define the input evidence and output records for periodic detection.
[Invariants] Cursor positions and phase bounds are explicit. Structural match
  is exact and intervals retain the caller’s measured totals.
[Timeouts & Locks] Bounded validation and slot cache. No I/O.
[Side-Effect Policy] Freezes evidence without changing Engine state.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import math
import sys
from types import MappingProxyType

from .engine_state import split_engine_state as _default_split_engine_state
from .interner import (
    CanonicalState,
    _DEFAULT_INTERNER,
    _STATE_METRICS,
    _canonical_equal,
    _next_state_version,
)
from .snapshots import (
    PeriodicContractError,
    _check_structural,
    _first_difference,
    _freeze_snapshot,
    _numeric_close,
    _require_float,
    _require_int,
    _require_run_id,
    _structural_equal,
)

_ADDITIVE_COUNTERS = (
    "service_by_resource", "busy_by_resource", "deliveries_by_counter", "hbm_accounting",
)
_CURSOR_SLOT_CACHE_ENTRIES = 128


@lru_cache(maxsize=_CURSOR_SLOT_CACHE_ENTRIES)
def _freeze_pair_slots(slots: tuple[int, ...]) -> tuple:
    return ("tuple", tuple(("int", value) for value in slots))


def _state_equal(first: SettledFrontier, second: SettledFrontier) -> bool:
    left = first._canonical
    right = second._canonical
    if left is not None and right is not None:
        return _canonical_equal(left, right)
    if left is None and right is None:
        # History evidence is the frozen view captured at observation, never
        # the public mutable input (which may have been rewritten since).
        left_raw = first._raw_snapshot
        right_raw = second._raw_snapshot
        if left_raw is None or right_raw is None:
            return False
        return _structural_equal(left_raw, right_raw)
    return False


def _state_difference(first: SettledFrontier, second: SettledFrontier) -> str | None:
    left = first._canonical
    right = second._canonical
    if left is not None and right is not None:
        if left.origin == "external" or right.origin == "external":
            return _first_difference(
                (left.template.template, left.dynamic),
                (right.template.template, right.dynamic),
                path="canonical.external",
            )
        if left.template.fingerprint != right.template.fingerprint:
            return "canonical.template_fingerprint"
        if left.template is not right.template and left.template.template != right.template.template:
            return _first_difference(left.template.template, right.template.template, path="canonical.template")
        if left.dynamic != right.dynamic:
            return _first_difference(left.dynamic, right.dynamic, path="canonical.dynamic")
        return None
    if left is None and right is None:
        left_raw = first._raw_snapshot
        right_raw = second._raw_snapshot
        if left_raw is None or right_raw is None:
            return "raw.missing_snapshot"
        return _first_difference(left_raw, right_raw, path="raw")
    return "canonical.mixed_state_form"


def _freeze_map(name: str, value: Mapping, *, value_check) -> Mapping:
    if not isinstance(value, Mapping):
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    frozen = {}
    for key, item in value.items():
        frozen[key] = value_check(key, item)
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class Cursor:
    """One active run's progress and its current structural phase class.

    ``slots`` must describe the live window in a tile-index-independent form
    (for example the sorted live slot offsets).  ``next_issue``/``retired`` are
    absolute counters and are deliberately excluded from the structural key.
    """

    run_id: Hashable
    phase_ordinal: int
    phase_end_exclusive: int
    next_issue: int
    retired: int
    slots: tuple | int

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        _require_int("phase_ordinal", self.phase_ordinal)
        _require_int("phase_end_exclusive", self.phase_end_exclusive)
        _require_int("next_issue", self.next_issue)
        _require_int("retired", self.retired)
        if isinstance(self.slots, bool) or not isinstance(self.slots, (int, tuple)):
            raise PeriodicContractError("periodic_cursor_slots_must_be_int_or_tuple")
        if isinstance(self.slots, int):
            _require_int("cursor_slots", self.slots, minimum=1)
        elif type(self.slots) is tuple and len(self.slots) <= 2 and all(type(slot) is int for slot in self.slots):
            object.__setattr__(self, "slots", _freeze_pair_slots(self.slots))
        else:
            _check_structural(self.slots, path=f"cursor[{self.run_id}].slots")
            object.__setattr__(
                self, "slots",
                _freeze_snapshot(
                    self.slots, path=f"cursor[{self.run_id}].slots"))
        if self.next_issue < self.retired:
            raise PeriodicContractError(f"periodic_cursor_retired_ahead:{self.run_id}")
        # ``phase_end_exclusive`` may legitimately be behind ``next_issue``
        # for one observation at a phase transition.  The bound calculation
        # then yields zero cycles and the next movement stays detailed.


@dataclass(frozen=True)
class SettledFrontier:
    """A fully settled Engine landmark plus the complete structural state.

    ``normalized_state`` is the adapter's exact, tile-index-independent view of
    the *whole* active Engine cohort: all live remaining maps, latency, edges,
    FIFO order, ready/release/completable order, allocation identities with
    pool capacities/units/scope, and any other field that can affect the next
    interval.  It is a tuple of ordinary nested containers only; callables,
    Engine objects and precomputed digests are rejected.
    """

    now: float
    epoch: Hashable
    normalized_state: tuple | CanonicalState
    cursors: tuple[Cursor, ...]
    sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "now", _require_float("now", self.now, allow_infinite=False))
        if not isinstance(self.epoch, (int, str)) or isinstance(self.epoch, bool):
            raise PeriodicContractError(f"periodic_epoch_invalid:{self.epoch!r}")
        state = self.normalized_state
        if isinstance(state, CanonicalState):
            # Trusted/interned canonical input: the template was validated by
            # StateInterner/TemplateRef; only the dynamic residue is checked.
            if state.template.generation != self.epoch:
                raise PeriodicContractError("periodic_canonical_generation_mismatch")
            object.__setattr__(self, "_canonical", state)
            object.__setattr__(self, "_raw_snapshot", None)
            object.__setattr__(self, "_state_form", "canonical")
            _STATE_METRICS["canonical_frames"] += 1
        elif isinstance(state, tuple):
            pkg = sys.modules.get("mindie_sim.prediction.engine_periodic")
            splitter = getattr(pkg, "split_engine_state", _default_split_engine_state) if pkg is not None else _default_split_engine_state
            split = splitter(state)
            canonical = None
            if split is not None:
                static, dynamic = split
                try:
                    template = _DEFAULT_INTERNER.intern_engine_static(
                        static, generation=self.epoch)
                    canonical = CanonicalState(
                        template, dynamic,
                        version=_next_state_version(), origin="engine")
                except PeriodicContractError:
                    # The restricted fast path accepted only known exact
                    # builtin scalar/container types.  Anything else keeps the
                    # generic validated path (which rejects rather than
                    # storing an opaque descendant).
                    canonical = None
            if canonical is not None:
                object.__setattr__(self, "_canonical", canonical)
                object.__setattr__(self, "_raw_snapshot", None)
                object.__setattr__(self, "_state_form", "engine")
                _STATE_METRICS["canonical_frames"] += 1
            else:
                counter = [0]
                _check_structural(state, counter=counter)
                object.__setattr__(self, "_canonical", None)
                object.__setattr__(
                    self, "_raw_snapshot",
                    _freeze_snapshot(state, path="normalized_state"))
                object.__setattr__(self, "_state_form", "raw")
                _STATE_METRICS["raw_frames"] += 1
                _STATE_METRICS["raw_nodes"] += counter[0]
        else:
            raise PeriodicContractError("periodic_normalized_state_must_be_tuple_or_canonical")
        if not isinstance(self.cursors, tuple):
            raise PeriodicContractError("periodic_cursors_must_be_tuple")
        seen: set[Hashable] = set()
        for cursor in self.cursors:
            if not isinstance(cursor, Cursor):
                raise PeriodicContractError("periodic_cursor_type_invalid")
            if cursor.run_id in seen:
                raise PeriodicContractError(f"periodic_duplicate_run_id:{cursor.run_id}")
            seen.add(cursor.run_id)
        _require_int("sequence", self.sequence)

    @property
    def state_form(self) -> str:
        return self._state_form

    @property
    def canonical_state(self) -> CanonicalState | None:
        return self._canonical


@dataclass(frozen=True)
class IntervalDelta:
    """Actual measured progress between two settled frontiers.

    ``issued_retired_by_run`` maps every active ``run_id`` to
    ``(issued_tiles, retired_tiles)`` in this interval.  ``service_by_resource``
    and ``busy_by_resource`` are real resource-unit / millisecond amounts.
    Deliveries and directional HBM accounting accumulate independently.
    Clock steps retain the increments before rounding at each landmark.
    ``per_unit_peaks`` maps a caller-defined pool/unit identity to the peak
    occupancy observed in the interval.  Nothing here is fabricated by the
    detector: the adapter must measure it from the Engine.
    """

    dt: float
    issued_retired_by_run: Mapping
    sequence_count: int
    service_by_resource: Mapping
    busy_by_resource: Mapping
    allocation_event_count: int
    per_unit_peaks: Mapping
    logical_completions: int
    crossed_external_or_opaque: bool
    allocation_grants: int = 0
    allocation_last_uses: int = 0
    deliveries_by_counter: Mapping[str, float] = field(default_factory=dict)
    hbm_accounting: Mapping[str, float] = field(default_factory=dict)
    clock_steps: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt", _require_float("interval_dt", self.dt))
        if self.dt < 0:
            raise PeriodicContractError(f"periodic_interval_dt_invalid:{self.dt!r}")

        def issued_check(key, value):
            _require_run_id(key, name="interval_run_id")
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise PeriodicContractError(f"periodic_interval_issued_invalid:{key!r}")
            return (
                _require_int("issued", value[0]),
                _require_int("retired", value[1]),
            )

        object.__setattr__(
            self,
            "issued_retired_by_run",
            _freeze_map(
                "issued_retired_by_run",
                self.issued_retired_by_run,
                value_check=issued_check,
            ),
        )

        def amount_check(key, value):
            if not isinstance(key, str) or not key:
                raise PeriodicContractError(f"periodic_interval_resource_invalid:{key!r}")
            return _require_float("interval_amount", value)

        for name in _ADDITIVE_COUNTERS:
            object.__setattr__(self, name, _freeze_map(name, getattr(self, name), value_check=amount_check))
        object.__setattr__(self, "clock_steps", tuple(_require_float("clock_step", dt) for dt in self.clock_steps))
        if any(dt < 0 for dt in self.clock_steps):
            raise PeriodicContractError("periodic_clock_step_negative")

        def peak_check(key, value):
            _check_structural(key, path="interval_peak_key")
            return _require_float("interval_peak", value)

        object.__setattr__(
            self,
            "per_unit_peaks",
            _freeze_map("per_unit_peaks", self.per_unit_peaks, value_check=peak_check),
        )
        _require_int("sequence_count", self.sequence_count)
        _require_int("allocation_event_count", self.allocation_event_count)
        _require_int("logical_completions", self.logical_completions)
        _require_int("allocation_grants", self.allocation_grants)
        _require_int("allocation_last_uses", self.allocation_last_uses)
        if not isinstance(self.crossed_external_or_opaque, bool):
            raise PeriodicContractError("periodic_interval_crossed_flag_must_be_bool")


@dataclass(frozen=True)
class RunCorrespondence:
    """Exact counter translation for one run in the skipped cohort."""

    run_id: Hashable
    phase_ordinal: int
    phase_end_exclusive: int
    next_issue_before: int
    next_issue_after: int
    retired_before: int
    retired_after: int
    tile_offset: int
    finished_offset: int

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        for name in (
            "phase_ordinal",
            "phase_end_exclusive",
            "next_issue_before",
            "next_issue_after",
            "retired_before",
            "retired_after",
            "tile_offset",
            "finished_offset",
        ):
            _require_int(name, getattr(self, name))
        if self.next_issue_after != self.next_issue_before + self.tile_offset:
            raise PeriodicContractError(f"periodic_correspondence_issue_mismatch:{self.run_id}")
        if self.retired_after != self.retired_before + self.tile_offset:
            raise PeriodicContractError(f"periodic_correspondence_retire_mismatch:{self.run_id}")


@dataclass(frozen=True)
class WindowTile:
    """One live-window tile whose absolute identity must translate on skip."""

    run_id: Hashable
    old_tile: int
    new_tile: int

    def __post_init__(self) -> None:
        _require_run_id(self.run_id)
        _require_int("old_tile", self.old_tile)
        _require_int("new_tile", self.new_tile)


@dataclass(frozen=True)
class CycleCorrespondence:
    """How live identity maps from the observed frontier to the skipped one.

    Event sequences of already-submitted live tasks are *not* rewritten: they
    keep their object identity and heap positions.  ``event_sequence_offset``
    is the amount by which the Engine's future ``_sequence`` must advance and
    the offset at which the skipped cycle's never-created events would have
    appeared.  ``tile_offset_by_run`` applies to tile-scoped allocation names
    and to ``_periodic_tasks`` metadata; persistent allocation names do not
    move.
    """

    runs: tuple[RunCorrespondence, ...]
    window: tuple[WindowTile, ...]
    event_sequence_offset: int
    tile_offset_by_run: Mapping

    def __post_init__(self) -> None:
        if not isinstance(self.runs, tuple):
            raise PeriodicContractError("periodic_correspondence_runs_invalid")
        if not isinstance(self.window, tuple):
            raise PeriodicContractError("periodic_correspondence_window_invalid")
        seen: set[Hashable] = set()
        for run in self.runs:
            if not isinstance(run, RunCorrespondence):
                raise PeriodicContractError("periodic_correspondence_run_invalid")
            if run.run_id in seen:
                raise PeriodicContractError(f"periodic_correspondence_duplicate_run:{run.run_id}")
            seen.add(run.run_id)
        for tile in self.window:
            if not isinstance(tile, WindowTile) or tile.run_id not in seen:
                raise PeriodicContractError("periodic_correspondence_window_invalid")
            run = next(item for item in self.runs if item.run_id == tile.run_id)
            if tile.new_tile - tile.old_tile != run.tile_offset:
                raise PeriodicContractError(f"periodic_correspondence_window_offset:{tile.run_id}")
        _require_int("event_sequence_offset", self.event_sequence_offset)

        def offset_check(key, value):
            _require_run_id(key, name="correspondence_run_id")
            return _require_int("tile_offset", value)

        object.__setattr__(
            self,
            "tile_offset_by_run",
            _freeze_map("tile_offset_by_run", self.tile_offset_by_run, value_check=offset_check),
        )
        for run in self.runs:
            if self.tile_offset_by_run[run.run_id] != run.tile_offset:
                raise PeriodicContractError(f"periodic_correspondence_offset_mismatch:{run.run_id}")

    def run(self, run_id: Hashable) -> RunCorrespondence:
        for item in self.runs:
            if item.run_id == run_id:
                return item
        raise KeyError(run_id)


@dataclass(frozen=True)
class CycleSkip:
    """A bounded, exact skip description; the adapter applies it explicitly.

    ``delta`` is the measured delta for **one** period.  Use
    :meth:`total_delta` (or multiply by ``cycles``) for accounting the whole
    skip.  ``tile_strides_by_run`` and ``seq_stride`` are also per period.
    """

    cycles: int
    period: float
    tile_strides_by_run: Mapping
    seq_stride: int
    delta: IntervalDelta
    correspondence: CycleCorrespondence

    def __post_init__(self) -> None:
        _require_int("cycles", self.cycles, minimum=1)
        object.__setattr__(self, "period", _require_float("period", self.period, positive=True))

        def stride_check(key, value):
            _require_run_id(key, name="skip_run_id")
            return _require_int("tile_stride", value)

        object.__setattr__(
            self,
            "tile_strides_by_run",
            _freeze_map(
                "tile_strides_by_run",
                self.tile_strides_by_run,
                value_check=stride_check,
            ),
        )
        _require_int("seq_stride", self.seq_stride, minimum=1)
        if not isinstance(self.delta, IntervalDelta):
            raise PeriodicContractError("periodic_skip_delta_invalid")
        if not isinstance(self.correspondence, CycleCorrespondence):
            raise PeriodicContractError("periodic_skip_correspondence_invalid")
        if not _numeric_close(self.delta.dt, self.period):
            raise PeriodicContractError("periodic_skip_delta_period_mismatch")
        if set(self.tile_strides_by_run) != {run.run_id for run in self.correspondence.runs}:
            raise PeriodicContractError("periodic_skip_correspondence_cohort")
        if not any(self.tile_strides_by_run.values()):
            raise PeriodicContractError("periodic_skip_without_tile_progress")
        if self.correspondence.event_sequence_offset != (self.cycles * self.seq_stride):
            raise PeriodicContractError("periodic_skip_sequence_offset_mismatch")
        for run in self.correspondence.runs:
            stride = self.tile_strides_by_run[run.run_id]
            if run.tile_offset != self.cycles * stride:
                raise PeriodicContractError(f"periodic_skip_tile_offset_mismatch:{run.run_id}")

    def total_delta(self) -> IntervalDelta:
        """Measured per-period delta multiplied by the exact skip count."""
        scale = self.cycles
        issued = {
            run_id: (issued * scale, retired * scale)
            for run_id, (issued, retired) in self.delta.issued_retired_by_run.items()
        }
        return IntervalDelta(
            dt=self.period * scale,
            issued_retired_by_run=issued,
            sequence_count=self.seq_stride * scale,
            allocation_event_count=self.delta.allocation_event_count * scale,
            per_unit_peaks=dict(self.delta.per_unit_peaks),
            logical_completions=self.delta.logical_completions * scale,
            crossed_external_or_opaque=False,
            allocation_grants=self.delta.allocation_grants * scale,
            allocation_last_uses=self.delta.allocation_last_uses * scale,
            **{field: {name: amount * scale for name, amount in getattr(self.delta, field).items()}
               for field in _ADDITIVE_COUNTERS},
        )


def _advance_periodic_clock(now: float, steps: tuple[float, ...], cycles: int) -> float:
    """Repeat the engine's rounded additions without replaying resource tasks."""
    for _ in range(cycles):
        for dt in steps:
            after = now + dt
            now = math.nextafter(now, math.inf) if dt > 0.0 and after == now else after
    return now
