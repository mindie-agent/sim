"""Bounded tile pipelines, finite phase templates, and interior folding.

[Role] Define ``PipelineProgram``, ``PipelinePhase``, ``PipelineTemplate``,
  ``PipelineTemplateProgram``, indexed-consumer ABI checks, and exact strictly-
  serial interior tile folding on the shared ``Engine`` contract.
[Invariants] Repeated tiles execute through a slot-bounded window including
  fill and drain without replacing finite work with steady-state II. Indexed
  templates validate every declared binding symbol and coordinate tuple.
[Timeouts & Locks] Bounded tile materialization capped by
  ``PIPELINE_COHORT_WINDOW`` and ``PIPELINE_FOLD_WINDOW``; no I/O or locks.
[Side-Effect Policy] Submits tasks only to the caller's ``Engine`` instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from ..engine import Allocation, Engine, Event
from .stages import Stage, _declares_no_work, scope_allocations

#: Bounded tile window a symmetric cohort may materialize to prove identity.
PIPELINE_COHORT_WINDOW = 512
#: Bounded tile window an interior fold may materialize.
PIPELINE_FOLD_WINDOW = 512
#: Bounded tile count eligible for callback-free static DAG submission.
_STATIC_UNROLL_MAX_TILES = 128
_MIN_STATIC_UNROLL_TILES = 2
_INDEX_COORD_ARITY = 5
_MIN_FOLD_TILES = 4
_MIN_FOLD_ENTRIES = 3


def _allocation_sig(items):
    return tuple(sorted((item.name, item.pool, item.size, tuple(item.units)) for item in items))


def _fold_stage_key(stage, *, fair_resources=None):
    """Exact physical identity of one single-resource allocation-free repeated stage."""
    if stage.program is not None or stage.fragment is not None:
        return None
    if (
        stage.latency > 0.0 or stage.coupled
        or stage.latency_from_first_service or stage.previous_lags
        or stage.deliveries or stage.keep_empty
        or stage.acquire or stage.free
        or stage.cohort_size > 1 or stage.cohort_widened or stage.cohort_members
        or len(stage.work) > 1
    ):
        return None
    if stage.work:
        if fair_resources is None:
            return None
        rname = next(iter(stage.work))
        if rname not in fair_resources:
            return None
    return (
        stage.name, stage.after, stage.previous, stage.priority,
        stage.cohort_size, stage.cohort_widened,
        tuple(sorted(stage.cohort_members.items())),
        tuple(sorted(stage.work.items())),
        tuple(sorted(stage.caps.items())),
        tuple((key, tuple(units)) for key, units in stage.placement.items()),
        tuple(sorted(stage.work_units.items())),
        tuple(sorted(stage.metadata.items())),
        _allocation_sig(stage.acquire),
        _allocation_sig(stage.free),
    )


def _fold_tile_key(stages, *, fair_resources=None):
    # Multi-stage tiles (e.g. load -> compute) cannot batch across tile
    # boundaries without shifting inter-stage overlap against concurrent peers.
    if len(stages) != 1:
        return None
    key = _fold_stage_key(stages[0], fair_resources=fair_resources)
    return None if key is None else (key,)


def _fold_scaled(stages, repeat: int):
    """One interior super-tile: identical stages whose work is the run's sum."""
    return tuple(
        replace(stage, work={key: value * repeat for key, value in stage.work.items()})
        for stage in stages
    )


def interior_fold_plan(program: PipelineProgram, *, fair_resources=None):
    """Return ``[(representative index, repeat, stages)]`` or ``None``."""
    count, slots = program.count, program.slots
    if program.homogeneous or count < _MIN_FOLD_TILES or slots != 1:
        return None
    if count > PIPELINE_FOLD_WINDOW:
        return None
    tiles = [tuple(program.tile(index)) for index in range(count)]
    if any(len(stages) != 1 for stages in tiles):
        return None
    keys = [_fold_tile_key(stages, fair_resources=fair_resources) for stages in tiles]
    if any(key is None for key in keys):
        return None
    period = None
    repeat = 0
    for candidate in range(1, min(slots, count - 1) + 1):
        matched = 0
        while matched + candidate < count and keys[matched] == keys[matched + candidate]:
            matched += 1
        blocks = (matched + candidate) // candidate
        if blocks < 2:
            continue
        if period is None or blocks * candidate > repeat * period:
            period, repeat = candidate, blocks
    if period is None:
        return None
    folded = [
        (residue, repeat, _fold_scaled(tiles[residue], repeat))
        for residue in range(period)
    ]
    for index in range(repeat * period, count):
        folded.append((index, 1, tiles[index]))
    if len(folded) < _MIN_FOLD_ENTRIES:
        return None
    return folded


def fold_interior_pipeline(program: PipelineProgram, *, fair_resources=None):
    """Return an exactly-folded equivalent program, or ``None``."""
    plan = interior_fold_plan(program, fair_resources=fair_resources)
    if plan is None:
        return None
    stages = tuple(entry[2] for entry in plan)

    def tile(index: int, stages=stages):
        return stages[index]

    return replace(program, count=len(stages), tile=tile)


@dataclass(frozen=True)
class PipelineProgram:
    """A repeated stage DAG with explicit stream ordering and buffer lifetime."""

    count: int
    slots: int
    tile: Callable[[int], Sequence[Stage]]
    minimum_ms: float = 0.0
    minimum_provenance: Mapping = field(default_factory=dict)
    setup_ms: float = 0.0
    homogeneous: bool = False
    producer_bindings: object = None
    interior_fold: bool = False
    template_start: bool = True

    def template(
        self,
        *,
        owner: int = 0,
        panels_per_output: int = 1,
        k_tail: bool = False,
        key_of=None,
        bindings=(),
    ) -> PipelineTemplate:
        """Build the exact finite phase sequence for this callback."""
        if not self.homogeneous:
            raise ModelingError("pipeline_template_requires_declared_homogeneity")
        if (
            isinstance(panels_per_output, bool)
            or not isinstance(panels_per_output, int)
            or panels_per_output < 1
        ):
            raise ValueError("pipeline_template_requires_positive_panels_per_output")
        if isinstance(self.count, bool) or self.count < 0 or self.count % panels_per_output:
            raise ValueError("pipeline_template_count_requires_whole_output_tiles")
        phases = []
        tile_calls = 0
        waves = self.count // panels_per_output
        for wave in range(waves):
            base = wave * panels_per_output
            key = tuple(key_of(base)) if key_of is not None else ()
            if panels_per_output == 1:
                stages = tuple(self.tile(base))
                tile_calls += 1
                phases.append(PipelinePhase("tile", stages, 1, key=key))
                continue
            stages = tuple(self.tile(base))
            tile_calls += 1
            phases.append(PipelinePhase("first", stages, 1, key=key))
            middle = panels_per_output - 2
            if middle > 0:
                stages = tuple(self.tile(base + 1))
                tile_calls += 1
                phases.append(PipelinePhase(
                    "middle", stages, middle,
                    key=tuple(key_of(base + 1)) if key_of is not None else (),
                ))
            stages = tuple(self.tile(base + panels_per_output - 1))
            tile_calls += 1
            phases.append(PipelinePhase(
                "last", stages, 1,
                key=tuple(key_of(base + panels_per_output - 1)) if key_of is not None else (),
            ))
        return PipelineTemplate(
            owner, self.slots, tuple(phases),
            setup_ms=self.setup_ms, minimum_ms=self.minimum_ms,
            bindings=tuple(dict(row) for row in bindings),
            tile_factory_calls=tile_calls,
            phase_objects=len(phases),
            stage_objects=sum(len(phase.stages) for phase in phases),
            static_phase_classes=len(phases),
            dynamic_instances=sum(phase.count for phase in phases),
        )

    def start_template(
        self,
        engine: Engine,
        name: str,
        ready: Event,
        *,
        owner: str = "",
        panels_per_output: int = 1,
        k_tail: bool = False,
        key_of=None,
        bindings=(),
    ) -> Event:
        """Submit a declared template, or the correct detailed finite window."""
        if not self.homogeneous:
            return self.start(engine, name, ready, owner=owner)
        template = self.template(
            panels_per_output=panels_per_output,
            k_tail=k_tail,
            key_of=key_of,
            bindings=bindings,
        )
        return PipelineTemplateProgram(template).start(engine, name, ready, owner=owner)

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        if self.count < 0 or self.slots < 1:
            raise ValueError("pipeline_requires_nonnegative_tiles_and_positive_slots")
        if self.homogeneous and self.template_start:
            return PipelineTemplateProgram(self.template()).start(
                engine, name, ready, owner=owner
            )
        if self.interior_fold:
            fair_resources = frozenset(
                rname for rname, res in engine.resources.items() if res.policy == "fair"
            )
            folded = fold_interior_pipeline(self, fair_resources=fair_resources)
            if folded is not None:
                return folded.start(engine, name, ready, owner=owner)
        if self.count == 0 and not self.setup_ms and not self.minimum_ms:
            return ready

        def scoped(items):
            return scope_allocations(name, items)

        if self.count == 1 and not self.setup_ms and not self.minimum_ms:
            return self._start_single_tile(engine, name, ready, scoped, owner)

        setup = (
            engine.submit(name + ":setup", after=(ready,), latency=self.setup_ms, owner=owner)
            if self.setup_ms else ready
        )
        observation = (
            engine.submit(name + ":span", after=(setup,), latency=self.minimum_ms, owner=owner)
            if self.minimum_ms else None
        )
        static_result = self._try_start_static_dag(engine, name, setup, observation, scoped, owner)
        if isinstance(static_result, Event):
            return static_result
        complete = engine.future(name + ":pipeline_complete")
        if static_result is None:
            stage_tail: dict[str, Event] = {}
            tile_events: dict[int, dict[str, Event]] = {}
            live_tiles: set[int] = set()
            next_tile, finished = 0, 0
            prefetched: dict[int, tuple[Stage, ...]] = {}
            static_tail: list[tuple[int, Event]] = []
        else:
            next_tile, prefetched_stages, stage_tail, tile_events, tile_retires = static_result
            prefetched = {next_tile: prefetched_stages}
            tail_start = max(0, next_tile - self.slots)
            finished = tail_start
            live_tiles = set(range(tail_start, next_tile)) if self.slots > 1 else set()
            static_tail = [(idx, tile_retires[idx]) for idx in range(tail_start, next_tile)]
        max_lag = 1
        serial_single_slot = self.slots == 1

        def retire(_event=None, tile_idx: int | None = None) -> None:
            nonlocal finished
            finished += 1
            if tile_idx is not None:
                live_tiles.discard(tile_idx)
            if finished == self.count:
                stage_tail.clear()
                tile_events.clear()
                live_tiles.clear()
                prefetched.clear()
                engine.resolve(complete)
            else:
                issue()

        def issue() -> None:
            nonlocal next_tile, finished, max_lag
            while next_tile < self.count:
                index = next_tile
                next_tile += 1
                local: dict[str, Event] = {}
                previous = dict(stage_tail)
                stages = prefetched.pop(index, None)
                if stages is None:
                    stages = tuple(self.tile(index))
                for st in stages:
                    for _, lag in st.previous_lags:
                        if lag > max_lag:
                            max_lag = lag
                consumed: set[str] = set()
                for stage in stages:
                    if stage.after:
                        consumed.update(stage.after)
                    deps = [local[key] for key in stage.after]
                    deps.extend(previous[key] for key in stage.previous if key in previous)
                    for dep_name, lag in stage.previous_lags:
                        target = index - lag
                        if target >= 0:
                            prior_tile = tile_events.get(target)
                            if prior_tile is not None and dep_name in prior_tile:
                                deps.append(prior_tile[dep_name])
                    if stage.name in stage_tail:
                        deps.append(stage_tail[stage.name])
                    task_name = f"{name}:tile{index}:{stage.name}"
                    if (
                        stage.program is None and not stage.acquire
                        and not stage.free and _declares_no_work(stage)
                    ):
                        ev = _zero_work_alias(engine, deps, setup)
                        if ev is not None:
                            local[stage.name] = stage_tail[stage.name] = ev
                            continue
                    event = stage.submit(
                        engine, task_name, tuple(deps) or (setup,), scoped, owner
                    )
                    local[stage.name] = stage_tail[stage.name] = event
                if not serial_single_slot:
                    live_tiles.add(index)
                tile_events[index] = local
                cutoff = index + 1 - max(self.slots, max_lag)
                for old_idx in tuple(tile_events):
                    if old_idx < cutoff and old_idx not in live_tiles:
                        del tile_events[old_idx]
                sinks = tuple(dict.fromkeys(
                    local[stage.name] for stage in stages if stage.name not in consumed
                )) or (setup,)
                if serial_single_slot and len(sinks) == 1:
                    sink = sinks[0]
                    if sink.time is not None:
                        finished += 1
                        if finished == self.count:
                            stage_tail.clear()
                            tile_events.clear()
                            prefetched.clear()
                            engine.resolve(complete)
                            return
                        continue
                    engine.on_complete(sink, retire, native_hook=True)
                    return
                engine.submit(
                    f"{name}:tile{index}:retire", after=sinks,
                    on_complete=lambda ev, _idx=index: retire(ev, _idx),
                    owner=owner, native_hook=True,
                )
                return

        initial_slots = max(0, min(self.slots, self.count) - len(static_tail))

        def begin(_: Event) -> None:
            if not self.count:
                engine.resolve(complete)
                return
            for _ in range(initial_slots):
                issue()

        for prev_idx, prev_ev in static_tail:
            engine.on_complete(
                prev_ev,
                lambda ev, _idx=prev_idx: retire(ev, _idx),
                native_hook=True,
            )
        if initial_slots == 0:
            if observation is None:
                return complete
        elif serial_single_slot:
            if setup.done and setup.time <= engine.now:
                begin(setup)
            else:
                engine.on_complete(setup, begin, native_hook=True)
            if observation is None:
                return complete
        else:
            engine.submit(
                name + ":begin", after=(setup,), on_complete=begin,
                owner=owner, native_hook=True,
            )
        return engine.submit(
            name + ":complete",
            after=(complete, observation) if observation is not None else (complete,),
            owner=owner,
        )

    @staticmethod
    def _tile_static_sinks(
        stages: tuple[Stage, ...], fifo_names: frozenset[str],
    ) -> tuple[bool, tuple[str, ...]]:
        """Check whether one tile's stages can submit statically without retire callbacks."""
        consumed: set[str] = set()
        for stage in stages:
            if (
                stage.program is not None or stage.fragment is not None
                or stage.acquire or stage.free or stage.deliveries
                or stage.latency_from_first_service
            ):
                return False, ()
            if fifo_names and stage.work and any(r in fifo_names for r in stage.work):
                return False, ()
            if stage.after:
                consumed.update(stage.after)
        sinks = tuple(stage.name for stage in stages if stage.name not in consumed)
        return True, sinks

    @staticmethod
    def _submit_static_tile(
        engine: Engine, name: str, index: int, stages: tuple[Stage, ...],
        slot_gate: Event, setup: Event, stage_tail: dict[str, Event],
        tile_events: dict[int, dict[str, Event]], scoped: Callable, owner: str,
    ) -> Event:
        """Submit one static-eligible tile and drop its Stage refs on return."""
        local: dict[str, Event] = {}
        previous = dict(stage_tail)
        consumed: set[str] = set()
        for stage in stages:
            if stage.after:
                consumed.update(stage.after)
            deps = [local[key] for key in stage.after]
            if not stage.after and slot_gate is not setup:
                deps.append(slot_gate)
            deps.extend(previous[key] for key in stage.previous if key in previous)
            for dep_name, lag in stage.previous_lags:
                target = index - lag
                if target >= 0:
                    prior_tile = tile_events.get(target)
                    if prior_tile is not None and dep_name in prior_tile:
                        deps.append(prior_tile[dep_name])
            if stage.name in stage_tail:
                deps.append(stage_tail[stage.name])
            task_name = f"{name}:tile{index}:{stage.name}"
            if _declares_no_work(stage):
                ev = _zero_work_alias(engine, deps, slot_gate)
                if ev is not None:
                    local[stage.name] = stage_tail[stage.name] = ev
                    continue
            event = stage.submit(
                engine, task_name, tuple(deps) or (slot_gate,), scoped, owner,
            )
            local[stage.name] = stage_tail[stage.name] = event
        tile_events[index] = local
        sinks = tuple(dict.fromkeys(
            local[stage.name] for stage in stages if stage.name not in consumed
        )) or (slot_gate,)
        return (
            sinks[0]
            if len(sinks) == 1
            else engine.submit(f"{name}:tile{index}:retire", after=sinks, owner=owner)
        )

    def _try_start_static_dag(
        self, engine: Engine, name: str, setup: Event,
        observation: Event | None, scoped: Callable, owner: str,
    ):
        """Stream bounded allocation-free tiles statically with zero Stage retention."""
        if not (_MIN_STATIC_UNROLL_TILES <= self.count <= _STATIC_UNROLL_MAX_TILES):
            return None
        if engine.memory is not None or engine._periodic_runs:
            return None
        fifo_names = engine.fifo_resource_names() if hasattr(engine, "fifo_resource_names") else frozenset()
        first_sinks: tuple[str, ...] | None = None
        stage_tail: dict[str, Event] = {}
        tile_events: dict[int, dict[str, Event]] = {}
        tile_retires: list[Event] = []
        check_sinks = self.slots > 1 and self.slots < self.count
        for index in range(self.count):
            stages = tuple(self.tile(index))
            ok, sink_names = self._tile_static_sinks(stages, fifo_names)
            if ok and check_sinks:
                if not sink_names:
                    ok = False
                elif first_sinks is None:
                    first_sinks = sink_names
                elif sink_names != first_sinks:
                    ok = False
            if not ok:
                return (index, stages, stage_tail, tile_events, tile_retires)
            slot_gate = tile_retires[index - self.slots] if index >= self.slots else setup
            tile_retire = self._submit_static_tile(
                engine, name, index, stages, slot_gate, setup,
                stage_tail, tile_events, scoped, owner,
            )
            stages = ()
            tile_retires.append(tile_retire)
        tail_deps = list(dict.fromkeys(tile_retires[-min(self.slots, self.count):]))
        if observation is not None:
            tail_deps.append(observation)
        if len(tail_deps) == 1:
            return tail_deps[0]
        return engine.submit(name + ":complete", after=tuple(tail_deps), owner=owner)

    def _start_single_tile(self, engine, name, ready, scoped, owner):
        def submit_tile0(trigger: Event) -> tuple[Event, ...]:
            local: dict[str, Event] = {}
            previous: dict[str, Event] = {}
            stages = tuple(self.tile(0))
            consumed: set[str] = set()
            for stage in stages:
                if stage.after:
                    consumed.update(stage.after)
                deps = [local[key] for key in stage.after]
                deps.extend(previous[key] for key in stage.previous if key in previous)
                if (
                    stage.program is None and not stage.acquire
                    and not stage.free and _declares_no_work(stage)
                ):
                    ev = _zero_work_alias(engine, deps, trigger)
                    if ev is not None:
                        local[stage.name] = ev
                        continue
                ev = stage.submit(
                    engine, f"{name}:tile0:{stage.name}",
                    tuple(deps) or (trigger,), scoped, owner,
                )
                local[stage.name] = ev
            sinks = tuple(dict.fromkeys(
                local[stage.name] for stage in stages if stage.name not in consumed
            ))
            return sinks or (trigger,)

        if ready.done and ready.time <= engine.now:
            sinks = submit_tile0(ready)
            if len(sinks) == 1:
                return sinks[0]
            return engine.submit(f"{name}:tile0:retire", after=sinks, owner=owner)

        complete_one = engine.future(name + ":pipeline_complete")

        def begin_one(trigger: Event) -> None:
            sinks = submit_tile0(trigger)
            if len(sinks) == 1:
                engine.on_complete(
                    sinks[0], lambda _e: engine.resolve(complete_one),
                    native_hook=True,
                )
            else:
                engine.submit(
                    f"{name}:tile0:retire", after=sinks,
                    on_complete=lambda _e: engine.resolve(complete_one),
                    owner=owner, native_hook=True,
                )

        engine.on_complete(ready, begin_one, native_hook=True)
        return complete_one


def _zero_work_alias(engine: Engine, deps: list[Event], fallback: Event) -> Event | None:
    if not deps:
        return fallback
    if len(deps) == 1:
        return deps[0]
    now = engine.now
    pending = [dep for dep in deps if not dep.done]
    if len(pending) == 1 and all(dep.time <= now for dep in deps if dep.done):
        return pending[0]
    if not pending:
        latest = max(deps, key=lambda dep: dep.time)
        if latest.time == now:
            return latest
    return None


@dataclass(frozen=True)
class PipelinePhase:
    """One repeated phase class of a finite pipeline template."""

    kind: str
    stages: tuple[Stage, ...]
    count: int
    key: tuple = ()
    residue_stages: tuple[tuple[Stage, ...], ...] = ()
    end: int | None = None

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("pipeline_phase_requires_kind")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("pipeline_phase_requires_positive_count")
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "key", tuple(self.key))
        object.__setattr__(self, "residue_stages", tuple(tuple(g) for g in self.residue_stages))
        if any(not group for group in self.residue_stages):
            raise ValueError("pipeline_phase_residue_requires_stages")
        if not self.stages:
            raise ValueError("pipeline_phase_requires_stages")

    def stages_for(self, index: int) -> tuple[Stage, ...]:
        if self.residue_stages:
            return self.residue_stages[index % len(self.residue_stages)]
        return tuple(self.stages)


@dataclass(frozen=True)
class PipelineTemplate:
    """Finite phase template for bounded Engine repeat execution."""

    owner: int
    slots: int
    phases: tuple[PipelinePhase, ...]
    setup: tuple[Stage, ...] = ()
    drain: tuple[Stage, ...] = ()
    setup_ms: float = 0.0
    minimum_ms: float = 0.0
    bindings: tuple[dict, ...] = ()
    motif_phases: tuple[PipelinePhase, ...] = ()
    motif_repeats: int = 1
    tile_factory_calls: int = 0
    stage_objects: int = 0
    phase_objects: int = 0
    static_phase_classes: int = 0
    dynamic_instances: int = 0
    indexed_panels: int | None = None
    index_of: Callable[[int], tuple[int, int, int, int, int]] | None = None

    def __post_init__(self):
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "setup", tuple(self.setup))
        object.__setattr__(self, "drain", tuple(self.drain))
        object.__setattr__(self, "bindings", tuple(dict(row) for row in self.bindings))
        object.__setattr__(self, "motif_phases", tuple(self.motif_phases))
        if self.motif_repeats < 1:
            raise ValueError("pipeline_template_motif_repeats_positive")
        if (self.indexed_panels is None) != (self.index_of is None):
            raise ValueError("pipeline_template_indexed_requires_both_fields")
        if self.indexed_panels is not None:
            if (
                isinstance(self.indexed_panels, bool)
                or not isinstance(self.indexed_panels, int)
                or self.indexed_panels < 0
            ):
                raise ValueError("pipeline_template_indexed_panels_invalid")
            if not callable(self.index_of):
                raise ValueError("pipeline_template_indexed_index_of_invalid")
            if isinstance(self.slots, bool) or not isinstance(self.slots, int) or self.slots < 1:
                raise ValueError("pipeline_template_indexed_requires_positive_slots")
            for phase in self.phases:
                if phase.count != 1:
                    raise ValueError("pipeline_template_indexed_phase_count_must_be_one")
                if phase.end is not None:
                    raise ValueError("pipeline_template_indexed_phase_end_must_be_none")
            if self.indexed_panels and not self.phases:
                raise ValueError("pipeline_template_indexed_requires_catalog")
            self._validate_indexed_bindings()

    def _validate_indexed_bindings(self) -> None:
        declared: dict[str, str] = {}
        for row in self.bindings:
            if not isinstance(row, Mapping):
                raise ValueError("pipeline_indexed_binding_row_invalid")
            name = row.get("name")
            scope = row.get("scope")
            if not isinstance(name, str) or not name:
                raise ValueError("pipeline_indexed_binding_name_invalid")
            if scope not in {"invocation", "output", "panel", "slot"}:
                raise ValueError(f"pipeline_indexed_binding_scope_invalid:{name}")
            if "owner" in row and row["owner"] != self.owner:
                raise ValueError(f"pipeline_indexed_binding_owner_mismatch:{name}")
            previous = declared.get(name)
            if previous is not None and previous != scope:
                raise ValueError(f"pipeline_indexed_binding_scope_conflict:{name}")
            declared[name] = scope
        stages = [*self.setup, *self.drain]
        for phase in self.phases:
            stages.extend(phase.stages)
            for group in phase.residue_stages:
                stages.extend(group)
        for stage in stages:
            for allocation in (*stage.acquire, *stage.free):
                if allocation.name not in declared:
                    raise ValueError(
                        f"pipeline_indexed_allocation_symbol_unbound:{allocation.name}"
                    )

    @property
    def tiles(self) -> int:
        if self.indexed_panels is not None:
            return self.indexed_panels
        return sum(phase.count for phase in self.phases)

    def stage_for(self, index: int) -> tuple[Stage, ...]:
        """Select the real Stage set for one absolute logical index."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("pipeline_stage_index_requires_integer")
        if self.indexed_panels is not None:
            if index < 0 or index >= self.indexed_panels:
                raise IndexError("pipeline_stage_index_out_of_range")
            try:
                mapped = self.index_of(index)
            except (IndexError, TypeError):
                raise
            except KeyError as exc:
                raise ModelingError("pipeline_index_mapping_invalid") from exc
            if not isinstance(mapped, tuple) or len(mapped) != _INDEX_COORD_ARITY:
                raise ModelingError("pipeline_index_mapping_invalid")
            phase_id, end, output, panel, slot = mapped
            if (
                isinstance(phase_id, bool) or not isinstance(phase_id, int)
                or not 0 <= phase_id < len(self.phases)
                or isinstance(end, bool) or not isinstance(end, int)
                or not index < end <= self.indexed_panels
                or isinstance(output, bool) or not isinstance(output, int) or output < 0
                or isinstance(panel, bool) or not isinstance(panel, int) or panel < 0
                or isinstance(slot, bool) or not isinstance(slot, int)
                or not 0 <= slot < self.slots
            ):
                raise ModelingError("pipeline_index_mapping_invalid")
            return self.phases[phase_id].stages_for(index)
        if index < 0:
            raise IndexError("pipeline_stage_index_out_of_range")
        cursor = 0
        for phase in self.phases:
            if index < cursor + phase.count:
                return phase.stages_for(index)
            cursor += phase.count
        raise IndexError("pipeline_stage_index_out_of_range")

    def reference_stages(self):
        """Yield logical Stage sets by absolute index; never hold a copy."""
        if self.indexed_panels is None:
            index = 0
            for phase in self.phases:
                for _ in range(phase.count):
                    yield phase.stages_for(index)
                    index += 1
            return
        for index in range(self.indexed_panels):
            yield self.stage_for(index)

    @property
    def allocation_bindings(self):
        """Read-only producer bindings; never inferred from display names."""
        return self.bindings


def indexed_pipeline_consumer_available() -> bool:
    """True when an indexed finite reader exists on the selected path."""
    try:
        from ..scheduling import start_pipeline_template
    except ImportError:
        return False
    version = getattr(start_pipeline_template, "indexed_panels_abi", None)
    return type(version) is int and version == 1


def _iter_template_stages(template: PipelineTemplate):
    yield from template.setup
    yield from template.drain
    for phase in template.phases:
        for group in (phase.residue_stages or (phase.stages,)):
            yield from group


def _template_has_prepared_collision(template: PipelineTemplate) -> bool:
    """True when two allocation-free stages collide in scheduling._prepared_for."""
    seen: dict[tuple, tuple] = {}
    for stage in _iter_template_stages(template):
        if stage.acquire or stage.free or stage.program is not None:
            continue
        cache_key = (
            stage.name, tuple(sorted(stage.work.items())), stage.latency,
            tuple(sorted(stage.caps.items())),
            tuple(sorted((k, tuple(u)) for k, u in stage.placement.items())),
            stage.coupled, stage.priority,
            tuple(sorted(stage.work_units.items())),
        )
        full_sig = (
            stage.latency_from_first_service, stage.deliveries,
            stage.cohort_size, stage.cohort_widened,
            tuple(sorted(stage.cohort_members.items())),
            stage.keep_empty, tuple(sorted(stage.metadata.items())),
        )
        prev = seen.get(cache_key)
        if prev is not None and prev != full_sig:
            return True
        seen[cache_key] = full_sig
    return False


def _template_needs_direct(template: PipelineTemplate) -> bool:
    if any(stage.previous_lags for stage in _iter_template_stages(template)):
        return True
    if template.indexed_panels is None and any(
        stage.program is not None
        for phase in template.phases
        if phase.residue_stages
        for group in phase.residue_stages
        for stage in group
    ):
        return True
    return _template_has_prepared_collision(template)


def _resolve_template_tile(template: PipelineTemplate, index: int):
    if template.indexed_panels is None:
        return template.stage_for(index), None
    if index < 0 or index >= template.indexed_panels:
        raise IndexError("pipeline_stage_index_out_of_range")
    try:
        mapped = template.index_of(index)
    except (IndexError, TypeError):
        raise
    except KeyError as exc:
        raise ModelingError("pipeline_index_mapping_invalid") from exc
    if not isinstance(mapped, tuple) or len(mapped) != _INDEX_COORD_ARITY:
        raise ModelingError("pipeline_index_mapping_invalid")
    phase_id, end, output, panel, slot = mapped
    if (
        isinstance(phase_id, bool) or not isinstance(phase_id, int)
        or not 0 <= phase_id < len(template.phases)
        or isinstance(end, bool) or not isinstance(end, int)
        or not index < end <= template.indexed_panels
        or isinstance(output, bool) or not isinstance(output, int) or output < 0
        or isinstance(panel, bool) or not isinstance(panel, int) or panel < 0
        or isinstance(slot, bool) or not isinstance(slot, int)
        or not 0 <= slot < template.slots
    ):
        raise ModelingError("pipeline_index_mapping_invalid")
    return template.phases[phase_id].stages_for(index), (output, panel, slot)


def _build_template_scoper(template: PipelineTemplate, name: str):
    acquire_phase: dict[str, set[int]] = {}
    free_phase: dict[str, set[int]] = {}
    for phase_index, phase in enumerate(template.phases):
        for group in (phase.residue_stages or (phase.stages,)):
            for stage in group:
                for alloc in stage.acquire:
                    acquire_phase.setdefault(alloc.name, set()).add(phase_index)
                for alloc in stage.free:
                    free_phase.setdefault(alloc.name, set()).add(phase_index)
    persistent = frozenset(
        aname for aname, acq in acquire_phase.items()
        if not free_phase.get(aname) or not free_phase[aname].issubset(acq)
    )
    binding_rows = {
        str(row["name"]): dict(row)
        for row in template.bindings if isinstance(row, Mapping) and row.get("name")
    }
    for stage in (*template.setup, *template.drain):
        for alloc in (*stage.acquire, *stage.free):
            binding_rows.setdefault(
                alloc.name, {"name": alloc.name, "scope": "invocation", "owner": template.owner},
            )
    indexed = template.indexed_panels is not None
    explicit_keys = ("identity", "physical", "physical_name", "external")

    def scoped_at(label: str, index: int, coords):
        output, panel, slot = coords if coords is not None else (None, None, None)

        def _scope(items):
            out = []
            for item in items:
                row = binding_rows.get(item.name, {})
                scope = row.get("scope")
                if indexed:
                    explicit = next(
                        (row[k] for k in explicit_keys if isinstance(row.get(k), str) and row[k]),
                        None,
                    )
                    if explicit is not None:
                        scoped_name = explicit
                    elif scope == "invocation":
                        scoped_name = f"{name}:owner{template.owner}:invocation0:{item.name}"
                    elif scope == "output":
                        ord_val = output if output is not None else int(row.get("ordinal", 0))
                        scoped_name = f"{name}:owner{template.owner}:output{ord_val}:{item.name}"
                    elif scope == "slot":
                        ord_val = slot if slot is not None else int(row.get("ordinal", 0))
                        scoped_name = f"{name}:owner{template.owner}:slot{ord_val}:{item.name}"
                    elif scope == "panel" or item.name not in persistent:
                        scoped_name = f"{name}:owner{template.owner}:panel{index}:{item.name}"
                    else:
                        scoped_name = f"{name}:owner{template.owner}:invocation0:{item.name}"
                else:
                    if scope == "invocation":
                        prefix = name
                    elif scope == "output":
                        digits = "".join(ch for ch in item.name if ch.isdigit())
                        prefix = f"{name}:output{digits or 0}"
                    elif scope == "slot":
                        prefix = f"{name}:slot{index % template.slots}"
                    elif scope == "panel":
                        prefix = label
                    elif item.name in persistent:
                        prefix = name
                    else:
                        prefix = label
                    scoped_name = f"{prefix}:{item.name}"
                out.append(Allocation(scoped_name, item.pool, item.size, item.units))
            return tuple(out)

        return _scope

    return scoped_at


def _start_template_direct(
    engine: Engine, template: PipelineTemplate, name: str, ready: Event, owner: str,
) -> Event:
    """Execute a PipelineTemplate losslessly with full binding scopes and lags.

    [Cohesive-Pipeline] Single-pass template instantiation binding scopes, reusable allocations, and lag edges.
    """
    scoped_at = _build_template_scoper(template, name)
    explicit_keys = ("identity", "physical", "physical_name", "external")
    reusable_alloc_names = frozenset(
        str(row["name"])
        for row in template.bindings
        if isinstance(row, Mapping) and row.get("name")
        and (
            row.get("scope") == "slot"
            or any(isinstance(row.get(k), str) and row.get(k) for k in explicit_keys)
        )
    )
    slot_alloc_free: dict[str, Event] = {}
    current = (
        engine.submit(name + ":setup", after=(ready,), latency=template.setup_ms, owner=owner)
        if template.setup_ms else ready
    )
    if template.setup:
        local_setup: dict[str, Event] = {}
        for stage in template.setup:
            deps = tuple(local_setup[key] for key in stage.after)
            local_setup[stage.name] = stage.submit(
                engine, f"{name}:setup:{stage.name}", deps or (current,),
                scoped_at(f"{name}:setup:{stage.name}", 0, (0, 0, 0)), owner,
            )
        current = engine.submit(
            name + ":setup_complete", after=tuple(local_setup.values()) or (current,), owner=owner,
        )
    observation = (
        engine.submit(name + ":span", after=(current,), latency=template.minimum_ms, owner=owner)
        if template.minimum_ms else None
    )
    total = template.tiles
    complete = engine.future(name + ":complete")
    stage_tail: dict[str, Event] = {}
    tile_events: dict[int, dict[str, Event]] = {}
    live_tiles: set[int] = set()
    next_tile, finished, max_lag = 0, 0, 1
    last_events: tuple[Event, ...] = (current,)
    drain_coords = (0, 0, 0)

    def _reusable_deps(scoped, stage) -> list[Event]:
        if not reusable_alloc_names or not stage.acquire:
            return []
        extra: list[Event] = []
        for raw, bound in zip(stage.acquire, scoped(stage.acquire)):
            if raw.name in reusable_alloc_names:
                prior = slot_alloc_free.get(bound.name)
                if prior is not None:
                    extra.append(prior)
        return extra

    def _note_reusable_frees(scoped, stage, event: Event) -> None:
        if not reusable_alloc_names or not stage.free:
            return
        for raw, bound in zip(stage.free, scoped(stage.free)):
            if raw.name in reusable_alloc_names:
                slot_alloc_free[bound.name] = event

    def finalize() -> None:
        stage_tail_snap = dict(stage_tail)
        stage_tail.clear()
        tile_events.clear()
        live_tiles.clear()
        if not template.drain:
            slot_alloc_free.clear()
            engine.resolve(complete)
            return
        local_drain: dict[str, Event] = {}
        drain_index = total - 1 if total else 0
        for stage in template.drain:
            deps = [local_drain[key] for key in stage.after]
            deps.extend(stage_tail_snap[key] for key in stage.previous if key in stage_tail_snap)
            scoped_drain = scoped_at(f"{name}:drain:{stage.name}", drain_index, drain_coords)
            deps.extend(_reusable_deps(scoped_drain, stage))
            ev = stage.submit(
                engine, f"{name}:drain:{stage.name}",
                tuple(dict.fromkeys(deps)) or last_events,
                scoped_drain, owner,
            )
            local_drain[stage.name] = ev
            _note_reusable_frees(scoped_drain, stage, ev)
        slot_alloc_free.clear()
        engine.submit(
            name + ":drain_complete", after=tuple(local_drain.values()) or last_events,
            on_complete=lambda _ev: engine.resolve(complete), owner=owner,
        )

    def retire(_event=None, tile_idx: int | None = None) -> None:
        nonlocal finished
        finished += 1
        if tile_idx is not None:
            live_tiles.discard(tile_idx)
        if finished == total:
            finalize()
        else:
            issue()

    def issue() -> None:
        nonlocal next_tile, max_lag, last_events, drain_coords
        if next_tile >= total:
            return
        index = next_tile
        next_tile += 1
        stages, coords = _resolve_template_tile(template, index)
        if index == total - 1 and coords is not None:
            drain_coords = coords
        for st in stages:
            for _, lag in st.previous_lags:
                if lag > max_lag:
                    max_lag = lag
        local: dict[str, Event] = {}
        previous = dict(stage_tail)
        consumed: set[str] = set()
        scoped = scoped_at(f"{name}:tile{index}", index, coords)
        for stage in stages:
            if stage.after:
                consumed.update(stage.after)
            deps = [local[key] for key in stage.after]
            deps.extend(previous[key] for key in stage.previous if key in previous)
            for dep_name, lag in stage.previous_lags:
                target = index - lag
                if target >= 0:
                    prior_tile = tile_events.get(target)
                    if prior_tile is not None and dep_name in prior_tile:
                        deps.append(prior_tile[dep_name])
            if stage.name in stage_tail:
                deps.append(stage_tail[stage.name])
            deps.extend(_reusable_deps(scoped, stage))
            ev = stage.submit(
                engine, f"{name}:tile{index}:{stage.name}",
                tuple(dict.fromkeys(deps)) or (current,), scoped, owner,
            )
            local[stage.name] = stage_tail[stage.name] = ev
            _note_reusable_frees(scoped, stage, ev)
        tile_events[index] = local
        live_tiles.add(index)
        cutoff = index - max(template.slots, max_lag)
        for old_idx in tuple(tile_events):
            if old_idx < cutoff and old_idx not in live_tiles:
                del tile_events[old_idx]
        sinks = tuple(dict.fromkeys(
            local[stage.name] for stage in stages if stage.name not in consumed
        )) or (current,)
        last_events = sinks
        engine.submit(
            f"{name}:tile{index}:retire", after=sinks,
            on_complete=lambda ev, _idx=index: retire(ev, _idx),
            owner=owner, native_hook=True,
        )

    def begin(_: Event) -> None:
        if not total:
            finalize()
            return
        for _ in range(min(template.slots, total)):
            issue()

    engine.submit(name + ":begin", after=(current,), on_complete=begin, owner=owner, native_hook=True)
    if observation is None:
        return complete
    return engine.submit(name + ":complete_outer", after=(complete, observation), owner=owner)


@dataclass(frozen=True)
class PipelineTemplateProgram:
    """Program adapter that submits a finite PipelineTemplate."""

    template: PipelineTemplate

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        stages = tuple(_iter_template_stages(self.template))
        if self.template.indexed_panels is not None and not indexed_pipeline_consumer_available():
            raise ModelingError("pipeline_indexed_consumer_unavailable")
        if _template_needs_direct(self.template):
            return _start_template_direct(engine, self.template, name, ready, owner)
        nested = any(stage.program is not None for stage in stages)
        allocated = any(stage.acquire or stage.free for stage in stages)
        has_residue = any(phase.residue_stages for phase in self.template.phases)
        from ..program.catalog import submit_template
        if (
            not nested and not allocated and not has_residue
            and self.template.indexed_panels is None
            and callable(getattr(engine, "submit_catalog", None))
        ):
            return submit_template(
                engine, self.template, name=name, ready=ready, owner=owner
            ).done
        try:
            from ..scheduling import start_pipeline_template
        except ImportError as exc:
            if callable(getattr(engine, "submit_catalog", None)):
                raise ModelingError("pipeline_template_nested_program_requires_consumer") from exc
            raise ModelingError("pipeline_template_requires_causal_catalog_engine") from exc
        if self.template.indexed_panels is not None:
            version = getattr(start_pipeline_template, "indexed_panels_abi", None)
            if type(version) is not int or version != 1:
                raise ModelingError("pipeline_indexed_consumer_unavailable")
        return start_pipeline_template(engine, self.template, name, ready, owner=owner)
