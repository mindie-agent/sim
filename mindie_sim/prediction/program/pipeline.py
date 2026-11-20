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
from ..engine import Engine, Event
from .stages import Stage, _declares_no_work, scope_allocations

#: Bounded tile window a symmetric cohort may materialize to prove identity.
PIPELINE_COHORT_WINDOW = 512
#: Bounded tile window an interior fold may materialize.
PIPELINE_FOLD_WINDOW = 512
_INDEX_COORD_ARITY = 5
_MIN_FOLD_TILES = 4
_MIN_FOLD_ENTRIES = 3


def _allocation_sig(items):
    return tuple(sorted((item.name, item.pool, item.size, len(item.units)) for item in items))


def _fold_stage_key(stage):
    """Exact physical identity of one repeated stage, work included."""
    if stage.program is not None or stage.fragment is not None:
        return None
    if stage.latency > 0.0 or stage.coupled:
        return None
    return (
        stage.name, stage.after, stage.previous, stage.priority,
        tuple(sorted(stage.work.items())),
        tuple(sorted(stage.caps.items())),
        tuple((key, tuple(units)) for key, units in stage.placement.items()),
        tuple(sorted(stage.work_units.items())),
        _allocation_sig(stage.acquire),
        _allocation_sig(stage.free),
    )


def _fold_tile_key(stages):
    rows = []
    for stage in stages:
        key = _fold_stage_key(stage)
        if key is None:
            return None
        rows.append(key)
    return tuple(rows)


def _fold_scaled(stages, repeat: int):
    """One interior super-tile: identical stages whose work is the run's sum."""
    return tuple(
        replace(stage, work={key: value * repeat for key, value in stage.work.items()})
        for stage in stages
    )


def interior_fold_plan(program: PipelineProgram):
    """Return ``[(representative index, repeat, stages)]`` or ``None``."""
    count, slots = program.count, program.slots
    if program.homogeneous or count < _MIN_FOLD_TILES or slots < 1:
        return None
    if count > PIPELINE_FOLD_WINDOW:
        return None
    tiles = [tuple(program.tile(index)) for index in range(count)]
    if any(not stages for stages in tiles):
        return None
    keys = [_fold_tile_key(stages) for stages in tiles]
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


def fold_interior_pipeline(program: PipelineProgram):
    """Return an exactly-folded equivalent program, or ``None``."""
    plan = interior_fold_plan(program)
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
            folded = fold_interior_pipeline(self)
            if folded is not None:
                return folded.start(engine, name, ready, owner=owner)
        if self.count == 0 and not self.setup_ms and not self.minimum_ms:
            return ready

        def scoped(items):
            return scope_allocations(name, items)

        def _zero_work_alias(deps: list[Event], fallback: Event) -> Event | None:
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

        if self.count == 1 and not self.setup_ms and not self.minimum_ms:
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
                        ev = _zero_work_alias(deps, trigger)
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
                    engine.on_complete(sinks[0], lambda _e: engine.resolve(complete_one))
                else:
                    engine.submit(
                        f"{name}:tile0:retire", after=sinks,
                        on_complete=lambda _e: engine.resolve(complete_one), owner=owner,
                    )

            engine.on_complete(ready, begin_one)
            return complete_one

        complete = engine.future(name + ":pipeline_complete")
        setup = (
            engine.submit(name + ":setup", after=(ready,), latency=self.setup_ms, owner=owner)
            if self.setup_ms else ready
        )
        observation = (
            engine.submit(name + ":span", after=(setup,), latency=self.minimum_ms, owner=owner)
            if self.minimum_ms else None
        )
        stage_tail: dict[str, Event] = {}
        next_tile, finished = 0, 0
        serial_single_slot = self.slots == 1

        def retire(_event=None) -> None:
            nonlocal finished
            finished += 1
            if finished == self.count:
                stage_tail.clear()
                engine.resolve(complete)
            else:
                issue()

        def issue() -> None:
            nonlocal next_tile, finished
            while next_tile < self.count:
                index = next_tile
                next_tile += 1
                local: dict[str, Event] = {}
                previous = dict(stage_tail)
                stages = tuple(self.tile(index))
                consumed: set[str] = set()
                for stage in stages:
                    if stage.after:
                        consumed.update(stage.after)
                    deps = [local[key] for key in stage.after]
                    deps.extend(previous[key] for key in stage.previous if key in previous)
                    if stage.name in stage_tail:
                        deps.append(stage_tail[stage.name])
                    task_name = f"{name}:tile{index}:{stage.name}"
                    if (
                        stage.program is None and not stage.acquire
                        and not stage.free and _declares_no_work(stage)
                    ):
                        ev = _zero_work_alias(deps, setup)
                        if ev is not None:
                            local[stage.name] = stage_tail[stage.name] = ev
                            continue
                    event = stage.submit(
                        engine, task_name, tuple(deps) or (setup,), scoped, owner
                    )
                    local[stage.name] = stage_tail[stage.name] = event
                sinks = tuple(dict.fromkeys(
                    local[stage.name] for stage in stages if stage.name not in consumed
                )) or (setup,)
                if serial_single_slot and len(sinks) == 1:
                    sink = sinks[0]
                    if sink.time is not None:
                        finished += 1
                        if finished == self.count:
                            stage_tail.clear()
                            engine.resolve(complete)
                            return
                        continue
                    engine.on_complete(sink, retire)
                    return
                engine.submit(
                    f"{name}:tile{index}:retire", after=sinks, on_complete=retire, owner=owner
                )
                return

        def begin(_: Event) -> None:
            if not self.count:
                engine.resolve(complete)
                return
            for _ in range(min(self.slots, self.count)):
                issue()

        if serial_single_slot:
            if setup.done and setup.time <= engine.now:
                begin(setup)
            else:
                engine.on_complete(setup, begin)
            if observation is None:
                return complete
        else:
            engine.submit(name + ":begin", after=(setup,), on_complete=begin, owner=owner)
        return engine.submit(
            name + ":complete",
            after=(complete, observation) if observation is not None else (complete,),
            owner=owner,
        )


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


@dataclass(frozen=True)
class PipelineTemplateProgram:
    """Program adapter that submits a finite PipelineTemplate."""

    template: PipelineTemplate

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        stages = (
            *self.template.setup,
            *self.template.drain,
            *(stage for phase in self.template.phases for stage in phase.stages),
        )
        nested = any(stage.program is not None for stage in stages)
        allocated = any(stage.acquire or stage.free for stage in stages)
        from ..program_catalog import submit_template
        if (
            not nested and not allocated
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
