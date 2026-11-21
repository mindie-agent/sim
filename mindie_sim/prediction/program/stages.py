"""Executable stage declarations and leaf kernel/parallel programs.

[Role] Define ``Stage``, ``Program``, ``NativeProgram``, ``ParallelProgram``,
  ``KernelProgram``, allocation scoping, and compiled leaf-kernel submission
  helpers on the shared ``Engine`` contract.
[Invariants] ``Stage`` mappings are frozen copies (``MappingProxyType``) and
  cohort widening metadata uses exact immutable builtins. Pure zero-work stages
  never invent service demand, and acquired allocations hold through completion.
[Timeouts & Locks] Pure virtual submission; compiled-plan caches are bounded by
  ``_COMPILED_CACHE_LIMIT`` with no I/O, threads, or locks.
[Side-Effect Policy] Submits tasks only to the caller's ``Engine`` instance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import ProgramFragment
from ..engine import Allocation, Engine, Event

_EMPTY_MAPPING = MappingProxyType({})
_STAGE_METADATA_BUILTINS = frozenset({bool, int, float, str})
#: Default live-stage bound for coalescing a repeated finite pipeline window.
PIPELINE_UNROLL_LIMIT = 64
_COMPILED_CACHE_LIMIT = 4096


def scope_allocations(name: str, items) -> tuple[Allocation, ...]:
    """Rename immutable buffer identities without reflecting dataclass fields."""
    if not items:
        return ()
    n = len(items)
    pfx = name + ":"
    if n == 1:
        a0 = items[0]
        return (Allocation(pfx + a0.name, a0.pool, a0.size, a0.units),)
    if n == 2:
        a0, a1 = items
        return (
            Allocation(pfx + a0.name, a0.pool, a0.size, a0.units),
            Allocation(pfx + a1.name, a1.pool, a1.size, a1.units),
        )
    return tuple(Allocation(pfx + a.name, a.pool, a.size, a.units) for a in items)


@dataclass(frozen=True)
class Stage:
    """One finite stage on the existing Engine's resources.

    ``work`` is issued service demand. ``acquire``/``free`` are grants and
    releases held until completion including ``latency``. ``coupled=True``
    requests fixed-proportion progress across all resources. ``priority`` is
    non-preemptive admission order. ``cohort_widened`` names resources whose
    ``placement`` is a disjoint union across ``cohort_size`` symmetric members
    (distinguishing widened disjoint slices from shared multi-unit placements);
    ``cohort_members`` optionally retains each member's original unit tuple so
    the Engine can split the cohort if mid-flight interference breaks symmetry.
    """

    name: str
    work: Mapping[str, float] = field(default_factory=dict)
    after: tuple[str, ...] = ()
    latency: float = 0.0
    caps: Mapping[str, float] = field(default_factory=dict)
    acquire: tuple[Allocation, ...] = ()
    free: tuple[Allocation, ...] = ()
    program: Program | None = None
    placement: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    previous: tuple[str, ...] = ()
    coupled: bool = False
    work_units: Mapping[str, str] = field(default_factory=dict)
    priority: int = 0
    fragment: ProgramFragment | None = None
    cohort_size: int = 1
    keep_empty: bool = False
    cohort_widened: tuple[str, ...] = ()
    cohort_members: Mapping[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    deliveries: tuple[tuple[str, float], ...] = ()
    latency_from_first_service: bool = False

    def __post_init__(self):
        work = MappingProxyType(dict(self.work)) if self.work else _EMPTY_MAPPING
        caps = MappingProxyType(dict(self.caps)) if self.caps else _EMPTY_MAPPING
        if not self.placement:
            placement = _EMPTY_MAPPING
        else:
            placement = MappingProxyType({
                k: u if u.__class__ is tuple else tuple(u)
                for k, u in self.placement.items()
            })
        if not self.work_units:
            units = _EMPTY_MAPPING
        else:
            units = MappingProxyType(dict(self.work_units))
            for resource, unit in units.items():
                if not isinstance(resource, str) or not isinstance(unit, str) or not unit:
                    raise ModelingError(f"stage_work_unit_invalid:{self.name}")
        object.__setattr__(self, "work", work)
        object.__setattr__(self, "caps", caps)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "work_units", units)
        if not (
            self.priority.__class__ is int and self.priority >= 0
            and self.cohort_size.__class__ is int and self.cohort_size >= 1
            and self.keep_empty.__class__ is bool and self.fragment is None
        ):
            if not isinstance(self.priority, int) or isinstance(self.priority, bool) or self.priority < 0:
                raise ModelingError(f"stage_priority_invalid:{self.name}")
            if not isinstance(self.cohort_size, int) or isinstance(self.cohort_size, bool) or self.cohort_size < 1:
                raise ModelingError(f"stage_cohort_size_invalid:{self.name}")
            if not isinstance(self.keep_empty, bool):
                raise ModelingError(f"stage_keep_empty_invalid:{self.name}")
            if self.fragment is not None and not isinstance(self.fragment, ProgramFragment):
                raise ModelingError(f"stage_fragment_invalid:{self.name}")
        for attr in ("after", "acquire", "free", "previous"):
            val = getattr(self, attr)
            if val.__class__ is not tuple:
                object.__setattr__(self, attr, tuple(val))
        if self.cohort_widened:
            widened = self.cohort_widened if self.cohort_widened.__class__ is tuple else tuple(self.cohort_widened)
            if any(type(key) is not str for key in widened):
                raise ModelingError(f"stage_cohort_widened_invalid:{self.name}")
            object.__setattr__(self, "cohort_widened", widened)
        elif self.cohort_widened.__class__ is not tuple:
            object.__setattr__(self, "cohort_widened", ())
        if self.cohort_members:
            members = MappingProxyType({
                key: tuple(m if m.__class__ is tuple else tuple(m) for m in rows)
                for key, rows in self.cohort_members.items()
            })
            if any(type(key) is not str or not rows for key, rows in members.items()):
                raise ModelingError(f"stage_cohort_members_invalid:{self.name}")
            object.__setattr__(self, "cohort_members", members)
        else:
            object.__setattr__(self, "cohort_members", _EMPTY_MAPPING)
        if self.metadata:
            raw_meta = dict(self.metadata)
            for mk, mv in raw_meta.items():
                if type(mk) is not str or mk == "occurrence" or type(mv) not in _STAGE_METADATA_BUILTINS:
                    raise ModelingError(f"stage_metadata_invalid:{self.name}:{mk!r}")
            object.__setattr__(self, "metadata", MappingProxyType(raw_meta))
        else:
            object.__setattr__(self, "metadata", _EMPTY_MAPPING)

    def _with_buffers(self, acquire: tuple[Allocation, ...], free: tuple[Allocation, ...]) -> Stage:
        """Clone an already-validated Stage replacing only tuple acquire/free."""
        clone = object.__new__(Stage)
        for attr in (
            "name", "work", "after", "latency", "caps", "program", "placement",
            "previous", "coupled", "work_units", "priority", "fragment",
            "cohort_size", "keep_empty", "cohort_widened", "cohort_members",
            "metadata", "deliveries", "latency_from_first_service",
        ):
            object.__setattr__(clone, attr, getattr(self, attr))
        object.__setattr__(clone, "acquire", acquire)
        object.__setattr__(clone, "free", free)
        object.__setattr__(clone, "_proto_stage", getattr(self, "_proto_stage", self))
        return clone

    def prepare(self, engine, name, scoped):
        """Freeze static stage validation on the public Engine surface."""
        prepare = getattr(engine, "prepare_stage", None)
        acquire = scoped(self.acquire)
        free = () if self.program else scoped(self.free)
        if callable(prepare):
            extra: dict[str, Any] = {}
            if self.cohort_widened:
                extra["cohort_widened"] = self.cohort_widened
            if self.cohort_members:
                extra["cohort_members"] = self.cohort_members
            if self.latency_from_first_service:
                extra["latency_from_first_service"] = True
            return prepare(
                name, self.work, latency=self.latency, caps=self.caps,
                acquire=acquire, free=free, placement=self.placement,
                coupled=self.coupled, priority=self.priority,
                cohort_size=self.cohort_size, **extra,
            )
        return _PreparedStageDeclaration(
            name, self.work, self.latency, self.caps, acquire, free,
            self.placement, self.coupled, self.priority, self.cohort_size,
            self.cohort_widened, self.cohort_members, self.metadata,
            self.deliveries, self.latency_from_first_service,
        )

    def submit(self, engine, name, predecessors, scoped, owner, prepared=None):
        acquire = scoped(self.acquire) if self.acquire else ()
        free = scoped(self.free) if self.free and self.program is None else ()
        if (
            self.program is not None and not self.work and self.latency <= 0.0
            and not self.acquire and not self.free and not self.caps
            and not self.coupled and self.priority == 0
            and not self.keep_empty and not self.metadata
            and self.fragment is None and len(predecessors) == 1
        ):
            return self.program.start(
                engine, name + ":program", predecessors[0], owner=owner
            )
        extra_kw: dict[str, Any] = {}
        if self.metadata:
            extra_kw["metadata"] = dict(self.metadata)
        if self.deliveries:
            extra_kw["deliveries"] = self.deliveries
        if prepared is not None and not isinstance(prepared, _PreparedStageDeclaration):
            sub_fn = getattr(engine, "_submit_prepared", None) or engine.submit_prepared
            event = sub_fn(prepared, name, after=predecessors, owner=owner, **extra_kw)
        else:
            prepare = getattr(engine, "prepared_stage", None)
            if prepare is not None and not self.program:
                prepared = prepare(self, acquire, free)
                sub_fn = getattr(engine, "_submit_prepared", None) or engine.submit_prepared
                event = sub_fn(prepared, name, after=predecessors, owner=owner, **extra_kw)
            else:
                kwargs: dict[str, Any] = dict(extra_kw)
                if self.cohort_size != 1:
                    kwargs["cohort_size"] = self.cohort_size
                if self.cohort_widened:
                    kwargs["cohort_widened"] = self.cohort_widened
                if self.cohort_members:
                    kwargs["cohort_members"] = self.cohort_members
                if self.latency_from_first_service:
                    kwargs["latency_from_first_service"] = True
                event = engine.submit(
                    name, self.work, after=predecessors, latency=self.latency,
                    caps=self.caps, acquire=acquire, free=free,
                    placement=self.placement, coupled=self.coupled,
                    priority=self.priority, owner=owner, **kwargs,
                )
        if self.program is not None:
            event = self.program.start(engine, name + ":program", event, owner=owner)
            if self.free:
                event = engine.submit(
                    name + ":free", after=(event,), free=scoped(self.free), owner=owner
                )
        return event


@dataclass(frozen=True, slots=True)
class _PreparedStageDeclaration:
    """Local static declaration used only when Engine lacks prepare_stage."""

    name: str
    work: Mapping[str, float]
    latency: float
    caps: Mapping[str, float]
    acquire: tuple[Allocation, ...]
    free: tuple[Allocation, ...]
    placement: Mapping[str, tuple[int, ...]]
    coupled: bool
    priority: int
    cohort_size: int = 1
    cohort_widened: tuple[str, ...] = ()
    cohort_members: Mapping[str, tuple[tuple[int, ...], ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    deliveries: tuple[tuple[str, float], ...] = ()
    latency_from_first_service: bool = False


def stage_from_fragment(
    fragment: ProgramFragment,
    *,
    name: str = "",
    after: tuple[str, ...] = (),
    latency: float = 0.0,
) -> Stage:
    """Project a fragment's finite work onto one existing Stage."""
    if not isinstance(fragment, ProgramFragment):
        raise ModelingError("stage_from_fragment_requires_program_fragment")
    blockers = fragment.stage_projection_blockers
    if blockers:
        raise ModelingError(
            "stage_from_fragment_unsupported:" + ",".join(blockers)
        )
    acquire = tuple(
        Allocation(effect.name, effect.pool, effect.size, effect.units)
        for effect in fragment.allocations
        if effect.action == "acquire" and effect.lifetime in {"stage", "slot"}
    )
    free = tuple(
        Allocation(effect.name, effect.pool, effect.size, effect.units)
        for effect in fragment.allocations
        if effect.action == "release" and effect.lifetime in {"stage", "slot"}
    )
    units = {item.resource: item.unit for item in fragment.work.issued()}
    return Stage(
        name or fragment.name, fragment.service_work, after=after,
        latency=latency, acquire=acquire, free=free, work_units=units,
        priority=fragment.priority, coupled=fragment.coupled, fragment=fragment,
    )


class Program(Protocol):
    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event: ...


@dataclass(frozen=True)
class NativeProgram:
    """Canonical executable native program plus its exact source ledger."""

    program: Any
    work: tuple[tuple[str, Fraction], ...]
    fidelity: str = "native_program"

    def start(
        self, engine: Any, name: str, ready: Any, *,
        peer_ready: Any = None, peer_send: Any = None,
        launch: Any = None, source_send: Any = None,
        send_placeholders: Any = None, transport: Any = None,
    ) -> Any:
        if (
            peer_ready is None and peer_send is None and launch is None
            and source_send is None and send_placeholders is None
            and transport is None
        ):
            return self.program.start(engine, name, ready)
        arguments = dict(
            peer_ready=peer_ready, peer_send=peer_send, launch=launch,
            source_send=source_send, send_placeholders=send_placeholders,
        )
        if transport is not None:
            arguments["transport"] = transport
        return self.program.start(engine, name, ready, **arguments)

    def start_with_handles(
        self, engine: Any, name: str, ready: Any, *,
        peer_ready: Any = None, peer_send: Any = None,
        launch: Any = None, source_send: Any = None,
        send_placeholders: Any = None, transport: Any = None,
    ):
        arguments = dict(
            peer_ready=peer_ready, peer_send=peer_send, launch=launch,
            source_send=source_send, send_placeholders=send_placeholders,
        )
        if transport is not None:
            arguments["transport"] = transport
        return self.program.start_with_handles(engine, name, ready, **arguments)


@dataclass(frozen=True)
class ParallelProgram:
    programs: tuple[Program, ...]

    def start(self, engine, name, ready, *, owner=""):
        if len(self.programs) == 1:
            return self.programs[0].start(engine, f"{name}:group0", ready, owner=owner)
        comp_cache = getattr(engine, "_compiled_programs", None)
        if comp_cache is not None and self.programs:
            hit = comp_cache.get(id(self))
            if hit is not None and hit[0] is self:
                plan = hit[1]
            else:
                all_ops = []
                branch_sinks = []
                next_slot = 1
                ok = True
                any_root_delay = False
                for index, program in enumerate(self.programs):
                    if not isinstance(program, KernelProgram):
                        ok = False
                        break
                    compiled = _compile_leaf_kernel_ops(
                        engine, program, f":group{index}", next_slot
                    )
                    if compiled is None:
                        ok = False
                        break
                    b_ops, b_sink, next_slot, _, b_root_delay = compiled
                    if b_root_delay:
                        any_root_delay = True
                    all_ops.extend(b_ops)
                    branch_sinks.append(b_sink)
                if ok:
                    unique_sinks = tuple(dict.fromkeys(branch_sinks)) or (0,)
                    join_slot = next_slot
                    pred_m = 1 if len(unique_sinks) == 1 else 2
                    pred_a = unique_sinks[0] if pred_m == 1 else unique_sinks
                    all_ops.append(
                        (":complete", engine._empty_task, None, None, pred_m, pred_a)
                    )
                    plan = (tuple(all_ops), join_slot, any_root_delay)
                else:
                    plan = False
                if len(comp_cache) >= _COMPILED_CACHE_LIMIT:
                    comp_cache.clear()
                comp_cache[id(self)] = (self, plan)
            if plan is not False:
                plan_ops, sink_slot, any_root_delay = plan
                if not any_root_delay or (
                    ready.__class__ is not tuple
                    and ready.time is not None
                    and (not engine._trace_maxlen or getattr(engine, "_compact_step", False))
                    and not engine._periodic_runs
                    and not engine.occurrence_metrics
                ):
                    ev = _run_compiled_ops(engine, name, ready, owner, plan_ops)
                    return ev[sink_slot]
        events = tuple(
            program.start(engine, f"{name}:group{index}", ready, owner=owner)
            for index, program in enumerate(self.programs)
        )
        return engine.submit(name + ":complete", after=events or (ready,), owner=owner)


def _declares_no_work(stage) -> bool:
    """True when a Stage owns no service demand, time, buffer or supply."""
    return (
        not stage.keep_empty and not any(stage.work.values())
        and stage.latency <= 0.0 and not stage.acquire and not stage.free
        and not any(stage.caps.values()) and not stage.coupled
        and stage.priority == 0 and stage.fragment is None
        and not stage.metadata
    )


def _is_pure_root_delay(stage, all_consumed: set[str], stages) -> bool:
    """True when a root stage is a pure clock delay before sole-dep stages."""
    return (
        not stage.after and stage.name in all_consumed and not stage.keep_empty
        and not any(stage.work.values()) and stage.latency > 0.0
        and not stage.acquire and not stage.free and not any(stage.caps.values())
        and not stage.coupled and stage.priority == 0 and stage.program is None
        and stage.fragment is None and not stage.previous and not stage.metadata
        and all(st.after == (stage.name,) for st in stages if stage.name in st.after)
    )


def _compile_leaf_kernel_ops(
    engine, prog: KernelProgram, group_prefix: str, next_slot: int
):
    """Compile a leaf KernelProgram into flat prepared submission instructions."""
    stages = prog.stages
    for s in stages:
        if s.program is not None or s.fragment is not None or s.metadata or s.deliveries or s.latency_from_first_service:
            return None
    all_consumed = {k for st in stages for k in st.after}
    slot_by_name: dict[str, int] = {}
    delay_by_name: dict[str, float] = {}
    consumed: set[str] = set()
    has_acquire = False
    has_root_delay = False
    ops = []
    for s in stages:
        if _is_pure_root_delay(s, all_consumed, stages):
            delay_by_name[s.name] = s.latency
            has_root_delay = True
            continue
        from_delay = None
        if s.after:
            consumed.update(s.after)
            if len(s.after) == 1 and s.after[0] in delay_by_name:
                from_delay = delay_by_name[s.after[0]]
                preds = (0,)
            else:
                preds = tuple(slot_by_name[k] for k in s.after)
        else:
            preds = (0,)
        if from_delay is None and _declares_no_work(s) and len(preds) == 1:
            slot_by_name[s.name] = preds[0]
            continue
        static = engine.prepared_stage(s, s.acquire, s.free)
        task_suf = f"{group_prefix}:{s.name}"
        if s.acquire:
            has_acquire = True
            acq_specs = tuple(
                (f"{group_prefix}:{a.name}", a.pool, a.size, a.units)
                for a in s.acquire
            )
        else:
            acq_specs = None
        free_specs = (
            tuple(f"{group_prefix}:{a.name}" for a in s.free) if s.free else None
        )
        if from_delay is not None:
            pred_m, pred_a = 0, from_delay
        elif len(preds) == 1:
            pred_m, pred_a = 1, preds[0]
        else:
            pred_m, pred_a = 2, preds
        my_slot = next_slot
        next_slot += 1
        slot_by_name[s.name] = my_slot
        ops.append((task_suf, static, acq_specs, free_specs, pred_m, pred_a))
    term_slots = [slot_by_name[s.name] for s in stages if s.name not in consumed]
    if prog.minimum_ms:
        obs_prep = engine._prepare(
            f"{group_prefix}:observation", None, latency=prog.minimum_ms
        )
        obs_slot = next_slot
        next_slot += 1
        ops.append((f"{group_prefix}:observation", obs_prep, None, None, 1, 0))
        term_slots.append(obs_slot)
    unique_terms = tuple(dict.fromkeys(term_slots)) or (0,)
    if len(unique_terms) == 1 and not prog.minimum_ms:
        sink_slot = unique_terms[0]
    else:
        join_slot = next_slot
        next_slot += 1
        pred_m = 1 if len(unique_terms) == 1 else 2
        pred_a = unique_terms[0] if pred_m == 1 else unique_terms
        ops.append((f"{group_prefix}:complete", engine._empty_task, None, None, pred_m, pred_a))
        sink_slot = join_slot
    return (
        tuple(ops), sink_slot, next_slot,
        has_acquire or has_root_delay, has_root_delay,
    )


def _build_scoped_allocs(name: str, specs) -> tuple[Allocation, ...]:
    if len(specs) == 1:
        a0 = specs[0]
        return (Allocation(name + a0[0], a0[1], a0[2], a0[3]),)
    return tuple(Allocation(name + s, p, sz, u) for s, p, sz, u in specs)


def _build_scoped_free_names(name: str, specs) -> tuple[str, ...]:
    if len(specs) == 1:
        return (name + specs[0],)
    return tuple(name + s for s in specs)


def _run_compiled_ops(engine, name: str, ready, owner: str, plan_ops):
    sub_prep = engine._submit_prepared
    now = engine.now
    ready_is_tuple = ready.__class__ is tuple
    ev = [None] if ready_is_tuple else [ready]
    ev_append = ev.append
    for task_suf, static, acq_specs, free_specs, pred_m, pred_a in plan_ops:
        rel = None
        if pred_m == 1:
            after = ready if (ready_is_tuple and pred_a == 0) else ev[pred_a]
        elif pred_m == 0:
            after = ()
            rel = now + pred_a
        elif ready_is_tuple and 0 in pred_a:
            after = tuple([*ready, *[ev[i] for i in pred_a if i != 0]])
        else:
            after = tuple(ev[i] for i in pred_a)
        if acq_specs is None and free_specs is None:
            ev_append(sub_prep(static, name + task_suf, after=after, release=rel, owner=owner))
        else:
            acq = _build_scoped_allocs(name, acq_specs) if acq_specs is not None else ()
            fr = _build_scoped_free_names(name, free_specs) if free_specs is not None else ()
            ev_append(sub_prep(
                static, name + task_suf, after=after, release=rel, owner=owner,
                acquire_override=acq, free_override=fr,
            ))
    return ev


@dataclass(frozen=True)
class KernelProgram:
    stages: tuple[Stage, ...]
    minimum_ms: float = 0.0

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        comp_cache = getattr(engine, "_compiled_programs", None)
        if comp_cache is not None:
            hit = comp_cache.get(id(self))
            if hit is not None and hit[0] is self:
                plan = hit[1]
            else:
                compiled = _compile_leaf_kernel_ops(engine, self, "", 1)
                if compiled is not None:
                    c_ops, c_sink, _, has_acq_or_delay, has_root_delay = compiled
                    single_fast = None
                    delay_fast = None
                    if (
                        len(c_ops) == 1 and c_sink == 1
                        and c_ops[0][2] is None and c_ops[0][3] is None
                    ):
                        if not has_root_delay and c_ops[0][4] == 1 and c_ops[0][5] == 0:
                            single_fast = (c_ops[0][0], c_ops[0][1])
                        elif c_ops[0][4] == 0:
                            delay_fast = (c_ops[0][0], c_ops[0][1], c_ops[0][5])
                    plan = (
                        c_ops, c_sink, single_fast, has_acq_or_delay,
                        has_root_delay, delay_fast,
                    )
                else:
                    plan = False
                if len(comp_cache) >= _COMPILED_CACHE_LIMIT:
                    comp_cache.clear()
                comp_cache[id(self)] = (self, plan)
            if plan is not False:
                (
                    plan_ops, sink_slot, single_fast, _has_ad,
                    has_root_delay, delay_fast,
                ) = plan
                if single_fast is not None:
                    return engine._submit_prepared(
                        single_fast[1], name + single_fast[0], after=ready, owner=owner,
                    )
                if not has_root_delay or (
                    ready.__class__ is not tuple
                    and ready.time is not None
                    and (not engine._trace_maxlen or getattr(engine, "_compact_step", False))
                    and not engine._periodic_runs
                    and not engine.occurrence_metrics
                ):
                    if delay_fast is not None:
                        return engine._submit_prepared(
                            delay_fast[1], name + delay_fast[0],
                            release=engine.now + delay_fast[2], owner=owner,
                        )
                    if not plan_ops:
                        return ready
                    ev = _run_compiled_ops(engine, name, ready, owner, plan_ops)
                    return ev[sink_slot]
        events: dict[str, Event] = {}
        consumed: set[str] = set()

        def scoped(items):
            return scope_allocations(name, items)

        for stage in self.stages:
            if stage.after:
                consumed.update(stage.after)
            predecessors = tuple(events[key] for key in stage.after) or (ready,)
            if (
                stage.program is None and _declares_no_work(stage)
                and len(predecessors) == 1
            ):
                events[stage.name] = predecessors[0]
                continue
            events[stage.name] = stage.submit(
                engine, f"{name}:{stage.name}", predecessors, scoped, owner
            )
        terminal_list = [
            events[stage.name] for stage in self.stages if stage.name not in consumed
        ]
        if self.minimum_ms:
            obs = engine.submit(
                name + ":observation", after=(ready,), latency=self.minimum_ms, owner=owner,
            )
            events["observation"] = obs
            terminal_list.append(obs)
        terminal = tuple(dict.fromkeys(terminal_list)) or (ready,)
        if len(terminal) == 1 and not self.minimum_ms:
            return terminal[0]
        return engine.submit(name + ":complete", after=terminal, owner=owner)
