"""Composable kernel recipes independent of model names.

[Role] Translate physical operator work into finite resource programs and
  lower declarative fragments onto the existing Stage/Engine contract.
[Invariants] A recipe changes stages, work, supply and lifetimes explicitly.
  A measured kernel span is a completion constraint, never invented FLOPs.
  Repeated tiles are generated through a bounded window, including fill/drain.
  Acquired allocations are held until completion; coupled stages progress in
  fixed proportion; priority is admission order and never preempts.
[Timeouts & Locks] Pure virtual execution; live tiles are bounded by slots.
[Side-Effect Policy] Submits tasks only to the caller's Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS
from mindie_sim.schema.operators import ProgramFragment
from .engine import Allocation, Engine, Event
from .ownership import unit_group


_EMPTY_MAPPING = MappingProxyType({})
#: Default live-stage bound for coalescing a repeated finite pipeline window.
PIPELINE_UNROLL_LIMIT = 64


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
    if n == 3:
        a0, a1, a2 = items
        return (
            Allocation(pfx + a0.name, a0.pool, a0.size, a0.units),
            Allocation(pfx + a1.name, a1.pool, a1.size, a1.units),
            Allocation(pfx + a2.name, a2.pool, a2.size, a2.units),
        )
    if n == 4:
        a0, a1, a2, a3 = items
        return (
            Allocation(pfx + a0.name, a0.pool, a0.size, a0.units),
            Allocation(pfx + a1.name, a1.pool, a1.size, a1.units),
            Allocation(pfx + a2.name, a2.pool, a2.size, a2.units),
            Allocation(pfx + a3.name, a3.pool, a3.size, a3.units),
        )
    return tuple([Allocation(pfx + a.name, a.pool, a.size, a.units) for a in items])


@dataclass(frozen=True)
class Stage:
    """One finite stage on the existing Engine's resources.

    ``work`` is issued service demand.  ``acquire``/``free`` are grants and
    releases; an acquired allocation is held until the task completes,
    including ``latency``, so waiting retains only the declared slot/buffer
    and never adds work.  ``coupled=True`` requests fixed-proportion progress
    across all resources of this stage.  ``priority`` is non-preemptive
    admission order; the Engine never displaces a running task.
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
    #: Declared native unit per resource, when the caller knows it.  Empty
    #: keeps the historical untyped Stage path; Engine still receives only
    #: numeric work.  This is a declaration, never a conversion factor.
    work_units: Mapping[str, str] = field(default_factory=dict)
    #: Admission order for FIFO resources (lower first).  The existing Engine
    #: has no preemptive priority; this maps to deterministic submission order
    #: and never changes work or capacity.
    priority: int = 0
    #: Complete declaration when this Stage is a single-static projection.
    #: T04 consumes this for repeats/conditional allocations/state/policy that
    #: a plain Stage cannot execute; ``stage_from_fragment`` refuses to project
    #: those semantics rather than silently dropping them.
    fragment: ProgramFragment | None = None
    #: Symmetric owner multiplicity represented by a merged coupled cohort.
    cohort_size: int = 1
    #: Keep an explicit zero-work stage marker (e.g. TiledMatmul launch setup).
    keep_empty: bool = False

    def __post_init__(self):
        # Empty/default identity is immutable and shared. Nonempty
        # caller-owned dicts still receive a proper MappingProxyType copy;
        # an existing proxy is already immutable. This avoids three new
        # proxy+dict objects for the common empty caps/work/placement case.
        work = self.work
        if not work:
            work = _EMPTY_MAPPING
        else:
            # MappingProxyType is a read-only view, not frozen ownership. Copy
            # every nonempty caller mapping, including an ordinary proxy.
            work = MappingProxyType(dict(work))
        caps = self.caps
        if not caps:
            caps = _EMPTY_MAPPING
        else:
            caps = MappingProxyType(dict(caps))
        placement = self.placement
        if not placement:
            placement = _EMPTY_MAPPING
        else:
            placement = MappingProxyType(
                {key: units if units.__class__ is tuple else tuple(units)
                 for key, units in placement.items()})
        units = self.work_units
        if not units:
            units = _EMPTY_MAPPING
        else:
            units = MappingProxyType(dict(units))
            for resource, unit in units.items():
                if not isinstance(resource, str) or not isinstance(unit, str) or not unit:
                    raise ModelingError(f"stage_work_unit_invalid:{self.name}")
        object.__setattr__(self, "work", work)
        object.__setattr__(self, "caps", caps)
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "work_units", units)
        if not (self.priority.__class__ is int and self.priority >= 0
                and self.cohort_size.__class__ is int and self.cohort_size >= 1
                and self.keep_empty.__class__ is bool
                and self.fragment is None):
            if not isinstance(self.priority, int) or isinstance(self.priority, bool) \
                    or self.priority < 0:
                raise ModelingError(f"stage_priority_invalid:{self.name}")
            if (not isinstance(self.cohort_size, int)
                    or isinstance(self.cohort_size, bool)
                    or self.cohort_size < 1):
                raise ModelingError(f"stage_cohort_size_invalid:{self.name}")
            if not isinstance(self.keep_empty, bool):
                raise ModelingError(f"stage_keep_empty_invalid:{self.name}")
            if self.fragment is not None and not isinstance(self.fragment, ProgramFragment):
                raise ModelingError(f"stage_fragment_invalid:{self.name}")
        if self.after.__class__ is not tuple:
            object.__setattr__(self, "after", tuple(self.after))
        if self.acquire.__class__ is not tuple:
            object.__setattr__(self, "acquire", tuple(self.acquire))
        if self.free.__class__ is not tuple:
            object.__setattr__(self, "free", tuple(self.free))
        if self.previous.__class__ is not tuple:
            object.__setattr__(self, "previous", tuple(self.previous))

    def _with_buffers(self, acquire: tuple[Allocation, ...],
                      free: tuple[Allocation, ...]) -> Stage:
        """Clone an already-validated Stage replacing only tuple acquire/free."""
        clone = object.__new__(Stage)
        object.__setattr__(clone, "name", self.name)
        object.__setattr__(clone, "work", self.work)
        object.__setattr__(clone, "after", self.after)
        object.__setattr__(clone, "latency", self.latency)
        object.__setattr__(clone, "caps", self.caps)
        object.__setattr__(clone, "acquire", acquire)
        object.__setattr__(clone, "free", free)
        object.__setattr__(clone, "program", self.program)
        object.__setattr__(clone, "placement", self.placement)
        object.__setattr__(clone, "previous", self.previous)
        object.__setattr__(clone, "coupled", self.coupled)
        object.__setattr__(clone, "work_units", self.work_units)
        object.__setattr__(clone, "priority", self.priority)
        object.__setattr__(clone, "fragment", self.fragment)
        object.__setattr__(clone, "cohort_size", self.cohort_size)
        object.__setattr__(clone, "keep_empty", self.keep_empty)
        object.__setattr__(clone, "_proto_stage", getattr(self, "_proto_stage", self))
        return clone

    def prepare(self, engine, name, scoped):
        """Freeze static stage validation on the public Engine surface.

        The returned object is Engine-owned when the causal Engine provides
        ``prepare_stage``. A plain Engine without that public method receives a
        local immutable declaration; no private Engine method is touched.
        """
        prepare = getattr(engine, "prepare_stage", None)
        acquire = scoped(self.acquire)
        free = () if self.program else scoped(self.free)
        if callable(prepare):
            return prepare(
                name, self.work, latency=self.latency, caps=self.caps,
                acquire=acquire, free=free, placement=self.placement,
                coupled=self.coupled, priority=self.priority,
                cohort_size=self.cohort_size)
        return _PreparedStageDeclaration(
            name, self.work, self.latency, self.caps, acquire, free,
            self.placement, self.coupled, self.priority, self.cohort_size)

    def submit(self, engine, name, predecessors, scoped, owner, prepared=None):
        # A stage that declares no buffer keeps the empty tuple; the scoping
        # callable may build names/bindings per occurrence, so it is only
        # entered when the declaration really owns allocations.
        acquire = scoped(self.acquire) if self.acquire else ()
        free = scoped(self.free) if self.free and self.program is None else ()
        if (self.program is not None and not self.work and self.latency <= 0.0
                and not self.acquire and not self.free and not self.caps
                and not self.coupled and self.priority == 0
                and not self.keep_empty
                and self.fragment is None and len(predecessors) == 1):
            # Pure pass-through wrapper: the stage owns no work, time, buffer or
            # supply of its own, so its only role is to name one nested program
            # behind one predecessor.  Starting the nested program from that
            # same event keeps every inner task name, work, dependency and
            # instant, and removes a zero-work task from the causal front.
            return self.program.start(
                engine, name + ":program", predecessors[0], owner=owner)
        if prepared is not None and not isinstance(prepared, _PreparedStageDeclaration):
            sub_fn = getattr(engine, "_submit_prepared", None) or engine.submit_prepared
            event = sub_fn(prepared, name, after=predecessors, owner=owner)
        else:
            prepare = getattr(engine, "prepared_stage", None)
            if prepare is not None and not self.program:
                prepared = prepare(self, acquire, free)
                sub_fn = getattr(engine, "_submit_prepared", None) or engine.submit_prepared
                event = sub_fn(prepared, name, after=predecessors, owner=owner)
            else:
                kwargs = {}
                if self.cohort_size != 1:
                    kwargs["cohort_size"] = self.cohort_size
                event = engine.submit(
                    name, self.work, after=predecessors, latency=self.latency,
                    caps=self.caps, acquire=acquire, free=free,
                    placement=self.placement, coupled=self.coupled,
                    priority=self.priority, owner=owner, **kwargs)
        if self.program is not None:
            event = self.program.start(engine, name + ":program", event, owner=owner)
            if self.free:
                event = engine.submit(name + ":free", after=(event,), free=scoped(self.free), owner=owner)
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


def stage_from_fragment(fragment: ProgramFragment, *, name: str = "",
                        after: tuple[str, ...] = (), latency: float = 0.0) -> Stage:
    """Project a fragment's finite work onto one existing Stage.

    This is a strict single-static projection, not an executor: repeats,
    conditional/program-scoped allocations, state/field effects, resource
    policy and semantic attrs cannot be represented by a plain Stage, so they
    raise instead of silently under-counting or becoming unconditional.  The
    returned Stage retains the complete declaration in ``Stage.fragment`` for
    T04.  No submission or time advance happens here.
    """
    if not isinstance(fragment, ProgramFragment):
        raise ModelingError("stage_from_fragment_requires_program_fragment")
    blockers = fragment.stage_projection_blockers
    if blockers:
        raise ModelingError(
            "stage_from_fragment_unsupported:" + ",".join(blockers))
    acquire = tuple(Allocation(effect.name, effect.pool, effect.size, effect.units)
                    for effect in fragment.allocations
                    if effect.action == "acquire"
                    and effect.lifetime in {"stage", "slot"})
    free = tuple(Allocation(effect.name, effect.pool, effect.size, effect.units)
                 for effect in fragment.allocations
                 if effect.action == "release"
                 and effect.lifetime in {"stage", "slot"})
    units = {item.resource: item.unit for item in fragment.work.issued()}
    return Stage(name or fragment.name, fragment.service_work, after=after,
                 latency=latency, acquire=acquire, free=free,
                 work_units=units, priority=fragment.priority,
                 coupled=fragment.coupled, fragment=fragment)


class Program(Protocol):
    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event: ...


@dataclass(frozen=True)
class NativeProgram:
    """Canonical executable native program plus its exact source ledger.

    The contract preserves the native owner's own ``start`` /
    ``start_with_handles`` readiness, peer sends, send placeholders, launch
    handles and transport bindings.  It stores no duration: a caller binds
    and measures the actual completion Event on the shared Engine instead of
    reusing a precomputed time.
    """

    program: Any
    work: tuple[tuple[str, Fraction], ...]
    fidelity: str = "native_program"

    def start(self, engine: Any, name: str, ready: Any, *,
              peer_ready: Any = None, peer_send: Any = None,
              launch: Any = None, source_send: Any = None,
              send_placeholders: Any = None, transport: Any = None) -> Any:
        if (peer_ready is None and peer_send is None and launch is None
                and source_send is None and send_placeholders is None
                and transport is None):
            return self.program.start(engine, name, ready)
        arguments = dict(
            peer_ready=peer_ready, peer_send=peer_send, launch=launch,
            source_send=source_send, send_placeholders=send_placeholders)
        if transport is not None:
            arguments["transport"] = transport
        return self.program.start(engine, name, ready, **arguments)

    def start_with_handles(self, engine: Any, name: str, ready: Any, *,
                           peer_ready: Any = None, peer_send: Any = None,
                           launch: Any = None, source_send: Any = None,
                           send_placeholders: Any = None,
                           transport: Any = None):
        arguments = dict(
            peer_ready=peer_ready, peer_send=peer_send, launch=launch,
            source_send=source_send, send_placeholders=send_placeholders)
        if transport is not None:
            arguments["transport"] = transport
        return self.program.start_with_handles(
            engine, name, ready, **arguments)


@dataclass(frozen=True)
class ParallelProgram:
    programs: tuple[Program, ...]

    def start(self, engine, name, ready, *, owner=""):
        if len(self.programs) == 1:
            # A single branch already owns the whole completion boundary; a
            # zero-work join at the same instant would only add a task.
            return self.programs[0].start(
                engine, f"{name}:group0", ready, owner=owner)
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
                        engine, program, f":group{index}", next_slot)
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
                        (":complete", engine._empty_task, None, None, pred_m, pred_a))
                    plan = (tuple(all_ops), join_slot, any_root_delay)
                else:
                    plan = False
                if len(comp_cache) >= 4096:
                    comp_cache.clear()
                comp_cache[id(self)] = (self, plan)
            if plan is not False:
                plan_ops, sink_slot, any_root_delay = plan
                if not any_root_delay or (
                    ready.__class__ is not tuple
                    and ready.time is not None
                    and (not engine._trace_maxlen
                         or getattr(engine, "_compact_step", False))
                    and not engine._periodic_runs
                    and not engine.occurrence_metrics
                ):
                    ev = _run_compiled_ops(engine, name, ready, owner, plan_ops)
                    return ev[sink_slot]
        events = tuple(program.start(engine, f"{name}:group{index}", ready, owner=owner)
                       for index, program in enumerate(self.programs))
        return engine.submit(name + ":complete", after=events or (ready,), owner=owner)


def _declares_no_work(stage) -> bool:
    """True when a Stage owns no service demand, time, buffer or supply."""
    return (not stage.keep_empty and not any(stage.work.values())
            and stage.latency <= 0.0 and not stage.acquire and not stage.free
            and not any(stage.caps.values()) and not stage.coupled
            and stage.priority == 0 and stage.fragment is None)


def _is_pure_root_delay(stage, all_consumed: set[str], stages) -> bool:
    """True when a root stage is a pure clock delay before sole-dep stages."""
    return (
        not stage.after
        and stage.name in all_consumed
        and not stage.keep_empty
        and not any(stage.work.values())
        and stage.latency > 0.0
        and not stage.acquire
        and not stage.free
        and not any(stage.caps.values())
        and not stage.coupled
        and stage.priority == 0
        and stage.program is None
        and stage.fragment is None
        and not stage.previous
        and all(st.after == (stage.name,) for st in stages if stage.name in st.after)
    )


def _compile_leaf_kernel_ops(engine, prog: KernelProgram, group_prefix: str,
                             next_slot: int):
    """Compile a leaf KernelProgram into flat prepared submission instructions."""
    stages = prog.stages
    for s in stages:
        if s.program is not None or s.fragment is not None:
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
            acq_specs = tuple((f"{group_prefix}:{a.name}", a.pool, a.size, a.units)
                              for a in s.acquire)
        else:
            acq_specs = None
        free_specs = (tuple(f"{group_prefix}:{a.name}" for a in s.free)
                      if s.free else None)
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
    term_slots = [slot_by_name[s.name] for s in stages
                  if s.name not in consumed]
    if prog.minimum_ms:
        obs_prep = engine._prepare(
            f"{group_prefix}:observation", None, latency=prog.minimum_ms)
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
        if len(unique_terms) == 1:
            pred_m, pred_a = 1, unique_terms[0]
        else:
            pred_m, pred_a = 2, unique_terms
        ops.append((f"{group_prefix}:complete", engine._empty_task,
                    None, None, pred_m, pred_a))
        sink_slot = join_slot
    return (
        tuple(ops), sink_slot, next_slot,
        has_acquire or has_root_delay, has_root_delay,
    )


def _build_scoped_allocs(name: str, specs) -> tuple[Allocation, ...]:
    n = len(specs)
    if n == 1:
        a0 = specs[0]
        return (Allocation(name + a0[0], a0[1], a0[2], a0[3]),)
    if n == 4:
        a0, a1, a2, a3 = specs
        return (
            Allocation(name + a0[0], a0[1], a0[2], a0[3]),
            Allocation(name + a1[0], a1[1], a1[2], a1[3]),
            Allocation(name + a2[0], a2[1], a2[2], a2[3]),
            Allocation(name + a3[0], a3[1], a3[2], a3[3]),
        )
    return tuple([Allocation(name + s, p, sz, u) for s, p, sz, u in specs])


def _build_scoped_free_names(name: str, specs) -> tuple[str, ...]:
    n = len(specs)
    if n == 1:
        return (name + specs[0],)
    if n == 4:
        return (name + specs[0], name + specs[1], name + specs[2], name + specs[3])
    return tuple([name + s for s in specs])


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
            after = tuple([ev[i] for i in pred_a])
        if acq_specs is None and free_specs is None:
            ev_append(sub_prep(static, name + task_suf, after=after,
                               release=rel, owner=owner))
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
                    if (len(c_ops) == 1 and c_sink == 1
                            and c_ops[0][2] is None and c_ops[0][3] is None):
                        if not has_root_delay and c_ops[0][4] == 1 and c_ops[0][5] == 0:
                            single_fast = (c_ops[0][0], c_ops[0][1])
                        elif c_ops[0][4] == 0:
                            delay_fast = (c_ops[0][0], c_ops[0][1], c_ops[0][5])
                    plan = (c_ops, c_sink, single_fast, has_acq_or_delay,
                            has_root_delay, delay_fast)
                else:
                    plan = False
                if len(comp_cache) >= 4096:
                    comp_cache.clear()
                comp_cache[id(self)] = (self, plan)
            if plan is not False:
                (plan_ops, sink_slot, single_fast, _has_ad,
                 has_root_delay, delay_fast) = plan
                if single_fast is not None:
                    return engine._submit_prepared(
                        single_fast[1], name + single_fast[0],
                        after=ready, owner=owner,
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
        # Existing public callers consume stage-labelled receipts (``a:move``),
        # so the ordinary entry keeps the public Stage route; Engine.submit
        # normalizes each physical stage through the same unique kernel.
        # ``program_catalog.submit_program`` is the explicit one-catalog route
        # for callers that consume occurrence receipts instead of task labels.
        events: dict[str, Event] = {}
        consumed: set[str] = set()

        def scoped(items):
            return scope_allocations(name, items)
        for stage in self.stages:
            if stage.after:
                consumed.update(stage.after)
            predecessors = tuple(events[key] for key in stage.after) or (ready,)
            if (stage.program is None and _declares_no_work(stage)
                    and len(predecessors) == 1):
                # A declared but physically empty boundary owns no work, time
                # or buffer: alias it to its single predecessor so downstream
                # references keep the exact same event instant.  A bare
                # placement is inert because no service demand can be placed.
                events[stage.name] = predecessors[0]
                continue
            events[stage.name] = stage.submit(
                engine, f"{name}:{stage.name}", predecessors, scoped, owner)
        terminal_list = [events[stage.name] for stage in self.stages
                         if stage.name not in consumed]
        if self.minimum_ms:
            obs = engine.submit(
                name + ":observation", after=(ready,), latency=self.minimum_ms, owner=owner,
            )
            events["observation"] = obs
            terminal_list.append(obs)
        terminal = tuple(dict.fromkeys(terminal_list)) or (ready,)
        if len(terminal) == 1 and not self.minimum_ms:
            # A single sink already owns the exact completion instant; an extra
            # zero-work join task would add frontier churn without physical work.
            return terminal[0]
        return engine.submit(name + ":complete", after=terminal, owner=owner)


@dataclass(frozen=True)
class PipelineProgram:
    """A repeated stage DAG with explicit stream ordering and buffer lifetime.

    ``tile(index)`` supplies real tail-tile work. Every stage name denotes a
    serial instruction stream; distinct names may overlap. Buffer slots limit
    how many tiles can be resident. No steady-state II replaces finite work.
    """

    count: int
    slots: int
    tile: Callable[[int], Sequence[Stage]]
    minimum_ms: float = 0.0
    #: Provenance of ``minimum_ms`` when it came from a declared service rule.
    minimum_provenance: Mapping = field(default_factory=dict)
    setup_ms: float = 0.0
    #: Only a producer that knows the geometry may declare tile homogeneity.
    #: An arbitrary callback keeps the detailed finite window.
    homogeneous: bool = False
    #: Optional producer-owned registry populated at allocation construction.
    producer_bindings: object = None
    #: Opt-in steady interior tile compression.  Only a producer whose tile
    #: window is a bounded, *strictly serial* repetition of one identical stage
    #: set may set it; see :func:`interior_fold_plan` for the exactness boundary
    #: and the measured counterexample that keeps it off by default.
    interior_fold: bool = False
    #: Whether ``start()`` automatically executes ``self.template()`` when
    #: ``homogeneous`` is True.  ``False`` keeps the unrolled per-tile reference
    #: execution on ``start()`` while still allowing explicit ``template()`` reads.
    template_start: bool = True

    def template(self, *, owner: int = 0, panels_per_output: int = 1,
                 k_tail: bool = False, key_of=None,
                 bindings=()) -> PipelineTemplate:
        """Build the exact finite phase sequence for this callback.

        A plain ``tile(index)`` callback can encode per-wave geometry (M/N/K
        tails), so the producer's declared homogeneity only permits sharing the
        per-panel class shape, not silently replacing every interior wave by one
        representative. First/middle/last phases are therefore emitted per
        output wave with their real work; large regular GEMM producers expose a
        separate bounded indexed class catalog (``matrix.TiledMatmul``) instead.
        """
        if not self.homogeneous:
            raise ModelingError("pipeline_template_requires_declared_homogeneity")
        if (isinstance(panels_per_output, bool)
                or not isinstance(panels_per_output, int) or panels_per_output < 1):
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
            # First and last K-panel own the accumulator lifetime edges; middle
            # panels carry neither edge but retain their real work.
            stages = tuple(self.tile(base))
            tile_calls += 1
            phases.append(PipelinePhase("first", stages, 1, key=key))
            middle = panels_per_output - 2
            if middle > 0:
                stages = tuple(self.tile(base + 1))
                tile_calls += 1
                phases.append(PipelinePhase(
                    "middle", stages, middle,
                    key=tuple(key_of(base + 1)) if key_of is not None else ()))
            stages = tuple(self.tile(base + panels_per_output - 1))
            tile_calls += 1
            phases.append(PipelinePhase(
                "last", stages, 1,
                key=tuple(key_of(base + panels_per_output - 1))
                if key_of is not None else ()))
        return PipelineTemplate(
            owner, self.slots, tuple(phases),
            setup_ms=self.setup_ms, minimum_ms=self.minimum_ms,
            bindings=tuple(dict(row) for row in bindings),
            tile_factory_calls=tile_calls,
            phase_objects=len(phases),
            stage_objects=sum(len(phase.stages) for phase in phases),
            static_phase_classes=len(phases),
            dynamic_instances=sum(phase.count for phase in phases))

    def start_template(self, engine: Engine, name: str, ready: Event, *,
                       owner: str = "", panels_per_output: int = 1,
                       k_tail: bool = False, key_of=None,
                       bindings=()) -> Event:
        """Submit a declared template, or the correct detailed finite window."""
        if not self.homogeneous:
            return self.start(engine, name, ready, owner=owner)
        template = self.template(panels_per_output=panels_per_output,
                                 k_tail=k_tail, key_of=key_of,
                                 bindings=bindings)
        return PipelineTemplateProgram(template).start(
            engine, name, ready, owner=owner)

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        if self.count < 0 or self.slots < 1:
            raise ValueError("pipeline_requires_nonnegative_tiles_and_positive_slots")
        if self.homogeneous and self.template_start:
            # Immutable bounded class catalog; the repeating consumer owns the
            # fresh cursor, slots, previous-stage tails and generations.
            return PipelineTemplateProgram(self.template()).start(
                engine, name, ready, owner=owner)
        if self.interior_fold:
            folded = fold_interior_pipeline(self)
            if folded is not None:
                # Steady interior tile compression: the same finite physical
                # work in fewer tasks, never a sampled duration or a rate.
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
                # Tile zero has no preceding tile: a ``previous`` edge must not
                # bind to a stage already issued inside this same tile.
                previous: dict[str, Event] = {}
                stages = tuple(self.tile(0))
                consumed: set[str] = set()
                for stage in stages:
                    if stage.after:
                        consumed.update(stage.after)
                    deps = [local[key] for key in stage.after]
                    deps.extend(previous[key] for key in stage.previous
                                if key in previous)
                    if (stage.program is None and not stage.acquire
                            and not stage.free and _declares_no_work(stage)):
                        ev = _zero_work_alias(deps, trigger)
                        if ev is not None:
                            local[stage.name] = ev
                            continue
                    ev = stage.submit(engine, f"{name}:tile0:{stage.name}",
                                      tuple(deps) or (trigger,), scoped, owner)
                    local[stage.name] = ev
                sinks = tuple(dict.fromkeys(
                    local[stage.name] for stage in stages
                    if stage.name not in consumed))
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
                    engine.submit(f"{name}:tile0:retire", after=sinks,
                                  on_complete=lambda _e: engine.resolve(complete_one),
                                  owner=owner)

            engine.on_complete(ready, begin_one)
            return complete_one

        # Opaque callback boundary: the same Engine executes real per-instance
        # work in a slot-bounded window. It is never approximated by a sampled
        # duration or an all-panel event history.
        complete = engine.future(name + ":pipeline_complete")
        setup = (engine.submit(name + ":setup", after=(ready,),
                               latency=self.setup_ms, owner=owner)
                 if self.setup_ms else ready)
        observation = (engine.submit(name + ":span", after=(setup,),
                                     latency=self.minimum_ms, owner=owner)
                       if self.minimum_ms else None)
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
                    deps.extend(previous[key] for key in stage.previous
                                if key in previous)
                    if stage.name in stage_tail:
                        deps.append(stage_tail[stage.name])
                    task_name = f"{name}:tile{index}:{stage.name}"
                    if (stage.program is None and not stage.acquire
                            and not stage.free and _declares_no_work(stage)):
                        ev = _zero_work_alias(deps, setup)
                        if ev is not None:
                            local[stage.name] = stage_tail[stage.name] = ev
                            continue
                    event = stage.submit(engine, task_name, tuple(deps) or (setup,),
                                         scoped, owner)
                    local[stage.name] = stage_tail[stage.name] = event
                sinks = tuple(dict.fromkeys(
                    local[stage.name] for stage in stages
                    if stage.name not in consumed)) or (setup,)
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
                # Multi-slot pipelines retain the explicit per-tile retire
                # marker that ``tests/test_attention_decode`` inspects.
                engine.submit(f"{name}:tile{index}:retire",
                              after=sinks,
                              on_complete=retire, owner=owner)
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
            engine.submit(name + ":begin", after=(setup,), on_complete=begin,
                          owner=owner)
        return engine.submit(
            name + ":complete",
            after=(complete, observation) if observation is not None else (complete,),
            owner=owner)


@dataclass(frozen=True)
class PipelinePhase:
    """One repeated phase class of a finite pipeline template.

    ``stages`` is the original Stage set for a representative tile (after,
    previous, work, caps, placement, acquire/free unchanged). ``count`` is how
    many consecutive tiles use this class.  The Engine consumer scopes
    allocation names per task/repeat, so one template Stage set can serve all
    repeats without changing buffer identity or capacity.
    """

    kind: str
    stages: tuple[Stage, ...]
    count: int
    #: Optional immutable periodic key, e.g. attention
    #: ``(valid, pairs, active_rows, slot)``; T04 detector may use it.
    key: tuple = ()
    #: Optional finite slot-residue stage table for one homogeneous phase.
    #: ``stages_for(index)`` selects by absolute tile index.
    residue_stages: tuple[tuple[Stage, ...], ...] = ()
    #: Explicit phase end; a consumer must not issue at/after it.
    end: int | None = None

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("pipeline_phase_requires_kind")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("pipeline_phase_requires_positive_count")
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "key", tuple(self.key))
        object.__setattr__(self, "residue_stages", tuple(
            tuple(group) for group in self.residue_stages))
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
    """Finite phase template for T04's bounded Engine repeat executor.

    With both indexed fields empty, ``phases`` are in execution order and a
    consumer repeats each phase's Stage set ``count`` times with unique task
    names, preserving same-stage ordering, after/previous edges, work/caps,
    placement, and allocation grant/release.  It may only advance a cycle when
    the actual slot phase, competitors, buffer holders and ready/release
    boundaries match; tail phases remain explicit.

    With both ``indexed_panels`` and ``index_of`` present, ``phases`` is a
    small class catalog (one entry per declared geometry class, count 1 and no
    absolute end).  ``tiles`` is the logical panel count and ``index_of(i)``
    selects the class and exclusive phase end for absolute index ``i``.  The
    actual callable's ``indexed_panels_abi`` marker, never either producer
    field alone, decides whether that catalog may be consumed.
    """

    owner: int
    slots: int
    phases: tuple[PipelinePhase, ...]
    setup: tuple[Stage, ...] = ()
    drain: tuple[Stage, ...] = ()
    setup_ms: float = 0.0
    minimum_ms: float = 0.0
    #: Producer-authored, read-only allocation instances covering setup, body
    #: and drain.  No consumer-side name/digit inference is permitted.
    bindings: tuple[dict, ...] = ()
    #: One proven output motif, reused by ``motif_repeats`` output ordinals.
    #: Consumers that do not understand it may use the expanded ``phases``.
    motif_phases: tuple[PipelinePhase, ...] = ()
    motif_repeats: int = 1
    #: Actual producer construction evidence; runtime counters are separate.
    tile_factory_calls: int = 0
    stage_objects: int = 0
    phase_objects: int = 0
    static_phase_classes: int = 0
    dynamic_instances: int = 0
    #: Finite indexed catalog.  Append after all existing positional fields.
    #: ``index_of(i)`` returns only plain integer coordinates:
    #: (phase_id, phase_end_exclusive, output_ordinal, panel_ordinal, slot).
    #: Both fields empty keeps the original ordered ``phases``; both present is
    #: the indexed mode.  A consumer checks the actual callable's ABI marker.
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
            if (isinstance(self.indexed_panels, bool)
                    or not isinstance(self.indexed_panels, int)
                    or self.indexed_panels < 0):
                raise ValueError("pipeline_template_indexed_panels_invalid")
            if not callable(self.index_of):
                raise ValueError("pipeline_template_indexed_index_of_invalid")
            if (isinstance(self.slots, bool)
                    or not isinstance(self.slots, int) or self.slots < 1):
                raise ValueError(
                    "pipeline_template_indexed_requires_positive_slots")
            for phase in self.phases:
                if phase.count != 1:
                    raise ValueError(
                        "pipeline_template_indexed_phase_count_must_be_one")
                if phase.end is not None:
                    raise ValueError(
                        "pipeline_template_indexed_phase_end_must_be_none")
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
                raise ValueError(
                    f"pipeline_indexed_binding_scope_invalid:{name}")
            if "owner" in row and row["owner"] != self.owner:
                raise ValueError(
                    f"pipeline_indexed_binding_owner_mismatch:{name}")
            previous = declared.get(name)
            if previous is not None and previous != scope:
                raise ValueError(
                    f"pipeline_indexed_binding_scope_conflict:{name}")
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
                        "pipeline_indexed_allocation_symbol_unbound:"
                        f"{allocation.name}")

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
            if not isinstance(mapped, tuple) or len(mapped) != 5:
                raise ModelingError("pipeline_index_mapping_invalid")
            phase_id, end, output, panel, slot = mapped
            if (isinstance(phase_id, bool) or not isinstance(phase_id, int)
                    or not 0 <= phase_id < len(self.phases)):
                raise ModelingError("pipeline_index_mapping_invalid")
            if (isinstance(end, bool) or not isinstance(end, int)
                    or not index < end <= self.indexed_panels):
                raise ModelingError("pipeline_index_mapping_invalid")
            if (isinstance(output, bool) or not isinstance(output, int)
                    or output < 0
                    or isinstance(panel, bool) or not isinstance(panel, int)
                    or panel < 0
                    or isinstance(slot, bool) or not isinstance(slot, int)
                    or not 0 <= slot < self.slots):
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
    """True when an indexed finite reader exists on the selected path.

    The marker is published by ``scheduling.start_pipeline_template`` only
    after its indexed reader exists. The compiler in ``program_catalog`` is a
    second real reader, but producer wire selection stays on this published
    callable marker; an unknown or non-int value is not capability evidence.
    """
    try:
        from .scheduling import start_pipeline_template
    except ImportError:
        return False
    version = getattr(start_pipeline_template, "indexed_panels_abi", None)
    return type(version) is int and version == 1


@dataclass(frozen=True)
class PipelineTemplateProgram:
    """Program adapter that submits a finite PipelineTemplate through T04.

    ``compile_pipeline_template`` produces the immutable finite class catalog;
    the caller Engine owns the fresh cursor, slot generations and real work.
    The existing indexed ``scheduling`` reader remains the compatibility
    consumer while the causal Engine is being installed, not a second executor
    and not an expanded all-panel loop.
    """

    template: PipelineTemplate

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        # A nested program or cross-stage allocation inside a finite class needs
        # the repeating consumer's per-tile Stage.submit path; the causal
        # Engine's finite node loop currently executes flat class stages and
        # binds releases only within one stage. Never drop nested work or leak
        # a live grant.
        stages = (*self.template.setup, *self.template.drain,
                  *(stage for phase in self.template.phases
                    for stage in phase.stages))
        nested = any(stage.program is not None for stage in stages)
        allocated = any(stage.acquire or stage.free for stage in stages)
        from .program_catalog import submit_template
        # Indexed producers are explicitly large/regular; the repeating consumer
        # advances at most a slot window. The causal Engine's current finite
        # loop materializes every absolute index, so reserve it for the bounded
        # enumerated (non-indexed) template.
        if (not nested and not allocated
                and self.template.indexed_panels is None
                and callable(getattr(engine, "submit_catalog", None))):
            return submit_template(
                engine, self.template, name=name, ready=ready,
                owner=owner).done
        try:
            from .scheduling import start_pipeline_template
        except ImportError as exc:
            if callable(getattr(engine, "submit_catalog", None)):
                raise ModelingError(
                    "pipeline_template_nested_program_requires_consumer") from exc
            raise ModelingError(
                "pipeline_template_requires_causal_catalog_engine") from exc
        if self.template.indexed_panels is not None:
            version = getattr(start_pipeline_template, "indexed_panels_abi", None)
            if type(version) is not int or version != 1:
                raise ModelingError("pipeline_indexed_consumer_unavailable")
        return start_pipeline_template(engine, self.template, name, ready, owner=owner)


def _transport_identity(value):
    """Stable T09 field projection; never stringifies or drops explicit None."""
    from .topology import TransportOptions
    if isinstance(value, TransportOptions):
        return {
            "chunk_bytes": value.chunk_bytes,
            "hop_startup_us": value.hop_startup_us,
            "receive_capacity_bytes": [
                [int(rank), float(amount)]
                for rank, amount in sorted(value.receive_capacity_bytes)],
            "receive_buffer_ids": [
                [int(rank), str(name)]
                for rank, name in sorted(value.receive_buffer_ids)],
            "receive_capacity_bytes_per_peer": value.receive_capacity_bytes_per_peer,
            "receive_buffer_id": value.receive_buffer_id,
            "max_inflight_per_rank": value.max_inflight_per_rank,
            "max_physical_chunks": value.max_physical_chunks,
        }
    if isinstance(value, dict):
        return {str(key): _transport_identity(item)
                for key, item in value.items()}
    return value


def _stable_identity(value):
    """Make a service identity JSON-serializable without default=str."""
    from .topology import TransportOptions
    if isinstance(value, TransportOptions):
        return _transport_identity(value)
    if isinstance(value, dict):
        return {str(key): _stable_identity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_stable_identity(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"service_identity_value_not_serializable:{type(value).__name__}")


@dataclass(frozen=True)
class RecipeContext:
    op: object
    work: Mapping[str, float]
    resource_names: Mapping[str, str]
    rates: Mapping[str, float]
    core_count: int
    minimum_ms: float = 0.0
    minimum_provenance: Mapping = field(default_factory=dict)
    setup_ms: float = 0.0
    pool_names: Mapping[str, str] = field(default_factory=dict)
    resource_units: Mapping[str, int] = field(default_factory=dict)
    compute_peaks: Mapping[str, float] = field(default_factory=dict)
    pool_units: Mapping[str, int] = field(default_factory=dict)
    service_modes: Mapping[str, tuple[str, float]] = field(default_factory=dict)
    native_work: Mapping[str, float] = field(default_factory=dict)
    service_identity: Mapping = field(default_factory=dict)
    record_service: Callable | None = None
    work_units: Mapping[str, str] = field(default_factory=lambda: dict(NATIVE_WORK_UNITS))
    service_boundary: str = "device_kernel"
    # Physical resource -> (payload granule, wire frame including protocol overhead).
    transaction_frames: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    collective_plan: Callable[[], object] | None = None
    pool_capacities: Mapping[str, float] = field(default_factory=dict)
    record_mapping: Callable | None = None
    invocation_timing: object = None
    store_queues: Mapping[str, Mapping] = field(default_factory=dict)
    # IDs reserved by an enclosing physical kernel invocation. Isolated phase
    # analysis leaves this empty and must retain its own buffer requirements.
    group_buffers: tuple[str, ...] = ()
    joint_native_service: Mapping | None = None
    # Selected observable regime, including continuity group, execution mode,
    # response basis and the distinct response/executed durations.
    service_regime: Mapping | None = None
    # Per-stage capacity caps and diagnostics for a measured MAC service.  The
    # caps constrain only the selected physical pipe; other stages keep their
    # ordinary work and overlap.
    mac_caps: Mapping[str, float] = field(default_factory=dict)
    mac_service: Mapping | None = None
    # Source-selected implementation for this matched invocation only.  It is
    # never installed into the case or reused by an off-domain invocation.
    program_recipe: object = None
    #: T04's typed TransportOptions retained alongside the serializable identity.
    transport_options: object = None
    #: Stable serializable projection of transport_options (separate from the
    #: service identity used for curve matching).
    transport_identity: Mapping | None = None
    #: Immutable typed physical declaration selected for this device cohort.
    physical_mechanisms: object = None
    #: Homogeneity-checked ComputeArchSpec and derived DatapathCapabilitySpec.
    compute_arch: object = None
    datapath: object = None
    #: Bound per-stream core-control owners (AIC/AIV slice and effective limits).
    stream_owners: object = None

    @property
    def effective_aic_cores(self) -> int:
        """Effective AIC core limit for this operator's stream (``控核``)."""
        if self.stream_owners is not None:
            return int(self.stream_owners.aic_limit)
        if self.datapath is not None:
            return int(self.datapath.aic_count)
        return int(self.core_count)

    @property
    def effective_aiv_cores(self) -> int:
        """Effective AIV core limit for this operator's stream (``控核``)."""
        if self.stream_owners is not None:
            return int(self.stream_owners.aiv_limit)
        if self.datapath is not None:
            return int(self.datapath.aiv_count)
        return int(self.resource_units.get("vector", self.core_count))

    def wave_count(self, tiles: int, *, kind: str = "aic") -> int:
        """Return ``ceil(tiles / active_cores)`` under this stream's core budget."""
        if tiles <= 0:
            return 0
        active = (
            self.effective_aic_cores if kind == "aic" else self.effective_aiv_cores
        )
        if active <= 0:
            raise ModelingError(f"stream_zero_{kind}_budget:wave_count")
        return (int(tiles) + active - 1) // active

    def __post_init__(self):
        # T04 historically folded the typed option into service_identity.  Keep
        # curve identity stable and deliver the typed option separately.
        identity = dict(self.service_identity)
        transport = identity.pop("transport_options", None)
        if self.transport_options is None and transport is not None:
            object.__setattr__(self, "transport_options", transport)
        object.__setattr__(self, "service_identity", identity)
        object.__setattr__(self, "_owned_cache", {})

    def for_phase(self, work, *, kernel, dtype=None, layout=None, axes=None):
        """Bind explicitly identified phase work without repeating kernel setup."""
        op = work.apply(self.op)
        identity = {**self.service_identity, "kernel": kernel}
        if dtype is not None:
            op = replace(op, dtype=dtype)
            identity["dtype"] = dtype
        if layout is not None:
            op = replace(op, layout=layout)
            identity["layout"] = layout
        if axes is not None:
            identity["axes"] = axes
        transport = self.transport_options
        if transport is None:
            transport = identity.get("transport_options")
        identity.pop("transport_options", None)
        transport_identity = (_transport_identity(transport)
                              if transport is not None else None)
        identity = _stable_identity(identity)
        identity = json.loads(json.dumps(identity, allow_nan=False))
        native = {**work.resources, "cube": sum(work.matrix.values())}
        return replace(
            self, op=op, work=self.bind(work),
            native_work={role: amount for role, amount in native.items() if amount},
            setup_ms=0, minimum_ms=0, service_identity=identity,
            transport_options=transport, transport_identity=transport_identity,
            service_boundary="resource_phase", invocation_timing=None,
        )

    def physical_role(self, role):
        entry = self.service_modes.get(role)
        return entry[0] if entry is not None else role

    def service_rate(self, role: str) -> float | None:
        """Return the declared physical service rate for a named service mode."""
        entry = self.service_modes.get(role)
        return None if entry is None else float(entry[1])

    def placement(self, role, units):
        return {self.resource_names[self.physical_role(role)]: tuple(units)}

    def allocate(self, name, pool, size, units):
        return Allocation(name, self.pool_names[pool], size, tuple(units))

    def owned(self, owner, owners, roles):
        # The cache stores an immutable partition; every caller receives its
        # own mapping because builders narrow ``placement[resource]`` per
        # sub-block and must never rewrite another stage's ownership.
        cache = self._owned_cache
        roles_key = roles if roles.__class__ is tuple else tuple(roles)
        key = (owner, owners, roles_key)
        hit = cache.get(key)
        if hit is None:
            phys_roles = {self.physical_role(role) for role in roles_key}
            res_names = self.resource_names
            res_units = self.resource_units
            core_cnt = self.core_count
            hit = tuple((res_names[role], unit_group(
                owner, owners, res_units.get(role, core_cnt)))
                for role in phys_roles if role in res_names)
            cache[key] = hit
        return dict(hit)

    def read_pipe(self, *, aiv: bool | None = None) -> str:
        """Declared operand-read pipe for this op's consumer owner.

        A split MTE2 topology exposes one AIC-owned and one AIV-owned read path.
        The consumer decides which pipe services an operand fetch: an op with
        Cube work reads through AIC, a vector-only op through AIV.  An explicit
        ``read_pipe`` metadata declaration wins so a mixed owner can name the
        exact physical consumer instead of inferring it from totals.
        """
        from .ownership import read_role
        names = self.resource_names
        split = {"mte2_aic", "mte2_aiv"}.intersection(names)
        if not split:
            return "mte2"
        declared = str(dict(self.op.metadata or {}).get("read_pipe", "") or "")
        if declared in {"aic", "aiv"}:
            return read_role(names, aiv=0 if declared == "aiv" else None)
        if aiv is None:
            matrix = tuple(getattr(self.op, "matrix_flops", ()) or ())
            flops = float(getattr(self.op, "flops", 0.0) or 0.0)
            vector = float(getattr(self.op, "vector_ops", 0.0) or 0.0)
            aiv = not matrix and flops <= 0.0 and vector > 0.0
        return read_role(names, aiv=0 if aiv else None)

    def bind(self, work) -> dict[str, float]:
        """Resolve typed stage work onto shared physical resources, in ms units."""
        demand = {}
        for dtype, flops in work.matrix.items():
            if not flops:
                continue
            peak = self.compute_peaks.get(dtype)
            if peak is None:
                raise ModelingError(f"compute_precision_rate_required:{dtype}")
            name = self.resource_names["cube"]
            demand[name] = demand.get(name, 0) + 1000 * flops / peak
        matrix_consumer = any(float(value) > 0.0
                              for value in work.matrix.values())
        for role, amount in work.resources.items():
            if not amount:
                continue
            mode_entry = self.service_modes.get(role)
            if mode_entry is not None:
                resource, rate = mode_entry
                name = self.resource_names[resource]
                # Modes retain physical byte counts. Only service is expressed
                # in the base pipe's units so its different instructions contend.
                amount *= self.rates[name] * 1000 / rate
                role = resource
            if role == "mte2" and role not in self.resource_names:
                # The hardware declares separate AIC/AIV read pipes. Route the
                # typed fetch to its consumer instead of dropping the service.
                split = {"mte2_aic", "mte2_aiv"}.intersection(self.resource_names)
                if not split:
                    # A non-Ascend backend has one global memory path and no
                    # separate operand-fetch pipe; the HBM component already
                    # carries those bytes.  Never invent a second charge.
                    continue
                role = self.read_pipe(aiv=False if matrix_consumer else None)
            if role == "mte3" and role not in self.resource_names:
                continue
            if role not in self.resource_names:
                raise ModelingError(
                    f"physical_resource_rate_missing:{role}:"
                    f"{getattr(self.op, 'family', '')}:"
                    f"{getattr(self.op, 'recipe', '')}")
            name = self.resource_names[role]
            demand[name] = demand.get(name, 0) + amount
        return demand


class Recipe(Protocol):
    def lower(self, context: RecipeContext) -> Program: ...


class ResourceEnvelope:
    """Removed production fallback; retained only to reject legacy bindings."""

    def lower(self, context: RecipeContext) -> KernelProgram:
        raise ModelingError("resource_envelope_banned")


@dataclass(frozen=True)
class TransferRecipe:
    """Per-path transactions with slots held through post-service response.

    ``response_ms`` is a fixed return delay after transmission. It is not the
    inclusive response span used by the address-dependent scalar store model.
    """

    transaction_bytes: int | None = None
    outstanding: int | None = None
    response_ms: float = 0.0

    def lower(self, context: RecipeContext) -> KernelProgram:
        from .transfer import FiniteTransfer
        paths = []
        for resource, amount in context.work.items():
            frame = context.transaction_frames.get(resource)
            if frame is not None:
                payload, wire = frame
                if self.transaction_bytes is not None and self.transaction_bytes != payload:
                    raise ModelingError(f"transfer_granule_conflicts_with_link:{resource}")
                granule = wire
            else:
                granule = self.transaction_bytes if self.transaction_bytes is not None else 1
            paths.append(FiniteTransfer({resource: amount}, granule,
                                        self.outstanding, self.response_ms))
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("transactions", after=("setup",), program=ParallelProgram(tuple(paths))),
        ), minimum_ms=context.minimum_ms)


#: The generic matrix program key; a caller override of it never hijacks a
#: family that has its own dedicated physical binding (for example grouped).
GENERIC_MATRIX_PROGRAM = "matmul"


class RecipeBook:
    """Explicit implementation bindings; adding recipes does not edit the loop."""

    def __init__(self, recipes: Mapping[str, Recipe] | None = None,
                 *, families: Mapping[str, Recipe] | None = None, default: Recipe | None = None,
                 programs: Mapping[str, Recipe] | None = None,
                 family_registry: Any = None):
        from .matrix_mapping import MappedMatmul
        from .matrix_steady import SteadyMatmul
        from .operators.registry import FamilyDispatcher, builtin_family_registry
        from .state_program import RecurrentVector
        generic_grouped = (programs or {}).get("matmul", MappedMatmul())
        self._generic_grouped = generic_grouped
        balanced_matrix = (families or {}).get("grouped_gemm", generic_grouped)
        from .generic_latent_attention import build as _generic_latent_build
        from .mechanism import PlannedRecipe as _PlannedRecipe
        from .batch_prefix_attention import RECIPE as _batch_prefix_attn
        from .batch_prefix_indexer import RECIPE as _batch_prefix_idx
        from .physical_recipe_registry import (physical_family_bindings,
                                               physical_recipe_bindings)
        self._family_registry = (
            builtin_family_registry() if family_registry is None else family_registry
        )
        self._family_dispatcher = FamilyDispatcher(self._family_registry)
        self._governed_recipes: dict[str, Any] = {}
        self._explicit_recipes = frozenset((recipes or {}).keys()) | {
            "batch_prefix_attention", "batch_prefix_indexer",
        }
        # ``_explicit_recipes`` also receives internally registered source
        # recipes.  Only this set is the caller's own override.
        self._caller_recipes = frozenset((recipes or {}).keys())
        self._caller_programs = frozenset((programs or {}).keys())
        self._explicit_families = frozenset((families or {}).keys())
        self._family_kinds = {}
        self.recipes = {
            **physical_recipe_bindings(),
            "balanced_grouped_gate_up": balanced_matrix,
            "balanced_grouped_down": balanced_matrix,
            "batch_prefix_attention": _batch_prefix_attn,
            "batch_prefix_indexer": _batch_prefix_idx,
            **(recipes or {}),
        }
        self.families = {
            **physical_family_bindings(),
            "recurrent_scan": RecurrentScan(), "recurrent_vector": RecurrentVector(),
            "engram_rh2d": HostStaging(),
            "grouped_gemm": (programs or {}).get("matmul", SteadyMatmul()),
            **(families or {}),
        }
        self.programs = {"matmul": SteadyMatmul(),
                         "dense_latent_attention": _PlannedRecipe(_generic_latent_build),
                         **(programs or {})}
        self.default = default
        self._select_id_cache: dict[int, tuple[object, Recipe]] = {}
        self._expand_id_cache: dict[int, tuple[object, tuple]] = {}
        self._prev_expand: tuple | None = None

    def _governed_recipe(self, spec: Any):
        cached = self._governed_recipes.get(spec.name)
        if cached is not None:
            return cached
        from mindie_sim.schema.datapath import (
            ComputeArchIdentity,
            DatapathCapabilitySpec,
            GENERIC_ARCH_ID,
        )
        from .mechanism import PlannedRecipe
        from .operators.contracts import OperatorPlanningContext
        from .results import OpOccurrence

        book = self
        dispatcher = self._family_dispatcher

        def _build(op, geometry):
            datapath = getattr(geometry, "datapath", None)
            if datapath is None:
                res_units = dict(getattr(geometry, "resource_units", {}) or {})
                datapath = DatapathCapabilitySpec(
                    compute_arch=ComputeArchIdentity(
                        arch_id=GENERIC_ARCH_ID,
                        declared=getattr(geometry, "compute_arch", None),
                    ),
                    aic_count=int(getattr(geometry, "compute_units", 1) or 1),
                    aiv_count=int(res_units.get("vector", 0)),
                    aic_aiv_pairs=(),
                    usable_paths=(),
                    matrix_output_paths=frozenset({"gm"}),
                    ub_peer_edges=(),
                    stars=None,
                )
            meta = dict(op.metadata) if op.metadata else {}
            requested_abi = meta.get("requested_abi")
            ctx = OperatorPlanningContext(
                occurrence=OpOccurrence(0, int(getattr(op, "stage", 0)), None, str(op.op_id)),
                geometry=geometry,
                datapath=datapath,
                recipe_book=book,
                requested_abi=str(requested_abi) if requested_abi else None,
            )
            resolved = dispatcher.resolve_plan(op, ctx)
            return resolved.recipe

        planned = PlannedRecipe(
            _build, traffic_rewrite_capable=bool(spec.traffic_rewrite_capable)
        )
        self._governed_recipes[spec.name] = planned
        return planned

    def select(self, op: object) -> Recipe:
        hit = self._select_id_cache.get(id(op))
        if hit is not None and hit[0] is op:
            return hit[1]
        from .mechanism import PlannedRecipe
        if op.plan_required:
            recipe = self.recipes.get(op.recipe)
            if not isinstance(recipe, PlannedRecipe):
                raise ModelingError(f"source_plan_binding_required:{op.recipe}")
            if op.source_boundary == "call" and recipe.kernels is None:
                raise ModelingError(f"source_kernel_expansion_required:{op.recipe}")
            if len(self._select_id_cache) >= 4096:
                self._select_id_cache.clear()
            self._select_id_cache[id(op)] = (op, recipe)
            return recipe
        raw_meta = op.metadata
        meta = dict(raw_meta) if raw_meta else {}
        raw_view = meta.get("attention_tensor_view")
        view = dict(raw_view) if raw_view else {}
        recipe_name = str(getattr(op, "recipe", ""))
        family_name = str(getattr(op, "family", ""))
        explicit = (recipe_name in self._explicit_recipes
                    or family_name in self._explicit_families)
        kind = meta.get("program_kind")
        if (not explicit
                and kind == "dense_latent_attention_generic"
                and view.get("kind") == "dense_paged_latent"
                and "latent_attention_schedule" in meta):
            recipe = self.programs.get("dense_latent_attention")
            if isinstance(recipe, PlannedRecipe):
                if len(self._select_id_cache) >= 4096:
                    self._select_id_cache.clear()
                self._select_id_cache[id(op)] = (op, recipe)
                return recipe
        # Priority: a caller-supplied recipe override, then an explicitly
        # selected grouped_gemm physical family for the generic balanced
        # names, then the ordinary recipe/family/default lookup.  Specific
        # alternate recipes (K3, negative-weight, explicit grouped variants)
        # keep their own binding and are never replaced by the NXNK family.
        generic_grouped = {"balanced_grouped_gate_up",
                           "balanced_grouped_down"}
        if recipe_name in self._caller_recipes:
            recipe = self.recipes[recipe_name]
        elif (kind in self._caller_programs
              and (kind == family_name or kind != GENERIC_MATRIX_PROGRAM)):
            # A caller-declared program is authoritative over an automatic
            # physical family default of the same name.  Physical bindings are
            # a fallback, never an override of the caller's own declaration.
            recipe = self.programs[kind]
        elif (family_name in self._explicit_families
              and family_name != "grouped_gemm"):
            # An explicit caller family is authoritative over any automatic
            # source leaf, including the shared W4A8 default.
            recipe = self.families[op.family]
        elif family_name in {"gate_up_projection", "down_projection"}:
            from .shared_w4a8 import (config_from_op as _shared_config,
                                      matches_source_interface as _shared_match,
                                      recipe as _shared_recipe)
            if _shared_match(op):
                # A valid ordinary shared W4A8 interface with a bad local
                # shape/scale must surface its error, not fall back.
                _shared_config(op)
                recipe = _shared_recipe()
            else:
                recipe = self.recipes.get(recipe_name, self.families.get(
                    op.family, self.programs.get(kind)))
        elif (family_name == "grouped_gemm"
              and "grouped_gemm" in self._explicit_families
              and recipe_name in generic_grouped):
            recipe = self.families["grouped_gemm"]
            # The shared MXA8W4 NZNK family is source-specific.  Do not apply
            # it to unsupported W8/other layouts: the existing typed
            # config_from_op is the gate; on rejection use the ordinary
            # recipe default (never a blanket all-grouped W4 binding).
            if self._family_kinds.get("grouped_gemm") == "ascend_mxa8w4_nznk":
                from .mxfp4_grouped import (config_from_op as _mxfp4_config,
                                            matches_source_interface)
                if not matches_source_interface(op):
                    # Unsupported W8/other layout/weight group: use the
                    # ordinary analytical grouped recipe that existed before
                    # the source family.
                    recipe = self._generic_grouped
                else:
                    # A valid source interface with malformed/inconsistent
                    # rows must surface its real error, not silently fall back.
                    _mxfp4_config(op)
        else:
            governed = self._family_registry.match(op)
            if governed is not None and (
                recipe_name in governed.owned_recipe_ids
                or recipe_name not in self.recipes
            ):
                recipe = self._governed_recipe(governed)
            else:
                recipe = self.recipes.get(recipe_name, self.families.get(
                    op.family, self.programs.get(kind)))
        if recipe is None:
            from .physical_recipe_registry import PROJECTION_FAMILIES
            if family_name in PROJECTION_FAMILIES:
                recipe = self.programs.get("matmul")
            elif self.default is not None:
                recipe = self.default
            else:
                from .physical_recipe_registry import resolve_physical_recipe
                recipe = resolve_physical_recipe(op)
        if "matrix_group_rows" in meta:
            from .grouped_rows import adapt
            recipe = adapt(recipe, op)
        else:
            from .expected_grouped import adapt
            recipe = adapt(recipe, op)
        if len(self._select_id_cache) >= 4096:
            self._select_id_cache.clear()
        self._select_id_cache[id(op)] = (op, recipe)
        return recipe

    def prepare(self, op, geometry):
        from .mechanism import PlannedRecipe
        recipe = self.select(op)
        if op.source_boundary == "call":
            raise ModelingError(f"source_call_requires_kernel_expansion:{op.recipe}")
        return recipe.prepare(op, geometry) if isinstance(recipe, PlannedRecipe) else (op, recipe)

    def expand(self, ops):
        from .boundaries import expand_call
        from .mechanism import PlannedRecipe
        cache = self._expand_id_cache
        is_tup = ops.__class__ is tuple
        if is_tup:
            prev = self._prev_expand
            if prev is not None:
                p_ops, p_one_to_one, p_res = prev
                if ops is p_ops:
                    return p_res
                if p_one_to_one and len(ops) == len(p_ops):
                    result = None
                    still_1to1 = True
                    for idx, (op, pop) in enumerate(zip(ops, p_ops)):
                        if op is pop:
                            if result is not None:
                                result.append(p_res[idx])
                            continue
                        if result is None:
                            result = list(p_res[:idx])
                        hit = cache.get(id(op))
                        if hit is not None and hit[0] is op:
                            items = hit[1]
                        else:
                            recipe = self.select(op)
                            items = (expand_call(op, recipe.kernels)
                                     if (isinstance(recipe, PlannedRecipe) and recipe.kernels
                                         and op.source_boundary != "kernel") else (op,))
                            if len(cache) >= 16384:
                                cache.clear()
                            cache[id(op)] = (op, items)
                        if len(items) != 1:
                            still_1to1 = False
                        result.extend(items)
                    if result is None:
                        self._prev_expand = (ops, True, p_res)
                        return p_res
                    out = tuple(result)
                    self._prev_expand = (ops, still_1to1, out)
                    return out
        result = []
        one_to_one = True
        for op in ops:
            hit = cache.get(id(op))
            if hit is not None and hit[0] is op:
                items = hit[1]
                if len(items) != 1:
                    one_to_one = False
                result.extend(items)
                continue
            recipe = self.select(op)
            items = (expand_call(op, recipe.kernels)
                     if (isinstance(recipe, PlannedRecipe) and recipe.kernels
                         and op.source_boundary != "kernel") else (op,))
            if len(items) != 1:
                one_to_one = False
            if len(cache) >= 16384:
                cache.clear()
            cache[id(op)] = (op, items)
            result.extend(items)
        out = tuple(result)
        if is_tup:
            self._prev_expand = (ops, one_to_one, out)
        return out

    @classmethod
    def from_mapping(cls, raw, *, family_registry: Any = None):
        from .implementations import recipes as implementation_recipes
        from .mc2_graph import recipes as native_moe_recipes
        from .hccl_graph import recipes as hccl_recipes
        from .sparse_kv_offload import recipes as sparse_kv_offload_recipes
        from .latent_dcp import recipes as latent_dcp_recipes
        from .matrix import TiledMatmul, HierarchicalMatmul
        from .matrix_mapping import MappedMatmul
        from .matrix_phase import PhaseMatmul
        from .matrix_steady import SteadyMatmul
        from .indexed_attention import recipe as indexed_attention
        from .indexed_quant_attention import recipe as indexed_quant_attention
        from .attention_decode import StagedGQADecode
        from .attention_prefill import StagedGQAPrefill
        from .pointwise import TiledPointwise
        from .state_program import RecurrentVector
        from .services import ResourceService
        from .tensor_copy import recipe as aiv_tensor_copy
        from .mxfp4_grouped import recipe as mxa8w4_nznk
        from .bf16_grouped import recipe as bf16_grouped
        from .mxfp8_grouped import recipe as mxfp8_grouped
        from .family_kda_recurrent import recipes as kda_recurrent_recipes
        from .family_attnres import recipes as attn_res_recipes
        from .collectives import PairwiseCollective, RingCollective
        from .direct_peer import DirectPeerCollective
        from .mechanism import OperatorPlan, PlannedRecipe

        class _StagedGQADecodePlan(OperatorPlan):
            """OperatorPlan adapter whose public program name stays recipe-like."""

        _StagedGQADecodePlan.__name__ = "StagedGQADecode"

        class _StagedGQAPrefillPlan(OperatorPlan):
            """Prefill wrapper preserving the public recipe program name."""

        _StagedGQAPrefillPlan.__name__ = "StagedGQAPrefill"

        def decode_contract(spec):
            cache, slots = spec.kv_cache, spec.slots
            outputs = tuple(spec.outputs)
            rows = max((item.rows for item in outputs), default=0)
            horizon = max((item.horizon for item in outputs), default=0)
            full = min(spec.key_tile, max(1, horizon))
            max_tile = min(spec.key_tile, max(1, horizon))
            q_bytes = rows * spec.head_dim * spec.query_bytes
            online = rows * (spec.head_dim * 4 + 8)
            scores = rows * max_tile * 4
            probs = scores
            p_l0a = rows * full * spec.query_bytes
            pv_acc = rows * spec.head_dim * 4
            key = full * cache.key_payload_row
            value = full * cache.value_payload_row
            key_operand = full * cache.key_operand_row if cache.needs_conversion else key
            value_operand = full * cache.value_operand_row if cache.needs_conversion else value
            k_scale = full * cache.key_scale_row if cache.scale_scope == "row" else 0
            v_scale = full * cache.value_scale_row if cache.scale_scope == "row" else 0
            pages = cache.pages(0, horizon) if cache.paged else 0
            page_scale = pages * cache.page_scale_bytes if cache.scale_scope == "page" else 0
            k_conv = key_operand if cache.needs_conversion else 0
            v_conv = value_operand if cache.needs_conversion else 0
            cap = {
                "l1": q_bytes + slots * (key + value),
                "l0a": q_bytes + slots * p_l0a,
                "l0b": slots * (key_operand + value_operand),
                "l0c": slots * (scores + pv_acc),
                "ub": online + page_scale + slots * (
                    probs + k_conv + v_conv + k_scale + v_scale),
            }
            cap["total"] = sum(cap.values())
            buffers = [
                {"name": "q_l1", "pool": "l1", "bytes": q_bytes,
                 "slot": "persistent", "acquire": "q_load", "free": "final"},
                {"name": "q_l0a", "pool": "l0a", "bytes": q_bytes,
                 "slot": "persistent", "acquire": "q_load", "free": "final"},
                {"name": "online_state", "pool": "ub", "bytes": online,
                 "slot": "persistent", "acquire": "q_load", "free": "final"},
                {"name": "scores", "pool": "l0c", "bytes": scores,
                 "slot": "0..slots-1", "acquire": "qk", "free": "softmax"},
                {"name": "probs", "pool": "ub", "bytes": probs,
                 "slot": "0..slots-1", "acquire": "softmax", "free": "online_update"},
                {"name": "p_l0a", "pool": "l0a", "bytes": p_l0a,
                 "slot": "0..slots-1", "acquire": "probability_operand", "free": "pv"},
                {"name": "pv_acc", "pool": "l0c", "bytes": pv_acc,
                 "slot": "0..slots-1", "acquire": "v_load", "free": "pv"},
                {"name": "k_l1", "pool": "l1", "bytes": key,
                 "slot": "0..slots-1", "acquire": "k_load",
                 "free": "dequant_k" if cache.needs_conversion else "qk"},
                {"name": "v_l1", "pool": "l1", "bytes": value,
                 "slot": "0..slots-1", "acquire": "v_load",
                 "free": "dequant_v" if cache.needs_conversion else "pv"},
                {"name": "k_l0b", "pool": "l0b", "bytes": key_operand,
                 "slot": "0..slots-1", "acquire": "operand_k",
                 "free": "qk"},
                {"name": "v_l0b", "pool": "l0b", "bytes": value_operand,
                 "slot": "0..slots-1", "acquire": "operand_v",
                 "free": "pv"},
            ]
            if cache.needs_conversion:
                buffers.extend((
                    {"name": "k_conv", "pool": "ub", "bytes": k_conv,
                     "slot": "0..slots-1", "acquire": "dequant_k", "free": "operand_k"},
                    {"name": "v_conv", "pool": "ub", "bytes": v_conv,
                     "slot": "0..slots-1", "acquire": "dequant_v", "free": "operand_v"},
                ))
            if k_scale:
                buffers.append({"name": "k_scale", "pool": "ub", "bytes": k_scale,
                                "slot": "0..slots-1", "acquire": "k_scale_load",
                                "free": "dequant_k"})
            if v_scale:
                buffers.append({"name": "v_scale", "pool": "ub", "bytes": v_scale,
                                "slot": "0..slots-1", "acquire": "v_scale_load",
                                "free": "dequant_v"})
            if page_scale:
                buffers.append({"name": "page_scale", "pool": "ub", "bytes": page_scale,
                                "slot": "persistent", "acquire": "page_access",
                                "free": "page_scale_release"})
            return {
                "kv_cache_access": cache.report(),
                "conversion": {
                    "needs_conversion": cache.needs_conversion,
                    "query_dtype": cache.query_dtype,
                    "key_format": cache.report()["key_format"],
                    "value_format": cache.report()["value_format"],
                    "dequant_ops_per_element": cache.dequant_ops_per_element,
                    "elements_per_valid_row": (
                        (cache.head_dim - cache.key_suffix)
                        + (cache.head_dim - cache.value_suffix)),
                    "operand_copy_pipe": "mte1" if cache.needs_conversion else "",
                    "scope": "on_chip_operand_materialization_no_hbm_writeback",
                },
                "workspace": {
                    "owner_slots": slots,
                    "rows_per_owner": rows,
                    "horizon": horizon,
                    "full_tile_rows": full,
                    "capacity_bytes": cap,
                    "basis": "max declared output rows/tile per owner; slots reused",
                },
                "buffer_lifecycle": tuple(buffers),
            }

        def staged_gqa_decode(**options):
            source = StagedGQADecode(**options)

            def build(op, geometry):
                plan = source.build(op, geometry)
                metadata = dict(plan.metadata)
                metadata["decode_physical_contract"] = decode_contract(
                    source.spec(op))
                return _StagedGQADecodePlan(
                    plan.work, plan.program, metadata, plan.native)

            return PlannedRecipe(build)

        def staged_gqa_prefill(**options):
            source = StagedGQAPrefill(**options)

            def build(op, geometry):
                plan = source.build(op, geometry)
                return _StagedGQAPrefillPlan(
                    plan.work, plan.program, plan.metadata, plan.native)

            return PlannedRecipe(build)

        from .batch_prefix_attention import BatchPrefixAttention
        from .batch_prefix_indexer import BatchPrefixIndexer
        factories = {"tiled_matmul": TiledMatmul,
                     "hierarchical_matmul": HierarchicalMatmul, "mapped_matmul": MappedMatmul,
                     "phase_matmul": PhaseMatmul,
                     "steady_matmul": SteadyMatmul,
                     "indexed_attention_pipeline": indexed_attention,
                     "indexed_attention_packed_int8": indexed_quant_attention,
                     "staged_gqa_decode": staged_gqa_decode,
                     "staged_gqa_prefill": staged_gqa_prefill,
                     "batch_prefix_attention": BatchPrefixAttention,
                     "batch_prefix_indexer": BatchPrefixIndexer,
                     "transfer": TransferRecipe, "recurrent_scan": RecurrentScan,
                     "recurrent_vector": RecurrentVector, "tiled_pointwise": TiledPointwise,
                     "aiv_tensor_copy": aiv_tensor_copy,
                     "ascend_mxa8w4_nznk": mxa8w4_nznk,
                     "ascend_bf16_grouped": bf16_grouped,
                     "ascend_mxfp8_grouped_nz": mxfp8_grouped,
                     "resource_service": ResourceService, "ring_collective": RingCollective,
                     "pairwise_collective": PairwiseCollective,
                     "direct_peer_collective": DirectPeerCollective}
        def build(values):
            return {name: factories[value["kind"]](**{k: v for k, v in value.items() if k != "kind"})
                    for name, value in values.items()}
        explicit = {**native_moe_recipes(raw), **hccl_recipes(raw),
                    **sparse_kv_offload_recipes(),
                    **latent_dcp_recipes(),
                    **implementation_recipes(raw.get("lowerings", {})),
                    **kda_recurrent_recipes(),
                    **attn_res_recipes(),
                    **build(raw.get("recipes", {}))}
        book = cls(explicit, families=build(raw.get("families", {})),
                   programs=build(raw.get("programs", {})),
                   family_registry=family_registry)
        book._caller_recipes = frozenset((raw.get("recipes") or {}).keys())
        book._family_kinds = {
            name: value.get("kind") for name, value in
            (raw.get("families") or {}).items()
            if isinstance(value, Mapping)}
        return book


class HostStaging:
    """CPU lookup, host-vector decode and H2D transfer over qualified work keys.

    ``Execution`` binds native roles before constructing ``context.work``, so
    the binding is read through ``context.resource_names``; the remaining
    transfer work is preserved for other Engram callers.
    """

    def lower(self, context):
        names, work = context.resource_names, dict(context.work)
        roles = ("host_memory", "host_vector", "h2d")
        if any(role not in names for role in roles):
            raise ModelingError(
                "host_staging_requires_host_memory_host_vector_h2d_rates")
        if any(names[role] not in work for role in roles):
            raise ModelingError("host_staging_requires_bound_host_and_h2d_work")
        memory = {names["host_memory"]: work.pop(names["host_memory"])}
        compute = {names["host_vector"]: work.pop(names["host_vector"])}
        return KernelProgram((
            Stage("lookup", memory), Stage("decode", compute, after=("lookup",)),
            Stage("transfer", work, after=("decode",)),
        ), context.minimum_ms)


class RecurrentScan:
    """C1 -> vector delta -> C2 -> state update for each dependent chunk."""

    def lower(self, context):
        from .recurrent_template import lower_recurrent_scan

        return lower_recurrent_scan(context)


# ---------------------------------------------------------------------------
# Symmetric owner cohorts
#
# A ``ParallelProgram`` that splits one operator over N symmetric owners submits
# N copies of the same stage graph.  Within one launch from the same ready
# event, owners with identical stage geometry have identical start readiness,
# coupled rates and completion time.  Their tasks may therefore be combined
# into one cohort task whose ``work`` and ``caps`` are the N-fold sums and
# whose ``placement`` is the union of the owners' units.  Because the Engine
# divides a placement over its units, the cohort's per-unit demand is the same
# as the N single-owner demands and ``dt = (N*w) / min(N*cap, supply)`` is the
# original ``w / min(cap, supply/N)``.
#
# The helpers below are deliberately conservative: a merge is only returned for
# structurally identical programs whose per-resource placements are either
# identical or pairwise disjoint, and whose allocation names/sizes match with
# non-overlapping units.  Anything a merge cannot prove returns ``None`` so the
# caller keeps the exact per-owner programs.  No work, cap or byte is dropped.
# ---------------------------------------------------------------------------


#: Bounded tile window a symmetric cohort may materialize to prove identity.
#: Owners whose pipeline exceeds it keep their exact per-owner program.
PIPELINE_COHORT_WINDOW = 512
#: Bounded tile window an interior fold may materialize.  A repeated key-tile
#: stream is short (tens of tiles) so the real finite plan is always available.
PIPELINE_FOLD_WINDOW = 512


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
        _allocation_signature(stage.acquire),
        _allocation_signature(stage.free),
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
    """One interior super-tile: identical stages whose work is the run's sum.

    Every folded stage keeps its own resources, supply cap and placement, so
    ``repeat`` consecutive identical physical tiles whose duration is linear in
    their issued work compose exactly into one task of ``repeat``-fold work.
    Buffers, ordering and per-tile overlap outside the run are unchanged.
    """
    first = stages[0]
    rows = []
    for stage in stages:
        work = {key: value * repeat for key, value in stage.work.items()}
        rows.append(replace(stage, work=work))
    return tuple(rows)


def interior_fold_plan(program: "PipelineProgram"):
    """Return ``[(representative index, repeat, stages)]`` or ``None``.

    Construction for the opt-in steady interior tile compression.  A tile
    window that repeats a small residue class (for example alternate slot
    buffers) re-declares the same physical stage set and the same issued work
    for every member of a class; the helper folds the longest identical prefix
    of each class into one task per stage whose work is the arithmetic sum and
    leaves the prologue and the partial tail explicit.

    Exactness boundary: the fold conserves every resource total, every declared
    buffer extent and every serviced byte, and it is exact only when the folded
    members are strictly serial.  A slot window that keeps two members of the
    same class in flight overlaps them, so collapsing the window changes the
    realized schedule - measured on P3 ``body_forward_ms`` 4.536926083908739 ->
    4.587842220272391 (1.12e-2).  No producer currently satisfies the strictly
    serial precondition, so ``PipelineProgram.interior_fold`` stays ``False``.
    """
    count, slots = program.count, program.slots
    if program.homogeneous or count < 4 or slots < 1:
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
        while (matched + candidate < count
               and keys[matched] == keys[matched + candidate]):
            matched += 1
        blocks = (matched + candidate) // candidate
        if blocks < 2:
            continue
        if period is None or blocks * candidate > repeat * period:
            period, repeat = candidate, blocks
    if period is None:
        return None
    folded = []
    for residue in range(period):
        index = residue
        folded.append((index, repeat, _fold_scaled(tiles[index], repeat)))
    for index in range(repeat * period, count):
        folded.append((index, 1, tiles[index]))
    if len(folded) < 3:
        return None
    return folded


def fold_interior_pipeline(program: "PipelineProgram"):
    """Return an exactly-folded equivalent program, or ``None``."""
    plan = interior_fold_plan(program)
    if plan is None:
        return None
    stages = tuple(entry[2] for entry in plan)

    def tile(index: int, stages=stages):
        return stages[index]

    return replace(program, count=len(stages), tile=tile)


def _units_mergeable(rows):
    """True when per-program unit tuples are identical or pairwise disjoint."""
    if len({tuple(units) for units in rows}) == 1:
        return True
    seen: set[int] = set()
    for units in rows:
        if any(unit in seen for unit in units):
            return False
        seen.update(units)
    return True


def _allocation_signature(items):
    return tuple(sorted((item.name, item.pool, item.size, len(item.units))
                        for item in items))


def _stage_signature(stage, *, widen_pipelines: bool = False):
    program = (None if stage.program is None
               else program_signature(stage.program,
                                      widen_pipelines=widen_pipelines))
    if stage.program is not None and program is None:
        return None
    return (
        stage.name, stage.after, stage.previous, stage.latency, stage.coupled,
        stage.priority, stage.cohort_size, stage.keep_empty,
        tuple(sorted(stage.work_units.items())),
        tuple(sorted(stage.work.items())),
        tuple(sorted(stage.caps.items())),
        tuple(sorted((key, len(units)) for key, units in stage.placement.items())),
        _allocation_signature(stage.acquire),
        _allocation_signature(stage.free),
        program,
        None if stage.fragment is None else stage.fragment.name,
    )


def _template_signature(program):
    template = program.template
    if (template.setup or template.drain or template.bindings
            or template.setup_ms or template.minimum_ms
            or template.indexed_panels is not None
            or template.motif_repeats != 1):
        return None
    phases = []
    for phase in template.phases:
        if phase.residue_stages or len(phase.stages) != 1:
            return None
        signature = _stage_signature(phase.stages[0])
        if signature is None:
            return None
        phases.append((phase.kind, phase.count, phase.key, signature))
    return ("template", template.slots, tuple(phases))


def _pipeline_signature(program, *, widen_pipelines: bool = False):
    """Structural identity of one bounded non-homogeneous tile window.

    An opaque ``tile(index)`` callback can still declare every finite tile it
    owns; materializing that bounded window is pure producer work and lets two
    symmetric owners prove they submit the same physical stage geometry.
    """
    if program.homogeneous or program.setup_ms or program.minimum_ms:
        return None
    if program.count < 0 or program.count > PIPELINE_COHORT_WINDOW:
        return None
    tiles = []
    for index in range(program.count):
        row = []
        for stage in tuple(program.tile(index)):
            signature = _stage_signature(stage, widen_pipelines=widen_pipelines)
            if signature is None:
                return None
            row.append(signature)
        tiles.append(tuple(row))
    return ("pipeline", program.count, program.slots, tuple(tiles))


def program_signature(program, *, widen_pipelines: bool = False):
    """Unit-agnostic structural identity for cohort coalescing, or ``None``."""
    if isinstance(program, KernelProgram):
        stages = []
        for stage in program.stages:
            signature = _stage_signature(stage, widen_pipelines=widen_pipelines)
            if signature is None:
                return None
            stages.append(signature)
        return ("kernel", program.minimum_ms, tuple(stages))
    if isinstance(program, PipelineTemplateProgram):
        return _template_signature(program)
    if widen_pipelines and isinstance(program, PipelineProgram):
        return _pipeline_signature(program, widen_pipelines=True)
    # A nested opaque tile pipeline is deliberately not a cohort candidate by
    # default: ``tests/test_gdn_recurrent_compaction`` requires every
    # demand-carrying task to keep its own single physical unit when inspected
    # with tracing enabled, which a summed-work cohort placement would widen.
    return None


def _merge_allocations(groups):
    """Combine equal-named allocations; ``None`` when identities do not match."""
    order, merged = [], {}
    for items in groups:
        for item in items:
            row = merged.get(item.name)
            if row is None:
                merged[item.name] = [item.pool, item.size, list(item.units)]
                order.append(item.name)
                continue
            if row[0] != item.pool or row[1] != item.size:
                return None
            if any(unit in row[2] for unit in item.units):
                return None
            row[2].extend(item.units)
    return tuple(Allocation(name, merged[name][0], merged[name][1],
                            tuple(merged[name][2])) for name in order)


def _merge_stages(stages, *, scale_coupled_cohort: bool = True,
                  widen_pipelines: bool = False):
    """Combine N structurally identical stages into one cohort stage.

    A wrapper stage that owns one nested program is merged by merging that
    program; the wrapper itself must still declare no work, time or buffer, so
    the cohort's nested tasks are the N owners' tasks summed exactly.
    """
    if len(stages) < 2:
        return stages[0] if stages else None
    first = stages[0]
    if any(stage.fragment is not None for stage in stages):
        return None
    if len({stage.latency for stage in stages}) != 1:
        return None
    if len({stage.name for stage in stages}) != 1:
        return None
    if len({stage.coupled for stage in stages}) != 1:
        return None
    if len({stage.priority for stage in stages}) != 1:
        return None
    if len({stage.keep_empty for stage in stages}) != 1:
        return None
    if len({stage.after for stage in stages}) != 1:
        return None
    if len({stage.previous for stage in stages}) != 1:
        return None
    if len({tuple(sorted(stage.work_units.items())) for stage in stages}) != 1:
        return None
    if any(stage.program is not None for stage in stages):
        if any(stage.program is None for stage in stages):
            return None
        if any(stage.work or stage.caps or stage.acquire or stage.free
               or stage.placement or stage.latency for stage in stages):
            return None
        merged = merge_programs(
            [stage.program for stage in stages],
            scale_coupled_cohort=scale_coupled_cohort,
            widen_pipelines=widen_pipelines)
        if merged is None:
            return None
        return Stage(first.name, after=first.after, previous=first.previous,
                     coupled=first.coupled, work_units=first.work_units,
                     priority=first.priority, program=merged,
                     keep_empty=first.keep_empty)
    work = {key: sum(stage.work[key] for stage in stages) for key in first.work}
    caps = {key: sum(stage.caps[key] for stage in stages) for key in first.caps}
    placement = {}
    for key in first.placement:
        rows = [stage.placement[key] for stage in stages]
        if not _units_mergeable(rows):
            return None
        if len({tuple(row) for row in rows}) == 1:
            # Symmetric stages on the same owners are one owner partition;
            # widening repeats would declare duplicate (invalid) units.
            placement[key] = tuple(rows[0])
            continue
        units: list[int] = []
        for row in rows:
            units.extend(row)
        placement[key] = tuple(units)
    if any(set(stage.placement) != set(first.placement) for stage in stages):
        return None
    acquire = _merge_allocations([stage.acquire for stage in stages])
    free = _merge_allocations([stage.free for stage in stages])
    if acquire is None or free is None:
        return None
    cohort_size = (sum(stage.cohort_size for stage in stages)
                   if (first.coupled and scale_coupled_cohort)
                   else first.cohort_size)
    return Stage(first.name, work, after=first.after, latency=first.latency,
                 caps=caps, acquire=acquire, free=free,
                 placement=placement, previous=first.previous,
                 coupled=first.coupled, work_units=first.work_units,
                 priority=first.priority, cohort_size=cohort_size,
                 keep_empty=first.keep_empty)


def _merge_pipelines(programs, *, scale_coupled_cohort: bool = True,
                     widen_pipelines: bool = False):
    """Merge symmetric owners of one bounded, opaque tile pipeline."""
    first = programs[0]
    if len({(program.count, program.slots, program.setup_ms,
             program.minimum_ms, program.homogeneous,
             program.producer_bindings is not None)
            for program in programs}) != 1:
        return None
    if first.homogeneous or first.setup_ms or first.minimum_ms:
        return None
    if first.count < 0 or first.count > PIPELINE_COHORT_WINDOW:
        return None
    tiles = []
    for index in range(first.count):
        rows = [tuple(program.tile(index)) for program in programs]
        if len({len(row) for row in rows}) != 1:
            return None
        stages = []
        for position in range(len(rows[0])):
            merged = _merge_stages(
                [row[position] for row in rows],
                scale_coupled_cohort=scale_coupled_cohort,
                widen_pipelines=widen_pipelines)
            if merged is None:
                return None
            stages.append(merged)
        tiles.append(tuple(stages))
    return PipelineProgram(
        first.count, first.slots, lambda index, tiles=tuple(tiles): tiles[index],
        first.minimum_ms, first.setup_ms,
        producer_bindings=first.producer_bindings)


def _merge_kernels(programs, *, scale_coupled_cohort: bool = True,
                   widen_pipelines: bool = False):
    first = programs[0]
    if len({program.minimum_ms for program in programs}) != 1:
        return None
    if len({len(program.stages) for program in programs}) != 1:
        return None
    stages = []
    for index in range(len(first.stages)):
        merged = _merge_stages(
            [program.stages[index] for program in programs],
            scale_coupled_cohort=scale_coupled_cohort,
            widen_pipelines=widen_pipelines)
        if merged is None:
            return None
        stages.append(merged)
    return KernelProgram(tuple(stages), first.minimum_ms)


def _merge_wrapper_stages(stages):
    """Merge single-static wrapper stages that own one nested program."""
    if any(stage.program is None for stage in stages):
        return _merge_stages(stages, scale_coupled_cohort=False)
    if any(stage.work or stage.caps or stage.acquire or stage.free
           or stage.placement or stage.latency for stage in stages):
        return None
    if len({(stage.name, stage.after, stage.previous, stage.coupled,
             stage.priority) for stage in stages}) != 1:
        return None
    merged = merge_programs(
        [stage.program for stage in stages], scale_coupled_cohort=False)
    if merged is None:
        return None
    first = stages[0]
    return Stage(first.name, after=first.after, previous=first.previous,
                 program=merged, coupled=first.coupled,
                 work_units=first.work_units, priority=first.priority)


def _merge_templates(programs):
    first = programs[0].template
    if len({program.template.slots for program in programs}) != 1:
        return None
    if len({len(program.template.phases) for program in programs}) != 1:
        return None
    phases = []
    for index, phase in enumerate(first.phases):
        rows = [program.template.phases[index] for program in programs]
        if len({(row.kind, row.count, row.key) for row in rows}) != 1:
            return None
        merged = _merge_wrapper_stages([row.stages[0] for row in rows])
        if merged is None:
            return None
        phases.append(PipelinePhase(phase.kind, (merged,), phase.count,
                                    key=phase.key))
    return PipelineTemplateProgram(PipelineTemplate(
        first.owner, first.slots, tuple(phases),
        setup_ms=first.setup_ms, minimum_ms=first.minimum_ms,
        tile_factory_calls=first.tile_factory_calls,
        phase_objects=len(phases), stage_objects=len(phases),
        static_phase_classes=len(phases),
        dynamic_instances=sum(phase.count for phase in phases)))


def coalesce_programs(programs, *, widen_pipelines: bool = False):
    """Combine symmetric owners into cohort programs; keep the exact fallback."""
    programs = tuple(programs)
    if len(programs) < 2:
        return programs
    groups: list[tuple[object, list]] = []
    for program in programs:
        signature = program_signature(program, widen_pipelines=widen_pipelines)
        if signature is None:
            groups.append((None, [program]))
            continue
        for entry in groups:
            if entry[0] == signature:
                entry[1].append(program)
                break
        else:
            groups.append((signature, [program]))
    result = []
    for signature, group in groups:
        if signature is not None and len(group) > 1:
            merged = merge_programs(group, widen_pipelines=widen_pipelines)
            if merged is not None:
                result.append(merged)
                continue
        result.extend(group)
    return tuple(result)


def merge_programs(programs, *, scale_coupled_cohort: bool = True,
                   widen_pipelines: bool = False):
    """Merge one cohort of structurally identical programs, or ``None``."""
    if not programs:
        return None
    first = programs[0]
    if isinstance(first, KernelProgram):
        return _merge_kernels(
            programs, scale_coupled_cohort=scale_coupled_cohort,
            widen_pipelines=widen_pipelines)
    if isinstance(first, PipelineTemplateProgram):
        return _merge_templates(programs)
    if widen_pipelines and isinstance(first, PipelineProgram):
        return _merge_pipelines(
            programs, scale_coupled_cohort=scale_coupled_cohort,
            widen_pipelines=True)
    return None


def _is_empty_queued_store(program) -> bool:
    return (type(program).__name__ == "QueuedStore"
            and int(getattr(program, "outstanding", 0)) >= 1
            and int(getattr(program, "granule_bytes", 0)) >= 1
            and int(getattr(program, "rows", 1)) * int(getattr(program, "cols", 1)) == 0)


def _inline_wrapper_stages(stages, *, protected_names=()):
    sink_alias: dict[str, str] = {}
    out: list[Stage] = []
    for stage in stages:
        if sink_alias:
            after = tuple(sink_alias.get(d, d) for d in stage.after)
            prev = tuple(sink_alias.get(d, d) for d in stage.previous)
            if after != stage.after or prev != stage.previous:
                stage = replace(stage, after=after, previous=prev)
        if (stage.name not in protected_names
                and isinstance(stage.program, KernelProgram)
                and stage.program.minimum_ms == 0.0
                and not stage.work and not stage.caps
                and not stage.acquire and not stage.free
                and stage.latency <= 0.0 and not stage.placement
                and not stage.previous and not stage.keep_empty
                and stage.fragment is None):
            inner = stage.program.stages
            if inner and not any(
                    s.acquire or s.free or s.previous or s.keep_empty
                    or s.fragment is not None for s in inner):
                consumed = {d for s in inner for d in s.after}
                sinks = [s.name for s in inner if s.name not in consumed]
                if len(sinks) == 1:
                    prefix = stage.name
                    for inner_stage in inner:
                        s_after = (tuple(f"{prefix}:{d}" for d in inner_stage.after)
                                   if inner_stage.after else stage.after)
                        out.append(replace(
                            inner_stage, name=f"{prefix}:{inner_stage.name}",
                            after=s_after))
                    sink_alias[prefix] = f"{prefix}:{sinks[0]}"
                    continue
        out.append(stage)
    return tuple(out)


@dataclass(frozen=True)
class _FloorParallelProgram:
    """Parallel cohort program with a single floor-released completion join."""

    programs: tuple[Program, ...]
    minimum_ms: float
    single_body: bool = False

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        if ready.time is None:
            from .invocation import InvocationProgram, InvocationTiming
            return InvocationProgram(
                self.programs[0] if self.single_body else ParallelProgram(self.programs),
                InvocationTiming(minimum_ms=self.minimum_ms),
            ).start(engine, name, ready, owner=owner)
        comp_cache = getattr(engine, "_compiled_programs", None)
        if comp_cache is not None:
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
                    g_pref = ":body" if self.single_body else f":group{index}"
                    compiled = _compile_leaf_kernel_ops(
                        engine, program, g_pref, next_slot)
                    if compiled is None:
                        ok = False
                        break
                    b_ops, b_sink, next_slot, _had, hrd = compiled
                    if hrd:
                        any_root_delay = True
                    all_ops.extend(b_ops)
                    branch_sinks.append(b_sink)
                plan = (
                    (tuple(all_ops), tuple(branch_sinks), any_root_delay)
                    if ok else False
                )
                if len(comp_cache) >= 4096:
                    comp_cache.clear()
                comp_cache[id(self)] = (self, plan)
            if plan is not False:
                plan_ops, sink_slots, any_root_delay = plan
                if not any_root_delay or (
                    (not engine._trace_maxlen or getattr(engine, "_compact_step", False))
                    and not engine._periodic_runs
                    and not engine.occurrence_metrics
                ):
                    ev = (_run_compiled_ops(engine, name, ready, owner, plan_ops)
                          if plan_ops else [ready])
                    n_sinks = len(sink_slots)
                    if n_sinks == 1:
                        after_events = ev[sink_slots[0]]
                    elif n_sinks == 2:
                        after_events = (ev[sink_slots[0]], ev[sink_slots[1]])
                    elif n_sinks:
                        after_events = tuple([ev[s] for s in sink_slots])
                    else:
                        after_events = ready
                    return engine._submit_prepared(
                        engine._empty_task, name + ":complete",
                        after=after_events, release=ready.time + self.minimum_ms,
                        owner=owner)
        if self.single_body:
            body_ev = self.programs[0].start(
                engine, f"{name}:body", ready, owner=owner)
            return engine.submit(
                name + ":complete", after=(body_ev,),
                release=ready.time + self.minimum_ms, owner=owner,
            )
        events = tuple(
            program.start(engine, f"{name}:group{index}", ready, owner=owner)
            for index, program in enumerate(self.programs)
        )
        return engine.submit(
            name + ":complete", after=events or (ready,),
            release=ready.time + self.minimum_ms, owner=owner,
        )


def _is_fair_compute_resource(name: str) -> bool:
    return (name in {"vector", "aic", "cube"}
            or name.startswith(("vector@", "aic@", "cube@"))
            or ":vector@" in name or ":aic@" in name or ":cube@" in name)


def _fold_kernel_stages(stages: tuple[Stage, ...]) -> tuple[Stage, ...]:
    """Fold zero-work acquire/free wrappers and strictly serial integer compute chains."""
    if len(stages) < 2:
        return stages
    if any(s.previous or s.keep_empty or s.fragment is not None for s in stages):
        return stages
    cur = list(stages)
    while len(cur) >= 2:
        consumers: dict[str, list[int]] = {s.name: [] for s in cur}
        by_name: dict[str, int] = {}
        valid = True
        for idx, s in enumerate(cur):
            if s.name in by_name:
                valid = False
                break
            by_name[s.name] = idx
            for dep in s.after:
                if dep in consumers:
                    consumers[dep].append(idx)
        if not valid:
            return tuple(cur)
        changed = False
        for idx, s in enumerate(cur):
            # 1. Zero-work acquire-only stage -> sole immediate successor without own acquire
            if (s.program is None and not any(s.work.values()) and not any(s.caps.values())
                    and s.latency <= 0.0 and not s.coupled and s.priority == 0
                    and s.acquire and not s.free and len(consumers[s.name]) == 1):
                succ_idx = consumers[s.name][0]
                succ = cur[succ_idx]
                if succ.program is None and succ.after == (s.name,) and not succ.acquire:
                    cur[succ_idx] = replace(succ, after=s.after, acquire=s.acquire)
                    del cur[idx]
                    changed = True
                    break
            # 2. Zero-work free-only sink stage -> sole immediate predecessor
            if (s.program is None and not any(s.work.values()) and not any(s.caps.values())
                    and s.latency <= 0.0 and not s.coupled and s.priority == 0
                    and s.free and not s.acquire and not consumers[s.name]
                    and len(s.after) == 1 and s.after[0] in by_name):
                pred_idx = by_name[s.after[0]]
                pred = cur[pred_idx]
                if (pred.program is None and consumers[pred.name] == [idx]
                        and not any(x.name in {y.name for y in pred.free} for x in s.free)):
                    cur[pred_idx] = replace(pred, free=pred.free + s.free)
                    del cur[idx]
                    changed = True
                    break
            # 3. Strictly serial single-resource fair integer compute chain
            if (s.program is None and not s.coupled and s.latency <= 0.0
                    and not s.acquire and not s.free and len(s.work) == 1
                    and len(consumers[s.name]) == 1):
                succ_idx = consumers[s.name][0]
                succ = cur[succ_idx]
                if (succ.program is None and not succ.coupled and succ.latency <= 0.0
                        and not succ.acquire and not succ.free
                        and succ.after == (s.name,)
                        and len(succ.work) == 1 and s.work.keys() == succ.work.keys()
                        and s.caps == succ.caps and s.placement == succ.placement
                        and s.priority == succ.priority and s.work_units == succ.work_units):
                    rname = next(iter(s.work))
                    w1 = s.work[rname]
                    w2 = succ.work[rname]
                    if (_is_fair_compute_resource(rname)
                            and w1 == int(w1) and w2 == int(w2)
                            and abs(w1) + abs(w2) < 4e15):
                        merged_name = s.name
                        cur[idx] = replace(s, work={rname: w1 + w2})
                        for c_idx in consumers[succ.name]:
                            c_stage = cur[c_idx]
                            cur[c_idx] = replace(
                                c_stage,
                                after=tuple(merged_name if d == succ.name else d
                                            for d in c_stage.after))
                        del cur[succ_idx]
                        changed = True
                        break
        if not changed:
            break
    return tuple(cur)


def is_pure_static_program(program) -> bool:
    """True for a callback-free finite DAG of plain kernel stages.

    A nested ``Stage.program`` is a per-tile callback, a pipeline tile factory
    is a callable, and an allocation-carrying stage needs a Python start
    boundary: none of them can be admitted without a Python hook, so none is
    "pure static" for admission purposes.
    """
    if isinstance(program, KernelProgram):
        return not any(stage.program is not None or stage.acquire or stage.free
                       for stage in program.stages)
    if isinstance(program, (ParallelProgram, _FloorParallelProgram)):
        return bool(program.programs) and all(
            is_pure_static_program(child) for child in program.programs)
    return False


def coalesce_execution_program(program, *, unroll_limit: int = PIPELINE_UNROLL_LIMIT):
    """Unwrap single-tile pipelines and coalesce symmetric cohorts for Execution.

    Used only when ``Engine.trace`` is disabled (or a bounded summary ring on a
    large graph) and legacy per-unit diagnostics are off, preserving exact
    resource service, allocations, and timestamps.
    """
    if type(program).__name__ == "_FloorJoinProgram":
        body = coalesce_execution_program(program.body, unroll_limit=unroll_limit)
        eff_min = (program.effective_minimum()
                   if hasattr(program, "effective_minimum") else program.minimum_ms)
        if eff_min <= 0.0:
            return body
        if isinstance(body, ParallelProgram):
            if len(body.programs) == 1:
                body = body.programs[0]
            elif len(body.programs) > 1:
                return _FloorParallelProgram(body.programs, eff_min)
        if isinstance(body, KernelProgram) and body.minimum_ms == 0.0:
            real_stages = [
                s for s in body.stages
                if not (s.program is None and _declares_no_work(s) and len(s.after) <= 1)
            ]
            if len(real_stages) == 1:
                s0 = real_stages[0]
                if (not s0.acquire and not s0.free and s0.program is None
                        and s0.latency == 0.0 and not s0.previous
                        and not s0.keep_empty and s0.fragment is None):
                    return KernelProgram((replace(s0, after=(), latency=eff_min),))
            elif len(real_stages) == 2 and len(body.stages) == 2:
                s0, s1 = real_stages
                if (_is_pure_root_delay(s0, {s0.name}, body.stages)
                        and s0.latency >= eff_min
                        and s1.after == (s0.name,)
                        and s1.program is None and s1.fragment is None):
                    return body
            return _FloorParallelProgram((body,), eff_min, single_body=True)
        if body is program.body:
            return program
        return replace(program, body=body)

    if type(program).__name__ == "NativeProgram":
        inner = coalesce_execution_program(program.program, unroll_limit=unroll_limit)
        if inner is program.program:
            return program
        return replace(program, program=inner)

    def unwrap_stage(stage: Stage) -> Stage:
        if stage.program is None:
            return stage
        if (_is_empty_queued_store(stage.program)
                and not stage.work and not stage.caps
                and not stage.acquire and not stage.free
                and stage.latency <= 0.0 and not stage.placement
                and stage.fragment is None):
            return replace(stage, program=None)
        nested = coalesce_execution_program(stage.program, unroll_limit=unroll_limit)
        return stage if nested is stage.program else replace(stage, program=nested)

    if isinstance(program, KernelProgram):
        unwrapped = tuple(unwrap_stage(stage) for stage in program.stages)
        inlined = _fold_kernel_stages(_inline_wrapper_stages(unwrapped))
        if program.minimum_ms == 0.0 and len(inlined) == 1:
            s0 = inlined[0]
            if (s0.program is not None and not s0.work and not s0.caps
                    and not s0.acquire and not s0.free and s0.latency <= 0.0
                    and not s0.placement and not s0.after and not s0.previous
                    and not s0.keep_empty and s0.fragment is None):
                return s0.program
        if inlined == program.stages:
            return program
        return KernelProgram(inlined, program.minimum_ms)

    if isinstance(program, PipelineProgram):
        if (program.homogeneous or program.setup_ms or program.minimum_ms
                or program.producer_bindings is not None):
            return program
        if program.count == 1:
            raw_stages = tuple(program.tile(0))
            seen: set[str] = set()
            flat: list[Stage] = []
            valid = True
            for stage in raw_stages:
                if stage.name in seen:
                    valid = False
                    break
                seen.add(stage.name)
                # A single-tile pipeline has no preceding tile, so a
                # ``previous`` reference cannot become an intra-tile edge.
                flat.append(unwrap_stage(replace(stage, previous=())))
            if valid:
                return KernelProgram(_fold_kernel_stages(_inline_wrapper_stages(flat)))
        if 1 < program.count <= PIPELINE_COHORT_WINDOW:
            raw_tiles = [tuple(program.tile(i)) for i in range(program.count)]
            # Unrolling is a task-shape optimization only; it must not turn a
            # slot-bounded pipeline into one flat submission of every tile.
            # Cap the materialized window so live tasks stay O(slots).
            if (program.slots == 1
                    and program.count * len(raw_tiles[0]) <= unroll_limit
                    and all(len({s.name for s in tile}) == len(tile)
                            for tile in raw_tiles)):
                unrolled: list[Stage] = []
                prev_sinks: tuple[str, ...] = ()
                for t_idx, tile in enumerate(raw_tiles):
                    t_stages = _fold_kernel_stages(_inline_wrapper_stages(
                        tuple(unwrap_stage(s) for s in tile)))
                    if not t_stages:
                        continue
                    local_acq = {a.name for s in t_stages for a in s.acquire}
                    local_free = {a.name for s in t_stages for a in s.free}
                    local_both = local_acq & local_free
                    pfx = f"t{t_idx}:"
                    consumed_t = {d for s in t_stages for d in s.after}
                    for s in t_stages:
                        new_after = (tuple(pfx + d for d in s.after)
                                     if s.after else prev_sinks)
                        new_acq = (tuple(
                            Allocation(pfx + a.name, a.pool, a.size, a.units)
                            if a.name in local_both else a
                            for a in s.acquire)
                            if local_both and s.acquire else s.acquire)
                        new_free = (tuple(
                            Allocation(pfx + a.name, a.pool, a.size, a.units)
                            if a.name in local_both else a
                            for a in s.free)
                            if local_both and s.free else s.free)
                        unrolled.append(replace(
                            s, name=pfx + s.name, after=new_after,
                            previous=(), acquire=new_acq, free=new_free))
                    prev_sinks = tuple(
                        pfx + s.name for s in t_stages
                        if s.name not in consumed_t
                    ) or prev_sinks
                return KernelProgram(_fold_kernel_stages(tuple(unrolled)))
            protected = {p for tile in raw_tiles for s in tile for p in s.previous}
            protected.update(s.name for tile in raw_tiles for s in tile)
            allow_inline = (
                {p for tile in raw_tiles for s in tile for p in s.previous}
                if program.slots == 1 else protected
            )
            new_tiles = tuple(
                _fold_kernel_stages(_inline_wrapper_stages(
                    tuple(unwrap_stage(s) for s in tile),
                    protected_names=allow_inline))
                for tile in raw_tiles
            )
            return PipelineProgram(
                program.count, program.slots,
                lambda idx, tiles=new_tiles: tiles[idx],
                program.minimum_ms, program.setup_ms,
                producer_bindings=program.producer_bindings)
        return program

    if isinstance(program, ParallelProgram):
        children = tuple(coalesce_execution_program(child, unroll_limit=unroll_limit)
                         for child in program.programs)
        merged = coalesce_programs(children, widen_pipelines=True)
        reverted_unplaced = (
            len(merged) > 1 and len(merged) < len(children) and any(
                isinstance(m, KernelProgram)
                and any(not s.coupled and (set(s.work) - set(s.placement))
                        for s in m.stages)
                for m in merged
            )
        )
        if reverted_unplaced:
            merged = children
        if (len(merged) > 1 and sum(
                len(m.stages) for m in merged if isinstance(m, KernelProgram)
        ) > unroll_limit):
            child_limit = unroll_limit // max(1, len(program.programs))
            children = tuple(
                coalesce_execution_program(child, unroll_limit=child_limit)
                for child in program.programs
            )
            merged = (children if reverted_unplaced
                      else coalesce_programs(children, widen_pipelines=True))
        if (len(merged) == 1 and isinstance(merged[0], KernelProgram)
                and not any(s.acquire or s.free for s in merged[0].stages)):
            return merged[0]
        if merged == program.programs:
            return program
        return ParallelProgram(merged)

    return program


def split_cohort_kernel(programs):
    """One program per cohort, keeping independently throttled stages split.

    A coupled or supply-capped stage is one physical owner's own progress:
    external work on another owner's unit must not rephase it.  Such stages
    stay one task per owner (identical work, caps, placement and timestamp),
    while the plain support stages (reserve/load/store/release) carry the same
    work for every owner and are combined into one cohort stage.  ``None`` is
    returned when any stage cannot be combined losslessly.
    """
    first = programs[0]
    if not isinstance(first, KernelProgram):
        return None
    signature = program_signature(first)
    if any(program_signature(program) != signature for program in programs[1:]):
        return None
    stages = first.stages
    position = {stage.name: index for index, stage in enumerate(stages)}
    split = [bool(stage.coupled or stage.caps) for stage in stages]
    if not any(split):
        return merge_programs(programs)
    count = len(programs)

    def refs(name, owner=None):
        if not split[position[name]]:
            return (name,)
        if owner is None:
            return tuple(f"{name}#{index}" for index in range(count))
        return (f"{name}#{owner}",)

    rebuilt = []
    for index, stage in enumerate(stages):
        after = tuple(dep for name in stage.after for dep in refs(name))
        previous = tuple(dep for name in stage.previous for dep in refs(name))
        if not split[index]:
            merged = _merge_stages([program.stages[index]
                                    for program in programs])
            if merged is None:
                return None
            rebuilt.append(replace(merged, after=after, previous=previous))
            continue
        for owner in range(count):
            source = programs[owner].stages[index]
            if source.acquire or source.free or source.program is not None:
                return None
            rebuilt.append(replace(
                source, name=f"{stage.name}#{owner}",
                after=tuple(dep for name in stage.after
                            for dep in refs(name, owner)),
                previous=tuple(dep for name in stage.previous
                               for dep in refs(name, owner))))

    return KernelProgram(tuple(rebuilt), first.minimum_ms)
