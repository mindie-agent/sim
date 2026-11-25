"""Kernel invocation descriptors and physical binding resolution.

[Role] Own invocation schemas and bind operator invocations to program primitives.
[Invariants] Deterministic binding without hidden physical defaults.
[Timeouts & Locks] Pure in-memory resolution; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.invocation import FloorConstraint, InvocationTiming
from mindie_sim.prediction.engine import Engine, Event
from mindie_sim.prediction.program import KernelProgram, Program, Stage
from mindie_sim.prediction.service_regime_domains.services import ResourceService


@dataclass(frozen=True)
class InvocationProgram:
    """Start one prepared body behind declared invocation timing."""

    body: Program
    timing: InvocationTiming
    #: Independently provenanced merged floors; empty keeps legacy timing only.
    constraints: tuple[FloorConstraint, ...] = ()
    #: Optional deferred body resolver evaluated when startup finishes.
    body_resolver: Callable[[], Program] | None = None

    def __post_init__(self):
        if not isinstance(self.timing, InvocationTiming):
            raise TypeError("invocation_program_requires_invocation_timing")
        constraints = tuple(self.constraints)
        if not all(isinstance(item, FloorConstraint) for item in constraints):
            raise TypeError("invocation_program_requires_floor_constraints")
        object.__setattr__(self, "constraints", constraints)

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        startup_ms = max((self.timing.startup_ms, *(item.startup_ms
                          for item in self.constraints)), default=self.timing.startup_ms)
        minimum_ms = max((self.timing.minimum_ms, *(item.minimum_ms
                          for item in self.constraints)), default=self.timing.minimum_ms)
        if startup_ms == 0 and ready.done:
            body_prog = self.body_resolver() if self.body_resolver is not None else self.body
            body = body_prog.start(engine, name + ":body", ready, owner=owner)
            if minimum_ms == 0:
                return body
            floor = engine.submit(
                name + ":minimum", after=(ready,), latency=minimum_ms, owner=owner,
            )
            return engine.submit(name + ":complete", after=(body, floor), owner=owner)

        completion = engine.future(name + ":complete")
        floor = (
            engine.submit(name + ":minimum", after=(ready,), latency=minimum_ms, owner=owner)
            if minimum_ms else None
        )

        def launch(startup: Event) -> None:
            # Expand the body only after startup has completed. This keeps idle
            # invocation state bounded and preserves the body's input readiness.
            body_prog = self.body_resolver() if self.body_resolver is not None else self.body
            body = body_prog.start(engine, name + ":body", startup, owner=owner)
            dependencies = (body,) if floor is None else (body, floor)
            engine.submit(
                name + ":body_complete", after=dependencies, owner=owner,
                on_complete=lambda _: engine.resolve(completion),
            )

        engine.submit(
            name + ":startup", after=(ready,), latency=startup_ms,
            on_complete=launch, owner=owner,
        )
        return completion


class InvocationPolicy:
    def __init__(self, rules=()):
        self.rules = []
        allowed = {"hardware", "runtime", "revision", "format", "kernel", "implementation",
                   "dtype", "layout", "replay_scope", "recipe", "family", "phase"}
        for raw in rules:
            match = dict(raw.get("match", {}))
            if match.keys() - allowed:
                raise ModelingError("invocation_unknown_identity_axes:" +
                                    ",".join(sorted(match.keys() - allowed)))
            timing = InvocationTiming.from_mapping({k: v for k, v in raw.items() if k != "match"})
            payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
            self.rules.append((match, timing, hashlib.sha256(payload.encode()).hexdigest()))

    def select(self, context, recipe, *, devices=()):
        op = context.op
        metadata = dict(op.metadata)
        explicit_class = metadata.get("invocation_class") or metadata.get(
            "program_execution", {}).get("execution_class")
        # A receive-only rank still enters an explicitly declared native
        # group. It may match a kernel rule even without locally charged work;
        # absent that rule, hardware class selection still requires evidence.
        if (context.service_boundary != "device_kernel"
                or (not context.work and not getattr(recipe, "algorithm", None)
                    and not (op.source_boundary == "kernel" and explicit_class)
                    and not metadata.get("device_kernel_group"))
                or op.family in {"barrier", "token_commit"}):
            return None
        identity = {**context.service_identity,
                    "hardware": context.service_identity["hardware"]["key"],
                    "recipe": op.recipe, "family": op.family, "phase": op.phase}
        matches = [(len(match), timing, key, match) for match, timing, key in self.rules
                   if all(identity.get(axis) == value for axis, value in match.items())]
        if not matches:
            selected = hardware_invocation(context, devices, explicit_class=explicit_class)
            if selected is None:
                return None
            timing, record = selected
        else:
            specificity = max(row[0] for row in matches)
            selected = [row for row in matches if row[0] == specificity]
            if len(selected) != 1:
                raise ModelingError("invocation_ambiguous_matching_rules")
            _, timing, key, match = selected[0]
            record = {"id": key, "match": match, **timing.to_dict()}
        if (isinstance(recipe, ResourceService) and timing.startup_ms
                and recipe.model.estimate(context.native_work.get(recipe.model.resource, 0))
                is not None):
            # An inclusive response already owns its invocation startup. A
            # second opaque device startup cannot be separated from that fit.
            raise ModelingError("invocation_startup_duplicates_resource_service_boundary")
        return timing, record


_HW_INV_CACHE = {}


def hardware_invocation(context, devices, *, explicit_class=None):
    """An enclosing physical kernel pays its declared execution classes once."""
    work = context.native_work
    if explicit_class:
        classes = (explicit_class,)
    elif work.get("link", 0) or context.op.family == "collective":
        # Communication libraries and pure DMA do not imply an AIV kernel.
        return None
    else:
        classes = tuple(role for role in ("cube", "vector") if work.get(role, 0))
    cache_key = (
        tuple(
            (device.rank, tuple((role, timing._cache_sig)
                                for role, timing in device.kernel_invocations))
            for device in devices
        ),
        classes,
    )
    if cache_key in _HW_INV_CACHE:
        return _HW_INV_CACHE[cache_key]
    selected = [
        {"rank": device.rank, "class": role, **timing.to_dict()}
        for device in devices for role, timing in device.kernel_invocations if role in classes
    ]
    if not selected:
        if len(_HW_INV_CACHE) >= 1024:
            _HW_INV_CACHE.pop(next(iter(_HW_INV_CACHE)))
        _HW_INV_CACHE[cache_key] = None
        return None
    # A representative TP/PCP shard conservatively uses its slowest member.
    # Mixed cube/vector is one invocation, not two stacked dispatch delays.
    timing = InvocationTiming(
        startup_ms=max(row["startup_ms"] for row in selected),
        minimum_ms=max(row["minimum_ms"] for row in selected),
        provenance={"basis": "hardware_execution_class", "declarations": selected},
    )
    payload = json.dumps(timing.to_dict(), sort_keys=True, separators=(",", ":"))
    res = timing, {
        "id": hashlib.sha256(payload.encode()).hexdigest(),
        "match": {"execution_classes": classes}, "source": "device.kernel_invocations",
        **timing.to_dict(),
    }
    if len(_HW_INV_CACHE) >= 1024:
        _HW_INV_CACHE.pop(next(iter(_HW_INV_CACHE)))
    _HW_INV_CACHE[cache_key] = res
    return res


@dataclass(frozen=True)
class _FloorJoinProgram:
    """Zero-startup invocation whose completion floor needs no second task.

    The declared floor completes at ``ready + minimum`` and the invocation
    completes at ``max(body, floor)``.  When the input is already ready that
    instant is known at submission, so one zero-work boundary with an explicit
    release expresses it exactly.  A not-yet-ready input keeps the general
    ``InvocationProgram`` (startup, floor and join) unchanged.
    """

    body: Program
    minimum_ms: float
    #: The selection provenance is carried as data; SOL admission reads it.
    provenance: Mapping = field(default_factory=dict)
    #: Independently provenanced merged floors; empty keeps legacy behavior.
    constraints: tuple[FloorConstraint, ...] = ()

    def effective_minimum(self):
        return max((self.minimum_ms, *(item.minimum_ms
                                       for item in self.constraints)),
                   default=self.minimum_ms)

    def start(self, engine, name, ready, *, owner=""):
        minimum = self.effective_minimum()
        if ready.time is None:
            return InvocationProgram(
                self.body,
                InvocationTiming(minimum_ms=minimum, provenance=self.provenance),
                self.constraints,
            ).start(engine, name, ready, owner=owner)
        body = self.body.start(engine, name + ":body", ready, owner=owner)
        return engine.submit(
            name + ":complete", after=(body,),
            release=ready.time + minimum, owner=owner,
        )


def _constraints(context, timing):
    """Keep each merged source's floor and provenance separate."""
    constraints = [FloorConstraint(timing.startup_ms, timing.minimum_ms,
                                   timing.provenance, "invocation")]
    minimum = float(getattr(context, "minimum_ms", 0.0) or 0.0)
    if minimum > 0:
        constraints.append(FloorConstraint(
            0.0, minimum,
            dict(getattr(context, "minimum_provenance", {}) or {}),
            "service_regime"))
    return tuple(constraints)


def _plan_exec_meta(recipe, context) -> Mapping | None:
    recipe_meta = getattr(recipe, "metadata", None)
    if isinstance(recipe_meta, Mapping) and (
        "launch_count" in recipe_meta or "launch_boundaries" in recipe_meta
    ):
        return recipe_meta
    op = getattr(context, "op", None)
    op_meta = getattr(op, "metadata", None) if op is not None else None
    if op_meta:
        prog_exec = dict(op_meta).get("program_execution")
        if isinstance(prog_exec, Mapping):
            return prog_exec
    return None


def _declared_launch_count(recipe, context) -> int:
    """Return declared sequential device kernel launch count (default 1)."""
    meta = _plan_exec_meta(recipe, context)
    raw = meta.get("launch_count") if meta is not None else None
    if type(raw) is int and raw >= 0:
        return raw
    return 1


def _declared_launch_boundaries(recipe, context) -> tuple[str, ...]:
    """Return declared sequential launch-boundary stage names when present."""
    meta = _plan_exec_meta(recipe, context)
    raw = meta.get("launch_boundaries") if meta is not None else None
    if isinstance(raw, (list, tuple)) and raw and all(type(item) is str and item for item in raw):
        return tuple(raw)
    return ()


def _wrap_launch_program(
    body: Program, timing: InvocationTiming, constraints: tuple[FloorConstraint, ...],
) -> Program:
    if timing.startup_ms == 0 and timing.minimum_ms > 0:
        return _FloorJoinProgram(body, timing.minimum_ms, timing.provenance, constraints)
    return InvocationProgram(body, timing, constraints)


def _wrap_boundary_stages(
    program: Program,
    boundaries: tuple[str, ...],
    timing: InvocationTiming,
    single_constraints: tuple[FloorConstraint, ...],
) -> Program | None:
    """Wrap each declared launch-boundary Stage in its own per-launch invocation."""
    if not isinstance(program, KernelProgram):
        return None
    stage_names = [s.name for s in program.stages]
    if len(stage_names) == len(boundaries) and set(stage_names) == set(boundaries):
        if any(s.acquire or s.free or s.previous for s in program.stages):
            return None
        wrapped = tuple(
            Stage(
                s.name,
                after=s.after,
                program=_wrap_launch_program(
                    KernelProgram((replace(s, after=()),)),
                    timing,
                    single_constraints,
                ),
            )
            for s in program.stages
        )
        return KernelProgram(wrapped, minimum_ms=program.minimum_ms)
    if (
        len(program.stages) == 2
        and program.stages[0].name == "setup"
        and program.stages[1].name == "body"
        and program.stages[1].program is not None
    ):
        inner = _wrap_boundary_stages(
            program.stages[1].program, boundaries, timing, single_constraints,
        )
        if inner is not None:
            return KernelProgram(
                (program.stages[0], replace(program.stages[1], program=inner)),
                minimum_ms=program.minimum_ms,
            )
    return None


def lower_invocation(recipe, context):
    timing = context.invocation_timing
    if timing is None:
        return recipe.lower(context)
    launch_count = _declared_launch_count(recipe, context)
    if launch_count == 0:
        return recipe.lower(replace(context, invocation_timing=None))
    body = recipe.lower(replace(context, minimum_ms=0, invocation_timing=None))
    if launch_count > 1 and (timing.startup_ms > 0.0 or timing.minimum_ms > 0.0):
        boundaries = _declared_launch_boundaries(recipe, context)
        if len(boundaries) == launch_count:
            single_constraints = (
                FloorConstraint(timing.startup_ms, timing.minimum_ms,
                                timing.provenance, "invocation"),
            )
            wrapped_body = _wrap_boundary_stages(body, boundaries, timing, single_constraints)
            if wrapped_body is not None:
                regime_min = float(getattr(context, "minimum_ms", 0.0) or 0.0)
                if regime_min > 0.0:
                    regime_prov = dict(getattr(context, "minimum_provenance", {}) or {})
                    regime_constraint = (
                        FloorConstraint(0.0, regime_min, regime_prov, "service_regime"),
                    )
                    return _FloorJoinProgram(
                        wrapped_body, regime_min, regime_prov, regime_constraint,
                    )
                return wrapped_body
        provenance = {**dict(timing.provenance), "launch_count": launch_count}
        timing = InvocationTiming(
            startup_ms=timing.startup_ms * launch_count,
            minimum_ms=timing.minimum_ms * launch_count,
            provenance=provenance,
        )
    constraints = _constraints(context, timing)
    # Keep ``timing`` as the invocation's own declaration; the merged floor
    # stays in ``constraints`` so its provenance is never overwritten.
    return _wrap_launch_program(body, timing, constraints)
