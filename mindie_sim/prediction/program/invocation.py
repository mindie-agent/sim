"""Kernel invocation descriptors and physical binding resolution.

[Role] Own invocation schemas and bind operator invocations to program primitives.
[Invariants] Deterministic binding without hidden physical defaults.
[Timeouts & Locks] Pure in-memory resolution; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.invocation import FloorConstraint, InvocationTiming
from mindie_sim.prediction.engine import Engine, Event
from mindie_sim.prediction.program import Program
from mindie_sim.prediction.service_regime_domains.services import ResourceService


@dataclass(frozen=True)
class InvocationProgram:
    """Start one prepared body behind declared invocation timing."""

    body: Program
    timing: InvocationTiming
    #: Independently provenanced merged floors; empty keeps legacy timing only.
    constraints: tuple[FloorConstraint, ...] = ()

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
            body = self.body.start(engine, name + ":body", ready, owner=owner)
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
            body = self.body.start(engine, name + ":body", startup, owner=owner)
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


#: arch3510 FP32 vector lanes and owners per die; both are declaration inputs.
FP32_LANES = 64
A5_VECTOR_UNITS = 64
#: The 32x32-block FP8 NZ projection form needs a dedicated L1 permutation and
#: tiling setup on every AIC before its declared matrix pipeline may start.
#: Its cycles are fixed by the kernel prologue and drain, so the total duration
#: stays monotonic in work: adding a tile never removes the setup span.
NZ_BLOCK32_WEIGHT_REPRESENTATION = "fp8_block32x32_fp8_e8m0"
NZ_BLOCK32_RECIPES = frozenset({"kv_a", "wkv_kv_a", "kv_a_proj"})
NZ_BLOCK32_PROLOGUE_CYCLES = 12375


def _arch35(context) -> bool:
    """A5 ``DAV_3510`` capability, read from the declared physical mechanism."""
    mechanisms = getattr(context, "physical_mechanisms", None)
    hierarchy = getattr(mechanisms, "memory_hierarchy", None)
    arch = dict(getattr(hierarchy, "provenance", {}) or {}).get("arch")
    if arch:
        return str(arch).startswith("DAV_3510")
    units = getattr(context, "resource_units", {}) or {}
    return int(units.get("vector", 0)) >= A5_VECTOR_UNITS


def _clock_per_ms(context) -> float:
    """Declared vector clock in cycles per millisecond."""
    names = context.resource_names
    vector = names.get("vector")
    if vector is not None and vector in context.rates:
        units = max(1, int(context.resource_units.get("vector", A5_VECTOR_UNITS)))
        return float(context.rates[vector]) / (units * FP32_LANES)
    cube = names.get("cube")
    return float(context.rates.get(cube, 1.0)) if cube else 1.0


def _block32_prologue_ms(context):
    """A5 32x32-block NZ matrix setup span, or ``None`` outside that domain."""
    if context.service_boundary != "device_kernel" or not _arch35(context):
        return None
    op = context.op
    if str(getattr(op, "recipe", "")) not in NZ_BLOCK32_RECIPES:
        return None
    metadata = dict(getattr(op, "metadata", ()) or ())
    if metadata.get("weight_representation") != NZ_BLOCK32_WEIGHT_REPRESENTATION:
        return None
    return NZ_BLOCK32_PROLOGUE_CYCLES / _clock_per_ms(context)


@dataclass(frozen=True)
class _PrologueProgram:
    """One declared kernel prologue/drain around an already-prepared body."""

    body: Program
    latency_ms: float

    def start(self, engine, name, ready, *, owner=""):
        prologue = engine.submit(name + ":prologue", after=(ready,),
                                 latency=self.latency_ms, owner=owner)
        return self.body.start(engine, name + ":body", prologue, owner=owner)


def lower_invocation(recipe, context):
    timing = context.invocation_timing
    prologue = _block32_prologue_ms(context)
    if timing is None:
        body = recipe.lower(context)
        return _PrologueProgram(body, prologue) if prologue else body
    body = recipe.lower(replace(context, minimum_ms=0, invocation_timing=None))
    if prologue:
        body = _PrologueProgram(body, prologue)
    constraints = _constraints(context, timing)
    # Keep ``timing`` as the invocation's own declaration; the merged floor
    # stays in ``constraints`` so its provenance is never overwritten.
    if timing.startup_ms == 0 and timing.minimum_ms > 0:
        return _FloorJoinProgram(body, timing.minimum_ms, timing.provenance,
                                 constraints)
    return InvocationProgram(body, timing, constraints)
