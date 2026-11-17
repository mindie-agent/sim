"""Canonical per-evaluation context for one prediction run.

[Role]
  Freeze the declared hardware representation once per evaluation, build the
  immutable execution bindings (topology, recipes, structural resource
  signature), and own the bounded evaluation-local caches used by physical
  grouping and isolated programs.  This module contains infrastructure only: it
  never prices an operator, lowers a program or executes device work.

[Invariants]
  * ``Device.to_dict()`` runs at most once per declared device during setup.
  * One evaluation constructs exactly one ``CostModel`` and one
    ``ExecutionBindings``; every isolated execution receives a fresh
    ``CostLedger`` from the caller so row ``basis`` records stay isolated.
  * Caches are evaluation-local and bounded.  There is no process-global
    ``id(object)`` prediction cache and no cross-case result cache.
  * A changed hardware counterfactual (rates, pools, topology, instruction
    services, quantization, runtime policy or recipe selection) produces a new
    evaluation context; the structural signature is explicit and hashable.

[Timeouts & Locks]
  Pure construction from already-parsed declarations.  No I/O, no locks and no
  device context initialization.

[Side-Effect Policy]
  Reads the parsed Case; never mutates it, the catalog, the filesystem or any
  global state.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from contextvars import ContextVar
import time
import hashlib
import json
from typing import Any, Mapping

from mindie_sim.schema.case import Case as CaseSpec


BOUNDED_CACHE_LIMIT = 4096
SIGNATURE_JSON_FLAGS = {"sort_keys": True, "separators": (",", ":"), "default": str}


def _canonical(value: Any) -> str:
    return json.dumps(value, **SIGNATURE_JSON_FLAGS)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class BoundedCache:
    """Small insertion-ordered cache with a hard size bound.

    Evaluation-local by construction: one instance is created by
    ``EvaluationContext`` and is never shared between cases.
    """

    __slots__ = ("_data", "limit", "hits", "misses", "evictions")

    def __init__(self, limit: int = BOUNDED_CACHE_LIMIT) -> None:
        self.limit = max(1, int(limit))
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            value = self._data.pop(key)
        except KeyError:
            self.misses += 1
            return default
        self._data[key] = value
        self.hits += 1
        return value

    def put(self, key: Any, value: Any) -> Any:
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.limit:
            self._data.popitem(last=False)
            self.evictions += 1
        return value

    __setitem__ = put

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._data), "limit": self.limit,
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions}


@dataclass(frozen=True)
class FrozenHardwareView:
    """Immutable hardware representation for one evaluation.

    ``_die_payloads`` holds one ``to_dict()`` result per declared device; the
    tuple is built during setup and is the only materialization of the mutable
    ``Device`` mapping in the evaluation.  ``dies`` exposes those same dict
    objects read-only to pricing/topology code that needs a mapping.
    """

    identity: str
    architecture: str
    hardware_key: str
    compute_peaks: Mapping[str, float]
    dies: tuple[Mapping[str, Any], ...]
    die_signatures: tuple[str, ...]
    signature: tuple

    @classmethod
    def from_machine(cls, machine: Any) -> "FrozenHardwareView":
        payloads: list[Mapping[str, Any]] = []
        signatures: list[str] = []
        for die in machine.dies:
            payload = die.to_dict()
            payloads.append(payload)
            signatures.append(_canonical(payload))
        identity = str(machine.hardware_key or machine.sku)
        peaks = dict(machine.compute_peaks)
        signature = (
            identity, str(machine.arch_family), str(machine.sku),
            tuple(signatures), _canonical(peaks),
        )
        return cls(identity=identity, architecture=str(machine.arch_family),
                   hardware_key=str(machine.hardware_key or machine.sku),
                   compute_peaks=peaks, dies=tuple(payloads),
                   die_signatures=tuple(signatures), signature=signature)

    def payload(self) -> dict[str, Any]:
        """The mapping expected by ``primitive_key`` and topology construction."""
        return {"identity": self.identity,
                "dies": list(self.dies),
                "compute_peaks": dict(self.compute_peaks)}

    def digest(self) -> str:
        return _digest(self.signature)


def _stage_geometry(case: CaseSpec,
                    hardware: FrozenHardwareView) -> tuple:
    """Per (replica, stage) declared die geometry, from the frozen payloads."""
    parallel = case.runtime.parallel
    width = parallel.tp * parallel.pcp
    geometry = []
    for replica in range(int(parallel.dp)):
        for stage in range(int(parallel.pp)):
            start = (replica * parallel.pp + stage) * width
            indexes = tuple(range(start, start + width))
            geometry.append((
                replica, stage, indexes,
                tuple(hardware.die_signatures[index] for index in indexes),
            ))
    return tuple(geometry)


def _runtime_signature(case: CaseSpec) -> tuple:
    runtime = case.runtime
    quantization = getattr(runtime, "quantization", None)
    return (
        str(runtime.vllm_impl), str(runtime.vllm_version),
        str(runtime.kernel_format), str(runtime.graph_capture_mode),
        str(runtime.collective_startup_mode), bool(runtime.async_scheduling),
        _canonical(quantization if isinstance(quantization, Mapping) else {}),
        _canonical({str(key): value for key, value in dict(runtime.kernels).items()
                    if key != "recipes"}),
    )


#: Cooperative per-candidate/outer sweep deadline.  Every pricing loop polls
#: this value; no signal handler, thread or subprocess is involved.
ACTIVE_DEADLINE_MONO: ContextVar[float | None] = ContextVar(
    "mindie_sim_deadline_mono", default=None)


def check_active_deadline() -> None:
    deadline = ACTIVE_DEADLINE_MONO.get()
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("simulation_deadline_exceeded")


@dataclass(frozen=True)
class ExecutionBindings:
    """Immutable bindings shared by graph, operator, cohort and efficiency runs."""

    hardware: FrozenHardwareView
    resource_signature: tuple
    stage_geometry: tuple
    resources: tuple = ()
    pools: tuple = ()
    topology: Any = None
    recipes: Any = None
    stage_profiles: tuple = ()

    @classmethod
    def create(cls, case: CaseSpec, recipes: Any = None) -> "ExecutionBindings":
        from .topology import Topology
        hardware = FrozenHardwareView.from_machine(case.machine)
        geometry = _stage_geometry(case, hardware)
        topology = Topology(
            case.machine, case.runtime.parallel, "",
            collective_startup_mode=case.runtime.collective_startup_mode)
        signature = (
            hardware.digest(), geometry, _runtime_signature(case),
            tuple(getattr(item, "name", str(item)) for item in topology.resources()),
        )
        from .execution import build_stage_profiles
        profiles = build_stage_profiles(case, "")
        return cls(hardware=hardware, resource_signature=signature,
                   stage_geometry=geometry, topology=topology, recipes=recipes,
                   stage_profiles=profiles)

    def signature_for(self, stage: int, replica: int) -> tuple:
        return (self.resource_signature, int(stage), int(replica))

    def profile_key(self, replica: int, stage: int) -> tuple:
        for profile in self.stage_profiles:
            if profile.replica == int(replica) and profile.stage == int(stage):
                return profile.profile_key
        return (self.resource_signature, int(replica), int(stage))

    def with_recipes(self, recipes: Any) -> "ExecutionBindings":
        if recipes is self.recipes:
            return self
        return ExecutionBindings(
            hardware=self.hardware, resource_signature=self.resource_signature,
            stage_geometry=self.stage_geometry, resources=self.resources,
            pools=self.pools, topology=self.topology, recipes=recipes,
            stage_profiles=self.stage_profiles)


@dataclass
class EvaluationContext:
    """One evaluation's frozen hardware, bindings, cost model and caches."""

    case: CaseSpec
    bindings: ExecutionBindings
    cost: Any
    physical_keys: BoundedCache = field(default_factory=BoundedCache)
    prepared_programs: BoundedCache = field(default_factory=BoundedCache)
    isolated_results: BoundedCache = field(default_factory=BoundedCache)
    #: ``case`` when the recipe book is the one the case declares, ``caller``
    #: when a caller supplied an override.  Only a case-declared book may be
    #: reused across evaluations of the same declaration.
    recipe_origin: str = "caller"
    construction_counts: dict[str, int] = field(
        default_factory=lambda: {"cost_models": 1, "bindings": 1,
                                 "contexts": 1})

    @classmethod
    def create(cls, case: CaseSpec, recipes: Any = None,
               primitives: Any = None) -> "EvaluationContext":
        from .cost import CostModel
        from .program import RecipeBook
        book = recipes or RecipeBook.from_mapping(case.runtime.kernels)
        bindings = ExecutionBindings.create(case, book)
        cost = CostModel(case, primitives, hardware=bindings.hardware)
        return cls(case=case, bindings=bindings, cost=cost,
                   recipe_origin="case" if recipes is None else "caller")

    def cost_summary(self) -> dict[str, Any]:
        return self.cost.summary()

    def hardware_payload(self) -> dict[str, Any]:
        return self.bindings.hardware.payload()


def context_for(case: CaseSpec, context: EvaluationContext | None,
                recipes: Any = None) -> EvaluationContext:
    """Reuse an explicit evaluation context or create one for a standalone call.

    The recipe book of an explicit context describes the case it was created
    from.  A caller that has consumed target declarations passes its own
    ``recipes`` to execution; it does not silently rebuild the frozen bindings.
    """
    if context is None:
        return EvaluationContext.create(case, recipes)
    return context
