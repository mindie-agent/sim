"""[Role] Optional per-operator plans for the existing resource service engine.

[Invariants] One plan replaces parent device work and preserves physical units.
[Timeouts & Locks] Pure bounded plan evaluation; no I/O, subprocesses or locks.
[Side-Effect Policy] No filesystem, device, network or global-state writes.

Demands retain their physical units (FLOPs, operations or bytes). A stage's
minimum duration is an opaque concurrent completion floor, never an additional
serial delay. Explicit rate limits constrain actual resource service using the
engine's fixed reservations. Plans replace rather than duplicate parent work.
This module performs no device, filesystem, network or external-provider I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol, Sequence

from .a3_types import A3InputError

RESOURCE_UNITS = {
    "cube": "flops", "vector": "ops", "hbm": "bytes",
    "mte1": "bytes", "mte2": "bytes", "mte3": "bytes",
}

#: Backend name of the captured-kernel provider (``profiled_kernel.py``).  It
#: lives with the plan contract so the cost model can recognise an observed
#: plan without importing the provider, which stays opt-in.
PROFILED_KERNEL_BACKEND = "profiled_kernel"


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise A3InputError(f"operator_plan_invalid:{label}")
    if not math.isfinite(value) or value < 0:
        raise A3InputError(f"operator_plan_invalid:{label}")
    return float(value)


@dataclass(frozen=True)
class EstimateContext:
    compute_dtype: str = "bf16"
    weight_dtype: str = ""
    activation_dtype: str = ""
    timing_scope: str = "device_event_warm"
    quantization: str = ""


@dataclass(frozen=True)
class ExecutionStage:
    name: str
    demands: tuple[tuple[str, float], ...] = ()
    deps: tuple[str, ...] = ()
    minimum_duration_ms: float = 0.0
    notes: tuple[str, ...] = ()
    rate_limits: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or ":" in self.name:
            raise A3InputError("operator_plan_stage_name_invalid")
        seen: set[str] = set()
        for resource, amount in self.demands:
            if resource not in RESOURCE_UNITS or resource in seen:
                raise A3InputError(f"operator_plan_resource_invalid:{resource}")
            seen.add(resource)
            _nonnegative(amount, resource)
        _nonnegative(self.minimum_duration_ms, "minimum_duration_ms")
        if len(set(self.deps)) != len(self.deps):
            raise A3InputError("operator_plan_duplicate_dependency")
        limited: set[str] = set()
        for resource, rate in self.rate_limits:
            if resource not in seen or dict(self.demands)[resource] <= 0 or resource in limited:
                raise A3InputError(f"operator_plan_rate_resource_invalid:{resource}")
            limited.add(resource)
            if _nonnegative(rate, f"{resource}_rate") == 0:
                raise A3InputError(f"operator_plan_rate_invalid:{resource}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "demands": dict(self.demands),
            "units": {key: RESOURCE_UNITS[key] for key, _ in self.demands},
            "deps": list(self.deps),
            "minimum_duration_ms": self.minimum_duration_ms,
            "notes": list(self.notes),
            "rate_limits_per_ms": dict(self.rate_limits),
        }


@dataclass(frozen=True)
class KernelPlan:
    backend: str
    version: str
    stages: tuple[ExecutionStage, ...]
    provenance: Mapping[str, Any]
    buffer_certificate: Mapping[str, Any] = field(default_factory=dict)
    includes_dispatch: bool = False

    def __post_init__(self) -> None:
        if not self.backend or not self.version or not self.stages:
            raise A3InputError("operator_plan_identity_or_stages_missing")
        if not isinstance(self.includes_dispatch, bool):
            raise A3InputError("operator_plan_dispatch_flag_invalid")
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise A3InputError("operator_plan_provenance_missing")
        seen: set[str] = set()
        for stage in self.stages:
            if not isinstance(stage, ExecutionStage) or stage.name in seen:
                raise A3InputError("operator_plan_stage_duplicate_or_invalid")
            if any(dep not in seen for dep in stage.deps):
                raise A3InputError(f"operator_plan_dependency_not_prior:{stage.name}")
            seen.add(stage.name)

    def resource_totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for stage in self.stages:
            for key, amount in stage.demands:
                totals[key] = totals.get(key, 0.0) + float(amount)
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "version": self.version,
            "stages": [stage.to_dict() for stage in self.stages],
            "resource_totals": self.resource_totals(),
            "buffer_certificate": dict(self.buffer_certificate),
            "provenance": dict(self.provenance),
            "includes_dispatch": self.includes_dispatch,
        }


@dataclass(frozen=True)
class KernelEstimate:
    plan: KernelPlan | None = None
    unsupported_reason: str = ""

    def __post_init__(self) -> None:
        if self.plan is None and not self.unsupported_reason:
            raise A3InputError("operator_estimate_reason_missing")
        if self.plan is not None and self.unsupported_reason:
            raise A3InputError("operator_estimate_ambiguous")


class OperatorEstimator(Protocol):
    def estimate(self, op: Any, machine: Any, context: EstimateContext) -> KernelEstimate: ...


@dataclass(frozen=True)
class CompositeEstimator:
    estimators: Sequence[OperatorEstimator]
    strict: bool = False

    def estimate(self, op: Any, machine: Any, context: EstimateContext) -> KernelEstimate:
        reasons = []
        for provider in self.estimators:
            result = provider.estimate(op, machine, context)
            if not isinstance(result, KernelEstimate):
                raise A3InputError("operator_estimator_result_invalid")
            if result.plan is not None:
                return result
            reasons.append(result.unsupported_reason)
        return KernelEstimate(unsupported_reason=";".join(reasons) or "no_estimators_configured")


def register_kernel_plan(
    engine: Any, plan: KernelPlan, stage_names: Mapping[str, str], *,
    task_id: str, dependencies: Sequence[str] = (), start_ms: float = 0.0,
    launch_ms: float = 0.0,
) -> str:
    """Append one finite plan; the returned zero-work gate owns no parent work."""
    from .a3_resource_engine import ResourceSpec

    _nonnegative(start_ms, "start_ms")
    _nonnegative(launch_ms, "launch_ms")
    # Preflight before mutating the persistent engine.
    for stage in plan.stages:
        for alias, amount in stage.demands:
            if amount > 0 and (alias not in stage_names or stage_names[alias] not in engine.resources):
                raise A3InputError(f"operator_plan_stage_resource_missing:{alias}")
    roots = tuple(dependencies)
    if launch_ms and not plan.includes_dispatch:
        resource = stage_names.get("launch")
        if resource is None or resource not in engine.resources:
            raise A3InputError("operator_plan_launch_resource_missing")
        dispatch_id = f"{task_id}:dispatch"
        engine.add_task(dispatch_id, {resource: launch_ms}, dependencies=roots, release_ms=start_ms,
                        owner=task_id, metadata={"kind": "operator_dispatch", "backend": plan.backend})
        roots = (dispatch_id,)
    ids: dict[str, str] = {}
    for stage in plan.stages:
        identifier = f"{task_id}:plan:{stage.name}"
        demands = {stage_names[key]: float(value) for key, value in stage.demands if value > 0}
        if stage.minimum_duration_ms:
            floor_name = f"{identifier}:service_floor"
            engine.resources[floor_name] = ResourceSpec(floor_name, 1.0, "ms", "per_task")
            engine.serviced.setdefault(floor_name, 0.0)
            demands[floor_name] = stage.minimum_duration_ms
        parents = tuple(ids[name] for name in stage.deps) if stage.deps else roots
        engine.add_task(identifier, demands, dependencies=parents, release_ms=start_ms,
                        reservations={stage_names[key]: min(float(rate), engine.resources[stage_names[key]].capacity)
                                      for key, rate in stage.rate_limits} or None,
                        reservation_mode="fixed",
                        owner=task_id, metadata={"kind": "operator_execution_stage",
                                                "backend": plan.backend, "stage": stage.name,
                                                "notes": list(stage.notes)})
        ids[stage.name] = identifier
    consumed = {dep for stage in plan.stages for dep in stage.deps}
    leaves = tuple(ids[stage.name] for stage in plan.stages if stage.name not in consumed)
    engine.add_task(task_id, {}, dependencies=leaves, release_ms=start_ms, owner=task_id,
                    metadata={"kind": "operator_plan_gate", "backend": plan.backend})
    return task_id


def evaluate_kernel_plan(plan: KernelPlan, machine: Any, context: EstimateContext,
                         launch_ms: float = 0.0, *, stage: int = 0) -> dict[str, Any]:
    """Isolated finite-plan reference; shared-graph execution can take longer."""
    from .a3_resources import build_resource_specs
    from .a3_resource_engine import ResourceEngine

    if isinstance(stage, bool) or not isinstance(stage, int) or stage < 0:
        raise A3InputError("operator_plan_stage_invalid")
    specs, stage_map = build_resource_specs(machine, stages=stage+1, compute_dtype=context.compute_dtype)
    engine = ResourceEngine(specs)
    register_kernel_plan(engine, plan, stage_map[stage], task_id="kernel", launch_ms=launch_ms)
    result = engine.run(until=("kernel",))
    return {
        "elapsed_ms": result["completed_at_ms"]["kernel"],
        "trace": result["trace"], "resource_serviced": dict(engine.serviced),
    }
