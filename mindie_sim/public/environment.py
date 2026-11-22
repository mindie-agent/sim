"""Independent declared-work execution without a model, request or SLO case.

[Role]
  Build the existing ``Execution``/Engine on declared hardware and runtime only,
  then price complete ``OpSpec`` work or a declared graph.  This is the
  operator/native/graph public entry: it does not invent a model name, request,
  workload or target latency.

[Invariants]
  * ``_StandaloneCase`` carries ``model=None`` and ``workload=None``; every
    supported path supplies an explicit operator dtype/recipe and does not read
    those fields.
  * The one existing execution engine performs all resource sharing and timing.
  * T06 user services enter through ``ProfileServiceRecipe`` at the resource
    phase boundary; a service cannot become a whole-operator duration floor.
  * A selected profile that does not match the op's single declared resource
    phase fails explicitly instead of falling back to analytic pricing.

[Timeouts & Locks]
  Bounded declared operators; no iteration history, I/O, locks or devices.

[Side-Effect Policy]
  Builds fresh Engine/Execution objects; inputs are not mutated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from typing import Any

from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.engine import Event
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS, Hardware
from mindie_sim.schema.model import SloSpec
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan

from mindie_sim.observations.resource_services import NormalizedResourceService
from mindie_sim.prediction.service_regime_domains.services import service_identity

from .profiles import UserProfile
from .services import ProfileServiceRecipe, conditions_context

_STANDALONE_WORK_ROLES = (
    ("hbm_bytes", "hbm"), ("mte2_bytes", "mte2"), ("mte3_bytes", "mte3"),
    ("vector_ops", "vector"), ("flops", "cube"), ("link_bytes", "link"),
)


@dataclass(frozen=True)
class _StandaloneCase:
    """The subset of Case consumed by the existing Execution boundary."""

    machine: Hardware
    model: None = None
    workload: None = None
    runtime: Software = field(default_factory=Software)
    slo: SloSpec = field(default_factory=SloSpec)


def _work_roles(op: OpSpec) -> dict[str, float]:
    roles: dict[str, float] = {}
    for field_name, role in _STANDALONE_WORK_ROLES:
        value = float(getattr(op, field_name, 0.0) or 0.0)
        if value:
            roles[role] = roles.get(role, 0.0) + value
    metadata = dict(op.metadata)
    for role, value in dict(metadata.get("resource_work", {}) or {}).items():
        amount = float(value or 0.0)
        if amount:
            roles[str(role)] = roles.get(str(role), 0.0) + amount
    return roles


def _selection_records(ops: Sequence[OpSpec]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for op in ops:
        values = dict(op.metadata).get("implementation_selections", ()) or ()
        for value in values:
            if isinstance(value, Mapping):
                records.append(dict(value))
    return records


def _is_zero_work(op: OpSpec) -> bool:
    # A source-call child may declare no OpSpec work because its PlannedRecipe
    # generates the physical work at lowering.  Only the explicit terminal
    # barrier is a pure synchronization node.
    return op.family == "barrier"


def _declared_boundary(op: OpSpec) -> str:
    return "resource_phase" if dict(op.metadata).get("device_kernel_group") \
        else "device_kernel"


def _identity_dtype(op: OpSpec) -> OpSpec:
    """Give a matrix-only op its identity dtype without changing its work."""
    if op.dtype or not op.matrix_flops:
        return op
    dtypes = sorted({str(dtype) for dtype, _ in op.matrix_flops})
    return replace(op, dtype=(dtypes[0] if len(dtypes) == 1
                              else "mixed:" + ",".join(dtypes)))


def _service_op(op: OpSpec, service: NormalizedResourceService,
                context) -> OpSpec:
    """Map declared OpSpec work onto the exact physical role a service names."""
    role = service.resource
    roles = _work_roles(op)
    if set(roles) == {role}:
        return op
    if (role in {"mte2_aiv", "mte2_aic"} and set(roles) == {"mte2"}
            and op.mte2_bytes):
        metadata = dict(op.metadata)
        resource_work = dict(metadata.get("resource_work", {}) or {})
        resource_work[role] = op.mte2_bytes
        metadata["resource_work"] = resource_work
        return replace(op, mte2_bytes=0.0,
                       metadata=tuple(sorted(metadata.items())))
    raise ModelingError(
        "user_profile_service_requires_one_declared_resource_phase:"
        + op.op_id + ":" + ",".join(sorted(roles)) + "!=" + role
        + ":placement=" + ",".join(sorted(context.resource_names)))


@dataclass(frozen=True)
class ExecutionEnvironment:
    """One declared hardware/runtime context for independent work."""

    machine: Hardware
    runtime: Software = field(default_factory=Software)
    namespace: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.machine, Hardware):
            raise ModelingError("standalone_environment_requires_hardware")
        if not isinstance(self.runtime, Software):
            raise ModelingError("standalone_environment_requires_software")
        if not isinstance(self.namespace, str):
            raise ModelingError("standalone_environment_namespace_requires_string")
        if not self.machine.dies:
            raise ModelingError("standalone_environment_requires_a_device")

    @classmethod
    def from_hardware(cls, machine: Hardware | Mapping[str, Any], *,
                      runtime: Software | Mapping[str, Any] | None = None,
                      parallel: ParallelSpec | None = None,
                      namespace: str = "") -> ExecutionEnvironment:
        if isinstance(machine, Mapping):
            machine = Hardware.from_mapping(machine)
        if not isinstance(machine, Hardware):
            raise ModelingError("standalone_environment_requires_hardware")
        if runtime is None:
            software = Software()
        elif isinstance(runtime, Mapping):
            software = Software.from_mapping(runtime)
        elif isinstance(runtime, Software):
            software = runtime
        else:
            raise ModelingError("standalone_environment_runtime_invalid")
        if parallel is not None:
            if not isinstance(parallel, ParallelSpec):
                raise ModelingError("standalone_environment_parallel_invalid")
            software = replace(software, parallel=parallel)
        return cls(machine=machine, runtime=software, namespace=namespace)

    @property
    def case(self) -> _StandaloneCase:
        return _StandaloneCase(
            machine=self.machine, model=None, workload=None,
            runtime=self.runtime, slo=SloSpec())

    def execution(self, *, recipes: RecipeBook | None = None,
                  on_operator=None, primitive_table: Any = None) -> Execution:
        from mindie_sim.prediction.evaluation import (
            ExecutionBindings,
            _resolve_default_primitive_table,
        )

        case = self.case
        book = recipes or RecipeBook.from_mapping(case.runtime.kernels)
        eff_table = (
            None if primitive_table is False
            else (primitive_table if primitive_table is not None else _resolve_default_primitive_table(case))
        )
        bindings = ExecutionBindings.create(case, book, primitive_table=eff_table)
        return Execution(
            case, CostModel(case, hardware=bindings.hardware),
            recipes=recipes, include_submission=False,
            namespace=self.namespace, on_operator=on_operator,
            bindings=bindings,
        )

    def _validate(self, ops: Sequence[OpSpec], *, allow_external_inputs: bool,
                  external_keys: Sequence[str] = ()) -> None:
        identifiers = {op.op_id for op in ops}
        external = set(external_keys)
        for op in ops:
            if not isinstance(op, OpSpec):
                raise ModelingError("standalone_entry_requires_op_spec")
            if not op.dtype and not op.matrix_flops:
                raise ModelingError(
                    "standalone_operator_requires_explicit_dtype:" + op.op_id)
            if op.stage < 0 or op.stage >= max(1, self.runtime.parallel.pp):
                raise ModelingError(
                    "standalone_operator_stage_outside_declared_placement:"
                    + op.op_id + ":" + str(op.stage))
            missing = set(op.deps) - identifiers
            if missing and (not allow_external_inputs
                            or not missing <= external):
                raise ModelingError(
                    "standalone_graph_requires_complete_dependencies:"
                    + op.op_id + ":" + ",".join(sorted(missing)))

    def _placement(self, execution: Execution, op: OpSpec, replica: int = 0):
        key = replica, op.stage
        if key not in execution.stage_contexts:
            raise ModelingError(
                "standalone_operator_stage_outside_declared_placement:"
                + op.op_id + ":" + str(op.stage))
        return (execution.stage_contexts[key], execution.stage_dies[key][0])

    def _placement_conditions(self, execution: Execution, op: OpSpec,
                              replay_scope: str, replica: int = 0):
        context, die = self._placement(execution, op, replica)
        identity = service_identity(
            execution.case, op, replay_scope, device=die)
        return conditions_context(context, identity), context, die, identity

    def _profiled(self, ops: Sequence[OpSpec], profile: UserProfile | None,
                  service_id: str | None, execution: Execution,
                  *, replay_scope: str = "none", replica: int = 0
                  ) -> tuple[tuple[OpSpec, ...], RecipeBook | None]:
        ops = tuple(_identity_dtype(op) for op in ops)
        if profile is None:
            return tuple(ops), None
        recipes: dict[str, Any] = {}
        normalized: list[OpSpec] = []
        for index, op in enumerate(ops):
            if _is_zero_work(op):
                normalized.append(op)
                continue
            metadata = dict(op.metadata)
            metadata.setdefault("kernel", op.recipe or op.family)
            op = replace(op, metadata=tuple(sorted(metadata.items())))
            conditions, stage_context, die, identity = self._placement_conditions(
                execution, op, replay_scope, replica)
            service = profile.service(
                service_id, conditions_context=conditions)
            if service.scope != "independent_resource":
                raise ModelingError(
                    "user_profile_joint_phase_requires_an_explicit_phase_boundary:"
                    + op.op_id)
            if op.source_boundary == "call":
                raise ModelingError(
                    "user_profile_service_requires_a_physical_kernel:"
                    + op.op_id)
            op = _service_op(op, service, stage_context)
            role = service.resource
            roles = _work_roles(op)
            if set(roles) != {role}:
                raise ModelingError(
                    "user_profile_service_requires_one_declared_resource_phase:"
                    + op.op_id + ":" + ",".join(sorted(roles)))
            expected_unit = stage_context.work_units.get(
                role, NATIVE_WORK_UNITS.get(role, "work"))
            if service.work_unit != expected_unit:
                raise ModelingError(
                    "user_profile_service_work_unit_mismatch:"
                    + op.op_id + ":" + role + ":" + service.work_unit
                    + "!=" + expected_unit)
            owner_count = stage_context.resource_units.get(
                role, stage_context.core_count)
            if service.active_units != owner_count:
                raise ModelingError(
                    "user_profile_service_active_units_mismatch:"
                    + op.op_id + ":" + str(service.active_units)
                    + "!=" + str(owner_count))
            if service.boundary != _declared_boundary(op):
                raise ModelingError(
                    "user_profile_service_boundary_mismatch:"
                    + op.op_id + ":" + service.boundary
                    + "!=" + _declared_boundary(op))
            key = f"user-profile:{service.service_id}:{op.op_id}:{index}"
            recipes[key] = ProfileServiceRecipe(service)
            normalized.append(replace(op, recipe=key))
        return tuple(normalized), RecipeBook(recipes=recipes)

    def _prepare(self, ops: Sequence[OpSpec], profile: UserProfile | None,
                 service_id: str | None, execution: Execution,
                 *, replay_scope: str = "none", replica: int = 0
                 ) -> tuple[tuple[OpSpec, ...], RecipeBook]:
        normalized, profile_book = self._profiled(
            ops, profile, service_id, execution, replay_scope=replay_scope,
            replica=replica)
        book = profile_book or execution.recipes
        return tuple(book.expand(normalized)), book

    def _execute(self, execution: Execution, expanded: Sequence[OpSpec], *,
                 input_ready_ms: float, replay_scope: str,
                 captured: list[dict[str, Any]], replica: int = 0,
                 external_inputs: Mapping[str, Any] | None = None,
                 wait: bool = True
                 ) -> dict[str, Any]:
        external: dict[str, Event] = {}
        for key, value in (external_inputs or {}).items():
            if isinstance(value, Event):
                external[key] = value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)) or float(value) < 0:
                    raise ModelingError(
                        "standalone_external_input_requires_finite_nonnegative:"
                        + str(key))
                external[key] = execution.engine.submit(
                    "standalone:external:" + str(key),
                    release=max(float(value), float(input_ready_ms)))
            else:
                raise ModelingError(
                    "standalone_external_input_requires_time_or_event:"
                    + str(key))
        iteration = getattr(execution, "_t15_submit_index", 0)
        try:
            setattr(execution, "_t15_submit_index", iteration + 1)
        except (AttributeError, TypeError):
            iteration = 0
        work = IterationWork(
            tuple(expanded), IterationPlan((), iteration=iteration), 1, 0, 0,
            None, "standalone", "standalone", replay_scope=replay_scope)
        step = execution.add(
            work, replica, input_ready_ms, external_inputs=external)
        if not wait:
            return {
                "execution": execution, "step": step,
                "complete_ms": None, "charge_ready_ms": None,
                "duration_ms": None, "input_ready_ms": input_ready_ms,
                "joins": [{
                    "id": op.op_id, "stream": op.stream,
                    "scope": replay_scope, "dependencies": list(op.deps),
                } for op in expanded if _is_zero_work(op)],
            }
        execution.engine.advance(until=(step.completion,))
        joins = [{
            "id": op.op_id, "stream": op.stream, "scope": replay_scope,
            "dependencies": list(op.deps),
        } for op in expanded if _is_zero_work(op)]
        charge_ready = next(
            (item["ready"].time for item in captured
             if item["operator"].get("family") != "barrier"),
            input_ready_ms)
        complete_ms = step.device_complete.time
        return {
            "execution": execution, "step": step, "joins": joins,
            "complete_ms": complete_ms,
            "charge_ready_ms": charge_ready,
            "duration_ms": complete_ms - charge_ready,
            "input_ready_ms": input_ready_ms,
        }

    def _run(self, ops: Sequence[OpSpec], profile: UserProfile | None,
             service_id: str | None, *, input_ready_ms: float = 0,
             replay_scope: str = "none", replica: int = 0,
             external_inputs: Mapping[str, Any] | None = None,
             execution: Execution | None = None,
             recipes: RecipeBook | None = None, wait: bool = True
             ) -> tuple[dict[str, Any], Execution, tuple[OpSpec, ...]]:
        if (not isinstance(input_ready_ms, (int, float))
                or isinstance(input_ready_ms, bool)
                or not math.isfinite(float(input_ready_ms))
                or input_ready_ms < 0):
            raise ModelingError(
                "standalone_input_ready_time_must_be_finite_and_nonnegative")
        input_ready_ms = float(input_ready_ms)
        if replay_scope not in {"none", "full", "piecewise"}:
            raise ModelingError("standalone_replay_scope_invalid")
        captured: list[dict[str, Any]] = []

        def observe(original, replica, program, ready, complete):
            captured.append({
                "physical_id": original.op_id,
                "operator": original.to_dict(),
                "replica": replica, "program": program,
                "ready": ready, "complete": complete,
            })

        borrowed = execution is not None
        previous_observer = None
        previous_recipes = None
        if execution is None:
            execution = self.execution(recipes=None, on_operator=observe)
        else:
            previous_observer = execution.on_operator
            previous_recipes = execution.recipes

            def observe_existing(original, replica, program, ready, complete):
                if previous_observer is not None:
                    previous_observer(original, replica, program, ready, complete)
                observe(original, replica, program, ready, complete)

            execution.on_operator = observe_existing
        try:
            if recipes is not None:
                execution.recipes = recipes
            expanded, book = self._prepare(
                ops, profile, service_id, execution, replay_scope=replay_scope,
                replica=replica)
            execution.recipes = book
            result = self._execute(
                execution, expanded, input_ready_ms=input_ready_ms,
                replay_scope=replay_scope, captured=captured, replica=replica,
                external_inputs=external_inputs, wait=wait)
        finally:
            if borrowed:
                execution.recipes = previous_recipes
                execution.on_operator = previous_observer
        rows = ([{
            "physical_id": item["physical_id"], "operator": item["operator"],
            "replica": item["replica"], "program": item["program"],
            "ready_ms": item["ready"].time, "complete_ms": item["complete"].time,
            "elapsed_ms": item["complete"].time - item["ready"].time,
        } for item in captured] if wait else [])
        result.update({
            "rows": rows,
            "serviced": ({key: value for key, value in
                          execution.engine.serviced.items() if value}
                         if wait else {}),
        })
        return result, execution, expanded

    def operator_report(self, ops: Sequence[OpSpec], *,
                        profile: UserProfile | None = None,
                        service_id: str | None = None,
                        input_ready_ms: float = 0,
                        recipes: RecipeBook | None = None,
                        replica: int = 0) -> dict[str, Any]:
        self._validate(ops, allow_external_inputs=True)
        rows = []
        timeline = []
        total = 0.0
        for op in ops:
            isolated = replace(op, deps=())
            result, execution, _ = self._run(
                (isolated,), profile, service_id,
                input_ready_ms=input_ready_ms, recipes=recipes, replica=replica)
            physical = result["rows"]
            if not physical:
                raise ModelingError(
                    "standalone_operator_produced_no_event:" + op.op_id)
            start_ms = result["charge_ready_ms"]
            complete_ms = result["complete_ms"]
            duration = result["duration_ms"]
            total += duration
            events = []
            for captured in physical:
                operator = captured["operator"]
                zero = operator.get("family") == "barrier"
                events.append({
                    **captured,
                    "zero_work": zero,
                    "serviced": {} if zero else None,
                    "service_attribution": (
                        "zero_work_sync" if zero else "call_aggregate_only"),
                })
            timeline.extend(events)
            charge_event = next(
                (event for event in events if not event["zero_work"]), events[0])
            rows.append({
                "physical_id": op.op_id,
                "operator": op.to_dict(),
                "occurrences": 1,
                "program": (op.recipe or charge_event["program"]),
                "duration_ms": duration,
                "total_ms": duration,
                "input_ready_ms": input_ready_ms,
                "ready_ms": start_ms,
                "complete_ms": complete_ms,
                "serviced": result["serviced"],
                "charge": "call",
                "service_attribution": "call_terminal_once",
                "physical_members": [event["physical_id"] for event in events],
                "physical_event_count": len(events),
                "physical_events": events,
                "basis": execution.cost.summary(),
                "runtime_uid": execution.engine.uid,
                "live_tasks": execution.engine.summary()["live_tasks"],
            })
        return {
            "schema": "mindie-sim.operators/1",
            "boundary": "standalone_declared_device_programs",
            "operator_occurrences": len(rows),
            "operator_events": len(timeline),
            "operator_sum_ms": total,
            "timeline": timeline,
            "implementation_selections": _selection_records(ops),
            "operators": rows,
        }

    def submit_graph(self, ops: Sequence[OpSpec], *,
                     profile: UserProfile | None = None,
                     service_id: str | None = None,
                     input_ready_ms: float = 0,
                     external_inputs: Mapping[str, Any] | None = None,
                     execution: Execution | None = None,
                     replica: int = 0,
                     recipes: RecipeBook | None = None):
        """Submit one declared graph and return its real ScheduledIteration.

        Callers that compose in an existing Engine use the returned step and
        Execution handles; ``graph_report`` is the waiting/projection view of
        the same path.
        """
        declaration = tuple(ops)
        if not declaration:
            raise ModelingError("standalone_graph_requires_op_spec_sequence")
        external = dict(external_inputs or {})
        self._validate(declaration, allow_external_inputs=bool(external),
                       external_keys=tuple(external))
        result, actual, normalized = self._run(
            declaration, profile, service_id, input_ready_ms=input_ready_ms,
            replica=replica, external_inputs=external, execution=execution,
            recipes=recipes, wait=False)
        return result["step"], actual, result, normalized

    def graph_report(self, ops: Sequence[OpSpec], *,
                     profile: UserProfile | None = None,
                     service_id: str | None = None,
                     input_ready_ms: float = 0,
                     external_inputs: Mapping[str, Any] | None = None,
                     execution: Execution | None = None,
                     replica: int = 0,
                     recipes: RecipeBook | None = None) -> dict[str, Any]:
        declaration = tuple(ops)
        if not declaration:
            raise ModelingError("standalone_graph_requires_op_spec_sequence")
        external = dict(external_inputs or {})
        self._validate(declaration, allow_external_inputs=bool(external),
                       external_keys=tuple(external))
        result, actual, normalized = self._run(
            declaration, profile, service_id, input_ready_ms=input_ready_ms,
            replica=replica, external_inputs=external, execution=execution,
            recipes=recipes)
        return {
            "schema": "mindie-sim.graph/1",
            "boundary": "standalone_declared_graph",
            "duration_ms": result["duration_ms"],
            "input_ready_ms": input_ready_ms,
            "charge_ready_ms": result["charge_ready_ms"],
            "complete_ms": result["complete_ms"],
            "operator_occurrences": sum(
                1 for row in result["rows"]
                if row["operator"].get("family") != "barrier"),
            "operator_events": len(result["rows"]),
            "timeline": [
                {**row,
                 "zero_work": row["operator"].get("family") == "barrier",
                 "serviced": ({} if row["operator"].get("family") == "barrier"
                              else None),
                 "service_attribution": (
                     "zero_work_sync"
                     if row["operator"].get("family") == "barrier"
                     else "call_aggregate_only")}
                for row in result["rows"]
            ],
            "zero_work_joins": result["joins"],
            "implementation_selections": _selection_records(normalized),
            "runtime_uid": actual.engine.uid,
            "execution": actual.summary(),
            "operators": [op.to_dict() for op in normalized],
        }

    def call_report(self, ops: Sequence[OpSpec], *, invocations: int = 1,
                    profile: UserProfile | None = None,
                    service_id: str | None = None,
                    mode: str = "eager", input_ready_ms: float = 0,
                    max_operators: int = 8192,
                    recipes: RecipeBook | None = None,
                    replica: int = 0) -> dict[str, Any]:
        if not isinstance(invocations, int) or isinstance(invocations, bool) \
                or invocations < 1:
            raise ModelingError("standalone_call_requires_positive_invocations")
        if mode not in {"eager", "graph"}:
            raise ModelingError("standalone_call_mode_requires_eager_or_graph")
        if (not isinstance(max_operators, int) or isinstance(max_operators, bool)
                or max_operators < 1):
            raise ModelingError(
                "standalone_call_requires_positive_operator_budget")
        declaration = tuple(ops)
        if not declaration:
            raise ModelingError("standalone_native_call_requires_op_spec_sequence")
        self._validate(declaration, allow_external_inputs=False)
        captured: list[dict[str, Any]] = []

        def observe(original, replica, program, ready, complete):
            captured.append({
                "physical_id": original.op_id,
                "operator": original.to_dict(),
                "replica": replica, "program": program,
                "ready": ready, "complete": complete,
            })

        execution = self.execution(recipes=recipes, on_operator=observe)
        expanded, book = self._prepare(
            declaration, profile, service_id, execution,
            replay_scope=("full" if mode == "graph" else "none"),
            replica=replica)
        if len(expanded) * invocations > max_operators:
            raise ModelingError(
                "standalone_call_expanded_operator_budget_exceeded")
        repeated: list[OpSpec] = []
        identifiers = {op.op_id for op in expanded}
        for invocation in range(invocations):
            for op in expanded:
                identifier = f"call{invocation}/{op.op_id}"
                deps = tuple(f"call{invocation}/{dep}" for dep in op.deps
                             if dep in identifiers)
                if set(op.deps) - identifiers:
                    raise ModelingError(
                        "standalone_call_requires_complete_dependencies:"
                        + op.op_id)
                repeated.append(replace(op, op_id=identifier, deps=deps))
        execution.recipes = book
        replay_scope = "full" if mode == "graph" else "none"
        result = self._execute(
            execution, tuple(repeated), input_ready_ms=input_ready_ms,
            replay_scope=replay_scope, captured=captured, replica=replica)
        rows = [{
            "physical_id": item["physical_id"], "operator": item["operator"],
            "replica": item["replica"], "program": item["program"],
            "ready_ms": item["ready"].time, "complete_ms": item["complete"].time,
            "elapsed_ms": item["complete"].time - item["ready"].time,
        } for item in captured]
        return {
            "schema": "mindie-sim.call-costs/1",
            "boundary": "standalone_declared_call",
            "duration_ms": result["duration_ms"],
            "input_ready_ms": input_ready_ms,
            "charge_ready_ms": result["charge_ready_ms"],
            "complete_ms": result["complete_ms"],
            "protocol": {"invocations": invocations, "mode": mode},
            "replay_scope": replay_scope,
            "physical_operators": len(expanded),
            "expanded_operators": len(repeated),
            "operator_occurrences": sum(
                1 for row in rows
                if row["operator"].get("family") != "barrier"),
            "operator_events": len(rows),
            "timeline": [
                {**row,
                 "zero_work": row["operator"].get("family") == "barrier",
                 "serviced": ({} if row["operator"].get("family") == "barrier"
                              else None),
                 "service_attribution": (
                     "zero_work_sync"
                     if row["operator"].get("family") == "barrier"
                     else "call_aggregate_only")}
                for row in rows
            ],
            "zero_work_joins": result["joins"],
            "implementation_selections": _selection_records(repeated),
            "runtime_uid": execution.engine.uid,
            "execution": execution.summary(),
            "operators": [op.to_dict() for op in repeated],
        }

__all__ = ["ExecutionEnvironment"]
