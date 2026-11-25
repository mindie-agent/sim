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

from mindie_sim.prediction.efficiency.cost import CostLedger, CostModel
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


@dataclass(frozen=True)
class _StandaloneCase:
    """The subset of Case consumed by the existing Execution boundary."""

    machine: Hardware
    model: None = None
    workload: None = None
    runtime: Software = field(default_factory=Software)
    slo: SloSpec = field(default_factory=SloSpec)


def _work_roles(
    op: OpSpec, cost_model: Any, replay_scope: str = "none",
) -> dict[str, float]:
    """Resolve declared resource roles through the single canonical CostModel."""
    model = getattr(cost_model, "model", cost_model)
    try:
        priced = model.price(op, replay_scope, ledger=CostLedger())
    except ValueError as exc:
        raise ModelingError(str(exc)) from exc
    return dict(priced.work)


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


def _service_op(
    op: OpSpec, service: NormalizedResourceService, context,
    cost_model: CostModel, replay_scope: str,
) -> OpSpec:
    """Map declared OpSpec work onto the exact physical role a service names."""
    role = service.resource
    roles = _work_roles(op, cost_model, replay_scope)
    if set(roles) == {role}:
        return op
    if role in {"mte2_aiv", "mte2_aic"} and set(roles) == {"mte2"}:
        metadata = dict(op.metadata)
        resource_work = dict(metadata.get("resource_work", {}) or {})
        mte2_total = float(roles["mte2"])
        resource_work.pop("mte2", None)
        resource_work[role] = float(resource_work.get(role, 0.0)) + mte2_total
        metadata["resource_work"] = resource_work
        return replace(op, mte2_bytes=0.0,
                       metadata=tuple(sorted(metadata.items())))
    raise ModelingError(
        "user_profile_service_requires_one_declared_resource_phase:"
        + op.op_id + ":" + ",".join(sorted(roles)) + "!=" + role
        + ":placement=" + ",".join(sorted(context.resource_names)))


def _overlay_recipes(base: RecipeBook, extra: Mapping[str, Any]) -> RecipeBook:
    """Overlay explicit recipe bindings onto ``base`` without mutating ``base``."""
    if not extra:
        return base
    merged = RecipeBook(
        recipes={},
        families={},
        programs={},
        default=base.default,
        family_registry=getattr(base, "_family_registry", None),
    )
    merged.recipes = {**base.recipes, **extra}
    merged.families = dict(base.families)
    merged.programs = dict(base.programs)
    merged._explicit_recipes = (
        frozenset(getattr(base, "_explicit_recipes", ())) | frozenset(extra)
    )
    merged._caller_recipes = (
        frozenset(getattr(base, "_caller_recipes", ())) | frozenset(extra)
    )
    merged._explicit_families = frozenset(getattr(base, "_explicit_families", ()))
    merged._caller_programs = frozenset(getattr(base, "_caller_programs", ()))
    merged._family_kinds = dict(getattr(base, "_family_kinds", {}))
    return merged


def _collective_replica_group(
    op: OpSpec, replica: int, topology: Any,
) -> tuple[int, ...]:
    """Return sorted DP replicas participating in ``op``'s collective group."""
    metadata = dict(op.metadata)
    parallel = topology.parallel
    stage = int(metadata.get("source_stage", op.stage))
    ext_dp = max(1, int(getattr(parallel, "external_dp", 1) or 1))
    dp_count = max(1, ext_dp * int(parallel.dp))
    owner = {
        int(rank): rep
        for rep in range(dp_count)
        for rank in topology.ranks(rep, stage)
    }
    axis = str(metadata.get("group_axis", "tp"))
    declared = metadata.get("group_ranks")
    if declared:
        groups: Sequence[Sequence[int]] = (tuple(declared),)
    elif axis == "etp":
        from mindie_sim.prediction.sharding.local_tp import expert_tp_groups
        groups = expert_tp_groups(topology, replica=replica, stage=stage)
    else:
        groups = tuple(topology.groups(axis, replica, stage, metadata=metadata))
    rep_ranks = {int(rk) for rk in topology.ranks(replica, stage)}
    matched: set[int] = set()
    for group in groups:
        group_ints = {int(rk) for rk in group}
        if declared or (group_ints & rep_ranks):
            matched.update(owner[rk] for rk in group_ints if rk in owner)
    if not matched:
        for group in groups:
            matched.update(owner[int(rk)] for rk in group if int(rk) in owner)
    return tuple(sorted(matched))


def _collective_key(op: OpSpec) -> str:
    meta = dict(op.metadata)
    grp_ranks = meta.get("group_ranks")
    if grp_ranks and meta.get("group_axis") == "local_tp":
        return f"{op.op_id}@{','.join(map(str, grp_ranks))}"
    return op.op_id


def _replica_submission_groups(
    works: Mapping[int, IterationWork],
    topology: Any,
    parallel: ParallelSpec,
    machine: Hardware,
) -> list[tuple[int, ...]]:
    """Partition active replicas into cohorts covering all cross-DP collectives."""
    from mindie_sim.prediction.execution.dispatch import cohorts
    from mindie_sim.prediction.sharding.local_tp import collective_requires_cohort

    active = list(works)
    parent = {rep: rep for rep in active}

    def find(rep: int) -> int:
        while parent[rep] != rep:
            parent[rep] = parent[parent[rep]]
            rep = parent[rep]
        return rep

    def union_all(members: Sequence[int]) -> None:
        present = [rep for rep in members if rep in parent]
        if len(present) <= 1:
            return
        root = find(present[0])
        for rep in present[1:]:
            parent[find(rep)] = root

    for group in cohorts(parallel, machine):
        union_all(group)
    for rep, work in works.items():
        for op in work.ops:
            if collective_requires_cohort(op, topology):
                union_all(_collective_replica_group(op, rep, topology))
    grouped: dict[int, list[int]] = {}
    for rep in active:
        grouped.setdefault(find(rep), []).append(rep)
    return [tuple(members) for members in grouped.values()]


def _operator_report_row(
    op: OpSpec,
    rep: int,
    *,
    multi_mapped: bool,
    physical: Sequence[Mapping[str, Any]],
    start_ms: float,
    complete_ms: float,
    duration: float,
    input_ready_ms: float,
    serviced: Mapping[str, Any],
    execution: Execution,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
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
    charge_event = next(
        (event for event in events if not event["zero_work"]), events[0])
    row = {
        "physical_id": op.op_id,
        **({"replica": rep} if multi_mapped else {}),
        "operator": op.to_dict(),
        "occurrences": 1,
        "program": (op.recipe or charge_event["program"]),
        "duration_ms": duration,
        "total_ms": duration,
        "input_ready_ms": input_ready_ms,
        "ready_ms": start_ms,
        "complete_ms": complete_ms,
        "serviced": serviced,
        "charge": "call",
        "service_attribution": "call_terminal_once",
        "physical_members": [event["physical_id"] for event in events],
        "physical_event_count": len(events),
        "physical_events": events,
        "basis": execution.cost.summary(),
        "runtime_uid": execution.engine.uid,
        "live_tasks": execution.engine.summary()["live_tasks"],
    }
    return row, events


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
            case, CostModel(case, hardware=bindings.hardware, primitive_table=bindings.primitive_table),
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

    def _replica_ops(
        self,
        ops: Sequence[OpSpec] | Mapping[int, Sequence[OpSpec] | OpSpec] | OpSpec,
        *,
        replica: int = 0,
    ) -> tuple[dict[int, tuple[OpSpec, ...]], bool]:
        dp = max(1, self.runtime.parallel.dp)
        if isinstance(ops, Mapping):
            mapped: dict[int, tuple[OpSpec, ...]] = {}
            for rep, rep_ops in ops.items():
                if isinstance(rep, bool) or not isinstance(rep, int) or rep < 0 or rep >= dp:
                    raise ModelingError(
                        "standalone_graph_replica_outside_declared_placement:" + str(rep))
                if isinstance(rep_ops, OpSpec):
                    mapped[rep] = (rep_ops,)
                elif isinstance(rep_ops, (str, bytes, bytearray)):
                    raise ModelingError("standalone_entry_requires_op_spec")
                else:
                    try:
                        mapped[rep] = tuple(rep_ops)
                    except TypeError as exc:
                        raise ModelingError("standalone_entry_requires_op_spec") from exc
            return mapped, True
        if isinstance(replica, bool) or not isinstance(replica, int) or replica < 0 or replica >= dp:
            raise ModelingError(
                "standalone_graph_replica_outside_declared_placement:" + str(replica))
        if isinstance(ops, OpSpec):
            seq = (ops,)
        elif isinstance(ops, (str, bytes, bytearray)):
            raise ModelingError("standalone_entry_requires_op_spec")
        else:
            try:
                seq = tuple(ops)
            except TypeError as exc:
                raise ModelingError("standalone_entry_requires_op_spec") from exc
        return {replica: seq}, False

    def _profiled_entries(
        self, ops: Sequence[OpSpec], profile: UserProfile | None,
        service_id: str | None, execution: Execution,
        *, replay_scope: str = "none", replica: int = 0, iteration: int = 0,
    ) -> tuple[tuple[OpSpec, ...], dict[str, Any]]:
        ops = tuple(_identity_dtype(op) for op in ops)
        if profile is None:
            return tuple(ops), {}
        recipes: dict[str, Any] = {}
        normalized: list[OpSpec] = []
        for index, op in enumerate(ops):
            if _is_zero_work(op):
                normalized.append(op)
                continue
            metadata = dict(op.metadata)
            metadata.setdefault("kernel", op.recipe or op.family)
            op = replace(op, metadata=tuple(sorted(metadata.items())))
            conditions, stage_context, _die, _identity = self._placement_conditions(
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
            op = _service_op(op, service, stage_context, execution.cost, replay_scope)
            role = service.resource
            roles = _work_roles(op, execution.cost, replay_scope)
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
            key = f"user-profile:{service.service_id}:r{replica}:i{iteration}:{op.op_id}:{index}"
            recipes[key] = ProfileServiceRecipe(service)
            normalized.append(replace(op, recipe=key))
        return tuple(normalized), recipes

    def _profiled(self, ops: Sequence[OpSpec], profile: UserProfile | None,
                  service_id: str | None, execution: Execution,
                  *, replay_scope: str = "none", replica: int = 0,
                  iteration: int = 0
                  ) -> tuple[tuple[OpSpec, ...], RecipeBook | None]:
        normalized, recipes = self._profiled_entries(
            ops, profile, service_id, execution,
            replay_scope=replay_scope, replica=replica, iteration=iteration)
        return normalized, (
            _overlay_recipes(execution.recipes, recipes)
            if profile is not None else None
        )

    def _prepare(self, ops: Sequence[OpSpec], profile: UserProfile | None,
                 service_id: str | None, execution: Execution,
                 *, replay_scope: str = "none", replica: int = 0,
                 iteration: int = 0
                 ) -> tuple[tuple[OpSpec, ...], RecipeBook]:
        normalized, profile_book = self._profiled(
            ops, profile, service_id, execution, replay_scope=replay_scope,
            replica=replica, iteration=iteration)
        book = profile_book or execution.recipes
        return tuple(book.expand(normalized)), book

    def _prepare_replicas(
        self,
        replica_ops: Mapping[int, Sequence[OpSpec]],
        profile: UserProfile | None,
        service_id: str | None,
        execution: Execution,
        *,
        replay_scope: str = "none",
        iteration: int = 0,
    ) -> tuple[dict[int, tuple[OpSpec, ...]], RecipeBook]:
        merged_recipes: dict[str, Any] = {}
        normalized_by_replica: dict[int, tuple[OpSpec, ...]] = {}
        for rep, rep_ops in replica_ops.items():
            normalized, rep_recipes = self._profiled_entries(
                rep_ops, profile, service_id, execution,
                replay_scope=replay_scope, replica=rep, iteration=iteration)
            normalized_by_replica[rep] = normalized
            merged_recipes.update(rep_recipes)
        book = (
            _overlay_recipes(execution.recipes, merged_recipes)
            if profile is not None else execution.recipes
        )
        expanded = {
            rep: tuple(book.expand(rep_ops))
            for rep, rep_ops in normalized_by_replica.items()
        }
        return expanded, book

    def _build_external_events(
        self,
        execution: Execution,
        external_inputs: Mapping[str, Any] | None,
        input_ready_ms: float,
    ) -> dict[str, Event]:
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
        return external

    @staticmethod
    def _check_cohort_completeness(
        execution: Execution, selected: Mapping[int, IterationWork],
    ) -> None:
        from mindie_sim.prediction.sharding.local_tp import collective_requires_cohort

        present = set(selected)
        for rep, work in selected.items():
            for op in work.ops:
                if not collective_requires_cohort(op, execution.topology):
                    continue
                peers = _collective_replica_group(op, rep, execution.topology)
                if not set(peers) <= present:
                    recipe = execution.recipes.select(op)
                    if getattr(recipe, "algorithm", None):
                        raise ModelingError("joint_collective_requires_complete_cohort")

    def _submit_replica_works(
        self,
        execution: Execution,
        works: Mapping[int, IterationWork],
        input_ready_ms: float,
        replica_externals: Mapping[int, Mapping[str, Event]],
    ) -> dict[int, Any]:
        groups = _replica_submission_groups(
            works, execution.topology, self.runtime.parallel, self.machine)
        steps: dict[int, Any] = {}
        for group in groups:
            selected = {rep: works[rep] for rep in group}
            self._check_cohort_completeness(execution, selected)
            if len(selected) > 1:
                steps.update(execution.add_group(
                    selected, input_ready_ms,
                    external_inputs=replica_externals))
            else:
                rep, work = next(iter(selected.items()))
                steps[rep] = execution.add(
                    work, rep, input_ready_ms,
                    external_inputs=replica_externals[rep])
        return steps

    def _execute_replicas(
        self,
        execution: Execution,
        expanded_by_replica: Mapping[int, Sequence[OpSpec]],
        *,
        input_ready_ms: float,
        replay_scope: str,
        captured: list[dict[str, Any]],
        external_inputs: Mapping[str, Any] | None = None,
        wait: bool = True,
        iteration: int | None = None,
    ) -> dict[str, Any]:
        external = self._build_external_events(
            execution, external_inputs, input_ready_ms)
        if iteration is None:
            iteration = getattr(execution, "_t15_submit_index", 0)
            try:
                setattr(execution, "_t15_submit_index", iteration + 1)
            except (AttributeError, TypeError):
                iteration = 0
        pp = max(1, self.runtime.parallel.pp)
        works = {
            rep: IterationWork(
                tuple(rep_ops), IterationPlan((), iteration=iteration),
                pp, 0, 0, None, "standalone", "standalone",
                replay_scope=replay_scope)
            for rep, rep_ops in expanded_by_replica.items()
        }
        replica_externals = {
            rep: {
                key: event for key, event in external.items()
                if any(key in op.deps for op in rep_ops)
            }
            for rep, rep_ops in expanded_by_replica.items()
        }
        steps = self._submit_replica_works(
            execution, works, input_ready_ms, replica_externals)
        multi_replica = len(expanded_by_replica) > 1
        joins = [{
            "id": op.op_id, "stream": op.stream, "scope": replay_scope,
            "dependencies": list(op.deps),
            **({"replica": rep} if multi_replica else {}),
        } for rep, rep_ops in expanded_by_replica.items()
          for op in rep_ops if _is_zero_work(op)]
        only_step = next(iter(steps.values())) if len(steps) == 1 else None
        if not wait:
            return {
                "execution": execution, "step": only_step, "steps": steps,
                "complete_ms": None, "charge_ready_ms": None,
                "duration_ms": None, "input_ready_ms": input_ready_ms,
                "joins": joins,
            }
        unique_completions = tuple(dict.fromkeys(
            step.completion for step in steps.values()))
        execution.engine.advance(until=unique_completions, first=False)
        ready_times = [
            item["ready"].time for item in captured
            if item["operator"].get("family") != "barrier"
        ]
        charge_ready = min(ready_times) if ready_times else input_ready_ms
        complete_ms = max(
            (step.device_complete.time for step in steps.values()),
            default=input_ready_ms,
        )
        return {
            "execution": execution, "step": only_step, "steps": steps,
            "joins": joins,
            "complete_ms": complete_ms,
            "charge_ready_ms": charge_ready,
            "duration_ms": complete_ms - charge_ready,
            "input_ready_ms": input_ready_ms,
        }

    def _execute(self, execution: Execution, expanded: Sequence[OpSpec], *,
                 input_ready_ms: float, replay_scope: str,
                 captured: list[dict[str, Any]], replica: int = 0,
                 external_inputs: Mapping[str, Any] | None = None,
                 wait: bool = True
                 ) -> dict[str, Any]:
        return self._execute_replicas(
            execution, {replica: expanded}, input_ready_ms=input_ready_ms,
            replay_scope=replay_scope, captured=captured,
            external_inputs=external_inputs, wait=wait)

    @staticmethod
    def _register_deferred_recipe_restore(
        execution: Execution,
        steps: Mapping[int, Any],
        previous_recipes: RecipeBook,
    ) -> None:
        """Keep deferred profile/custom recipes live until lazy submission finishes."""
        pending_events = tuple(dict.fromkeys(
            step.completion for step in steps.values()
            if not getattr(step.completion, "done", False)
        ))
        if not pending_events:
            execution.recipes = previous_recipes
            return
        active_submits = int(getattr(execution, "_t15_pending_submits", 0))
        if active_submits == 0:
            execution._t15_saved_recipes = previous_recipes
        execution._t15_pending_submits = active_submits + 1
        remaining = [len(pending_events)]

        def _on_step_done(_ev, ex=execution, rem=remaining):
            rem[0] -= 1
            if rem[0] > 0:
                return
            left = max(0, int(getattr(ex, "_t15_pending_submits", 1)) - 1)
            ex._t15_pending_submits = left
            if left == 0 and hasattr(ex, "_t15_saved_recipes"):
                ex.recipes = ex._t15_saved_recipes
                delattr(ex, "_t15_saved_recipes")

        for ev in pending_events:
            execution.engine.on_complete(ev, _on_step_done, native_hook=True)

    def _run_replicas(
        self,
        replica_ops: Mapping[int, Sequence[OpSpec]],
        profile: UserProfile | None,
        service_id: str | None,
        *,
        input_ready_ms: float = 0,
        replay_scope: str = "none",
        external_inputs: Mapping[str, Any] | None = None,
        execution: Execution | None = None,
        recipes: RecipeBook | None = None,
        wait: bool = True,
    ) -> tuple[dict[str, Any], Execution, dict[int, tuple[OpSpec, ...]]]:
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

        def observe(original, obs_replica, program, ready, complete):
            captured.append({
                "physical_id": original.op_id,
                "operator": original.to_dict(),
                "replica": obs_replica, "program": program,
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
            if wait:
                def observe_existing(original, obs_replica, program, ready, complete):
                    if previous_observer is not None:
                        previous_observer(original, obs_replica, program, ready, complete)
                    observe(original, obs_replica, program, ready, complete)

                execution.on_operator = observe_existing
        iteration = getattr(execution, "_t15_submit_index", 0)
        try:
            setattr(execution, "_t15_submit_index", iteration + 1)
        except (AttributeError, TypeError):
            iteration = 0
        defer_restore = borrowed and not wait
        succeeded = False
        try:
            if recipes is not None:
                if borrowed and getattr(execution, "_t15_pending_submits", 0) > 0:
                    pending_profiles = {
                        k: v for k, v in previous_recipes.recipes.items()
                        if k.startswith("user-profile:")
                    }
                    execution.recipes = _overlay_recipes(recipes, pending_profiles)
                else:
                    execution.recipes = recipes
            expanded, book = self._prepare_replicas(
                replica_ops, profile, service_id, execution,
                replay_scope=replay_scope, iteration=iteration)
            execution.recipes = book
            result = self._execute_replicas(
                execution, expanded, input_ready_ms=input_ready_ms,
                replay_scope=replay_scope, captured=captured,
                external_inputs=external_inputs, wait=wait,
                iteration=iteration)
            if defer_restore and (profile is not None or recipes is not None):
                self._register_deferred_recipe_restore(
                    execution, result["steps"], previous_recipes)
            succeeded = True
        finally:
            if borrowed and (not defer_restore or not succeeded):
                execution.recipes = previous_recipes
            if borrowed and wait:
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

    def _run(self, ops: Sequence[OpSpec], profile: UserProfile | None,
             service_id: str | None, *, input_ready_ms: float = 0,
             replay_scope: str = "none", replica: int = 0,
             external_inputs: Mapping[str, Any] | None = None,
             execution: Execution | None = None,
             recipes: RecipeBook | None = None, wait: bool = True
             ) -> tuple[dict[str, Any], Execution, tuple[OpSpec, ...]]:
        result, actual, expanded = self._run_replicas(
            {replica: ops}, profile, service_id,
            input_ready_ms=input_ready_ms, replay_scope=replay_scope,
            external_inputs=external_inputs, execution=execution,
            recipes=recipes, wait=wait)
        return result, actual, expanded[replica]

    def _operator_cohort_plan(
        self,
        replica_ops: Mapping[int, Sequence[OpSpec]],
        recipes: RecipeBook | None,
    ) -> dict[tuple[int, int], tuple[Any, dict[int, OpSpec]]]:
        """Group complete cross-DP collective occurrences across replicas for isolated pricing."""
        if len(replica_ops) <= 1:
            return {}
        has_candidate = any(
            op.link_bytes > 0 or "group_axis" in dict(op.metadata)
            for rep_ops in replica_ops.values()
            for op in rep_ops
        )
        if not has_candidate:
            return {}
        from mindie_sim.prediction.sharding.local_tp import collective_requires_cohort

        probe = self.execution(recipes=recipes)
        seen_occ: dict[tuple[int, int, str], int] = {}
        buckets: dict[tuple[tuple[int, ...], int, str, int], dict[int, tuple[int, OpSpec]]] = {}
        for rep, rep_ops in replica_ops.items():
            for idx, op in enumerate(rep_ops):
                if not collective_requires_cohort(op, probe.topology):
                    continue
                peers = _collective_replica_group(op, rep, probe.topology)
                coll_key = _collective_key(op)
                occ_key = (rep, op.stage, coll_key)
                occ = seen_occ.get(occ_key, 0)
                seen_occ[occ_key] = occ + 1
                buckets.setdefault((peers, op.stage, coll_key, occ), {})[rep] = (idx, op)
        member_to_cohort: dict[tuple[int, int], tuple[Any, dict[int, OpSpec]]] = {}
        for cohort_key, members in buckets.items():
            peers = cohort_key[0]
            if not set(peers) <= set(members):
                continue
            cohort_ops = {p: members[p][1] for p in peers}
            for p in peers:
                member_to_cohort[(p, members[p][0])] = (cohort_key, cohort_ops)
        return member_to_cohort

    def operator_report(
        self,
        ops: Sequence[OpSpec] | Mapping[int, Sequence[OpSpec] | OpSpec],
        *,
        profile: UserProfile | None = None,
        service_id: str | None = None,
        input_ready_ms: float = 0,
        recipes: RecipeBook | None = None,
        replica: int = 0,
    ) -> dict[str, Any]:
        replica_ops, multi_mapped = self._replica_ops(ops, replica=replica)
        for rep_ops in replica_ops.values():
            self._validate(rep_ops, allow_external_inputs=True)
        member_to_cohort = self._operator_cohort_plan(replica_ops, recipes)
        cohort_results: dict[Any, tuple[dict[str, Any], Execution]] = {}
        rows = []
        timeline = []
        total = 0.0
        all_ops: list[OpSpec] = []
        for rep, rep_ops in replica_ops.items():
            all_ops.extend(rep_ops)
            for idx, op in enumerate(rep_ops):
                cohort_info = member_to_cohort.get((rep, idx))
                if cohort_info is not None:
                    cohort_key, cohort_ops = cohort_info
                    if cohort_key not in cohort_results:
                        isolated_cohort = {
                            p: (replace(peer_op, deps=()),)
                            for p, peer_op in cohort_ops.items()
                        }
                        cohort_results[cohort_key] = self._run_replicas(
                            isolated_cohort, profile, service_id,
                            input_ready_ms=input_ready_ms, recipes=recipes,
                        )[:2]
                    result, execution = cohort_results[cohort_key]
                    physical = [row for row in result["rows"] if row["replica"] == rep]
                    ready_times = [
                        row["ready_ms"] for row in physical
                        if row["operator"].get("family") != "barrier"
                    ]
                    start_ms = min(ready_times) if ready_times else float(input_ready_ms)
                    complete_ms = max(
                        (row["complete_ms"] for row in physical),
                        default=float(input_ready_ms),
                    )
                    duration = complete_ms - start_ms
                else:
                    isolated = replace(op, deps=())
                    result, execution, _ = self._run(
                        (isolated,), profile, service_id,
                        input_ready_ms=input_ready_ms, recipes=recipes, replica=rep)
                    physical = result["rows"]
                    start_ms = result["charge_ready_ms"]
                    complete_ms = result["complete_ms"]
                    duration = result["duration_ms"]
                if not physical:
                    raise ModelingError(
                        "standalone_operator_produced_no_event:" + op.op_id)
                total += duration
                row, events = _operator_report_row(
                    op, rep, multi_mapped=multi_mapped, physical=physical,
                    start_ms=start_ms, complete_ms=complete_ms, duration=duration,
                    input_ready_ms=input_ready_ms, serviced=result["serviced"],
                    execution=execution)
                timeline.extend(events)
                rows.append(row)
        return {
            "schema": "mindie-sim.operators/1",
            "boundary": "standalone_declared_device_programs",
            "operator_occurrences": len(rows),
            "operator_events": len(timeline),
            "operator_sum_ms": total,
            "timeline": timeline,
            "implementation_selections": _selection_records(all_ops),
            "operators": rows,
        }

    def submit_graph(
        self,
        ops: Sequence[OpSpec] | Mapping[int, Sequence[OpSpec] | OpSpec],
        *,
        profile: UserProfile | None = None,
        service_id: str | None = None,
        input_ready_ms: float = 0,
        external_inputs: Mapping[str, Any] | None = None,
        execution: Execution | None = None,
        replica: int = 0,
        recipes: RecipeBook | None = None,
    ):
        """Submit one declared graph and return its real ScheduledIteration.

        Callers that compose in an existing Engine use the returned step and
        Execution handles; ``graph_report`` is the waiting/projection view of
        the same path.
        """
        replica_ops, mapped = self._replica_ops(ops, replica=replica)
        if not replica_ops or not any(replica_ops.values()):
            raise ModelingError("standalone_graph_requires_op_spec_sequence")
        external = dict(external_inputs or {})
        for rep_ops in replica_ops.values():
            self._validate(rep_ops, allow_external_inputs=bool(external),
                           external_keys=tuple(external))
        result, actual, normalized = self._run_replicas(
            replica_ops, profile, service_id, input_ready_ms=input_ready_ms,
            external_inputs=external, execution=execution,
            recipes=recipes, wait=False)
        return (
            result["steps"] if mapped else result["step"],
            actual,
            result,
            normalized if mapped else normalized[replica],
        )

    def graph_report(
        self,
        ops: Sequence[OpSpec] | Mapping[int, Sequence[OpSpec] | OpSpec],
        *,
        profile: UserProfile | None = None,
        service_id: str | None = None,
        input_ready_ms: float = 0,
        external_inputs: Mapping[str, Any] | None = None,
        execution: Execution | None = None,
        replica: int = 0,
        recipes: RecipeBook | None = None,
    ) -> dict[str, Any]:
        replica_ops, _ = self._replica_ops(ops, replica=replica)
        if not replica_ops or not any(replica_ops.values()):
            raise ModelingError("standalone_graph_requires_op_spec_sequence")
        external = dict(external_inputs or {})
        for rep_ops in replica_ops.values():
            self._validate(rep_ops, allow_external_inputs=bool(external),
                           external_keys=tuple(external))
        result, actual, normalized_by_replica = self._run_replicas(
            replica_ops, profile, service_id, input_ready_ms=input_ready_ms,
            external_inputs=external, execution=execution, recipes=recipes)
        all_normalized = tuple(
            op for rep_ops in normalized_by_replica.values() for op in rep_ops
        )
        return {
            "schema": "mindie-sim.graph/1",
            "boundary": "standalone_declared_graph",
            "duration_ms": result["duration_ms"],
            "input_ready_ms": input_ready_ms,
            "charge_ready_ms": result["charge_ready_ms"],
            "complete_ms": result["complete_ms"],
            "replicas": list(replica_ops),
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
            "implementation_selections": _selection_records(all_normalized),
            "runtime_uid": actual.engine.uid,
            "execution": actual.summary(),
            "operators": [op.to_dict() for op in all_normalized],
        }

    def call_report(self, ops: Sequence[OpSpec], *, invocations: int = 1,
                    profile: UserProfile | None = None,
                    service_id: str | None = None,
                    mode: str = "eager", input_ready_ms: float = 0,
                    max_operators: int = 8192,
                    recipes: RecipeBook | None = None,
                    replica: int = 0) -> dict[str, Any]:
        if isinstance(ops, Mapping):
            raise ModelingError("standalone_native_call_requires_op_spec_sequence")
        if not isinstance(invocations, int) or isinstance(invocations, bool) \
                or invocations < 1:
            raise ModelingError("standalone_call_requires_positive_invocations")
        if mode not in {"eager", "graph"}:
            raise ModelingError("standalone_call_mode_requires_eager_or_graph")
        if (not isinstance(max_operators, int) or isinstance(max_operators, bool)
                or max_operators < 1):
            raise ModelingError(
                "standalone_call_requires_positive_operator_budget")
        replica_ops, _ = self._replica_ops(ops, replica=replica)
        declaration = replica_ops[replica]
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
