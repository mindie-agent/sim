"""Main Execution runner compiling iteration DAGs onto the shared Engine.

[Role] Coordinate stage topology, host and STARS submission, iteration DAGs,
  and summaries on one shared Engine.
[Invariants] Reuse one CostModel and ExecutionBindings per evaluation. Host
  bus resources are unique per node, root complex and direction. Configured
  STARS queues run at real kernel boundaries, including without host
  submission, skipping empty views and avoiding step-end launch floors.
[Timeouts & Locks] Finite virtual time, bounded caches, no locks.
[Side-Effect Policy] Reads SKU calibration and updates simulation state and caches.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache, partial
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork
from ..cohort_traffic import bind_cohort_traffic
from ..collectives.topology import Topology
from ..cost import CostModel
from ..engine import Engine, Event, Resource
from ..execution_state import memory_bindings_for
from ..mechanism import Geometry
from ..program import RecipeBook
from ..runtime.l2_residency import AdmissionMemoryView, primitive_working_set_sweep
from ..serving_io import ServingIO
from ..step_metrics import StepSummary
from .cohort import CollectiveParticipant, _CohortMixin
from .context import (
    INLINE_FRONTIER_TASK_CAP,
    StageProfile,
    _ContextMixin,
    build_stage_profiles,
)
from .lazy_submit import _LazySubmitMixin
from .step_helpers import (
    attach_stars_queues,
    bind_stream_plans,
    build_execution_summary,
    build_stars_summary,
    build_submission_summary,
    finalize_step,
    observe_collective_readiness,
    schedule_eager_moe_split_barrier,
    schedule_op_offload_reads,
    schedule_op_offload_writes,
    schedule_step_offload_tails,
    try_lazy_add,
)
from .streams import (
    META_CACHE_LIMIT,
    _NO_REV_OPS_CACHE,
    ResourceRates,
    _LedgerBoundCost,
    _isolated_script_key,
    _ledger_recorder,
    _rank_token,
    mark_no_rev_collective,
    meta_scan_flags,
)
from .submission import (
    ScheduledIteration,
    SubmissionPolicy,
    is_zero_kernel_view,
)

HOST_LOOKAHEAD_DEFAULT = 8
FRAGMENT_CACHE_LIMIT_DEFAULT = 64


@lru_cache
def _sku_working_set_sweep(sku):
    from mindie_sim.primitives.loader import load_manifest, load_primitives

    for key, entry in load_manifest()["tables"].items():
        if sku in (key, entry.get("sku"), *entry.get("compatible_skus", ())):
            return primitive_working_set_sweep(load_primitives(sku))
    return {}


class Execution(_ContextMixin, _CohortMixin, _LazySubmitMixin):
    schedule_replay = False
    _isolated_scripts: dict = {}

    def __init__(
        self, case: CaseSpec, cost: CostModel, *, recipes: RecipeBook | None = None,
        submission: SubmissionPolicy | None = None, trace_limit: int = 0,
        engine: Engine | None = None, namespace: str = "",
        include_submission: bool = True, on_operator=None,
        include_serving_io: bool = False,
        client_tails=None, memory_policy=None,
        legacy_diagnostics: bool | None = None,
        fragment_state=None, memory=None,
        lazy_frontier: bool = True, host_lookahead: int = HOST_LOOKAHEAD_DEFAULT,
        bindings=None, ledger=None, isolated_replica: int | None = None,
        evaluation=None, accel_eager_attach: bool = False,
        stream_plans=None,
    ):
        from ..cost import CostLedger
        from ..invocation_binding import InvocationPolicy
        from ..offload import resolve_case_offload_policy
        from ..participant_profile import normalise as _normalise_profile
        from ..service_regimes import RegimePolicy
        from mindie_sim.observations.resource_services import normalize_resource_service

        self.case, self.bindings = case, bindings
        self.evaluation = evaluation
        self._shared_eval_cache = (
            None if evaluation is None else evaluation.prepared_programs)
        self.ledger = ledger if ledger is not None else CostLedger()
        self.cost_model = cost
        self.cost = _LedgerBoundCost(cost, self.ledger)
        self.namespace = namespace
        self.memory = memory
        self._active_requests: tuple = ()
        self._active_padded_tokens: int = 0
        self._active_layout: str = ""
        metadata = getattr(case.runtime, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            metadata = dict(metadata)
        if legacy_diagnostics is None:
            legacy_diagnostics = bool(
                metadata.get("legacy_diagnostics", False)
                or getattr(cost, "primitives", None) is not None)
        self.legacy_diagnostics = bool(legacy_diagnostics)
        self.include_submission = include_submission
        self.on_operator = on_operator
        self.lazy_frontier = bool(lazy_frontier)
        self.host_lookahead = max(1, int(host_lookahead))
        self.accel_eager_attach = bool(accel_eager_attach)
        self.recipes = recipes or RecipeBook.from_mapping(case.runtime.kernels)
        self.invocations = InvocationPolicy(case.runtime.kernels.get("invocations", ()))
        self.regimes = RegimePolicy(case.runtime.kernels.get("service_regimes", ()))
        self.submission = submission or SubmissionPolicy(**case.runtime.submission)
        self.submission_source = "api.submission" if submission is not None else "runtime.submission"
        self.serving_io = ServingIO(case.runtime.serving_io if include_serving_io else {},
                                    namespace, case.runtime.parallel.dp, client_tails)
        self.offload_policy = resolve_case_offload_policy(case, memory_policy)
        self._offload_memory = None
        self.step_summary = StepSummary()
        self.stage_names: dict[tuple[int, int], dict[str, str]] = {}
        self.stage_pools: dict[tuple[int, int], dict[str, str]] = {}
        self.stage_dies: dict[tuple[int, int], tuple[Any, ...]] = {}
        self.streams: dict[tuple[int, int, str], Event] = {}
        self.hosts: dict[tuple[int, int], Event] = {}
        self.submission_counts = {"eager_invocations": 0, "graph_replays": 0}
        self.prepared = {}
        self.schedules = {}

        resources, pools, pool_units = self._build_engine_resources(
            case, bindings, namespace, isolated_replica, stream_plans)
        self._bind_engine(engine, trace_limit, resources, pools, pool_units)

        attach_stars_queues(self)

        self.resource_rates = ResourceRates(self.engine.resources)
        self.stage_geometry = {
            key: replace(
                Geometry.from_devices(
                    dies, links=case.machine.links,
                    transfer_channels=case.machine.transfer_channels,
                ),
                datapath=self.stage_profiles[key].datapath,
            )
            for key, dies in self.stage_dies.items()
        }
        self._stage_geom_keys: dict = {}
        self.stage_contexts = {key: self._stage_context(key, dies)
                               for key, dies in self.stage_dies.items()}
        self.work = dict.fromkeys(("flops", "vector_ops", "hbm_bytes", "link_bytes"), 0.0)
        self.conflicts = 0
        self.recipes_used: dict[str, int] = {}
        self._collectives = None
        self._cohort_views = None
        self.native_group_joins: dict = {}
        self.native_group_launches: dict = {}
        self._program_cache: dict = {}
        self._prepare_cache: dict = {}
        self._lazy_op_cache: dict = {}
        self._lazy_op_id_cache: dict = {}
        self._lazy_stage_id_caches: dict = {}
        self._lazy_safe_id_cache: dict = {}
        self._prev_lazy_dag = None
        self._prev_lazy_entries = None
        self._prev_add_topology = None
        self._compact_step: bool = False
        self._last_invocation_record = None
        self._service_identity_cache: dict = {}
        self._op_key_cache: dict = {}
        self._transport_options: dict = {}
        self.replay_counts = {"record": 0, "shadow": 0, "replay": 0, "real": 0}
        self._trace_key = None
        self._trace_shadows: list = []
        self.readiness_report = metadata.get("collective_readiness")
        self.readiness_observations: list = []
        self.readiness_unresolved: list = []
        self._joint_collective_limit = max(0, int(trace_limit))
        self._joint_collective_trace = {} if self._joint_collective_limit else None
        self.joint_collective_stats = (
            {"observations": 0, "retained": 0,
             "max_ready_spread_ms": 0.0, "max_rendezvous_delay_ms": 0.0,
             "critical_rank": None}
            if self._joint_collective_trace is not None else None)
        self._joint_collective_ordinals: dict = {}
        self.participant_profile = _normalise_profile(
            metadata.get("collective_participant_profile"))
        self.fragment_conductors: dict = {}
        self.engine.ensure_state_ledger(fragment_state)
        self.state_accesses = None
        self.scheduled_accesses = None
        self.resource_services = {}
        profile_svc = metadata.get("resource_service_profile")
        for record in (dict(profile_svc or {}).get("services", ())
                       if isinstance(profile_svc, Mapping) else ()):
            service = normalize_resource_service(record)
            self.resource_services[service.service_id] = service

    def _build_engine_resources(
        self, case, bindings, namespace, isolated_replica, stream_plans,
    ):
        from ..offload import offload_resources

        resources, pools, pool_units = [], {}, {}
        resources.extend(self.serving_io.resources())
        resources.extend(offload_resources(
            self.offload_policy, namespace, case.runtime.parallel.dp, case.runtime.parallel.pp))
        p = case.runtime.parallel
        selected_replica = None if isolated_replica is None else int(isolated_replica)
        self.stage_profiles: dict[tuple[int, int], StageProfile] = {}
        profiles = None
        if (bindings is not None and not namespace
                and getattr(bindings, "stage_profiles", ())):
            profiles = tuple(
                profile for profile in bindings.stage_profiles
                if selected_replica is None or profile.replica == selected_replica)
        if not profiles:
            profiles = build_stage_profiles(
                case, namespace,
                replicas=(None if selected_replica is None else (selected_replica,)),
                primitive_table=getattr(bindings, "primitive_table", None))
        if selected_replica is not None and not profiles:
            raise ModelingError("isolated_replica_out_of_range")

        active_ranks: set[int] = set()
        scheduler_replicas: set[int] = set()
        seen_host_rc: set[str] = set()
        for profile in profiles:
            replica, stage = profile.replica, profile.stage
            if replica not in scheduler_replicas:
                resources.append(Resource(
                    f"{namespace}scheduler@r{replica}", 1, "ms", "fifo"))
                scheduler_replicas.add(replica)
            self.stage_dies[replica, stage] = profile.dies
            self.stage_names[replica, stage] = profile.names
            self.stage_pools[replica, stage] = profile.stage_pools
            self.stage_profiles[replica, stage] = profile
            resources.extend(profile.resources)
            for rc_res in getattr(profile, "host_rc_resources", ()):
                if rc_res.name not in seen_host_rc:
                    seen_host_rc.add(rc_res.name)
                    resources.append(rc_res)
            pools.update(profile.pools)
            pool_units.update(profile.pool_units)
            active_ranks.update(int(die.rank) for die in profile.dies)

        self.stream_owners, self.stream_quota_by_res, stream_res = bind_stream_plans(
            stream_plans, self.stage_profiles, self.namespace)
        resources.extend(stream_res)
        self.topology = (
            bindings.topology
            if bindings is not None and bindings.topology is not None and not namespace
            else Topology(case.machine, p, namespace,
                          collective_startup_mode=case.runtime.collective_startup_mode))
        if selected_replica is None:
            resources.extend(self.topology.resources())
        else:
            resources.extend(self.topology.resources_for_ranks(active_ranks))

        return resources, pools, pool_units

    def _bind_engine(self, engine, trace_limit, resources, pools, pool_units):
        self._new_engine = partial(Engine, tuple(resources), pools=dict(pools),
                                   pool_units=dict(pool_units), trace_limit=int(trace_limit),
                                   accel_eager_attach=self.accel_eager_attach)
        if engine is None and self.memory is not None:
            bound = getattr(self.memory, "runtime_state", None)
            if isinstance(bound, Engine):
                engine = bound
        if engine is not None:
            engine.extend(resources, pools, pool_units)
        self._attach_engine(engine if engine is not None else self._new_engine())

    def _attach_engine(self, engine):
        self.engine = engine
        if self.memory is not None and not engine.bind_memory(self.memory, pools=("hbm",)):
            raise ModelingError("execution_memory_ledger_conflict")
        view = engine.admission_memory_view
        if view is None:
            dies = self.case.machine.dies
            capacities = {int(die.rank): int(dict(die.buffers).get("l2", 0)) for die in dies}
            capacities.setdefault(0, capacities[int(dies[0].rank)])
            profile = next(iter(self.stage_profiles.values()))
            sweep = primitive_working_set_sweep(profile.primitive_table)
            if not sweep and profile.dies[0].sku:
                sweep = _sku_working_set_sweep(profile.dies[0].sku)
            view = AdmissionMemoryView(capacities, memory_access=profile.datapath.memory_access,
                                       ws_sweep=sweep)
            engine.attach_admission_memory_view(view)
        self.admission_memory_view = view
        self.allocation_versions = engine._allocation_versions

    def reset_engine(self) -> None:
        """Start the next isolated price with cold memory and fresh bindings."""
        from ..cost import CostLedger
        self.ledger = CostLedger()
        self.cost = _LedgerBoundCost(self.cost_model, self.ledger)
        self.recipes_used = {}
        self._attach_engine(self._new_engine())
        for cache in (self._prepare_cache, self._program_cache, self._lazy_op_cache,
                      self._lazy_stage_id_caches, self.fragment_conductors):
            cache.clear()
        self._shared_eval_cache = None
        self.engine.ensure_state_ledger(None)
        attach_stars_queues(self)
        self.resource_rates = ResourceRates(self.engine.resources)
        self.stage_contexts = {key: self._stage_context(key, dies)
                               for key, dies in self.stage_dies.items()}
        self._collectives = None
        self._cohort_views = None
        self.native_group_joins = {}
        self.native_group_launches = {}
        self._service_identity_cache = {}
        self._prev_lazy_dag = None
        self._prev_lazy_entries = None
        self._prev_add_topology = None

    def _op_key(self, op):
        key = self._op_key_cache.get(id(op))
        if key is not None and key[0] is op:
            return key[1]
        from ..physical_key import operator_metadata_key
        value = operator_metadata_key(op.metadata, op.stage, op.layer)
        if len(self._op_key_cache) >= META_CACHE_LIMIT:
            self._op_key_cache.clear()
        self._op_key_cache[id(op)] = (op, value)
        return value

    def ledger_recorder(self, method):
        """Bind one CostModel recorder to this execution's private ledger."""
        return _ledger_recorder(method, self.ledger)

    @property
    def fragment_state(self):
        """Live view of the one typed state ledger owned by this runtime."""
        return self.engine.ensure_state_ledger()

    def submit_catalog(self, catalog, binding, *, name: str, after=(),
                       owner: str = ""):
        """Submit one immutable catalog on this Execution's RuntimeState."""
        self.engine.ensure_state_ledger()
        return self.engine.submit_catalog(
            catalog, binding, name=name, after=tuple(after), owner=owner)

    def bind_memory(self, memory, *, replica: int = 0, stage: int = 0,
                    pools=("hbm",)):
        """Idempotently bind this Execution's real T08 ledger."""
        if memory is None:
            return self.memory
        if self.memory is not None and self.memory is not memory:
            return None
        if not self.engine.bind_memory(memory, pools=pools):
            return None
        self.memory = memory
        return memory

    def fragment_conductor(
        self, *, replica: int = 0, stage: int = 0,
        resource_bindings=None, pool_bindings=None, unit_conversions=None,
        state=None, detail: bool = False,
        cache_limit: int = FRAGMENT_CACHE_LIMIT_DEFAULT,
        repeat_limit: int = 1_000_000, prefix: str | None = None,
        memory=None, memory_pools=("hbm",), memory_bindings=None,
        memory_owner=None, memory_storage=None, memory_ranges=None,
        memory_generation=None, memory_event=None, memory_host_bytes: float = 0.0,
        memory_unit: str | None = None,
    ):
        """One shared-context conductor for typed model/framework fragments."""
        from ..scheduling import FragmentConductor
        names = dict(self.stage_names.get((replica, stage), {}))
        names.update(dict(resource_bindings or {}))
        pools = dict(self.stage_pools.get((replica, stage), {}))
        pools.update(dict(pool_bindings or {}))
        if state is None:
            state = self.fragment_state
        ledger = memory if memory is not None else self.memory
        physical_memory_pools = tuple(
            pools.get(str(item), str(item)) for item in memory_pools)
        return FragmentConductor(
            self.engine, resource_bindings=names, pool_bindings=pools,
            unit_conversions=unit_conversions, state=state, detail=detail,
            memory=ledger, memory_replica=replica, memory_stage=stage,
            memory_pools=physical_memory_pools, memory_bindings=memory_bindings,
            memory_owner=memory_owner, memory_storage=memory_storage,
            memory_ranges=memory_ranges, memory_generation=memory_generation,
            memory_event=memory_event, memory_host_bytes=memory_host_bytes,
            memory_unit=memory_unit,
            cache_limit=cache_limit, repeat_limit=repeat_limit,
            prefix=prefix or f"fragment@{replica}:s{stage}:",
        )

    def submit_fragment(self, fragment, *, replica: int = 0, stage: int = 0,
                        **submit_kwargs):
        """Submit into the cached shared-context conductor for a stage."""
        key = (replica, stage)
        conductor = self.fragment_conductors.get(key)
        if conductor is None:
            conductor = self.fragment_conductor(replica=replica, stage=stage)
            self.fragment_conductors[key] = conductor
        return conductor.submit(fragment, **submit_kwargs)

    def stage_submit_boundary(self, replica: int = 0, stage: int = 0,
                              name: str = "host_submit", *, step=None):
        """Start event of a real host launch, or ``None``."""
        if step is not None:
            matches = [event for rep, stg, event in step.host_submits
                       if rep == replica and stg == stage]
            return matches[-1] if matches else None
        launch = self.hosts.get((replica, stage))
        if launch is None:
            return None
        return self.engine.start_boundary(launch, name)

    def _start_program(self, context, recipe, name, ready, *, owner,
                       local_ready=None, peer_ready=None, rank_index=None,
                       launch=None, source_send=None,
                       send_placeholders=None, replica=None,
                       memory_ledger=None, memory_pools=None,
                       memory_bindings=None, on_resolve=None, begin_after=None):
        """Keep per-core programs resident only while their operator can run."""
        engine = self.engine
        entry = {"ready": ready}
        inherited = local_ready if local_ready is not None else ready
        stage = int(context.op.stage)

        if (begin_after is None and on_resolve is None and memory_ledger is None
                and not memory_bindings and not Execution.schedule_replay
                and len(engine._tasks) < INLINE_FRONTIER_TASK_CAP
                and not self._program_uses_fifo(context, engine)):
            op = context.op
            transport, prog_key, do_compact, replayable = self._program_inputs(
                context, recipe, engine, replica, op, owner)
            if not replayable:
                (program, mapped_recs, svc_calls), _from_cache = (
                    self._lower_operator_program(
                        context, recipe, engine, replica, name, prog_key, do_compact))
                from ..program import is_pure_static_program
                if is_pure_static_program(program):
                    if mapped_recs and context.record_mapping is not None:
                        for mrec in mapped_recs:
                            context.record_mapping(mrec)
                    if svc_calls and context.record_service is not None:
                        for sargs, skwargs in svc_calls:
                            context.record_service(*sargs, **skwargs)
                    native_resources = self._native_resource_names(context, program)
                    self._record_single_native(
                        context, program, replica, name, transport=transport)
                    with engine.memory_scope(
                            memory_ledger,
                            replica=(0 if replica is None else replica),
                            stage=stage,
                            pools=(memory_pools if memory_pools is not None else ("hbm",)),
                            bindings=memory_bindings):
                        return self._start_engine_program(
                            program, engine, name, inherited, owner,
                            local_ready=local_ready, peer_ready=peer_ready,
                            rank_index=rank_index, launch=launch,
                            source_send=source_send,
                            send_placeholders=send_placeholders,
                            resource_names=native_resources,
                            transport=transport)

        complete = engine.future(name + ":program_complete")

        def begin(_):
            op = context.op
            cur_ready = entry["ready"]
            transport, prog_key, do_compact, replayable = self._program_inputs(
                context, recipe, engine, replica, op, owner)
            token = _rank_token(context)
            if replayable:
                script_key = _isolated_script_key(prog_key, context, token)
                if (engine._trace_prefix is not None and engine._trace_private
                        and engine._trace_clean and engine._trace_token != token
                        and not self._outside_trace(engine)):
                    self._trace_shadows.append((token, complete))
                    self.replay_counts["shadow"] += 1
                    return
                if engine._trace_prefix is None and engine.replay_ready():
                    engine.begin_trace(name, token)
                    self._trace_key = script_key
                    self._trace_shadows = []
                    self.replay_counts["record"] += 1
                else:
                    self.replay_counts["real"] += 1
            else:
                self.replay_counts["real"] += 1

            recording = getattr(engine, "_trace_prefix", None) == name

            def finished(_event=None):
                if recording and getattr(engine, "_trace_prefix", None) == name:
                    self._seal_trace(engine, complete)
                else:
                    engine.resolve(complete)
                if on_resolve is not None:
                    on_resolve()

            (program, mapped_recs, svc_calls), _from_cache = (
                self._lower_operator_program(
                    context, recipe, engine, replica, name, prog_key, do_compact))
            if mapped_recs and context.record_mapping is not None:
                for mrec in mapped_recs:
                    context.record_mapping(mrec)
            if svc_calls and context.record_service is not None:
                for sargs, skwargs in svc_calls:
                    context.record_service(*sargs, **skwargs)
            native_resources = self._native_resource_names(context, program)
            self._record_single_native(
                context, program, replica, name, transport=transport)
            finish = self._start_engine_program(
                program, engine, name, cur_ready, owner,
                local_ready=local_ready, peer_ready=peer_ready,
                rank_index=rank_index, launch=launch,
                source_send=source_send,
                send_placeholders=send_placeholders,
                resource_names=native_resources, transport=transport)
            engine.on_complete(finish, finished)

        def begin_with_binding(event):
            with engine.memory_scope(
                    memory_ledger, replica=(0 if replica is None else replica),
                    stage=stage,
                    pools=(memory_pools if memory_pools is not None else ("hbm",)),
                    bindings=memory_bindings):
                begin(event)

        if begin_after is None:
            engine.submit(name + ":begin", after=(inherited,), owner=owner,
                          on_complete=begin_with_binding, native_hook=True)
        else:
            boundary = engine.submit(
                name + ":begin", after=tuple(begin_after), owner=owner,
                on_complete=begin_with_binding, native_hook=True)
            entry["ready"] = boundary
        return complete

    def add(self, work: IterationWork, replica: int, release_ms: float, *,
            external_inputs=None, emitted_tokens=None,
            memory_bindings=None, state_accesses=None) -> ScheduledIteration:
        work = self._prepare_iteration_work(work, replica)
        engine = self.engine
        prefix = f"{self.namespace}r{replica}:i{work.plan.iteration}"
        ids: dict[str, Event] = dict(external_inputs or {})
        declared_commits: dict[str, Event] = {}
        host_submits: list[tuple[int, int, Event]] = []
        scope = work.replay_scope or ("full" if work.graph_bucket is not None else "none")
        if self.include_submission and self.case.runtime.async_scheduling and replica in self.prepared:
            schedule = self.prepared.pop(replica)
        else:
            scheduler_ms = self.submission.scheduler_ms if self.include_submission else 0
            schedule = engine.submit(prefix + ":schedule", {f"{self.namespace}scheduler@r{replica}": scheduler_ms},
                                     release=release_ms, owner=f"replica:{replica}")
        self.schedules[replica] = schedule
        prepared, input_ready = self.serving_io.input(self, prefix, replica, schedule, work)
        physical_ops = self.recipes.expand(work.ops)
        external = dict(external_inputs or {})

        lazy_result = try_lazy_add(
            self, work, replica, release_ms, scope=scope, schedule=schedule,
            prepared=prepared, input_ready=input_ready, physical_ops=physical_ops,
            prefix=prefix, external=external, emitted_tokens=emitted_tokens,
            memory_bindings=memory_bindings, state_accesses=state_accesses)
        if lazy_result is not None:
            return lazy_result

        from ..kernel_groups import KernelGroups, declaration
        from ..local_tp_groups import collective_requires_cohort
        graph_launch: dict[object, tuple[Event, Event]] = {}
        layer_entries: dict[int, Event] = {}
        expert_layer_events: dict[tuple[int, int], Event | None] = {}
        last_expert_done: dict[int, Event] = {}
        kv_layer_events: dict[tuple[int, int], list[Event]] = {}
        last_kv_done: dict[int, Event] = {}
        kv_writeback_layers: set[tuple[int, int]] = set()
        kv_writeback_events: list[Event] = []
        all_bound_accesses: list = []
        owned_offload: list[Event] = []
        pending_draft: dict[int, list] = {}
        last_draft_done: dict[int, Event] = {}
        kernel_groups = KernelGroups(self, physical_ops, replica, prefix)
        step_replay_ids: dict[tuple[int, int], int] = {}

        for op in physical_ops:
            original = op
            op, recipe = self.prepare(op, replica)
            stage_key = replica, op.stage
            name = prefix + ":" + op.op_id
            names = self.stage_names[stage_key]
            deps = [input_ready, *(ids[key] for key in op.deps)]
            bound_accesses = schedule_op_offload_reads(
                self, op, work, replica, prefix, name, input_ready, deps, ids,
                layer_entries, expert_layer_events, last_expert_done,
                kv_layer_events, last_kv_done, state_accesses, all_bound_accesses)
            deps.extend(kernel_groups.entry_dependencies(op, ids))
            stream_key = replica, op.stage, op.stream or "main"
            stream_tail = kernel_groups.stream_tail(op, stream_key)
            if stream_tail is not None:
                deps.append(stream_tail)
            context = self._context(op, replica, scope, recipe=recipe)
            context = replace(context, group_buffers=kernel_groups.buffer_scope(op))
            owner = f"replica:{replica}"
            captured = scope == "full" or (
                scope == "piecewise" and bool(dict(op.metadata).get("graph_captured")))
            capture_key = (
                op.stage, dict(op.metadata).get("graph_segment", 0) if scope == "piecewise" else 0)
            kernel_group = declaration(op)
            submit = (bool(kernel_group.get("entry")) if kernel_group else
                      bool(context.work or getattr(recipe, "algorithm", None)))
            host_launch = None
            if submit and self.include_submission:
                host_launch = self._submit_eager_host_launch(
                    engine, stage_key, name, names, owner, prepared, deps,
                    captured, capture_key, scope, graph_launch, host_submits)
                deps.append(host_launch)

            stars_issued = self._issue_eager_descriptors(
                engine, op, stage_key, owner, submit, captured, capture_key,
                host_launch, prepared, deps, step_replay_ids)

            barrier_entry = schedule_eager_moe_split_barrier(
                self, op, replica, scope, name, deps, ids
            )
            ready = engine.submit(name + ":ready", after=tuple(deps) or (schedule,), owner=owner)
            if (self.readiness_report
                    and op.family in {"collective", "allgather", "reduce_scatter",
                                      "allreduce", "transport"}
                    and op.link_bytes > 0
                    and dict(op.metadata).get("group_axis") not in {"ep", "local_tp"}):
                observe_collective_readiness(self, op, replica, ready)
            if op.layer >= 0 and op.layer not in layer_entries:
                layer_entries[op.layer] = ready
            ready = kernel_groups.phase_ready(op, ready, name)
            recipe_name = op.recipe or type(recipe).__name__
            self.recipes_used[recipe_name] = self.recipes_used.get(recipe_name, 0) + 1
            cross_dp = collective_requires_cohort(op, self.topology)
            if cross_dp and getattr(recipe, "algorithm", None) and self._collectives is None:
                raise ModelingError("joint_collective_requires_complete_cohort")
            if self._collectives is not None and cross_dp:
                complete = engine.future(name + ":collective")
                op_meta = dict(op.metadata)
                grp_ranks = op_meta.get("group_ranks")
                coll_key = (
                    f"{op.op_id}@{','.join(map(str, grp_ranks))}"
                    if grp_ranks and op_meta.get("group_axis") == "local_tp"
                    else op.op_id)
                self._collectives.setdefault(coll_key, []).append(CollectiveParticipant(
                    replica, ready, context, recipe, name, complete))
            else:
                physical_hbm = self.stage_pools.get(stage_key, {}).get("hbm", "hbm")
                complete = self._start_program(
                    context, recipe, name, ready, owner=owner, replica=replica,
                    memory_ledger=self.memory, memory_pools=(physical_hbm,),
                    memory_bindings=memory_bindings_for(op, memory_bindings))
            complete = kernel_groups.finish(op, context, recipe, complete, stream_key)
            if stars_issued:
                group = kernel_groups.group(op)
                physical_done = group.completion if group is not None else complete
                for s_queue, s_rec, s_alloc in stars_issued:
                    s_queue.bind_kernel_execution(
                        engine, s_rec, s_alloc,
                        kernel_start=ready, kernel_done=physical_done, owner=owner)
            ids[op.op_id] = complete
            schedule_op_offload_writes(
                self, op, work, replica, prefix, complete, bound_accesses,
                last_expert_done, last_kv_done, kv_writeback_events,
                owned_offload, last_draft_done, pending_draft)
            if self.on_operator is not None:
                op_start = barrier_entry if barrier_entry is not None else ready
                self.on_operator(original, replica, "JointNativeService" if context.joint_native_service
                                 is not None else type(recipe).__name__, op_start, complete)
            if op.family == "token_commit":
                declared_commits.update((request, complete) for request in op.requests)
            for key in self.work:
                self.work[key] += float(getattr(op, key))

        schedule_step_offload_tails(
            self, work, replica, prefix, input_ready, physical_ops, ids,
            pending_draft, last_draft_done, all_bound_accesses, kv_layer_events,
            kv_writeback_layers, kv_writeback_events, owned_offload)
        return finalize_step(
            self, work, replica, release_ms, prefix=prefix, scope=scope,
            input_ready=input_ready, physical_ops=physical_ops, ids=ids,
            declared_commits=declared_commits, host_submits=host_submits,
            kv_layer_events=kv_layer_events,
            kv_writeback_events=kv_writeback_events,
            emitted_tokens=emitted_tokens)

    def _prepare_iteration_work(self, work: IterationWork, replica: int) -> IterationWork:
        if self._collectives is None:
            if _NO_REV_OPS_CACHE.get(id(work.ops)) is not work.ops:
                has_rev = any(
                    op.metadata and meta_scan_flags(op.metadata)[1]
                    for op in work.ops)
                if has_rev:
                    work = replace(work, ops=bind_cohort_traffic(
                        self.case, {replica: work.ops})[replica])
                else:
                    mark_no_rev_collective(work.ops)
        work = self._lower_local_tp_work(work, replica)
        if (getattr(work, "recipe", None) != "source"
                and getattr(work, "plan", None) is not None
                and getattr(work.plan, "requests", ())):
            self._active_requests = tuple(work.plan.requests)
        if getattr(work, "padded_tokens", 0):
            self._active_padded_tokens = int(work.padded_tokens)
        return work

    def _submit_eager_host_launch(
        self, engine, stage_key, name, names, owner, prepared, deps,
        captured, capture_key, scope, graph_launch, host_submits,
    ):
        if captured and capture_key in graph_launch:
            host_launch, _start = graph_launch[capture_key]
        else:
            dies = self.stage_dies[stage_key]
            duration = self.submission.launch_ms(dies, captured=captured)
            host_deps = (prepared,) + ((self.hosts[stage_key],) if stage_key in self.hosts else ())
            if captured and scope == "piecewise" and self.submission.piecewise_sync_ms > 0.0:
                if deps:
                    host_deps = tuple(deps) + host_deps
                if self.submission_counts["graph_replays"] > 0:
                    duration += float(self.submission.piecewise_sync_ms)
            host_launch = engine.submit(name + ":submit", {names["host"]: duration},
                                        after=host_deps, owner=owner)
            self.hosts[stage_key] = host_launch
            self.submission_counts["graph_replays" if captured else "eager_invocations"] += 1
            start = engine.start_boundary(host_launch, name + ":host_submit")
            if captured:
                graph_launch[capture_key] = (host_launch, start)
            host_submits.append((*stage_key, start))
        return host_launch

    def _issue_eager_descriptors(
        self, engine, op, stage_key, owner, submit, captured, capture_key,
        host_launch, prepared, deps, step_replay_ids,
    ):
        stage_stars = self.stars_queues.get(stage_key, ())
        stars_issued: list[tuple[Any, Any, Any]] = []
        if stage_stars and submit and not is_zero_kernel_view(op):
            replay_slot = (
                (stage_key, "graph", capture_key)
                if captured else (stage_key, "eager", op.op_id)
            )
            if replay_slot not in step_replay_ids:
                rep_id = self._stars_replay_seq.get(stage_key, 0)
                self._stars_replay_seq[stage_key] = rep_id + 1
                step_replay_ids[replay_slot] = rep_id
            self._stars_occ_seq += 1
            desc_ready = host_launch if host_launch is not None else prepared
            for die_rank, s_queue in stage_stars:
                s_rec, s_alloc = s_queue.issue_descriptor(
                    engine, replay=step_replay_ids[replay_slot],
                    stream=op.stream or "main",
                    kernel_occurrence=self._stars_occ_seq,
                    descriptor_ready=desc_ready, owner=owner, die=die_rank)
                stars_issued.append((s_queue, s_rec, s_alloc))
                deps.append(s_rec.issue_done_event)
        return stars_issued

    def prepare_next(self, replica):
        if self.include_submission and self.case.runtime.async_scheduling:
            self.prepared[replica] = self.engine.submit(
                f"{self.namespace}r{replica}:prepare_next",
                {f"{self.namespace}scheduler@r{replica}": self.submission.scheduler_ms},
                after=(self.schedules[replica],), owner=f"replica:{replica}")

    def submission_summary(self) -> dict:
        return build_submission_summary(self)

    def stars_summary(self) -> dict[str, Any]:
        """Return STARS descriptor/uncovered summaries per stage and die."""
        return build_stars_summary(self)

    def summary(self) -> dict:
        return build_execution_summary(self)
