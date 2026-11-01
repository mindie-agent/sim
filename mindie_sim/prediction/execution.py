"""Compile iteration DAGs onto the shared physical execution engine.

[Role] Bind rank placement, host submission, kernel recipes and device streams.
[Invariants] Model names do not select execution. Same-stream order and data
  dependencies are distinct from resource contention. Host submission can run
  ahead of device completion. Link work is never reduced by overlap.
[Timeouts & Locks] Finite virtual time; at most active iteration DAGs are live.
[Side-Effect Policy] Mutates only the simulation Engine and aggregate counters.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import re
from typing import Any, Mapping

from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS
from .cost import CostModel
from .cohort_traffic import bind_cohort_traffic
from .collectives import plan_options as collective_options
from .engine import Engine, Event, Resource
from .mechanism import Geometry, Work
from .program import RecipeBook, RecipeContext
from .topology import Topology
from .serving_io import ServingIO
from .step_metrics import StepSummary


def _freeze_key(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_key(item) for item in value)
    if isinstance(value, list):
        return ("__list__", tuple(_freeze_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("__set__", tuple(sorted((_freeze_key(item) for item in value), key=repr)))
    if isinstance(value, Mapping):
        return ("__map__", tuple(sorted(((str(k), _freeze_key(v)) for k, v in value.items()), key=lambda pair: pair[0])))
    try:
        hash(value)
        return value
    except TypeError:
        return ("__repr__", type(value), repr(value))


_LAYER_PREFIX_RE = re.compile(r"\bs\d+\.l\d+\.")
_DIAGNOSTIC_ONLY_META_KEYS = frozenset({
    "structure_subgraphs", "fusion_semantic_ids", "fusion_external_outputs",
})


def _op_meta_key(op) -> Any:
    def _norm(value: Any) -> Any:
        if isinstance(value, str):
            return _LAYER_PREFIX_RE.sub("s_.l_.", value)
        if isinstance(value, (int, float, bool, bytes, type(None))):
            return value
        if isinstance(value, tuple):
            return tuple(_norm(item) for item in value)
        if isinstance(value, list):
            return ("__list__", tuple(_norm(item) for item in value))
        if isinstance(value, (set, frozenset)):
            return ("__set__", tuple(sorted((_norm(item) for item in value), key=repr)))
        if isinstance(value, Mapping):
            return ("__map__", tuple(sorted(
                ((str(k), _norm(v)) for k, v in value.items()
                 if k not in _DIAGNOSTIC_ONLY_META_KEYS),
                key=lambda pair: pair[0],
            )))
        try:
            hash(value)
            return value
        except TypeError:
            return ("__repr__", type(value), repr(value))

    return _norm(dict(op.metadata))


def _meta_diff(orig: Any, res: Any) -> Any:
    if isinstance(orig, Mapping) and isinstance(res, Mapping):
        removed = frozenset(orig.keys() - res.keys())
        updated = {
            k: (_meta_diff(orig[k], v) if k in orig else ("__set__", v))
            for k, v in res.items() if k not in orig or orig[k] != v
        }
        return ("__dict__", removed, updated)
    if (isinstance(orig, tuple) and isinstance(res, tuple) and len(orig) == len(res)
            and orig and all(isinstance(x, Mapping) for x in orig)
            and all(isinstance(x, Mapping) for x in res)):
        return ("__tuple_map__", tuple(_meta_diff(a, b) for a, b in zip(orig, res)))
    return ("__set__", res)


def _meta_patch(orig: Any, diff: Any) -> Any:
    kind = diff[0]
    if kind == "__set__":
        return diff[1]
    if kind == "__dict__" and isinstance(orig, Mapping):
        _, removed, updated = diff
        out = {k: v for k, v in orig.items() if k not in removed}
        for k, sub in updated.items():
            out[k] = _meta_patch(out[k], sub) if k in out else sub[1]
        return out
    if kind == "__tuple_map__" and isinstance(orig, tuple) and len(orig) == len(diff[1]):
        return tuple(_meta_patch(a, sub) for a, sub in zip(orig, diff[1]))
    return diff[-1]


class ResourceRates(Mapping):
    """Read-only live rate view; PD extends the same Engine with new namespaces."""

    def __init__(self, resources):
        self.resources = resources

    def __getitem__(self, key):
        return self.resources[key].rate

    def __iter__(self):
        return iter(self.resources)

    def __len__(self):
        return len(self.resources)


def _execution_geometry(device):
    """A representative shard requires identical physical owner topology."""
    roles = dict(device.rates).keys() | {"cube", "host"}
    return (
        device.compute_units,
        tuple(sorted((role, device.units(role)) for role in roles)),
        tuple(sorted((pool, device.pool_units(pool)) for pool, _ in device.buffers)),
        tuple(sorted((mode, role) for mode, (role, _) in device.service_modes)),
        tuple(sorted(dict(device.compute_peaks))),
        tuple(sorted((role, dict(device.work_units).get(role, NATIVE_WORK_UNITS.get(role, "work")))
                     for role in roles | dict(device.service_modes).keys())),
    )


@dataclass(frozen=True)
class SubmissionPolicy:
    graph_replay_ms: float | None = None
    scheduler_ms: float = 0
    completion_ms: float = 0
    kernel_launch_ms: float | None = None

    def launch_ms(self, dies, *, captured=False):
        """Software submission overrides the explicit legacy device parameter."""
        eager = (self.kernel_launch_ms if self.kernel_launch_ms is not None
                 else max(d.launch_us for d in dies) / 1000)
        return (self.graph_replay_ms if captured and self.graph_replay_ms is not None
                else eager)


@dataclass(frozen=True)
class ScheduledIteration:
    replica: int
    iteration: int
    completion: Event
    start_ms: float
    mode: str
    valid_tokens: int
    padded_tokens: int
    graph_bucket: int | None
    replay_scope: str
    operators: int
    commits: Mapping[str, Event]
    device_complete: Event
    host_results: Mapping[str, Event]
    client_deliveries: Mapping[str, Event]
    implementation_selections: tuple[dict, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica": self.replica, "iteration": self.iteration,
            "start_ms": self.start_ms, "end_ms": self.completion.time,
            "duration_ms": self.completion.time - self.start_ms,
            "mode": self.mode, "valid_tokens": self.valid_tokens,
            "padded_tokens": self.padded_tokens, "graph_bucket": self.graph_bucket,
            "replay_scope": self.replay_scope, "operators": self.operators,
            "device_end_ms": self.device_complete.time,
            "host_result_ready_ms": max((event.time for event in self.host_results.values()), default=None),
            "implementation_selections": list(self.implementation_selections),
        }


@dataclass(frozen=True)
class CollectiveParticipant:
    replica: int
    ready: Event
    context: RecipeContext
    recipe: Any
    name: str
    completion: Event


class Execution:
    def __init__(
        self, case: CaseSpec, cost: CostModel, *, recipes: RecipeBook | None = None,
        submission: SubmissionPolicy | None = None, trace_limit: int = 0,
        engine: Engine | None = None, namespace: str = "",
        include_submission: bool = True, on_operator=None,
        include_serving_io: bool = False,
        client_tails=None,
    ):
        self.case, self.cost = case, cost
        self.namespace = namespace
        self.include_submission = include_submission
        self.on_operator = on_operator
        self.recipes = recipes or RecipeBook.from_mapping(case.runtime.kernels)
        from .invocation_binding import InvocationPolicy
        self.invocations = InvocationPolicy(case.runtime.kernels.get("invocations", ()))
        from .service_regimes import RegimePolicy
        self.regimes = RegimePolicy(case.runtime.kernels.get("service_regimes", ()))
        self.submission = submission or SubmissionPolicy(**case.runtime.submission)
        self.submission_source = "api.submission" if submission is not None else "runtime.submission"
        self.serving_io = ServingIO(case.runtime.serving_io if include_serving_io else {},
                                    namespace, case.runtime.parallel.dp, client_tails)
        self.step_summary = StepSummary()
        self.stage_names: dict[tuple[int, int], dict[str, str]] = {}
        self.stage_dies: dict[tuple[int, int], tuple[Any, ...]] = {}
        self.streams: dict[tuple[int, int, str], Event] = {}
        self.hosts: dict[tuple[int, int], Event] = {}
        self.submission_counts = {"eager_invocations": 0, "graph_replays": 0}
        self.prepared = {}
        self.schedules = {}
        resources, pools, pool_units = [], {}, {}
        resources.extend(self.serving_io.resources())
        p = case.runtime.parallel
        width = p.tp * p.pcp
        for replica in range(p.dp):
            resources.append(Resource(f"{namespace}scheduler@r{replica}", 1, "ms", "fifo"))
            for stage in range(p.pp):
                start = (replica * p.pp + stage) * width
                dies = tuple(case.machine.dies[start:start + width])
                if len(dies) != width:
                    raise ModelingError("placement_exceeds_physical_devices")
                geometry = _execution_geometry(dies[0])
                if any(_execution_geometry(die) != geometry for die in dies[1:]):
                    raise ModelingError(
                        "representative_shard_requires_matching_resource_geometry:"
                        f"replica={replica}:stage={stage};"
                        "heterogeneous_owners_require_per_rank_execution")
                self.stage_dies[replica, stage] = dies
                roles = dict(dies[0].rates)
                work_units = {**NATIVE_WORK_UNITS, **dict(dies[0].work_units)}
                rates = {role: (min(dict(d.rates)[role] for d in dies) / 1000,
                                work_units.get(role, "work")) for role in roles}
                rates.update({"cube": (1, "peak_ms"), "host": (1, "ms")})
                names = {}
                for kind, (rate, unit) in rates.items():
                    label = {"cube": "matrix", "hbm": "memory"}.get(kind, kind)
                    name = f"{namespace}{label}@r{replica}:s{stage}"
                    resources.append(Resource(
                        name, rate, unit, "fifo" if kind == "host" else "fair",
                        dies[0].units(kind)))
                    names[kind] = name
                self.stage_names[replica, stage] = names
                for pool in dict(dies[0].buffers):
                    key = f"{namespace}{pool}@r{replica}:s{stage}"
                    pools[key] = min(dict(d.buffers)[pool] for d in dies)
                    pool_units[key] = dies[0].pool_units(pool)
        self.topology = Topology(case.machine, p, namespace)
        resources.extend(self.topology.resources())
        if engine is not None:
            engine.extend(resources, pools, pool_units)
        self.engine = engine if engine is not None else Engine(
            resources, pools=pools, pool_units=pool_units, trace_limit=trace_limit)
        self.resource_rates = ResourceRates(self.engine.resources)
        # Hardware bindings are stage properties. Thousands of pending kernels
        # retain their own work/readiness, but share these immutable inputs.
        self.stage_geometry = {key: Geometry.from_devices(dies)
                               for key, dies in self.stage_dies.items()}
        self.stage_contexts = {key: self._stage_context(key, dies)
                               for key, dies in self.stage_dies.items()}
        self.work = dict.fromkeys(("flops", "vector_ops", "hbm_bytes", "link_bytes"), 0.0)
        self.conflicts = 0
        self.recipes_used: dict[str, int] = {}
        self._collectives = None
        self._program_cache: dict = {}
        self._prepare_cache: dict = {}
        self._service_identity_cache: dict = {}

    def add_group(self, works, release_ms, *, external_inputs=None, emitted_tokens=None):
        """Compile an EP cohort before releasing any of its collective events."""
        graphs = bind_cohort_traffic(self.case, {r: work.ops for r, work in works.items()})
        works = {r: replace(work, ops=graphs[r]) for r, work in works.items()}
        self._collectives = {}
        steps = {replica: self.add(
            work, replica, release_ms,
            external_inputs=(external_inputs or {}).get(replica),
            emitted_tokens=(emitted_tokens or {}).get(replica),
        ) for replica, work in works.items()}
        for key, participants in self._collectives.items():
            if len(participants) != len(works):
                raise ValueError(f"collective_participants_mismatch:{key}")
            if any(getattr(item.recipe, "algorithm", None) for item in participants):
                self._start_joint_collective(key, participants)
                continue
            dependencies = tuple(item.ready for item in participants)
            ready = self.engine.submit(f"group:{key}:ready", after=dependencies, owner="expert_group")
            # A cohort synchronizes participants; it must not replace their
            # phase ordering, invocation caps, buffers or calibrated services.
            completions, promises = [], []
            for item in participants:
                completions.append(self._start_program(
                    item.context, item.recipe, item.name, ready, owner="expert_group"))
                promises.append(item.completion)

            def release(_, events=tuple(promises)):
                for event in events:
                    self.engine.resolve(event)

            self.engine.submit(f"group:{key}", after=completions,
                               on_complete=release, owner="expert_group")
        self._collectives = None
        return steps

    def _start_joint_collective(self, key, participants):
        """Bind one physical cohort while retaining each replica's readiness."""
        first = participants[0]
        algorithm = getattr(first.recipe, "algorithm", None)
        plan_options = collective_options(first.recipe, first.context.op)
        path_model = getattr(first.recipe, "path_model", None)
        op = first.context.op
        declaration = lambda item: (
            dict(item.metadata).get("collective", item.family),
            dict(item.metadata).get("group_axis", "tp"),
            dict(item.metadata).get("source_stage", item.stage),
            dict(item.metadata).get("steps"),
            dict(item.metadata).get("peer_traffic"),
            dict(item.metadata).get("peer_volume_basis", "per_rank"),
        )
        for item in participants:
            if (getattr(item.recipe, "algorithm", None) != algorithm
                    or collective_options(item.recipe, item.context.op) != plan_options
                    or getattr(item.recipe, "path_model", None) != path_model
                    or declaration(item.context.op) != declaration(op)):
                raise ModelingError(f"joint_collective_declarations_differ:{key}")
            if any(value for role, value in item.context.native_work.items() if role != "link"):
                raise ModelingError(f"{algorithm}_collective_requires_transport_only_work")
        plan = self.topology.joint_plan(
            op, {item.replica: item.context.native_work.get("link", 0)
                 for item in participants}, algorithm=algorithm, **plan_options)
        plan.validate_path_model(path_model)
        source_stage = int(dict(op.metadata).get("source_stage", op.stage))
        ready_by_rank = {}
        for item in participants:
            timing = item.context.invocation_timing
            ready = item.ready
            if timing is not None and timing.startup_ms:
                ready = self.engine.submit(
                    item.name + ":device_startup", after=(ready,), latency=timing.startup_ms,
                    owner=f"replica:{item.replica}")
            ready_by_rank.update({rank: ready for rank in
                                  self.topology.ranks(item.replica, source_stage)})
        completions = plan.start_participants(
            self.engine, first.name + ":cohort", ready_by_rank, owner="expert_group")
        for item in participants:
            dependencies = [completions[rank] for rank in
                            self.topology.ranks(item.replica, source_stage)]
            timing = item.context.invocation_timing
            minimum = max(item.context.minimum_ms, timing.minimum_ms if timing else 0)
            if minimum:
                dependencies.append(self.engine.submit(
                    item.name + ":observation", after=(item.ready,),
                    latency=minimum, owner=f"replica:{item.replica}"))
            self.engine.submit(
                item.name + ":cohort_complete", after=dependencies,
                owner=f"replica:{item.replica}",
                on_complete=lambda _, promise=item.completion: self.engine.resolve(promise))
        return completions

    def _start_program(self, context, recipe, name, ready, *, owner):
        """Keep per-core programs resident only while their operator can run."""
        engine = self.engine
        complete = engine.future(name + ":program_complete")

        def begin(_):
            op = context.op
            live_rates = tuple(
                (role, context.rates.get(res_name))
                for role, res_name in sorted(context.resource_names.items())
                if res_name in context.rates
            )
            inv_t = context.invocation_timing
            prog_key = (
                owner, _freeze_key(recipe), op.family, op.recipe, op.phase, op.stage,
                op.dtype, op.layout, op.semantic_shape, op.local_shape, op.padded_shape,
                op.launch_shape, op.flops, op.matrix_flops, op.mte2_bytes,
                op.mte3_bytes, op.vector_ops, op.hbm_bytes, op.link_bytes,
                op.link_scope, op.source_boundary, op.returns,
                _op_meta_key(op), context.setup_ms, context.minimum_ms,
                (inv_t.startup_ms, inv_t.minimum_ms) if inv_t is not None else None,
                _freeze_key(context.joint_native_service),
                tuple(sorted(context.work.items())),
                live_rates,
                context.group_buffers,
            )
            cached_prog = self._program_cache.get(prog_key)
            if cached_prog is not None:
                program, mapped_recs, svc_calls = cached_prog
                if mapped_recs and context.record_mapping is not None:
                    for mrec in mapped_recs:
                        context.record_mapping(deepcopy(mrec))
                if svc_calls and context.record_service is not None:
                    for sargs, skwargs in svc_calls:
                        context.record_service(*deepcopy(sargs), **deepcopy(skwargs))
                finish = program.start(engine, name, ready, owner=owner)
                engine.submit(name + ":resolve", after=(finish,), owner=owner,
                              on_complete=lambda _: engine.resolve(complete))
                return

            mapped_list = []
            svc_list = []
            orig_rec = context.record_mapping
            orig_svc = context.record_service

            def cap_map(rec):
                mapped_list.append(deepcopy(rec))
                if orig_rec is not None:
                    orig_rec(rec)

            def cap_svc(*args, **kwargs):
                svc_list.append((deepcopy(args), deepcopy(kwargs)))
                if orig_svc is not None:
                    orig_svc(*args, **kwargs)

            run_ctx = replace(
                context,
                record_mapping=cap_map if orig_rec is not None else None,
                record_service=cap_svc if orig_svc is not None else None,
            )
            program = self.lower_program(run_ctx, recipe)
            if len(self._program_cache) >= 768:
                self._program_cache.pop(next(iter(self._program_cache)))
            self._program_cache[prog_key] = (program, tuple(mapped_list), tuple(svc_list))
            finish = program.start(engine, name, ready, owner=owner)
            engine.submit(name + ":resolve", after=(finish,), owner=owner,
                          on_complete=lambda _: engine.resolve(complete))

        # Host submission and graph dependencies are already declared. Expanding
        # every future layer's per-core pipelines here multiplies inactive state
        # by the device's core count without changing any physical work.
        engine.submit(name + ":begin", after=(ready,), owner=owner, on_complete=begin)
        return complete

    @staticmethod
    def lower_program(context, recipe):
        if context.joint_native_service is not None:
            from .program import KernelProgram, Stage
            return KernelProgram((Stage("joint_native_service", context.work),))
        from .invocation_binding import lower_invocation
        return lower_invocation(recipe, context)

    def prepare(self, op, replica=0):
        geom = self.stage_geometry[replica, op.stage]
        prep_key = (
            id(self.recipes), _freeze_key(geom), op.stage, op.family, op.recipe, op.phase,
            op.dtype, op.layout, op.semantic_shape, op.local_shape, op.padded_shape,
            op.launch_shape, op.flops, op.matrix_flops, op.mte2_bytes,
            op.mte3_bytes, op.vector_ops, op.hbm_bytes, op.link_bytes,
            op.link_scope, op.source_boundary, op.returns,
            _op_meta_key(op),
        )
        if prep_key in self._prepare_cache:
            cached_op, meta_diff, recipe = self._prepare_cache[prep_key]
            meta = op.metadata
            if meta_diff is not None:
                patched = _meta_patch(dict(op.metadata), meta_diff)
                meta = tuple(sorted(patched.items()))
            return replace(
                cached_op, op_id=op.op_id, layer=op.layer, deps=op.deps,
                requests=op.requests, stream=op.stream, notes=op.notes,
                metadata=meta,
            ), recipe
        res_op, recipe = self.recipes.prepare(op, geom)
        orig_meta = dict(op.metadata)
        res_meta = dict(res_op.metadata)
        meta_diff = _meta_diff(orig_meta, res_meta) if orig_meta != res_meta else None
        if len(self._prepare_cache) >= 512:
            self._prepare_cache.pop(next(iter(self._prepare_cache)))
        self._prepare_cache[prep_key] = (res_op, meta_diff, recipe)
        return res_op, recipe

    def _stage_context(self, key, dies):
        replica, stage = key
        names = self.stage_names[key]
        return RecipeContext(
            None, {}, names, self.resource_rates,
            dies[0].aic_count,
            pool_names={pool: f"{self.namespace}{pool}@r{replica}:s{stage}"
                        for pool in dict(dies[0].buffers)},
            resource_units={role: dies[0].units(role) for role in names},
            compute_peaks={dtype: min(d.peak(dtype) for d in dies)
                           for dtype, _ in dies[0].compute_peaks},
            pool_units={pool: dies[0].pool_units(pool)
                        for pool in dict(dies[0].buffers)},
            pool_capacities={pool: min(dict(d.buffers)[pool] for d in dies)
                             for pool in dict(dies[0].buffers)},
            record_mapping=self.cost.record_mapping,
            service_modes={mode: (role, min(dict(d.service_modes)[mode][1] for d in dies))
                           for mode, (role, _) in dies[0].service_modes},
            record_service=self.cost.record_service,
            work_units={**NATIVE_WORK_UNITS, **dict(dies[0].work_units)},
            store_queues={name: dict(queue) for name, queue in dies[0].store_queues
                          if all(dict(d.store_queues).get(name) == queue for d in dies)},
        )

    def _context(self, op: Any, replica: int, scope: str, *, recipe=None) -> RecipeContext:
        from .services import service_identity
        from .kernel_groups import declaration
        cost = self.cost.price(op, scope)
        dies = self.stage_dies[replica, op.stage]
        template = self.stage_contexts[replica, op.stage]
        peaks = template.compute_peaks
        additional = {dtype for dtype, _ in op.matrix_flops} - peaks.keys()
        if additional:
            peaks = {**peaks, **{dtype: min(d.peak(dtype) for d in dies) for dtype in additional}}
        meta = dict(op.metadata)
        sid_key = (
            replica, op.stage, dies[0].rank, dies[0].compute_units,
            _freeze_key(dict(dies[0].resource_units)),
            op.family, op.recipe, op.dtype, op.layout, scope,
            meta.get("kernel"), meta.get("kernel_name"),
            _freeze_key(meta.get("implementation")),
            _freeze_key(meta.get("service_axes")),
            bool(meta.get("hccl_selection")),
            _freeze_key(meta),
        )
        sid = self._service_identity_cache.get(sid_key)
        if sid is None:
            sid = service_identity(self.case, op, scope, device=dies[0])
            if len(self._service_identity_cache) >= 1024:
                self._service_identity_cache.pop(next(iter(self._service_identity_cache)))
            self._service_identity_cache[sid_key] = sid
        context = replace(template, op=op, native_work=cost.work, compute_peaks=peaks,
            service_identity=sid,
            service_boundary="resource_phase" if declaration(op) else "device_kernel")
        demands, frames, setup = {}, {}, 0.0
        if "peer_traffic" in dict(op.metadata) and not cost.work.get("link", 0):
            # A zero sender still participates and receives. Validate its row
            # even though CostModel omits zero work from the demand mapping.
            self.topology.accounting(op, replica, volume=cost.work.get("link", 0))
        for resource, amount in cost.work.items():
            if resource == "link":
                routed = self.topology.accounting(op, replica, volume=amount)
                demands.update(routed.wire_by_resource)
                frames.update(routed.transaction_frames)
                setup = routed.setup_ms
                continue
            if resource == "cube":
                portions = op.matrix_flops or ((op.dtype or self.case.model.compute_dtype, amount),)
                bound = context.bind(Work(matrix=dict(portions)))
            else:
                bound = context.bind(Work(resources={resource: amount}))
            for name, value in bound.items():
                demands[name] = demands.get(name, 0) + value
        measured = cost.timing.duration_ms if cost.timing is not None else 0
        physical = max((value / self.engine.resources[key].rate for key, value in demands.items()), default=0)
        # Context binding also serves independently constructed source phases.
        # Invocation callers pass their selected recipe; binding native work
        # must not require an unrelated source-dispatch registration.
        algorithm = getattr(recipe, "algorithm", "ring")
        context = replace(context, work=demands, minimum_ms=measured, setup_ms=setup,
                          transaction_frames=frames,
                          collective_plan=lambda: self.topology.joint_plan(
                              op, {replica: cost.work.get("link", 0)}, algorithm=algorithm,
                              **collective_options(recipe, op)))
        response = self.regimes.resolve(context, self.cost.record_regime)
        if response is not None and response.execution is not None:
            if getattr(recipe, "algorithm", None):
                raise ModelingError("joint_native_service_requires_single_local_kernel")
            demand = response.bind_joint(context)
            self.cost.record_joint_native(response.execution, cost)
            declaration = {**response.execution,
                           "reference_program": type(recipe).__name__,
                           "reference_primitive_ms": measured if cost.timing is not None else None}
            return replace(context, work=demand, minimum_ms=0, setup_ms=0,
                           invocation_timing=None, joint_native_service=declaration)
        self.conflicts += int(bool(measured and measured < physical))
        selected = self.invocations.select(context, recipe, devices=dies)
        if selected is not None:
            timing, record = selected
            self.cost.record_invocation(record)
            context = replace(context, invocation_timing=timing)
        if response is not None:
            context = replace(context, minimum_ms=max(context.minimum_ms, response.duration_ms))
        return context

    def add(self, work: IterationWork, replica: int, release_ms: float, *,
            external_inputs=None, emitted_tokens=None) -> ScheduledIteration:
        if self._collectives is None:
            work = replace(work, ops=bind_cohort_traffic(
                self.case, {replica: work.ops})[replica])
        engine = self.engine
        prefix = f"{self.namespace}r{replica}:i{work.plan.iteration}"
        ids: dict[str, Event] = dict(external_inputs or {})
        commits: dict[str, Event] = {}
        scope = work.replay_scope or ("full" if work.graph_bucket is not None else "none")
        if self.include_submission and self.case.runtime.async_scheduling and replica in self.prepared:
            schedule = self.prepared.pop(replica)
        else:
            scheduler_ms = self.submission.scheduler_ms if self.include_submission else 0
            schedule = engine.submit(prefix + ":schedule", {f"{self.namespace}scheduler@r{replica}": scheduler_ms},
                                     release=release_ms, owner=f"replica:{replica}")
        self.schedules[replica] = schedule
        prepared, input_ready = self.serving_io.input(self, prefix, replica, schedule, work)
        graph_launch: dict[int, Event] = {}
        physical_ops = self.recipes.expand(work.ops)
        from .kernel_groups import KernelGroups, declaration
        kernel_groups = KernelGroups(self, physical_ops, replica, prefix)
        for op in physical_ops:
            original = op
            op, recipe = self.prepare(op, replica)
            stage_key = replica, op.stage
            name = prefix + ":" + op.op_id
            names = self.stage_names[stage_key]
            deps = [input_ready, *(ids[key] for key in op.deps)]
            deps.extend(kernel_groups.entry_dependencies(op, ids))
            stream_key = replica, op.stage, op.stream or "main"
            stream_tail = kernel_groups.stream_tail(op, stream_key)
            if stream_tail is not None:
                deps.append(stream_tail)
            context = self._context(op, replica, scope, recipe=recipe)
            context = replace(context, group_buffers=kernel_groups.buffer_scope(op))
            owner = f"replica:{replica}"
            captured = scope == "full" or (
                scope == "piecewise" and bool(dict(op.metadata).get("graph_captured"))
            )
            capture_key = op.stage, dict(op.metadata).get("graph_segment", 0) if scope == "piecewise" else 0
            kernel_group = declaration(op)
            submit = (bool(kernel_group.get("entry")) if kernel_group else
                      bool(context.work or getattr(recipe, "algorithm", None)))
            if submit and self.include_submission:
                if captured and capture_key in graph_launch:
                    launch = graph_launch[capture_key]
                else:
                    dies = self.stage_dies[stage_key]
                    duration = self.submission.launch_ms(dies, captured=captured)
                    host_deps = (prepared,) + ((self.hosts[stage_key],) if stage_key in self.hosts else ())
                    launch = engine.submit(name + ":submit", {names["host"]: duration},
                                           after=host_deps, owner=owner)
                    self.hosts[stage_key] = launch
                    self.submission_counts[
                        "graph_replays" if captured else "eager_invocations"] += 1
                    if captured:
                        graph_launch[capture_key] = launch
                deps.append(launch)
            ready = engine.submit(name + ":ready", after=tuple(deps) or (schedule,), owner=owner)
            ready = kernel_groups.phase_ready(op, ready, name)
            recipe_name = op.recipe or type(recipe).__name__
            self.recipes_used[recipe_name] = self.recipes_used.get(recipe_name, 0) + 1
            cross_dp = (dict(op.metadata).get("group_axis") == "ep"
                        and self.case.runtime.parallel.ep_spans_dp)
            if cross_dp and getattr(recipe, "algorithm", None) and self._collectives is None:
                raise ModelingError("joint_collective_requires_complete_cohort")
            grouped = self._collectives is not None and cross_dp
            if grouped:
                complete = engine.future(name + ":collective")
                self._collectives.setdefault(op.op_id, []).append(CollectiveParticipant(
                    replica, ready, context, recipe, name, complete))
            else:
                complete = self._start_program(context, recipe, name, ready, owner=owner)
            complete = kernel_groups.finish(op, context, recipe, complete, stream_key)
            ids[op.op_id] = complete
            if self.on_operator is not None:
                self.on_operator(original, replica, "JointNativeService" if context.joint_native_service
                                 is not None else type(recipe).__name__, ready, complete)
            if op.family == "token_commit":
                commits.update((request, complete) for request in op.requests)
            for key in self.work:
                self.work[key] += float(getattr(op, key))
        device_complete = engine.submit(prefix + ":device_complete",
                                        after=tuple(ids[op.op_id] for op in physical_ops) or (input_ready,),
                                        owner=f"replica:{replica}")
        completion = engine.submit(prefix + ":complete", after=(device_complete,),
                                   latency=self.submission.completion_ms if self.include_submission else 0,
                                   owner=f"replica:{replica}")
        for view in work.plan.requests:
            commits.setdefault(view.request_id, completion)
        host_results, clients, host_complete = self.serving_io.output(
            self, prefix, replica, work, commits, completion,
            emitted_tokens if emitted_tokens is not None else {
                view.request_id: int(view.draft_advanced_tokens or 1)
                for view in work.plan.requests if view.samples})
        if self.serving_io.enabled:
            completion = engine.submit(prefix + ":step_ready", after=(completion, host_complete),
                                       owner=f"replica:{replica}")
        step = ScheduledIteration(replica, work.plan.iteration, completion, release_ms,
                                  work.mode, work.valid_tokens, work.padded_tokens,
                                  work.graph_bucket, scope, len(physical_ops), commits,
                                  device_complete, host_results, clients, work.implementation_selections)
        engine.submit(prefix + ":step_metrics", after=(completion,),
                      on_complete=lambda _: self.step_summary.record(step))
        return step

    def prepare_next(self, replica):
        if self.include_submission and self.case.runtime.async_scheduling:
            self.prepared[replica] = self.engine.submit(
                f"{self.namespace}r{replica}:prepare_next",
                {f"{self.namespace}scheduler@r{replica}": self.submission.scheduler_ms},
                after=(self.schedules[replica],), owner=f"replica:{replica}")

    def submission_summary(self) -> dict:
        return {
            "included": self.include_submission,
            **self.submission_counts,
            "host_service_ms": sum(
                self.engine.serviced.get(names["host"], 0)
                for names in self.stage_names.values()),
            "stages": [
                {"replica": replica, "stage": stage,
                 "kernel_launch_ms": self.submission.launch_ms(dies),
                 "graph_replay_ms": self.submission.launch_ms(dies, captured=True),
                 "kernel_parameter": (f"{self.submission_source}.kernel_launch_ms"
                                      if self.submission.kernel_launch_ms is not None
                                      else "machine.device.launch_us"),
                 "replay_parameter": (f"{self.submission_source}.graph_replay_ms"
                                      if self.submission.graph_replay_ms is not None
                                      else "kernel_launch_parameter")}
                for (replica, stage), dies in self.stage_dies.items()
            ],
            "boundary": "host enqueue service; may overlap ready device work",
            "zero_parameter": "excludes undeclared host cost; not measured zero overhead",
        }

    def summary(self) -> dict:
        return {
            **self.engine.summary(), "work": dict(self.work),
            **({"work_basis": "logical_and_analytical_reference; executed_service_is_in_serviced"}
               if self.cost.joint_native else {}),
            "recipes": dict(self.recipes_used),
            "submission": self.submission_summary(),
            "serving_io": self.serving_io.summary(),
            "step_summary": self.step_summary.to_dict(),
            "rank_model": "representative_local_shard_per_stage",
        }
