"""Plan lowering and Engine evaluation for the Day-0 graph optimizer.

[Role] Lower a physical plan to the one causal Engine with real UB/prefetch
  pools and explicit event edges, and expose capture/SOL audits over the
  effective execution DAG.
[Invariants] Shared HBM, MTE/on-chip, Vector, Cube and host submission is
  conserved. A kernel holds its per-owner UB reservation until retirement; a
  prefetch holds a shared pool reservation until both its copy and consumer
  retire, and a destination buffer is acquired before its first write.
  Foreground service outranks the prefetch priority class in the Engine's fair
  arbitration. Cross-stream crossings pay explicit record/wait latency. Capture
  savings exist only when a real replay counterfactual ran.
[Timeouts & Locks] Finite virtual events; no threads, sleeps, I/O or locks.
[Side-Effect Policy] Builds private Engine instances and returns records.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import heapq
import math

from mindie_sim.prediction.engine import Allocation, Engine, Resource

from .engine_plan_phases import (
    conserved_fusion_work, phase_acquisitions, reconciled_phase_work,
)
from .fusion_feasibility import local_nbytes, single_op_ub
from .fusion_graph import OpNode, TypedGraph
from .fusion_recipe import FusionCandidate
from .plan_model import EngineEvaluation, HardwareProfile, PlanNode
from .weight_prefetch import PrefetchDecision

def _work(totals: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((k, float(v)) for k, v in totals.items() if abs(v) > 1e-12))


def _baseline_access(graph: TypedGraph, op: OpNode,
                     hardware: HardwareProfile) -> tuple[float, float]:
    reads = sum(float(local_nbytes(graph.values[v], hardware.tensor_parallel))
                for v in op.inputs)
    writes = sum(float(local_nbytes(graph.values[v], hardware.tensor_parallel))
                 for v in op.outputs if not graph.values[v].alias_of)
    return reads, writes


_DEGREE_STEP = 1


def _topological(nodes: tuple[PlanNode, ...],
                 external: tuple[str, ...] = ()) -> tuple[PlanNode, ...]:
    by_id = {n.node_id: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("plan_duplicate_node")
    missing = {dep for node in nodes for dep in node.deps
               if dep not in by_id and dep not in external}
    if missing:
        raise ValueError("plan_dependency_unavailable:" + ",".join(sorted(missing)))
    in_degree: dict[str, int] = {}
    children: dict[str, list[PlanNode]] = {n.node_id: [] for n in nodes}
    for node in nodes:
        deps = {d for d in node.deps if d in by_id}
        in_degree[node.node_id] = len(deps)
        for dep in deps:
            children[dep].append(node)
    ordered: list[PlanNode] = []
    ready = deque(n for n in nodes if not in_degree[n.node_id])
    while ready:
        node = ready.popleft()
        ordered.append(node)
        for other in children[node.node_id]:
            in_degree[other.node_id] -= _DEGREE_STEP
            if not in_degree[other.node_id]:
                ready.append(other)
    if len(ordered) != len(nodes):
        raise ValueError("plan_dependency_cycle")
    return tuple(ordered)


def _stream_predecessor(order: tuple[PlanNode, ...]) -> dict[str, str]:
    previous: dict[str, str] = {}
    tails: dict[str, str] = {}
    for node in order:
        if node.kind != "kernel":
            continue
        if node.stream in tails:
            previous[node.node_id] = tails[node.stream]
        tails[node.stream] = node.node_id
    return previous


def _stable_topological(nodes: tuple[PlanNode, ...],
                        external: tuple[str, ...] = ()) -> tuple[PlanNode, ...]:
    """Topological order stable in the caller's declared tuple order.

    The detailed ``Execution`` submission path walks physical operators in
    declaration order and appends same-stream predecessors as it goes.  Its
    effective DAG therefore preserves that order among ready nodes; a FIFO
    frontier would export ``aa -> cc -> bb`` for ``aa, bb, cc``.
    """
    by_id = {n.node_id: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("plan_duplicate_node")
    missing = {dep for node in nodes for dep in node.deps
               if dep not in by_id and dep not in external}
    if missing:
        raise ValueError("plan_dependency_unavailable:" + ",".join(sorted(missing)))
    rank = {n.node_id: idx for idx, n in enumerate(nodes)}
    in_degree: dict[str, int] = {}
    children: dict[str, list[PlanNode]] = {n.node_id: [] for n in nodes}
    for node in nodes:
        deps = {d for d in node.deps if d in by_id}
        in_degree[node.node_id] = len(deps)
        for dep in deps:
            children[dep].append(node)
    heap = [(rank[n.node_id], n) for n in nodes if not in_degree[n.node_id]]
    heapq.heapify(heap)
    ordered: list[PlanNode] = []
    while heap:
        _, node = heapq.heappop(heap)
        ordered.append(node)
        for other in children[node.node_id]:
            in_degree[other.node_id] -= _DEGREE_STEP
            if not in_degree[other.node_id]:
                heapq.heappush(heap, (rank[other.node_id], other))
    if len(ordered) != len(nodes):
        raise ValueError("plan_dependency_cycle")
    return tuple(ordered)


def _pools(hardware: HardwareProfile) -> tuple[dict[str, float], dict[str, int]]:
    """One physical L1 pool shared by Cube holders and prefetch grants.

    ``prefetch_capacity_bytes`` is the declared total over all L1 units, so the
    Engine's per-unit capacity is that total divided by the unit count.  A
    prefetch and an existing Cube/group L1 holder therefore contend for the
    same capacity instead of living in separate pools.
    """
    # The declared usable per-owner budget caps the physical UB; admitting
    # reservations against the raw physical size would execute past the
    # declared software budget.
    ub_capacity = float(hardware.ub_bytes_per_aiv)
    if hardware.usable_ub_bytes is not None:
        ub_capacity = min(ub_capacity, float(hardware.usable_ub))
    pools = {"ub": ub_capacity}
    units = {"ub": int(hardware.ub_units)}
    prefetch_units = max(1, int(hardware.prefetch_units))
    if hardware.prefetch_capacity_bytes > 0:
        pools["l1"] = float(hardware.prefetch_capacity_bytes) / prefetch_units
        units["l1"] = prefetch_units
    cube = hardware.cube_capacities
    if cube is not None and cube.complete:
        for name in ("l1", "l0a", "l0b", "l0c"):
            if name == "l1" and "l1" in pools:
                continue
            pools[name] = float(getattr(cube, name))
            units[name] = 1
    return pools, units


def _resources(hardware: HardwareProfile, totals: dict[str, float]) -> list[Resource]:
    resources = [
        Resource("hbm", hardware.hbm_bytes_per_ms, "bytes"),
        Resource("vector", hardware.vector_ops_per_ms, "ops"),
        Resource("cube", hardware.cube_flops_per_ms, "flops"),
        Resource("mte1", hardware.onchip_bytes_per_ms, "bytes"),
        Resource("host", 1.0, "ms", policy="fifo"),
    ]
    for name, rate in (("mte2", hardware.mte2_bytes_per_ms),
                       ("mte3", hardware.mte3_bytes_per_ms),
                       ("fixpipe", hardware.onchip_bytes_per_ms)):
        if totals.get(name):
            if rate <= 0:
                raise ValueError(f"plan_resource_rate_missing:{name}")
            resources.append(Resource(name, rate, "bytes"))
    for name in sorted(totals):
        if name in {"hbm", "vector", "cube", "mte1", "mte2", "mte3", "fixpipe",
                    "host", "event_record", "event_wait"}:
            continue
        raise ValueError(f"plan_resource_undeclared:{name}")
    return resources


def _ub_allocations(node: PlanNode) -> tuple[Allocation, ...]:
    found = []
    for index, (name, pool, size, units) in enumerate(node.ub_allocations):
        if size > 0:
            found.append(Allocation(f"{node.node_id}:ub:{index}:{name}", pool,
                                    float(size), tuple(units)))
    if node.prefetch_alloc is not None:
        name, pool, size, units = node.prefetch_alloc
        found.append(Allocation(f"{node.node_id}:reserve:{name}", pool, float(size),
                                tuple(units)))
    return tuple(found)


def _phase_allocations(node: PlanNode) -> dict[str, tuple[Allocation, ...]]:
    found: dict[str, tuple[Allocation, ...]] = {}
    for phase, items in node.phase_acquisitions:
        allocations = []
        for index, (name, pool, size, units) in enumerate(items):
            if size > 0:
                allocations.append(Allocation(
                    f"{node.node_id}:phase:{phase}:{index}:{name}", pool,
                    float(size), tuple(units)))
        found[phase] = tuple(allocations)
    return found


def _free_allocations(node: PlanNode) -> tuple[Allocation, ...]:
    if node.free_alloc is None:
        return ()
    name, pool, size, units = node.free_alloc
    return (Allocation(name, pool, float(size), tuple(units)),)


def evaluate_plan(nodes: tuple[PlanNode, ...], hardware: HardwareProfile, *,
                  capture_mode: str | None = None,
                  measure_prefetch: bool = True) -> EngineEvaluation:
    """Lower a plan to public Engine tasks and run the one causal Engine."""
    try:
        order = _topological(nodes)
        totals: dict[str, float] = {}
        for node in order:
            for resource, amount in node.work:
                if resource in {"event_record", "event_wait"}:
                    continue
                totals[resource] = totals.get(resource, 0.0) + float(amount)
        pools, pool_units = _pools(hardware)
        for node in order:
            # Every concurrent reservation of one owner is summed per physical
            # unit.  A single allocation below capacity but a sum above it on
            # the same unit is still infeasible; reservations on distinct units
            # are independent owners and must not be added together.
            demands: dict[tuple[str, int], int] = {}

            def reserve(pool: str, size: int, units) -> None:
                for unit in (tuple(units) or (0,)):
                    key = (pool, int(unit))
                    demands[key] = demands.get(key, 0) + int(size)

            for _name, pool, size, units in node.ub_allocations:
                reserve(pool, size, units)
            if node.prefetch_alloc is not None:
                _name, pool, size, units = node.prefetch_alloc
                reserve(pool, size, units)
            for _phase, items in node.phase_acquisitions:
                for _name, pool, size, units in items:
                    reserve(pool, size, units)
            for (pool, _unit), total in sorted(demands.items()):
                capacity = pools.get(pool)
                if capacity is not None and float(total) > float(capacity):
                    raise ValueError(
                        "single_op_ub_capacity_exceeded:"
                        f"{node.node_id}:{pool}:{int(total)}>{int(capacity)}")
        engine = Engine(_resources(hardware, totals), pools=pools,
                        pool_units=pool_units)
        entry = engine.submit("entry")
        launch: dict[str, object] = {}
        launches = 0
        if capture_mode == "full":
            capture = engine.submit("capture:replay", {"host": hardware.launch_ms},
                                    after=(entry,), owner="optimizer")
            launches = 1 if any(n.kind == "kernel" for n in order) else 0
            launch = {n.node_id: capture for n in order if n.kind == "kernel"}
        else:
            previous = entry
            for node in order:
                if node.kind != "kernel":
                    continue
                launch[node.node_id] = engine.submit(
                    f"launch:{node.node_id}", {"host": hardware.launch_ms},
                    after=(previous,), owner="optimizer")
                previous = launch[node.node_id]
                launches += 1
        stream_prev = _stream_predecessor(order)
        events: dict[str, object] = {}
        started: dict[str, float] = {}
        completed: dict[str, float] = {}
        gate_events: list[tuple[object, tuple[object, ...]]] = []
        first_events: dict[str, object] = {}
        by_id = {n.node_id: n for n in order}
        for node in order:
            predecessors = []
            if node.kind == "kernel":
                predecessors.append(launch[node.node_id])
            elif node.kind == "prefetch" and node.prefetch_for in launch:
                predecessors.append(launch[node.prefetch_for])
            for dep in node.deps:
                if dep in events:
                    predecessors.append(events[dep])
            previous_stream = stream_prev.get(node.node_id)
            if previous_stream and previous_stream not in node.deps:
                predecessors.append(events[previous_stream])
            for dep in node.deps:
                parent = by_id.get(dep)
                if parent is None or parent.stream == node.stream:
                    continue
                record = engine.submit(
                    f"event:record:{dep}->{node.node_id}", {},
                    after=(events[dep],), latency=hardware.record_ms, owner="optimizer")
                wait = engine.submit(
                    f"event:wait:{dep}->{node.node_id}", {},
                    after=(record,), latency=hardware.wait_ms, owner="optimizer")
                predecessors.append(wait)
            acquisitions = _ub_allocations(node)
            phase_allocations = _phase_allocations(node)
            latency = hardware.device_min_ms
            ready = tuple(dict.fromkeys(predecessors))
            if node.phase_work:
                phase_events = []
                phase_ready = ready
                acquired: list[Allocation] = []
                first_events[node.node_id] = None
                for phase_index, (phase, phase_demand) in enumerate(node.phase_work):
                    acquire = list(acquisitions if phase_index == 0 else ())
                    acquire.extend(phase_allocations.get(phase, ()))
                    acquired.extend(acquire)
                    phase_event = engine.submit(
                        f"body:{node.node_id}:{phase}", dict(phase_demand),
                        after=phase_ready,
                        latency=latency if phase_index == 0 else 0.0,
                        caps=dict(node.caps), priority=node.priority,
                        owner="optimizer", acquire=tuple(acquire))
                    phase_events.append(phase_event)
                    if first_events.get(node.node_id) is None:
                        first_events[node.node_id] = phase_event
                    phase_ready = (phase_event,)
                if acquired:
                    event = engine.submit(
                        f"body:{node.node_id}:release", after=(phase_events[-1],),
                        free=tuple(dict.fromkeys(acquired)), owner="optimizer")
                else:
                    event = phase_events[-1]
            else:
                event = engine.submit(
                    f"body:{node.node_id}", node.work_dict(),
                    after=ready, latency=latency,
                    caps=dict(node.caps), priority=node.priority, owner="optimizer",
                    acquire=acquisitions,
                    free=(() if node.kind == "prefetch"
                          else tuple(acquisitions) + _free_allocations(node)))
                first_events[node.node_id] = event
            events[node.node_id] = event
            if node.prefetch_for and node.kind == "prefetch":
                gate_events.append((node, tuple(predecessors)))
        terminal = engine.submit("plan:complete", after=tuple(events.values()))
        prefetch_at_gate: dict[str, float] = {}
        if measure_prefetch and gate_events:
            gates = []
            for prefetch_node, predecessors in gate_events:
                consumer = prefetch_node.prefetch_for
                target = by_id.get(consumer)
                ready = tuple(dict.fromkeys(
                    [launch[consumer]] if target is not None and consumer in launch else ()))
                if target is not None:
                    ready = ready + tuple(events[d] for d in target.deps if d in events)
                gate = engine.submit(f"gate:{prefetch_node.node_id}", after=ready)
                gates.append((prefetch_node, gate))
            # Measure each consumer at its own start gate, never after all
            # gates advanced to the latest consumer.
            pending = {node.node_id: (node, gate) for node, gate in gates}
            while pending:
                engine.advance(until=tuple(gate for _, gate in pending.values()))
                done = [(node_id, item) for node_id, item in pending.items()
                        if item[1].done]
                if not done:
                    break
                for node_id, (prefetch_node, gate) in done:
                    event = events[prefetch_node.node_id]
                    target_bytes = prefetch_node.work_dict().get("hbm", 0.0)
                    at_gate = float(gate.time if gate.time is not None else engine.now)
                    if event.done and event.time <= at_gate + 1e-12:
                        prefetch_at_gate[node_id] = target_bytes
                    else:
                        prefetch_at_gate[node_id] = float(
                            engine.service_of(event).get("hbm", 0.0))
                    del pending[node_id]
        engine.advance(until=(terminal,), first=False)
        for node in order:
            event = events[node.node_id]
            completed[node.node_id] = event.time
            first = first_events.get(node.node_id)
            started[node.node_id] = (first.started if first is not None
                                     and first.started is not None else 0.0)
            if node.node_id not in prefetch_at_gate and node.kind == "prefetch":
                prefetch_at_gate[node.node_id] = float(
                    node.work_dict().get("hbm", 0.0) if event.done else
                    engine.service_of(event).get("hbm", 0.0))
        for node_id, event in launch.items():
            started[f"launch:{node_id}"] = (event.started
                                            if event.started is not None else 0.0)
        return EngineEvaluation(
            "completed", terminal.time, launches, totals.get("hbm", 0.0),
            totals.get("cube", 0.0), totals.get("vector", 0.0),
            tuple(sorted(started.items())), tuple(sorted(completed.items())),
            "", tuple(sorted(prefetch_at_gate.items())), None,
            _work(totals))
    except TimeoutError:
        raise
    except Exception as exc:  # Engine contract errors are recorded, never scored as zero
        return EngineEvaluation("failed", math.inf, 0, 0.0, 0.0, 0.0,
                                reason=f"{type(exc).__name__}:{exc}")


def baseline_plan(graph: TypedGraph,
                  hardware: HardwareProfile) -> tuple[PlanNode, ...]:
    nodes = []
    for op in graph.ops.values():
        declared = dict(op.work)
        reads, writes = _baseline_access(graph, op, hardware)
        work = dict(declared)
        work["hbm"] = float(declared.get("hbm", reads + writes))
        # One row is the irreducible minimum tile: a single unfused operator
        # can always execute row-by-row, so only that footprint decides
        # feasibility.  (Fused regions choose their own tile rows.)
        ub = single_op_ub(graph, op.op_id, tile_rows=1,
                          capacity_per_owner=hardware.ub_bytes_per_aiv,
                          usable_ub_bytes=hardware.usable_ub, owner=op.owner,
                          allow_cube=op.resource == "cube",
                          cube_resources=hardware.cube_capacities)
        nodes.append(PlanNode(
            node_id=op.op_id, work=_work(work), deps=graph.predecessors(op.op_id),
            stream=op.stream or "main", kind="kernel", args=(op.op_id,),
            hbm_read=reads, hbm_write=writes, collective=op.collective,
            state_effect=op.state_effect, capture_boundary=op.capture_boundary,
            phase=op.phase,
            capture_ineligible=bool(op.param("capture_ineligible")),
            ub_allocations=tuple((a.name, a.pool, a.total_bytes, a.units)
                                 for a in ub),
            numerical_policy=op.numerical_policy, pointer_stable=False))
    return tuple(nodes)


def _traffic_split(candidate: FusionCandidate) -> tuple[float, float]:
    reads = sum(v.candidate_read_bytes for v in candidate.analysis.traffic.values)
    writes = sum(v.candidate_write_bytes + v.new_spill_bytes
                 for v in candidate.analysis.traffic.values)
    return float(reads), float(writes)


def apply_fusion(plan: tuple[PlanNode, ...],
                 candidate: FusionCandidate) -> tuple[PlanNode, ...]:
    members = set(candidate.member_ops)
    present = {n.node_id for n in plan}
    if not members <= present:
        raise ValueError("fusion_member_no_longer_standalone")
    member_nodes = tuple(n for n in plan if n.node_id in members)
    external_deps = tuple(dict.fromkeys(
        dep for node in member_nodes for dep in node.deps if dep not in members))
    index = min(i for i, n in enumerate(plan) if n.node_id in members)
    reads, writes = _traffic_split(candidate)
    conserved = conserved_fusion_work(member_nodes, candidate)
    executed_phases = reconciled_phase_work(candidate, conserved)
    ub = tuple((a.name, a.pool, a.total_bytes, a.units)
               for a in candidate.analysis.worksheet.allocations)
    acquisition_groups = phase_acquisitions(candidate)
    fused = PlanNode(
        node_id="fused:" + "+".join(candidate.member_ops),
        work=conserved,
        deps=external_deps, stream=candidate.stream_name, kind="kernel",
        args=candidate.member_ops, recipe_id=candidate.recipe_id,
        hbm_read=reads, hbm_write=writes, retained=candidate.retained_outputs,
        ub_allocations=ub, numerical_policy=candidate.numerical_policy,
        pointer_stable=all(n.pointer_stable for n in member_nodes),
        capture_dynamic=any(n.capture_dynamic for n in member_nodes),
        capture_ineligible=any(n.capture_ineligible for n in member_nodes),
        capture_boundary=next((n.capture_boundary for n in member_nodes
                               if n.capture_boundary), ""),
        collective=any(n.collective for n in member_nodes),
        state_effect=next((n.state_effect for n in member_nodes
                           if n.state_effect), ""),
        group_buffers=candidate.group_buffers,
        phase_work=executed_phases,
        phase_acquisitions=acquisition_groups)
    fused_id = fused.node_id
    remaining = [replace(n, deps=tuple(dict.fromkeys(
        fused_id if dep in members else dep for dep in n.deps)))
        for n in plan if n.node_id not in members]
    remaining.insert(index, fused)
    return tuple(remaining)


def apply_stream_move(plan: tuple[PlanNode, ...], node_ids: tuple[str, ...],
                      stream: str = "aux", *, hardware: HardwareProfile) -> tuple[PlanNode, ...]:
    if hardware.max_streams < 2:
        raise ValueError("stream_capability_insufficient")
    chosen = set(node_ids)
    for node in plan:
        if node.node_id in chosen and (node.collective or node.state_effect
                                       or node.capture_boundary):
            raise ValueError(f"stream_move_preserves_barrier:{node.node_id}")
    return tuple(replace(n, stream=stream) if n.node_id in chosen else n for n in plan)


def apply_prefetch(plan: tuple[PlanNode, ...], decision: PrefetchDecision,
                   cube_node: PlanNode, hardware: HardwareProfile) -> tuple[PlanNode, ...]:
    """Insert a low-priority prefetch and a release that frees its reservation."""
    if not decision.allowed or decision.bytes_prefetched <= 0:
        raise ValueError("prefetch_decision_not_accepted")
    if hardware.prefetch_capacity_bytes < decision.bytes_prefetched:
        raise ValueError("prefetch_exceeds_shared_pool")
    prefetch_id = f"prefetch_W_{decision.op_id}"
    release_id = f"release_W_{decision.op_id}"
    units = tuple(range(max(1, int(hardware.prefetch_units))))
    per_unit = float(decision.bytes_prefetched) / len(units)
    prefetch = PlanNode(
        node_id=prefetch_id, work=_work({"hbm": float(decision.bytes_prefetched)}),
        deps=(), stream="prefetch", kind="prefetch", prefetch_for=cube_node.node_id,
        hbm_read=float(decision.bytes_prefetched),
        priority=max(hardware.prefetch_priority, cube_node.priority + 1),
        caps=(("hbm", float(decision.spare_bytes_per_ms)),),
        prefetch_alloc=(prefetch_id, "l1", per_unit, units),
        numerical_policy="exact")
    release = PlanNode(
        node_id=release_id, work=(), deps=(cube_node.node_id, prefetch_id),
        stream="prefetch", kind="release", prefetch_for=cube_node.node_id,
        priority=prefetch.priority,
        free_alloc=(f"{prefetch_id}:reserve:{prefetch_id}", "l1",
                    per_unit, units))
    result = []
    for node in plan:
        result.append(node)
        if node.node_id == cube_node.node_id:
            result.extend((prefetch, release))
    return tuple(result)


def settle_prefetch(plan: tuple[PlanNode, ...], prefetch_id: str,
                    completed_bytes: float) -> tuple[PlanNode, ...]:
    """Credit only measured completed bytes and add the consumer join edge."""
    prefetch = next((n for n in plan if n.node_id == prefetch_id), None)
    if prefetch is None:
        return plan
    done = int(max(0.0, min(float(completed_bytes), float(prefetch.hbm_read))))
    consumer = prefetch.prefetch_for
    if done <= 0:
        return tuple(n for n in plan if n.node_id != prefetch_id
                     and not (n.kind == "release" and n.prefetch_for == consumer
                              and prefetch_id in n.deps))
    result = []
    for node in plan:
        if node.node_id == prefetch_id:
            result.append(replace(node, work=_work({"hbm": float(done)}),
                                  hbm_read=float(done)))
        elif node.kind == "release" and node.prefetch_for == consumer:
            result.append(node)
        elif node.node_id == consumer:
            work = node.work_dict()
            if done > work.get("hbm", 0.0) + 1e-9:
                raise ValueError("prefetch_exceeds_foreground_hbm_work")
            work["hbm"] = max(0.0, work.get("hbm", 0.0) - done)
            work["mte1"] = work.get("mte1", 0.0) + done
            phases = []
            remaining = float(done)
            for phase, phase_demand in node.phase_work:
                demand = dict(phase_demand)
                if remaining and demand.get("hbm"):
                    cut = min(remaining, demand["hbm"])
                    demand["hbm"] -= cut
                    demand["mte1"] = demand.get("mte1", 0.0) + cut
                    remaining -= cut
                phases.append((phase, demand))
            result.append(replace(node, work=_work(work), phase_work=tuple(phases),
                                  hbm_read=max(0.0, node.hbm_read - done),
                                  deps=tuple(dict.fromkeys((*node.deps, prefetch_id)))))
        else:
            result.append(node)
    return tuple(result)


def window_foreground_hbm(plan: tuple[PlanNode, ...], evaluation: EngineEvaluation,
                          start_ms: float, end_ms: float) -> float:
    """Time-local foreground HBM bytes whose service overlaps the window.

    Whole-kernel sums before the window and after it are excluded; a partially
    overlapping kernel contributes only its overlapping service fraction.
    """
    if end_ms <= start_ms:
        return 0.0
    total = 0.0
    for node in plan:
        if node.kind != "kernel":
            continue
        amount = node.work_dict().get("hbm", 0.0)
        if amount <= 0:
            continue
        node_start = evaluation.start(node.node_id)
        node_end = evaluation.complete(node.node_id)
        if not (node_start < end_ms and node_end > start_ms):
            continue
        span = max(1e-12, node_end - node_start)
        overlap = max(0.0, min(node_end, end_ms) - max(node_start, start_ms))
        total += amount * overlap / span
    return total


def plan_has_members(plan: tuple[PlanNode, ...], members: tuple[str, ...]) -> bool:
    ids = {n.node_id for n in plan}
    return all(m in ids for m in members)


def audit_capture(plan: tuple[PlanNode, ...], hardware: HardwareProfile, *,
                  counterfactual: EngineEvaluation | None = None) -> dict:
    """Static eligibility plus a real replay counterfactual; never an estimate."""
    blockers = []
    for node in plan:
        if node.collective:
            blockers.append(f"collective:{node.node_id}")
        if node.state_effect in {"write", "read_write"}:
            blockers.append(f"state_write:{node.node_id}")
        if node.capture_boundary:
            blockers.append(f"capture_boundary:{node.node_id}")
        if node.kind == "kernel" and any(resource == "host"
                                         for resource, _ in node.work):
            blockers.append(f"host_barrier:{node.node_id}")
        if node.capture_dynamic:
            blockers.append(f"dynamic_shape:{node.node_id}")
        if node.capture_ineligible:
            blockers.append(f"capture_ineligible:{node.node_id}")
        if node.phase == "prefill":
            blockers.append(f"prefill_capture_unsupported:{node.node_id}")
        if node.kind == "prefetch" and not node.pointer_stable:
            blockers.append(f"prefetch_pointer_stability_unknown:{node.node_id}")
    if not hardware.capability_declared("aclgraph_capture"):
        blockers.append("capture_capability_undeclared")
    elif not hardware.capability("aclgraph_capture"):
        blockers.append("capture_capability_missing")
    if any(n.kind == "prefetch" for n in plan) \
            and not hardware.capability("aclgraph_prefetch"):
        blockers.append("prefetch_pointer_stability_unknown")
    if blockers:
        return {"status": "blocked", "mode": "aclgraph_full",
                "blockers": tuple(dict.fromkeys(blockers)),
                "host_bubble_saved_ms": None, "evaluated": False}
    if counterfactual is None:
        return {"status": "unknown", "mode": "aclgraph_full", "blockers": (),
                "host_bubble_saved_ms": None, "evaluated": False,
                "reason": "capture_counterfactual_not_run"}
    if counterfactual.status != "completed":
        return {"status": "unknown", "mode": "aclgraph_full",
                "blockers": ("capture_counterfactual_failed",),
                "host_bubble_saved_ms": None, "evaluated": False,
                "reason": counterfactual.reason}
    saved = 0.0
    return {"status": "eligible", "mode": "aclgraph_full", "blockers": (),
            "host_bubble_saved_ms": saved, "evaluated": True,
            "normal_ms": None, "capture_ms": counterfactual.duration_ms}


def capture_counterfactual(plan: tuple[PlanNode, ...],
                           hardware: HardwareProfile,
                           normal: EngineEvaluation) -> tuple[EngineEvaluation, dict]:
    if normal.status != "completed":
        return EngineEvaluation("failed", math.inf, 0, 0.0, 0.0, 0.0,
                                reason="normal_plan_failed"), audit_capture(
                                    plan, hardware)
    captured = evaluate_plan(plan, hardware, capture_mode="full",
                             measure_prefetch=False)
    audit = audit_capture(plan, hardware, counterfactual=captured)
    if audit["evaluated"]:
        audit["normal_ms"] = normal.duration_ms
        audit["host_bubble_saved_ms"] = max(0.0, normal.duration_ms - captured.duration_ms)
    return captured, audit


def effective_edges(plan: tuple[PlanNode, ...]) -> tuple[dict, ...]:
    """Read-only projection of every edge the Engine actually evaluated.

    The export walks the same topological execution order as ``evaluate_plan``;
    stream order must follow execution order rather than the raw source tuple.
    """
    edges = []
    by_id = {n.node_id: n for n in plan}
    order = _topological(tuple(n for n in plan if n.kind != "release"))
    stream_prev = _stream_predecessor(order)
    for node in order:
        for dep in node.deps:
            kind = "data"
            if dep.startswith("prefetch_W_"):
                kind = "prefetch_completion"
            elif by_id.get(dep) is not None and by_id[dep].stream != node.stream:
                kind = "event_cross_stream"
            edges.append({"from": dep, "to": node.node_id, "kind": kind})
        if node.node_id in stream_prev:
            edges.append({"from": stream_prev[node.node_id], "to": node.node_id,
                          "kind": "stream_order"})
        if node.kind == "prefetch" and node.prefetch_for:
            consumer = by_id.get(node.prefetch_for)
            if consumer is not None and node.node_id in consumer.deps:
                edges.append({"from": node.node_id, "to": node.prefetch_for,
                              "kind": "prefetch_join"})
        if node.free_alloc is not None and node.prefetch_for:
            edges.append({"from": node.prefetch_for, "to": node.node_id,
                          "kind": "reservation_release"})
    for node in plan:
        if node.kind != "release":
            continue
        for dep in node.deps:
            edges.append({"from": dep, "to": node.node_id,
                          "kind": "reservation_release"})
        if node.prefetch_for and node.prefetch_for in node.deps:
            edges.append({"from": node.prefetch_for, "to": node.node_id,
                          "kind": "prefetch_join"})
    unique = []
    seen = set()
    for edge in edges:
        key = (edge["from"], edge["to"], edge["kind"])
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return tuple(unique)


def necessary_sol(plan: tuple[PlanNode, ...], hardware: HardwareProfile,
                  external: tuple[str, ...] = ()) -> dict:
    durations = {}
    for node in plan:
        work = node.work_dict()
        durations[node.node_id] = max(
            work.get("hbm", 0.0) / hardware.hbm_bytes_per_ms if work.get("hbm") else 0.0,
            work.get("cube", 0.0) / hardware.cube_flops_per_ms if work.get("cube") else 0.0,
            work.get("vector", 0.0) / hardware.vector_ops_per_ms if work.get("vector") else 0.0,
            work.get("mte1", 0.0) / hardware.onchip_bytes_per_ms if work.get("mte1") else 0.0)
    order = _topological(plan, external)
    ends: dict[str, float] = {}
    for node in order:
        ends[node.node_id] = max((ends.get(d, 0.0) for d in node.deps), default=0.0) \
            + durations[node.node_id]
    work = {k: sum(n.work_dict().get(k, 0.0) for n in plan)
            for k in ("hbm", "cube", "vector", "mte1")}
    cp = max(ends.values(), default=0.0)
    sol = max(work["hbm"] / hardware.hbm_bytes_per_ms,
              work["cube"] / hardware.cube_flops_per_ms,
              work["vector"] / hardware.vector_ops_per_ms,
              work["mte1"] / hardware.onchip_bytes_per_ms, cp)
    return {"duration_ms": sol, "critical_path_ms": cp,
            "hbm_lower_bound_ms": work["hbm"] / hardware.hbm_bytes_per_ms,
            "cube_lower_bound_ms": work["cube"] / hardware.cube_flops_per_ms,
            "vector_lower_bound_ms": work["vector"] / hardware.vector_ops_per_ms,
            "onchip_lower_bound_ms": work["mte1"] / hardware.onchip_bytes_per_ms,
            "basis": "necessary_lower_bound"}
