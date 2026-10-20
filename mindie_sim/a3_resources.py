"""Event-driven shared-resource service engine.

[Role]
  Advance a dependency DAG of work tasks on explicit resource scopes.  A task
  starts only when its dependencies are complete and carries remaining work on
  one or more resources; at every completion/arrival the active set is
  recomputed and capacity is shared according to the declared policy.

[Invariants]
  * Wall time is produced by event advancement, never by summing isolated task
    durations or taking a max of independent totals.
  * Sharing conserves capacity: for equal-share, each active demander receives
    ``capacity / active_count``; total serviced work equals total demand.
  * A zero-work task completes at its dependency/release time.  A task never
    starts before both are visible.
  * A producer's transfer starts only after the producer event completes.
    Chunked transfers expose each chunk completion separately; a consumer that
    depends on chunk 0 is not blocked by chunk 1.
  * Distinct resource names are physically disjoint.  Local MTE and global HBM
    are separate resources, so a local-only conflict does not serialize HBM.
  * One engine resource per declared physical link.  A task charges the wire
    its fabric scope names (SIO pair vs intra-node switch vs RoCE); an op that
    names a fabric the machine does not declare raises instead of silently
    charging the default link.
  * Replica sharing follows the declared fabric semantics: a switched pipe is
    equal-shared by concurrent replicas, while a mesh/direct pair wire is only
    shared by a concurrent replica whose rank set occupies the same wire.
  * A whole-kernel calibrated duration is represented by work on this engine,
    not re-added as a separate memory duration.

[Timeouts & Locks]
  Deterministic in-memory loop bounded by task/event count and a configurable
  horizon.  No wall-clock sleeps, locks or worker threads.

[Side-Effect Policy]
  No filesystem, network, device or global mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .a3_types import A3InputError, LINK_SCOPE_ALIASES, MachineSpec

_EPS = 1e-12


from .a3_resource_engine import (
    ResourceEngine, ResourceError, ResourceSpec, ResourceTrace, WorkTask,
)


def select_link_for_ranks(
    machine: MachineSpec, ranks: Sequence[int], *,
    channel: str | None = None, semantics: str | None = None,
) -> Any:
    """Choose a physical link by membership, never by array order.

    ``channel``/``semantics`` are optional declared filters so a caller that
    knows which fabric it needs (SIO vs HCCS vs RoCE) cannot be handed an
    unrelated link merely because it also covers the rank set.

    Among covering links the *most specific* one wins (fewest members): a
    two-die TP group inside one A3 package really runs on its SIO pair, not on
    the 8-way switch that also happens to cover it.  When two wires cover the
    same rank set (a single-package model: SIO pair and intra-node switch both
    cover ranks 0..1) the faster dedicated wire wins; the name breaks any
    remaining tie so the result never depends on declaration order.
    """
    rank_set = set(int(rank) for rank in ranks)
    candidates = [
        link for link in machine.links
        if (channel is None or link.channel == channel)
        and (semantics is None or link.semantics == semantics)
    ]
    covering = [
        link for link in candidates
        if rank_set and rank_set.issubset(set(link.members))
    ]
    if covering:
        return min(
            covering,
            key=lambda link: (
                len(link.members), -link.capacity_bytes_per_ms, link.name,
            ),
        )
    overlapping = [
        link for link in candidates
        if rank_set & set(link.members)
    ]
    if overlapping:
        return min(
            overlapping,
            key=lambda link: (
                len(link.members), -link.capacity_bytes_per_ms, link.name,
            ),
        )
    if candidates:
        return sorted(candidates, key=lambda link: link.name)[0]
    if channel is not None or semantics is not None:
        # The caller named a fabric the machine does not declare.  Returning an
        # unrelated link here would silently price traffic on the wrong wire.
        raise ResourceError(
            f"link_fabric_unavailable:{channel or ''}:{semantics or ''}"
        )
    if machine.links:
        return sorted(machine.links, key=lambda link: link.name)[0]
    return None


def plan_replica_links(
    machine: MachineSpec, parallel: Any,
    active_replicas: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Resolve each DP replica's ranks/link and its declared sharing rule.

    A switched fabric is one aggregate pipe: concurrent replicas equal-share
    it.  A mesh (A2's full 8x8 HCCS) or a direct pair wire (A3 SIO) gives each
    rank pair its own wire, so the only contention is another *concurrent
    replica that occupies an overlapping rank*, i.e. the number of pair streams
    on the same wire.  Dividing a mesh by the replica count (the old rule)
    under-predicts A2 bandwidth by up to the DP degree.
    """
    groups = []
    ranks_per_replica = max(1, parallel.pp * parallel.tp)
    for replica in range(parallel.dp):
        ranks = tuple(
            replica * ranks_per_replica + offset
            for offset in range(ranks_per_replica)
        )
        link = select_link_for_ranks(machine, ranks)
        groups.append({
            "replica": replica, "ranks": ranks,
            "link": link,
            "link_name": link.name if link is not None else None,
        })
    if active_replicas is not None:
        active = set(int(index) for index in active_replicas)
        groups = [group for group in groups if group["replica"] in active]
    switched_counts: dict[str | None, int] = {}
    for group in groups:
        semantics = str(getattr(group["link"], "semantics", "") or "unknown")
        group["transport_semantics"] = semantics
        if semantics in {"mesh", "direct"}:
            # Pair-dedicated wire: concurrency is measured by overlapping rank
            # sets, not by how many replicas exist in total.
            concurrent = 1 + sum(
                1 for other in groups
                if other is not group
                and other["link_name"] == group["link_name"]
                and set(other["ranks"]) & set(group["ranks"])
            )
            group["concurrent_pairs"] = concurrent
            group["share_count"] = concurrent
            group["capacity_scale"] = 1.0 / concurrent
            group["sharing_rule"] = (
                "mesh_pair_wire" if semantics == "mesh" else "dedicated_pair_wire"
            )
        else:
            switched_counts[group["link_name"]] = (
                switched_counts.get(group["link_name"], 0) + 1
            )
    for group in groups:
        if "sharing_rule" in group:
            continue
        count = switched_counts.get(group["link_name"], 1)
        group["concurrent_pairs"] = 1
        group["share_count"] = count
        group["capacity_scale"] = 1.0 / max(1, count)
        group["sharing_rule"] = "switched_equal_share"
    return groups


def build_resource_specs(
    machine: MachineSpec,
    stages: int = 1,
    *,
    link_name: str = "d2d",
    link_capacity_scale: float = 1.0,
    compute_dtype: str = "bf16",
    link: Any = None,
    ranks: Sequence[int] | None = None,
) -> tuple[dict[str, ResourceSpec], dict[int, dict[str, str]]]:
    """Create per-PP-stage local resources plus shared link resources."""
    resources: dict[str, ResourceSpec] = {}
    stage_map: dict[int, dict[str, str]] = {}
    if stages < 1:
        raise ResourceError("stages_invalid")
    for stage in range(stages):
        die = machine.dies[stage % len(machine.dies)]
        names = {
            "cube": f"cube@s{stage}",
            "vector": f"vector@s{stage}",
            "hbm": f"hbm@s{stage}",
            "mte2": f"mte2@s{stage}",
            "mte3": f"mte3@s{stage}",
            "mte1": f"mte1@s{stage}",
            "launch": f"launch@s{stage}",
            "serial": f"serial@s{stage}",
            "cpu": f"cpu@s{stage}",
            "aicpu": f"aicpu@s{stage}",
            "hbm_capacity": f"hbm_capacity@s{stage}",
        }
        cube_peak = machine.peak_flops_for(compute_dtype)
        resources[names["cube"]] = ResourceSpec(names["cube"], cube_peak / 1000.0, "flops")
        resources[names["vector"]] = ResourceSpec(names["vector"], die.vector_ops_per_s / 1000.0, "ops")
        resources[names["hbm"]] = ResourceSpec(names["hbm"], die.hbm_bw_bytes_per_s / 1000.0, "bytes")
        resources[names["mte2"]] = ResourceSpec(names["mte2"], die.mte_bw_bytes_per_s / 1000.0, "bytes")
        resources[names["mte3"]] = ResourceSpec(names["mte3"], die.mte_bw_bytes_per_s / 1000.0, "bytes")
        mte1_rate = (
            float(getattr(die, "mte1_bw_bytes_per_s", 0.0) or 0.0)
            or float(die.mte_bw_bytes_per_s)
        ) / 1000.0
        resources[names["mte1"]] = ResourceSpec(names["mte1"], mte1_rate, "bytes")
        resources[names["launch"]] = ResourceSpec(names["launch"], 1.0, "ms")
        resources[names["serial"]] = ResourceSpec(names["serial"], 1.0, "ms")
        resources[names["cpu"]] = ResourceSpec(names["cpu"], 1.0, "ms")
        resources[names["aicpu"]] = ResourceSpec(names["aicpu"], 1.0, "ms")
        resources[names["hbm_capacity"]] = ResourceSpec(
            names["hbm_capacity"], die.hbm_bytes, "bytes",
        )
        stage_map[stage] = names
    if link is None and ranks is not None:
        link = select_link_for_ranks(machine, ranks)
    if link is None:
        link = next((item for item in machine.links if item.name == link_name), None)
    if link is None and machine.links:
        link = sorted(machine.links, key=lambda item: item.name)[0]
    # One engine resource per declared physical link.  Ops that name a fabric
    # (SIO vs HCCS vs RoCE) then charge their own wire and do not serialize on
    # an unrelated one; ops that do not name one keep the replica's default.
    link_resources: dict[str, str] = {}
    for item in machine.links:
        name = f"link:{item.name}"
        if name in resources:
            link_resources.setdefault(item.name, name)
            continue
        capacity = max(_EPS, item.capacity_bytes_per_ms)
        if link is not None and item.name == link.name:
            capacity = max(_EPS, item.capacity_bytes_per_ms * link_capacity_scale)
        resources[name] = ResourceSpec(name, capacity, "bytes", scope=item.scope)
        link_resources[item.name] = name
    if link is not None:
        default_name = link_resources[link.name]
        for stage in stage_map:
            # Insert the default first: preparation tasks pick the first
            # ``link:`` value in iteration order as their transfer wire.
            stage_map[stage]["link"] = default_name
            for item in machine.links:
                # Instance-scoped tokens only: channel tokens are resolved
                # through ``link_for_scope`` below so several wires of one
                # channel (A3 SIO per package) resolve deterministically.
                for token in {item.name.lower(), item.scope.lower()}:
                    stage_map[stage][f"link_scope:{token}"] = link_resources[item.name]
            for alias in sorted(LINK_SCOPE_ALIASES):
                try:
                    resolved = machine.link_for_scope(alias)
                except A3InputError:
                    continue
                if resolved is not None:
                    stage_map[stage][f"link_scope:{alias}"] = link_resources[resolved.name]
    return resources, stage_map


def _finish_stage_graph(
    engine: ResourceEngine, *, task_id: str, source_id: str,
    stages: Sequence[Mapping[str, Any]], stage_ids: Sequence[str],
    deps: Sequence[str], release_ms: float, suffix: str,
    owner_prefix: str,
) -> str:
    """Join independent terminal fragments before exposing one op output.

    A source op can expand into several parallel terminal stages (for example
    vector work and a link fragment).  Consumers must depend on all of them,
    not merely on whichever stage happened to be added last.
    """
    if not stage_ids:
        raise ResourceError(f"stage_graph_empty:{source_id}")
    used: set[int] = set()
    for stage in stages:
        for index in stage.get("deps", ()) or ():
            used.add(int(index))
    terminals = [index for index in range(len(stage_ids)) if index not in used]
    if len(terminals) <= 1:
        return stage_ids[terminals[0]] if terminals else stage_ids[-1]
    join_id = f"{task_id}:{suffix}_join"
    engine.add_task(
        join_id, {}, dependencies=tuple(stage_ids[index] for index in terminals),
        release_ms=release_ms, owner=f"{owner_prefix}:join",
        metadata={"family": "stage_join", "source": source_id},
    )
    return join_id


def _ensure_core_resource(
    engine: ResourceEngine, base_name: str, core_id: int, total_cores: int,
    *, divide_by_cores: bool = True,
) -> str:
    """Create/read one global per-engine per-worker capacity slice."""
    safe_base = str(base_name).replace(":", "_")
    name = f"{safe_base}@core{int(core_id)}"
    if name not in engine.resources:
        base = engine.resources.get(base_name)
        if base is None:
            raise ResourceError(f"stage_resource_missing:core_base:{base_name}")
        capacity = float(base.capacity)
        if divide_by_cores and total_cores > 0:
            capacity /= float(total_cores)
        engine.resources[name] = ResourceSpec(
            name, capacity, base.unit, "core", "equal_share",
        )
        engine.serviced.setdefault(name, 0.0)
    return name


def _link_resource_for_op(
    op: Any, stage_names: Mapping[str, str], override: str | None,
) -> str:
    """Pick the engine link resource for an op's declared fabric scope."""
    if override:
        return override
    default = stage_names.get("link")
    declared = str(getattr(op, "link_scope", "") or "").strip().lower()
    if not declared or declared in {"d2d", "default", "shared"}:
        if default is None:
            raise ResourceError(f"link_resource_missing_for:{op.op_id}")
        return default
    mapped = stage_names.get(f"link_scope:{declared}")
    if mapped is None:
        # The op names a fabric the machine does not declare.  Charging the
        # default link here would silently price SIO traffic on HCCS.
        raise ResourceError(f"link_scope_unresolved:{op.op_id}:{declared}")
    return mapped


def add_costed_ops(
    engine: ResourceEngine,
    costed_ops: Sequence[Any],
    stage_map: Mapping[int, Mapping[str, str]],
    *,
    prefix: str,
    start_ms: float,
    link_scope: str | None = None,
    release_overrides: Mapping[str, float] | None = None,
    external_deps: Sequence[str] = (),
    family_external_deps: Mapping[str, Sequence[str]] | None = None,
    replacement_tasks: Mapping[str, Sequence[str]] | None = None,
    op_external_deps: Mapping[str, Sequence[str]] | None = None,
    core_counts: Mapping[str, int] | None = None,
) -> dict[str, str]:
    """Register costed op DAGs, mapping generic demands to stage resources."""
    id_map: dict[str, str] = {}
    released = release_overrides or {}
    for costed in costed_ops:
        op = costed.op
        source_id = op.op_id
        task_id = f"{prefix}:{source_id}"
        id_map[source_id] = task_id
        demands = costed.demand_map()
        envelope_ms = float(demands.pop("completion_envelope_ms", 0.0) or 0.0)
        if not math.isfinite(envelope_ms) or envelope_ms < 0:
            raise ResourceError(f"completion_envelope_invalid:{source_id}")
        mapped: dict[str, float] = {}
        stage_names = stage_map.get(op.stage, {})
        for key, value in demands.items():
            if key == "link":
                resource = _link_resource_for_op(op, stage_names, link_scope)
                mapped[resource] = mapped.get(resource, 0.0) + value
            else:
                resource = stage_names.get(key)
                if resource is None:
                    raise ResourceError(f"stage_resource_missing:{op.stage}:{key}")
                mapped[resource] = mapped.get(resource, 0.0) + value
        if envelope_ms > 0:
            # A measured whole-span completion is a per-task elapsed-time
            # assumption, not physical link/AIV/AICPU/serial occupancy.  A
            # private capacity-1 resource lets this task finish no earlier than
            # the measured span without inventing contention with other tasks.
            envelope_name = f"{task_id}:completion_envelope"
            engine.resources[envelope_name] = ResourceSpec(
                envelope_name, 1.0, "ms", "per_task", "equal_share",
            )
            engine.serviced.setdefault(envelope_name, 0.0)
            mapped[envelope_name] = envelope_ms
        serial_ms = float(dict(op.metadata).get("serial_ms", 0.0) or 0.0)
        if serial_ms > 0:
            serial_name = stage_names.get("serial")
            if serial_name is None:
                raise ResourceError(f"stage_resource_missing:{op.stage}:serial")
            mapped[serial_name] = mapped.get(serial_name, 0.0) + serial_ms
        deps = tuple(id_map.get(dep, f"{prefix}:{dep}") for dep in op.deps)
        if not deps and external_deps:
            deps = tuple(external_deps)
        extra = (family_external_deps or {}).get(op.family, ())
        if extra:
            deps = tuple(dict.fromkeys((*deps, *extra)))
        specific = (op_external_deps or {}).get(source_id, ())
        if specific:
            deps = tuple(dict.fromkeys((*deps, *tuple(str(v) for v in specific))))
        replacements = (replacement_tasks or {}).get(source_id, ())
        if isinstance(replacements, str):
            replacements = (replacements,)
        if replacements:
            # The preparation task replaces this semantic operation.  Keep a
            # zero-work bridge so downstream dependencies still resolve, but
            # do not charge the original compute/traffic twice.
            bridge_deps = tuple(dict.fromkeys(
                (*deps, *tuple(str(item) for item in replacements))
            ))
            engine.add_task(
                task_id, {}, dependencies=bridge_deps,
                release_ms=released.get(source_id, start_ms),
                owner=f"prepare_replaced:{source_id}",
                metadata={
                    "family": op.family, "stage": op.stage,
                    "recipe": op.recipe, "replacement_tasks": list(replacements),
                },
            )
            continue
        comm_stages = tuple(
            dict(stage) for stage in dict(op.metadata).get("comm_stages", ())
        )
        if comm_stages:
            stage_ids: list[str] = []
            comm_launch = float(costed.demand_map().get("launch", 0.0))
            for stage_index, stage in enumerate(comm_stages):
                stage_id = f"{task_id}:comm{stage_index}"
                demands: dict[str, float] = {}
                work = dict(stage.get("work", {}) or {})
                for logical, resource_key in (
                    ("aicpu_service", "aicpu"), ("vector_ops", "vector"),
                    ("mte2_bytes", "mte2"), ("mte3_bytes", "mte3"),
                    ("link_edge_bytes", "link"),
                ):
                    number = float(work.get(logical, 0.0) or 0.0)
                    if number > 0:
                        resource = (
                            _link_resource_for_op(op, stage_names, link_scope)
                            if resource_key == "link"
                            else stage_names.get(resource_key)
                        )
                        if resource is None:
                            raise ResourceError(
                                f"stage_resource_missing:{op.stage}:{resource_key}"
                            )
                        demands[resource] = demands.get(resource, 0.0) + number
                hbm_bytes = sum(
                    float(work.get(key, 0.0) or 0.0) for key in (
                        "local_hbm_read", "local_hbm_write",
                        "peer_hbm_read", "peer_hbm_write",
                    )
                )
                if hbm_bytes > 0:
                    resource = stage_names.get("hbm")
                    if resource is None:
                        raise ResourceError(f"stage_resource_missing:{op.stage}:hbm")
                    demands[resource] = demands.get(resource, 0.0) + hbm_bytes
                if stage_index == 0 and comm_launch > 0:
                    launch = stage_names.get("launch")
                    if launch is not None:
                        demands[launch] = demands.get(launch, 0.0) + comm_launch
                internal = tuple(stage.get("deps", ()) or ())
                stage_deps = (
                    tuple(stage_ids[int(index)] for index in internal)
                    if internal else deps
                )
                reservations = dict(stage.get("reservations", {}) or {})
                capacities = dict(stage.get("reservation_capacities", {}) or {})
                modeled_reservations: dict[str, float] = {}
                slot_aliases = {
                    "aiv_slots": "vector",
                    "aiv_core_slots": "vector",
                    "aiv_cores": "vector",
                    "vector_cores": "vector",
                }
                for name, amount in reservations.items():
                    physical = name if name in engine.resources else (
                        stage_names.get(name)
                        or stage_names.get(slot_aliases.get(name, ""))
                    )
                    if physical is not None and physical in engine.resources:
                        raw_amount = float(amount)
                        total_cores = None
                        if name in {"aiv_slots", "aiv_core_slots", "aiv_cores"}:
                            total_cores = capacities.get(
                                "aiv_cores", capacities.get(name)
                            )
                            if total_cores is None or float(total_cores) <= 0:
                                # A numeric core count cannot be interpreted as
                                # ops/ms.  Unknown capacity stays unmodeled.
                                continue
                            capacity_units = float(
                                engine.resources[physical].capacity
                            )
                            raw_amount = (
                                capacity_units * float(amount)
                                / float(total_cores)
                            )
                        # Logical AIV/vector reservations throttle the physical
                        # per-die compute resource, never a same-named global
                        # shadow pool that serializes independent dies.
                        modeled_reservations[physical] = (
                            modeled_reservations.get(physical, 0.0)
                            + raw_amount
                        )
                        continue
                    capacity = (
                        capacities.get(name, capacities.get(physical))
                        if physical is not None else capacities.get(name)
                    )
                    if capacity is None or float(capacity) <= 0:
                        continue
                    engine.resources[name] = ResourceSpec(
                        name, float(capacity), "units", "shared", "equal_share",
                    )
                    engine.serviced.setdefault(name, 0.0)
                    modeled_reservations[name] = float(amount)
                engine.add_task(
                    stage_id, demands, dependencies=stage_deps,
                    reservations=modeled_reservations or None,
                    release_ms=released.get(source_id, start_ms),
                    owner=f"{source_id}:{stage.get('kind', stage_index)}",
                    metadata={
                        "family": op.family, "stage": op.stage,
                        "recipe": op.recipe, "comm_stage_kind": stage.get("kind"),
                        "comm_engine": stage.get("engine"),
                        "reservations": reservations,
                        "modeled_reservations": modeled_reservations,
                    },
                )
                stage_ids.append(stage_id)
            id_map[source_id] = _finish_stage_graph(
                engine, task_id=task_id, source_id=source_id,
                stages=comm_stages, stage_ids=stage_ids, deps=deps,
                release_ms=released.get(source_id, start_ms),
                suffix="comm", owner_prefix=source_id,
            )
            continue
        stages = tuple(dict(stage) for stage in dict(op.metadata).get("fusion_stages", ()))
        if stages:
            stage_ids: list[str] = []
            summary_demands = costed.demand_map()
            launch_demand = float(summary_demands.get("launch", 0.0))
            for stage_index, stage in enumerate(stages):
                stage_id = f"{task_id}:stage{stage_index}:{stage.get('name', stage_index)}"
                stage_demands: dict[str, float] = {}
                stage_resources = dict(stage.get("resources", {}) or {})
                resource_aliases = {
                    "flops": "cube", "cube": "cube",
                    "vector_ops": "vector", "vector": "vector",
                    "hbm_bytes": "hbm", "hbm": "hbm",
                    "mte1": "mte1", "mte2": "mte2", "mte2_bytes": "mte2",
                    "mte3": "mte3", "mte3_bytes": "mte3",
                }
                stage_amounts: dict[str, float] = {}
                for key, value in (
                    ("flops", stage.get("flops", 0.0)),
                    ("vector_ops", stage.get("vector_ops", 0.0)),
                    ("hbm_bytes", stage.get("hbm_bytes", 0.0)),
                    ("mte2_bytes", stage.get("mte2_bytes", 0.0)),
                    ("mte3_bytes", stage.get("mte3_bytes", 0.0)),
                ):
                    resource_key = resource_aliases.get(str(key))
                    if resource_key is not None:
                        stage_amounts[resource_key] = float(value or 0.0)
                for key, value in stage_resources.items():
                    resource_key = resource_aliases.get(str(key))
                    if resource_key is None:
                        continue
                    # An explicit resources entry is authoritative for that
                    # physical resource; do not double-count the flat alias.
                    stage_amounts[resource_key] = float(value or 0.0)
                assignment = dict(stage.get("core_assignment", {}) or {})
                core_ids = tuple(
                    int(value) for value in assignment.get("core_ids", ())
                )
                counts = dict(core_counts or {})
                for resource_key, number in stage_amounts.items():
                    if number <= 0:
                        continue
                    resource = stage_names.get(resource_key)
                    if resource is None and resource_key == "mte1":
                        # Compatibility for callers that have not yet declared
                        # an MTE1 local-transfer resource: charge on MTE2
                        # rather than silently dropping the move.
                        resource = stage_names.get("mte2")
                    if resource is None:
                        raise ResourceError(f"stage_resource_missing:{op.stage}:{resource_key}")
                    if core_ids and resource_key in {"cube", "vector", "mte1"}:
                        total = int(counts.get(
                            "aiv" if resource_key == "vector" else "aic", 0,
                        ) or 0)
                        share = float(number) / float(len(core_ids))
                        for core_id in core_ids:
                            core_resource = _ensure_core_resource(
                                engine, resource, core_id, total,
                                divide_by_cores=(resource_key != "mte1"),
                            )
                            stage_demands[core_resource] = (
                                stage_demands.get(core_resource, 0.0) + share
                            )
                    else:
                        stage_demands[resource] = (
                            stage_demands.get(resource, 0.0) + number
                        )
                stage_reservations: dict[str, float] = {}
                width = dict(stage.get("execution_width", {}) or {})
                counts = dict(core_counts or {})
                for width_key, core_key, resource_key in (
                    ("aic_cores", "aic", "cube"), ("aiv_cores", "aiv", "vector"),
                ):
                    active = float(width.get(width_key, 0) or 0)
                    total = float(counts.get(core_key, 0) or 0)
                    resource = stage_names.get(resource_key)
                    if active <= 0 or total <= 0 or resource is None:
                        continue
                    capacity = float(engine.resources[resource].capacity)
                    stage_reservations[resource] = capacity * active / total
                if stage.get("launch") and launch_demand > 0:
                    resource = stage_names.get("launch")
                    if resource is not None:
                        stage_demands[resource] = stage_demands.get(resource, 0.0) + launch_demand
                internal = tuple(stage.get("deps", ()))
                stage_deps = (
                    tuple(stage_ids[int(index)] for index in internal)
                    if internal else deps
                )
                engine.add_task(
                    stage_id, stage_demands, dependencies=stage_deps,
                    reservations=stage_reservations or None,
                    release_ms=released.get(source_id, start_ms),
                    owner=f"{source_id}:{stage.get('name', stage_index)}",
                    metadata={"family": stage.get("family", op.family),
                              "stage": op.stage, "recipe": op.recipe,
                              "fused_parent": source_id},
                )
                stage_ids.append(stage_id)
            id_map[source_id] = _finish_stage_graph(
                engine, task_id=task_id, source_id=source_id,
                stages=stages, stage_ids=stage_ids, deps=deps,
                release_ms=released.get(source_id, start_ms),
                suffix="fused", owner_prefix=source_id,
            )
            continue
        engine.add_task(
            task_id, mapped, dependencies=deps,
            release_ms=released.get(source_id, start_ms),
            owner=source_id,
            metadata={
                "family": op.family, "stage": op.stage, "recipe": op.recipe,
                "completion_envelope_ms": envelope_ms,
            },
        )
    return id_map


def simulate_microbatch_pipeline(
    stage_times_ms: Sequence[float],
    microbatches: int,
    *,
    transfer_ms: float = 0.0,
    start_ms: float = 0.0,
) -> dict[str, Any]:
    """Finite PP microbatch schedule with optional stage-transfer latency.

    Stage resources are FIFO (one microbatch occupies a stage at a time); a
    microbatch transfers between adjacent stages before the downstream stage
    task is released.  This is deliberately not ``total_time / pp``.
    """
    if not stage_times_ms:
        raise ResourceError("pipeline_stages_empty")
    if microbatches < 1:
        raise ResourceError("pipeline_microbatches_invalid")
    if not math.isfinite(float(transfer_ms)) or float(transfer_ms) < 0:
        raise ResourceError("pipeline_transfer_invalid")
    resources: dict[str, ResourceSpec] = {}
    for index, duration in enumerate(stage_times_ms):
        if not math.isfinite(float(duration)) or float(duration) < 0:
            raise ResourceError(f"pipeline_stage_time_invalid:{index}")
        resources[f"stage{index}"] = ResourceSpec(
            f"stage{index}", 1.0, "stage_slot", policy="fifo",
        )
    if transfer_ms > 0:
        resources["transfer"] = ResourceSpec("transfer", 1.0, "transfer_slot", policy="fifo")
    engine = ResourceEngine(resources)
    last_stage = len(stage_times_ms) - 1
    stage_end: dict[tuple[int, int], str] = {}
    for mb in range(microbatches):
        dependency: str | None = None
        for stage, duration in enumerate(stage_times_ms):
            task_id = f"mb{mb}.s{stage}"
            engine.add_task(
                task_id,
                {f"stage{stage}": float(duration)},
                dependencies=(dependency,) if dependency else (),
                release_ms=start_ms,
                priority=mb,
                owner=f"microbatch:{mb}",
                metadata={"microbatch": mb, "stage": stage},
            )
            stage_end[(mb, stage)] = task_id
            dependency = task_id
            if stage < last_stage and transfer_ms > 0:
                transfer_id = f"mb{mb}.transfer{stage}"
                engine.add_task(
                    transfer_id,
                    {"transfer": float(transfer_ms)},
                    dependencies=(task_id,),
                    release_ms=start_ms,
                    priority=mb,
                    owner=f"microbatch:{mb}",
                    metadata={"microbatch": mb, "stage": stage, "kind": "transfer"},
                )
                dependency = transfer_id
    result = engine.run(start_ms=start_ms)
    first_token = max(
        (result["completed_at_ms"][stage_end[(0, stage)]] for stage in range(last_stage + 1)),
        default=start_ms,
    )
    last_completion = max(
        (result["completed_at_ms"][stage_end[(microbatches - 1, stage)]]
         for stage in range(last_stage + 1)),
        default=start_ms,
    )
    return {
        "status": "modeled",
        "stage_times_ms": [float(value) for value in stage_times_ms],
        "microbatches": microbatches,
        "transfer_ms": float(transfer_ms),
        "first_token_ms": first_token,
        "makespan_ms": result["makespan_ms"],
        "last_token_ms": last_completion,
        "timeline": [
            {
                "microbatch": mb, "stage": stage,
                "task_id": stage_end[(mb, stage)],
                "start_ms": next(item.start_ms for item in engine.trace if item.task_id == stage_end[(mb, stage)]),
                "end_ms": result["completed_at_ms"][stage_end[(mb, stage)]],
            }
            for mb in range(microbatches) for stage in range(last_stage + 1)
        ],
        "note": "FIFO stage slots; finite microbatch schedule, not II*tiles or total/PP",
    }


__all__ = [
    "ResourceError", "ResourceSpec", "WorkTask", "ResourceTrace", "ResourceEngine",
    "build_resource_specs", "add_costed_ops", "simulate_microbatch_pipeline",
    "select_link_for_ranks", "plan_replica_links", "fusion_pool_capacities",
]


def fusion_pool_capacities(die: Any) -> dict[str, int] | None:
    """Return declared L1/UB/L0C byte capacities, or None for unified-only.

    L2 is not a kernel-local pool.  A synthetic unified capacity must stay
    labelled as such by the fusion witness; only explicit profile/caller
    fields become physical pool declarations.
    """
    values = {
        "l1": int(getattr(die, "l1_bytes", 0) or 0),
        "ub": int(getattr(die, "ub_bytes", 0) or 0),
        "l0a": int(getattr(die, "l0a_bytes", 0) or 0),
        "l0b": int(getattr(die, "l0b_bytes", 0) or 0),
        "l0c": int(getattr(die, "l0c_bytes", 0) or 0),
    }
    return values if all(value > 0 for value in values.values()) else None
