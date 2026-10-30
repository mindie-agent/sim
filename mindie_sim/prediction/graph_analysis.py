"""Subgraph and pure-model costs on the ordinary execution engine.

[Role] Compose operators, streams and cohorts before serving policy is applied.
[Invariants] Graph spans and sums of isolated costs have distinct boundaries.
  Cut dependencies require declared inputs. Resource contention conserves work.
  A precedence bound is not an optimization saving or measured accuracy.
[Timeouts & Locks] Finite input graph; O(graph + active program state), no I/O.
[Side-Effect Policy] No device contexts, measurements or changes to input cases.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import math
from mindie_sim.coverage import model_coverage

from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan
from .cost import CostModel
from .cohort_traffic import bind_cohort_traffic
from .dispatch import cohorts
from .execution import Execution
from .operator_analysis import cohort_operator_costs, operator_costs, physical_id, _replay_scope
from .program import RecipeBook
from .structure import parse_case


def _name(replica, identifier):
    return f"r{replica}:{identifier}"


def _validate_graph(ops, external):
    available = set(external)
    for op in ops:
        if op.op_id in available:
            raise ValueError(f"graph_duplicate_node_or_input:{op.op_id}")
        missing = set(op.deps) - available
        if missing:
            raise ValueError(f"graph_dependency_unavailable:{op.op_id}:{','.join(sorted(missing))}")
        available.add(op.op_id)


def _precedence_bound(rows, external, start):
    """Longest isolated-cost path, including streams and collective rendezvous."""
    ends = {key: max(0, value - start) for key, value in external.items()}
    parents, waiting, successors = {}, {}, {}
    indexed = {row["id"]: row for row in rows}
    for row in rows:
        waiting[row["id"]] = sum(dep not in ends for dep in row["predecessors"])
        for dep in row["predecessors"]:
            successors.setdefault(dep, []).append(row["id"])
    ready = deque(key for key, count in waiting.items() if not count)
    while ready:
        key = ready.popleft()
        row = indexed[key]
        predecessor = max(row["predecessors"], key=ends.__getitem__, default=None)
        parents[key] = predecessor
        ends[key] = ((ends[predecessor] if predecessor is not None else 0)
                     + row.get("precedence_ms", row["isolated_ms"]))
        for successor in successors.get(key, ()):
            waiting[successor] -= 1
            if not waiting[successor]:
                ready.append(successor)
    if len(parents) != len(rows):
        raise ValueError("graph_stream_or_collective_dependency_cycle")
    leaf = max(indexed, key=ends.__getitem__, default=None)
    path, cursor = [], leaf
    while cursor in parents:
        path.append(cursor)
        cursor = parents[cursor]
    return {
        "duration_ms": ends[leaf] if leaf is not None else 0,
        "operators": list(reversed(path)),
        "basis": ("local_dependencies_with_zero_joint_transport_cost"
                  if any("precedence_ms" in row for row in rows)
                  else "isolated_costs_with_data_stream_and_collective_dependencies"),
    }


def graph_costs(case, ops, *, recipes=None, primitives=None, external_inputs=None,
                input_ready_ms=0, include_submission=False, replay_scope=None,
                isolated=True, trace_limit=0):
    """Run a declared DAG or replica->DAG mapping without a request scheduler.

    External input keys are operator dependency names and values are absolute
    virtual ready times, shared across selected replicas. A sequence selects
    replica 0. Cross-DP expert collectives require their complete cohort mapping.
    """
    from .body_defaults import consume_target_declaration
    from .implementation_selection import graph_selections
    case = consume_target_declaration(parse_case(case), ops)
    if input_ready_ms < 0 or not math.isfinite(input_ready_ms):
        raise ValueError("graph_input_ready_time_must_be_finite_and_nonnegative")
    external = dict(external_inputs or {})
    if any(value < 0 or not math.isfinite(value) for value in external.values()):
        raise ValueError("graph_external_input_time_must_be_finite_and_nonnegative")
    book = recipes or RecipeBook.from_mapping(case.runtime.kernels)
    graphs = {r: tuple(book.expand(items)) for r, items in
              (ops.items() if isinstance(ops, Mapping) else ((0, ops),))}
    graphs = bind_cohort_traffic(case, graphs)
    scopes = {r: _replay_scope(items, replay_scope) for r, items in graphs.items()}
    if any(value not in {"none", "full", "piecewise"} for value in scopes.values()):
        raise ValueError("graph_replay_scope_invalid")
    replay_scope = next(iter(set(scopes.values()))) if len(set(scopes.values())) == 1 else "mixed"
    if any(r not in range(case.runtime.parallel.dp) for r in graphs):
        raise ValueError("graph_replica_outside_declared_placement")
    for items in graphs.values():
        _validate_graph(items, external)
        if any(op.family in {"sampling", "token_commit", "acceptance"} for op in items):
            raise ValueError("graph_boundary_excludes_sampling_and_token_commit")
    groups = cohorts(case.runtime.parallel)
    group_for = {r: group for group in groups for r in group}
    for r, items in graphs.items():
        if case.runtime.parallel.ep_spans_dp and any(
            dict(op.metadata).get("group_axis") == "ep" for op in items
        ) and set(group_for[r]) - graphs.keys():
            raise ValueError("graph_expert_collective_requires_complete_cohort")
    records, streams = [], {}
    external_times = {_name(r, key): max(value, input_ready_ms)
                      for r in graphs for key, value in external.items()}

    def observe(op, replica, program, ready, complete):
        key = _name(replica, op.op_id)
        stream = replica, op.stage, op.stream or "main"
        predecessors = [_name(replica, dep) for dep in op.deps]
        if stream in streams:
            predecessors.append(streams[stream])
        streams[stream] = key
        records.append({
            "id": key, "op": op, "replica": replica, "program": program,
            "predecessors": tuple(dict.fromkeys(predecessors)),
            "ready": ready, "complete": complete,
            "collective_algorithm": getattr(book.select(op), "algorithm", None),
        })

    cost = CostModel(case, primitives)
    execution = Execution(case, cost, recipes=book, include_submission=include_submission,
                          on_operator=observe, trace_limit=trace_limit)
    engine = execution.engine
    engine.now = input_ready_ms
    inputs = {r: {key: engine.submit(
        _name(r, "input:" + key), release=max(value, input_ready_ms),
    ) for key, value in external.items()
        if any(key in op.deps for op in items)} for r, items in graphs.items()}
    works = {r: IterationWork(
        items, IterationPlan(()), case.runtime.parallel.pp, 0, 0, None,
        "graph", "declared_graph", replay_scope=scopes[r],
        implementation_selections=graph_selections(items),
    ) for r, items in graphs.items()}
    steps = {}
    for group in groups:
        selected = {r: works[r] for r in group if r in works}
        if len(selected) > 1:
            steps.update(execution.add_group(selected, input_ready_ms, external_inputs=inputs))
        elif selected:
            r, work = next(iter(selected.items()))
            steps[r] = execution.add(work, r, input_ready_ms, external_inputs=inputs[r])
    engine.advance(until=tuple(step.completion for step in steps.values()), first=False)
    end = max((step.completion.time for step in steps.values()), default=input_ready_ms)
    duration = end - input_ready_ms
    # Aggregate recipes retain their rendezvous. Causal transport plans retain
    # rank entry and round dependencies instead of one all-cohort ready gate.
    joins = {}
    if case.runtime.parallel.ep_spans_dp:
        for record in records:
            op, r = record["op"], record["replica"]
            if dict(op.metadata).get("group_axis") == "ep":
                joins.setdefault((group_for[r], op.op_id), []).append(record)
    for participants in joins.values():
        if all(item["collective_algorithm"] for item in participants):
            for item in participants:
                item["joint_collective"] = True
            continue
        dependencies = tuple(dict.fromkeys(dep for item in participants for dep in item["predecessors"]))
        ready_ms = max(item["ready"].time for item in participants)
        for item in participants:
            item["predecessors"] = dependencies
            item["collective_ready_ms"] = ready_ms

    costs, durations, isolated_sum, cohort_cache = [], {}, 0.0, {}
    if isolated:
        for participants in joins.values():
            if not participants[0].get("joint_collective"):
                continue
            signature = tuple((item["replica"], item["op"].stage, physical_id(item["op"]))
                              for item in participants)
            if signature not in cohort_cache:
                report = cohort_operator_costs(
                    case, {item["replica"]: item["op"] for item in participants},
                    recipes=book, primitives=primitives, replay_scope=scopes)
                report["id"] = f"cohort{len(cohort_cache)}"
                report["occurrences"] = 0
                for row in report["operators"]:
                    row.update(cohort_id=report["id"], occurrences=0, total_ms=0)
                    costs.append(row)
                cohort_cache[signature] = report
            report = cohort_cache[signature]
            report["occurrences"] += 1
            for item, row in zip(participants, report["operators"]):
                row["occurrences"] += 1
                row["total_ms"] += row["duration_ms"]
                item["joint_isolated_ms"] = row["duration_ms"]
                item["cohort_id"] = report["id"]
        joint_nodes = {(item["replica"], item["op"].op_id)
                       for members in joins.values() for item in members
                       if item.get("joint_collective")}
        for r, items in graphs.items():
            local = tuple(op for op in items if (r, op.op_id) not in joint_nodes)
            report = operator_costs(case, local, recipes=book, primitives=primitives,
                                    replay_scope=scopes[r], replica=r)
            for row in report["operators"]:
                costs.append({"replica": r, **row})
                durations[r, row["operator"]["stage"], row["physical_id"]] = row["duration_ms"]
        isolated_sum = sum(row["total_ms"] for row in costs)
    rows = []
    for record in records:
        op, r = record["op"], record["replica"]
        ready_ms = record.get("collective_ready_ms", record["ready"].time)
        row = {
            "id": record["id"], "replica": r, "op_id": op.op_id, "family": op.family,
            "recipe": op.recipe, "program": record["program"], "stage": op.stage,
            "layer": op.layer, "stream": op.stream or "main", "phase": op.phase,
            "source_boundary": op.source_boundary, "predecessors": list(record["predecessors"]),
            "ready_ms": ready_ms, "complete_ms": record["complete"].time,
            "elapsed_ms": record["complete"].time - ready_ms,
        }
        # Copy declared identities, not nested phase DAGs or guessed name roles.
        metadata = dict(op.metadata)
        boundary = metadata.get("fusion_boundary", {})
        for key in ("fusion_role", "fusion_group_key", "native_kind"):
            if key in metadata:
                row[key] = metadata[key]
        if boundary:
            row["native_implementation"] = boundary.get("native")
            row["physical_source_program"] = (
                metadata.get("fusion_program_profile") or metadata.get("fusion_source_profile")
                or boundary.get("program_profile") or boundary.get("traffic_profile"))
        if record.get("joint_collective"):
            row["collective_algorithm"] = record["collective_algorithm"]
        if "collective_ready_ms" in record:
            row["local_ready_ms"] = record["ready"].time
            row["rendezvous_wait_ms"] = ready_ms - record["ready"].time
        if isolated:
            if record.get("joint_collective"):
                row.update(
                    isolated_ms=record["joint_isolated_ms"],
                    precedence_ms=0, cohort_id=record["cohort_id"],
                    isolation_boundary="complete_collective_cohort")
            else:
                row["isolated_ms"] = (0 if op.family == "barrier"
                                      else durations[r, op.stage, physical_id(op)])
        rows.append(row)
    resources = [{
        "resource": name, "work": amount, "unit": engine.resources[name].unit,
        "capacity_per_ms": engine.resources[name].rate,
        "capacity_bound_ms": amount / engine.resources[name].rate,
        "busy_ms": engine.busy_ms[name],
    } for name, amount in engine.serviced.items() if amount]
    resources.sort(key=lambda row: row["capacity_bound_ms"], reverse=True)
    result = {
        "schema": "mindie-sim.graph/1",
        **model_coverage(case.model),
        "boundary": "submitted_device_graph" if include_submission else "device_graph",
        "input_ready_ms": input_ready_ms, "complete_ms": end, "duration_ms": duration,
        "replicas": list(graphs), "replay_scope": replay_scope,
        "replay_scope_by_replica": scopes,
        "operator_occurrences": sum(row["family"] != "barrier" for row in rows),
        "timeline": rows, "resources": resources,
        "execution": execution.summary(), "parameters": cost.summary(),
        "events": list(engine.trace),
    }
    if isolated:
        path = _precedence_bound(rows, external_times, input_ready_ms)
        capacity_bound = max((row["capacity_bound_ms"] for row in resources), default=0)
        result["isolated"] = {
            "boundary": "sum_of_isolated_device_programs",
            "operator_sum_ms": isolated_sum, "operators": costs,
            **({"collective_cohorts": [
                {key: value for key, value in report.items() if key != "operators"}
                for report in cohort_cache.values()
            ], "sum_interpretation": "sum_of_replica_representative_completion_spans"}
               if cohort_cache else {}),
        }
        result["bounds"] = {
            "precedence": path, "resource_capacity_ms": capacity_bound,
            "combined_ms": max(path["duration_ms"], capacity_bound),
            "interpretation": "lower_bounds_for_this_declared_graph_not_optimization_savings",
        }
    return result
