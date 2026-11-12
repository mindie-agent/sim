"""Subgraph and pure-model costs on the ordinary execution engine.

[Role] Compose operators, streams and cohorts before serving policy is applied.
[Invariants] Graph spans and sums of isolated costs have distinct boundaries.
  Cut dependencies require declared inputs. Resource contention conserves work.
  A precedence bound is not an optimization saving or measured accuracy.
  Compact results retain the same event boundaries and exact aggregate counts
  without materializing per-event rows or copying the engine trace.
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
from .native_binding import bind_native_call
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


def _bounded_event_profile(engine):
    """Separate completed physical events from optional debug-trace retention."""
    retained = len(engine.trace)
    return {
        "completed_physical_events": engine.completed_count,
        "completed_tasks": engine.completed_count,
        "trace_retained": retained,
        "trace_dropped": engine.trace_dropped,
        "trace_events": retained + engine.trace_dropped,
    }


def _matrix_wrapper_projection(metadata):
    """Publish only the selected wrapper's semantic kind/source role.

    The nested metadata is the explicit source-selection result; absent it the
    ordinary matrix row keeps no kind. This is attribution, not a second op.
    """
    wrapper = metadata.get("matrix_wrapper")
    if not isinstance(wrapper, Mapping):
        return {}
    native = wrapper.get("native_kind")
    if not isinstance(native, str) or not native:
        return {}
    result = {"native_kind": native}
    role = wrapper.get("source_role")
    if isinstance(role, str) and role:
        result["source_role"] = role
    source = wrapper.get("source")
    basis = (wrapper.get("source_basis")
             or (source.get("basis") if isinstance(source, Mapping) else None))
    if isinstance(basis, str) and basis:
        result["source_basis"] = basis
    return result


def graph_costs(case, ops, *, recipes=None, primitives=None, external_inputs=None,
                input_ready_ms=0, include_submission=False, replay_scope=None,
                isolated=True, trace_limit=0, compact=False):
    """Run a declared DAG or replica->DAG mapping without a request scheduler.

    External input keys are operator dependency names and values are absolute
    virtual ready times, shared across selected replicas. A sequence selects
    replica 0. Cross-DP expert collectives require their complete cohort mapping.

    ``compact=True`` is additive: it omits the per-event ``timeline`` and
    ``events`` lists and returns the same duration, operator events and
    readiness/completion boundaries as bounded ``profile``/``event_profile``
    reductions. No synthetic time histogram is built; each replica's declared
    identity and source ready/complete boundaries are retained.
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
        # A real native cohort owns its own launch-local readiness. Use the
        # actual launch record, not a family label: a plan that merely emits a
        # native-looking op is not evidence that peer work ran.
        if execution.native_group_launches.get(participants[0]["op"].op_id):
            for item in participants:
                item["native_local_ready"] = True
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
    profile = None
    if compact:
        profile = {
            "duration_ms": duration, "operator_events": 0,
            "by_family": {}, "by_operator": {},
            "by_rank": {}, "by_layer": {},
        }
    rows = []
    for record in records:
        op, r = record["op"], record["replica"]
        ready_ms = record.get("collective_ready_ms", record["ready"].time)
        complete_ms = record["complete"].time
        elapsed = complete_ms - ready_ms
        # Copy declared identities, not nested phase DAGs or guessed name roles.
        metadata = dict(op.metadata)
        boundary = metadata.get("fusion_boundary", {})
        # One normalized actual-invocation descriptor for the per-event row and
        # the compact per-rank projection. It is metadata, never priced work.
        native_call = bind_native_call(op)
        native_payload = native_call.to_dict() if native_call is not None else None
        if compact:
            profile["operator_events"] += 1
            family = profile["by_family"].setdefault(
                op.family, {"operator_events": 0, "elapsed_ms": 0.0,
                            "replicas": {}})
            family["operator_events"] += 1
            family["elapsed_ms"] += elapsed
            rank_totals = profile["by_rank"].setdefault(
                r, {"operator_events": 0, "elapsed_ms": 0.0,
                    "ready_ms": ready_ms, "complete_ms": complete_ms})
            rank_totals["operator_events"] += 1
            rank_totals["elapsed_ms"] += elapsed
            rank_totals["ready_ms"] = min(rank_totals["ready_ms"], ready_ms)
            rank_totals["complete_ms"] = max(rank_totals["complete_ms"], complete_ms)
            layer_totals = profile["by_layer"].setdefault(
                op.layer, {"operator_events": 0, "elapsed_ms": 0.0})
            layer_totals["operator_events"] += 1
            layer_totals["elapsed_ms"] += elapsed
            # One canonical operator entry, with each replica's actual declared
            # identity and source boundaries retained below it. No synthetic
            # histogram and no first-rank label leaking onto another rank.
            identity = f"s{op.stage}:{op.op_id}"
            stats = profile["by_operator"].get(identity)
            if stats is None:
                stats = {
                    "op_id": op.op_id, "stage": op.stage,
                    "operator_events": 0, "ready_ms": ready_ms,
                    "complete_ms": complete_ms, "elapsed_ms": 0.0,
                    "replicas": {},
                }
                profile["by_operator"][identity] = stats
            stats["operator_events"] += 1
            stats["ready_ms"] = min(stats["ready_ms"], ready_ms)
            stats["complete_ms"] = max(stats["complete_ms"], complete_ms)
            stats["elapsed_sum_ms"] = stats.get("elapsed_sum_ms", 0.0) + elapsed
            stats["elapsed_ms"] = max(stats.get("elapsed_ms", 0.0), elapsed)
            rank = stats["replicas"].get(r)
            if rank is None:
                rank = {
                    "operator_events": 0, "ready_ms": ready_ms,
                    "complete_ms": complete_ms, "elapsed_ms": 0.0,
                    "family": op.family, "layer": op.layer,
                    "stream": op.stream or "main",
                    "phase": op.phase, "recipe": op.recipe,
                    "program": record["program"],
                    "source_boundary": op.source_boundary,
                }
                stats["replicas"][r] = rank
            if native_payload is not None:
                rank["native_call"] = native_payload
            rank["operator_events"] += 1
            rank["ready_ms"] = min(rank["ready_ms"], ready_ms)
            rank["complete_ms"] = max(rank["complete_ms"], complete_ms)
            rank["elapsed_ms"] += elapsed
            for key in ("fusion_role", "fusion_group_key", "native_kind"):
                if key in metadata:
                    rank[key] = metadata[key]
            rank.update(_matrix_wrapper_projection(metadata))
            if boundary:
                rank["native_implementation"] = boundary.get("native")
                rank["physical_source_program"] = (
                    metadata.get("fusion_program_profile")
                    or metadata.get("fusion_source_profile")
                    or boundary.get("program_profile") or boundary.get("traffic_profile"))
            if record.get("joint_collective"):
                rank["collective_algorithm"] = record["collective_algorithm"]
            if record.get("native_local_ready"):
                rank["native_local_ready_ms"] = record["ready"].time
            if "collective_ready_ms" in record:
                wait = ready_ms - record["ready"].time
                rank["local_ready_ms"] = record["ready"].time
                rank["rendezvous_wait_ms"] = max(
                    rank.get("rendezvous_wait_ms", 0.0), wait)
                rank["rendezvous_events"] = rank.get("rendezvous_events", 0) + 1
            if isolated:
                if record.get("joint_collective"):
                    iso = record["joint_isolated_ms"]
                    record["isolated_ms"] = iso
                    record["precedence_ms"] = 0
                    rank.update(cohort_id=record["cohort_id"],
                                isolation_boundary="complete_collective_cohort")
                else:
                    iso = (0 if op.family == "barrier"
                           else durations[r, op.stage, physical_id(op)])
                    record["isolated_ms"] = iso
                rank["isolated_ms"] = rank.get("isolated_ms", 0.0) + iso
                stats["isolated_ms"] = stats.get("isolated_ms", 0.0) + iso
                family["isolated_ms"] = family.get("isolated_ms", 0.0) + iso
            family_rank = family["replicas"].setdefault(
                r, {"operator_events": 0, "elapsed_ms": 0.0,
                    "ready_ms": ready_ms, "complete_ms": complete_ms})
            family_rank["operator_events"] += 1
            family_rank["elapsed_ms"] += elapsed
            family_rank["ready_ms"] = min(family_rank["ready_ms"], ready_ms)
            family_rank["complete_ms"] = max(family_rank["complete_ms"], complete_ms)
            continue
        row = {
            "id": record["id"], "replica": r, "op_id": op.op_id, "family": op.family,
            "recipe": op.recipe, "program": record["program"], "stage": op.stage,
            "layer": op.layer, "stream": op.stream or "main", "phase": op.phase,
            "source_boundary": op.source_boundary, "predecessors": list(record["predecessors"]),
            "ready_ms": ready_ms, "complete_ms": record["complete"].time,
            "elapsed_ms": record["complete"].time - ready_ms,
        }
        for key in ("fusion_role", "fusion_group_key", "native_kind"):
            if key in metadata:
                row[key] = metadata[key]
        row.update(_matrix_wrapper_projection(metadata))
        if native_payload is not None:
            row["native_call"] = native_payload
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
    isolated_profile = None
    if compact and isolated:
        isolated_profile = {
            "operator_sum_ms": isolated_sum,
            "by_family": {}, "by_operator": {},
        }
        for row in costs:
            operator = row["operator"]
            replica = row.get("replica")
            key = (f"r{replica}:{operator['op_id']}" if replica is not None
                   else f"{row.get('cohort_id', 'cohort')}:{operator['op_id']}")
            item = isolated_profile["by_operator"].setdefault(key, {
                "replica": replica, "op_id": operator["op_id"],
                "family": operator["family"], "stage": operator["stage"],
                "layer": operator["layer"], "physical_id": row["physical_id"],
                "occurrences": 0, "total_ms": 0.0,
            })
            item["occurrences"] += row["occurrences"]
            item["total_ms"] += row["total_ms"]
            family = isolated_profile["by_family"].setdefault(
                operator["family"], {"occurrences": 0, "total_ms": 0.0})
            family["occurrences"] += row["occurrences"]
            family["total_ms"] += row["total_ms"]
        costs = []
    resources = [{
        "resource": name, "work": amount, "unit": engine.resources[name].unit,
        "capacity_per_ms": engine.resources[name].rate,
        "capacity_bound_ms": amount / engine.resources[name].rate,
        "busy_ms": engine.busy_ms[name],
    } for name, amount in engine.serviced.items() if amount]
    resources.sort(key=lambda row: row["capacity_bound_ms"], reverse=True)
    head = {
        "schema": "mindie-sim.graph/1",
        **model_coverage(case.model),
        "boundary": "submitted_device_graph" if include_submission else "device_graph",
        "input_ready_ms": input_ready_ms, "complete_ms": end, "duration_ms": duration,
        "replicas": list(graphs), "replay_scope": replay_scope,
        "replay_scope_by_replica": scopes,
        "operator_occurrences": sum(
            record["op"].family != "barrier" for record in records),
    }
    execution_summary, parameter_summary = execution.summary(), cost.summary()
    if compact:
        result = {
            **head, "resources": resources,
            "execution": execution_summary, "parameters": parameter_summary,
        }
        profile["steps_by_replica"] = {
            replica: {
                "start_ms": step.start_ms,
                "device_end_ms": step.device_complete.time,
                "end_ms": step.completion.time,
            }
            for replica, step in steps.items()
        }
        native_records = [
            record for record in records
            if (record["op"].family in {"mc2_transport", "mc2_protocol"}
                or "mc2_phase" in dict(record["op"].metadata))
        ]
        native_groups = list(execution_summary.get("native_groups", ()) or ())
        if native_records:
            launched_ids = {group["op_id"] for group in native_groups}
            launched = [record for record in native_records
                        if record["op"].op_id in launched_ids]
            protocol = [record for record in native_records
                        if record["op"].op_id not in launched_ids]
            bases = sorted({
                str(basis) for group in native_groups
                for basis in (group.get("payload_basis") or {}).values()})
            source_basis = next(
                (basis for basis in bases if "source_force_eplb" in basis),
                next((basis for basis in bases if basis != "actual_zero"),
                     bases[0] if bases else "unknown"))
            for record in native_records:
                if source_basis != "unknown":
                    break
                meta = dict(record["op"].metadata)
                for key in ("source_force_eplb", "source_token_mask"):
                    source = meta.get(key)
                    if isinstance(source, Mapping) and source.get("basis"):
                        source_basis = str(source["basis"])
                        break
            exact = bool(native_groups) and all(
                basis.startswith("source_force_eplb") or basis == "actual_zero"
                for group in native_groups
                for basis in (group.get("payload_basis") or {}).values())
            expectation = any(
                basis not in {"actual_zero"} and
                not basis.startswith("source_force_eplb")
                for group in native_groups
                for basis in (group.get("payload_basis") or {}).values())
            directed = sum(int(group.get("directed_edges", 0))
                           for group in native_groups)
            tp = int(case.runtime.parallel.tp)
            bridge_status = "actual" if launched else "partial"
            bridge_reason = ("" if launched else
                             "ordinary_frontend_native_group_not_materialized")
            profile["native_bridge"] = {
                "status": bridge_status,
                "reason": bridge_reason,
                "source_basis": source_basis,
                "known_vs_expected": (
                    "not_materialized" if not launched else
                    "declared_destination_expectation" if expectation else
                    "exact_directed_source_frames" if exact and directed else
                    "exact_zero_source_frames" if exact else "unknown"),
                "physical_tp_scope": ("resolved" if tp == 1 else
                                      "unresolved_tp_rank_mapping"),
                "native_operators": len(launched),
                "native_launch_groups": len(native_groups),
                "native_kernel_calls": len(native_groups),
                "launch_modes": sorted({
                    str(group.get("launch_mode", ""))
                    for group in native_groups if group.get("launch_mode")}),
                "protocol_operators": len(protocol),
                "directed_edges": directed,
                "directed_frames": sum(int(group.get("directed_frames", 0))
                                       for group in native_groups),
                "transport_tx_bytes": sum(float(group.get(
                    "transport_tx_bytes", 0.0)) for group in native_groups),
                "transport_rx_bytes": sum(float(group.get(
                    "transport_rx_bytes", 0.0)) for group in native_groups),
                "transport_frames": sum(int(group.get(
                    "transport_frames", 0)) for group in native_groups),
                "transport_edges": sum(int(group.get(
                    "transport_edges", 0)) for group in native_groups),
                "total_edges": sum(int(group.get(
                    "total_edges", 0)) for group in native_groups),
                "payload_edges": sum(int(group.get(
                    "payload_edges", 0)) for group in native_groups),
                "count_edges": sum(int(group.get(
                    "count_edges", 0)) for group in native_groups),
                "operators": native_groups,
            }
        result.update(
            result_mode="compact", profile=profile,
            event_profile=_bounded_event_profile(engine),
            timeline_omitted=True, events_omitted=True,
        )
    else:
        # Keep the historical detailed key order byte-for-byte.
        result = {
            **head, "timeline": rows, "resources": resources,
            "execution": execution_summary, "parameters": parameter_summary,
            "events": list(engine.trace),
        }
    if isolated:
        path = _precedence_bound(records if compact else rows,
                                 external_times, input_ready_ms)
        capacity_bound = max((row["capacity_bound_ms"] for row in resources), default=0)
        result["isolated"] = {
            "boundary": "sum_of_isolated_device_programs",
            "operator_sum_ms": isolated_sum,
            **({"by_family": isolated_profile["by_family"],
                "by_operator": isolated_profile["by_operator"],
                "operators_omitted": True}
               if compact else {"operators": costs}),
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
