"""Joint collective cohorts, native peer launches, and trace replay helpers.

[Role]
  Bind cross-DP EP/local-TP cohorts, launch joint collective and native peer
  programs onto the shared ``Engine``, record bounded joint-collective
  observations, and manage isolated private schedule trace replay.

[Invariants]
  * All participants in a cross-DP cohort bind before any collective event
    is released; mismatches raise explicit ``ModelingError`` / ``ValueError``.
  * Joint-collective observations copy numeric timestamps on completion and
    never retain live ``Event`` handles.

[Timeouts & Locks]
  Bounded per-execution trace caches (``ISOLATED_SCRIPT_CACHE_LIMIT``,
  ``NATIVE_GROUP_SUMMARY_LIMIT``); no I/O or locks.

[Side-Effect Policy]
  Submits tasks to the caller's Engine and mutates only bounded trace state.
"""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from dataclasses import dataclass, replace
from typing import Any

from mindie_sim.schema.common import ModelingError
from ..cohort_traffic import bind_cohort_traffic
from ..collectives import plan_options as collective_options
from ..engine import Event
from ..program import RecipeContext
from .cohort_launch import _RankEngine, _bind_native_resource_view, launch_native_cohort
from .cohort_native import _CohortNativeMixin
from .streams import _collective_native_basis

ISOLATED_SCRIPT_CACHE_LIMIT = 1024


@dataclass(frozen=True)
class CollectiveParticipant:
    replica: int
    ready: Event
    context: RecipeContext
    recipe: Any
    name: str
    completion: Event


class _CohortMixin(_CohortNativeMixin):
    """Cross-DP cohort binding, native launches, and isolated replay traces."""

    def add_group(self, works, release_ms, *, external_inputs=None,
                  emitted_tokens=None, memory_bindings=None,
                  state_accesses=None):
        """Compile an EP/local-TP cohort before releasing any collective events."""
        graphs = bind_cohort_traffic(self.case, {r: work.ops for r, work in works.items()})
        from ..local_projection_tp import lower_local_projection_tp_cohort
        graphs = lower_local_projection_tp_cohort(
            graphs, model=self.case.model, runtime=self.case.runtime,
            machine=self.case.machine)
        works = {r: replace(work, ops=graphs[r]) for r, work in works.items()}
        self._collectives = {}
        self._cohort_views = {
            int(rank): work.plan.requests
            for r, work in works.items()
            if work.plan.requests or work.mode != "graph"
            for stage in range(max(1, int(self.case.runtime.parallel.pp)))
            for rank in self.topology.ranks(r, stage)
        }

        def per_replica(value, replica):
            if isinstance(value, _AbcMapping) and replica in value:
                return value[replica]
            return value

        steps = {replica: self.add(
            work, replica, release_ms,
            external_inputs=(external_inputs or {}).get(replica),
            emitted_tokens=(emitted_tokens or {}).get(replica),
            memory_bindings=per_replica(memory_bindings, replica),
            state_accesses=per_replica(state_accesses, replica),
        ) for replica, work in works.items()}
        for key, participants in self._collectives.items():
            expected_count = len(works)
            first_meta = dict(participants[0].context.op.metadata)
            if first_meta.get("group_axis") == "local_tp" and first_meta.get("group_ranks"):
                src_stage = int(first_meta.get("source_stage", participants[0].context.op.stage))
                group_set = {int(rk) for rk in first_meta["group_ranks"]}
                expected_count = sum(
                    1 for r in works
                    if set(self.topology.ranks(r, src_stage)) & group_set)
            if len(participants) != expected_count:
                raise ValueError(f"collective_participants_mismatch:{key}")
            if any(getattr(item.recipe, "algorithm", None) for item in participants):
                self._start_joint_collective(key, participants)
                continue
            native = any(getattr(item.recipe, "native", False)
                         for item in participants)
            if native:
                self._schedule_native_cohort(key, participants)
                continue
            if all(item.context.op.family == "mc2_transport"
                   for item in participants):
                programs = {item.replica: self.lower_program(
                    item.context, item.recipe) for item in participants}
                from ..program import NativeProgram
                if all(isinstance(program, NativeProgram)
                       and hasattr(program.program, "start_with_handles")
                       for program in programs.values()):
                    if self._start_native_cohort(key, participants, programs):
                        continue
            source_stage = int(dict(participants[0].context.op.metadata).get(
                "source_stage", participants[0].context.op.stage))
            local_ready_by_rank = {}
            for item in participants:
                local_ready_by_rank.update({
                    rank: item.ready for rank in
                    self.topology.ranks(item.replica, source_stage)})
            observation = self._begin_joint_observation(
                participants[0].context.op, key, participants,
                local_ready_by_rank, source_stage)
            dependencies = tuple(item.ready for item in participants)
            ready = self.engine.submit(f"group:{key}:ready", after=dependencies, owner="expert_group")
            completions, promises = [], []
            for item in participants:
                completions.append(self._start_program(
                    item.context, item.recipe, item.name, ready,
                    owner="expert_group", replica=item.replica))
                promises.append(item.completion)

            def release(event, events=tuple(promises), observation=observation,
                        local_ready_by_rank=local_ready_by_rank,
                        completions=tuple(completions), ready=ready,
                        participants=tuple(participants),
                        source_stage=source_stage):
                if observation is not None:
                    observation["barrier_ms"] = ready.time
                    observation["participants"] = []
                    for item, completion in zip(participants, completions):
                        ranks = self.topology.ranks(item.replica, source_stage)
                        local = {rank: local_ready_by_rank[rank].time
                                 for rank in ranks}
                        done = {rank: completion.time for rank in ranks}
                        observation["participants"].append({
                            "replica": item.replica, "ranks": list(ranks),
                            "ready_ms": local, "completion_ms": done,
                            "wait_ms": {rank: ready.time - local[rank]
                                        for rank in ranks},
                            "parent_complete_ms": event.time,
                        })
                        observation["ready_ms_by_rank"].update(local)
                        observation["completion_ms_by_rank"].update(done)
                        observation["parent_complete_ms"][item.replica] = event.time
                    self._finish_joint_collective(observation)
                for item in events:
                    self.engine.resolve(item)

            self.engine.submit(f"group:{key}", after=completions,
                               on_complete=release, owner="expert_group")
        self._collectives = None
        self._cohort_views = None
        return steps

    def _begin_joint_observation(self, op, key, participants, ready_by_rank,
                                 source_stage):
        """Allocate one bounded record for an actual cohort rendezvous."""
        if self._joint_collective_trace is None:
            return None
        collective, _expected_bytes, expected_dtype = _collective_native_basis(
            op, len(ready_by_rank))
        axis = dict(op.metadata).get("group_axis", "tp")
        occurrence = self._joint_collective_ordinals.get((op.family, axis), 0)
        self._joint_collective_ordinals[(op.family, axis)] = occurrence + 1
        link_by_rank, native_by_rank = {}, {}
        for item in participants:
            ranks = self.topology.ranks(item.replica, source_stage)
            _, native, _ = _collective_native_basis(
                item.context.op, len(ready_by_rank))
            value = (int(native) if native is not None
                     and native.denominator == 1 else None)
            link_value = item.context.native_work.get(
                "link", item.context.op.link_bytes)
            for rank in ranks:
                link_by_rank[rank] = link_value
                native_by_rank[rank] = value
        native_total = (sum(native_by_rank.values())
                        if native_by_rank
                        and all(value is not None
                                for value in native_by_rank.values())
                        else None)
        return {
            "op_id": op.op_id, "group": key,
            "family": op.family, "group_axis": axis,
            "collective": collective, "dtype": expected_dtype,
            "native_input_bytes": native_total,
            "native_input_bytes_by_rank": native_by_rank,
            "link_bytes": sum(link_by_rank.values()),
            "link_bytes_by_rank": link_by_rank,
            "source_stage": source_stage,
            "rank_count": len(ready_by_rank), "occurrence": occurrence,
            "participants": [], "ready_ms_by_rank": {},
            "completion_ms_by_rank": {}, "parent_complete_ms": {},
        }

    def _start_joint_collective(self, key, participants):
        """Bind one physical cohort while retaining each replica's readiness."""
        first = participants[0]
        algorithm = getattr(first.recipe, "algorithm", None)
        plan_options = collective_options(first.recipe, first.context.op)
        path_model = getattr(first.recipe, "path_model", None)
        op = first.context.op

        def declaration(item):
            return (
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
        participants_vol = {
            item.replica: item.context.native_work.get("link", 0)
            for item in participants
        }
        plan = (
            self.topology.band_collective_plan(
                op, participants_vol, algorithm=algorithm,
            )
            if (algorithm in {"ring", "pairwise"} and not plan_options)
            else None
        )
        if plan is None:
            plan = self.topology.joint_plan(
                op, participants_vol, algorithm=algorithm, **plan_options)
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
        if self.readiness_report:
            from ..collective_readiness import bind_pipeline_ready
            try:
                collective, expected_bytes, expected_dtype = _collective_native_basis(
                    op, len(ready_by_rank))
                context = bind_pipeline_ready(
                    self.readiness_report, ready_by_rank,
                    expected_collective=collective,
                    expected_bytes=(int(expected_bytes)
                                    if expected_bytes is not None
                                    and expected_bytes.denominator == 1
                                    else None),
                    expected_dtype=expected_dtype,
                    expected_group_size=len(ready_by_rank),
                    connection_id=dict(op.metadata).get("collective_connection_id"),
                    ordinal=dict(op.metadata).get("collective_call_ordinal"),
                    call_id=dict(op.metadata).get("collective_call_id"))
            except ModelingError as exc:
                self.readiness_unresolved.append({
                    "op_id": op.op_id, "group": key, "reason": str(exc)})
            else:
                self.readiness_observations.append({
                    "op_id": op.op_id, "group": key,
                    "ready_by_rank": tuple(sorted(ready_by_rank)),
                    "context": context,
                })
        completions = plan.start_participants(
            self.engine, first.name + ":cohort", ready_by_rank, owner="expert_group")
        observation = self._begin_joint_observation(
            op, key, participants, ready_by_rank, source_stage)
        pending = [len(participants)] if observation is not None else None
        for item in participants:
            ranks = self.topology.ranks(item.replica, source_stage)
            dependencies = [completions[rank] for rank in ranks]
            timing = item.context.invocation_timing
            minimum = max(item.context.minimum_ms, timing.minimum_ms if timing else 0)
            if minimum:
                dependencies.append(self.engine.submit(
                    item.name + ":observation", after=(item.ready,),
                    latency=minimum, owner=f"replica:{item.replica}"))

            def cohort_complete(event, item=item, ranks=ranks,
                                observation=observation, pending=pending):
                if observation is not None:
                    ready = {rank: ready_by_rank[rank].time for rank in ranks}
                    completion = {rank: completions[rank].time for rank in ranks}
                    observation["participants"].append({
                        "replica": item.replica, "ranks": list(ranks),
                        "ready_ms": ready, "completion_ms": completion,
                        "parent_complete_ms": event.time,
                    })
                    observation["ready_ms_by_rank"].update(ready)
                    observation["completion_ms_by_rank"].update(completion)
                    observation["parent_complete_ms"][item.replica] = event.time
                    pending[0] -= 1
                    if pending[0] == 0:
                        self._finish_joint_collective(observation)
                self.engine.resolve(item.completion)

            self.engine.submit(
                item.name + ":cohort_complete", after=dependencies,
                owner=f"replica:{item.replica}", on_complete=cohort_complete)
        return completions

    def _finish_joint_collective(self, observation: dict) -> None:
        """Reduce one completed cohort to bounded numeric observations."""
        observation["participants"].sort(key=lambda item: item["replica"])
        ready = [value for value in observation["ready_ms_by_rank"].values()
                 if value is not None]
        completion = [value for value in observation["completion_ms_by_rank"].values()
                      if value is not None]
        if ready:
            observation["rendezvous_ready_ms"] = observation.get(
                "barrier_ms", max(ready))
            observation["ready_spread_ms"] = max(ready) - min(ready)
            observation["critical_rank"] = max(
                observation["ready_ms_by_rank"],
                key=lambda rank: (observation["ready_ms_by_rank"][rank]
                                  if observation["ready_ms_by_rank"][rank] is not None
                                  else float("-inf")))
        if completion:
            observation["last_completion_ms"] = max(completion)
        from ..participant_profile import select_call
        observation["source"] = select_call(
            self.participant_profile, primitive=observation["collective"],
            occurrence=observation["occurrence"],
            group_size=observation["rank_count"], dtype=observation["dtype"])
        stats = self.joint_collective_stats
        stats["observations"] += 1
        spread = observation.get("ready_spread_ms") or 0.0
        if spread > stats["max_ready_spread_ms"]:
            stats["max_ready_spread_ms"] = spread
            stats["max_rendezvous_delay_ms"] = spread
            stats["critical_rank"] = observation.get("critical_rank")
        trace = self._joint_collective_trace
        trace_key = (observation["op_id"], observation["group"])
        if trace_key in trace:
            trace[trace_key]["occurrences"] += 1
            trace[trace_key]["last_ready_spread_ms"] = spread
            trace[trace_key]["last_parent_complete_ms"] = max(
                observation["parent_complete_ms"].values(), default=None)
        else:
            observation["occurrences"] = 1
            if len(trace) >= self._joint_collective_limit:
                trace.pop(next(iter(trace)))
            trace[trace_key] = observation
        stats["retained"] = len(trace)

    def _outside_trace(self, engine) -> bool:
        prefix = getattr(engine, "_trace_prefix", None)
        if prefix is None:
            return False
        return any(
            (task.remaining or task.allocation_demands)
            and not task.event.name.startswith(prefix)
            for task in engine._tasks.values()
        )

    def _join_isolated_replay(self, engine, script_key, cached, token, complete) -> bool:
        steps, peaks = cached
        pending = engine._script_pending
        if pending is None:
            if not engine.replay_ready():
                return False
            engine.queue_script(script_key, steps, [(token, complete)], peaks)
            return True
        if engine._script_pending_key != script_key:
            return False
        pending[1].append((token, complete))
        return True

    def _seal_trace(self, engine, complete) -> None:
        key = self._trace_key
        shadows = self._trace_shadows
        self._trace_key = None
        self._trace_shadows = []
        if getattr(engine, "_trace_prefix", None) is None:
            engine.resolve(complete)
            return
        clean, steps, peaks = engine.end_trace()
        scripts = type(self)._isolated_scripts
        if key is not None and clean:
            scripts[key] = (steps, peaks)
            if len(scripts) > ISOLATED_SCRIPT_CACHE_LIMIT:
                scripts.pop(next(iter(scripts)))
            for token, shadow in shadows:
                self._apply_private_account(engine, steps, peaks, token)
                engine.resolve(shadow)
        else:
            for _token, shadow in shadows:
                engine.resolve(shadow)
        engine.resolve(complete)

    def _apply_private_account(self, engine, steps, peaks, token) -> None:
        for dt, busy, adds in steps:
            for name in busy:
                concrete = name.replace("@r*", token, 1)
                if concrete in engine.busy_ms:
                    engine.busy_ms[concrete] += dt
            for name, used in adds:
                concrete = name.replace("@r*", token, 1)
                if concrete in engine.serviced:
                    engine.serviced[concrete] += used
        for pool, unit, value in peaks:
            concrete = pool.replace("@r*", token, 1)
            if concrete not in engine.peak_allocated:
                continue
            pair = concrete, unit
            if value > engine.peak_unit_allocated.get(pair, 0.0):
                engine.peak_unit_allocated[pair] = value
            if value > engine.peak_allocated[concrete]:
                engine.peak_allocated[concrete] = value

    def _schedule_native_cohort(self, key, participants):
        started = [False]

        def begin(_event):
            if started[0]:
                return
            started[0] = True
            programs = {item.replica: self.lower_program(
                item.context, item.recipe) for item in participants}
            from ..program import NativeProgram
            if not all(isinstance(program, NativeProgram)
                       and hasattr(program.program, "start_with_handles")
                       for program in programs.values()):
                raise ModelingError(
                    f"native_peer_plan_requires_executable_cohort:{key}")
            if not self._start_native_cohort(key, participants, programs,
                                             defer=True):
                raise ModelingError(
                    f"native_peer_plan_requires_executable_cohort:{key}")

        for item in participants:
            self.engine.submit(
                f"group:{key}:native_wait:r{item.replica}",
                after=(item.ready,), owner="expert_group",
                on_complete=begin)
        return True

    def _start_native_cohort(self, key, participants, programs, *, defer=False):
        """Two-phase native launch for one ordinary EP cohort."""
        source_stage = int(dict(participants[0].context.op.metadata).get(
            "source_stage", participants[0].context.op.stage))
        launches = []
        for item in participants:
            program = programs[item.replica]
            rank = self.participant_rank_index(item.replica, source_stage)
            if rank is None:
                return False
            launches.append({
                "program": program,
                "name": f"group:{key}:r{item.replica}",
                "ready": item.ready,
                "rank_index": rank,
                "peer_count": int(getattr(program.program, "peer_count", 0)),
                "node_ranks": int(getattr(self.case.machine, "ranks_per_node", 0)),
                "resource_names": self.native_resource_names(
                    item.context, program, rank=rank, replica=item.replica),
                "transport": self.native_transport(item.context.op, item.replica),
            })
        result = launch_native_cohort(
            self.engine, launches, name_prefix=f"group:{key}",
            owner="expert_group", defer=defer)
        for item in participants:
            name = f"group:{key}:r{item.replica}"
            parent = result["parents"][name]
            self.engine.submit(
                name + ":release", after=(parent,),
                owner=f"replica:{item.replica}",
                on_complete=lambda _event, promise=item.completion:
                self.engine.resolve(promise))
        self.native_group_joins[key] = result["join"]
        raw = dict(participants[0].context.op.metadata).get("mc2_phase", {})
        payload_edges = sum(
            int(phases.get("payload_exchange", {}).get("edges", 0))
            for phases in result["transport_by_rank"].values())
        count_edges = sum(
            int(phases.get("count_status_exchange", {}).get("edges", 0))
            for phases in result["transport_by_rank"].values())
        payload_frames = sum(
            int(phases.get("payload_exchange", {}).get("frames", 0))
            for phases in result["transport_by_rank"].values())
        merged_work = {}
        for name, work in result["work_by_rank"].items():
            merged = {pipe: amount for pipe, amount in work
                      if "link" not in str(pipe).lower()}
            for pipe, amount in result["transport_work_by_rank"].get(
                    name, {}).items():
                merged[pipe] = merged.get(pipe, 0.0) + float(amount)
            merged_work[name] = tuple(sorted(merged.items()))
        self.native_group_launches[key] = {
            "op_id": key,
            "kind": str(raw.get("kind", "")),
            "launch_mode": "cohort",
            "ranks": tuple(sorted(result["rank_indices"].values())),
            "payload_basis": dict(result["payload_basis"]),
            "payload_packet_bytes": dict(result["payload_packet_bytes"]),
            "payload_frames": dict(result["payload_frames"]),
            "link_by_phase_by_rank": dict(result["link_by_phase_by_rank"]),
            "work_by_rank": merged_work,
            "transport_by_rank": {
                name: dict(phases)
                for name, phases in result["transport_by_rank"].items()},
            "transport_work_by_rank": {
                name: dict(work)
                for name, work in result["transport_work_by_rank"].items()},
            "transport_tx_bytes": float(result["transport_tx_bytes"]),
            "transport_rx_bytes": float(result["transport_rx_bytes"]),
            "transport_frames": int(result["transport_frames"]),
            "transport_edges": payload_edges,
            "total_edges": int(result["transport_edges"]),
            "payload_edges": payload_edges,
            "count_edges": count_edges,
            "directed_edges": (payload_edges or int(result["directed_edges"])),
            "directed_frames": (payload_frames or int(result["directed_frames"])),
            "protocol_edges": int(result["protocol_edges"]),
            "unresolved_edges": int(result["unresolved_edges"]),
            "partial": tuple(result["partial"]),
        }
        return True
