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

        from ..runtime.dp_cohort_dedup import is_symmetric_dp_cohort
        if is_symmetric_dp_cohort(
            self, works, external_inputs=external_inputs,
            emitted_tokens=emitted_tokens, memory_bindings=memory_bindings,
            state_accesses=state_accesses,
        ):
            return self._add_symmetric_group(
                works, release_ms, external_inputs=external_inputs,
            )

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
                               on_complete=release, owner="expert_group",
                               native_hook=True)
        self._collectives = None
        self._cohort_views = None
        return steps

    def _collective_peer_replicas(self, item0, replicas):
        """Exact participating DP replicas for one canonical collective op."""
        meta = dict(item0.context.op.metadata)
        axis = meta.get("group_axis")
        src_stage = int(meta.get("source_stage", item0.context.op.stage))
        if axis == "local_tp" and meta.get("group_ranks"):
            group_set = {int(rk) for rk in meta["group_ranks"]}
            return tuple(
                r for r in replicas
                if set(self.topology.ranks(r, src_stage)) & group_set
            )
        if axis in {"cp", "dcp_dp", "dcp_a2a"}:
            groups = self.topology.groups(axis, item0.replica, src_stage, metadata=meta)
            rank_set = {int(rk) for grp in groups for rk in grp}
            if rank_set:
                matched = tuple(
                    r for r in replicas
                    if any(int(rk) in rank_set for rk in self.topology.ranks(r, src_stage))
                )
                if matched:
                    return matched
        if axis == "ep":
            from ..dispatch import cohorts
            for grp in cohorts(self.case.runtime.parallel, self.case.machine):
                if item0.replica in grp:
                    return tuple(r for r in replicas if r in grp)
        return replicas

    def _synthesize_cohort_participants(self, item0, works, peer_replicas):
        """Build full-cohort CollectiveParticipants sharing canonical ready."""
        op_id = item0.context.op.op_id
        out = []
        for rep in peer_replicas:
            if rep == item0.replica:
                out.append(item0)
                continue
            wr = works[rep]
            op_r = next((op for op in wr.ops if op.op_id == op_id), item0.context.op)
            ctx_r = self._context(op_r, rep, wr.replay_scope)
            name_r = f"{self.namespace}r{rep}:i{wr.plan.iteration}:{op_r.op_id}"
            comp_r = self.engine.future(name_r + ":collective")
            out.append(CollectiveParticipant(
                rep, item0.ready, ctx_r, item0.recipe, name_r, comp_r,
            ))
        return out

    def _add_symmetric_group(self, works, release_ms, *, external_inputs=None):
        """Execute one canonical replica for a provably symmetric DP cohort."""
        from ..runtime.dp_cohort_dedup import (
            can_dedup_collective_to_canonical,
            is_symmetric_native_program,
            mirror_deduplicated_step,
            scale_dedup_accounting,
        )
        replicas = tuple(sorted(works))
        r0 = replicas[0]
        peers = replicas[1:]
        base_work = dict(self.work)
        base_recipes = dict(self.recipes_used)
        base_cost_counts = dict(self.cost.counts)

        step0 = self.add(
            works[r0], r0, release_ms,
            external_inputs=(external_inputs or {}).get(r0),
        )
        can_r0_only = can_dedup_collective_to_canonical(self.topology, self.engine)
        used_r0_only_links = False

        for key, participants in self._collectives.items():
            item0 = participants[0]
            peer_replicas = self._collective_peer_replicas(item0, replicas)
            if getattr(item0.recipe, "algorithm", None):
                synth = self._synthesize_cohort_participants(item0, works, peer_replicas)
                self._start_joint_collective(key, synth)
                continue
            if getattr(item0.recipe, "native", False):
                if can_r0_only and self._schedule_canonical_native_cohort(
                    key, item0, peer_replicas, is_symmetric_native_program,
                ):
                    used_r0_only_links = True
                    continue
                synth = self._synthesize_cohort_participants(item0, works, peer_replicas)
                self._schedule_native_cohort(key, synth)
                continue
            if item0.context.op.family == "mc2_transport":
                prog0 = self.lower_program(item0.context, item0.recipe)
                from ..program import NativeProgram
                if (isinstance(prog0, NativeProgram)
                        and hasattr(prog0.program, "start_with_handles")):
                    if can_r0_only and is_symmetric_native_program(prog0, len(peer_replicas)):
                        if self._start_canonical_native_cohort(
                            key, item0, prog0, peer_replicas,
                        ):
                            used_r0_only_links = True
                            continue
                    synth = self._synthesize_cohort_participants(item0, works, peer_replicas)
                    progs = {
                        p.replica: (prog0 if p.replica == r0
                                    else self.lower_program(p.context, p.recipe))
                        for p in synth
                    }
                    if self._start_native_cohort(key, synth, progs):
                        continue
            if can_r0_only:
                comp0 = self._start_program(
                    item0.context, item0.recipe, item0.name, item0.ready,
                    owner="expert_group", replica=item0.replica,
                )
                self.engine.on_complete(
                    comp0,
                    lambda _ev, promise=item0.completion: self.engine.resolve(promise),
                    native_hook=True,
                )
                used_r0_only_links = True
            else:
                synth = self._synthesize_cohort_participants(item0, works, peer_replicas)
                ready = self.engine.submit(
                    f"group:{key}:ready", after=(item0.ready,), owner="expert_group",
                )
                completions = [
                    self._start_program(
                        p.context, p.recipe, p.name, ready,
                        owner="expert_group", replica=p.replica,
                    )
                    for p in synth
                ]
                promises = tuple(p.completion for p in synth)

                def _release(_ev, events=promises):
                    for ev in events:
                        self.engine.resolve(ev)

                self.engine.submit(
                    f"group:{key}", after=completions,
                    on_complete=_release, owner="expert_group", native_hook=True,
                )

        scale_dedup_accounting(
            self, base_work, base_recipes, base_cost_counts, len(replicas),
        )
        dedup_list = getattr(self, "_dedup_cohorts", None)
        if dedup_list is None:
            dedup_list = []
            self._dedup_cohorts = dedup_list
        dedup_list.append({
            "canonical": r0,
            "peers": peers,
            "mirror_links": used_r0_only_links,
        })
        self._collectives = None
        self._cohort_views = None
        steps = {r0: step0}
        for rep in peers:
            steps[rep] = mirror_deduplicated_step(step0, rep, works[rep])
        return steps

    def _schedule_canonical_native_cohort(
        self, key, item0, peer_replicas, is_symmetric_fn,
    ) -> bool:
        """Defer-launch one canonical native MC2 rank when routing is symmetric."""
        prog0 = self.lower_program(item0.context, item0.recipe)
        from ..program import NativeProgram
        if not (isinstance(prog0, NativeProgram)
                and is_symmetric_fn(prog0, len(peer_replicas))):
            return False
        return self._start_canonical_native_cohort(
            key, item0, prog0, peer_replicas, defer=True,
        )

    def _start_canonical_native_cohort(
        self, key, item0, prog0, peer_replicas, *, defer: bool = False,
    ) -> bool:
        """Launch canonical rank-0 MC2 program with full-cohort peer fan-out."""
        source_stage = int(dict(item0.context.op.metadata).get(
            "source_stage", item0.context.op.stage))
        rank0 = self.participant_rank_index(item0.replica, source_stage)
        if rank0 is None:
            return False
        inner = prog0.program
        peer_count = int(getattr(inner, "peer_count", 0) or len(peer_replicas))
        if peer_count <= 0 or not 0 <= rank0 < peer_count:
            return False
        res_names = self.native_resource_names(
            item0.context, prog0, rank=rank0, replica=item0.replica)
        transport = dict(self.native_transport(item0.context.op, item0.replica))
        r0_ranks = tuple(self.topology.ranks(item0.replica, source_stage))
        if r0_ranks and transport.get("destination_memory"):
            r0_mems = tuple(
                self.destination_window_resource(int(rk), source_stage)
                for rk in r0_ranks
            )
            group_ranks = tuple(transport.get("group") or ())
            if group_ranks and any(r0_mems):
                transport["destination_memory"] = {
                    int(rk): r0_mems[idx % len(r0_mems)]
                    for idx, rk in enumerate(group_ranks)
                }
        bound = _bind_native_resource_view(prog0, res_names, self.engine)
        source_view = getattr(bound.program, "view", None)
        if source_view is None:
            return False
        labels = tuple(i for i in range(peer_count) if i != rank0)
        node = int(getattr(self.case.machine, "ranks_per_node", 0))
        frames_for = getattr(bound.program, "frames_for", None)
        frames_raw = frames_for(rank0) if callable(frames_for) else getattr(source_view, "payload_frames", None)
        if isinstance(frames_raw, _AbcMapping):
            p_frames = tuple(sorted((int(p), int(f)) for p, f in frames_raw.items() if int(f) > 0))
        elif frames_raw:
            p_frames = tuple(sorted((int(x[0]), int(x[1])) for x in frames_raw if int(x[1]) > 0))
        else:
            p_frames = ()
        basis_for = getattr(bound.program, "payload_basis_for", None)
        p_basis = str(basis_for(rank0) if callable(basis_for)
                      else getattr(source_view, "payload_basis", "unknown"))
        local_for = getattr(bound.program, "local_frames_for", None)
        p_local = int((local_for(rank0) if callable(local_for)
                       else getattr(source_view, "payload_local_frames", 0)) or 0)
        view = replace(
            source_view,
            rank=rank0, peer_labels=labels,
            peer_classes=tuple(getattr(source_view, "peer_classes", ()) if source_view.rank == rank0 else ()),
            peer_weights=tuple(getattr(source_view, "peer_weights", ()) if source_view.rank == rank0 else ()),
            node_ranks=node,
            intra_latency_ms=getattr(source_view, "intra_latency_ms", 0),
            inter_latency_ms=getattr(source_view, "inter_latency_ms", 0),
            resource_names=tuple(sorted((str(k), str(v)) for k, v in dict(res_names).items())),
            payload_frames=p_frames,
            payload_packet_bytes=int(getattr(source_view, "payload_packet_bytes", 0) or 0),
            payload_basis=p_basis,
            payload_local_frames=p_local,
        )
        name0 = f"group:{key}:r{item0.replica}"
        phase_names = tuple(p.name for p in getattr(bound.program, "phases", ()))

        def _launch(_event=None):
            placeholders = {}
            owned_futures = []
            for pname in phase_names:
                for peer in labels:
                    ev_p = self.engine.future(
                        f"{name0}:sym_send:{rank0}->{peer}:{pname}")
                    owned_futures.append(ev_p)
                    placeholders[(rank0, peer, pname)] = ev_p
                    placeholders[(peer, rank0, pname)] = ev_p
            rank_eng = _RankEngine(self.engine, res_names)
            finish, _handles = bound.start_with_handles(
                rank_eng, name0, item0.ready,
                launch=view, send_placeholders=placeholders, transport=transport,
            )

            def _complete_canonical(_e, promise=item0.completion, pending=tuple(owned_futures)):
                for ev_p in pending:
                    if not ev_p.done:
                        self.engine.resolve(ev_p)
                self.engine.resolve(promise)

            self.engine.on_complete(finish, _complete_canonical, native_hook=True)

        if defer:
            self.engine.submit(
                name0 + ":begin", after=(item0.ready,),
                owner="expert_group", on_complete=_launch, native_hook=True,
            )
        else:
            _launch()

        payload, count = (bound.program.transport_account(transport)
                          if hasattr(bound.program, "transport_account") else ({}, {}))
        n_peers = len(peer_replicas)
        phases_set = {p.name for p in getattr(bound.program, "phases", ())}
        p_edges = int(payload.get("edges", 0)) * n_peers
        c_edges = (int(count.get("edges", 0)) * n_peers) if "count_status_exchange" in phases_set else 0
        p_frames_tot = int(payload.get("frames", 0)) * n_peers
        c_frames_tot = (int(count.get("frames", 0)) * n_peers) if "count_status_exchange" in phases_set else 0
        tx_b = (float(payload.get("tx_bytes", 0.0)) + (float(count.get("tx_bytes", 0.0)) if "count_status_exchange" in phases_set else 0.0)) * n_peers
        rx_b = (float(payload.get("rx_bytes", 0.0)) + (float(count.get("rx_bytes", 0.0)) if "count_status_exchange" in phases_set else 0.0)) * n_peers
        raw = dict(item0.context.op.metadata).get("mc2_phase", {})
        self.native_group_launches[key] = {
            "op_id": key,
            "kind": str(raw.get("kind", "")),
            "launch_mode": "cohort",
            "ranks": tuple(range(n_peers)),
            "payload_basis": {f"group:{key}:r{r}": p_basis for r in peer_replicas},
            "payload_packet_bytes": {f"group:{key}:r{r}": view.payload_packet_bytes for r in peer_replicas},
            "payload_frames": {f"group:{key}:r{r}": dict(p_frames) for r in peer_replicas},
            "link_by_phase_by_rank": {},
            "work_by_rank": {},
            "transport_by_rank": {},
            "transport_work_by_rank": {},
            "transport_tx_bytes": tx_b,
            "transport_rx_bytes": rx_b,
            "transport_frames": p_frames_tot + c_frames_tot,
            "transport_edges": p_edges,
            "total_edges": p_edges + c_edges,
            "payload_edges": p_edges,
            "count_edges": c_edges,
            "directed_edges": p_edges,
            "directed_frames": p_frames_tot,
            "protocol_edges": 0,
            "unresolved_edges": 0,
            "partial": (),
        }
        return True

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
                owner=f"replica:{item.replica}", on_complete=cohort_complete,
                native_hook=True)
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
                on_complete=begin, native_hook=True)
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
                owner=f"replica:{item.replica}", native_hook=True,
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
