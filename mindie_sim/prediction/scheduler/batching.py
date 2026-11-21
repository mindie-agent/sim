"""Batch admission and request scheduling.

[Role] Schedule bounded batches and commit request, cache and state changes.
[Invariants] Cache is reserved before work and token commits follow actual output.
[Timeouts & Locks] Work is bounded by active requests and configured batch budgets.
[Side-Effect Policy] Updates per-run request progress and the Memory ledger.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import RequestSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

from ..capacity import Memory
from .progress import DraftPolicy, Progress, running_first

_PREEMPTED = object()


def _live_ids(by_id: Mapping[str, Progress]) -> set[str]:
    """Initialize the live request IDs owned by one scheduler instance."""
    return set(by_id)


class Scheduler:
    def __init__(self, case: CaseSpec, requests: list[RequestSpec], memory: Memory, replica: int = 0,
                 order: Callable[[Progress], tuple] = running_first, eligible: Callable = lambda state: True,
                 handoff_after_prefill: bool = False):
        self.case, self.memory, self.replica, self.order = case, memory, replica, order
        self.eligible = eligible
        self.handoff_after_prefill = handoff_after_prefill
        self.states = [Progress(request) for request in requests]
        self.by_id = {state.request.request_id: state for state in self.states}
        # Bounded per-decision projection: completed requests leave this set.
        self._live_ids = _live_ids(self.by_id)
        self._infeasible_ids = set()
        self.drafts = {state.request.request_id: DraftPolicy.for_request(case, state.request)
                       for state in self.states}
        self.iteration = 0
        self.in_flight: tuple[RequestView, ...] = ()
        #: Plan iteration currently in flight; a stale callback cannot publish
        #: the next batch.
        self._in_flight_iteration: int | None = None
        self.blocked = False
        #: Why the last ``schedule`` call produced or withheld a batch.  It is
        #: feedback for an assembler, not a second scheduler state machine.
        self.last_reason = "not_scheduled"
        self.successors = {}
        tails = {}
        for state in self.states:
            chain = state.request.chain_id
            if chain:
                if chain in tails:
                    self.successors[tails[chain]] = state
                    state.release_ms = float("inf")
                tails[chain] = state.request.request_id

    @property
    def done(self) -> bool:
        return all(state.done or not self.eligible(state) for state in self.states)

    def next_arrival(self, now: float) -> float | None:
        return min((state.release_ms for state in self.states
                    if not state.done and self.eligible(state)
                    and now < state.release_ms < float("inf")), default=None)

    def schedule(self, now: float, *, defer_capacity_failure: bool = False) -> IterationPlan | None:
        if self.in_flight:
            raise ModelingError("scheduler_batch_already_in_flight")
        while True:
            result = self._schedule(now, defer_capacity_failure=defer_capacity_failure)
            if result is not _PREEMPTED:
                return result

    def _refresh_waiting_states(self) -> None:
        """Recompute the preemption hold flag from the same Memory state.

        A direct T04 ``Memory.release_async_consumer`` is the real last-read
        edge even when the scheduler wrapper was not the caller.  An old
        ``waiting_for_release`` flag must not survive that release, and a still
        held reader must keep the request out of the next batch.
        """
        for state in self.states:
            if not state.waiting_for_release:
                continue
            request = state.request.request_id
            held = False
            if hasattr(self.memory, "pending_async_consumers"):
                held = bool(self.memory.pending_async_consumers(
                    self.replica, request))
            if not held and hasattr(self.memory, "is_retired"):
                held = bool(self.memory.is_retired(self.replica, request))
            if not held:
                state.waiting_for_release = False

    def _minimum_request_tokens(self, state: Progress) -> int:
        """Smallest non-optimistic row count that can execute this request."""
        if (state.computed >= state.prompt_target
                or self.case.runtime.enable_chunked_prefill):
            return state.computed + 1
        return state.prompt_target

    def _minimum_request_fits(self, state: Progress) -> bool:
        """Whether reclaiming every live holder could ever fit this request.

        This uses the same Memory geometry as admission.  A request whose
        minimum KV span exceeds the mounted per-stage budget is permanently
        infeasible; no release callback can change that.
        """
        if not hasattr(self.memory, "cache_bytes"):
            return True
        try:
            needed = self.memory.cache_bytes(
                self._minimum_request_tokens(state))
            capacity = self.memory.capacity[self.replica]
        except (AttributeError, IndexError, TypeError, ValueError):
            return True
        return all(float(need) <= float(cap) + max(1e-9, abs(float(cap)) * 1e-12)
                   for need, cap in zip(needed, capacity))

    def _mark_permanent(self, state: Progress) -> None:
        state.reason = "insufficient_cache_or_token_budget"
        self._live_ids.discard(state.request.request_id)
        self._infeasible_ids.add(state.request.request_id)
        self.retire_state(state.request.request_id, force=True)
        successor = self.successors.get(state.request.request_id)
        while successor is not None:
            successor.reason = "closed_loop_predecessor_unserved"
            self._live_ids.discard(successor.request.request_id)
            self._infeasible_ids.add(successor.request.request_id)
            successor = self.successors.get(successor.request.request_id)

    def _schedule(self, now, *, defer_capacity_failure):
        runtime = self.case.runtime
        self._refresh_waiting_states()
        eligible = [state for state in self.states
                    if not state.done and self.eligible(state)
                    and state.release_ms <= now]
        arrived = sorted((state for state in eligible
                          if not state.waiting_for_release), key=self.order)
        if not arrived:
            if eligible and self._has_reclaimable_capacity(eligible):
                self.last_reason = "waiting_for_release"
            elif eligible and not any(self._minimum_request_fits(state)
                                      for state in eligible):
                for state in eligible:
                    self._mark_permanent(state)
                self.last_reason = "capacity_or_budget_exhausted"
            else:
                self.last_reason = "no_arrived_request"
            return None
        active = sum(state.started for state in arrived)
        budget = runtime.max_num_batched_tokens
        input_budget = runtime.max_num_input_tokens or budget
        views = []
        for state in arrived:
            if budget <= 0:
                break
            if not state.started and active >= runtime.max_num_seqs:
                continue
            decode = state.computed >= state.prompt_target
            draft = self.drafts[state.request.request_id]
            remaining = state.request.output_tokens - state.generated
            proposal = min(draft.proposal, max(0, remaining - 1), budget - 1) if decode else 0
            q = 1 if decode else min(state.prompt_target - state.computed, budget, input_budget)
            if not decode and not runtime.enable_chunked_prefill and q < state.prompt_target - state.computed:
                continue
            if q <= 0:
                continue
            allocation = q + proposal
            lookahead_steps = getattr(self.memory, "async_in_flight_lookahead_steps", 0)
            lookahead = (
                min(max(0, remaining - allocation), lookahead_steps * allocation)
                if decode and lookahead_steps > 0
                else 0
            )
            # The lookahead reservation is an admission optimization, not a
            # correctness requirement.  If the next-step headroom is missing,
            # execute the current batch with only the rows it really reserves;
            # do not turn a transient async reservation into an unserved request.
            if not self.memory.resize(
                    self.replica, state.request.request_id,
                    state.computed + allocation + lookahead):
                lookahead = 0
                if not self.memory.resize(
                        self.replica, state.request.request_id,
                        state.computed + allocation):
                    continue
            state.pending_context = state.computed + allocation
            state.reserved_tokens = allocation
            if not state.started:
                state.started = True
                state.admission_ms = now if state.admission_ms is None else state.admission_ms
                active += 1
            advance = draft.committed(state, proposal) if proposal else 1
            views.append(RequestView(
                request_id=state.request.request_id, q_len=q, context=state.computed,
                samples=decode or state.computed + q >= state.prompt_target,
                phase="decode" if decode else "prefill", prefix_tokens=state.request.prefix_tokens,
                prefix_cache_key=state.request.prefix_cache_key,
                priority=state.request.priority, media_encoded=state.media_encoded,
                draft_verify=bool(proposal), draft_proposal_tokens=proposal,
                draft_prepare=bool(draft.mode) and remaining > 2,
                draft_advanced_tokens=advance if decode else 0,
                draft_accepted_tokens=advance - 1 if decode else 0,
                prompt_tokens=state.request.prompt_tokens,
            ))
            budget -= allocation
            if not decode:
                input_budget -= q
        if not views:
            running = [state for state in arrived if state.started]
            if len(running) > 1 and not defer_capacity_failure:
                victim = running[-1]
                # Preemption revokes this request's logical demand.  A
                # still-running reader keeps its physical allocation until its
                # real completion; only the confirmed cancel/finish branch may
                # force-release the corresponding references.
                self.retire_state(victim.request.request_id, force=False)
                victim.recompute_to = victim.computed
                victim.computed = min(victim.request.prefix_tokens,
                                      max(0, victim.request.prompt_tokens - 1))
                victim.started = False
                victim.preemptions += 1
                victim.state_version += 1
                victim.pending_context = None
                victim.reserved_tokens = 0
                victim.waiting_for_release = bool(
                    self.memory.pending_async_consumers(
                        self.replica, victim.request.request_id)
                    or self.memory.is_retired(
                        self.replica, victim.request.request_id))
                self.last_reason = "preempted"
                return _PREEMPTED
            # A PD transfer, async consumer, delayed release, selected/shared
            # reservation or grant owns the same Memory until a real event.
            # That event can release capacity without evicting or failing a
            # waiting request.  A request whose minimum span already exceeds
            # the mounted budget is permanently infeasible regardless of any
            # currently reclaimable reference.
            permanent = [state for state in arrived
                         if not self._minimum_request_fits(state)]
            feasible = [state for state in arrived
                        if state not in permanent]
            if permanent:
                for state in permanent:
                    self._mark_permanent(state)
            if feasible and (defer_capacity_failure
                             or self.memory.has_delayed_releases(self.replica)
                             or self._has_reclaimable_capacity(feasible)):
                self.last_reason = "waiting_for_release"
                return None
            if not permanent:
                for state in feasible:
                    self._mark_permanent(state)
            # The loop has exhausted admission and preemption. This is a real
            # capacity/budget inability, never an absence-of-measurements gate.
            self.last_reason = "capacity_or_budget_exhausted"
            return None
        self.in_flight = tuple(views)
        self.last_reason = "scheduled"
        result = IterationPlan(tuple(views), iteration=self.iteration,
                               output_tokens=tuple((s.request.request_id, s.request.output_tokens) for s in arrived))
        self._in_flight_iteration = result.iteration
        self.iteration += 1
        return result

    def commit(self, at_ms: float, times: Mapping[str, float] | None = None) -> None:
        if not self.in_flight:
            return
        self.memory.step_delayed_releases(self.replica)
        for view in self.in_flight:
            state = self.by_id[view.request_id]
            state.commit(view, (times or {}).get(view.request_id, at_ms))
            if state.done:
                self._live_ids.discard(view.request_id)
                if self.memory.async_delayed_release_steps > 0 and not self.done:
                    self.memory.defer_release(self.replica, view.request_id)
                else:
                    self.retire_state(view.request_id)
                successor = self.successors.get(view.request_id)
                if successor is not None and not self.case.runtime.serving_io:
                    successor.release_ms = max(successor.request.arrival_ms,
                                               state.finish_ms + successor.request.gap_ms)
            else:
                if not self.memory.resize(self.replica, view.request_id, state.computed):
                    raise ModelingError(
                        "committed_kv_state_exceeds_reserved_capacity:"
                        + view.request_id)
        if self.done:
            self.memory.flush_delayed_releases(self.replica)
        self.in_flight = ()
        self._in_flight_iteration = None

    def logical_views(self, plan: IterationPlan) -> tuple[RequestView, ...]:
        """Real request rows in a plan; graph-padding lanes are not requests.

        A padding lane is physical graph work for an idle replica.  It has no
        Progress, commit, delivery or request KV.  Real but unregistered rows
        still fail loudly.
        """
        views = []
        for view in plan.requests:
            if getattr(view, "padding", False):
                continue
            if view.request_id not in self.by_id:
                raise ModelingError(
                    "scheduler_emission_request_not_registered:"
                    + view.request_id)
            views.append(view)
        return tuple(views)

    def emissions(self, plan: IterationPlan) -> dict[str, int]:
        """Tokens this plan will commit, derived from the same accept policy.

        An assembler passes this to ``Execution.add(emitted_tokens=...)``.  It
        is a read-only projection before ``commit``; commit applies the same
        integer counts and advances the state version once.  Only logical rows
        emit; a pure padding plan emits nothing and never touches Progress.
        """
        emitted = {}
        for view in self.logical_views(plan):
            state = self.by_id[view.request_id]
            if view.phase == "decode":
                if not view.draft_verify and int(view.q_len) != 1:
                    raise ModelingError(
                        "non_speculative_decode_forward_must_commit_one_token:"
                        + view.request_id)
                count = int(view.draft_advanced_tokens) if view.draft_verify else 1
            else:
                count = 1 if (view.samples and not state.generated) else 0
            count = min(count, state.request.output_tokens - state.generated)
            if count > 0:
                emitted[view.request_id] = count
        return emitted

    def commit_batch(self, step, at_ms: float | None = None) -> bool:
        """Commit only the batch that is actually in flight.

        ``step.commits`` carries per-request ``token_commit`` events; when a
        request has no declared commit node, its step completion time is used
        as the observation boundary.  A stale callback from an already
        published iteration cannot advance the next in-flight request; a
        repeated callback for the current iteration is a no-op.
        """
        if getattr(step, "mode", None) == "retire":
            completion = getattr(step, "completion", None)
            when = at_ms if at_ms is not None else getattr(completion, "time", None)
            if when is None:
                raise ModelingError("scheduled_batch_has_no_completion_event")
            if completion is not None and completion is getattr(self, "_last_retire_completion", None):
                return False
            self._last_retire_completion = completion
            if hasattr(self.memory, "has_delayed_releases") and self.memory.has_delayed_releases(self.replica):
                self.memory.step_delayed_releases(self.replica)
                return True
            return False
        if not self.in_flight or self._in_flight_iteration is None:
            return False
        iteration = getattr(step, "iteration", None)
        if iteration is not None:
            # PP microbatches carry plan_iteration = plan_iteration * count +
            # microbatch; the service still owns one scheduler transaction per
            # plan.  Accept only this plan's microbatch range, never a stale
            # callback from an earlier or later transaction.
            microbatches = max(1, min(
                int(getattr(self.case.runtime, "pipeline_microbatches", 1)),
                max(1, len(self.in_flight))))
            base = int(self._in_flight_iteration) * microbatches
            if not base <= int(iteration) < base + microbatches:
                return False
        completion = getattr(step, "completion", None)
        when = at_ms if at_ms is not None else getattr(completion, "time", None)
        if when is None:
            raise ModelingError("scheduled_batch_has_no_completion_event")
        times = {
            request: event.time
            for request, event in getattr(step, "commits", {}).items()
            if event is not None and event.time is not None
        }
        self.commit(when, times)
        return True

    def bind_execution(self, execution: Any) -> None:
        """Make Execution-owned offload readers use this exact Memory."""
        execution.memory = self.memory
        execution._offload_memory = self.memory

    def _has_reclaimable_capacity(self, states=None) -> bool:
        """True when a real T08 release/hold event can make room.

        This is a view of the same Memory instance, not a second capacity
        estimate.  It only names already-accounted grants/selected/shared
        reservations, delayed releases, and async consumer holds that can be
        released by their actual completion or cancellation.  A request whose
        own minimum span can never fit is not made feasible by any release.
        """
        if states:
            states = [state for state in states
                      if self._minimum_request_fits(state)]
            if not states:
                return False
        memory = self.memory
        if hasattr(memory, "has_delayed_releases") and memory.has_delayed_releases(self.replica):
            return True
        for attr in ("granted", "selected_reserved", "shared_reserved"):
            rows = getattr(memory, attr, None)
            if rows and any(float(value) > 0.0 for value in rows[self.replica]):
                return True
        holders = getattr(memory, "owners", {}) or {}
        for replica, request in holders:
            if replica != self.replica:
                continue
            if hasattr(memory, "pending_async_consumers"):
                if memory.pending_async_consumers(replica, request):
                    return True
            if hasattr(memory, "is_retired") and memory.is_retired(replica, request):
                return True
        return False

    def grant_state(self, grants, *, stage: int = 0) -> bool:
        """Atomically grant actual T08 state ranges in this replica budget."""
        return self.memory.grant_batch(self.replica, grants, stage=stage)

    def release_state_grants(self, names) -> float:
        return self.memory.release_grants(self.replica, names)

    def retain_shared_state(self, owner: str, stages, *, version: int = 0) -> bool:
        return self.memory.retain_shared_pages(
            self.replica, owner, stages, version=version)

    def release_shared_state(self, owner: str) -> bool:
        return self.memory.release_shared_pages(self.replica, owner)

    def hold_state_consumer(self, request: str, consumer: str, *,
                            storage: str | None = None, ranges=None,
                            generation: int | None = None, event=None) -> bool:
        return self.memory.hold_for_async_consumer(
            self.replica, request, consumer, storage=storage, ranges=ranges,
            generation=generation, event=event)

    def release_state_consumer(self, request: str, consumer: str) -> bool:
        released = self.memory.release_async_consumer(
            self.replica, request, consumer)
        if released and not self.memory.pending_async_consumers(
                self.replica, request) and not self.memory.is_retired(
                    self.replica, request):
            for state in self.states:
                if state.request.request_id == request:
                    state.waiting_for_release = False
        return released

    def retire_state(self, request: str, *, force: bool = False) -> None:
        """Retire a request through T08's async-hold-aware release edge."""
        if force and hasattr(self.memory, "force_release"):
            self.memory.force_release(self.replica, request)
            for state in self.states:
                if state.request.request_id == request:
                    state.waiting_for_release = False
        else:
            self.memory.release(self.replica, request)

    def _async_consumer_bindings(self, request: str) -> dict[str, dict]:
        holds = getattr(self.memory, "async_consumer_holds", {}).get(
            (self.replica, request), {})
        return {consumer: dict(snapshot.binding)
                for consumer, snapshot in holds.items()
                if getattr(snapshot, "binding", None)}

    def feedback(self, now: float | None = None, *, include_done: bool = False) -> dict:
        """Bounded state/capacity snapshot for the next assembly decision.

        It reports committed state, the in-flight transaction and free KV
        headroom.  It does not contain a token history and does not estimate a
        duration; an assembler uses it to choose the next batch or stop.
        The default is the bounded per-decision form (``include_done=False``);
        request metrics consume the final per-request rows separately.
        ``include_done=True`` is available for a terminal snapshot.
        """
        in_flight = {view.request_id for view in self.in_flight}
        owners = getattr(self.memory, "owners", {})
        memory_live = getattr(self.memory, "live", None)
        memory_capacity = getattr(self.memory, "capacity", None)
        memory_granted = getattr(self.memory, "granted", None)
        memory_selected = getattr(self.memory, "selected_reserved", None)
        memory_shared = getattr(self.memory, "shared_reserved", None)
        live = list(memory_live[self.replica]) if memory_live else []
        capacity = list(memory_capacity[self.replica]) if memory_capacity else []
        clock = 0.0 if now is None else float(now)
        if include_done:
            selected_states = list(self.states)
        else:
            selected_states = [self.by_id[request_id]
                               for request_id in sorted(self._live_ids)
                               if not self.by_id[request_id].done]
            selected_ids = {state.request.request_id for state in selected_states}
            # Terminal failures stay visible in the bounded admission record;
            # they are not a history of every done request.
            selected_states.extend(self.by_id[request_id]
                                   for request_id in sorted(self._infeasible_ids)
                                   if request_id not in selected_ids)
        selected_ids = {state.request.request_id for state in selected_states}
        requests = []
        for state in selected_states:
            request_id = state.request.request_id
            if state.done:
                status = "done"
            elif request_id in in_flight:
                status = "in_flight"
            elif state.started:
                status = "running"
            elif state.release_ms <= clock:
                status = "ready"
            else:
                status = "waiting_arrival"
            cache = owners.get((self.replica, request_id), ())
            requests.append({
                "request_id": request_id,
                "status": status,
                "location": state.location,
                "visible_context_tokens": state.computed,
                "generated_tokens": state.generated,
                "output_tokens": state.request.output_tokens,
                "pending_context_tokens": state.pending_context,
                "reserved_tokens": state.reserved_tokens,
                "state_version": state.state_version,
                "preemptions": state.preemptions,
                "accepted_tokens": state.accepted,
                "rolled_back_tokens": state.rolled_back,
                "cache_bytes": float(sum(cache)),
                "logical_cache_bytes_by_stage": [
                    float(value) for value in
                    getattr(self.memory, "logical_owners", {}).get(
                        (self.replica, request_id), ())],
                "request_version": getattr(self.memory, "request_versions", {}).get(
                    (self.replica, request_id)),
                "pending_async_consumers": list(self.memory.pending_async_consumers(
                    self.replica, request_id)) if hasattr(
                        self.memory, "pending_async_consumers") else [],
                "async_consumer_bindings": self._async_consumer_bindings(request_id),
                "retired": bool(self.memory.is_retired(
                    self.replica, request_id)) if hasattr(
                        self.memory, "is_retired") else False,
                "first_token_abs_ms": state.first_token_ms,
                "finish_ms": state.finish_ms,
                "reason": state.reason,
            })
        infeasible = sorted(state.request.request_id for state in selected_states
                            if state.reason is not None)
        return {
            "schema": "mindie-sim.scheduler-feedback/1",
            "replica": self.replica,
            "iteration": self.iteration,
            "last_reason": self.last_reason,
            "done": self.done,
            "in_flight_requests": sorted(in_flight),
            "pending_retired_requests": sorted(
                request for replica, request in
                getattr(self.memory, "retired_requests", ())
                if replica == self.replica),
            "admission": {
                "waiting_for_release": self.last_reason == "waiting_for_release",
                "capacity_or_budget_exhausted": (
                    self.last_reason == "capacity_or_budget_exhausted"),
                "infeasible_requests": infeasible,
            },
            "capacity": {
                "cache_bytes_by_stage": [float(value) for value in live],
                "kv_capacity_bytes_by_stage": [float(value) for value in capacity],
                "headroom_bytes_by_stage": [
                    float(cap - used) for cap, used in zip(capacity, live)
                ] if capacity else [],
                "granted_bytes_by_stage": [
                    float(value) for value in memory_granted[self.replica]
                ] if memory_granted else [],
                "selected_reserved_bytes_by_stage": [
                    float(value) for value in memory_selected[self.replica]
                ] if memory_selected else [],
                "shared_reserved_bytes_by_stage": [
                    float(value) for value in memory_shared[self.replica]
                ] if memory_shared else [],
                "owner_bytes_by_request": {
                    request: float(sum(sizes))
                    for (replica, request), sizes in owners.items()
                    if replica == self.replica and request in selected_ids
                },
            },
            "requests": requests,
            "draft": {
                request: {"mode": policy.mode, "proposal_tokens": policy.proposal,
                          "basis": policy.basis}
                for request, policy in self.drafts.items()
                if request in selected_ids
            },
        }

    def next_runtime(self, plan: IterationPlan):
        metadata = dict(self.case.runtime.metadata)
        modes = {draft.mode for draft in self.drafts.values() if draft.mode}
        if modes:
            metadata["draft_mode"] = next(iter(modes))
            metadata["draft_proposal_tokens"] = str(max(d.proposal for d in self.drafts.values()))
        return replace(self.case.runtime, metadata=tuple(sorted(metadata.items())))
