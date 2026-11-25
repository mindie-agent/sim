"""Batch admission and request scheduling.

[Role] Schedule bounded batches and commit request, cache and state changes.
[Invariants] Cache is reserved before work and token commits follow actual output.
[Timeouts & Locks] Work is bounded by active requests and configured batch budgets.
[Side-Effect Policy] Updates per-run request progress and the Memory ledger.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any, Callable, Mapping

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import RequestSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

from ..capacity import Memory
from .progress import DraftPolicy, Progress, running_first

_PREEMPTED = object()
_RETRY = object()
_COMMIT_HISTORY_LIMIT = 2048


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
        self._next_state_slot = 1
        self._state_slots: dict[str, int] = {}
        self.in_flight: tuple[RequestView, ...] = ()
        #: Plan iteration currently in flight; a stale callback cannot publish
        #: the next batch.
        self._in_flight_iteration: int | None = None
        #: Completion handles already committed (retaining references prevents
        #: id() reuse), stopping stale callbacks from advancing a subsequent
        #: batch without rejecting a new batch whose PP microbatch count shrunk.
        self._committed_completion_refs: deque[Any] = deque(maxlen=_COMMIT_HISTORY_LIMIT)
        self._committed_completions: set[int] = set()
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
            if result not in (_PREEMPTED, _RETRY):
                return result

    def _resize_memory(self, request_id: str, tokens: int, *, query_tokens: int) -> bool:
        """Reserve or update per-request KV/state demand with its active verify width."""
        if isinstance(self.memory, Memory):
            return self.memory.resize(
                self.replica, request_id, tokens, query_tokens=query_tokens,
            )
        return self.memory.resize(self.replica, request_id, tokens)

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

        Uses the same Memory admission geometry (offload, staging, host DRAM,
        and per-rank budgets) at the smallest legal step (minimum token span
        and ``query_tokens=1``, since batch clipping can legally reduce decode
        verify width to 1).  A request whose minimum step exceeds the static
        token budget or mounted HBM/host budget is permanently infeasible.
        """
        runtime = self.case.runtime
        budget = runtime.max_num_batched_tokens
        input_budget = runtime.max_num_input_tokens or budget
        decode = state.computed >= state.prompt_target
        if budget <= 0 or (not decode and input_budget <= 0):
            return False
        if not decode and not runtime.enable_chunked_prefill:
            if state.prompt_target - state.computed > min(budget, input_budget):
                return False
        if hasattr(self.memory, "admission_decision"):
            decision = self.memory.admission_decision(
                self.replica, state.request.request_id,
                self._minimum_request_tokens(state),
                query_tokens=1,
            )
            return not bool(decision.get("permanent", False))
        if not hasattr(self.memory, "cache_bytes") or not hasattr(self.memory, "capacity"):
            return True
        capacity_table = self.memory.capacity
        if not isinstance(capacity_table, (list, tuple)) or self.replica >= len(capacity_table):
            return True
        needed = self.memory.cache_bytes(
            self._minimum_request_tokens(state))
        capacity = capacity_table[self.replica]
        return all(float(need) <= float(cap) + max(1e-9, abs(float(cap)) * 1e-12)
                   for need, cap in zip(needed, capacity))

    def _mark_permanent(self, state: Progress) -> None:
        state.reason = "insufficient_cache_or_token_budget"
        state.pending_context = None
        state.reserved_tokens = 0
        self._state_slots.pop(state.request.request_id, None)
        self._live_ids.discard(state.request.request_id)
        self._infeasible_ids.add(state.request.request_id)
        # Preserve live async DMA consumer holds until their real completion.
        self.retire_state(state.request.request_id, force=False)
        successor = self.successors.get(state.request.request_id)
        while successor is not None:
            successor.reason = "closed_loop_predecessor_unserved"
            self._live_ids.discard(successor.request.request_id)
            self._infeasible_ids.add(successor.request.request_id)
            successor = self.successors.get(successor.request.request_id)

    def _handle_unscheduled_arrived(
        self, arrived: list[Progress], *, defer_capacity_failure: bool,
    ):
        """Resolve an empty scheduling pass via permanent retirement, preemption, or wait."""
        permanent = [state for state in arrived
                     if not self._minimum_request_fits(state)]
        feasible = [state for state in arrived
                    if state not in permanent]
        if permanent:
            for state in permanent:
                self._mark_permanent(state)
            if feasible:
                return _RETRY
            self.last_reason = "capacity_or_budget_exhausted"
            return None
        running = [state for state in arrived if state.started]
        if len(running) > 1 and not defer_capacity_failure:
            victim = running[-1]
            # Preemption revokes this request's logical demand.  A
            # still-running reader keeps its physical allocation until its
            # real completion; only the confirmed cancel/finish branch may
            # force-release the corresponding references.
            self.retire_state(victim.request.request_id, force=False)
            committed_context = (
                victim.request.prompt_tokens + max(0, victim.generated - 1)
            )
            victim.recompute_to = max(
                victim.recompute_to, victim.computed, committed_context,
            )
            victim.computed = min(victim.request.prefix_tokens,
                                  max(0, victim.request.prompt_tokens - 1))
            victim.started = False
            victim.preemptions += 1
            victim.state_version += 1
            victim.state_generation += 1
            victim.last_state_publication = ""
            self._state_slots.pop(victim.request.request_id, None)
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
        if feasible and (defer_capacity_failure
                         or self.memory.has_delayed_releases(self.replica)
                         or self._has_reclaimable_capacity(feasible)):
            self.last_reason = "waiting_for_release"
            return None
        waiting = [state for state in feasible if not state.started]
        if running and waiting:
            # Running requests held slots/KV while failing their next chunk;
            # retire only the blocked active set and retry waiting requests.
            for state in running:
                self._mark_permanent(state)
            return _RETRY
        for state in feasible:
            self._mark_permanent(state)
        # The loop has exhausted admission and preemption. This is a real
        # capacity/budget inability, never an absence-of-measurements gate.
        self.last_reason = "capacity_or_budget_exhausted"
        return None

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
            proposal = draft.proposal_for_step(state, remaining, budget) if decode else 0
            q = 1 if decode else min(state.prompt_target - state.computed, budget, input_budget)
            if not decode and not runtime.enable_chunked_prefill and q < state.prompt_target - state.computed:
                continue
            if q <= 0:
                continue
            allocation = q + proposal
            verify_width = 1 + proposal
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
            if not self._resize_memory(
                    state.request.request_id,
                    state.computed + allocation + lookahead,
                    query_tokens=verify_width):
                lookahead = 0
                if not self._resize_memory(
                        state.request.request_id,
                        state.computed + allocation,
                        query_tokens=verify_width):
                    continue
            state.pending_context = state.computed + allocation
            state.reserved_tokens = allocation
            if not state.started:
                state.started = True
                state.admission_ms = now if state.admission_ms is None else state.admission_ms
                active += 1
            views.append(self._build_scheduled_view(
                state, draft, decode=decode, q=q, proposal=proposal, remaining=remaining,
            ))
            budget -= allocation
            if not decode:
                input_budget -= q
        if not views:
            return self._handle_unscheduled_arrived(
                arrived, defer_capacity_failure=defer_capacity_failure,
            )
        self.in_flight = tuple(views)
        self.last_reason = "scheduled"
        result = IterationPlan(tuple(views), iteration=self.iteration,
                               output_tokens=tuple((s.request.request_id, s.request.output_tokens) for s in arrived))
        self._in_flight_iteration = result.iteration
        self.iteration += 1
        return result

    def _build_scheduled_view(
        self, state: Progress, draft: DraftPolicy, *, decode: bool, q: int, proposal: int, remaining: int,
    ) -> RequestView:
        advance = draft.committed(state, proposal) if proposal else 1
        samples = decode or state.computed + q >= state.prompt_target
        step_emitted = min(
            advance if decode else (1 if samples and not state.generated else 0),
            remaining,
        )
        remaining_after = max(0, remaining - step_emitted)
        next_decode_step = state.decode_steps + (1 if decode else 0)
        next_proposal = (
            draft.next_proposal_for_step(next_decode_step, remaining_after)
            if samples and draft.proposal > 0 else 0
        )
        req_id = state.request.request_id
        slot = self._state_slots.get(req_id)
        if slot is None:
            slot = self._next_state_slot
            self._next_state_slot += 1
            self._state_slots[req_id] = slot
        return RequestView(
            request_id=req_id, q_len=q, context=state.computed, samples=samples,
            phase="decode" if decode else "prefill", prefix_tokens=state.request.prefix_tokens,
            prefix_cache_key=state.request.prefix_cache_key,
            priority=state.request.priority, media_encoded=state.media_encoded,
            draft_verify=bool(proposal), draft_proposal_tokens=proposal,
            draft_prepare=(not decode) and draft.requires_draft_prepare and remaining > 2,
            draft_advanced_tokens=advance if decode else 0,
            draft_accepted_tokens=advance - 1 if decode else 0,
            prompt_tokens=state.request.prompt_tokens,
            state_version=state.state_version, state_slot=slot,
            state_generation=state.state_generation,
            previous_state_publication=state.last_state_publication,
            next_draft_proposal_tokens=next_proposal,
        )

    def commit(self, at_ms: float, times: Mapping[str, float] | None = None) -> None:
        if not self.in_flight:
            return
        self.memory.step_delayed_releases(self.replica)
        committed_iter = (
            self._in_flight_iteration
            if self._in_flight_iteration is not None
            else max(0, self.iteration - 1)
        )
        for view in self.in_flight:
            state = self.by_id[view.request_id]
            state.commit(view, (times or {}).get(view.request_id, at_ms))
            slot = int(view.state_slot or self._state_slots.get(view.request_id, 0))
            pub_kind = (
                "recover_ssm_commit"
                if (view.phase == "decode" and view.draft_verify and view.draft_proposal_tokens > 0)
                else ("decode_state" if view.phase == "decode" else "prefill_state")
            )
            state.last_state_publication = (
                f"{pub_kind}:iter{committed_iter}:req={view.request_id}"
                f":slot={slot}:gen={state.state_generation}:v{state.state_version}"
            )
            if state.done:
                self._live_ids.discard(view.request_id)
                self._state_slots.pop(view.request_id, None)
                if self.memory.async_delayed_release_steps > 0 and not self.done:
                    self.memory.defer_release(self.replica, view.request_id)
                else:
                    self.retire_state(view.request_id)
                successor = self.successors.get(view.request_id)
                if successor is not None and not self.case.runtime.serving_io:
                    successor.release_ms = max(successor.request.arrival_ms,
                                               state.finish_ms + successor.request.gap_ms)
            else:
                verify_width = (
                    1 + int(view.draft_proposal_tokens)
                    if view.phase == "decode" else 1
                )
                if not self._resize_memory(
                        view.request_id, state.computed,
                        query_tokens=verify_width):
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
        completion = getattr(step, "completion", None)
        if completion is not None and id(completion) in self._committed_completions:
            return False
        orig_iter = getattr(step, "original_iteration", None)
        iteration = getattr(step, "iteration", None)
        if orig_iter is not None:
            if int(orig_iter) != int(self._in_flight_iteration):
                return False
        elif iteration is not None:
            iter_val = int(iteration)
            # PP microbatches carry plan_iteration = original_iteration * cohort_count +
            # microbatch where cohort_count is the cohort-wide max microbatch count in
            # [local_microbatches, pipeline_microbatches] (may exceed local len(in_flight)
            # on asymmetric DP replicas). Accept only this transaction's valid range.
            max_mb = max(1, int(getattr(self.case.runtime, "pipeline_microbatches", 1)))
            local_mb = max(1, min(max_mb, max(1, len(self.in_flight))))
            cur_iter = int(self._in_flight_iteration)
            matched = iter_val == cur_iter or any(
                cur_iter * count <= iter_val < cur_iter * count + count
                for count in range(local_mb, max_mb + 1)
            )
            if not matched:
                return False
        when = at_ms if at_ms is not None else getattr(completion, "time", None)
        if when is None:
            raise ModelingError("scheduled_batch_has_no_completion_event")
        times = {
            request: event.time
            for request, event in getattr(step, "commits", {}).items()
            if event is not None and event.time is not None
        }
        in_flight_ids = {view.request_id for view in self.in_flight}
        if times and not (set(times) <= in_flight_ids):
            return False
        if completion is not None:
            if len(self._committed_completion_refs) >= _COMMIT_HISTORY_LIMIT:
                evicted = self._committed_completion_refs.popleft()
                self._committed_completions.discard(id(evicted))
            self._committed_completion_refs.append(completion)
            self._committed_completions.add(id(completion))
        self.commit(when, times)
        return True

    def bind_execution(self, execution: Any) -> None:
        """Make Execution-owned offload readers use this exact Memory."""
        execution.memory = self.memory
        execution._offload_memory = self.memory

    def _affected_replicas(self) -> tuple[int, ...]:
        """Replicas whose pools can constrain admission on ``self.replica``."""
        p = self.case.runtime.parallel
        dp_width = max(1, int(getattr(p, "dcp_dp_width", 1) or 1))
        if (bool(getattr(p, "cp_spans_dp", False))
                and dp_width > 1
                and getattr(self.memory, "rank_token_owners", None) is None):
            start = (self.replica // dp_width) * dp_width
            end = min(int(p.dp), start + dp_width)
            return tuple(range(start, end))
        return (self.replica,)

    def _replica_headroom_by_stage(
        self, capacity: list[float], live: list[float],
    ) -> tuple[list[float], list[float], list[float]]:
        """Derive per-stage ``(immediate, post_reclaim, reclaimable)`` headroom bytes.

        Consumes ``ResidentMemoryReceipt`` (immediate = ``usable - live``,
        post-reclaim = ``usable - (live - reclaimable)``, never historical peak)
        when backed by the unified Memory ledger.  Stage reclaimable capacity
        is the exact difference between stage post-reclaim and immediate minima.
        """
        if hasattr(self.memory, "resident_memory_receipt"):
            receipt = self.memory.resident_memory_receipt()
            by_stage: dict[int, list[Any]] = {}
            for die in receipt.dies:
                if die.replica == self.replica:
                    by_stage.setdefault(die.stage, []).append(die)
            if by_stage:
                stages = sorted(by_stage)
                immediate = [
                    float(min(d.immediate_free_hbm_headroom_bytes for d in by_stage[st]))
                    for st in stages
                ]
                post_reclaim = [
                    float(min(d.post_reclaim_free_hbm_headroom_bytes for d in by_stage[st]))
                    for st in stages
                ]
                reclaimable = [
                    float(max(0.0, post - imm))
                    for imm, post in zip(immediate, post_reclaim)
                ]
                return immediate, post_reclaim, reclaimable
        if hasattr(self.memory, "physical_usage"):
            usage = self.memory.physical_usage(self.replica)
            caps = usage.get("rank_capacity_bytes_per_stage_rank") or ()
            occ = usage.get("hbm_bytes_per_stage_rank") or ()
            if caps and occ:
                immediate = [
                    float(min(float(c) - float(u) for c, u in zip(c_row, u_row)))
                    for c_row, u_row in zip(caps, occ)
                    if c_row and u_row
                ]
                return immediate, list(immediate), [0.0] * len(immediate)
        immediate = [float(cap - used) for cap, used in zip(capacity, live)] if capacity else []
        return immediate, list(immediate), [0.0] * len(immediate)

    def _has_reclaimable_capacity(self, states=None) -> bool:
        """True when a real T08 release/hold event can make room.

        Inspects the authoritative ``ResidentMemoryReceipt`` post-reclaim
        margin, HBM and host-DRAM grants/reservations, grant readers, delayed
        releases, async consumer holds, and peer DCP-cohort holders across all
        affected replicas.  A request whose minimum span can never fit is not
        made feasible by any release.
        """
        if states:
            states = [state for state in states
                      if self._minimum_request_fits(state)]
            if not states:
                return False
        memory = self.memory
        replicas = set(self._affected_replicas())
        if hasattr(memory, "resident_memory_receipt"):
            receipt = memory.resident_memory_receipt()
            if any(
                die.post_reclaim_free_hbm_headroom_bytes > die.immediate_free_hbm_headroom_bytes
                for die in receipt.dies
                if die.replica in replicas
            ):
                return True
        if states and hasattr(memory, "admission_decision"):
            for state in states:
                decision = memory.admission_decision(
                    self.replica, state.request.request_id,
                    self._minimum_request_tokens(state),
                    query_tokens=1,
                )
                if decision.get("wake_events"):
                    return True
        if hasattr(memory, "has_delayed_releases"):
            if any(memory.has_delayed_releases(rep) for rep in replicas):
                return True
        for attr in ("granted", "selected_reserved", "shared_reserved"):
            rows = getattr(memory, attr, None)
            if rows and any(
                float(value) > 0.0
                for rep in replicas if rep < len(rows)
                for value in rows[rep]
            ):
                return True
        granted_host = getattr(memory, "granted_host", None)
        if granted_host and any(
            float(granted_host[rep]) > 0.0
            for rep in replicas if rep < len(granted_host)
        ):
            return True
        grant_readers = getattr(memory, "grant_readers", {}) or {}
        if any(replica in replicas and bool(readers)
               for (replica, _grant), readers in grant_readers.items()):
            return True
        holders = getattr(memory, "owners", {}) or {}
        for replica, request in holders:
            if replica not in replicas:
                continue
            if replica != self.replica:
                return True
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
        immediate_headroom, post_reclaim_headroom, reclaimable_by_stage = (
            self._replica_headroom_by_stage(capacity, live)
        )
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
                "headroom_bytes_by_stage": immediate_headroom,
                "immediate_free_hbm_headroom_bytes_by_stage": immediate_headroom,
                "post_reclaim_free_hbm_headroom_bytes_by_stage": post_reclaim_headroom,
                "reclaimable_bytes_by_stage": reclaimable_by_stage,
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
