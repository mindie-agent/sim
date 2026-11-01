"""Request scheduling driven by resource completion and cache ownership.

[Role] Build bounded prefill/decode batches and commit real output events.
[Invariants] Padding never produces tokens. Speculation produces integer token
  counts only after verification. KV is reserved before execution and retained
  until consumers finish. Preemption frees bytes and causes actual recomputation.
[Timeouts & Locks] Pure in-memory transitions, no external work or waits.
[Side-Effect Policy] Mutates per-request progress and the prediction Memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import hashlib
import math
from typing import Callable, Mapping

from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.model import RequestSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from .capacity import Memory
from .serving_io import TokenTimeline

_PREEMPTED = object()


@dataclass
class Progress:
    request: RequestSpec
    computed: int = 0
    generated: int = 0
    first_token_ms: float | None = None
    finish_ms: float | None = None
    last_token_ms: float | None = None
    admission_ms: float | None = None
    itl_sum: float = 0
    itl_count: int = 0
    itl_max: float = 0
    iterations: int = 0
    decode_steps: int = 0
    started: bool = False
    preemptions: int = 0
    recompute_to: int = 0
    reason: str | None = None
    media_encoded: bool = False
    proposed: int = 0
    accepted: int = 0
    rolled_back: int = 0
    acceptance_credit: float = 0
    release_ms: float = 0
    location: str = "prefill"
    host_result: TokenTimeline = field(default_factory=TokenTimeline)
    client_delivery: TokenTimeline = field(default_factory=TokenTimeline)

    def __post_init__(self):
        # A fully cached prefix still needs its final hidden state for sampling.
        self.computed = min(self.request.prefix_tokens, max(0, self.request.prompt_tokens - 1))
        self.media_encoded = self.request.media_encoded
        self.release_ms = self.request.arrival_ms

    @property
    def prompt_target(self) -> int:
        return max(self.request.prompt_tokens, self.recompute_to)

    @property
    def done(self) -> bool:
        return self.finish_ms is not None or self.reason is not None

    def commit(self, view: RequestView, at_ms: float) -> None:
        advance = 0
        if view.phase == "decode":
            advance = int(view.draft_advanced_tokens) if view.draft_verify else 1
            advance = min(advance, self.request.output_tokens - self.generated)
            self.computed += advance
            self.decode_steps += 1
            self.proposed += view.draft_proposal_tokens
            self.accepted += max(0, advance - 1)
            self.rolled_back += max(0, view.draft_proposal_tokens - advance + 1)
        else:
            self.computed += view.q_len
            if view.samples and not self.generated:
                advance = min(1, self.request.output_tokens)
        self.iterations += 1
        if advance:
            if self.first_token_ms is None:
                self.first_token_ms = at_ms
                self.itl_count += advance - 1
            else:
                gap = at_ms - self.last_token_ms
                self.itl_sum += gap
                self.itl_count += advance
                self.itl_max = max(self.itl_max, gap)
            self.last_token_ms = at_ms
            self.generated += advance
        if view.q_len and self.request.media_tokens:
            self.media_encoded = True
        if self.generated >= self.request.output_tokens:
            self.finish_ms = at_ms

    def to_dict(self) -> dict:
        arrival = self.release_ms if math.isfinite(self.release_ms) else self.request.arrival_ms
        return {
            "request_id": self.request.request_id, "arrival_ms": arrival,
            "admission_ms": self.admission_ms, "prompt_tokens": self.request.prompt_tokens,
            "output_tokens": self.request.output_tokens, "generated_tokens": self.generated,
            "first_token_abs_ms": self.first_token_ms, "finish_ms": self.finish_ms,
            "last_token_abs_ms": self.last_token_ms,
            "ttft_ms": None if self.first_token_ms is None else self.first_token_ms - arrival,
            "tpot_ms": ((self.finish_ms - self.first_token_ms) / (self.request.output_tokens - 1)
                        if self.finish_ms is not None and self.request.output_tokens > 1 else None),
            "e2e_ms": None if self.finish_ms is None else self.finish_ms - arrival,
            "status": "served" if self.finish_ms is not None else "unserved",
            "reason": self.reason, "iterations": self.iterations, "preemptions": self.preemptions,
            "token_event_semantics": "committed_batch" if self.proposed else "discrete",
            "itl": {"count": self.itl_count,
                    "mean_ms": self.itl_sum / self.itl_count if self.itl_count else None,
                    "max_ms": self.itl_max if self.itl_count else None},
            "speculative": {"proposed_tokens": self.proposed, "accepted_tokens": self.accepted,
                            "rolled_back_tokens": self.rolled_back},
        }


@dataclass(frozen=True)
class DraftPolicy:
    mode: str = ""
    proposal: int = 0
    mean: float = 1.0
    trace: tuple[int, ...] = ()
    basis: str = "disabled"
    distribution: tuple[float, ...] = ()
    seed: int = 0

    @classmethod
    def for_request(cls, case: CaseSpec, request: RequestSpec) -> DraftPolicy:
        runtime, metadata = case.runtime, dict(case.runtime.metadata)
        mtp = getattr(case.model.auxiliary, "mtp", None)
        mode = str(metadata.get("draft_mode", "") or "")
        proposal = int(metadata.get("draft_proposal_tokens", getattr(mtp, "proposal_tokens", 0)) or 0)
        mean = dict(runtime.mtp_acceptance_by_output_tokens).get(request.output_tokens,
                                                                  runtime.mtp_acceptance_length)
        if not mode and mtp is not None and mean > 1:
            mode = "mtp"
        basis, distribution = "declared_mean_integer_credit", ()
        raw_spec = metadata.get("speculative")
        if runtime.dspark is not None:
            spec = runtime.dspark.step_spec()
            mode, proposal, mean = "dspark", spec.draft_len, spec.expected_committed_tokens()
            basis = "declared_distribution"
            distribution = spec.acceptance_distribution()
        elif raw_spec:
            spec = SpeculativeSpec.from_mapping(json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec)
            proposal, mean = spec.draft_len, spec.expected_committed_tokens()
            if not mode and mtp is not None:
                mode = "mtp"
            basis = "declared_distribution"
            distribution = spec.acceptance_distribution()
        accepted = metadata.get("draft_accept_tokens")
        if accepted is not None:
            mean, basis = int(accepted) + 1, "declared_acceptance"
            distribution = ()
        trace_raw = metadata.get("draft_accept_trace")
        trace = ()
        if trace_raw:
            data = json.loads(trace_raw) if isinstance(trace_raw, str) else trace_raw
            if isinstance(data, dict):
                data = data.get(request.request_id, data.get("accepted", []))
            trace, basis = tuple(int(value) for value in data), "declared_trace_then_mean"
        if not mode and mean > 1:
            raise ValueError("draft_structure_required_for_speculative_acceptance")
        if not mode:
            return cls()
        if proposal == 0 and distribution == (1.0,):
            return cls(basis="confidence_declined_all_proposals")
        if proposal <= 0 or not 1 <= mean <= proposal + 1 or any(value < 0 or value > proposal for value in trace):
            raise ValueError("draft_acceptance_exceeds_declared_proposals")
        return cls(mode, proposal, float(mean), trace, basis, distribution,
                   int(metadata.get("random_seed", 0)))

    def committed(self, state: Progress, proposal: int) -> int:
        if state.decode_steps < len(self.trace):
            return 1 + min(proposal, self.trace[state.decode_steps])
        if self.distribution:
            material = f"{self.seed}:{state.request.request_id}:{state.decode_steps}".encode()
            draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2 ** 64
            cumulative = 0
            for accepted, probability in enumerate(self.distribution):
                cumulative += probability
                if draw < cumulative:
                    return 1 + min(proposal, accepted)
            return 1 + proposal
        credit = state.acceptance_credit + (self.mean - 1) * proposal / max(1, self.proposal)
        accepted = math.floor(credit + 1e-12)
        state.acceptance_credit = credit - accepted
        return 1 + accepted


def running_first(state: Progress) -> tuple:
    return not state.started, state.request.priority, state.request.arrival_ms, state.request.request_id


class Scheduler:
    def __init__(self, case: CaseSpec, requests: list[RequestSpec], memory: Memory, replica: int = 0,
                 order: Callable[[Progress], tuple] = running_first, eligible: Callable = lambda state: True,
                 handoff_after_prefill: bool = False):
        self.case, self.memory, self.replica, self.order = case, memory, replica, order
        self.eligible = eligible
        self.handoff_after_prefill = handoff_after_prefill
        self.states = [Progress(request) for request in requests]
        self.by_id = {state.request.request_id: state for state in self.states}
        self.drafts = {state.request.request_id: DraftPolicy.for_request(case, state.request)
                       for state in self.states}
        self.iteration = 0
        self.in_flight: tuple[RequestView, ...] = ()
        self.blocked = False
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
        while True:
            result = self._schedule(now, defer_capacity_failure=defer_capacity_failure)
            if result is not _PREEMPTED:
                return result

    def _schedule(self, now, *, defer_capacity_failure):
        runtime = self.case.runtime
        arrived = sorted((state for state in self.states
                          if not state.done and self.eligible(state) and state.release_ms <= now), key=self.order)
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
            lookahead = (
                min(max(0, remaining - allocation), self.memory.async_in_flight_lookahead_steps * allocation)
                if decode and self.memory.async_in_flight_lookahead_steps > 0
                else 0
            )
            if not self.memory.resize(self.replica, state.request.request_id, state.computed + allocation + lookahead):
                continue
            if not state.started:
                state.started = True
                state.admission_ms = now if state.admission_ms is None else state.admission_ms
                active += 1
            advance = draft.committed(state, proposal) if proposal else 1
            views.append(RequestView(
                request_id=state.request.request_id, q_len=q, context=state.computed,
                samples=decode or state.computed + q >= state.prompt_target,
                phase="decode" if decode else "prefill", prefix_tokens=state.request.prefix_tokens,
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
            # A PD transfer or async delayed release owns cache until completion.
            # That event can release capacity without evicting or failing a waiting request.
            if defer_capacity_failure or self.memory.has_delayed_releases(self.replica):
                return None
            running = [state for state in arrived if state.started]
            if len(running) > 1:
                victim = running[-1]
                self.memory.release(self.replica, victim.request.request_id)
                victim.recompute_to = victim.computed
                victim.computed = min(victim.request.prefix_tokens, max(0, victim.request.prompt_tokens - 1))
                victim.started = False
                victim.preemptions += 1
                return _PREEMPTED
            # The loop has exhausted admission and preemption. This is a real
            # capacity/budget inability, never an absence-of-measurements gate.
            for state in arrived:
                state.reason = "insufficient_cache_or_token_budget"
                self.memory.release(self.replica, state.request.request_id)
                successor = self.successors.get(state.request.request_id)
                while successor is not None:
                    successor.reason = "closed_loop_predecessor_unserved"
                    successor = self.successors.get(successor.request.request_id)
            return None
        self.in_flight = tuple(views)
        result = IterationPlan(tuple(views), iteration=self.iteration,
                               output_tokens=tuple((s.request.request_id, s.request.output_tokens) for s in arrived))
        self.iteration += 1
        return result

    def commit(self, at_ms: float, times: Mapping[str, float] | None = None) -> None:
        self.memory.step_delayed_releases(self.replica)
        for view in self.in_flight:
            state = self.by_id[view.request_id]
            state.commit(view, (times or {}).get(view.request_id, at_ms))
            if state.done:
                if self.memory.async_delayed_release_steps > 0 and not self.done:
                    self.memory.defer_release(self.replica, view.request_id)
                else:
                    self.memory.release(self.replica, view.request_id)
                successor = self.successors.get(view.request_id)
                if successor is not None and not self.case.runtime.serving_io:
                    successor.release_ms = max(successor.request.arrival_ms,
                                               state.finish_ms + successor.request.gap_ms)
            else:
                self.memory.resize(self.replica, view.request_id, state.computed)
        if self.done:
            self.memory.flush_delayed_releases(self.replica)
        self.in_flight = ()

    def next_runtime(self, plan: IterationPlan):
        metadata = dict(self.case.runtime.metadata)
        modes = {draft.mode for draft in self.drafts.values() if draft.mode}
        if modes:
            metadata["draft_mode"] = next(iter(modes))
            metadata["draft_proposal_tokens"] = str(max(d.proposal for d in self.drafts.values()))
        return replace(self.case.runtime, metadata=tuple(sorted(metadata.items())))
