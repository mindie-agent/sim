"""Request progress and draft acceptance.

[Role] Track committed request progress and derive integer draft acceptance.
[Invariants] Only verified token advances commit. Padding never produces tokens.
[Timeouts & Locks] Bounded per-request state and in-memory acceptance policy.
[Side-Effect Policy] Updates per-request progress, acceptance credit and token
  timelines.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import RequestSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.workload import RequestView

from ..serving_io import TokenTimeline

_DRAFT_ACCEPTANCE_HASH_BITS = 64


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
    #: Preempted logical request waiting for a real async reader release.
    waiting_for_release: bool = False
    recompute_to: int = 0
    reason: str | None = None
    media_encoded: bool = False
    proposed: int = 0
    accepted: int = 0
    rolled_back: int = 0
    acceptance_credit: float = 0
    release_ms: float = 0
    location: str = "prefill"
    #: Monotonic counter of applied state transitions (commit, preemption or
    #: rollback).  It is not a token identity and never replaces ``computed``.
    state_version: int = 0
    #: Context reserved by the batch currently in flight.  Visible only after
    #: ``commit`` so consumers cannot read a candidate state.
    pending_context: int | None = None
    reserved_tokens: int = 0
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
            if not view.draft_verify and int(view.q_len) != 1:
                raise ModelingError(
                    "non_speculative_decode_forward_must_commit_one_token:"
                    + view.request_id)
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
        self.state_version += 1
        self.pending_context = None
        self.reserved_tokens = 0

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
            "state_version": self.state_version,
            "visible_context_tokens": self.computed,
            "committed_tokens": self.generated,
            "pending_context_tokens": self.pending_context,
            "reserved_tokens": self.reserved_tokens,
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
            draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2 ** _DRAFT_ACCEPTANCE_HASH_BITS
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
