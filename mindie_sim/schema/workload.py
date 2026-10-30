"""Serving iteration contracts.

[Role] Describe scheduled token work separately from output commits.
[Invariants] Padding is not output; proposal and accepted counts are separate.
[Timeouts & Locks] Finite immutable records, no locks or I/O.
[Side-Effect Policy] None.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class RequestView:
    request_id: str
    q_len: int
    context: int
    samples: bool
    phase: str
    prefix_tokens: int = 0
    priority: int = 0
    padding: bool = False
    semantic_phase: str = ""
    execution_mode: str = ""
    media_encoded: bool = False
    draft_verify: bool = True
    draft_prepare: bool = True
    draft_proposal_tokens: int = 0
    draft_accepted_tokens: int = 0
    # Integer commits sampled or allocated by the acceptance policy.
    draft_advanced_tokens: int = 0
    prompt_tokens: int = 0
    global_q_len: int = 0
    query_ranges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.semantic_phase:
            object.__setattr__(self, "semantic_phase", self.phase)
        if not self.execution_mode:
            object.__setattr__(self, "execution_mode", self.phase)

    @property
    def kv_len(self) -> int:
        return self.context + (self.global_q_len or self.q_len)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "q_len": self.q_len,
            "context": self.context, "kv_len": self.kv_len,
            "samples": self.samples, "phase": self.phase,
            "semantic_phase": self.semantic_phase,
            "execution_mode": self.execution_mode,
            "prefix_tokens": self.prefix_tokens, "padding": self.padding,
            "media_encoded": self.media_encoded,
            "draft_verify": self.draft_verify,
            "draft_prepare": self.draft_prepare,
            "draft_proposal_tokens": self.draft_proposal_tokens,
            "draft_accepted_tokens": self.draft_accepted_tokens,
            "draft_advanced_tokens": self.draft_advanced_tokens,
            "prompt_tokens": self.prompt_tokens,
        }

@dataclass(frozen=True)
class IterationPlan:
    requests: tuple[RequestView, ...]
    iteration: int = 0
    reason: str = "scheduled"
    #: Declared full output length per scheduled request.  Carries the target
    #: acceptance lookup key without inferring it from a table average.
    output_tokens: tuple[tuple[str, int], ...] = ()
    all_request_ids: tuple[str, ...] = ()
    request_statuses: tuple[tuple[str, str], ...] = ()

    @property
    def valid_tokens(self) -> int:
        return sum(view.q_len for view in self.requests)

    @property
    def modes(self) -> set[str]:
        return {view.phase for view in self.requests}

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "requests": [view.to_dict() for view in self.requests],
            "valid_tokens": self.valid_tokens,
            "mode": selected_mode(self),
        }


def selected_mode(plan: IterationPlan) -> str:
    if not plan.requests:
        return "idle"
    modes = plan.modes
    if modes == {"decode"}:
        return "decode"
    if modes == {"prefill"}:
        return "prefill"
    return "mixed"
