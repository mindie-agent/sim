"""Serving iteration contracts.

[Role] Describe scheduled token work separately from output commits.
[Invariants] Padding is not output; proposal and accepted counts are separate.
[Timeouts & Locks] Finite immutable records, no locks or I/O.
[Side-Effect Policy] None.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

from .common import ModelingError

SYNTHETIC_EAGLE3_GQA_PROVENANCE = "synthetic_eagle3_gqa"
SPECULATIVE_WORKLOAD_METHODS = frozenset({
    "none", "mtp", "dspark", "eagle", "eagle3", "ngram", "ngram_gpu",
})
SpeculativeWorkloadMethod = Literal[
    "none", "mtp", "dspark", "eagle", "eagle3", "ngram", "ngram_gpu",
]


@dataclass(frozen=True, slots=True)
class SpeculativeWorkloadPolicy:
    """Explicit speculative-decoding policy separating real stacks from synthetic exploration.

    Actual deployment simulation requires a real speculative checkpoint
    (``allow_synthetic_gqa=False``). Hypothetical modeling may explicitly opt
    into a synthetic EAGLE3 GQA draft by setting ``allow_synthetic_gqa=True``
    with mandatory provenance tag ``"synthetic_eagle3_gqa"``.
    """

    spec_method: SpeculativeWorkloadMethod = "none"
    num_speculative_tokens: int = 0
    allow_synthetic_gqa: bool = False
    synthetic_draft_provenance: str = ""

    def __post_init__(self) -> None:
        if self.spec_method not in SPECULATIVE_WORKLOAD_METHODS:
            raise ModelingError(f"speculative_workload_method_invalid:{self.spec_method!r}")
        if isinstance(self.num_speculative_tokens, bool) or not isinstance(self.num_speculative_tokens, int) or self.num_speculative_tokens < 0:
            raise ModelingError(f"speculative_workload_tokens_invalid:{self.num_speculative_tokens!r}")
        if not isinstance(self.allow_synthetic_gqa, bool):
            raise ModelingError("speculative_workload_allow_synthetic_invalid")
        if self.num_speculative_tokens > 0 and self.spec_method == "none":
            raise ModelingError("speculative_workload_method_required_for_positive_tokens")
        if self.allow_synthetic_gqa:
            if self.spec_method != "eagle3":
                raise ModelingError(f"synthetic_gqa_requires_eagle3_method:{self.spec_method!r}")
            if self.num_speculative_tokens < 1:
                raise ModelingError("synthetic_gqa_requires_positive_speculative_tokens")
            prov = self.synthetic_draft_provenance.strip() if isinstance(self.synthetic_draft_provenance, str) else ""
            if not prov:
                object.__setattr__(self, "synthetic_draft_provenance", SYNTHETIC_EAGLE3_GQA_PROVENANCE)
            elif prov != SYNTHETIC_EAGLE3_GQA_PROVENANCE:
                raise ModelingError(f"synthetic_gqa_provenance_mismatch:{prov!r}")
        elif self.synthetic_draft_provenance:
            raise ModelingError(
                f"synthetic_gqa_provenance_forbids_disabled_opt_in:{self.synthetic_draft_provenance!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_method": self.spec_method,
            "num_speculative_tokens": self.num_speculative_tokens,
            "allow_synthetic_gqa": self.allow_synthetic_gqa,
            "synthetic_draft_provenance": self.synthetic_draft_provenance,
        }


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
    #: Physical prefix-pool identity of the request's immutable prefix.
    prefix_cache_key: str = ""
    #: Optional outer physical tensor padding lengths for rectangular BSND/BNSD/BSH layouts.
    padded_q_len: int = 0
    padded_seq_len: int = 0
    #: Monotonic per-request recurrent/KV state version, slot & prior publication.
    state_version: int = 0
    state_slot: int | None = None
    state_generation: int = 0
    previous_state_publication: str = ""
    #: Next-step speculative proposal width decoupled from the current verified width.
    next_draft_proposal_tokens: int | None = None
    #: Optional typed Mamba/SSM block table row & boundary descriptor for align-mode RecoverSSM.
    mamba_block_table: tuple[int, ...] = ()
    recover_ssm_boundary: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.semantic_phase:
            object.__setattr__(self, "semantic_phase", self.phase)
        if not self.execution_mode:
            object.__setattr__(self, "execution_mode", self.phase)
        # A padding view is a physical graph allocation, not an active
        # request.  It has no logical history; any active context belongs to
        # the graph/source declaration, never to this pseudo-request.
        if self.padding:
            object.__setattr__(self, "context", 0)

    @property
    def logical_q_len(self) -> int:
        return 0 if self.padding else self.q_len

    @property
    def kv_len(self) -> int:
        if self.padding:
            return 0
        return self.context + (self.global_q_len or self.q_len)

    @property
    def physical_q_len(self) -> int:
        """Outer physical query stride (``padded_q_len`` for BSND/BNSD/BSH, else ``q_len``)."""
        return max(int(self.q_len), int(self.padded_q_len))

    @property
    def physical_kv_len(self) -> int:
        """Outer physical KV stride (``padded_seq_len`` for BSND/BNSD/BSH, else ``kv_len``)."""
        return max(int(self.kv_len), int(self.padded_seq_len))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "request_id": self.request_id, "q_len": self.q_len,
            "context": self.context, "kv_len": self.kv_len,
            "samples": self.samples, "phase": self.phase,
            "semantic_phase": self.semantic_phase,
            "execution_mode": self.execution_mode,
            "prefix_tokens": self.prefix_tokens, "priority": self.priority,
            "padding": self.padding,
            "media_encoded": self.media_encoded,
            "draft_verify": self.draft_verify,
            "draft_prepare": self.draft_prepare,
            "draft_proposal_tokens": self.draft_proposal_tokens,
            "draft_accepted_tokens": self.draft_accepted_tokens,
            "draft_advanced_tokens": self.draft_advanced_tokens,
            "prompt_tokens": self.prompt_tokens,
            # The execution geometry may have been split across CP/PCP ranks.
            # ``kv_len`` above is derived from ``global_q_len``, so omitting the
            # global width and owned query spans would export an inconsistent
            # request: its cache length could no longer be reconstructed from
            # the local ``q_len``.
            "global_q_len": self.global_q_len,
            "query_ranges": self.query_ranges,
            "prefix_cache_key": self.prefix_cache_key,
        }
        if self.mamba_block_table:
            payload["mamba_block_table"] = self.mamba_block_table
        if self.recover_ssm_boundary:
            payload["recover_ssm_boundary"] = dict(self.recover_ssm_boundary)
        if self.padded_q_len:
            payload["padded_q_len"] = self.padded_q_len
        if self.padded_seq_len:
            payload["padded_seq_len"] = self.padded_seq_len
        if self.state_version:
            payload["state_version"] = self.state_version
        if self.state_slot is not None:
            payload["state_slot"] = self.state_slot
        if self.state_generation:
            payload["state_generation"] = self.state_generation
        if self.previous_state_publication:
            payload["previous_state_publication"] = self.previous_state_publication
        if self.next_draft_proposal_tokens is not None:
            payload["next_draft_proposal_tokens"] = self.next_draft_proposal_tokens
        return payload

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
        return sum(view.logical_q_len for view in self.requests)

    @property
    def modes(self) -> set[str]:
        return {view.phase for view in self.requests}

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "reason": self.reason,
            "requests": [view.to_dict() for view in self.requests],
            "valid_tokens": self.valid_tokens,
            "mode": selected_mode(self),
            # Acceptance lookup key and completion/status bindings are part of
            # the scheduled plan, not derivable from the request rows.
            "output_tokens": self.output_tokens,
            "all_request_ids": self.all_request_ids,
            "request_statuses": self.request_statuses,
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
