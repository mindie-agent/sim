"""Per-request execution state for predictive and normative A3 timelines.

[Role]
  Track committed prompt/decode/KV state, first-token and finish timestamps,
  media tower-once state, draft proposal/acceptance/rollback state, and the
  explicit cost-backend protocol used only by validation tests.

[Invariants]
  * Prompt positions expanded for media are visible to the scheduler budget.
    A media tower executes at most once per request; already-encoded input and
    later chunks/recompute never re-execute it.
  * Draft acceptance is caller-declared.  Missing evidence is an explicit
    fallback (0 accepted -> one fallback token), never a measured distribution.
    Accepted tokens commit KV; rejected proposal tokens are rollback-only.
  * A verify iteration charges the real target verify graph once and advances
    generated/KV/output by accepted+1 (capped by remaining output).  The final
    output tail falls back to a normal one-token decode without full verify.
  * Request state updates are deterministic and independent of global DP step
    barriers.

[Timeouts & Locks]
  Pure in-memory arithmetic.  No I/O, locks, sleeps or device access.

[Side-Effect Policy]
  Mutates only its own RequestState fields; no filesystem/network/global state.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .a3_types import A3InputError, CaseSpec


@dataclass
class RequestState:
    request: Any
    computed: int = 0
    generated: int = 0
    first_token_ms: float | None = None
    token_times_ms: list[float] = field(default_factory=list)
    token_event_times_ms: list[float] = field(default_factory=list)
    finish_ms: float | None = None
    kv_blocks: int = 0
    started: bool = False
    preempted: int = 0
    unserved_reason: str | None = None
    media_encoded: bool = False
    draft_mode: str = ""
    draft_proposal_tokens: int = 0
    draft_accept_default: int = 0
    draft_accept_trace: list[int] = field(default_factory=list)
    draft_accept_cursor: int = 0
    draft_acceptance_source: str = "not_draft"
    draft_proposed_total: int = 0
    draft_accepted_total: int = 0
    draft_rollback_tokens: int = 0
    draft_temporary_tokens: int = 0
    draft_fallback_tokens: int = 0
    draft_steps: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.computed == 0:
            self.computed = self.request.prefix_tokens
        self.media_encoded = bool(
            getattr(self.request, "media_encoded", False)
        ) or bool(self.media_encoded)

    @property
    def prompt(self) -> int:
        return self.request.prompt_tokens

    @property
    def output(self) -> int:
        return self.request.output_tokens

    @property
    def prompt_done(self) -> bool:
        return self.computed >= self.prompt

    def note_media_execution(self, view: Any) -> None:
        """Persist tower execution for this request after its first prefill.

        Catalog lowering is stateless between forwards; the scheduler carries
        this flag in the next view's ``media_encoded`` field so later chunks,
        recompute and already-encoded input never execute the tower again.
        """
        if self.media_encoded:
            return
        if int(getattr(self.request, "media_tokens", 0) or 0) <= 0:
            return
        if int(getattr(view, "context", 0) or 0) != 0:
            return
        if str(getattr(view, "semantic_phase", getattr(view, "phase", ""))) == "decode":
            return
        self.media_encoded = True

    def configure_draft(self, runtime: Any, model: Any) -> None:
        """Bind explicit draft acceptance inputs to this request.

        No measured acceptance distribution is invented.  If the caller only
        declares a draft mode, acceptance remains an explicit unspecified
        fallback (0 accepted -> one fallback token); callers should supply
        ``draft_accept_tokens`` or ``draft_accept_trace``.
        """
        meta = dict(getattr(runtime, "metadata", ()) or ())
        mode = str(meta.get("draft_mode", "") or "").lower()
        if not mode:
            return
        aux = getattr(model, "auxiliary", None)
        spec = None
        if aux is not None:
            dspark_mode = mode in {"dspark", "verify", "draft_verify"}
            spec = getattr(aux, "dspark", None) if dspark_mode else getattr(aux, "mtp", None)
            if spec is None:
                spec = getattr(aux, "mtp", None) or getattr(aux, "dspark", None)
        proposal = int(getattr(spec, "proposal_tokens", 0) or 0)
        if proposal <= 0:
            proposal = int(meta.get("draft_proposal_tokens", 0) or 0)
        self.draft_mode = mode
        self.draft_proposal_tokens = max(0, proposal)
        source = "unspecified_fallback_zero"
        if meta.get("draft_accept_tokens") is not None:
            try:
                self.draft_accept_default = max(
                    0, min(proposal, int(meta.get("draft_accept_tokens", 0) or 0))
                )
            except (TypeError, ValueError):
                self.draft_accept_default = 0
            source = "draft_accept_tokens"
        raw_trace = meta.get("draft_accept_trace")
        if raw_trace:
            try:
                parsed = (
                    json.loads(raw_trace)
                    if isinstance(raw_trace, str) else raw_trace
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            values: list[int] = []
            if isinstance(parsed, Mapping):
                selected = parsed.get(self.request.request_id, parsed)
                if isinstance(selected, Mapping):
                    selected = selected.get("accepted")
                if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
                    values = [int(v) for v in selected]
                elif selected is not None:
                    values = [int(selected)]
            elif isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
                if all(isinstance(v, Mapping) for v in parsed):
                    for item in parsed:
                        rid = item.get("request_id")
                        if rid is None or str(rid) == self.request.request_id:
                            values.append(int(item.get("accepted_tokens", 0) or 0))
                else:
                    values = [int(v) for v in parsed]
            if values:
                self.draft_accept_trace = [
                    max(0, min(proposal, value)) for value in values
                ]
                source = "draft_accept_trace"
        self.draft_acceptance_source = source

    def peek_draft_accept(self) -> int:
        if not self.draft_mode:
            return 0
        if self.draft_accept_cursor < len(self.draft_accept_trace):
            return self.draft_accept_trace[self.draft_accept_cursor]
        return self.draft_accept_default

    def commit_view(self, view: Any, end_ms: float) -> int:
        """Commit one scheduled view and return committed token advance."""
        self.started = True
        computed_advance = int(getattr(view, "q_len", 0) or 0)
        generated_advance = 1 if bool(getattr(view, "samples", False)) else 0
        if bool(getattr(view, "draft_verify", False)):
            proposal = max(0, int(getattr(view, "draft_proposal_tokens", 0) or 0))
            accepted = max(
                0, min(proposal, int(getattr(view, "draft_accepted_tokens", 0) or 0))
            )
            remaining = max(0, self.output - self.generated)
            advance = min(remaining, accepted + 1)
            computed_advance = advance
            generated_advance = advance
            self.draft_proposed_total += proposal
            self.draft_accepted_total += min(accepted, max(0, remaining - 1))
            self.draft_rollback_tokens += max(0, proposal - accepted)
            self.draft_temporary_tokens += proposal
            if accepted == 0:
                self.draft_fallback_tokens += 1
            self.draft_steps.append({
                "iteration": len(self.draft_steps),
                "proposal_tokens": proposal,
                "accepted_tokens": accepted,
                "advanced_tokens": advance,
                "rollback_tokens": max(0, proposal - accepted),
                "fallback": accepted == 0,
                "acceptance_source": self.draft_acceptance_source,
            })
            self.draft_accept_cursor += 1
        computed_advance = max(0, computed_advance)
        generated_advance = max(0, min(generated_advance, self.output - self.generated))
        self.computed += computed_advance
        self.note_media_execution(view)
        if bool(getattr(view, "samples", False)):
            if self.first_token_ms is None:
                self.first_token_ms = end_ms
            self.generated += generated_advance
            self.token_times_ms.append(end_ms)
            self.token_event_times_ms.extend([end_ms] * generated_advance)
            if self.generated >= self.output:
                self.finish_ms = end_ms
        return computed_advance

    @property
    def finished(self) -> bool:
        return self.finish_ms is not None or self.generated >= self.output

    def to_dict(self) -> dict[str, Any]:
        itl = [self.token_times_ms[i] - self.token_times_ms[i - 1]
               for i in range(1, len(self.token_times_ms))]
        token_itl = [
            self.token_event_times_ms[i] - self.token_event_times_ms[i - 1]
            for i in range(1, len(self.token_event_times_ms))
        ]
        tpot = None
        if self.first_token_ms is not None and self.finish_ms is not None and self.generated > 1:
            tpot = (self.finish_ms - self.first_token_ms) / (self.generated - 1)
        ttft = (
            self.first_token_ms - self.request.arrival_ms
            if self.first_token_ms is not None else None
        )
        return {
            "request_id": self.request.request_id,
            "arrival_ms": self.request.arrival_ms,
            "prompt_tokens": self.prompt,
            "output_tokens": self.output,
            "prefix_tokens": self.request.prefix_tokens,
            "text_prompt_tokens": getattr(self.request, "text_prompt_tokens", 0),
            "media_tokens": getattr(self.request, "media_tokens", 0),
            "media_encoded": self.media_encoded,
            "draft_mode": self.draft_mode,
            "draft_proposal_tokens": self.draft_proposal_tokens,
            "draft_proposed_total": self.draft_proposed_total,
            "draft_accepted_total": self.draft_accepted_total,
            "draft_rollback_tokens": self.draft_rollback_tokens,
            "draft_temporary_tokens": self.draft_temporary_tokens,
            "draft_fallback_tokens": self.draft_fallback_tokens,
            "draft_acceptance_source": self.draft_acceptance_source,
            "draft_steps": list(self.draft_steps),
            "computed_tokens": self.computed,
            "kv_blocks": self.kv_blocks,
            "generated_tokens": self.generated,
            "ttft_ms": ttft,
            "first_token_abs_ms": self.first_token_ms,
            "itl_ms": itl,
            "itl_scope": "engine_output_burst_interval",
            "token_itl_ms": token_itl,
            "token_event_times_ms": list(self.token_event_times_ms),
            "tpot_ms": tpot,
            "e2e_ms": (self.finish_ms - self.request.arrival_ms) if self.finish_ms is not None else None,
            "finish_ms": self.finish_ms,
            "first_token_ms": self.first_token_ms,
            "preemptions": self.preempted,
            "status": "served" if self.finished else "unserved",
            "unserved_reason": self.unserved_reason,
        }



class TokenLinearCostBackend:
    """Deterministic validation backend: Tbatch = base + per_token*tokens.

    This is an explicitly injected backend for scheduler/algorithm tests.  It
    is never presented as an A3 measured cost model.
    """

    def __init__(self, *, base_ms: float = 1.0, per_token_ms: float = 0.001) -> None:
        self.base_ms = float(base_ms)
        self.per_token_ms = float(per_token_ms)

    def iteration_duration_ms(self, work: Any, case: CaseSpec) -> float:
        return self.base_ms + self.per_token_ms * work.valid_tokens


def _backend_duration(cost_backend: Any, work: Any, case: CaseSpec) -> float:
    if cost_backend is None:
        raise A3InputError("cost_backend_missing")
    if hasattr(cost_backend, "iteration_duration_ms"):
        value = cost_backend.iteration_duration_ms(work, case)
    elif callable(cost_backend):
        value = cost_backend(work, case)
    else:
        raise A3InputError("cost_backend_protocol_invalid")
    if isinstance(value, Mapping):
        value = value.get("duration_ms")
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise A3InputError("cost_backend_duration_invalid") from exc
    if not math.isfinite(duration) or duration < 0:
        raise A3InputError("cost_backend_duration_invalid")
    return duration


__all__ = ["RequestState", "TokenLinearCostBackend", "with_media_runtime"]


def with_media_runtime(case: CaseSpec) -> CaseSpec:
    """Merge catalog media scheduling metadata into an already-built case.

    ``CaseSpec.from_mapping`` does this at the public parse boundary; this
    helper covers programmatic ``CaseSpec`` construction and is idempotent.
    The prompt position expansion itself happens in ``RequestSpec``.
    """
    from dataclasses import replace

    try:
        from .a3_catalog_media import media_runtime_metadata
    except ImportError:
        return case
    media_meta = media_runtime_metadata(case.workload)
    if not media_meta:
        return case
    merged = dict(case.runtime.metadata)
    merged.update(media_meta)
    return replace(
        case,
        runtime=replace(
            case.runtime, metadata=tuple(sorted(merged.items())),
        ),
    )


def runtime_with_draft_next_counts(
    runtime: Any,
    views: Sequence[Any],
    states: Sequence[Any],
) -> Any:
    """Attach the symbolic next-iteration candidate counts for catalog9.

    The scheduler remains the owner of configured/scheduled counts.  This
    function only decides whether another candidate block is useful after the
    current target body: zero when the request is on its final output token.
    """
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    if not metadata.get("draft_mode"):
        return runtime
    by_id = {
        str(getattr(state, "request").request_id): state for state in states
    }
    counts: dict[str, int] = {}
    for view in views:
        request_id = str(getattr(view, "request_id", ""))
        state = by_id.get(request_id)
        if state is None:
            continue
        remaining = max(0, int(getattr(state, "output", 0)) - int(state.generated))
        if bool(getattr(view, "draft_verify", False)):
            advance = min(
                remaining,
                int(getattr(view, "draft_accepted_tokens", 0) or 0) + 1,
            )
        elif bool(getattr(view, "samples", False)):
            advance = 1
        else:
            advance = 0
        remaining_after = max(0, remaining - advance)
        configured = max(
            0, int(getattr(state, "draft_proposal_tokens", 0) or 0)
        )
        can_verify_next = int(
            getattr(runtime, "max_num_batched_tokens", 0) or 0
        ) >= 2
        counts[request_id] = (
            configured if remaining_after >= 2 and can_verify_next else 0
        )
    metadata["draft_next_counts"] = json.dumps(counts, sort_keys=True)
    return replace(runtime, metadata=tuple(sorted(metadata.items())))
