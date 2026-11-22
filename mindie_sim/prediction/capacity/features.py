"""Runtime-aware batch/active/retiring capacity facts.

[Role]
  Extract the physical batch cache calculation shared by Memory and SLA so a
  shared prefix is charged once per DP replica, retiring waves do not multiply it
  again, and speculative verify windows use their real query width.

[Invariants]
  * Active and retiring windows are unions of physical cache ranges, not a
    constant bytes/token slope.
  * Shared prefix bytes are counted once; private suffix/state/draft bytes are
    per request.
  * Workspace and per-rank reserve checks remain the caller's responsibility.

[Timeouts & Locks]
  Bounded request/layer arithmetic; no I/O or locks.

[Side-Effect Policy]
  Read-only; returns plain dictionaries and immutable footprints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindie_sim.catalog.cache_ownership import (
    BatchCacheFootprint, batch_cache_footprint,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView


DECODE = "decode"


@dataclass(frozen=True)
class CapacityWindow:
    """Active/retiring physical cache peaks for one candidate batch."""

    active_bytes: float
    retiring_bytes: float
    wave_bytes: float
    peak_bytes: float
    footprint: BatchCacheFootprint


def _synthetic_views(sequences: int, context: int, tokens_per_seq: int,
                     prefix_tokens: int) -> tuple[RequestView, ...]:
    return tuple(
        RequestView(
            f"capacity-{index}", int(tokens_per_seq), int(context), True,
            DECODE, prefix_tokens=int(prefix_tokens))
        for index in range(int(sequences)))


def batch_cache_bytes(
    case: Any, *, sequences: int, context: int, tokens_per_seq: int,
    prefix_tokens: int = 0, phase: str = DECODE, padded: bool = True,
) -> BatchCacheFootprint:
    """Physical batch cache footprint for one DP replica."""
    if sequences < 1 or context < 0 or tokens_per_seq < 1:
        raise ModelingError("capacity_batch_geometry_invalid")
    views = _synthetic_views(sequences, context, tokens_per_seq, prefix_tokens)
    return batch_cache_footprint(case, requests=views, replica=0,
                                 phase=phase, padded=padded)


def capacity_window(
    case: Any, *, sequences: int, context: int, tokens_per_seq: int,
    prefix_tokens: int = 0, retiring_sequences: int = 0,
    lookahead_sequences: int = 0, phase: str = DECODE,
    padded: bool = True,
) -> CapacityWindow:
    """Active, retiring and lookahead peaks from the same physical ranges."""
    active = batch_cache_bytes(
        case, sequences=sequences, context=context, tokens_per_seq=tokens_per_seq,
        prefix_tokens=prefix_tokens, phase=phase, padded=padded)
    retiring = (batch_cache_bytes(
        case, sequences=max(1, int(retiring_sequences)), context=context,
        tokens_per_seq=tokens_per_seq, prefix_tokens=prefix_tokens,
        phase=phase, padded=padded) if int(retiring_sequences) > 0 else None)
    lookahead = (batch_cache_bytes(
        case, sequences=max(1, int(lookahead_sequences)), context=context,
        tokens_per_seq=tokens_per_seq, prefix_tokens=prefix_tokens,
        phase=phase, padded=padded) if int(lookahead_sequences) > 0 else None)
    active_total = active.total
    retiring_total = 0.0 if retiring is None else retiring.total
    lookahead_total = 0.0 if lookahead is None else lookahead.total
    # The live peak is one physical union over every concurrent window, so a
    # shared prefix is not charged again for each retiring wave.
    combined = batch_cache_bytes(
        case, sequences=sequences + max(0, int(retiring_sequences))
        + max(0, int(lookahead_sequences)), context=context,
        tokens_per_seq=tokens_per_seq, prefix_tokens=prefix_tokens,
        phase=phase, padded=padded)
    peak = max(active_total, combined.total)
    return CapacityWindow(
        float(active_total), float(retiring_total), float(lookahead_total),
        float(peak), active)


def speculative_window_tokens(runtime: Any, *, view: Any = None,
                              tokens_per_seq: int | None = None) -> int:
    """Resolved capacity query width (kept separate from accepted mean)."""
    from mindie_sim.catalog.speculative_state import resolve_speculative_query_tokens

    return resolve_speculative_query_tokens(
        runtime, view=view, tokens_per_seq=tokens_per_seq)
