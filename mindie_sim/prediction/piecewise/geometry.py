"""Ragged request and bounded analysis-domain geometry.

[Role] Type the dynamic ``X`` input: per-request query/context ranges, explicit
  graph bucket/media/routing state and the bounded domain used for finite
  piecewise enumeration.
[Invariants] No implicit homogeneous approximation replaces a ragged vector.
  A graph bucket pads physical tokens, never valid attention pairs.  Domain
  misses raise.
[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Immutable records only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from mindie_sim.schema.common import ModelingError


def local_batch(concurrency: int, dp: int) -> int:
    """Requests per DP replica under even scheduler admission."""
    return max(1, math.ceil(max(1, int(concurrency)) / max(1, int(dp))))


def ep_batch(concurrency: int, dp: int, ep: int) -> int:
    """Total requests participating in one expert-parallel AllToAll cohort."""
    dp_n, ep_n = max(1, int(dp)), max(1, int(ep))
    return max(1, math.ceil(max(1, int(concurrency)) * min(dp_n, ep_n) / dp_n))


def clamp_graph_bucket(bucket: int, batch: int, dp: int, buckets: tuple[int, ...] | list[int] = ()) -> int:
    """Select a valid local graph capture bucket covering the per-DP-rank batch."""
    bucket = max(0, int(bucket))
    if bucket <= 0:
        return 0
    lb = max(1, int(batch))
    if buckets:
        for b in sorted(int(x) for x in buckets if int(x) > 0):
            if b >= lb:
                return b
        return 0
    if bucket < lb:
        return 0
    if int(dp) <= 1:
        return bucket
    local_cap = 1 << (lb - 1).bit_length()
    return min(bucket, max(lb, local_cap))


@dataclass(frozen=True)
class AnalysisDomain:
    max_tokens: int = 1_048_576
    max_batch: int = 4096
    max_q: int = 262_144
    max_s_kv: int = 1_048_576
    max_seq: int = 1_048_576
    max_query_tiles: int = 65536
    max_kv_tiles: int = 65536
    max_work: int = 1_000_000_000_000
    max_moe_rows: int = 1_048_576
    max_index_rows: int = 262_144
    max_media_tokens: int = 65536

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"analysis_domain_invalid:{name}")

    def bound(self, key: str) -> int:
        if key not in self.__dataclass_fields__:
            raise ModelingError(f"analysis_domain_key_unknown:{key}")
        return int(getattr(self, key))

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


def _normalize_query_ranges(ranges: Any, q_len: int) -> tuple[tuple[int, int], ...]:
    """Validate and canonicalise selected rows into sorted disjoint intervals.

    Merging touching intervals makes the physical selection independent of the
    caller's iteration order; the same rows always produce the same tile and
    causal-bound work.
    """
    parsed: list[tuple[int, int]] = []
    try:
        items = tuple(ranges)
    except TypeError:
        raise ModelingError(
            f"request_query_range_invalid:{ranges!r}:{q_len}") from None
    for item in items:
        try:
            start, end = item
        except (TypeError, ValueError):
            raise ModelingError(
                f"request_query_range_invalid:{item!r}:{q_len}") from None
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)):
            raise ModelingError(
                f"request_query_range_invalid:{start}:{end}:{q_len}")
        if start < 0 or start > end or end > q_len:
            raise ModelingError(
                f"request_query_range_invalid:{start}:{end}:{q_len}")
        parsed.append((start, end))
    parsed.sort()
    merged: list[tuple[int, int]] = []
    for start, end in parsed:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            if end > previous_end:
                merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return tuple(merged)


@dataclass(frozen=True)
class RequestGeometry:
    """One ragged request: query length, pre-iteration context and ranges."""

    q_len: int
    context: int
    phase: str = "decode"
    query_ranges: tuple[tuple[int, int], ...] = ()
    replica: int = 0
    draft_len: int = 1

    def __post_init__(self) -> None:
        for name in ("q_len", "context", "draft_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"request_geometry_invalid:{name}")
        if self.phase not in {"prefill", "decode", "draft", "draft_prefill", "verify"}:
            raise ModelingError(f"request_phase_invalid:{self.phase}")
        object.__setattr__(self, "query_ranges",
                           _normalize_query_ranges(self.query_ranges, self.q_len))

    @property
    def kv_len(self) -> int:
        return self.context + self.q_len

    @property
    def selected_q_len(self) -> int:
        """Valid query rows selected by ``query_ranges`` (all rows if empty)."""
        if not self.query_ranges:
            return self.q_len
        return sum(end - start for start, end in self.query_ranges)

    def pairs(self, *, window: int = 0, causal: bool = True) -> int:
        if not causal:
            return self.selected_q_len * self.kv_len
        ranges = self.query_ranges or ((0, self.q_len),)
        total = 0
        for start, end in ranges:
            total += _range_window_sum(start, end, self.context, window)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {"q_len": self.q_len, "context": self.context, "phase": self.phase,
                "query_ranges": [list(item) for item in self.query_ranges],
                "replica": self.replica, "draft_len": self.draft_len}


def _range_window_sum(start: int, end: int, context: int, window: int) -> int:
    """Exact sum over query positions of valid causal key counts."""
    if end <= start:
        return 0
    if window <= 0:
        return (end - start) * context + end * (end + 1) // 2 - start * (start + 1) // 2
    split = max(start, min(end, window - context - 1))
    ramp = (split - start) * context + split * (split + 1) // 2 - start * (start + 1) // 2
    return ramp + (end - split) * window


@dataclass(frozen=True)
class WorkloadGeometry:
    """Ragged request vector and declared graph/media state for one slice."""

    requests: tuple[RequestGeometry, ...]
    graph_bucket: int = 0
    cohort_batch: int = 0
    physical_tokens: int = 0
    media_tokens: int = 0
    routing: tuple[tuple[int, tuple[int, ...]], ...] = ()
    phase: str = ""
    local_batch: int = 0

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        if not requests:
            raise ModelingError("workload_requests_empty")
        if not all(isinstance(request, RequestGeometry) for request in requests):
            raise ModelingError("workload_request_record_invalid")
        object.__setattr__(self, "requests", requests)
        for name in ("graph_bucket", "cohort_batch", "physical_tokens", "media_tokens", "local_batch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"workload_geometry_invalid:{name}")
        object.__setattr__(self, "routing",
                           tuple((int(layer), tuple(counts)) for layer, counts in self.routing))

    @classmethod
    def homogeneous(cls, batch: int, q_len: int, s_kv: int, *,
                    phase: str = "decode", **kwargs: Any) -> "WorkloadGeometry":
        if batch < 1 or q_len < 1 or s_kv < q_len:
            raise ModelingError("workload_homogeneous_invalid")
        return cls(tuple(RequestGeometry(q_len, s_kv - q_len, phase=phase)
                         for _ in range(batch)), phase=phase, **kwargs)

    @property
    def batch(self) -> int:
        return len(self.requests)

    @property
    def tokens(self) -> int:
        return sum(request.q_len for request in self.requests)

    @property
    def selected_tokens(self) -> int:
        """Valid query rows after applying every request's query ranges."""
        return sum(request.selected_q_len for request in self.requests)

    @property
    def s_sum(self) -> int:
        return sum(request.kv_len for request in self.requests)

    @property
    def s_max(self) -> int:
        return max(request.kv_len for request in self.requests)

    @property
    def q_max(self) -> int:
        return max(request.q_len for request in self.requests)

    @property
    def q2_sum(self) -> int:
        return sum(request.q_len * request.q_len for request in self.requests)

    @property
    def effective_tokens(self) -> int:
        if self.graph_bucket:
            return max(self.tokens, self.graph_bucket)
        return self.physical_tokens or self.tokens

    @property
    def phase_name(self) -> str:
        return self.phase or self.requests[0].phase

    def pairs(self, *, window: int = 0, causal: bool = True) -> int:
        return sum(request.pairs(window=window, causal=causal) for request in self.requests)

    def routing_for(self, layer: int) -> tuple[int, ...] | None:
        for key, counts in self.routing:
            if key == layer:
                return counts
        return None

    def expanded_to(self, batch: int) -> "WorkloadGeometry":
        """Expand request vector to ``batch`` while keeping cohort metadata and ragged distribution."""
        batch = max(1, int(batch))
        if batch == self.batch:
            return self
        shapes = {(r.q_len, r.context, r.phase, r.query_ranges, r.draft_len) for r in self.requests}
        loc = self.local_batch or self.batch
        if len(shapes) > 1:
            if batch < self.batch or batch % self.batch != 0:
                raise ModelingError(f"workload_ragged_cohort_expansion_ambiguous:{self.batch}->{batch}")
            reps = batch // self.batch
            return replace(self, requests=tuple(replace(r) for _ in range(reps) for r in self.requests),
                           cohort_batch=batch, local_batch=loc)
        proto = self.requests[0]
        return replace(self, requests=tuple(replace(proto) for _ in range(batch)),
                       cohort_batch=batch, local_batch=loc)

    def for_subgraph(self, kind: str) -> "WorkloadGeometry":
        """Lift EP-wide subgraphs (routed MoE F3 and prefill F2) to cohort batch."""
        if kind == "F3" or (kind == "F2" and self.phase_name == "prefill"):
            return self.expanded_to(self.cohort_batch or self.batch)
        return self

    def bindings(self) -> dict[str, Any]:
        return {
            "T": self.effective_tokens, "T_VALID": self.tokens,
            "T_SELECTED": self.selected_tokens,
            "selected_tokens": self.selected_tokens,
            "B": self.batch, "EP_B": self.cohort_batch or self.batch,
            "Q": self.q_max, "S": self.s_max,
            "S_SUM": self.s_sum, "Q2": self.q2_sum,
            "PAIRS": self.pairs(), "PAIRS_NC": self.pairs(causal=False),
            "BUCKET": self.graph_bucket or self.effective_tokens,
            "MEDIA": self.media_tokens, "media_tokens": self.media_tokens,
            "tokens": self.effective_tokens, "valid_tokens": self.tokens,
            "batch": self.batch, "q_max": self.q_max, "s_max": self.s_max,
            "s_sum": self.s_sum, "q2": self.q2_sum,
        }

    def validate(self, domain: AnalysisDomain) -> None:
        if self.batch > domain.max_batch:
            raise ModelingError(f"analysis_domain_miss:batch:{self.batch}>{domain.max_batch}")
        if self.tokens > domain.max_tokens:
            raise ModelingError(f"analysis_domain_miss:tokens:{self.tokens}>{domain.max_tokens}")
        if self.physical_tokens > domain.max_tokens:
            raise ModelingError(
                "analysis_domain_miss:physical_tokens:"
                f"{self.physical_tokens}>{domain.max_tokens}")
        if self.effective_tokens > domain.max_tokens:
            raise ModelingError(
                f"analysis_domain_miss:effective_tokens:"
                f"{self.effective_tokens}>{domain.max_tokens}")
        if self.media_tokens > domain.max_media_tokens:
            raise ModelingError(
                f"analysis_domain_miss:media:{self.media_tokens}>{domain.max_media_tokens}")
        for request in self.requests:
            if request.q_len > domain.max_q:
                raise ModelingError(f"analysis_domain_miss:q:{request.q_len}>{domain.max_q}")
            if request.kv_len > domain.max_s_kv:
                raise ModelingError(
                    f"analysis_domain_miss:s_kv:{request.kv_len}>{domain.max_s_kv}")
        if self.graph_bucket > domain.max_seq:
            raise ModelingError(f"analysis_domain_miss:bucket:{self.graph_bucket}>{domain.max_seq}")

    def to_dict(self) -> dict[str, Any]:
        return {"requests": [request.to_dict() for request in self.requests],
                "graph_bucket": self.graph_bucket,
                "physical_tokens": self.physical_tokens,
                "media_tokens": self.media_tokens, "tokens": self.tokens}
