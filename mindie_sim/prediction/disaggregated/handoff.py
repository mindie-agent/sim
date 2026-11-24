"""Cache handoff binding for prefill and decode.

[Role] Map published source ranges to physical destination shards and bind releases.
[Invariants] Handoff bytes and releases follow the physical owners and original
  versions.
[Timeouts & Locks] Finite spans and Engine dependencies for the current handoff.
[Side-Effect Policy] Queries endpoint memory and registers release callbacks on the
  shared Engine.
"""
from collections.abc import Mapping

from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.context_parallel import cp_interleave
from mindie_sim.schema.common import ModelingError

from ..pd_connector import PDHandoff, PDTransfer
from ..disaggregated.ranges import address, owned_runs, transfers_for_runs


def _local_head_spans(count, tp):
    """Physical KV-head ownership for one TP rank, including legal replicas."""
    del count, tp
    raise NotImplementedError(
        "LEGACY_PURGED: deleted cyclic modulo GQA head mapping (index*local)%count; "
        "implement contiguous r//R GQA replica-block head partition (S6-W5-10)"
    )


def _token_span(memory, index, tokens):
    """Actual half-open token rows owned by one physical rank (or the span)."""
    del memory, index, tokens
    raise NotImplementedError(
        "LEGACY_PURGED: deleted contiguous/modulo CP token slicing; implement exact "
        "DCP-interleaved MLA / contiguous GQA integer token ownership (S6-W5-10)"
    )


def _head_token_span(index, tokens, head_spans, cp):
    """Token rows paired with one rank's replicated head span."""
    del index, tokens, head_spans, cp
    raise NotImplementedError(
        "LEGACY_PURGED: deleted modulo head-token pairing; implement contiguous r//R GQA "
        "+ DCP-interleaved MLA rank ownership (S6-W5-10)"
    )


def _uncovered_intervals(covered, lo, hi):
    """Remaining half-open subranges of ``[lo, hi)`` after prior transfers."""
    pieces = [(int(lo), int(hi))]
    for held_lo, held_hi in covered:
        following = []
        for piece_lo, piece_hi in pieces:
            if held_hi <= piece_lo or held_lo >= piece_hi:
                following.append((piece_lo, piece_hi))
                continue
            if piece_lo < held_lo:
                following.append((piece_lo, min(piece_hi, held_lo)))
            if held_hi < piece_hi:
                following.append((max(piece_lo, held_hi), piece_hi))
        pieces = following
    return tuple((piece_lo, piece_hi) for piece_lo, piece_hi in pieces
                 if piece_hi > piece_lo)


def _published_for(published, pool, request, layer, component):
    """Read one truly named producer/output record supplied by T04/F-STATE."""
    if not published:
        return None
    table = published.get(pool, published) if isinstance(published, Mapping) else None
    if not isinstance(table, Mapping):
        return None
    for key in ((request, layer, component), (layer, component), component):
        try:
            value = table.get(key)
        except TypeError:
            continue
        if value is not None:
            return dict(value)
    return None


def _component_versions(memory, replica, request, layer, component):
    """Per-output version: named component binding first, request version next."""
    table = getattr(memory, "component_versions", None)
    if isinstance(table, Mapping):
        for key in ((replica, request, layer, component),
                    (request, layer, component)):
            try:
                value = table.get(key)
            except TypeError:
                continue
            if value is not None:
                return int(value)
    return int(memory.request_versions.get((replica, request), 0))


HEAD_COMPONENTS = frozenset({"gqa_kv", "window"})


def _pd_handoff_binding(request, tokens, placement, cases, memory, execution,
                        decode_local, volume, demands, shared_buffer_prefix=None,
                        shared_buffer_capacity=None, published=None):
    """Bind actual source/destination layer, component, head and token ranges."""
    del (
        request, tokens, placement, cases, memory, execution,
        decode_local, volume, demands, shared_buffer_prefix,
        shared_buffer_capacity, published,
    )
    raise NotImplementedError(
        "LEGACY_PURGED: deleted float PDTransfer emission, 1e-6 tolerance, empty-transfer "
        "bypass, TP2 GDN rank-0 source hotspot, and RD2H rank-0 hotspot; implement integer "
        "half-open [start, end) PDTransfer with exact 0 B conservation and TP-sharded "
        "recurrent/conv + MLA/GQA rank ownership (S6-W5-11)"
    )


def _consumer_release_done(engine, target, on_consumer, rank):
    if not target.done:
        engine.resolve(target)
    if on_consumer is not None:
        callback = on_consumer(rank)
        if callback is not None:
            return callback()
    return None


def _pending_release_targets(memory):
    """Real hold/consumer events currently retaining this service's ledgers.

    These are the only tails the serving loop may advance to when destination
    admission is waiting for a release.  An unrelated Engine timer is never a
    target.
    """
    events = []
    for ledger in memory.values():
        for holds in getattr(ledger, "async_consumer_holds", {}).values():
            for snapshot in holds.values():
                binding = dict(getattr(snapshot, "binding", ()) or ())
                event = binding.get("event")
                if event is not None and not getattr(event, "done", False):
                    events.append(event)
    return tuple(events)


def _register_consumer_release(engine, request, releases, device_complete,
                               on_consumer=None):
    """Bind the real decode device-complete event before Engine advance."""
    for rank, release in releases.items():
        if release.done:
            continue
        suffix = "" if len(releases) == 1 and rank == 1 else f":r{rank}"
        engine.submit(
            f"pd:{request}:consumer_boundary{suffix}",
            after=(device_complete,),
            metadata={"boundary": "pd_decode_consumer_release",
                      "request_id": request, "rank": rank},
            on_complete=lambda _event, target=release, rank=rank:
                _consumer_release_done(engine, target, on_consumer, rank))


class _CohortStageTopology:
    """Minimal disjoint stage-rank topology over one request's P or D shard cohort."""

    def __init__(self, parallel, *, rank_offset: int = 0) -> None:
        self._pp = max(1, int(parallel.pp))
        dcp_dp = max(1, int(getattr(parallel, "dcp_dp_width", 1) or 1))
        self._width = max(1, int(parallel.tp) * int(parallel.pcp) * dcp_dp)
        self._offset = int(rank_offset)

    def ranks(self, replica: int, stage: int) -> tuple[int, ...]:
        base = self._offset + (int(replica) * self._pp + int(stage)) * self._width
        return tuple(range(base, base + self._width))


def compute_pd_handoff_plan(
    prefill_case,
    decode_case,
    *,
    tokens: int,
    prefill_memory=None,
    decode_memory=None,
) -> dict:
    """Compute exact shard-aware PD KV/Indexer handoff bytes via ``_pd_handoff_binding``."""
    del prefill_case, decode_case, tokens, prefill_memory, decode_memory
    raise NotImplementedError(
        "LEGACY_PURGED: deleted orphaned post-hoc draft_kv_bytes residual & float handoff "
        "plan; emit explicit integer draft KV PDTransfer records inside PDHandoff gated by "
        "SpeculativeExecutionContract (S6-W5-12)"
    )
