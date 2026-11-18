"""Prefill/decode pools on the same causal engine.

[Role] Schedule prefill, destination admission, cache handoff and decode.
[Invariants] Destination KV is reserved before transfer. The source retains
  state until the transfer completes. TTFT and token intervals include actual
  arrival, queue and handoff times. Transfer bytes are physical destination
  shards, independent of overlap. No post-hoc latency addition.
[Timeouts & Locks] Finite requests and active transfers only; no external work.
[Side-Effect Policy] Mutates only per-run state.
"""
import math
from collections import defaultdict
from collections.abc import Mapping

from .attention import AttentionPolicy
from .capacity import Memory
from .cost import CostModel
from .dispatch import cohorts, submit_batch
from .engine import Engine
from .execution import Execution
from .frontend import Frontend
from .metrics import serving_boundaries, summarize
from .pd_connector import PDConnector, PDHandoff, PDTransfer
from .scheduler import Scheduler
from .structure import parse_case
from mindie_sim.schema.common import ModelingError

#: Bounded default diagnostic trace sizes for the direct PD entry.
DEFAULT_PD_TRACE_LIMIT = 2
DEFAULT_PD_EVENT_LIMIT = 256
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.coverage import combine_coverage


def _endpoint_execution(execution, memory, pool):
    """Attach the endpoint's own T08 ledger to its real T04 tasks.

    T04's per-task entry is ``Engine.submit(memory=...)``: every
    ``Execution.add``/``_start_program`` path forwards ``self.memory`` there,
    so each endpoint's Scheduler and Execution submit to the same T08 ledger
    while sharing one Engine.  ``Engine.bind_memory`` intentionally keeps one
    *default* ledger; constructing a second Execution with a differing default
    would replace that process-wide slot, so PD passes ``memory=None`` at
    construction and binds the endpoint ledger here.  This replaces the old
    name-prefix router and its decode default.
    """
    if memory is None:
        raise ModelingError("pd_endpoint_memory_required:" + pool)
    execution.memory = memory
    return memory


def _memory_binding_report(engine, memory, execution):
    """Expose the real per-endpoint ledgers without a name router or default."""
    return {
        "engine": id(engine),
        "same_engine": all(pool.engine is engine for pool in execution.values()),
        "pools": {name: id(ledger) for name, ledger in memory.items()},
        "executions": {name: id(pool.memory)
                       for name, pool in execution.items()},
        "bound": {name: execution[name].memory is memory[name]
                  for name in memory},
        "name_router": False,
        "default_ledger": None if engine.memory is None else id(engine.memory),
        "budget_versions": {name: int(getattr(ledger, "budget_version", 0))
                            for name, ledger in memory.items()},
        "grant_records": {name: len(getattr(ledger, "grant_records", {}))
                          for name, ledger in memory.items()},
    }


def _local_head_spans(count, tp):
    """Physical KV-head ownership for one TP rank, including legal replicas.

    ``local_kv_heads`` is ``max(1, ceil(count/tp))``.  When the TP degree
    exceeds the head count, several ranks hold the *same* head (GQA/MQA
    replication); the previous clipped arithmetic gave the extra ranks an empty
    span and silently dropped their real copy.  Ownership here is the local
    consecutive block starting at ``rank*local`` modulo the head count.
    """
    count = max(0, int(count))
    tp = max(1, int(tp))
    if count <= 0:
        return ()
    local = max(1, (count + tp - 1) // tp)
    spans = []
    for index in range(tp):
        start = (index * local) % count
        end = min(count, start + local)
        if end <= start:
            end = count
        spans.append((start, end))
    return tuple(spans)


def _token_span(memory, index, tokens):
    """Actual half-open token rows owned by one physical rank (or the span)."""
    tokens = max(0, int(tokens))
    owners = getattr(memory, "rank_token_owners", None)
    if owners is not None and 0 <= index < len(owners):
        lo, hi = owners[index]
        return (max(0, min(tokens, int(lo))), max(0, min(tokens, int(hi))))
    cp = max(1, int(memory.case.runtime.parallel.cp))
    pcp = max(1, int(memory.case.runtime.parallel.pcp))
    width = max(1, int(memory.case.runtime.parallel.tp) * pcp)
    if cp > 1:
        # Without a T05 owner partition, the declared CP/PCP axis still owns a
        # contiguous local row range; never claim the whole token span.
        axis = index % cp
        lo = tokens * axis // cp
        hi = tokens * (axis + 1) // cp
        return (lo, hi)
    return (0, tokens)


def _head_token_span(index, tokens, head_spans, cp):
    """Token rows paired with one rank's replicated head span.

    When TP exceeds the KV-head count several ranks hold the same head, while
    the CP/DCP axis splits tokens.  The rank grid must pair every head with
    every token shard: head replicas advance the head span, and the next rank
    slot inside that replica group advances the token shard.  Enumerating both
    axes from the same linear index leaves half of the ``(token, head)`` pairs
    uncovered.
    """
    tokens = max(0, int(tokens))
    cp = max(1, int(cp))
    distinct = len({(int(lo), int(hi)) for lo, hi in head_spans})
    slots = distinct or 1
    axis = (int(index) // slots) % cp
    return (tokens * axis // cp, tokens * (axis + 1) // cp)


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
    """Bind actual source/destination layer, component, head and token ranges.

    The transfer set is the intersection of the source's published component
    storage with the destination's real storage need.  Replicated KV heads
    (TP > KV heads) stay as separate destination copies; PP stages keep their
    own source rank; the sparse SFA connector contributes one Host RD2H copy
    and per-destination hot-HBM/index bytes instead of rebuilding the static
    dense total.  Each transfer also carries the source/destination
    storage/generation and half-open byte range later consumed by T08/T09.
    """
    from collections import defaultdict as _defaultdict
    prefill_exec = execution["prefill"]
    decode_exec = execution["decode"]
    prefill_rep = placement["prefill", request]
    decode_rep = placement["decode", request]
    prefill_case, decode_case = cases["prefill"], cases["decode"]
    prefill_p, decode_p = prefill_case.runtime.parallel, decode_case.runtime.parallel
    prefill_mem, decode_mem = memory["prefill"], memory["decode"]
    source_version = prefill_mem.request_versions.get((prefill_rep, request), 0)
    destination_version = decode_mem.request_versions.get((decode_rep, request), 0)
    sparse = getattr(decode_mem, "sparse_offload", None)
    offload = bool(sparse) and not bool(
        dict(sparse.get("options", {})).get("keep_device_kv_cache"))
    offload_layers = set(sparse.get("offload_layers", ())) if offload else set()
    use_copy_sfa = bool(dict(sparse.get("options", {})).get(
        "use_fused_copy_sfa", True)) if offload else False
    topk_buffer = int(sparse.get("topk_buffer_size", 0)) if offload else 0
    hot_tokens = (min(int(tokens), topk_buffer)
                  if use_copy_sfa and topk_buffer > 0 else 0)
    hot_ratio = (float(hot_tokens) / float(tokens)
                 if tokens > 0 and hot_tokens > 0 else 0.0)
    transfers = []
    destination_bytes = _defaultdict(float)
    host_bytes = _defaultdict(float)
    source_ranks = set()
    destination_ranks = set()
    source_offsets = _defaultdict(float)
    destination_offsets = _defaultdict(float)

    def emit(source_rank, destination_rank, component, payload, *,
             source_stage, destination_stage, layer, source_span,
             destination_span, source_tokens, destination_tokens,
             tier="hbm", source_storage, destination_storage,
             source_component=None, destination_component=None):
        payload = float(payload)
        if payload <= 0.0:
            return
        source_component = source_component or component
        destination_component = destination_component or component
        src_rec = _published_for(published, "prefill", request, layer,
                                 source_component)
        dst_rec = _published_for(published, "decode", request, layer,
                                 destination_component)
        producer = (str(src_rec.get("output")) if src_rec and src_rec.get("output")
                    else f"prefill:{request}:s{source_stage}:l{layer}:{source_component}")
        output = (str(dst_rec.get("output")) if dst_rec and dst_rec.get("output")
                  else f"decode:{request}:s{destination_stage}:l{layer}:{destination_component}")
        src_version = (int(src_rec["version"]) if src_rec and src_rec.get("version")
                       is not None else _component_versions(
                           prefill_mem, prefill_rep, request, layer,
                           source_component))
        dst_version = (int(dst_rec["version"]) if dst_rec and dst_rec.get("version")
                       is not None else _component_versions(
                           decode_mem, decode_rep, request, layer,
                           destination_component))
        if src_rec and src_rec.get("storage"):
            src_storage = str(src_rec["storage"])
        else:
            src_storage = source_storage
        if dst_rec and dst_rec.get("storage"):
            dst_storage = str(dst_rec["storage"])
        else:
            dst_storage = destination_storage
        src_key = (source_rank, src_storage)
        dst_key = (destination_rank, dst_storage, tier)
        src_offset = source_offsets[src_key]
        dst_offset = destination_offsets[dst_key]
        source_offsets[src_key] += payload
        destination_offsets[dst_key] += payload
        transfers.append(PDTransfer(
            source_rank, destination_rank, payload,
            source_stage=source_stage,
            destination_stage=destination_stage,
            source_layers=(layer,), destination_layers=(layer,),
            component=component,
            source_version=src_version,
            destination_version=dst_version,
            source_heads=source_span,
            destination_heads=destination_span,
            source_token_span=source_tokens,
            destination_token_span=destination_tokens,
            producer=producer, output=output,
            destination_tier=tier,
            source_storage=src_storage,
            destination_storage=dst_storage,
            source_byte_range=(src_offset, src_offset + payload),
            destination_byte_range=(dst_offset, dst_offset + payload),
            source_scope=(str(src_rec.get("scope")) if src_rec else
                          f"prefill:r{prefill_rep}:s{source_stage}:rank{source_rank}"),
            destination_scope=(str(dst_rec.get("scope")) if dst_rec else
                               f"decode:r{decode_rep}:s{destination_stage}:rank{destination_rank}"),
        ))
        if tier == "host":
            host_bytes[destination_rank] += payload
        else:
            destination_bytes[destination_rank] += payload
        source_ranks.add(source_rank)
        destination_ranks.add(destination_rank)

    for layer in prefill_case.model.layer_plans:
        source_stage = prefill_mem.layer_stage[layer.index]
        destination_stage = decode_mem.layer_stage[layer.index]
        source = tuple(prefill_exec.topology.ranks(prefill_rep, source_stage))
        destination = tuple(decode_exec.topology.ranks(decode_rep, destination_stage))
        if not source or not destination:
            raise ModelingError("pd_handoff_missing_stage_ranks:" + request)
        num_heads = int(getattr(layer, "num_kv_heads", 0) or 0)
        source_spans = _local_head_spans(num_heads, prefill_p.tp)
        target_spans = _local_head_spans(num_heads, decode_p.tp)
        source_owners = getattr(prefill_mem, "rank_token_owners", None)
        target_owners = getattr(decode_mem, "rank_token_owners", None)

        def views(case, parallel, ranks, ledger, owners, head_spans):
            if owners is not None:
                # A T05 owner partition is the exact physical fact.
                spans = tuple(
                    _token_span(ledger, index, tokens)
                    for index in range(len(ranks)))
                component_owners = owners
            else:
                spans = tuple(
                    _head_token_span(index, tokens, head_spans, parallel.cp)
                    for index in range(len(ranks)))
                # Feed the same exact partition to the cache ledger: an even
                # ``rows / cp`` split would bill 160/160 B for the [0,2) and
                # [2,5) shards instead of the exact 128/192 B they own.
                component_owners = spans
            result = []
            for index, rank in enumerate(ranks):
                components = layer_cache_components(
                    case.model, layer, parallel, tokens, capacity_padded=False,
                    interleave=case.runtime.kv_block_size,
                    rank=index, owner_spans=component_owners)
                result.append((index, rank, components, spans[index]))
            return result

        src_views = views(prefill_case, prefill_p, source, prefill_mem,
                          source_owners, source_spans)
        dst_views = views(decode_case, decode_p, destination, decode_mem,
                          target_owners, target_spans)
        components = {name for _, _, values, _ in src_views for name in values}
        components |= {name for _, _, values, _ in dst_views for name in values}
        gqa_components = {name for name in components
                          if name in HEAD_COMPONENTS and num_heads > 0}
        split_kv = (num_heads > 0
                    and _published_for(published, "prefill", request,
                                       layer.index, "gqa_k") is not None
                    and _published_for(published, "prefill", request,
                                       layer.index, "gqa_v") is not None)
        for component in sorted(components):
            window = int(getattr(layer, "window", 0) or 0)
            windowed = component == "window" or (
                component in {"gqa_kv", "mla_kv"} and window > 0)
            window_start = max(0, tokens - window) if windowed else 0
            if component in gqa_components:
                for dest_index, dest_rank, dest_values, dest_tokens in dst_views:
                    need = float(dest_values.get(component, 0.0))
                    dspan = target_spans[dest_index % max(1, len(target_spans))]
                    d_lo, d_hi = dest_tokens
                    if windowed:
                        d_lo, d_hi = max(window_start, d_lo), min(tokens, d_hi)
                    if (need <= 0.0 or dspan[1] <= dspan[0]
                            or d_hi <= d_lo):
                        continue
                    # A destination row needs a per-head payload only for the
                    # token rows a source rank actually owns.  The head and
                    # token axes are independent intersections.
                    per_head_token = (need / float(dspan[1] - dspan[0])
                                      / float(d_hi - d_lo))
                    covered: dict[int, list[tuple[int, int]]] = {}
                    for src_index, src_rank, src_values, src_tokens in src_views:
                        sspan = source_spans[src_index % max(1, len(source_spans))]
                        common = [head for head in range(dspan[0], dspan[1])
                                  if sspan[0] <= head < sspan[1]]
                        if not common:
                            continue
                        s_lo, s_hi = src_tokens
                        if windowed:
                            s_lo, s_hi = (max(window_start, s_lo),
                                          min(tokens, s_hi))
                        lo, hi = max(s_lo, d_lo), min(s_hi, d_hi)
                        if hi <= lo:
                            continue
                        src_storage = f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}"
                        dst_storage = f"pd:kv:{request}:decode:r{decode_rep}:rank{dest_rank}"
                        grouped: dict[tuple[int, int], list[int]] = {}
                        for head in common:
                            for piece_lo, piece_hi in _uncovered_intervals(
                                    covered.get(head, ()), lo, hi):
                                grouped.setdefault(
                                    (piece_lo, piece_hi), []).append(head)
                                covered.setdefault(head, []).append(
                                    (piece_lo, piece_hi))
                        for (piece_lo, piece_hi), heads in grouped.items():
                            payload = (per_head_token * (piece_hi - piece_lo)
                                       * len(heads))
                            if split_kv:
                                for suffix in ("k", "v"):
                                    emit(src_rank, dest_rank, f"gqa_{suffix}",
                                         payload * 0.5,
                                         source_stage=source_stage,
                                         destination_stage=destination_stage,
                                         layer=layer.index, source_span=sspan,
                                         destination_span=dspan,
                                         source_tokens=(piece_lo, piece_hi),
                                         destination_tokens=(piece_lo, piece_hi),
                                         source_storage=src_storage,
                                         destination_storage=dst_storage,
                                         source_component=f"gqa_{suffix}",
                                         destination_component=f"gqa_{suffix}")
                            else:
                                emit(src_rank, dest_rank, component, payload,
                                     source_stage=source_stage,
                                     destination_stage=destination_stage,
                                     layer=layer.index, source_span=sspan,
                                     destination_span=dspan,
                                     source_tokens=(piece_lo, piece_hi),
                                     destination_tokens=(piece_lo, piece_hi),
                                     source_storage=src_storage,
                                     destination_storage=dst_storage)
                    missing = [head for head in range(dspan[0], dspan[1])
                               if sum(piece_hi - piece_lo for piece_lo, piece_hi
                                      in covered.get(head, ())) < d_hi - d_lo]
                    if missing:
                        raise ModelingError(
                            "pd_handoff_source_head_uncovered:"
                            f"{request}:{layer.index}:{dest_rank}:{missing}")
                continue
            host_done = False
            for dest_index, dest_rank, dest_values, dest_tokens in dst_views:
                need = float(dest_values.get(component, 0.0))
                if windowed:
                    dest_tokens = (max(window_start, dest_tokens[0]),
                                   min(tokens, dest_tokens[1]))
                d_lo, d_hi = dest_tokens
                dst_storage = f"pd:kv:{request}:decode:r{decode_rep}:rank{dest_rank}"
                if offload and component == "mla_kv" and layer.index in offload_layers:
                    if not host_done:
                        src_index, src_rank, src_values, src_tokens = src_views[0]
                        if windowed:
                            src_tokens = (max(window_start, src_tokens[0]),
                                          min(tokens, src_tokens[1]))
                        host_payload = float(src_values.get("mla_kv", 0.0))
                        if host_payload > 0.0:
                            host_storage = (f"pd:host:{request}:decode:"
                                            f"r{decode_rep}")
                            emit(src_rank, dest_rank, component, host_payload,
                                 source_stage=source_stage,
                                 destination_stage=destination_stage,
                                 layer=layer.index,
                                 source_span=(0, num_heads),
                                 destination_span=(0, num_heads),
                                 source_tokens=src_tokens,
                                 destination_tokens=dest_tokens,
                                 tier="host",
                                 source_storage=f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}",
                                 destination_storage=host_storage)
                        host_done = True
                    hot = need * hot_ratio
                    if hot > 0.0:
                        src_index, src_rank, src_values, src_tokens = src_views[0]
                        if windowed:
                            src_tokens = (max(window_start, src_tokens[0]),
                                          min(tokens, src_tokens[1]))
                        emit(src_rank, dest_rank, component, hot,
                             source_stage=source_stage,
                             destination_stage=destination_stage,
                             layer=layer.index,
                             source_span=(0, num_heads),
                             destination_span=(0, num_heads),
                             source_tokens=src_tokens,
                             destination_tokens=dest_tokens,
                             source_storage=f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}",
                             destination_storage=dst_storage)
                    continue
                if need <= 0.0:
                    continue
                if component in {"mla_kv", "window"} and d_hi > d_lo:
                    # A dense non-GQA cache row is owned by the token axis.
                    # Intersect each source shard with the destination's own
                    # rows; TP replicas holding the same interval deduplicate.
                    per_token = need / float(d_hi - d_lo)
                    covered_tokens: list[tuple[int, int]] = []
                    for _src_index, src_rank, _src_values, src_tokens in src_views:
                        s_lo, s_hi = src_tokens
                        if windowed:
                            s_lo, s_hi = (max(window_start, s_lo),
                                          min(tokens, s_hi))
                        lo, hi = max(s_lo, d_lo), min(s_hi, d_hi)
                        if hi <= lo:
                            continue
                        for piece_lo, piece_hi in _uncovered_intervals(
                                covered_tokens, lo, hi):
                            emit(src_rank, dest_rank, component,
                                 per_token * (piece_hi - piece_lo),
                                 source_stage=source_stage,
                                 destination_stage=destination_stage,
                                 layer=layer.index,
                                 source_span=(0, num_heads),
                                 destination_span=(0, num_heads),
                                 source_tokens=(piece_lo, piece_hi),
                                 destination_tokens=(piece_lo, piece_hi),
                                 source_storage=f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}",
                                 destination_storage=dst_storage)
                            covered_tokens.append((piece_lo, piece_hi))
                    covered_rows = sum(piece_hi - piece_lo
                                       for piece_lo, piece_hi in covered_tokens)
                    if covered_rows < d_hi - d_lo:
                        raise ModelingError(
                            "pd_handoff_token_range_uncovered:"
                            f"{request}:{layer.index}:{component}:{dest_rank}")
                    continue
                remaining = need
                for src_index, src_rank, src_values, src_tokens in src_views:
                    if remaining <= 1e-9:
                        break
                    if windowed:
                        src_tokens = (max(window_start, src_tokens[0]),
                                      min(tokens, src_tokens[1]))
                    available = float(src_values.get(component, 0.0))
                    take = min(available, remaining)
                    if take <= 0.0:
                        continue
                    emit(src_rank, dest_rank, component, take,
                         source_stage=source_stage,
                         destination_stage=destination_stage,
                         layer=layer.index,
                         source_span=(0, num_heads),
                         destination_span=(0, num_heads),
                         source_tokens=src_tokens,
                         destination_tokens=dest_tokens,
                         source_storage=f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}",
                         destination_storage=dst_storage)
                    remaining -= take
                if remaining > max(1e-9, need * 1e-12):
                    raise ModelingError(
                        "pd_handoff_component_range_uncovered:"
                        f"{request}:{layer.index}:{component}:{remaining}")
    if not transfers:
        return PDHandoff.build(
            request, (), memory_work=tuple(sorted(demands.items()))), (), ()
    expected = sum(destination_bytes.values()) + sum(host_bytes.values())
    if abs(float(volume) - expected) > max(1e-6, abs(float(volume)) * 1e-9):
        raise ModelingError(
            "pd_handoff_physical_range_not_conserved:"
            f"{request}:{volume}!={expected}")
    if shared_buffer_prefix is not None:
        buffers = {
            rank: f"{shared_buffer_prefix}:rank{rank}"
            for rank in destination_ranks}
    else:
        buffers = {
            rank: f"pd:kv:{request}:decode:r{decode_rep}:rank{rank}"
            for rank in destination_ranks}
    handoff = PDHandoff.build(
        request, transfers, memory_work=tuple(sorted(demands.items())),
        receive_capacity_bytes=dict(destination_bytes),
        receive_buffer_ids=buffers,
        receive_pool_capacity_bytes=(
            {rank: float(shared_buffer_capacity)
             for rank in destination_ranks}
            if shared_buffer_capacity is not None else None),
        receive_host_bytes=dict(host_bytes))
    return handoff, tuple(sorted(source_ranks)), tuple(sorted(destination_ranks))


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


def predict_pd(raw, *, primitives=None, recipes=None, submission=None, memory_policy=None,
               lower=None, trace_limit=DEFAULT_PD_TRACE_LIMIT,
               event_limit=DEFAULT_PD_EVENT_LIMIT, scope="serving", phase=None,
               tokens=None, context=None, sequences=None, published=None):
    # The direct PD entry enforces the same structural timing gate as predict.
    from .unmodeled_gate import assert_timing_supported
    assert_timing_supported(raw, entry="predict")
    if scope == "body":
        from .api import predict
        return predict(raw, scope=scope, phase=phase, tokens=tokens, context=context,
                       sequences=sequences, primitives=primitives, recipes=recipes,
                       lower=lower, event_limit=event_limit)
    if scope != "serving" or any(v is not None for v in (phase, tokens, context, sequences)):
        raise ModelingError("prediction_geometry_requires_body_scope")
    pools = raw["pools"]
    common = {key: value for key, value in raw.items() if key not in {"pools", "connector"}}
    cases = {name: parse_case({**common, **pools[name], "model": raw["model"], "workload": raw["workload"]})
             for name in ("prefill", "decode")}
    if cases["prefill"].model.is_standalone_draft:
        raise ModelingError("draft_requires_target_model:attach_auxiliary_to_target;catalog_inspection_available")
    identifiers = [r.request_id for r in cases["prefill"].workload.requests]
    if len(identifiers) != len(set(identifiers)):
        raise ModelingError("request_ids_must_be_unique")
    connector = PDConnector(raw["connector"])
    engine = Engine(connector.resources(), trace_limit=event_limit)
    memory = {name: Memory(case, memory_policy) for name, case in cases.items()}
    client_declared = [case.runtime.serving_io.get("client_delivery_ms") is not None
                       for case in cases.values()]
    if any(client_declared) and not all(client_declared):
        raise ModelingError("pd_client_delivery_boundary_requires_both_pool_declarations")
    client_tails = {}
    execution = {name: Execution(case, CostModel(case, primitives), recipes=recipes,
                                submission=submission, engine=engine, namespace=name + ":",
                                include_serving_io=True, client_tails=client_tails,
                                memory_policy=memory[name].policy,
                                memory=None)
                 for name, case in cases.items()}
    for name in cases:
        _endpoint_execution(execution[name], memory[name], name)
    lowerers = {name: lower or Frontend(attention_policy=AttentionPolicy(**case.runtime.attention_policy))
                for name, case in cases.items()}
    schedulers, states, placement = {}, [], {}
    for name, case in cases.items():
        groups = [[] for _ in range(case.runtime.parallel.dp)]
        chain_placement = {}
        for index, request in enumerate(case.workload.requests):
            replica = chain_placement.setdefault(request.chain_id, index % len(groups)) if request.chain_id else index % len(groups)
            groups[replica].append(request)
            placement[name, request.request_id] = replica
        schedulers[name] = [Scheduler(case, group, memory[name], replica,
                           eligible=lambda state, pool=name: state.location == pool,
                           handoff_after_prefill=name == "prefill")
                           for replica, group in enumerate(groups)]
    states = [state for scheduler in schedulers["prefill"] for state in scheduler.states]
    shared = {state.request.request_id: state for state in states}
    for scheduler in schedulers["decode"]:
        scheduler.states = [shared[state.request.request_id] for state in scheduler.states]
        scheduler.by_id = {state.request.request_id: state for state in scheduler.states}
        scheduler.successors = {key: shared[value.request.request_id] for key, value in scheduler.successors.items()}
    engine.now = min((r.request.arrival_ms for r in states), default=0)
    flights, transfers, traces = {}, {}, []
    # Real event handles for the three distinct PD boundaries.  They are not
    # fabricated sleeps: source is the prefill device-complete event, grant is
    # the decode admission event, and release is resolved by the first actual
    # decode batch that consumes the handoff.
    source_ready, consumer_release = {}, {}
    consumer_callbacks = {}
    source_release_state = {}
    consumer_registered = set()
    handoff_records = []
    handoff_bytes, handoffs, iterations = 0, 0, 0
    while not all(state.done for state in states):
        # Admit to the destination before making any external cache service
        # ready. Waiting transfers retain source ownership and exert backpressure.
        for state in states:
            if state.done or state.location != "waiting_transfer":
                continue
            request = state.request.request_id
            target = placement["decode", request]
            active = sum(s.location in {"decode", "transferring"} and not s.done
                         for s in schedulers["decode"][target].states)
            if active >= cases["decode"].runtime.max_num_seqs:
                continue
            if not memory["decode"].resize(target, request, state.computed + 1):
                continue
            source_event = source_ready.get(request)
            if source_event is None:
                raise ModelingError(
                    "pd_handoff_missing_source_ready:" + request)
            p = cases["decode"].runtime.parallel
            if memory["decode"].sparse_offload:
                volume, decode_local = memory["decode"].pd_handoff_volume_and_stage_hbm(state.computed)
            else:
                decode_local = memory["decode"].cache_bytes(state.computed, padded=False)
                volume = sum(decode_local) * p.tp * p.pcp
            demands = {}
            for pool in ("prefill", "decode"):
                replica = placement[pool, request]
                local = (decode_local if pool == "decode"
                         else memory[pool].cache_bytes(state.computed, padded=False))
                for stage, stage_bytes in enumerate(local):
                    roles = execution[pool].stage_names[replica, stage]
                    demands[roles["hbm"]] = stage_bytes
            handoff, source_ranks, destination_ranks = _pd_handoff_binding(
                request, state.computed, placement, cases, memory, execution,
                decode_local, volume, demands,
                shared_buffer_prefix=(connector.receive_buffer_id
                                      if connector.shared_receive_buffer else None),
                shared_buffer_capacity=connector.receive_capacity_bytes,
                published=published)
            if not handoff.transfers:
                transfers[request] = engine.submit(
                    f"pd:{request}", owner=f"request:{request}",
                    metadata={"boundary": "pd_empty_handoff",
                              "request_id": request})
                handoff_records.append({
                    "request_id": request, "transport_binding": "empty",
                    "source_ranks": list(source_ranks),
                    "destination_ranks": list(destination_ranks),
                    "transfers": [], "receive_buffer_ids": {},
                    "receive_capacity_bytes": {},
                    "memory_work": dict(demands),
                    "chunk_bytes": connector.chunk_bytes,
                    "transport_event": transfers[request]})
                state.location = "decode"
                continue
            storage_holds = {}
            if set(source_ranks) & set(destination_ranks):
                # The fixture's two pools may occupy the same physical ranks;
                # keep the existing pool-endpoint connector identity for that
                # local case.  Disjoint pools use the actual rank binding below.
                grant = engine.submit(
                    f"pd:{request}:destination_grant",
                    owner=f"decode:r{target}",
                    metadata={"boundary": "pd_destination_admission",
                              "request_id": request,
                              "physical_ranks": list(destination_ranks),
                              "tokens": state.computed + 1})
                release = engine.future(
                    f"pd:{request}:consumer_release")
                consumer_release[request] = {1: release}
                transfers[request] = connector.submit(
                    engine, "pd:" + request, volume, demands,
                    ready=source_event, destination_ready=grant,
                    release_events={1: (release,)})
                transport_binding = "pool_endpoint_overlapping_ranks"
            else:
                grant_by_rank = {
                    rank: engine.submit(
                        f"pd:{request}:destination_grant:r{rank}",
                        owner=f"decode:r{target}",
                        metadata={"boundary": "pd_destination_admission",
                                  "request_id": request, "rank": rank,
                                  "receive_buffer_id": dict(
                                      handoff.receive_buffer_ids)[rank],
                                  "tokens": state.computed + 1})
                    for rank in destination_ranks}
                releases = {
                    rank: (engine.future(
                        f"pd:{request}:consumer_release:r{rank}"),)
                    for rank in destination_ranks}
                consumer_release[request] = {
                    rank: events[0] for rank, events in releases.items()}
                destination_ranges = defaultdict(list)
                for item in handoff.hbm_transfers():
                    if item.destination_byte_range:
                        destination_ranges[item.destination_rank].append(
                            item.destination_byte_range)
                release_callbacks = {}
                for rank, events in releases.items():
                    buffer_id = dict(handoff.receive_buffer_ids)[rank]
                    consumer_name = f"pd:{request}:decode:r{rank}"
                    generation = memory["decode"].request_versions.get(
                        (target, request), 0)
                    hold_event = events[0]
                    fallback = ((0.0, float(handoff.capacity_for(rank) or 0.0)),)
                    held = memory["decode"].hold_for_async_consumer(
                        target, request, consumer_name,
                        storage=buffer_id,
                        ranges=tuple(destination_ranges.get(rank, fallback)),
                        generation=generation, event=hold_event)
                    if not held:
                        raise ModelingError(
                            f"pd_decode_memory_hold_rejected:{request}:r{rank}")
                    storage_holds[rank] = buffer_id
                    keep = (memory["decode"], target, request, consumer_name,
                            generation, hold_event)
                    def release_hold(keep=keep):
                        pool, replica, req, consumer, gen, event = keep
                        return pool.release_async_consumer(
                            replica, req, consumer,
                            generation=gen, event=event)
                    # Fixed (pool, replica, request, rank, generation, event)
                    # identity; never re-derived from a display name later.
                    consumer_callbacks[request, rank] = release_hold
                    release_callbacks[rank] = release_hold
                source_released = [False]
                source_release_state[request] = source_released
                source_callbacks = {}
                prefill_replica = placement["prefill", request]
                for source_rank in source_ranks:
                    def release_source(rank=source_rank, done=source_released,
                                       replica=prefill_replica):
                        if done[0]:
                            return
                        done[0] = True
                        memory["prefill"].release(replica, request)
                    source_callbacks[source_rank] = release_source
                transfers[request] = connector.submit_handoff(
                    engine, handoff,
                    source_ready_by_rank={rank: source_event
                                          for rank in source_ranks},
                    destination_ready_by_rank=grant_by_rank,
                    release_events=releases,
                    source_release_callbacks=source_callbacks,
                    release_callbacks_by_rank=release_callbacks,
                    name="pd:" + request)
                transport_binding = "actual_rank"
            handoff_records.append({
                "request_id": request,
                "transport_binding": transport_binding,
                "source_ranks": list(source_ranks),
                "destination_ranks": list(destination_ranks),
                "transfers": [
                    {"source_rank": item.source_rank,
                     "destination_rank": item.destination_rank,
                     "payload_bytes": item.payload_bytes,
                     "source_stage": item.source_stage,
                     "destination_stage": item.destination_stage,
                     "source_layers": list(item.source_layers),
                     "destination_layers": list(item.destination_layers),
                     "component": item.component,
                     "source_version": item.source_version,
                     "destination_version": item.destination_version,
                     "source_heads": list(item.source_heads),
                     "destination_heads": list(item.destination_heads),
                     "source_token_span": list(item.source_token_span),
                     "destination_token_span": list(item.destination_token_span),
                     "producer": item.producer,
                     "output": item.output,
                     "destination_tier": item.destination_tier,
                     "source_storage": item.source_storage,
                     "destination_storage": item.destination_storage,
                     "source_byte_range": list(item.source_byte_range),
                     "destination_byte_range": list(item.destination_byte_range),
                     "source_scope": item.source_scope,
                     "destination_scope": item.destination_scope}
                    for item in handoff.transfers],
                "receive_buffer_ids": {
                    str(rank): value for rank, value in
                    handoff.receive_buffer_ids},
                "receive_capacity_bytes": {
                    str(rank): value for rank, value in
                    handoff.receive_capacity_bytes},
                "receive_pool_capacity_bytes": {
                    str(rank): value for rank, value in
                    handoff.receive_pool_capacity_bytes},
                "receive_host_bytes": {
                    str(rank): value for rank, value in
                    handoff.receive_host_bytes},
                "host_bytes": sum(float(item.payload_bytes)
                                  for item in handoff.host_transfers()),
                "memory_work": dict(handoff.memory_work),
                "chunk_bytes": connector.chunk_bytes,
                "storage_holds": {str(rank): value
                                  for rank, value in storage_holds.items()},
                "transport_event": transfers[request],
            })
            state.location = "transferring"
            handoff_bytes += volume
            handoffs += 1
        for name, case in cases.items():
            for group in cohorts(case.runtime.parallel):
                if any((name, replica) in flights for replica in group):
                    continue
                pending_release = any(state.location in {"waiting_transfer", "transferring"}
                                      and not state.done for state in states)
                plans = {r: schedulers[name][r].schedule(
                    engine.now, defer_capacity_failure=name == "prefill" and pending_release)
                         if not schedulers[name][r].done else None
                         for r in group}
                if any(plans.values()):
                    steps = submit_batch(case, schedulers[name], plans, execution[name], lowerers[name], engine.now)
                    flights.update({(name, replica): step for replica, step in steps.items()})
                    if name == "decode":
                        for request, step in ((request, steps[replica])
                                              for replica in steps
                                              for request in steps[replica].commits):
                            pending = {
                                rank: release
                                for rank, release in (consumer_release.get(request) or {}).items()
                                if not release.done
                                and (request, rank) not in consumer_registered}
                            consumer_registered.update(
                                (request, rank) for rank in pending)
                            _register_consumer_release(
                                engine, request, pending, step.device_complete,
                                on_consumer=lambda rank, request=request,
                                registry=consumer_callbacks:
                                    registry.get((request, rank)))
                    iterations += len(steps)
                else:
                    for r in group:
                        if not schedulers[name][r].done and memory[name].has_delayed_releases(r):
                            pool_exec = execution[name]
                            prep = pool_exec.prepared.pop(r, None)
                            sched_ms = pool_exec.submission.scheduler_ms if pool_exec.include_submission else 0.0
                            comp_ms = pool_exec.submission.completion_ms if pool_exec.include_submission else 0.0
                            sched_ev = engine.submit(
                                f"{pool_exec.namespace}r{r}:retire_schedule:{iterations}",
                                {f"{pool_exec.namespace}scheduler@r{r}": sched_ms},
                                after=(prep,) if prep is not None else (),
                                release=engine.now, owner=f"replica:{r}",
                            )
                            comp_ev = engine.submit(
                                f"{pool_exec.namespace}r{r}:retire_complete:{iterations}",
                                after=(sched_ev,), latency=comp_ms, release=engine.now, owner=f"replica:{r}",
                            )
                            from .execution import ScheduledIteration
                            flights[name, r] = ScheduledIteration(
                                r, schedulers[name][r].iteration, comp_ev, engine.now,
                                "retire", 0, 0, None, "none", 0, {}, comp_ev, {}, {}, (),
                            )
        arrivals = [time for name, group in schedulers.items() for replica, scheduler in enumerate(group)
                    if (name, replica) not in flights if (time := scheduler.next_arrival(engine.now)) is not None]
        horizon = min(arrivals, default=math.inf)
        targets = (tuple(s.completion for s in flights.values())
                   + tuple(transfers.values())
                   + tuple(client_tails.values())
                   + _pending_release_targets(memory))
        if not targets:
            if math.isfinite(horizon):
                engine.advance(horizon=horizon)
                engine.now = horizon
                continue
            for state in states:
                if not state.done:
                    state.reason = "decode_destination_capacity"
                    for name in cases:
                        memory[name].release(placement[name, state.request.request_id], state.request.request_id)
            break
        engine.advance(until=targets, horizon=horizon)
        for key, step in list(flights.items()):
            if not step.completion.done:
                continue
            name, replica = key
            scheduler = schedulers[name][replica]
            scheduler.commit(step.completion.time, {key: event.time for key, event in step.commits.items()})
            if name == "prefill":
                for request in step.commits:
                    source_ready.setdefault(request, step.device_complete)
                for state in scheduler.states:
                    if not state.done and state.first_token_ms is not None and state.location == "prefill":
                        state.location = "waiting_transfer"
            if step.mode != "retire" and len(traces) < trace_limit:
                traces.append({"pool": name, **step.to_dict()})
            del flights[key]
        for request, event in list(transfers.items()):
            if event.done:
                # The actual-rank path releases the source exactly at its
                # per-rank last-use callback; the pool-endpoint path keeps its
                # existing whole-handoff retirement.
                if source_release_state.get(request) is None:
                    memory["prefill"].release(
                        placement["prefill", request], request)
                shared[request].location = "decode"
                del transfers[request]
    # A request that never reaches decode must not leave a transport credit
    # allocation or consumer-release task unresolved.  This explicit release
    # is recorded at the actual end of the serving loop, not as elapsed sleep.
    final_releases = []
    for request, releases in list(consumer_release.items()):
        for rank, release in releases.items():
            if release.done:
                continue
            suffix = ("" if len(releases) == 1 and rank == 1
                      else f":r{rank}")
            cancel = consumer_callbacks.get((request, rank))
            source_done = source_release_state.get(request)
            def finish(_event, target=release, cancel=cancel,
                       request=request, source_state=source_done):
                if not target.done:
                    engine.resolve(target)
                if cancel is not None:
                    cancel()
                if source_state is not None and not source_state[0]:
                    source_state[0] = True
                    memory["prefill"].release(
                        placement["prefill", request], request)
            final_releases.append(engine.submit(
                f"pd:{request}:final_release{suffix}",
                metadata={"boundary": "pd_unconsumed_final_release",
                          "request_id": request, "rank": rank},
                on_complete=finish))
    serving_end = engine.now
    # Complete only this service's causal tail.  An unrelated long timer is
    # not a target and stays pending; same-time service descendants (receive
    # release decisions, consumer cancellation) settle at serving_end.
    device_tail = (tuple(final_releases)
                   + tuple(consumer_release.get(request, {}).get(rank, None)
                           for request in consumer_release
                           for rank in consumer_release[request])
                   + tuple(transfers.values()))
    device_tail = tuple(event for event in device_tail if event is not None)
    if device_tail:
        engine.advance(until=device_tail, first=False, horizon=serving_end)
    engine.advance(horizon=serving_end)
    delivery_tail = tuple(event for event in client_tails.values() if event is not None)
    if delivery_tail:
        engine.advance(until=delivery_tail, first=False)
        engine.advance(horizon=engine.now)
    for ledger in memory.values():
        ledger.flush_delayed_releases()
    rows, boundaries = serving_boundaries(
        states, cases["prefill"].slo,
        host_results=any(pool.serving_io.enabled for pool in execution.values()),
        client_delivery=all(client_declared))
    return {
        "schema": "mindie-sim.predict/2", "deployment": "disaggregated",
        **combine_coverage(case.model for case in cases.values()),
        "status": "modeled" if all(state.finish_ms is not None for state in states) else "infeasible",
        "metrics": summarize(rows, cases["prefill"].slo, now_ms=serving_end),
        "metric_boundaries": boundaries,
        "throughput_window_boundary": "legacy_scheduler_completion",
        "requests": rows, "memory": {name: ledger.to_dict() for name, ledger in memory.items()},
        "execution": {**engine.summary(), "iterations": iterations,
                      "handoffs": handoffs, "handoff_bytes": handoff_bytes,
                      "connector": connector.summary(),
                      "memory_binding": _memory_binding_report(engine, memory, execution),
                      "pd_handoffs": [
                          {key: value for key, value in record.items()
                           if key != "transport_event"}
                          | {"complete_ms": record["transport_event"].time}
                          for record in handoff_records],
                      "submission": {name: pool.submission_summary()
                                     for name, pool in execution.items()},
                      "serving_io": {name: pool.serving_io.summary()
                                     for name, pool in execution.items()},
                      "step_summary": {name: pool.step_summary.to_dict()
                                       for name, pool in execution.items()},
                      "transfer_layout": "destination_shards"},
        "traces": traces, "events": list(engine.trace), "metric_boundary": "device_token_commit",
    }
