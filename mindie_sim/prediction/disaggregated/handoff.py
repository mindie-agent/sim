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
from mindie_sim.schema.common import ModelingError

from ..pd_connector import PDHandoff, PDTransfer
from ..disaggregated.ranges import address, owned_runs, transfers_for_runs


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
    transfers = []
    destination_bytes = _defaultdict(float)
    host_bytes = _defaultdict(float)
    source_ranks = set()
    destination_ranks = set()
    plane_bases, source_extents = {}, _defaultdict(float)
    destination_planes, destination_offsets = {}, _defaultdict(float)

    def emit(source_rank, destination_rank, component, payload, *,
             source_stage, destination_stage, layer, source_span,
             destination_span, source_tokens, destination_tokens,
             tier="hbm", source_storage, destination_storage,
             source_component=None, destination_component=None,
             source_size, source_displacement=0, destination_size, destination_displacement=0):
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
        src_offset = address(plane_bases, source_extents,
                             (source_rank, src_storage, src_version),
                             (layer, source_component), source_size,
                             source_displacement, payload)
        dst_key = (destination_rank, dst_storage, dst_version, tier)
        dst_offset = address(destination_planes, destination_offsets, dst_key,
                             (layer, destination_component), destination_size,
                             destination_displacement, payload)
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
        _bind_layer_handoff(
            layer, request, tokens, published, emit, prefill_exec, decode_exec,
            prefill_rep, decode_rep, prefill_case, decode_case, prefill_p, decode_p,
            prefill_mem, decode_mem, offload, offload_layers, hot_tokens)
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


def _handoff_rank_views(case, layer, parallel, ranks, ledger, owners,
                        head_spans, has_head_axis, tokens):
    if owners is not None:
        # A T05 owner partition is the exact physical fact.
        spans = tuple(
            _token_span(ledger, index, tokens)
            for index in range(len(ranks)))
        component_owners = owners
    else:
        spans = tuple(
            _head_token_span(index, tokens, head_spans if has_head_axis else (), parallel.cp)
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


def _bind_layer_handoff(
    layer, request, tokens, published, emit, prefill_exec, decode_exec,
    prefill_rep, decode_rep, prefill_case, decode_case, prefill_p, decode_p,
    prefill_mem, decode_mem, offload, offload_layers, hot_tokens,
):
    source_stage = prefill_mem.layer_stage[layer.index]
    destination_stage = decode_mem.layer_stage[layer.index]
    source = tuple(prefill_exec.topology.ranks(prefill_rep, source_stage))
    destination = tuple(decode_exec.topology.ranks(decode_rep, destination_stage))
    if not source or not destination:
        raise ModelingError("pd_handoff_missing_stage_ranks:" + request)
    num_heads = int(getattr(layer, "num_kv_heads", 0) or 0)
    has_head_axis = layer.attention in {"gqa", "projection_only"}
    source_spans = _local_head_spans(num_heads, prefill_p.tp)
    target_spans = _local_head_spans(num_heads, decode_p.tp)
    source_owners = getattr(prefill_mem, "rank_token_owners", None)
    target_owners = getattr(decode_mem, "rank_token_owners", None)


    src_views = _handoff_rank_views(
        prefill_case, layer, prefill_p, source, prefill_mem,
        source_owners, source_spans, has_head_axis, tokens)
    dst_views = _handoff_rank_views(
        decode_case, layer, decode_p, destination, decode_mem,
        target_owners, target_spans, has_head_axis, tokens)
    components = {name for _, _, values, _ in src_views for name in values}
    components |= {name for _, _, values, _ in dst_views for name in values}
    gqa_components = {name for name in HEAD_COMPONENTS
                      if has_head_axis and num_heads > 0}
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
            _emit_head_transfers(
                emit, request, component, layer, source_stage, destination_stage,
                prefill_rep, decode_rep, src_views, dst_views, source_spans,
                target_spans, windowed, window_start, tokens, split_kv)
            continue
        row_component = component in {"mla_kv", "window", "compressed_kv", "index_k"}
        if row_component:
            _emit_row_transfers(
                emit, request, component, layer, source_stage, destination_stage,
                prefill_rep, decode_rep, src_views, dst_views, prefill_case,
                decode_case, prefill_p, decode_p, tokens, window_start, num_heads,
                offload, offload_layers, hot_tokens, destination)
            continue
        for dest_index, dest_rank, dest_values, dest_tokens in dst_views:
            need = float(dest_values.get(component, 0))
            dst_storage = f"pd:kv:{request}:decode:r{decode_rep}:rank{dest_rank}"
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
                     destination_storage=dst_storage, source_size=available,
                     destination_size=need, destination_displacement=need - remaining)
                remaining -= take
            if remaining > max(1e-9, need * 1e-12):
                raise ModelingError(
                    "pd_handoff_component_range_uncovered:"
                    f"{request}:{layer.index}:{component}:{remaining}")


def _emit_head_transfers(
    emit, request, component, layer, source_stage, destination_stage,
    prefill_rep, decode_rep, src_views, dst_views, source_spans, target_spans,
    windowed, window_start, tokens, split_kv,
):
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
                spans_all_rows = (piece_lo, piece_hi) == (s_lo, s_hi)
                spans_all_rows &= (piece_lo, piece_hi) == (d_lo, d_hi)
                consecutive = heads[-1] - heads[0] + 1 == len(heads)
                groups = ([heads] if spans_all_rows and consecutive
                          else ([head] for head in heads))
                planes = ("gqa_k", "gqa_v") if split_kv else (component,)
                for group in groups:
                    first = (group[0] - sspan[0]) * (s_hi - s_lo)
                    offset = first + piece_lo - s_lo
                    for plane in planes:
                        unit = per_head_token / len(planes)
                        emit(src_rank, dest_rank, plane,
                             unit * (piece_hi - piece_lo) * len(group),
                             source_stage=source_stage,
                             destination_stage=destination_stage,
                             layer=layer.index, source_span=sspan,
                             destination_span=dspan,
                             source_tokens=(piece_lo, piece_hi),
                             destination_tokens=(piece_lo, piece_hi),
                             source_storage=src_storage,
                             destination_storage=dst_storage,
                             source_size=src_values[component] / len(planes),
                             source_displacement=unit * offset,
                             destination_size=need / len(planes),
                             destination_displacement=unit * (
                                 (group[0] - dspan[0]) * (d_hi - d_lo)
                                 + piece_lo - d_lo))
        missing = [head for head in range(dspan[0], dspan[1])
                   if sum(piece_hi - piece_lo for piece_lo, piece_hi
                          in covered.get(head, ())) < d_hi - d_lo]
        if missing:
            raise ModelingError(
                "pd_handoff_source_head_uncovered:"
                f"{request}:{layer.index}:{dest_rank}:{missing}")


def _emit_row_transfers(
    emit, request, component, layer, source_stage, destination_stage,
    prefill_rep, decode_rep, src_views, dst_views, prefill_case, decode_case,
    prefill_p, decode_p, tokens, window_start, num_heads, offload,
    offload_layers, hot_tokens, destination,
):
    def manifest(case, parallel, entries):
        return [(rank, owned_runs(component, layer, case.model, parallel,
                                  index, span, tokens, window_start),
                 values.get(component, 0))
                for index, rank, values, span in entries]

    sources = manifest(prefill_case, prefill_p, src_views)
    destinations = manifest(decode_case, decode_p, dst_views)
    sparse_layer = offload and component == "mla_kv" and layer.index in offload_layers
    receivers = []
    if sparse_layer:
        receivers.append((destination[0], ((window_start, tokens),),
                          "host", f"pd:host:{request}:decode:r{decode_rep}"))
    for rank, runs, need in destinations:
        if need <= 0:
            continue
        if sparse_layer:
            runs = tuple((max(lo, tokens - hot_tokens), hi) for lo, hi in runs
                         if hi > max(lo, tokens - hot_tokens))
        storage = f"pd:kv:{request}:decode:r{decode_rep}:rank{rank}"
        receivers.append((rank, runs, "hbm", storage))
    ratio = (max(1, layer.compress_ratio, layer.index_kpool)
             if component == "index_k" else max(1, layer.compress_ratio)
             if component == "compressed_kv" else 1)
    for dst_rank, runs, tier, storage in receivers:
        target_rows = sum(hi - lo for lo, hi in runs)
        received = 0
        for src_rank, lo, hi, size, offset, unit in transfers_for_runs(sources, runs):
            token_span = (lo * ratio, hi * ratio)
            emit(src_rank, dst_rank, component, (hi - lo) * unit,
                 source_stage=source_stage, destination_stage=destination_stage,
                 layer=layer.index, source_span=(0, num_heads),
                 destination_span=(0, num_heads), source_tokens=token_span,
                 destination_tokens=token_span, tier=tier,
                 source_storage=f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}",
                 destination_storage=storage, source_size=size,
                 source_displacement=offset, destination_size=target_rows * unit,
                 destination_displacement=received * unit)
            received += hi - lo


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


def _union_interval_length(intervals) -> float:
    """Exact total length of the union of half-open ``(lo, hi)`` byte intervals."""
    ordered = sorted(
        (float(lo), float(hi)) for lo, hi in intervals if float(hi) > float(lo)
    )
    if not ordered:
        return 0.0
    total = 0.0
    cur_lo, cur_hi = ordered[0]
    for lo, hi in ordered[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    return total + (cur_hi - cur_lo)


def compute_pd_handoff_plan(
    prefill_case,
    decode_case,
    *,
    tokens: int,
    prefill_memory=None,
    decode_memory=None,
) -> dict:
    """Compute exact shard-aware PD KV/Indexer handoff bytes via ``_pd_handoff_binding``.

    Accounts for TP, DCP (including ``dcp_dp_width > 1`` cohorts), PCP, PP,
    sliding-window clipping, MLA/GQA/Indexer components, and sparse offload,
    returning both direct D2D transfer totals and deduplicated unique KV bytes
    for three-tier HBM/DRAM/SSD KV pooling.
    """
    from collections import defaultdict as _defaultdict
    from types import SimpleNamespace
    from ..capacity import Memory
    from ..structure import parse_case

    p_case = parse_case(prefill_case)
    d_case = parse_case(decode_case)
    p_mem = prefill_memory if prefill_memory is not None else Memory(p_case)
    d_mem = decode_memory if decode_memory is not None else Memory(d_case)
    n_tokens = max(0, int(tokens))
    if n_tokens <= 0:
        return {
            "total_transfer_bytes": 0.0,
            "unique_kv_bytes": 0.0,
            "hbm_transfer_bytes": 0.0,
            "host_transfer_bytes": 0.0,
            "per_layer_bytes": {},
            "tail_layer_bytes": 0.0,
            "source_rank_bytes": {},
            "destination_rank_bytes": {},
            "unique_source_rank_bytes": {},
            "source_ranks": (),
            "destination_ranks": (),
            "transfer_count": 0,
        }
    p_topo = _CohortStageTopology(p_case.runtime.parallel, rank_offset=0)
    p_span = p_topo._pp * p_topo._width
    d_topo = _CohortStageTopology(d_case.runtime.parallel, rank_offset=p_span)
    execution = {
        "prefill": SimpleNamespace(topology=p_topo),
        "decode": SimpleNamespace(topology=d_topo),
    }
    d_par = d_case.runtime.parallel
    d_cohort_width = d_par.tp * d_par.pcp * max(1, int(d_par.dcp_dp_width))
    if d_mem.sparse_offload:
        volume, decode_local = d_mem.pd_handoff_volume_and_stage_hbm(n_tokens)
        draft_kv_bytes = 0.0
    else:
        target_stage = [0.0] * max(1, int(d_par.pp))
        for layer in d_case.model.layer_plans:
            comps = layer_cache_components(
                d_case.model, layer, d_par, n_tokens,
                capacity_padded=False, interleave=d_case.runtime.kv_block_size,
            )
            target_stage[d_mem.layer_stage[layer.index]] += sum(
                float(v) for v in comps.values()
            )
        decode_local = tuple(target_stage)
        volume = float(sum(decode_local)) * float(d_cohort_width)
        full_decode_bytes = float(sum(d_mem.cache_bytes(n_tokens, padded=False))) * float(d_cohort_width)
        both_draft = bool(
            dict(p_case.runtime.metadata).get("draft_proposal_tokens", 0)
            and dict(d_case.runtime.metadata).get("draft_proposal_tokens", 0)
        )
        draft_kv_bytes = max(0.0, full_decode_bytes - volume) if both_draft else 0.0
    req_id = "__pd_handoff__"
    placement = {("prefill", req_id): 0, ("decode", req_id): 0}
    handoff, src_ranks, dst_ranks = _pd_handoff_binding(
        req_id, n_tokens, placement,
        {"prefill": p_case, "decode": d_case},
        {"prefill": p_mem, "decode": d_mem},
        execution, decode_local, volume, {},
    )
    src_bytes: dict[int, float] = _defaultdict(float)
    dst_bytes: dict[int, float] = _defaultdict(float)
    per_layer: dict[int, float] = _defaultdict(float)
    unique_ranges: dict[tuple[int, str], list[tuple[float, float]]] = _defaultdict(list)
    hbm_total = 0.0
    host_total = 0.0
    for item in handoff.transfers:
        payload = float(item.payload_bytes)
        src_bytes[int(item.source_rank)] += payload
        dst_bytes[int(item.destination_rank)] += payload
        if item.destination_tier == "host":
            host_total += payload
        else:
            hbm_total += payload
        for layer_idx in item.source_layers or (0,):
            per_layer[int(layer_idx)] += payload / float(max(1, len(item.source_layers or (0,))))
        if item.source_byte_range:
            unique_ranges[(int(item.source_rank), str(item.source_storage))].append(
                (float(item.source_byte_range[0]), float(item.source_byte_range[1]))
            )
    unique_by_src: dict[int, float] = _defaultdict(float)
    for (s_rank, _storage), spans in unique_ranges.items():
        unique_by_src[s_rank] += _union_interval_length(spans)
    total_bytes = hbm_total + host_total
    unique_total = sum(unique_by_src.values()) if unique_by_src else total_bytes
    tail_bytes = float(per_layer[max(per_layer)]) if per_layer else 0.0
    return {
        "total_transfer_bytes": float(total_bytes),
        "unique_kv_bytes": float(unique_total),
        "draft_kv_bytes": float(draft_kv_bytes),
        "hbm_transfer_bytes": float(hbm_total),
        "host_transfer_bytes": float(host_total),
        "per_layer_bytes": {k: float(v) for k, v in sorted(per_layer.items())},
        "tail_layer_bytes": float(tail_bytes),
        "source_rank_bytes": {k: float(v) for k, v in sorted(src_bytes.items())},
        "destination_rank_bytes": {k: float(v) for k, v in sorted(dst_bytes.items())},
        "unique_source_rank_bytes": {k: float(v) for k, v in sorted(unique_by_src.items())},
        "source_ranks": tuple(int(r) for r in src_ranks),
        "destination_ranks": tuple(int(r) for r in dst_ranks),
        "transfer_count": len(handoff.transfers),
    }
