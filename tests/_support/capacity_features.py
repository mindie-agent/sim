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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mindie_sim.catalog.cache import (
    layer_cache_components,
    layer_stage_map,
)
from mindie_sim.catalog.cache_ownership import (
    DEFAULT_INTERLEAVE,
    SHARING_SHARED,
    STATE_COMPONENTS,
    BatchCacheFootprint,
    _active_draft_kind,
    _union_elementary_spans,
    batch_cache_footprint,
    cache_component_ranges,
)
from mindie_sim.catalog.cache_pages import CachePages
from mindie_sim.catalog.cache_pools import (
    _ceil_div,
    _draft_is_dcp_sharded,
    _draft_row_bytes_int,
)
from mindie_sim.catalog.context_parallel import (
    cp_interleave,
    dcp_indexer_sharded,
    layer_dcp_factors,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView

from mindie_sim.prediction.capacity.cache import _DCP_KV_SHARDED_COMPONENTS


DECODE = "decode"


@dataclass(frozen=True)
class CapacityWindow:
    """Active/retiring physical cache peaks for one candidate batch."""

    active_bytes: float
    retiring_bytes: float
    wave_bytes: float
    peak_bytes: float
    footprint: BatchCacheFootprint


def _resolve_views(
    case: Any,
    sequences: int,
    context: int,
    tokens_per_seq: int,
    prefix_tokens: int,
    *,
    phase: str = DECODE,
    requests: Sequence[RequestView] | None = None,
    prefix_cache_keys: Sequence[str] | None = None,
    prefix_groups: int | None = None,
) -> tuple[RequestView, ...]:
    if requests is not None:
        return tuple(requests)
    workload_reqs = tuple(getattr(getattr(case, "workload", None), "requests", ()) or ())
    if prefix_groups is not None and int(prefix_groups) < 1:
        raise ModelingError("capacity_prefix_groups_invalid")
    src_prefixes = [int(getattr(req, "prefix_tokens", 0) or 0) for req in workload_reqs]
    heterogeneous_prefixes = len(set(src_prefixes)) > 1
    views: list[RequestView] = []
    for index in range(int(sequences)):
        src = workload_reqs[index % len(workload_reqs)] if workload_reqs else None
        if prefix_cache_keys:
            p_key = str(prefix_cache_keys[index % len(prefix_cache_keys)] or "")
        elif prefix_groups is not None:
            p_key = f"group-{index % int(prefix_groups)}"
        else:
            p_key = str(getattr(src, "prefix_cache_key", "") or "") if src is not None else ""
        if int(prefix_tokens) <= 0:
            req_prefix = 0
        elif heterogeneous_prefixes and src is not None:
            req_prefix = min(int(prefix_tokens), int(getattr(src, "prefix_tokens", 0) or 0))
        else:
            req_prefix = int(prefix_tokens)
        views.append(RequestView(
            f"capacity-{index}", int(tokens_per_seq), int(context), True,
            str(phase), prefix_tokens=req_prefix, prefix_cache_key=p_key,
        ))
    return tuple(views)


def _page_group_stage_costs(group: Any, stage_map: tuple[int, ...]) -> list[tuple[int, float]]:
    meta = float(group.metadata_bytes_per_page)
    if group.bytes_per_group_page:
        return [(int(stage_map[group.layers[0]]), float(group.bytes_per_group_page) + meta)]
    per_layer = float(group.bytes_per_layer_page) + meta
    return [(int(stage_map[layer]), per_layer) for layer in group.layers]


def _accumulate_uncovered_ranges(
    case: Any,
    views: Sequence[RequestView],
    *,
    phase: str,
    padded: bool,
    stage_map: tuple[int, ...],
    covered: Mapping[int, set[str]],
    shared_mode: bool,
    shared_once: list[float],
    private: list[float],
    state: list[float],
    draft: list[float],
    has_aux_group: bool,
) -> None:
    from mindie_sim.schema.runtime_features import canonical_prefix_cache_key

    model, runtime = case.model, case.runtime
    parallel = runtime.parallel
    interleave = cp_interleave(runtime)
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    is_prefill = str(phase or DECODE).strip().lower() == "prefill"
    shared_spans: dict[tuple[int, str, str, int, int], list[tuple[int, int, float]]] = {}
    draft_kind = "" if has_aux_group else _active_draft_kind(model, runtime)
    for view in views:
        length = max(0, int(getattr(view, "kv_len", 0) or 0))
        q_tok = 1 if is_prefill else max(1, int(getattr(view, "q_len", 1) or 1))
        declared = max(0, min(int(getattr(view, "prefix_tokens", 0) or 0), length))
        generation = int(getattr(view, "prefix_generation", 0) or 0)
        p_key = (
            canonical_prefix_cache_key("shared", getattr(view, "prefix_cache_key", ""))
            if shared_mode else f"private:{getattr(view, 'request_id', '')}"
        )
        for plan in model.layer_plans:
            layer_cov = covered.get(plan.index, ())
            ranges = cache_component_ranges(
                model, plan, parallel, sequence_tokens=length,
                prefix_tokens=declared if shared_mode else 0,
                interleave=interleave, rank=None, capacity_padded=padded,
                runtime=runtime, query_tokens=q_tok, block_size=block_size,
                generation=generation, phase=phase,
            )
            stage = int(stage_map[plan.index])
            for item in ranges:
                if item.component in layer_cov:
                    continue
                if item.sharing == SHARING_SHARED:
                    span_len = max(1, item.end_token - item.start_token)
                    rate = float(item.allocated_bytes) / float(span_len)
                    s_key = (stage, item.component, p_key, generation, plan.index)
                    shared_spans.setdefault(s_key, []).append(
                        (item.start_token, item.end_token, rate)
                    )
                else:
                    val = float(item.allocated_bytes)
                    private[stage] += val
                    if item.component in STATE_COMPONENTS:
                        state[stage] += val
        if draft_kind:
            from mindie_sim.catalog.cache import stage_cache_components

            d_stages = stage_cache_components(
                model, parallel, length, capacity_padded=padded,
                draft_kind=draft_kind, interleave=interleave,
                block_size=block_size, runtime=runtime,
                query_tokens=q_tok, phase=phase,
            )
            draft[-1] += float(d_stages[-1].get("draft_kv", 0.0))
            d_state = float(d_stages[-1].get("draft_state", 0.0))
            draft[-1] += d_state
            state[-1] += d_state
    for (stage, _comp, _p_key, _gen, _layer), spans in shared_spans.items():
        shared_once[stage] += _union_elementary_spans(spans)


def _explicit_page_group_footprint(
    case: Any,
    views: Sequence[RequestView],
    *,
    phase: str,
    padded: bool,
) -> tuple[BatchCacheFootprint, CachePages] | None:
    if not padded:
        return None
    model, runtime = case.model, case.runtime
    if dict(getattr(model, "metadata", ()) or ()).get("cache_allocator") == "paired_sfa":
        return None
    raw_groups = tuple(dict(getattr(runtime, "memory", ()) or ()).get("cache_page_groups") or ())
    if not raw_groups:
        return None
    from mindie_sim.schema.runtime_features import canonical_prefix_cache_key

    parallel = runtime.parallel
    stages = max(1, int(parallel.pp))
    stage_map = layer_stage_map(model, parallel, runtime)
    max_q = max((max(1, int(getattr(v, "q_len", 1) or 1)) for v in views), default=1)
    comp_names = {
        plan.index: set(layer_cache_components(
            model, plan, parallel, runtime.max_model_len,
            capacity_padded=False, runtime=runtime, query_tokens=max_q,
        ))
        for plan in model.layer_plans
    }
    pages_obj = CachePages(raw_groups, comp_names, stage_map)
    shared_mode = getattr(runtime, "prefix_cache_mode", "private") == "shared"
    shared_once = [0.0] * stages
    private = [0.0] * stages
    state = [0.0] * stages
    draft = [0.0] * stages
    boundary = [0.0] * stages
    seen_shared_pages: dict[tuple[str, int, int], set[int]] = {}
    has_aux = any(group.auxiliary for group in pages_obj.groups)
    for view in views:
        length = max(0, int(getattr(view, "kv_len", 0) or 0))
        declared = max(0, min(int(getattr(view, "prefix_tokens", 0) or 0), length))
        generation = int(getattr(view, "prefix_generation", 0) or 0)
        p_key = (
            canonical_prefix_cache_key("shared", getattr(view, "prefix_cache_key", ""))
            if shared_mode else f"private:{getattr(view, 'request_id', '')}"
        )
        for group_idx, group in enumerate(pages_obj.groups):
            total_pages = group.pages(length, runtime.max_model_len)
            stage_costs = _page_group_stage_costs(group, stage_map)
            if group.fixed_pages:
                for st, cost in stage_costs:
                    val = float(total_pages) * cost
                    if group.auxiliary:
                        draft[st] += val
                    else:
                        private[st] += val
                        state[st] += val
                continue
            tpp = max(1, int(group.tokens_per_page))
            if group.resident_tokens is not None and length > group.resident_tokens:
                win_start = length - int(group.resident_tokens)
                sh_lo = _ceil_div(win_start, tpp)
                sh_hi = min(declared, length) // tpp
                shareable_ids = (
                    range(sh_lo, sh_hi)
                    if (shared_mode and sh_hi > sh_lo and group.allocation == "live")
                    else range(0)
                )
            else:
                eff_len = min(length, group.resident_tokens) if group.resident_tokens is not None else length
                eff_decl = min(declared, eff_len)
                shareable_cap = max(0, total_pages - group.recycling_pages)
                sh_count = (
                    min(shareable_cap, eff_decl // tpp)
                    if (shared_mode and eff_decl > 0 and group.allocation == "live")
                    else 0
                )
                shareable_ids = range(sh_count)
            shared_count = len(shareable_ids)
            if shared_count > 0:
                seen = seen_shared_pages.setdefault((p_key, generation, group_idx), set())
                new_ids = set(shareable_ids) - seen
                if new_ids:
                    for st, cost in stage_costs:
                        shared_once[st] += float(len(new_ids)) * cost
                    seen.update(new_ids)
            priv_pages = max(0, total_pages - shared_count)
            for st, cost in stage_costs:
                if group.auxiliary:
                    draft[st] += float(priv_pages) * cost
                else:
                    private[st] += float(priv_pages) * cost
                    if group.recycling_pages > 0:
                        state[st] += float(min(priv_pages, group.recycling_pages)) * cost
    _accumulate_uncovered_ranges(
        case, views, phase=phase, padded=padded, stage_map=stage_map,
        covered=pages_obj.covered, shared_mode=shared_mode,
        shared_once=shared_once, private=private, state=state, draft=draft,
        has_aux_group=has_aux,
    )
    totals = tuple(shared_once[s] + private[s] + draft[s] + boundary[s] for s in range(stages))
    footprint = BatchCacheFootprint(
        tuple(shared_once), tuple(private), tuple(state),
        tuple(draft), tuple(boundary), totals,
        requests=len(views), stages=stages,
    )
    return footprint, pages_obj


def _dcp_dp_cohort_draft_extra(
    case: Any,
    views: Sequence[RequestView],
    *,
    phase: str,
    padded: bool,
    dp_width: int,
) -> float:
    from mindie_sim.catalog.draft import optional_draft_spec
    from mindie_sim.catalog.draft_geometry import draft_layer_plans
    from mindie_sim.catalog.draft_spec import effective_draft_context_cap
    from mindie_sim.catalog.cache import _context_shard_blocks, uses_inherited_draft_cache

    model, runtime = case.model, case.runtime
    parallel = runtime.parallel
    draft_kind = _active_draft_kind(model, runtime)
    if not draft_kind or getattr(model, "auxiliary", None) is None:
        return 0.0
    if dict(getattr(model, "metadata", ()) or ()).get("cache_allocator") == "paired_sfa":
        return 0.0
    draft = optional_draft_spec(model, draft_kind)
    draft_cap = effective_draft_context_cap(draft, runtime)
    interleave = cp_interleave(runtime)
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    sharded_idx = dcp_indexer_sharded(model, runtime)
    is_prefill = str(phase or DECODE).strip().lower() == "prefill"
    extra = 0.0
    if uses_inherited_draft_cache(model, draft_kind):
        for view in views:
            length = max(0, int(getattr(view, "kv_len", 0) or 0))
            eff_len = min(length, draft_cap) if draft_cap > 0 else length
            q_tok = 1 if is_prefill else max(1, int(getattr(view, "q_len", 1) or 1))
            for d_plan in draft_layer_plans(model, draft_kind):
                _, _, d_dp = layer_dcp_factors(d_plan, parallel)
                if d_dp <= 1:
                    continue
                comps = layer_cache_components(
                    model, d_plan, parallel, eff_len, capacity_padded=padded,
                    interleave=interleave, block_size=block_size,
                    runtime=runtime, query_tokens=q_tok, phase=phase,
                )
                for name, val in comps.items():
                    if name in _DCP_KV_SHARDED_COMPONENTS or (name == "index_k" and sharded_idx):
                        extra += float(val) * float(d_dp - 1)
    else:
        row_bytes = _draft_row_bytes_int(model, draft, tp=parallel.tp)
        draft_sharded = _draft_is_dcp_sharded(
            model, parallel, draft_kind,
            dcp=int(getattr(parallel, "cp", 1) or 1),
            draft_page=block_size * row_bytes,
        )
        if draft_sharded:
            draft_cp = max(1, int(getattr(parallel, "cp", 1) or 1))
            for view in views:
                length = max(0, int(getattr(view, "kv_len", 0) or 0))
                eff_len = min(length, draft_cap) if draft_cap > 0 else length
                rows = (
                    float(_context_shard_blocks(eff_len, block_size, draft_cp))
                    if padded else float(eff_len) / float(draft_cp)
                )
                extra += rows * float(row_bytes * int(draft.layers)) * float(dp_width - 1)
    return extra


def _apply_dcp_dp_cohort_placement(
    case: Any,
    footprint: BatchCacheFootprint,
    views: Sequence[RequestView],
    *,
    phase: str,
    padded: bool,
    covered_by_layer: Mapping[int, set[str]],
) -> BatchCacheFootprint:
    parallel = case.runtime.parallel
    dp_width = int(getattr(parallel, "dcp_dp_width", 1) or 1)
    if dp_width <= 1 or not case.model.layer_plans:
        return footprint
    from mindie_sim.schema.runtime_features import canonical_prefix_cache_key

    model, runtime = case.model, case.runtime
    stages = footprint.stages
    stage_map = layer_stage_map(model, parallel, runtime)
    interleave = cp_interleave(runtime)
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    sharded_idx = dcp_indexer_sharded(model, runtime)
    shared_mode = getattr(runtime, "prefix_cache_mode", "private") == "shared"
    is_prefill = str(phase or DECODE).strip().lower() == "prefill"
    extra_shared = [0.0] * stages
    extra_private = [0.0] * stages
    extra_draft = [0.0] * stages
    shared_spans: dict[tuple[int, str, str, int, int, int], list[tuple[int, int, float]]] = {}
    for view in views:
        length = max(0, int(getattr(view, "kv_len", 0) or 0))
        q_tok = 1 if is_prefill else max(1, int(getattr(view, "q_len", 1) or 1))
        declared = max(0, min(int(getattr(view, "prefix_tokens", 0) or 0), length))
        generation = int(getattr(view, "prefix_generation", 0) or 0)
        p_key = (
            canonical_prefix_cache_key("shared", getattr(view, "prefix_cache_key", ""))
            if shared_mode else f"private:{getattr(view, 'request_id', '')}"
        )
        for plan in model.layer_plans:
            _, _, layer_dp = layer_dcp_factors(plan, parallel)
            if layer_dp <= 1:
                continue
            layer_cov = covered_by_layer.get(plan.index, ()) if padded else ()
            ranges = cache_component_ranges(
                model, plan, parallel, sequence_tokens=length,
                prefix_tokens=declared if shared_mode else 0,
                interleave=interleave, rank=None, capacity_padded=padded,
                runtime=runtime, query_tokens=q_tok, block_size=block_size,
                generation=generation, phase=phase,
            )
            stage = int(stage_map[plan.index])
            scale = layer_dp - 1
            for item in ranges:
                if item.component in layer_cov:
                    continue
                is_sharded = (
                    item.component in _DCP_KV_SHARDED_COMPONENTS
                    or (item.component == "index_k" and sharded_idx)
                )
                if not is_sharded:
                    continue
                if item.sharing == SHARING_SHARED:
                    span_len = max(1, item.end_token - item.start_token)
                    rate = float(item.allocated_bytes) / float(span_len)
                    s_key = (stage, item.component, p_key, generation, plan.index, scale)
                    shared_spans.setdefault(s_key, []).append(
                        (item.start_token, item.end_token, rate)
                    )
                else:
                    extra_private[stage] += float(item.allocated_bytes) * float(scale)
    for (stage, _comp, _p_key, _gen, _layer, scale), spans in shared_spans.items():
        extra_shared[stage] += _union_elementary_spans(spans) * float(scale)
    extra_draft[-1] += _dcp_dp_cohort_draft_extra(
        case, views, phase=phase, padded=padded, dp_width=dp_width,
    )
    shared_once = tuple(a + b for a, b in zip(footprint.shared_once_bytes, extra_shared))
    private = tuple(a + b for a, b in zip(footprint.private_per_request_bytes, extra_private))
    draft = tuple(a + b for a, b in zip(footprint.draft_bytes, extra_draft))
    totals = tuple(
        shared_once[s] + private[s] + draft[s] + footprint.boundary_padding_bytes[s]
        for s in range(stages)
    )
    return BatchCacheFootprint(
        shared_once, private, footprint.state_bytes,
        draft, footprint.boundary_padding_bytes, totals,
        requests=footprint.requests, stages=stages,
    )


def _clamp_footprint_draft_cap(
    case: Any,
    footprint: BatchCacheFootprint,
    views: Sequence[RequestView],
    *,
    phase: str,
    padded: bool,
) -> BatchCacheFootprint:
    """Ensure ``batch_cache_bytes`` honors ``effective_draft_context_cap``."""
    from mindie_sim.catalog.cache import stage_cache_components
    from mindie_sim.catalog.draft import optional_draft_spec
    from mindie_sim.catalog.draft_spec import effective_draft_context_cap

    model, runtime = case.model, case.runtime
    draft_kind = _active_draft_kind(model, runtime)
    if not draft_kind or getattr(model, "auxiliary", None) is None:
        return footprint
    if dict(getattr(model, "metadata", ()) or ()).get("cache_allocator") == "paired_sfa":
        return footprint
    draft = optional_draft_spec(model, draft_kind)
    draft_cap = effective_draft_context_cap(draft, runtime)
    if draft_cap <= 0 or not any(int(getattr(v, "kv_len", 0) or 0) > draft_cap for v in views):
        return footprint
    parallel = runtime.parallel
    interleave = cp_interleave(runtime)
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    is_prefill = str(phase or DECODE).strip().lower() == "prefill"
    capped_draft = 0.0
    for view in views:
        length = max(0, int(getattr(view, "kv_len", 0) or 0))
        q_tok = 1 if is_prefill else max(1, int(getattr(view, "q_len", 1) or 1))
        d_stages = stage_cache_components(
            model, parallel, length, capacity_padded=padded,
            draft_kind=draft_kind, interleave=interleave, block_size=block_size,
            runtime=runtime, query_tokens=q_tok, phase=phase,
        )
        capped_draft += float(d_stages[-1].get("draft_kv", 0.0)) + float(d_stages[-1].get("draft_state", 0.0))
    draft_list = list(footprint.draft_bytes)
    delta = capped_draft - float(draft_list[-1])
    if abs(delta) <= 1e-9:
        return footprint
    draft_list[-1] = capped_draft
    totals = list(footprint.total_bytes)
    totals[-1] = max(0.0, float(totals[-1]) + delta)
    return BatchCacheFootprint(
        footprint.shared_once_bytes, footprint.private_per_request_bytes,
        footprint.state_bytes, tuple(draft_list),
        footprint.boundary_padding_bytes, tuple(totals),
        requests=footprint.requests, stages=footprint.stages,
    )


def batch_cache_bytes(
    case: Any, *, sequences: int, context: int, tokens_per_seq: int,
    prefix_tokens: int = 0, phase: str = DECODE, padded: bool = True,
    requests: Sequence[RequestView] | None = None,
    prefix_cache_keys: Sequence[str] | None = None,
    prefix_groups: int | None = None,
) -> BatchCacheFootprint:
    """Physical batch cache footprint for one DP replica (with DP-cohort DCP & CachePages)."""
    if requests is None and (sequences < 1 or context < 0 or tokens_per_seq < 1):
        raise ModelingError("capacity_batch_geometry_invalid")
    views = _resolve_views(
        case, sequences, context, tokens_per_seq, prefix_tokens,
        phase=phase, requests=requests,
        prefix_cache_keys=prefix_cache_keys, prefix_groups=prefix_groups,
    )
    if not views:
        raise ModelingError("capacity_batch_geometry_invalid")
    page_result = _explicit_page_group_footprint(case, views, phase=phase, padded=padded)
    if page_result is not None:
        footprint, pages_obj = page_result
        covered = pages_obj.covered
    else:
        footprint = batch_cache_footprint(
            case, requests=views, replica=0, phase=phase, padded=padded,
        )
        footprint = _clamp_footprint_draft_cap(
            case, footprint, views, phase=phase, padded=padded,
        )
        covered = {}
    return _apply_dcp_dp_cohort_placement(
        case, footprint, views, phase=phase, padded=padded, covered_by_layer=covered,
    )


def capacity_window(
    case: Any, *, sequences: int, context: int, tokens_per_seq: int,
    prefix_tokens: int = 0, retiring_sequences: int = 0,
    lookahead_sequences: int = 0, phase: str = DECODE,
    padded: bool = True,
    requests: Sequence[RequestView] | None = None,
    prefix_cache_keys: Sequence[str] | None = None,
    prefix_groups: int | None = None,
) -> CapacityWindow:
    """Active, retiring and lookahead peaks from the same physical ranges."""
    active = batch_cache_bytes(
        case, sequences=sequences, context=context, tokens_per_seq=tokens_per_seq,
        prefix_tokens=prefix_tokens, phase=phase, padded=padded,
        requests=requests, prefix_cache_keys=prefix_cache_keys,
        prefix_groups=prefix_groups)
    retiring = (batch_cache_bytes(
        case, sequences=max(1, int(retiring_sequences)), context=context,
        tokens_per_seq=tokens_per_seq, prefix_tokens=prefix_tokens,
        phase=phase, padded=padded, prefix_cache_keys=prefix_cache_keys,
        prefix_groups=prefix_groups) if int(retiring_sequences) > 0 else None)
    lookahead = (batch_cache_bytes(
        case, sequences=max(1, int(lookahead_sequences)), context=context,
        tokens_per_seq=tokens_per_seq, prefix_tokens=prefix_tokens,
        phase=phase, padded=padded, prefix_cache_keys=prefix_cache_keys,
        prefix_groups=prefix_groups) if int(lookahead_sequences) > 0 else None)
    active_total = active.total
    retiring_total = 0.0 if retiring is None else retiring.total
    lookahead_total = 0.0 if lookahead is None else lookahead.total
    # The live peak is one physical union over every concurrent window, so a
    # shared prefix is not charged again for each retiring wave.
    extra_seqs = max(0, int(retiring_sequences)) + max(0, int(lookahead_sequences))
    if requests is not None and extra_seqs == 0:
        combined = active
    elif requests is not None:
        extra_views = _resolve_views(
            case, extra_seqs, context, tokens_per_seq, prefix_tokens,
            phase=phase, requests=None, prefix_cache_keys=prefix_cache_keys,
            prefix_groups=prefix_groups,
        )
        combined = batch_cache_bytes(
            case, sequences=len(requests) + extra_seqs, context=context,
            tokens_per_seq=tokens_per_seq, prefix_tokens=prefix_tokens,
            phase=phase, padded=padded, requests=(*tuple(requests), *extra_views),
        )
    else:
        combined = batch_cache_bytes(
            case, sequences=sequences + extra_seqs, context=context,
            tokens_per_seq=tokens_per_seq, prefix_tokens=prefix_tokens,
            phase=phase, padded=padded, prefix_cache_keys=prefix_cache_keys,
            prefix_groups=prefix_groups)
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
