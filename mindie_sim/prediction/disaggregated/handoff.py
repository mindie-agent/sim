"""Cache handoff binding for prefill and decode.

[Role] Map published source ranges to physical destination shards and bind releases
  with exact integer half-open ``[start, end)`` byte intervals.
[Invariants] Handoff bytes and releases follow upstream-exact GQA/MLA/Mamba/Draft
  rank ownership, exact 0-byte disjoint integer partition conservation, and
  structural ``MooncakeHybridConnector`` legality (AG014-clean).
[Timeouts & Locks] Finite spans and Engine dependencies for the current handoff.
[Side-Effect Policy] Queries endpoint memory and registers release callbacks on the
  shared Engine.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from mindie_sim.catalog.cache import gbsa_cache_layout, layer_cache_components
from mindie_sim.catalog.cache_format import (
    _resolve_index_cache_format,
    cache_format,
    gqa_kv_formats,
)
from mindie_sim.catalog.context_parallel import (
    cp_interleave,
    dcp_indexer_sharded,
    layer_dcp_degree,
    layer_dcp_factors,
)
from mindie_sim.catalog.draft import optional_draft_spec
from mindie_sim.catalog.draft_geometry import draft_layer_plans
from mindie_sim.schema.common import ModelingError

from ..pd_connector import PDHandoff, PDTransfer
from ..disaggregated.ranges import (
    address,
    canonical_physical_rank,
    exact_int_bytes,
    interleaved_token_runs,
    owned_runs,
    transfers_for_runs,
)

_NON_NEURAL_SPEC_METHODS = frozenset({"none", "ngram", "ngram_gpu"})

HEAD_COMPONENTS = frozenset({"gqa_kv", "window"})
ROW_COMPONENTS = frozenset({"mla_kv", "window", "compressed_kv", "index_k"})
_TOKEN_ADDITIVE_COMPONENTS = HEAD_COMPONENTS | ROW_COMPONENTS | frozenset({"gqa_k", "gqa_v"})
SHARDED_STATE_COMPONENTS = frozenset({"recurrent_state", "conv_state"})
TRANSIENT_STATE_COMPONENTS = frozenset({"replay_records", "state_transient"})
VALID_PD_CONNECTORS = frozenset({
    "auto", "MooncakeConnector", "MooncakePushWorker", "MooncakeHybridConnector",
})
_LINEAR_ATTENTION_KINDS = frozenset({"gdn", "kda", "mamba", "mamba2", "linear"})
_CONV_PROJECTION_COUNT = 3


def resolve_and_validate_pd_connector(
    model_spec: Any,
    p_par: Any,
    d_par: Any,
    *,
    deployment_mode: str = "pd_disaggregated",
    connector_kind: str = "auto",
    disable_hybrid_kv_cache_manager: bool = False,
    p_pp_layer_partition: tuple[int, ...] = (),
) -> tuple[bool, str, str | None]:
    """Structural PD connector resolver & gate (AG014-clean: zero model_name checks)."""
    if connector_kind not in VALID_PD_CONNECTORS:
        raise ModelingError(f"pd_connector_unknown:{connector_kind}")
    if deployment_mode != "pd_disaggregated":
        return True, "none", None

    plans = tuple(getattr(model_spec, "layer_plans", ()) or ())
    has_recurrent = any(
        str(getattr(lp, "attention", "") or "").strip().lower() in _LINEAR_ATTENTION_KINDS
        for lp in plans
    )
    has_non_full = any(
        str(getattr(lp, "attention", "") or "").strip().lower() in _LINEAR_ATTENTION_KINDS
        or int(getattr(lp, "window", 0) or 0) > 0
        or int(getattr(lp, "compress_ratio", 0) or 0) > 1
        for lp in plans
    )
    kv_group_kinds = {
        (
            "mamba"
            if str(getattr(lp, "attention", "") or "").strip().lower() in _LINEAR_ATTENTION_KINDS
            else f"swa_{int(getattr(lp, 'window', 0) or 0)}"
            if int(getattr(lp, "window", 0) or 0) > 0
            else f"cmp_{int(getattr(lp, 'compress_ratio', 0) or 0)}"
            if int(getattr(lp, "compress_ratio", 0) or 0) > 1
            else "full"
        )
        for lp in plans
    }
    use_hybrid = (not disable_hybrid_kv_cache_manager) and has_non_full and len(kv_group_kinds) > 1
    meta = dict(getattr(model_spec, "metadata", ()) or ())
    is_paired_sfa = str(meta.get("cache_allocator") or "").strip().lower() == "paired_sfa"

    if connector_kind == "auto":
        resolved = "MooncakeHybridConnector" if (has_recurrent and not is_paired_sfa) else "MooncakePushWorker"
    else:
        resolved = connector_kind

    if resolved != "MooncakeHybridConnector":
        return True, resolved, None

    if int(d_par.pp) != 1:
        return False, resolved, "mooncake_hybrid_connector_requires_decode_pp_1"
    if int(p_par.cp) != 1 or int(d_par.cp) != 1:
        return False, resolved, "mooncake_hybrid_connector_requires_dcp_1"
    if int(d_par.pcp) != 1:
        return False, resolved, "mooncake_hybrid_connector_requires_decode_pcp_1"
    if int(p_par.pp) > 1 and int(p_par.pcp) > 1:
        return False, resolved, "mooncake_hybrid_connector_forbids_pp_and_pcp"
    if int(p_par.tp) < int(d_par.tp):
        return False, resolved, "mooncake_hybrid_connector_requires_prefill_tp_ge_decode_tp"
    if has_recurrent:
        if not use_hybrid:
            return False, resolved, "mooncake_hybrid_connector_mamba_requires_hybrid_groups"
        if int(p_par.tp) != int(d_par.tp):
            return False, resolved, "mooncake_hybrid_connector_mamba_requires_equal_tp"
    if int(p_par.pp) > 1 and p_pp_layer_partition and sum(p_pp_layer_partition) != len(plans):
        return False, resolved, "mooncake_hybrid_connector_invalid_prefill_pp_partition"
    return True, resolved, None


def _local_head_spans(count, tp):
    """Contiguous ``r // replicas`` KV-head ownership matching upstream ``push_worker.py``."""
    count = max(0, int(count))
    tp = max(1, int(tp))
    if count <= 0:
        return ()
    if max(count, tp) % min(count, tp) != 0:
        raise ModelingError(
            f"pd_handoff_unsupported_head_partition:heads={count},tp={tp}"
        )
    replicas = max(1, tp // count)
    local = max(1, count // tp)
    return tuple(
        ((r // replicas) * local, (r // replicas + 1) * local)
        for r in range(tp)
    )


def _upstream_rank_coord(index: int, parallel: Any) -> int:
    """Upstream ``push_worker.py`` rank coordinate with PCP innermost."""
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    idx = int(index)
    tp_i = idx % tp
    pcp_i = (idx // tp) % pcp
    dp_i = idx // (tp * pcp)
    return (dp_i * tp + tp_i) * pcp + pcp_i


def _token_span(memory, index, tokens):
    """Actual half-open integer token rows owned by one physical rank."""
    tokens = max(0, int(tokens))
    owners = getattr(memory, "rank_token_owners", None)
    if owners is not None and 0 <= index < len(owners):
        lo, hi = owners[index]
        return (max(0, min(tokens, int(lo))), max(0, min(tokens, int(hi))))
    parallel = memory.case.runtime.parallel
    cp = max(1, int(parallel.cp))
    if cp > 1:
        axis = _upstream_rank_coord(index, parallel) % cp
        return (tokens * axis // cp, tokens * (axis + 1) // cp)
    return (0, tokens)


def _head_token_span(index, tokens, head_spans, cp, *, total_ranks: int | None = None,
                     parallel: Any = None):
    """Exact half-open integer token rows paired with one rank's contiguous head span."""
    tokens = max(0, int(tokens))
    cp = max(1, int(cp))
    if cp <= 1:
        return (0, tokens)
    if head_spans:
        tp = len(head_spans)
        distinct = len({(int(lo), int(hi)) for lo, hi in head_spans})
        rep_per_head = max(1, tp // max(1, distinct))
        if total_ranks is not None:
            outer_planes = max(1, int(total_ranks) // max(1, tp))
            slots_per_head = outer_planes * rep_per_head
            if slots_per_head < cp or slots_per_head % cp != 0:
                raise ModelingError(
                    f"pd_handoff_cp_slots_not_divisible:slots={slots_per_head},cp={cp}"
                )
        outer_slot = int(index) // tp
        intra_ord = (int(index) % tp) % rep_per_head
        axis = (outer_slot * rep_per_head + intra_ord) % cp
    elif parallel is not None:
        axis = _upstream_rank_coord(index, parallel) % cp
    else:
        axis = int(index) % cp
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


@dataclass(frozen=True)
class _ActiveDraftHandoff:
    """Resolved compatible neural draft cache contract for PD transfer."""

    kind: str
    inherited: bool
    eff_tokens: int
    p_draft_spec: Any
    d_draft_spec: Any
    p_draft_plans: tuple[Any, ...]
    d_draft_plans: tuple[Any, ...]
    p_row_bytes: int = 0
    d_row_bytes: int = 0
    p_sharded: bool = False
    d_sharded: bool = False


@dataclass(frozen=True)
class PdExpectedDemand:
    """Authoritative integer handoff demand shared by sweep and live serving."""

    total_volume_bytes: int
    hbm_volume_bytes: int
    host_volume_bytes: int
    draft_kv_bytes: int
    draft_state_bytes: int
    decode_local_stage_bytes: tuple[float, ...]
    active_draft: _ActiveDraftHandoff | None


def _spec_method_from_runtime(runtime: Any) -> str:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    return str(meta.get("spec_method") or "").strip().lower()


def _resolve_active_draft_handoff(
    p_case: Any, d_case: Any, p_mem: Any, d_mem: Any, tokens: int,
) -> _ActiveDraftHandoff | None:
    """Resolve compatible neural draft cache transfer contract (0 for ngram/none)."""
    d_method = _spec_method_from_runtime(d_case.runtime)
    if d_method in _NON_NEURAL_SPEC_METHODS:
        return None
    d_kind = str(getattr(d_mem, "draft_kind", "") or "")
    if not d_kind:
        return None
    d_spec = optional_draft_spec(d_case.model, d_kind)
    if d_spec is None or int(getattr(d_spec, "layers", 0) or 0) <= 0:
        return None
    p_method = _spec_method_from_runtime(p_case.runtime)
    p_kind = str(getattr(p_mem, "draft_kind", "") or "")
    p_spec = (
        optional_draft_spec(p_case.model, p_kind)
        if p_kind and p_method not in _NON_NEURAL_SPEC_METHODS
        else None
    )
    if p_spec is None or int(getattr(p_spec, "layers", 0) or 0) <= 0:
        raise ModelingError("pd_handoff_missing_producer_draft_cache")
    if (
        int(p_spec.layers) != int(d_spec.layers)
        or str(p_spec.attention_kind) != str(d_spec.attention_kind)
    ):
        raise ModelingError(
            "pd_handoff_incompatible_draft_cache:"
            f"p=({p_spec.layers},{p_spec.attention_kind}),"
            f"d=({d_spec.layers},{d_spec.attention_kind})"
        )
    p_cap = int(p_mem._draft_context_cap(p_spec) or 0)
    d_cap = int(d_mem._draft_context_cap(d_spec) or 0)
    eff_tokens = int(tokens)
    if p_cap > 0:
        eff_tokens = min(eff_tokens, p_cap)
    if d_cap > 0:
        eff_tokens = min(eff_tokens, d_cap)
    if eff_tokens <= 0:
        return None
    if d_mem._uses_inherited_draft_cache(d_case.model):
        p_plans = tuple(draft_layer_plans(p_case.model, p_kind))
        d_plans = tuple(draft_layer_plans(d_case.model, d_kind))
        return _ActiveDraftHandoff(
            kind=d_kind, inherited=True, eff_tokens=eff_tokens,
            p_draft_spec=p_spec, d_draft_spec=d_spec,
            p_draft_plans=p_plans, d_draft_plans=d_plans,
        )
    from mindie_sim.prediction.capacity.cache import _draft_is_dcp_sharded, _draft_row_bytes_int

    p_par, d_par = p_case.runtime.parallel, d_case.runtime.parallel
    p_block = max(1, int(cp_interleave(p_case.runtime)))
    d_block = max(1, int(cp_interleave(d_case.runtime)))
    p_row = int(_draft_row_bytes_int(p_case.model, p_spec, tp=p_par.tp))
    d_row = int(_draft_row_bytes_int(d_case.model, d_spec, tp=d_par.tp))
    p_shd = bool(_draft_is_dcp_sharded(
        p_case.model, p_par, p_kind,
        dcp=int(getattr(p_par, "cp", 1) or 1), draft_page=p_block * p_row,
    ))
    d_shd = bool(_draft_is_dcp_sharded(
        d_case.model, d_par, d_kind,
        dcp=int(getattr(d_par, "cp", 1) or 1), draft_page=d_block * d_row,
    ))
    return _ActiveDraftHandoff(
        kind=d_kind, inherited=False, eff_tokens=eff_tokens,
        p_draft_spec=p_spec, d_draft_spec=d_spec,
        p_draft_plans=(), d_draft_plans=(),
        p_row_bytes=p_row, d_row_bytes=d_row,
        p_sharded=p_shd, d_sharded=d_shd,
    )


def _handoff_rank_views(
    case, layer, parallel, ranks, ledger, owners,
    head_spans, has_head_axis, tokens,
    *, stage: int = 0, replica: int = 0, phase: str = "prefill",
):
    eff_cp = layer_dcp_degree(layer, parallel)
    if owners is not None:
        spans = tuple(_token_span(ledger, index, tokens) for index in range(len(ranks)))
        component_owners = owners
    else:
        spans = tuple(
            _head_token_span(
                index, tokens, head_spans if has_head_axis else (), eff_cp,
                total_ranks=len(ranks), parallel=parallel,
            )
            for index in range(len(ranks))
        )
        # Preserve exact DCP interleaved ownership in layer_cache_components
        # whenever caller did not supply explicit T05 owner spans.
        component_owners = None
    interleave = cp_interleave(case.runtime)
    dsa_non_dcp_idx = (
        owners is None
        and eff_cp > 1
        and getattr(parallel, "cp_plan", "") == "dsa_cp"
        and getattr(layer, "owns_index_k", False)
        and not dcp_indexer_sharded(case.model, case.runtime)
        and layer_dcp_factors(layer, parallel)[2] <= 1
    )
    result = []
    for index, rank in enumerate(ranks):
        if owners is not None:
            cache_rank = index
        elif dsa_non_dcp_idx:
            cache_rank = index % eff_cp
        else:
            cache_rank = canonical_physical_rank(
                index, parallel, stage=stage, replica=replica, cohort_size=len(ranks),
            )
        raw_components = layer_cache_components(
            case.model, layer, parallel, tokens, capacity_padded=False,
            interleave=interleave,
            rank=cache_rank, owner_spans=component_owners,
            runtime=case.runtime, query_tokens=1, phase=phase,
        )
        components = {
            name: exact_int_bytes(val, f"pd_component_bytes:{name}")
            for name, val in raw_components.items()
            if name not in TRANSIENT_STATE_COMPONENTS
        }
        result.append((index, rank, components, spans[index], cache_rank))
    return result


def _sparse_offload_params(decode_mem: Any, tokens: int) -> tuple[bool, set[int], int]:
    sparse = getattr(decode_mem, "sparse_offload", None)
    offload = bool(sparse) and not bool(
        dict(sparse.get("options", {})).get("keep_device_kv_cache")
    )
    if not offload:
        return False, set(), 0
    offload_layers = set(sparse.get("offload_layers", ()))
    use_copy_sfa = bool(dict(sparse.get("options", {})).get("use_fused_copy_sfa", True))
    topk_buffer = int(sparse.get("topk_buffer_size", 0))
    hot_tokens = min(int(tokens), topk_buffer) if use_copy_sfa and topk_buffer > 0 else 0
    return True, offload_layers, hot_tokens


def compute_expected_pd_handoff_demand(
    prefill_case: Any,
    decode_case: Any,
    prefill_memory: Any,
    decode_memory: Any,
    tokens: int,
) -> PdExpectedDemand:
    """Authoritative exact integer demand shared by ``compute_pd_handoff_plan`` and ``serving.py``."""
    n_tokens = max(0, int(tokens))
    d_par = decode_case.runtime.parallel
    pp = max(1, int(d_par.pp))
    if n_tokens <= 0:
        return PdExpectedDemand(0, 0, 0, 0, 0, tuple(0.0 for _ in range(pp)), None)
    d_topo = _CohortStageTopology(d_par, machine=decode_case.machine, rank_offset=0)
    offload, offload_layers, hot_tokens = _sparse_offload_params(decode_memory, n_tokens)
    target_owners = getattr(decode_memory, "rank_token_owners", None)
    stage_hbm_total = [0] * pp
    host_total = 0
    draft_kv_total = 0
    draft_state_total = 0

    for layer in decode_case.model.layer_plans:
        st = int(decode_memory.layer_stage[layer.index])
        dst_ranks = d_topo.ranks(0, st)
        num_heads = int(getattr(layer, "num_kv_heads", 0) or 0)
        has_head_axis = layer.attention in {"gqa", "projection_only"}
        target_spans = _local_head_spans(num_heads, d_par.tp) if has_head_axis else ()
        dst_views = _handoff_rank_views(
            decode_case, layer, d_par, dst_ranks, decode_memory,
            target_owners, target_spans, has_head_axis, n_tokens,
            stage=st, replica=0, phase="decode",
        )
        window = int(getattr(layer, "window", 0) or 0)
        host_counted = False
        for _index, _rank, values, span, cache_rank in dst_views:
            for comp, comp_bytes in values.items():
                if comp_bytes <= 0:
                    continue
                if offload and comp == "mla_kv" and layer.index in offload_layers:
                    win_start = max(0, n_tokens - window) if window > 0 else 0
                    runs = owned_runs(
                        comp, layer, decode_case.model, d_par, cache_rank, span,
                        n_tokens, win_start, runtime=decode_case.runtime,
                        explicit_span=(target_owners is not None),
                    )
                    owned_rows = sum(hi - lo for lo, hi in runs)
                    if owned_rows > 0:
                        if comp_bytes % owned_rows != 0:
                            raise ModelingError(
                                f"pd_handoff_offload_non_integer_row_bytes:{comp_bytes}:{owned_rows}"
                            )
                        row_unit = comp_bytes // owned_rows
                        if not host_counted:
                            host_total += (n_tokens - win_start) * row_unit
                            host_counted = True
                        hot_rows = sum(
                            hi - max(lo, n_tokens - hot_tokens)
                            for lo, hi in runs if hi > max(lo, n_tokens - hot_tokens)
                        )
                        stage_hbm_total[st] += hot_rows * row_unit
                else:
                    stage_hbm_total[st] += comp_bytes

    active_draft = _resolve_active_draft_handoff(
        prefill_case, decode_case, prefill_memory, decode_memory, n_tokens,
    )
    if active_draft is not None:
        draft_st = pp - 1
        dst_ranks = d_topo.ranks(0, draft_st)
        if active_draft.inherited:
            for d_layer in active_draft.d_draft_plans:
                num_heads = int(getattr(d_layer, "num_kv_heads", 0) or 0)
                has_head_axis = d_layer.attention in {"gqa", "projection_only"}
                target_spans = _local_head_spans(num_heads, d_par.tp) if has_head_axis else ()
                dst_views = _handoff_rank_views(
                    decode_case, d_layer, d_par, dst_ranks, decode_memory,
                    target_owners, target_spans, has_head_axis, active_draft.eff_tokens,
                    stage=draft_st, replica=0, phase="decode",
                )
                for _index, _rank, values, _span, _cache_rank in dst_views:
                    for comp, comp_bytes in values.items():
                        if comp_bytes <= 0:
                            continue
                        stage_hbm_total[draft_st] += comp_bytes
                        if comp in SHARDED_STATE_COMPONENTS:
                            draft_state_total += comp_bytes
                        else:
                            draft_kv_total += comp_bytes
        else:
            d_cp = int(d_par.cp) if active_draft.d_sharded else 1
            for index in range(len(dst_ranks)):
                runs = _draft_token_runs(
                    decode_case, d_par, decode_memory, index, active_draft.eff_tokens,
                    d_cp, stage=draft_st, replica=0, cohort_size=len(dst_ranks),
                )
                owned_rows = sum(hi - lo for lo, hi in runs)
                layer_bytes = owned_rows * active_draft.d_row_bytes
                total_rank_draft = layer_bytes * int(active_draft.d_draft_spec.layers)
                stage_hbm_total[draft_st] += total_rank_draft
                draft_kv_total += total_rank_draft

    hbm_total = sum(stage_hbm_total)
    stage_width = max(1, d_topo._width)
    decode_local = tuple(float(st_bytes) / float(stage_width) for st_bytes in stage_hbm_total)
    return PdExpectedDemand(
        total_volume_bytes=hbm_total + host_total,
        hbm_volume_bytes=hbm_total,
        host_volume_bytes=host_total,
        draft_kv_bytes=draft_kv_total,
        draft_state_bytes=draft_state_total,
        decode_local_stage_bytes=decode_local,
        active_draft=active_draft,
    )


def _conv_projection_byte_widths(layer: Any, row_bytes: int, tp_ratio: int) -> tuple[int, ...]:
    """Projection byte widths inside one conv-state row (``base_worker.py:197-260``)."""
    attn = str(getattr(layer, "attention", "") or "").strip().lower()
    if attn == "kda" and row_bytes % _CONV_PROJECTION_COUNT == 0:
        third = row_bytes // _CONV_PROJECTION_COUNT
        if third % tp_ratio == 0:
            return (third, third, third)
    if attn == "gdn":
        conv_k = max(2, int(getattr(layer, "conv_kernel", 0) or 2))
        conv_rows = conv_k - 1
        conv_elems = int(getattr(layer, "conv_state_elements", 0) or 0)
        row_elems = conv_elems // conv_rows if conv_rows > 0 and conv_elems > 0 else 0
        state_elems = int(getattr(layer, "num_heads", 0) or 0) * int(getattr(layer, "head_dim", 0) or 0)
        if row_elems > 0 and 0 < state_elems < row_elems and (row_elems - state_elems) % 2 == 0:
            q_elems = (row_elems - state_elems) // 2
            if row_bytes % row_elems == 0:
                elem_b = row_bytes // row_elems
                q_b = q_elems * elem_b
                v_b = state_elems * elem_b
                if q_b % tp_ratio == 0 and v_b % tp_ratio == 0:
                    return (q_b, q_b, v_b)
    return (row_bytes,)


def _emit_conv_displaced_slices(
    emit, *, src_rank: int, dst_rank: int, component: str,
    source_stage: int, destination_stage: int, logical_layer: int, num_heads: int,
    src_tokens: tuple[int, int], dst_tokens: tuple[int, int],
    src_storage: str, dst_storage: str, src_size: int, dst_size: int,
    layer: Any, tp_ratio: int, rank_offset: int, src_is_wider: bool,
) -> None:
    """Emit exact 2D conv-state per-row, per-projection displaced integer byte slices."""
    conv_rows = max(1, int(getattr(layer, "conv_kernel", 0) or 1) - 1)
    if src_size % conv_rows != 0 or dst_size % conv_rows != 0:
        slice_len = dst_size if src_is_wider else src_size
        emit(
            src_rank, dst_rank, component, slice_len,
            source_stage=source_stage, destination_stage=destination_stage,
            layer=logical_layer, source_span=(0, num_heads), destination_span=(0, num_heads),
            source_tokens=src_tokens, destination_tokens=dst_tokens,
            source_storage=src_storage, destination_storage=dst_storage,
            source_size=src_size,
            source_displacement=(rank_offset * slice_len if src_is_wider else 0),
            destination_size=dst_size,
            destination_displacement=(0 if src_is_wider else rank_offset * slice_len),
        )
        return
    wide_row = (src_size if src_is_wider else dst_size) // conv_rows
    narrow_row = (dst_size if src_is_wider else src_size) // conv_rows
    proj_widths = _conv_projection_byte_widths(layer, wide_row, tp_ratio)
    for row_idx in range(conv_rows):
        wide_row_base = row_idx * wide_row
        narrow_row_base = row_idx * narrow_row
        wide_off = 0
        narrow_off = 0
        for p_wide in proj_widths:
            p_narrow = p_wide // tp_ratio
            wide_disp = wide_row_base + wide_off + rank_offset * p_narrow
            narrow_disp = narrow_row_base + narrow_off
            emit(
                src_rank, dst_rank, component, p_narrow,
                source_stage=source_stage, destination_stage=destination_stage,
                layer=logical_layer, source_span=(0, num_heads), destination_span=(0, num_heads),
                source_tokens=src_tokens, destination_tokens=dst_tokens,
                source_storage=src_storage, destination_storage=dst_storage,
                source_size=src_size,
                source_displacement=(wide_disp if src_is_wider else narrow_disp),
                destination_size=dst_size,
                destination_displacement=(narrow_disp if src_is_wider else wide_disp),
            )
            wide_off += p_wide
            narrow_off += p_narrow


def _emit_state_transfers(
    emit, request, emitted_component, raw_component, layer, logical_layer,
    source_stage, destination_stage, prefill_rep, decode_rep,
    src_views, dst_views, prefill_p, decode_p, num_heads,
):
    """Exact Mamba/KDA/GDN TP-sharded state slicing (zero rank-0 hotspot & exact offsets)."""
    is_sharded = raw_component in SHARDED_STATE_COMPONENTS
    attn = str(getattr(layer, "attention", "") or "").strip().lower()
    if not is_sharded:
        for dest_index, dest_rank, dest_values, dest_tokens, _ in dst_views:
            need = dest_values.get(raw_component, 0)
            if need <= 0:
                continue
            src_index, src_rank, src_values, src_tokens, _ = src_views[dest_index % len(src_views)]
            avail = src_values.get(raw_component, 0)
            if avail != need:
                raise ModelingError(
                    f"pd_handoff_cache_abi_mismatch:{request}:{logical_layer}:{emitted_component}:src={avail}:dst={need}"
                )
            emit(
                src_rank, dest_rank, emitted_component, need,
                source_stage=source_stage, destination_stage=destination_stage,
                layer=logical_layer, source_span=(0, num_heads), destination_span=(0, num_heads),
                source_tokens=src_tokens, destination_tokens=dest_tokens,
                source_storage=f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}",
                destination_storage=f"pd:kv:{request}:decode:r{decode_rep}:rank{dest_rank}",
                source_size=avail, source_displacement=0,
                destination_size=need, destination_displacement=0,
            )
        return

    kda_sharded_pcp = attn == "kda"
    s_p = int(prefill_p.tp) * (int(prefill_p.pcp) if kda_sharded_pcp else 1)
    s_d = int(decode_p.tp) * (int(decode_p.pcp) if kda_sharded_pcp else 1)
    if max(s_p, s_d) % min(s_p, s_d) != 0:
        raise ModelingError(f"pd_mamba_tp_ratio_non_integer:src={s_p},dst={s_d}")
    tp_ratio = max(s_p, s_d) // min(s_p, s_d)

    def _shard_and_plane(idx: int, par: Any) -> tuple[int, int]:
        tp_n = max(1, int(par.tp))
        pcp_n = max(1, int(par.pcp))
        tp_i = idx % tp_n
        pcp_i = (idx // tp_n) % pcp_n
        dp_i = idx // (tp_n * pcp_n)
        if kda_sharded_pcp:
            return tp_i * pcp_n + pcp_i, dp_i
        return tp_i, dp_i * pcp_n + pcp_i

    src_by_shard: dict[int, list[tuple[int, int, int, tuple[int, int]]]] = defaultdict(list)
    for src_index, src_rank, src_values, src_tokens, _ in src_views:
        avail = src_values.get(raw_component, 0)
        if avail > 0:
            j_p, p_plane = _shard_and_plane(src_index, prefill_p)
            src_by_shard[j_p].append((p_plane, src_rank, avail, src_tokens))

    for dest_index, dest_rank, dest_values, dest_tokens, _ in dst_views:
        need = dest_values.get(raw_component, 0)
        if need <= 0:
            continue
        j_d, d_plane = _shard_and_plane(dest_index, decode_p)
        dst_storage = f"pd:kv:{request}:decode:r{decode_rep}:rank{dest_rank}"
        if s_p <= s_d:
            # 1 producer feeds tp_ratio receivers (s_p == s_d => tp_ratio == 1, rank_offset == 0).
            j_p = j_d // tp_ratio
            rank_offset = j_d % tp_ratio
            cands = src_by_shard.get(j_p, ())
            if not cands:
                raise ModelingError(
                    f"pd_handoff_component_range_uncovered:{request}:{logical_layer}:{emitted_component}"
                )
            _, src_rank, avail, src_tokens = cands[d_plane % len(cands)]
            if avail != need * tp_ratio:
                raise ModelingError(
                    f"pd_handoff_cache_abi_mismatch:{request}:{logical_layer}:{emitted_component}:src={avail}:dst={need}"
                )
            src_storage = f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}"
            if tp_ratio == 1:
                emit(
                    src_rank, dest_rank, emitted_component, need,
                    source_stage=source_stage, destination_stage=destination_stage,
                    layer=logical_layer, source_span=(0, num_heads), destination_span=(0, num_heads),
                    source_tokens=src_tokens, destination_tokens=dest_tokens,
                    source_storage=src_storage, destination_storage=dst_storage,
                    source_size=avail, source_displacement=0,
                    destination_size=need, destination_displacement=0,
                )
            elif raw_component == "conv_state":
                _emit_conv_displaced_slices(
                    emit, src_rank=src_rank, dst_rank=dest_rank, component=emitted_component,
                    source_stage=source_stage, destination_stage=destination_stage,
                    logical_layer=logical_layer, num_heads=num_heads,
                    src_tokens=src_tokens, dst_tokens=dest_tokens,
                    src_storage=src_storage, dst_storage=dst_storage,
                    src_size=avail, dst_size=need, layer=layer,
                    tp_ratio=tp_ratio, rank_offset=rank_offset, src_is_wider=True,
                )
            else:
                emit(
                    src_rank, dest_rank, emitted_component, need,
                    source_stage=source_stage, destination_stage=destination_stage,
                    layer=logical_layer, source_span=(0, num_heads), destination_span=(0, num_heads),
                    source_tokens=src_tokens, destination_tokens=dest_tokens,
                    source_storage=src_storage, destination_storage=dst_storage,
                    source_size=avail, source_displacement=rank_offset * need,
                    destination_size=need, destination_displacement=0,
                )
        else:
            # tp_ratio producers merge into 1 receiver.
            for rank_offset in range(tp_ratio):
                j_p = j_d * tp_ratio + rank_offset
                cands = src_by_shard.get(j_p, ())
                if not cands:
                    raise ModelingError(
                        f"pd_handoff_component_range_uncovered:{request}:{logical_layer}:{emitted_component}"
                    )
                _, src_rank, avail, src_tokens = cands[d_plane % len(cands)]
                if need != avail * tp_ratio:
                    raise ModelingError(
                        f"pd_handoff_cache_abi_mismatch:{request}:{logical_layer}:{emitted_component}:src={avail}:dst={need}"
                    )
                src_storage = f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}"
                if raw_component == "conv_state":
                    _emit_conv_displaced_slices(
                        emit, src_rank=src_rank, dst_rank=dest_rank, component=emitted_component,
                        source_stage=source_stage, destination_stage=destination_stage,
                        logical_layer=logical_layer, num_heads=num_heads,
                        src_tokens=src_tokens, dst_tokens=dest_tokens,
                        src_storage=src_storage, dst_storage=dst_storage,
                        src_size=avail, dst_size=need, layer=layer,
                        tp_ratio=tp_ratio, rank_offset=rank_offset, src_is_wider=False,
                    )
                else:
                    emit(
                        src_rank, dest_rank, emitted_component, avail,
                        source_stage=source_stage, destination_stage=destination_stage,
                        layer=logical_layer, source_span=(0, num_heads), destination_span=(0, num_heads),
                        source_tokens=src_tokens, destination_tokens=dest_tokens,
                        source_storage=src_storage, destination_storage=dst_storage,
                        source_size=avail, source_displacement=0,
                        destination_size=need, destination_displacement=rank_offset * avail,
                    )


def _gqa_kv_plane_weights(model: Any, layer: Any, *, phase: str = "decode") -> tuple[int, int]:
    """Return exact positive integer ``(k_bytes, v_bytes)`` weights per head-token."""
    if model is None:
        return 1, 1
    k_fmt, v_fmt, is_shared = gqa_kv_formats(model, plan=layer, phase=phase)
    k_dim = max(1, int(getattr(layer, "head_dim", 0) or getattr(model, "head_dim", 1) or 1))
    v_dim = max(1, int(getattr(layer, "v_head_dim", 0) or k_dim))
    if is_shared and k_dim == v_dim:
        return 1, 1
    k_b = max(1, int(k_fmt.row_bytes(k_dim)))
    v_b = max(1, int(v_fmt.row_bytes(v_dim)))
    return k_b, v_b


def _assert_matching_cache_abi(
    prefill_case: Any, decode_case: Any, layer: Any, raw_component: str,
    request: str, logical_layer: int,
) -> None:
    """Fail closed when prefill and decode declare incompatible cache wire formats."""
    if raw_component in {"gqa_kv", "index_k"}:
        if gbsa_cache_layout(prefill_case.runtime) != gbsa_cache_layout(decode_case.runtime):
            raise ModelingError(
                f"pd_handoff_cache_abi_mismatch:{request}:{logical_layer}:{raw_component}"
            )
    if raw_component == "gqa_kv":
        p_fmt = gqa_kv_formats(prefill_case.model, plan=layer, phase="prefill")
        d_fmt = gqa_kv_formats(decode_case.model, plan=layer, phase="decode")
        if p_fmt != d_fmt:
            raise ModelingError(
                f"pd_handoff_cache_abi_mismatch:{request}:{logical_layer}:{raw_component}"
            )
        return
    if raw_component == "index_k":
        p_fmt = _resolve_index_cache_format(prefill_case.model, layer, phase="prefill")
        d_fmt = _resolve_index_cache_format(decode_case.model, layer, phase="decode")
        if p_fmt != d_fmt:
            raise ModelingError(
                f"pd_handoff_cache_abi_mismatch:{request}:{logical_layer}:{raw_component}"
            )
        return
    if raw_component in ROW_COMPONENTS:
        p_fmt = cache_format(prefill_case.model, raw_component, plan=layer, phase="prefill")
        d_fmt = cache_format(decode_case.model, raw_component, plan=layer, phase="decode")
        if p_fmt != d_fmt:
            raise ModelingError(
                f"pd_handoff_cache_abi_mismatch:{request}:{logical_layer}:{raw_component}"
            )


def _consecutive_int_runs(values) -> list[list[int]]:
    ordered = sorted(int(v) for v in values)
    if not ordered:
        return []
    runs: list[list[int]] = [[ordered[0]]]
    for val in ordered[1:]:
        if val == runs[-1][-1] + 1:
            runs[-1].append(val)
        else:
            runs.append([val])
    return runs


def _packed_run_offsets(runs: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    prefix = 0
    offsets: list[int] = []
    for lo, hi in runs:
        offsets.append(prefix)
        prefix += hi - lo
    return tuple(offsets)


def _intersect_head_runs(
    src_runs: tuple[tuple[int, int], ...],
    src_offsets: tuple[int, ...],
    dst_runs: tuple[tuple[int, int], ...],
    dst_offsets: tuple[int, ...],
):
    for (d_lo, d_hi), d_prefix in zip(dst_runs, dst_offsets):
        for (s_lo, s_hi), s_prefix in zip(src_runs, src_offsets):
            lo, hi = max(s_lo, d_lo), min(s_hi, d_hi)
            if hi > lo:
                yield s_lo, s_prefix, d_lo, d_prefix, lo, hi


def _emit_head_piece_groups(
    emit, request, emitted_component, logical_layer, source_stage, destination_stage,
    src_rank, dest_rank, src_storage, dst_storage, sspan, dspan,
    src_runs, dst_runs, src_rows, dst_rows, src_comp_bytes, need, per_head_token,
    s_lo, s_prefix, d_lo, d_prefix, grouped, split_kv, model, layer,
) -> None:
    del src_runs, dst_runs, src_rows, dst_rows
    if split_kv:
        k_w, v_w = _gqa_kv_plane_weights(model, layer, phase="decode")
        tot_w = k_w + v_w
        plane_specs = (("gqa_k", k_w), ("gqa_v", v_w))
        for total_val in (per_head_token, src_comp_bytes, need):
            if (total_val * k_w) % tot_w != 0 or (total_val * v_w) % tot_w != 0:
                raise ModelingError(f"pd_transfer_split_kv_non_integer:{request}:{logical_layer}")
    else:
        tot_w = 1
        plane_specs = ((emitted_component, 1),)
    src_heads = sspan[1] - sspan[0]
    dst_heads = dspan[1] - dspan[0]
    for (piece_lo, piece_hi), heads in grouped.items():
        for group in _consecutive_int_runs(heads):
            head_slice = (group[0], group[-1] + 1)
            full_row = len(group) == src_heads == dst_heads
            token_chunks = (
                ((piece_lo, piece_hi),)
                if full_row
                else tuple((tok, tok + 1) for tok in range(piece_lo, piece_hi))
            )
            for t_lo, t_hi in token_chunks:
                src_off = (s_prefix + (t_lo - s_lo)) * src_heads + (group[0] - sspan[0])
                dst_off = (d_prefix + (t_lo - d_lo)) * dst_heads + (group[0] - dspan[0])
                for plane, plane_w in plane_specs:
                    unit = (per_head_token * plane_w) // tot_w
                    src_plane_size = (src_comp_bytes * plane_w) // tot_w
                    dst_plane_size = (need * plane_w) // tot_w
                    payload = unit * (t_hi - t_lo) * len(group)
                    emit(
                        src_rank, dest_rank, plane, payload,
                        source_stage=source_stage, destination_stage=destination_stage,
                        layer=logical_layer, source_span=head_slice, destination_span=head_slice,
                        source_tokens=(t_lo, t_hi), destination_tokens=(t_lo, t_hi),
                        source_storage=src_storage, destination_storage=dst_storage,
                        source_size=src_plane_size, source_displacement=unit * src_off,
                        destination_size=dst_plane_size, destination_displacement=unit * dst_off,
                    )


def _emit_head_transfers(
    emit, request, emitted_component, raw_component, layer, logical_layer,
    source_stage, destination_stage, prefill_rep, decode_rep,
    src_views, dst_views, source_spans, target_spans,
    prefill_case, decode_case, prefill_p, decode_p,
    window_start, tokens, split_kv, *, source_owners=None, target_owners=None,
):
    """Upstream ``push_worker.py:482-505`` GQA head transfer with exact interleaved DCP runs."""
    _assert_matching_cache_abi(prefill_case, decode_case, layer, raw_component, request, logical_layer)
    tp_p = max(1, int(prefill_p.tp))
    pcp_p = max(1, int(prefill_p.pcp))
    tp_d = max(1, int(decode_p.tp))
    pcp_d = max(1, int(decode_p.pcp))
    distinct_p = len(set(source_spans)) if source_spans else 1
    replicas_p = max(1, tp_p // max(1, distinct_p))
    p_dcp = layer_dcp_degree(layer, prefill_p)
    d_dcp = layer_dcp_degree(layer, decode_p)

    def _build_manifest(case, parallel, views, *, explicit_span: bool):
        out = []
        for idx, rank, values, span, cache_rank in views:
            runs = owned_runs(
                raw_component, layer, case.model, parallel, cache_rank, span,
                tokens, window_start, runtime=case.runtime, explicit_span=explicit_span,
            )
            out.append((idx, rank, values, runs, _packed_run_offsets(runs), sum(hi - lo for lo, hi in runs)))
        return out

    src_manifest = _build_manifest(prefill_case, prefill_p, src_views, explicit_span=(source_owners is not None))
    dst_manifest = _build_manifest(decode_case, decode_p, dst_views, explicit_span=(target_owners is not None))

    for dest_index, dest_rank, dest_values, dst_runs, dst_offsets, dst_rows in dst_manifest:
        need = dest_values.get(raw_component, 0)
        dspan = target_spans[dest_index % max(1, len(target_spans))]
        if need <= 0 or dspan[1] <= dspan[0] or dst_rows <= 0:
            continue
        denom = (dspan[1] - dspan[0]) * dst_rows
        if need % denom != 0:
            raise ModelingError(
                f"pd_transfer_non_integer_head_bytes:{request}:{logical_layer}:{need}:{denom}"
            )
        per_head_token = need // denom
        d_tp = dest_index % tp_d
        d_pcp = (dest_index // tp_d) % pcp_d
        if p_dcp == 1 and d_dcp == 1:
            ordered_sources = [
                entry for entry in src_manifest
                if ((entry[0] // tp_p) % pcp_p) == (d_pcp % pcp_p)
                and (entry[0] % tp_p) % replicas_p == (d_tp % replicas_p)
            ]
        else:
            shift = dest_index % max(1, len(src_manifest))
            ordered_sources = list(src_manifest[shift:]) + list(src_manifest[:shift])

        covered: dict[int, list[tuple[int, int]]] = {}
        dst_storage = f"pd:kv:{request}:decode:r{decode_rep}:rank{dest_rank}"
        for src_index, src_rank, src_values, src_runs, src_offsets, src_rows in ordered_sources:
            sspan = source_spans[src_index % max(1, len(source_spans))]
            common = [h for h in range(dspan[0], dspan[1]) if sspan[0] <= h < sspan[1]]
            src_comp_bytes = src_values.get(raw_component, 0)
            if not common or src_comp_bytes <= 0 or src_rows <= 0:
                continue
            src_denom = (sspan[1] - sspan[0]) * src_rows
            if src_comp_bytes % src_denom != 0 or (src_comp_bytes // src_denom) != per_head_token:
                raise ModelingError(
                    f"pd_handoff_cache_abi_mismatch:{request}:{logical_layer}:{raw_component}"
                )
            src_storage = f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}"
            for s_lo, s_prefix, d_lo, d_prefix, lo, hi in _intersect_head_runs(
                src_runs, src_offsets, dst_runs, dst_offsets,
            ):
                grouped: dict[tuple[int, int], list[int]] = {}
                for head in common:
                    for piece_lo, piece_hi in _uncovered_intervals(covered.get(head, ()), lo, hi):
                        grouped.setdefault((piece_lo, piece_hi), []).append(head)
                        covered.setdefault(head, []).append((piece_lo, piece_hi))
                if grouped:
                    _emit_head_piece_groups(
                        emit, request, emitted_component, logical_layer,
                        source_stage, destination_stage, src_rank, dest_rank,
                        src_storage, dst_storage, sspan, dspan,
                        src_runs, dst_runs, src_rows, dst_rows,
                        src_comp_bytes, need, per_head_token,
                        s_lo, s_prefix, d_lo, d_prefix, grouped,
                        split_kv, decode_case.model, layer,
                    )
        missing = [
            h for h in range(dspan[0], dspan[1])
            if sum(p_hi - p_lo for p_lo, p_hi in covered.get(h, ())) < dst_rows
        ]
        if missing:
            raise ModelingError(
                f"pd_handoff_source_head_uncovered:{request}:{logical_layer}:{dest_rank}:{missing}"
            )


def _emit_row_transfers(
    emit, request, emitted_component, raw_component, layer, logical_layer,
    source_stage, destination_stage, prefill_rep, decode_rep,
    src_views, dst_views, prefill_case, decode_case, prefill_p, decode_p,
    tokens, window_start, num_heads, offload, offload_layers, hot_tokens,
    destination, *, source_owners=None, target_owners=None,
):
    """Upstream ``push_worker.py:470-481`` DCP-group-filtered MLA/index row transfer."""
    _assert_matching_cache_abi(prefill_case, decode_case, layer, raw_component, request, logical_layer)

    def manifest(case, parallel, entries, *, explicit_span: bool):
        return [
            (
                index,
                rank,
                owned_runs(
                    raw_component, layer, case.model, parallel, cache_rank, span,
                    tokens, window_start, runtime=case.runtime,
                    explicit_span=explicit_span,
                ),
                values.get(raw_component, 0),
            )
            for index, rank, values, span, cache_rank in entries
        ]

    src_manifest = manifest(prefill_case, prefill_p, src_views, explicit_span=(source_owners is not None))
    dst_manifest = manifest(decode_case, decode_p, dst_views, explicit_span=(target_owners is not None))
    all_sources = [(rank, runs, size) for _idx, rank, runs, size in src_manifest]

    # Upstream push_worker.py:478-481 replica group partitioning (zero 8x rank-0 hotspot).
    if raw_component == "index_k":
        from mindie_sim.catalog.context_parallel import dcp_indexer_sharded

        eff_p_dcp = (
            layer_dcp_degree(layer, prefill_p)
            if (dcp_indexer_sharded(prefill_case.model, prefill_case.runtime)
                or prefill_p.cp_plan == "dsa_cp")
            else 1
        )
    else:
        eff_p_dcp = layer_dcp_degree(layer, prefill_p)
    p_total = len(src_manifest)
    use_groups = (
        source_owners is None
        and eff_p_dcp >= 1
        and p_total >= eff_p_dcp
        and p_total % eff_p_dcp == 0
        and (p_total // eff_p_dcp) > 1
    )
    replicas = (p_total // eff_p_dcp) if use_groups else 1
    grouped_sources: dict[int, list[tuple[int, tuple[tuple[int, int], ...], int]]] = defaultdict(list)
    if use_groups:
        for src_idx, rank, runs, size in src_manifest:
            j_p = _upstream_rank_coord(src_idx, prefill_p)
            grouped_sources[j_p // eff_p_dcp].append((rank, runs, size))

    sparse_layer = offload and raw_component == "mla_kv" and logical_layer in offload_layers
    receivers: list[tuple[int, int, tuple[tuple[int, int], ...], str, str, int | None]] = []
    dst_row_unit: int | None = None
    for _dst_idx, _rank, runs, need in dst_manifest:
        n_rows = sum(hi - lo for lo, hi in runs)
        if need > 0 and n_rows > 0:
            if need % n_rows != 0:
                raise ModelingError(
                    f"pd_transfer_non_integer_row_bytes:need={need},rows={n_rows}"
                )
            dst_row_unit = need // n_rows
            break
    if sparse_layer:
        host_slot = logical_layer % len(destination)
        receivers.append((
            host_slot, destination[host_slot], ((window_start, tokens),),
            "host", f"pd:host:{request}:decode:r{decode_rep}", dst_row_unit,
        ))
    for dst_idx, rank, runs, need in dst_manifest:
        if need <= 0:
            continue
        full_rows = sum(hi - lo for lo, hi in runs)
        rank_unit = (need // full_rows) if full_rows > 0 else dst_row_unit
        if sparse_layer:
            runs = tuple(
                (max(lo, tokens - hot_tokens), hi)
                for lo, hi in runs if hi > max(lo, tokens - hot_tokens)
            )
            if not runs:
                continue
        storage = f"pd:kv:{request}:decode:r{decode_rep}:rank{rank}"
        receivers.append((dst_idx, rank, runs, "hbm", storage, rank_unit))

    ratio = (
        max(1, int(layer.compress_ratio or 1), int(layer.index_kpool or 1))
        if raw_component == "index_k"
        else max(1, int(layer.compress_ratio or 1))
        if raw_component == "compressed_kv"
        else 1
    )
    for dst_idx, dst_rank, runs, tier, storage, expected_unit in receivers:
        if use_groups:
            j_d = _upstream_rank_coord(dst_idx, decode_p) if tier == "hbm" else dst_idx
            cand_sources = grouped_sources.get(j_d % replicas) or all_sources
        else:
            cand_sources = all_sources
        target_rows = sum(hi - lo for lo, hi in runs)
        received = 0
        for src_rank, lo, hi, size, offset, unit in transfers_for_runs(cand_sources, runs):
            if expected_unit is not None and unit != expected_unit:
                raise ModelingError(
                    f"pd_handoff_cache_abi_mismatch:{request}:{logical_layer}:{raw_component}:src={unit}:dst={expected_unit}"
                )
            token_span = (lo * ratio, hi * ratio)
            emit(
                src_rank, dst_rank, emitted_component, (hi - lo) * unit,
                source_stage=source_stage, destination_stage=destination_stage,
                layer=logical_layer, source_span=(0, num_heads), destination_span=(0, num_heads),
                source_tokens=token_span, destination_tokens=token_span, tier=tier,
                source_storage=f"pd:kv:{request}:prefill:r{prefill_rep}:rank{src_rank}",
                destination_storage=storage, source_size=size, source_displacement=offset,
                destination_size=target_rows * unit, destination_displacement=received * unit,
            )
            received += hi - lo


def _cohort_stage_ranks(exec_obj: Any, parallel: Any, replica: int, stage: int) -> tuple[int, ...]:
    """Return full DCP-DP cohort stage ranks for both ``_CohortStageTopology`` and live ``Topology``."""
    topo = exec_obj.topology
    if isinstance(topo, _CohortStageTopology):
        return tuple(topo.ranks(replica, stage))
    dcp_dp = max(1, int(getattr(parallel, "dcp_dp_width", 1) or 1))
    if dcp_dp <= 1:
        return tuple(topo.ranks(replica, stage))
    first_rep = (int(replica) // dcp_dp) * dcp_dp
    out: list[int] = []
    for dp_off in range(dcp_dp):
        out.extend(topo.ranks(first_rep + dp_off, stage))
    return tuple(out)


def _draft_token_runs(
    case: Any,
    parallel: Any,
    memory: Any,
    index: int,
    eff_tokens: int,
    cp: int,
    *,
    stage: int,
    replica: int,
    cohort_size: int,
) -> tuple[tuple[int, int], ...]:
    """Exact half-open interleaved token runs owned by one rank for non-inherited draft KV."""
    toks = max(0, int(eff_tokens))
    if toks <= 0:
        return ()
    cp_n = max(1, int(cp))
    if cp_n <= 1:
        return ((0, toks),)
    if getattr(memory, "rank_token_owners", None) is not None:
        lo, hi = _token_span(memory, index, toks)
        return ((lo, hi),) if hi > lo else ()
    block = max(1, int(cp_interleave(case.runtime)))
    cache_rank = canonical_physical_rank(
        index, parallel, stage=stage, replica=replica, cohort_size=cohort_size,
    )
    return interleaved_token_runs(toks, cp_n, block, cache_rank % cp_n)


def pd_destination_rank_offset(prefill_case: Any, decode_case: Any) -> int:
    """Canonical pool-level connector rank shift (0 when P and D declare disjoint die ranks)."""
    p_dies = tuple(getattr(getattr(prefill_case, "machine", None), "dies", ()) or ())
    d_dies = tuple(getattr(getattr(decode_case, "machine", None), "dies", ()) or ())
    p_ranks = (
        {int(d.rank) for d in p_dies}
        if p_dies else set(range(max(1, int(prefill_case.runtime.parallel.world_size))))
    )
    d_ranks = (
        {int(d.rank) for d in d_dies}
        if d_dies else set(range(max(1, int(decode_case.runtime.parallel.world_size))))
    )
    if not (p_ranks and d_ranks and (p_ranks & d_ranks)):
        return 0
    offset = max(p_ranks) + 1 - min(d_ranks)
    return offset if offset > 0 else max(p_ranks) + 1


def _bind_layer_handoff(
    layer, logical_layer, source_stage, destination_stage, request, tokens,
    published, emit, prefill_exec, decode_exec, prefill_rep, decode_rep,
    prefill_case, decode_case, prefill_p, decode_p, prefill_mem, decode_mem,
    offload, offload_layers, hot_tokens, *, component_prefix: str = "",
):
    source = _cohort_stage_ranks(prefill_exec, prefill_p, prefill_rep, source_stage)
    destination = _cohort_stage_ranks(decode_exec, decode_p, decode_rep, destination_stage)
    if not source or not destination:
        raise ModelingError("pd_handoff_missing_stage_ranks:" + request)
    num_heads = int(getattr(layer, "num_kv_heads", 0) or 0)
    has_head_axis = layer.attention in {"gqa", "projection_only"}
    source_spans = _local_head_spans(num_heads, prefill_p.tp) if has_head_axis else ()
    target_spans = _local_head_spans(num_heads, decode_p.tp) if has_head_axis else ()
    source_owners = getattr(prefill_mem, "rank_token_owners", None)
    target_owners = getattr(decode_mem, "rank_token_owners", None)

    src_views = _handoff_rank_views(
        prefill_case, layer, prefill_p, source, prefill_mem,
        source_owners, source_spans, has_head_axis, tokens,
        stage=source_stage, replica=prefill_rep, phase="prefill",
    )
    dst_views = _handoff_rank_views(
        decode_case, layer, decode_p, destination, decode_mem,
        target_owners, target_spans, has_head_axis, tokens,
        stage=destination_stage, replica=decode_rep, phase="decode",
    )
    components = {name for _, _, values, _, _ in src_views for name in values}
    components |= {name for _, _, values, _, _ in dst_views for name in values}
    gqa_components = {name for name in HEAD_COMPONENTS if has_head_axis and num_heads > 0}
    split_kv = (
        not component_prefix
        and num_heads > 0
        and _published_for(published, "prefill", request, logical_layer, "gqa_k") is not None
        and _published_for(published, "prefill", request, logical_layer, "gqa_v") is not None
    )
    for raw_component in sorted(components):
        emitted_component = f"{component_prefix}{raw_component}"
        window = int(getattr(layer, "window", 0) or 0)
        windowed = raw_component == "window" or (
            raw_component in {"gqa_kv", "mla_kv"} and window > 0
        )
        window_start = max(0, tokens - window) if windowed else 0
        if raw_component in gqa_components:
            _emit_head_transfers(
                emit, request, emitted_component, raw_component, layer, logical_layer,
                source_stage, destination_stage, prefill_rep, decode_rep,
                src_views, dst_views, source_spans, target_spans,
                prefill_case, decode_case, prefill_p, decode_p,
                window_start, tokens, split_kv,
                source_owners=source_owners, target_owners=target_owners,
            )
            continue
        if raw_component in ROW_COMPONENTS:
            _emit_row_transfers(
                emit, request, emitted_component, raw_component, layer, logical_layer,
                source_stage, destination_stage, prefill_rep, decode_rep,
                src_views, dst_views, prefill_case, decode_case, prefill_p, decode_p,
                tokens, window_start, num_heads, offload, offload_layers, hot_tokens,
                destination, source_owners=source_owners, target_owners=target_owners,
            )
            continue
        _emit_state_transfers(
            emit, request, emitted_component, raw_component, layer, logical_layer,
            source_stage, destination_stage, prefill_rep, decode_rep,
            src_views, dst_views, prefill_p, decode_p, num_heads,
        )


def _emit_non_inherited_gqa_draft_handoff(
    draft_info: _ActiveDraftHandoff,
    *,
    body_layers: int,
    request: str,
    emit,
    source_stage: int,
    destination_stage: int,
    source: tuple[int, ...],
    destination: tuple[int, ...],
    prefill_rep: int,
    decode_rep: int,
    prefill_case: Any,
    decode_case: Any,
    prefill_mem: Any,
    decode_mem: Any,
    prefill_p: Any,
    decode_p: Any,
    num_heads: int,
    p_cp: int,
    d_cp: int,
    comp: str,
) -> None:
    """Emit exact integer head+token intersected ``PDTransfer`` records for GQA draft caches."""
    src_head_spans = _local_head_spans(num_heads, prefill_p.tp)
    dst_head_spans = _local_head_spans(num_heads, decode_p.tp)
    p_local_h = src_head_spans[0][1] - src_head_spans[0][0]
    d_local_h = dst_head_spans[0][1] - dst_head_spans[0][0]
    if (
        p_local_h <= 0
        or d_local_h <= 0
        or draft_info.p_row_bytes % p_local_h != 0
        or draft_info.d_row_bytes % d_local_h != 0
    ):
        raise ModelingError(
            f"pd_draft_non_integer_head_bytes:{request}:{draft_info.p_row_bytes}:{draft_info.d_row_bytes}"
        )
    per_head_token = draft_info.d_row_bytes // d_local_h
    if draft_info.p_row_bytes // p_local_h != per_head_token:
        raise ModelingError(
            f"pd_draft_incompatible_head_bytes:{request}:{draft_info.p_row_bytes // p_local_h}:{per_head_token}"
        )
    toks = draft_info.eff_tokens
    tp_p, pcp_p = max(1, int(prefill_p.tp)), max(1, int(prefill_p.pcp))
    tp_d, pcp_d = max(1, int(decode_p.tp)), max(1, int(decode_p.pcp))
    replicas_p = max(1, tp_p // max(1, len(set(src_head_spans))))
    src_entries = []
    for s_idx, s_rank in enumerate(source):
        s_runs = _draft_token_runs(
            prefill_case, prefill_p, prefill_mem, s_idx, toks, p_cp,
            stage=source_stage, replica=prefill_rep, cohort_size=len(source),
        )
        src_entries.append((
            s_idx, s_rank, src_head_spans[s_idx % len(src_head_spans)],
            s_runs, _packed_run_offsets(s_runs), sum(hi - lo for lo, hi in s_runs),
        ))
    dst_entries = []
    for d_idx, d_rank in enumerate(destination):
        d_runs = _draft_token_runs(
            decode_case, decode_p, decode_mem, d_idx, toks, d_cp,
            stage=destination_stage, replica=decode_rep, cohort_size=len(destination),
        )
        dst_entries.append((
            d_idx, d_rank, dst_head_spans[d_idx % len(dst_head_spans)],
            d_runs, _packed_run_offsets(d_runs), sum(hi - lo for lo, hi in d_runs),
        ))
    for layer_offset in range(int(draft_info.d_draft_spec.layers)):
        logical_layer = body_layers + layer_offset
        for d_idx, d_rank, dspan, d_runs, d_offsets, dst_rows in dst_entries:
            if dst_rows <= 0 or dspan[1] <= dspan[0]:
                continue
            dst_size = dst_rows * draft_info.d_row_bytes
            dst_storage = f"pd:kv:{request}:decode:r{decode_rep}:rank{d_rank}"
            if p_cp == 1 and d_cp == 1:
                d_tp, d_pcp = d_idx % tp_d, (d_idx // tp_d) % pcp_d
                ordered = [
                    entry for entry in src_entries
                    if ((entry[0] // tp_p) % pcp_p) == (d_pcp % pcp_p)
                    and (entry[0] % tp_p) % replicas_p == (d_tp % replicas_p)
                ]
            else:
                shift = d_idx % max(1, len(src_entries))
                ordered = src_entries[shift:] + src_entries[:shift]
            covered: dict[int, list[tuple[int, int]]] = {}
            for _s_idx, s_rank, sspan, s_runs, s_offsets, src_rows in ordered:
                common = [h for h in range(dspan[0], dspan[1]) if sspan[0] <= h < sspan[1]]
                if not common or src_rows <= 0:
                    continue
                src_size = src_rows * draft_info.p_row_bytes
                src_storage = f"pd:kv:{request}:prefill:r{prefill_rep}:rank{s_rank}"
                for s_lo, s_prefix, d_lo, d_prefix, lo, hi in _intersect_head_runs(
                    s_runs, s_offsets, d_runs, d_offsets,
                ):
                    grouped: dict[tuple[int, int], list[int]] = {}
                    for head in common:
                        for piece_lo, piece_hi in _uncovered_intervals(covered.get(head, ()), lo, hi):
                            grouped.setdefault((piece_lo, piece_hi), []).append(head)
                            covered.setdefault(head, []).append((piece_lo, piece_hi))
                    if grouped:
                        _emit_head_piece_groups(
                            emit, request, comp, logical_layer,
                            source_stage, destination_stage, s_rank, d_rank,
                            src_storage, dst_storage, sspan, dspan,
                            s_runs, d_runs, src_rows, dst_rows,
                            src_size, dst_size, per_head_token,
                            s_lo, s_prefix, d_lo, d_prefix, grouped,
                            False, None, None,
                        )
            missing = [
                h for h in range(dspan[0], dspan[1])
                if sum(p_hi - p_lo for p_lo, p_hi in covered.get(h, ())) < dst_rows
            ]
            if missing:
                raise ModelingError(
                    f"pd_handoff_source_head_uncovered:{request}:{logical_layer}:{d_rank}:{missing}"
                )


def _bind_non_inherited_draft_handoff(
    draft_info: _ActiveDraftHandoff, body_layers: int, request: str, emit,
    prefill_exec, decode_exec, prefill_rep: int, decode_rep: int,
    prefill_case: Any, decode_case: Any, prefill_mem: Any, decode_mem: Any,
    prefill_p: Any, decode_p: Any,
) -> None:
    """Emit explicit integer ``PDTransfer`` records for paired-SFA / non-inherited draft caches."""
    source_stage = max(0, int(prefill_p.pp) - 1)
    destination_stage = max(0, int(decode_p.pp) - 1)
    source = _cohort_stage_ranks(prefill_exec, prefill_p, prefill_rep, source_stage)
    destination = _cohort_stage_ranks(decode_exec, decode_p, decode_rep, destination_stage)
    if not source or not destination:
        raise ModelingError("pd_handoff_missing_stage_ranks:" + request)
    p_cp = int(prefill_p.cp) if draft_info.p_sharded else 1
    d_cp = int(decode_p.cp) if draft_info.d_sharded else 1
    comp = f"draft.{draft_info.d_draft_spec.attention_kind}_kv"
    num_heads = int(getattr(draft_info.d_draft_spec, "num_kv_heads", 0) or 0)
    if str(getattr(draft_info.d_draft_spec, "attention_kind", "")) != "mla" and num_heads > 0:
        _emit_non_inherited_gqa_draft_handoff(
            draft_info, body_layers=body_layers, request=request, emit=emit,
            source_stage=source_stage, destination_stage=destination_stage,
            source=source, destination=destination,
            prefill_rep=prefill_rep, decode_rep=decode_rep,
            prefill_case=prefill_case, decode_case=decode_case,
            prefill_mem=prefill_mem, decode_mem=decode_mem,
            prefill_p=prefill_p, decode_p=decode_p,
            num_heads=num_heads, p_cp=p_cp, d_cp=d_cp, comp=comp,
        )
        return
    toks = draft_info.eff_tokens
    src_runs_all = []
    for s_idx, s_rank in enumerate(source):
        s_runs = _draft_token_runs(
            prefill_case, prefill_p, prefill_mem, s_idx, toks, p_cp,
            stage=source_stage, replica=prefill_rep, cohort_size=len(source),
        )
        src_runs_all.append((s_idx, s_rank, s_runs, sum(hi - lo for lo, hi in s_runs) * draft_info.p_row_bytes))

    replicas = len(source) // max(1, p_cp) if len(source) % max(1, p_cp) == 0 else 1
    grouped: dict[int, list[tuple[int, tuple[tuple[int, int], ...], int]]] = defaultdict(list)
    if replicas > 1:
        for s_idx, s_rank, runs, size in src_runs_all:
            j_p = _upstream_rank_coord(s_idx, prefill_p)
            grouped[j_p // p_cp].append((s_rank, runs, size))
    flat_sources = [(s_rank, runs, size) for _idx, s_rank, runs, size in src_runs_all]

    for layer_offset in range(int(draft_info.d_draft_spec.layers)):
        logical_layer = body_layers + layer_offset
        for d_idx, d_rank in enumerate(destination):
            d_runs = _draft_token_runs(
                decode_case, decode_p, decode_mem, d_idx, toks, d_cp,
                stage=destination_stage, replica=decode_rep, cohort_size=len(destination),
            )
            target_rows = sum(hi - lo for lo, hi in d_runs)
            if target_rows <= 0:
                continue
            j_d = _upstream_rank_coord(d_idx, decode_p)
            cand_sources = grouped.get(j_d % replicas) or flat_sources if replicas > 1 else flat_sources
            received = 0
            dst_storage = f"pd:kv:{request}:decode:r{decode_rep}:rank{d_rank}"
            for s_rank, lo, hi, size, offset, unit in transfers_for_runs(cand_sources, d_runs):
                emit(
                    s_rank, d_rank, comp, (hi - lo) * unit,
                    source_stage=source_stage, destination_stage=destination_stage,
                    layer=logical_layer, source_span=(0, num_heads), destination_span=(0, num_heads),
                    source_tokens=(lo, hi), destination_tokens=(lo, hi), tier="hbm",
                    source_storage=f"pd:kv:{request}:prefill:r{prefill_rep}:rank{s_rank}",
                    destination_storage=dst_storage, source_size=size, source_displacement=offset,
                    destination_size=target_rows * draft_info.d_row_bytes,
                    destination_displacement=received * draft_info.d_row_bytes,
                )
                received += hi - lo


def _verify_disjoint_integer_partition(
    request: str,
    transfers: list[PDTransfer],
    source_extents: Mapping[tuple[int, str, int], int],
    destination_extents: Mapping[tuple[int, str, int], int],
    expected_volume: int,
) -> None:
    """Verify exact 0-byte disjoint integer half-open ``[start, end)`` range conservation."""
    if not transfers:
        if expected_volume > 0:
            raise ModelingError(
                f"pd_handoff_empty_transfers_with_nonzero_demand:{request}:{expected_volume}"
            )
        return
    emitted_total = 0
    dst_intervals: dict[tuple[int, str, int], list[tuple[int, int]]] = defaultdict(list)
    for item in transfers:
        emitted_total += item.payload_bytes
        if not item.source_byte_range or not item.destination_byte_range:
            raise ModelingError(f"pd_handoff_missing_byte_range:{request}:{item.component}")
        s_lo, s_hi = item.source_byte_range
        s_key = (item.source_rank, item.source_storage, item.source_version)
        if s_lo < 0 or s_hi > source_extents.get(s_key, 0):
            raise ModelingError(f"pd_handoff_source_range_out_of_bounds:{request}:{s_key}")
        d_key = (item.destination_rank, item.destination_storage, item.destination_version)
        dst_intervals[d_key].append(item.destination_byte_range)
    if emitted_total != expected_volume:
        raise ModelingError(
            f"pd_handoff_physical_range_not_conserved:{request}:{expected_volume}!={emitted_total}"
        )
    for d_key, spans in dst_intervals.items():
        ordered = sorted(spans)
        cursor = 0
        for d_lo, d_hi in ordered:
            if d_lo < cursor:
                raise ModelingError(
                    f"pd_handoff_destination_range_overlap:{request}:{d_key}:{d_lo}<{cursor}"
                )
            if d_lo > cursor:
                raise ModelingError(
                    f"pd_handoff_destination_range_gap:{request}:{d_key}:{cursor}->{d_lo}"
                )
            cursor = d_hi
        if cursor != destination_extents.get(d_key, 0):
            raise ModelingError(
                f"pd_handoff_destination_extent_mismatch:{request}:{d_key}:{cursor}"
            )


def _make_pd_transfer_emitter(
    *,
    request,
    published,
    prefill_mem,
    decode_mem,
    prefill_rep: int,
    decode_rep: int,
    plane_bases: dict[Any, int],
    source_extents: dict[tuple[int, str, int], int],
    destination_planes: dict[Any, int],
    destination_offsets: dict[tuple[int, str, int], int],
    transfers: list[PDTransfer],
    destination_bytes: dict[int, int],
    host_bytes: dict[int, int],
    source_ranks: set[int],
    destination_ranks: set[int],
):
    """Build the integer-byte ``PDTransfer`` emitter bound to one handoff plan."""
    def emit(source_rank, destination_rank, component, payload, *,
             source_stage, destination_stage, layer, source_span,
             destination_span, source_tokens, destination_tokens,
             tier="hbm", source_storage, destination_storage,
             source_component=None, destination_component=None,
             source_size, source_displacement=0, destination_size, destination_displacement=0):
        payload_i = exact_int_bytes(payload, f"pd_transfer_payload:{component}")
        if payload_i <= 0:
            return
        src_size_i = exact_int_bytes(source_size, f"pd_transfer_src_size:{component}")
        src_disp_i = exact_int_bytes(source_displacement, f"pd_transfer_src_disp:{component}")
        dst_size_i = exact_int_bytes(destination_size, f"pd_transfer_dst_size:{component}")
        dst_disp_i = exact_int_bytes(destination_displacement, f"pd_transfer_dst_disp:{component}")
        src_comp = source_component or component
        dst_comp = destination_component or component
        src_rec = _published_for(published, "prefill", request, layer, src_comp)
        dst_rec = _published_for(published, "decode", request, layer, dst_comp)
        producer = (
            str(src_rec.get("output")) if src_rec and src_rec.get("output")
            else f"prefill:{request}:s{source_stage}:l{layer}:{src_comp}"
        )
        output = (
            str(dst_rec.get("output")) if dst_rec and dst_rec.get("output")
            else f"decode:{request}:s{destination_stage}:l{layer}:{dst_comp}"
        )
        src_version = (
            int(src_rec["version"]) if src_rec and src_rec.get("version") is not None
            else _component_versions(prefill_mem, prefill_rep, request, layer, src_comp)
        )
        dst_version = (
            int(dst_rec["version"]) if dst_rec and dst_rec.get("version") is not None
            else _component_versions(decode_mem, decode_rep, request, layer, dst_comp)
        )
        src_stor = str(src_rec["storage"]) if src_rec and src_rec.get("storage") else source_storage
        dst_stor = str(dst_rec["storage"]) if dst_rec and dst_rec.get("storage") else destination_storage
        src_offset = address(
            plane_bases, source_extents, (int(source_rank), src_stor, src_version),
            (int(layer), src_comp), src_size_i, src_disp_i, payload_i,
        )
        dst_offset = address(
            destination_planes, destination_offsets, (int(destination_rank), dst_stor, dst_version),
            (int(layer), dst_comp), dst_size_i, dst_disp_i, payload_i,
        )
        transfers.append(PDTransfer(
            int(source_rank), int(destination_rank), payload_i,
            source_stage=int(source_stage), destination_stage=int(destination_stage),
            source_layers=(int(layer),), destination_layers=(int(layer),),
            component=component,
            source_version=src_version, destination_version=dst_version,
            source_heads=source_span, destination_heads=destination_span,
            source_token_span=source_tokens, destination_token_span=destination_tokens,
            producer=producer, output=output, destination_tier=tier,
            source_storage=src_stor, destination_storage=dst_stor,
            source_byte_range=(src_offset, src_offset + payload_i),
            destination_byte_range=(dst_offset, dst_offset + payload_i),
            source_scope=(
                str(src_rec.get("scope")) if src_rec
                else f"prefill:r{prefill_rep}:s{source_stage}:rank{source_rank}"
            ),
            destination_scope=(
                str(dst_rec.get("scope")) if dst_rec
                else f"decode:r{decode_rep}:s{destination_stage}:rank{destination_rank}"
            ),
        ))
        if tier == "host":
            host_bytes[int(destination_rank)] += payload_i
        else:
            destination_bytes[int(destination_rank)] += payload_i
        source_ranks.add(int(source_rank))
        destination_ranks.add(int(destination_rank))

    return emit


def _pd_handoff_binding(request, tokens, placement, cases, memory, execution,
                        decode_local, volume, demands, shared_buffer_prefix=None,
                        shared_buffer_capacity=None, published=None, *,
                        active_draft: _ActiveDraftHandoff | None = None,
                        source_plane_bases_out: dict[Any, int] | None = None):
    """Bind actual source/destination layer, component, head and token integer ranges."""
    del decode_local
    prefill_exec = execution["prefill"]
    decode_exec = execution["decode"]
    prefill_rep = placement["prefill", request]
    decode_rep = placement["decode", request]
    prefill_case, decode_case = cases["prefill"], cases["decode"]
    prefill_p, decode_p = prefill_case.runtime.parallel, decode_case.runtime.parallel
    prefill_mem, decode_mem = memory["prefill"], memory["decode"]
    offload, offload_layers, hot_tokens = _sparse_offload_params(decode_mem, tokens)
    expected_volume = exact_int_bytes(volume, "pd_handoff_expected_volume")

    transfers: list[PDTransfer] = []
    destination_bytes: dict[int, int] = defaultdict(int)
    host_bytes: dict[int, int] = defaultdict(int)
    source_ranks: set[int] = set()
    destination_ranks: set[int] = set()
    plane_bases: dict[Any, int] = {}
    source_extents: dict[tuple[int, str, int], int] = defaultdict(int)
    destination_planes: dict[Any, int] = {}
    destination_offsets: dict[tuple[int, str, int], int] = defaultdict(int)

    emit = _make_pd_transfer_emitter(
        request=request, published=published,
        prefill_mem=prefill_mem, decode_mem=decode_mem,
        prefill_rep=prefill_rep, decode_rep=decode_rep,
        plane_bases=plane_bases, source_extents=source_extents,
        destination_planes=destination_planes, destination_offsets=destination_offsets,
        transfers=transfers, destination_bytes=destination_bytes,
        host_bytes=host_bytes, source_ranks=source_ranks, destination_ranks=destination_ranks,
    )

    for layer in prefill_case.model.layer_plans:
        _bind_layer_handoff(
            layer, int(layer.index),
            int(prefill_mem.layer_stage[layer.index]),
            int(decode_mem.layer_stage[layer.index]),
            request, tokens, published, emit, prefill_exec, decode_exec,
            prefill_rep, decode_rep, prefill_case, decode_case, prefill_p, decode_p,
            prefill_mem, decode_mem, offload, offload_layers, hot_tokens,
        )

    if active_draft is not None:
        body_len = len(prefill_case.model.layer_plans)
        if active_draft.inherited:
            src_st = max(0, int(prefill_p.pp) - 1)
            dst_st = max(0, int(decode_p.pp) - 1)
            for idx, d_layer in enumerate(active_draft.d_draft_plans):
                _bind_layer_handoff(
                    d_layer, body_len + idx, src_st, dst_st,
                    request, active_draft.eff_tokens, published, emit,
                    prefill_exec, decode_exec, prefill_rep, decode_rep,
                    prefill_case, decode_case, prefill_p, decode_p,
                    prefill_mem, decode_mem, False, set(), 0,
                    component_prefix="draft.",
                )
        else:
            _bind_non_inherited_draft_handoff(
                active_draft, body_len, request, emit, prefill_exec, decode_exec,
                prefill_rep, decode_rep, prefill_case, decode_case,
                prefill_mem, decode_mem, prefill_p, decode_p,
            )

    if source_plane_bases_out is not None:
        source_plane_bases_out.update(plane_bases)

    _verify_disjoint_integer_partition(
        request, transfers, source_extents, destination_offsets, expected_volume,
    )
    if not transfers:
        return PDHandoff.build(
            request, (), memory_work=tuple(sorted(demands.items()))), (), ()

    buffers = (
        {rank: f"{shared_buffer_prefix}:rank{rank}" for rank in destination_ranks}
        if shared_buffer_prefix is not None
        else {rank: f"pd:kv:{request}:decode:r{decode_rep}:rank{rank}" for rank in destination_ranks}
    )
    pool_caps = (
        {rank: exact_int_bytes(shared_buffer_capacity, "pd_shared_buffer_capacity")
         for rank in destination_bytes}
        if shared_buffer_capacity is not None else None
    )
    handoff = PDHandoff.build(
        request, transfers, memory_work=tuple(sorted(demands.items())),
        receive_capacity_bytes=dict(destination_bytes),
        receive_buffer_ids=buffers,
        receive_pool_capacity_bytes=pool_caps,
        receive_host_bytes=dict(host_bytes),
    )
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
    """Real hold/consumer events currently retaining this service's ledgers."""
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

    def __init__(self, parallel, *, machine=None, rank_offset: int = 0) -> None:
        self._pp = max(1, int(parallel.pp))
        self._tp = max(1, int(parallel.tp))
        self._pcp = max(1, int(parallel.pcp))
        self._dcp_dp = max(1, int(getattr(parallel, "dcp_dp_width", 1) or 1))
        self._local_width = max(1, self._tp * self._pcp)
        self._width = max(1, self._local_width * self._dcp_dp)
        self._dies = tuple(getattr(machine, "dies", ()) or ()) if machine is not None else ()
        self._offset = int(rank_offset)

    def ranks(self, replica: int, stage: int) -> tuple[int, ...]:
        first_rep = (int(replica) // self._dcp_dp) * self._dcp_dp
        if self._dies:
            out: list[int] = []
            for dp_off in range(self._dcp_dp):
                rep_i = first_rep + dp_off
                start = (rep_i * self._pp + int(stage)) * self._local_width
                chunk = self._dies[start:start + self._local_width]
                if len(chunk) != self._local_width:
                    out = []
                    break
                out.extend(int(dev.rank) + self._offset for dev in chunk)
            if out:
                return tuple(out)
        base = self._offset + (int(replica) * self._pp + int(stage)) * self._width
        return tuple(range(base, base + self._width))


def _union_interval_length(intervals) -> int:
    """Exact integer length of the union of half-open ``[lo, hi)`` byte intervals."""
    ordered = sorted(
        (int(lo), int(hi)) for lo, hi in intervals if int(hi) > int(lo)
    )
    if not ordered:
        return 0
    total = 0
    cur_lo, cur_hi = ordered[0]
    for lo, hi in ordered[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    return total + (cur_hi - cur_lo)


def _summarize_unique_logical_handoff_bytes(
    handoff: PDHandoff, p_case, p_topo, plane_bases: Mapping[Any, int] | None = None,
) -> tuple[int, int]:
    """Compute replica-deduplicated logical ``(total_bytes, non_token_state_bytes)``."""
    logical_row_units: dict[tuple[Any, ...], tuple[int, int]] = {}
    logical_row_spans: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    logical_plane_spans: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    logical_state_spans: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    p_par = p_case.runtime.parallel
    p_rank_to_idx: dict[tuple[int, int], int] = {}
    for st in range(p_topo._pp):
        for idx, rk in enumerate(p_topo.ranks(0, st)):
            p_rank_to_idx[(st, rk)] = idx

    def _plane_relative_range(item: PDTransfer, layer_id: int) -> tuple[int, int]:
        lo, hi = int(item.source_byte_range[0]), int(item.source_byte_range[1])
        if plane_bases:
            b_key = (
                (int(item.source_rank), str(item.source_storage), int(item.source_version)),
                (int(layer_id), str(item.component)),
            )
            base = plane_bases.get(b_key)
            if base is not None:
                return (lo - int(base), hi - int(base))
        return (0, hi - lo)

    for item in handoff.transfers:
        layer_id = item.source_layers[0] if item.source_layers else 0
        raw_comp = item.component.removeprefix("draft.")
        if raw_comp in ROW_COMPONENTS and item.source_token_span:
            t_lo, t_hi = item.source_token_span
            if t_hi > t_lo:
                key = (layer_id, item.component)
                logical_row_spans[key].append((t_lo, t_hi))
                logical_row_units[key] = (item.payload_bytes, t_hi - t_lo)
        elif raw_comp in HEAD_COMPONENTS or raw_comp in {"gqa_k", "gqa_v"}:
            if item.source_token_span and item.source_heads:
                t_lo, t_hi = item.source_token_span
                h_lo, h_hi = item.source_heads
                if t_hi > t_lo and h_hi > h_lo:
                    n_units = (h_hi - h_lo) * (t_hi - t_lo)
                    for head_id in range(h_lo, h_hi):
                        h_key = (layer_id, item.component, head_id)
                        logical_row_spans[h_key].append((t_lo, t_hi))
                        logical_row_units[h_key] = (item.payload_bytes, n_units)
                    continue
            key = (layer_id, item.component, item.source_heads, item.source_token_span)
            if item.source_byte_range:
                logical_plane_spans[key].append(_plane_relative_range(item, layer_id))
        elif raw_comp in SHARDED_STATE_COMPONENTS:
            s_idx = p_rank_to_idx.get((item.source_stage, item.source_rank), 0)
            layer_obj = (
                p_case.model.layer_plans[layer_id]
                if 0 <= layer_id < len(p_case.model.layer_plans) else None
            )
            attn = str(getattr(layer_obj, "attention", "") or "").strip().lower()
            tp_n = max(1, int(p_par.tp))
            pcp_n = max(1, int(p_par.pcp))
            shard_id = ((s_idx % tp_n) * pcp_n + (s_idx // tp_n) % pcp_n) if attn == "kda" else (s_idx % tp_n)
            key = (layer_id, item.component, shard_id)
            if item.source_byte_range:
                logical_state_spans[key].append(_plane_relative_range(item, layer_id))
        else:
            key = (layer_id, item.component, "rep")
            if item.source_byte_range:
                logical_state_spans[key].append(_plane_relative_range(item, layer_id))
    unique_total = 0
    for key, spans in logical_row_spans.items():
        union_toks = _union_interval_length(spans)
        pay_ref, span_ref = logical_row_units[key]
        unique_total += (union_toks * pay_ref) // span_ref
    for spans in logical_plane_spans.values():
        unique_total += _union_interval_length(spans)
    unique_state = sum(_union_interval_length(spans) for spans in logical_state_spans.values())
    return int(unique_total + unique_state), int(unique_state)


def compute_pd_handoff_plan(
    prefill_case,
    decode_case,
    *,
    tokens: int,
    prefill_memory=None,
    decode_memory=None,
    connector_kind: str = "auto",
    strict_connector: bool = False,
) -> dict:
    """[Cohesive-Pipeline] Compute exact integer shard-aware PD KV/Indexer/State/Draft handoff plan."""
    from ..capacity import Memory
    from ..structure import parse_case

    p_case = parse_case(prefill_case) if isinstance(prefill_case, Mapping) else prefill_case
    d_case = parse_case(decode_case) if isinstance(decode_case, Mapping) else decode_case
    conn_ok, resolved_conn, conn_reason = resolve_and_validate_pd_connector(
        d_case.model, p_case.runtime.parallel, d_case.runtime.parallel,
        deployment_mode="pd_disaggregated", connector_kind=connector_kind,
    )
    if not conn_ok and (strict_connector or connector_kind == "MooncakeHybridConnector"):
        raise ModelingError(str(conn_reason or "pd_connector_incompatible"))

    p_mem = prefill_memory if prefill_memory is not None else Memory(p_case)
    d_mem = decode_memory if decode_memory is not None else Memory(d_case)
    n_tokens = max(0, int(tokens))
    if n_tokens <= 0:
        return {
            "connector_kind": resolved_conn,
            "hybrid_connector_valid": bool(conn_ok),
            "hybrid_connector_reason": conn_reason,
            "total_transfer_bytes": 0,
            "unique_kv_bytes": 0,
            "unique_state_bytes": 0,
            "draft_kv_bytes": 0,
            "draft_state_bytes": 0,
            "hbm_transfer_bytes": 0,
            "host_transfer_bytes": 0,
            "per_layer_bytes": {},
            "tail_layer_bytes": 0,
            "source_rank_bytes": {},
            "destination_rank_bytes": {},
            "destination_hbm_rank_bytes": {},
            "destination_host_rank_bytes": {},
            "unique_source_rank_bytes": {},
            "unique_source_rank_state_bytes": {},
            "destination_rank_offset": 0,
            "source_ranks": (),
            "destination_ranks": (),
            "transfer_count": 0,
            "transfers": (),
        }

    demand = compute_expected_pd_handoff_demand(p_case, d_case, p_mem, d_mem, n_tokens)
    p_topo = _CohortStageTopology(p_case.runtime.parallel, machine=p_case.machine, rank_offset=0)
    d_offset = pd_destination_rank_offset(p_case, d_case)
    d_topo = _CohortStageTopology(d_case.runtime.parallel, machine=d_case.machine, rank_offset=d_offset)
    execution = {
        "prefill": SimpleNamespace(topology=p_topo),
        "decode": SimpleNamespace(topology=d_topo),
    }
    req_id = "__pd_handoff__"
    placement = {("prefill", req_id): 0, ("decode", req_id): 0}
    src_plane_bases: dict[Any, int] = {}
    handoff, src_ranks, dst_ranks = _pd_handoff_binding(
        req_id, n_tokens, placement,
        {"prefill": p_case, "decode": d_case},
        {"prefill": p_mem, "decode": d_mem},
        execution, demand.decode_local_stage_bytes, demand.total_volume_bytes, {},
        active_draft=demand.active_draft,
        source_plane_bases_out=src_plane_bases,
    )

    src_bytes: dict[int, int] = defaultdict(int)
    dst_bytes: dict[int, int] = defaultdict(int)
    dst_hbm_bytes: dict[int, int] = defaultdict(int)
    dst_host_bytes: dict[int, int] = defaultdict(int)
    per_layer: dict[int, int] = defaultdict(int)
    unique_ranges: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    unique_state_ranges: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    hbm_total = 0
    host_total = 0
    body_layers = len(p_case.model.layer_plans)
    for item in handoff.transfers:
        payload = item.payload_bytes
        src_bytes[item.source_rank] += payload
        dst_bytes[item.destination_rank] += payload
        if item.destination_tier == "host":
            host_total += payload
            dst_host_bytes[item.destination_rank] += payload
        else:
            hbm_total += payload
            dst_hbm_bytes[item.destination_rank] += payload
        layers_tup = item.source_layers or (0,)
        if len(layers_tup) == 1:
            per_layer[layers_tup[0]] += payload
        else:
            share = payload // len(layers_tup)
            for layer_idx in layers_tup:
                per_layer[layer_idx] += share
        if item.source_byte_range:
            key_r = (item.source_rank, item.source_storage)
            unique_ranges[key_r].append(item.source_byte_range)
            raw_comp = item.component.removeprefix("draft.")
            if raw_comp not in _TOKEN_ADDITIVE_COMPONENTS:
                unique_state_ranges[key_r].append(item.source_byte_range)

    unique_by_src = {
        s_rank: sum(
            _union_interval_length(spans)
            for (rank_k, _stor), spans in unique_ranges.items()
            if rank_k == s_rank
        )
        for s_rank in sorted({rk for rk, _ in unique_ranges})
    }
    unique_state_by_src = {
        s_rank: sum(
            _union_interval_length(spans)
            for (rank_k, _stor), spans in unique_state_ranges.items()
            if rank_k == s_rank
        )
        for s_rank in sorted({rk for rk, _ in unique_state_ranges})
    }
    unique_total, unique_state_total = _summarize_unique_logical_handoff_bytes(
        handoff, p_case, p_topo, src_plane_bases,
    )
    total_bytes = hbm_total + host_total
    if unique_total <= 0:
        unique_total = total_bytes
    body_layer_keys = [k for k in per_layer if k < body_layers]
    tail_key = max(body_layer_keys) if body_layer_keys else (max(per_layer) if per_layer else -1)
    tail_bytes = per_layer[tail_key] if tail_key >= 0 else 0

    return {
        "connector_kind": resolved_conn,
        "hybrid_connector_valid": bool(conn_ok),
        "hybrid_connector_reason": conn_reason,
        "total_transfer_bytes": int(total_bytes),
        "unique_kv_bytes": int(unique_total),
        "unique_state_bytes": int(unique_state_total),
        "draft_kv_bytes": int(demand.draft_kv_bytes),
        "draft_state_bytes": int(demand.draft_state_bytes),
        "hbm_transfer_bytes": int(hbm_total),
        "host_transfer_bytes": int(host_total),
        "per_layer_bytes": {int(k): int(v) for k, v in sorted(per_layer.items())},
        "tail_layer_bytes": int(tail_bytes),
        "source_rank_bytes": {int(k): int(v) for k, v in sorted(src_bytes.items())},
        "destination_rank_bytes": {int(k): int(v) for k, v in sorted(dst_bytes.items())},
        "destination_hbm_rank_bytes": {int(k): int(v) for k, v in sorted(dst_hbm_bytes.items())},
        "destination_host_rank_bytes": {int(k): int(v) for k, v in sorted(dst_host_bytes.items())},
        "unique_source_rank_bytes": {int(k): int(v) for k, v in sorted(unique_by_src.items())},
        "unique_source_rank_state_bytes": {int(k): int(v) for k, v in sorted(unique_state_by_src.items())},
        "destination_rank_offset": int(d_offset),
        "source_ranks": tuple(int(r) for r in src_ranks),
        "destination_ranks": tuple(int(r) for r in dst_ranks),
        "transfer_count": len(handoff.transfers),
        "transfers": handoff.transfers,
    }
