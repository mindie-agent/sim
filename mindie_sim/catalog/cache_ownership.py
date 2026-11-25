"""Physical shared/private cache ownership ranges and batch footprints.

[Role]
  Turn declared prefix identities into physical cache ranges.  A shared prefix
  owner is reserved once per DP replica; each request owns only its private
  suffix/COW tail, recurrent state and draft cache.  The same ranges feed
  capacity admission and read-traffic planning.

[Invariants]
  * Token ranges are intersected with actual ownership; a partially filled
    prefix block that receives private tokens is a private COW tail.
  * Sliding-window prefixes retain only live window rows; compressed groups
    share only complete groups entirely inside the immutable prefix.
  * Different prefix keys never pool; private mode ignores the key and charges
    each request exactly as before.  B=1/P=0/private reproduce old totals.
  * State and draft state stay per request; target-shared draft components use
    the inherited draft plans, not ``draft.layers * dense_row * tokens``.

[Timeouts & Locks]
  Bounded by requests x layers; no I/O or locks.

[Side-Effect Policy]
  Pure immutable records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from typing import Any

from mindie_sim.catalog.cache import (
    _LINEAR_ATTENTION_KINDS, layer_cache_components, layer_stage_map,
    owned_token_rows,
)
from mindie_sim.catalog.cache_pools import _ceil_div
from mindie_sim.catalog.context_parallel import (
    cp_interleave,
    layer_dcp_degree,
    layer_dcp_factors,
    owned_positions,
)
from mindie_sim.schema.common import ModelingError


ZERO = 0
ONE = 1
ZERO_FLOAT = 0.0
DEFAULT_INTERLEAVE = 128
FIRST = 0
SECOND = 1
SHARING_SHARED = "shared"
SHARING_PRIVATE = "private"
STATE_COMPONENTS = frozenset({
    "recurrent_state", "conv_state", "replay_records", "state_transient",
    "compressor_state", "index_compressor_state", "index_pool_tail",
    "ple_conv_state",
})
WINDOW_BOUND_COMPONENTS = frozenset({"window", "gqa_kv", "mla_kv"})
_TARGET_ONLY_KV_META_KEYS = (
    "kv_cache_tag",
    "kv_cache_type",
    "cache_representations",
    "layer_cache_representations",
    "kv_scale_dtype",
)


@dataclass(frozen=True)
class CacheRange:
    """One physical cache interval with explicit sharing ownership and page IDs."""

    layer: int
    component: str
    representation: str
    start_token: int
    end_token: int
    prefix_key: str
    generation: int
    sharing: str
    rank: int | None
    allocated_bytes: float
    valid_bytes: float
    owner_key: str = ""
    page_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class BatchCacheFootprint:
    """Per-stage physical footprint for one DP replica's request batch."""

    shared_once_bytes: tuple[float, ...]
    private_per_request_bytes: tuple[float, ...]
    state_bytes: tuple[float, ...]
    draft_bytes: tuple[float, ...]
    boundary_padding_bytes: tuple[float, ...]
    total_bytes: tuple[float, ...]
    requests: int = ZERO
    stages: int = ZERO

    def _sum(self, values: tuple[float, ...]) -> float:
        return float(math.fsum(values))

    @property
    def shared_once_total(self) -> float:
        return self._sum(self.shared_once_bytes)

    @property
    def private_total(self) -> float:
        return self._sum(self.private_per_request_bytes)

    @property
    def state_total(self) -> float:
        return self._sum(self.state_bytes)

    @property
    def draft_total(self) -> float:
        return self._sum(self.draft_bytes)

    @property
    def total(self) -> float:
        return self._sum(self.total_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_once_bytes": list(self.shared_once_bytes),
            "private_per_request_bytes": list(self.private_per_request_bytes),
            "state_bytes": list(self.state_bytes),
            "draft_bytes": list(self.draft_bytes),
            "boundary_padding_bytes": list(self.boundary_padding_bytes),
            "total_bytes": list(self.total_bytes),
            "requests": int(self.requests),
            "stages": int(self.stages),
        }


def _layer_stages(model: Any, parallel: Any, runtime: Any = None) -> dict[int, int]:
    stage_map = layer_stage_map(model, parallel, runtime)
    return {int(plan.index): stage_map[int(plan.index)]
            for plan in model.layer_plans
            if 0 <= int(plan.index) < len(stage_map)}


def _block_tokens(tokens: int, interleave: int) -> int:
    if interleave <= ONE:
        return int(tokens)
    return (int(tokens) // int(interleave)) * int(interleave)


def _component_prefix_tokens(plan: Any, tokens: int) -> int:
    ratio = max(ZERO, int(getattr(plan, "compress_ratio", ZERO) or ZERO))
    if ratio > ONE:
        return (int(tokens) // ratio) * ratio
    return int(tokens)


def _page_id_span(start_token: int, end_token: int, page_tokens: int) -> tuple[int, ...]:
    lo = max(0, int(start_token))
    hi = max(lo, int(end_token))
    if hi <= lo:
        return ()
    stride = max(1, int(page_tokens))
    first_page = lo // stride
    last_page = _ceil_div(hi, stride)
    return tuple(range(first_page, last_page))


def _rank_owned_subintervals(
    layer: Any,
    parallel: Any,
    start: int,
    end: int,
    *,
    interleave: int,
    rank: int,
    owner_spans: Any,
) -> tuple[tuple[int, int], ...]:
    """Exact disjoint half-open token intervals owned by ``rank`` inside ``[start, end)``."""
    lo_bound, hi_bound = max(0, int(start)), max(0, int(end))
    if hi_bound <= lo_bound:
        return ()
    if owner_spans is not None:
        spans = tuple((int(lo), int(hi)) for lo, hi in owner_spans)
        r_idx = int(rank)
        if not 0 <= r_idx < len(spans):
            raise ModelingError(f"cache_owner_rank_out_of_range:{rank}")
        out: list[tuple[int, int]] = []
        lo, hi = spans[r_idx]
        a, b = max(lo_bound, lo), min(hi_bound, hi)
        if b > a:
            out.append((a, b))
        if r_idx == len(spans) - 1:
            max_hi = max((span_hi for _, span_hi in spans), default=0)
            tail_a = max(lo_bound, max_hi)
            if hi_bound > tail_a:
                out.append((tail_a, hi_bound))
        return tuple(out)
    cp = layer_dcp_degree(layer, parallel)
    if cp <= 1:
        return ((lo_bound, hi_bound),)
    pcp_f, tp_g, dp_w = layer_dcp_factors(layer, parallel)
    tp_dim = max(1, int(parallel.tp))
    pcp_dim = max(1, int(parallel.pcp))
    pp_dim = max(1, int(parallel.pp))
    dp_dim = max(1, int(parallel.dp))
    r_int = int(rank)
    tp_i = r_int % tp_dim
    pcp_i = (r_int // tp_dim) % pcp_dim
    dp_i = (r_int // (tp_dim * pcp_dim * pp_dim)) % dp_dim
    if dp_w > 1:
        cp_rank = ((dp_i % dp_w) * (pcp_f * tp_g) + (pcp_i % pcp_f) * tp_g + (tp_i % tp_g)) % cp
    else:
        cp_rank = ((tp_i % tp_g) * pcp_f + (pcp_i % pcp_f)) % cp
    stride = max(1, int(interleave))
    first_chunk = lo_bound // stride
    chunk = first_chunk + ((cp_rank - (first_chunk % cp)) % cp)
    out = []
    while chunk * stride < hi_bound:
        a = max(lo_bound, chunk * stride)
        b = min(hi_bound, (chunk + 1) * stride)
        if b > a:
            out.append((a, b))
        chunk += cp
    return tuple(out)


def _rank_owned_interval(
    layer: Any,
    parallel: Any,
    start: int,
    end: int,
    *,
    interleave: int,
    rank: int,
    owner_spans: Any,
) -> int:
    """Exact token count owned by ``rank`` in half-open interval ``[start, end)``."""
    return sum(
        b - a
        for a, b in _rank_owned_subintervals(
            layer, parallel, start, end,
            interleave=interleave, rank=rank, owner_spans=owner_spans,
        )
    )


def cache_component_ranges(
    model: Any, layer: Any, parallel: Any, *, sequence_tokens: int,
    prefix_tokens: int, interleave: int, rank: int | None,
    capacity_padded: bool, owner_spans=None, runtime: Any = None,
    query_tokens: int | None = None, block_size: int | None = None,
    generation: int = 0, phase: str = "decode",
) -> tuple[CacheRange, ...]:
    """Physical shared-prefix and private-suffix ranges for one layer.

    Always computes ``valid_bytes`` from ``capacity_padded=False`` and
    ``allocated_bytes`` from ``capacity_padded=True`` (when requested), and
    preserves explicit ``phase`` through every ``layer_cache_components`` call.
    """
    length = max(0, int(sequence_tokens))
    page_tokens = max(
        1,
        int(block_size if block_size is not None else (
            getattr(runtime, "kv_block_size", 0) or interleave
        )),
    )
    prefix = max(0, min(int(prefix_tokens), length))
    prefix = _component_prefix_tokens(layer, prefix)
    prefix = _block_tokens(prefix, page_tokens)
    unpadded = layer_cache_components(
        model, layer, parallel, length, capacity_padded=False,
        interleave=interleave, block_size=page_tokens, rank=rank,
        owner_spans=owner_spans, runtime=runtime, query_tokens=query_tokens,
        phase=phase,
    )
    if prefix <= 0:
        prefix_values = {name: 0.0 for name in unpadded}
    else:
        prefix_values = layer_cache_components(
            model, layer, parallel, prefix, capacity_padded=False,
            interleave=interleave, block_size=page_tokens, rank=rank,
            owner_spans=owner_spans, runtime=runtime, query_tokens=query_tokens,
            phase=phase,
        )
    if capacity_padded:
        allocated = layer_cache_components(
            model, layer, parallel, length, capacity_padded=True,
            interleave=interleave, block_size=page_tokens, rank=rank,
            owner_spans=owner_spans, runtime=runtime, query_tokens=query_tokens,
            phase=phase,
        )
    else:
        allocated = unpadded
    win = max(0, int(getattr(layer, "window", 0) or 0))
    gen = int(generation)
    ranges: list[CacheRange] = []
    for name, value in unpadded.items():
        valid = float(value)
        allocated_value = float(allocated.get(name, valid))
        if name in STATE_COMPONENTS or valid <= 0.0:
            component = str(name)
            ranges.append(CacheRange(
                int(layer.index), component, str(component), 0, length,
                "", gen, SHARING_PRIVATE, rank, allocated_value, valid,
                f"layer{layer.index}:{component}:private:{rank}", (),
            ))
            continue
        if win > 0 and str(name) in WINDOW_BOUND_COMPONENTS:
            win_start = max(0, length - win)
            sh_start = win_start
            sh_end = max(win_start, min(prefix, length))
            if sh_end <= sh_start:
                shared = 0.0
            elif rank is not None:
                owned_live = _rank_owned_interval(
                    layer, parallel, win_start, length,
                    interleave=interleave, rank=rank, owner_spans=owner_spans,
                )
                owned_shared = _rank_owned_interval(
                    layer, parallel, sh_start, sh_end,
                    interleave=interleave, rank=rank, owner_spans=owner_spans,
                )
                shared = (
                    valid * (float(owned_shared) / float(owned_live))
                    if owned_live > 0 and owned_shared > 0 else 0.0
                )
            else:
                live_len = max(0, length - win_start)
                live_shared = max(0, sh_end - win_start)
                shared = (
                    valid * (float(live_shared) / float(live_len))
                    if live_len > 0 and live_shared > 0 else 0.0
                )
        else:
            shared = min(valid, float(prefix_values.get(name, 0.0)))
            sh_start = 0
            sh_end = prefix
        private = max(0.0, valid - shared)
        if shared > 0.0:
            if rank is not None:
                sub_spans = _rank_owned_subintervals(
                    layer, parallel, sh_start, sh_end,
                    interleave=interleave, rank=rank, owner_spans=owner_spans,
                )
                total_sub = sum(b - a for a, b in sub_spans)
                if sub_spans and total_sub > 0:
                    for sub_a, sub_b in sub_spans:
                        sub_bytes = shared * (float(sub_b - sub_a) / float(total_sub))
                        ranges.append(CacheRange(
                            int(layer.index), str(name), str(name), sub_a, sub_b,
                            "", gen, SHARING_SHARED, rank, sub_bytes, sub_bytes,
                            f"layer{layer.index}:{name}:{sub_a}:{sub_b}:shared",
                            _page_id_span(sub_a, sub_b, page_tokens),
                        ))
                else:
                    ranges.append(CacheRange(
                        int(layer.index), str(name), str(name), sh_start, sh_end,
                        "", gen, SHARING_SHARED, rank, shared, shared,
                        f"layer{layer.index}:{name}:{sh_start}:{sh_end}:shared",
                        _page_id_span(sh_start, sh_end, page_tokens),
                    ))
            else:
                ranges.append(CacheRange(
                    int(layer.index), str(name), str(name), sh_start, sh_end,
                    "", gen, SHARING_SHARED, rank, shared, shared,
                    f"layer{layer.index}:{name}:{sh_start}:{sh_end}:shared",
                    _page_id_span(sh_start, sh_end, page_tokens),
                ))
        ranges.append(CacheRange(
            int(layer.index), str(name), str(name), sh_end, length,
            "", gen, SHARING_PRIVATE, rank, max(0.0, allocated_value - shared),
            private, f"layer{layer.index}:{name}:private:{rank}",
            _page_id_span(sh_end, length, page_tokens),
        ))
    return tuple(ranges)


def _prefixed(ranges: Sequence[CacheRange], prefix_key: str,
              generation: int) -> tuple[CacheRange, ...]:
    return tuple(
        CacheRange(
            item.layer, item.component, item.representation, item.start_token,
            item.end_token, prefix_key, generation, item.sharing, item.rank,
            item.allocated_bytes, item.valid_bytes,
            (f"prefix:{prefix_key}:g{generation}:{item.owner_key}"
             if item.sharing == SHARING_SHARED else item.owner_key),
            item.page_ids)
        for item in ranges)


def _effective_draft_sequence_tokens(
    model: Any, draft_kind: str, sequence_tokens: int, runtime: Any = None,
) -> int:
    """Clamp draft sequence length by ``effective_draft_context_cap`` at source."""
    from mindie_sim.catalog.draft import optional_draft_spec
    from mindie_sim.catalog.draft_spec import effective_draft_context_cap

    seq = max(0, int(sequence_tokens))
    if not draft_kind or getattr(model, "auxiliary", None) is None:
        return seq
    draft = optional_draft_spec(model, draft_kind)
    cap = effective_draft_context_cap(draft, runtime)
    return min(seq, cap) if cap > 0 else seq


def _draft_cache_model(model: Any, draft_kind: str) -> Any:
    """Model view carrying the draft's own resolved KV quantization contract."""
    from mindie_sim.catalog.draft import draft_quant, optional_draft_spec

    if not draft_kind or getattr(model, "auxiliary", None) is None:
        return model
    draft = optional_draft_spec(model, draft_kind)
    dq = draft_quant(model, draft)
    target_kv = getattr(getattr(model, "quant", None), "kv_dtype", "")
    differs = dq.kv_dtype != target_kv
    if not draft.independent_context and dq == getattr(model, "quant", None) and not differs:
        return model
    meta = dict(getattr(model, "metadata", ()) or ())
    if draft.independent_context or differs:
        for key in _TARGET_ONLY_KV_META_KEYS:
            meta.pop(key, None)
    return replace(model, quant=dq, kv_dtype=dq.kv_dtype, metadata=tuple(meta.items()))


def _draft_cache_parallel(
    model: Any,
    parallel: Any,
    draft_kind: str,
    *,
    interleave: int = DEFAULT_INTERLEAVE,
) -> Any:
    """Resolve draft parallel placement honoring ``draft.dcp_sharded``."""
    from mindie_sim.catalog.cache import uses_inherited_draft_cache
    from mindie_sim.catalog.cache_pools import _draft_is_dcp_sharded, _draft_row_bytes_int
    from mindie_sim.catalog.draft import optional_draft_spec

    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp <= 1 or not draft_kind or getattr(model, "auxiliary", None) is None:
        return parallel
    draft = optional_draft_spec(model, draft_kind)
    if getattr(draft, "dcp_sharded", None) is False:
        sharded = False
    elif uses_inherited_draft_cache(model, draft_kind):
        sharded = True
    else:
        row_bytes = _draft_row_bytes_int(
            model, draft, tp=max(1, int(getattr(parallel, "tp", 1) or 1)),
        )
        sharded = _draft_is_dcp_sharded(
            model, parallel, draft_kind, dcp=cp,
            draft_page=max(1, int(interleave)) * row_bytes,
        )
    if sharded:
        return parallel
    pcp = int(getattr(parallel, "pcp", 1) or 1)
    return replace(parallel, cp=1, dcp=1, cp_plan="pcp" if pcp > 1 else "none")


def draft_cache_components(
    model: Any, parallel: Any, *, draft_kind: str, sequence_tokens: int,
    runtime: Any = None, rank: int | None = None, capacity_padded: bool = True,
    owner_spans=None, query_tokens: int | None = None, phase: str = "decode",
) -> Mapping[str, float]:
    """Draft cache components honoring draft quant, context cap, DCP policy, and ``draft_kv``."""
    from mindie_sim.catalog.draft_geometry import draft_layer_plans

    components: dict[str, float] = {}
    interleave = max(
        ONE, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    eff_seq = _effective_draft_sequence_tokens(model, draft_kind, sequence_tokens, runtime)
    draft_model = _draft_cache_model(model, draft_kind)
    draft_parallel = _draft_cache_parallel(model, parallel, draft_kind, interleave=interleave)
    for plan in draft_layer_plans(model, draft_kind):
        values = layer_cache_components(
            draft_model, plan, draft_parallel, eff_seq, capacity_padded=capacity_padded,
            interleave=interleave, rank=rank, owner_spans=owner_spans,
            runtime=runtime, query_tokens=query_tokens, phase=phase)
        for name, value in values.items():
            components[f"draft.{name}"] = components.get(f"draft.{name}", ZERO_FLOAT) + float(value)
    components["draft_kv"] = float(math.fsum(
        value for name, value in components.items()
        if not name.endswith(("recurrent_state", "conv_state", "replay_records",
                              "state_transient"))))
    components["draft_state"] = float(math.fsum(
        value for name, value in components.items()
        if name.endswith(("recurrent_state", "conv_state", "replay_records",
                          "state_transient"))))
    return components


_NON_NEURAL_DRAFT_MODES = frozenset({"none", "ngram", "ngram_gpu"})


def _active_draft_kind(model: Any, runtime: Any) -> str:
    """Resolve active neural draft kind strictly from SpeculativeExecutionContract / runtime.

    Non-neural methods (``none``, ``ngram``, ``ngram_gpu``) always return ``""``
    so zero neural draft weights or KV cache are allocated.
    """
    aux = getattr(model, "auxiliary", None)
    if aux is None or runtime is None:
        return ""
    meta = dict(getattr(runtime, "metadata", ()) or ())
    contract = getattr(runtime, "speculative_contract", None) or meta.get("speculative_contract")
    if contract is not None:
        method = str(
            getattr(contract, "spec_method", None)
            if not isinstance(contract, Mapping)
            else contract.get("spec_method", "")
        ).strip().lower()
        k_val = int(
            getattr(contract, "num_speculative_tokens", 0)
            if not isinstance(contract, Mapping)
            else (contract.get("num_speculative_tokens", 0) or 0)
        )
        if method in _NON_NEURAL_DRAFT_MODES or k_val <= 0:
            return ""
        return "dspark" if method in {"dspark", "dflash"} else method
    spine = meta.get("single_spine")
    spec_method = str(meta.get("spec_method") or "").strip().lower()
    if not spec_method and isinstance(spine, Mapping):
        scen = spine.get("scenario_contract")
        if isinstance(scen, Mapping):
            pol = scen.get("speculative_policy")
            if isinstance(pol, Mapping):
                spec_method = str(pol.get("spec_method") or "").strip().lower()
    if spec_method in _NON_NEURAL_DRAFT_MODES:
        return ""
    declared_mode = str(meta.get("draft_mode") or "").strip().lower()
    if declared_mode in _NON_NEURAL_DRAFT_MODES:
        return ""
    if getattr(runtime, "dspark", None) is not None and (aux.dspark is not None or aux.mtp is not None):
        return "dspark"
    if declared_mode:
        if declared_mode in {"dspark", "dflash", "verify", "draft_verify"}:
            return "dspark" if (aux.dspark is not None or (aux.mtp is not None and aux.mtp.kind in {"mtp_dspark", "dspark"})) else ""
        return declared_mode if aux.mtp is not None else ""
    if spec_method:
        if spec_method in {"dspark", "dflash"}:
            return "dspark" if (aux.dspark is not None or (aux.mtp is not None and aux.mtp.kind in {"mtp_dspark", "dspark"})) else ""
        return spec_method if aux.mtp is not None else ""
    return ""


def _view_generation(view: Any) -> int:
    raw = getattr(view, "prefix_generation", None)
    if raw is None:
        raw = getattr(view, "generation", 0)
    return int(raw or 0)


def _resolve_group_stage(
    group: Mapping[str, Any],
    layer_stages: Mapping[int, int],
    stages: int,
) -> int:
    layers = group.get("layers") or ()
    if layers:
        first = int(layers[0])
        if first in layer_stages:
            return int(layer_stages[first])
    members = group.get("members") or ()
    if members:
        first = int(members[0]["layer"])
        if first in layer_stages:
            return int(layer_stages[first])
    return max(0, stages - 1)


def _paired_group_covered_members(groups: Sequence[Mapping[str, Any]]) -> tuple[set[int], set[tuple[int, str]]]:
    full_layers: set[int] = set()
    member_pairs: set[tuple[int, str]] = set()
    for group in groups:
        if group.get("auxiliary"):
            continue
        if "members" in group and group["members"]:
            for item in group["members"]:
                member_pairs.add((int(item["layer"]), str(item["component"])))
        else:
            for layer_id in group.get("layers") or ():
                full_layers.add(int(layer_id))
    return full_layers, member_pairs


def _paired_sfa_batch_footprint(
    case: Any,
    *,
    requests: Sequence[Any],
    phase: str,
    padded: bool,
    layer_stages: Mapping[int, int],
    stages: int,
    shared_mode: bool,
) -> BatchCacheFootprint:
    from mindie_sim.catalog.paired_sfa import page_groups
    from mindie_sim.schema.runtime_features import canonical_prefix_cache_key

    model, runtime = case.model, case.runtime
    parallel = runtime.parallel
    query_tokens = max(
        (max(1, int(getattr(v, "q_len", 1) or 1)) for v in requests),
        default=1,
    )
    groups = page_groups(case, layer_stages, query_tokens)
    covered_layers, covered_members = _paired_group_covered_members(groups)
    shared_once = [0.0] * stages
    private = [0.0] * stages
    state = [0.0] * stages
    draft = [0.0] * stages
    boundary = [0.0] * stages
    shared_page_sets: dict[tuple[str, int, int], set[int]] = {}
    interleave = cp_interleave(runtime)
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    active_draft = _active_draft_kind(model, runtime)

    for view in requests:
        length = max(0, int(getattr(view, "kv_len", 0) or 0))
        q_tok = max(1, int(getattr(view, "q_len", 1) or 1))
        declared = max(0, min(int(getattr(view, "prefix_tokens", 0) or 0), length))
        generation = _view_generation(view)
        key = (
            canonical_prefix_cache_key("shared", getattr(view, "prefix_cache_key", ""))
            if shared_mode else f"private:{getattr(view, 'request_id', '')}"
        )
        for group_idx, group in enumerate(groups):
            stage = _resolve_group_stage(group, layer_stages, stages)
            tpp = max(1, int(group["tokens_per_page"]))
            page_bytes = (
                float(group["bytes_per_group_page"])
                + float(group.get("metadata_bytes_per_page", 0) or 0)
            )
            is_aux = bool(group.get("auxiliary", False))
            fixed = int(group.get("fixed_pages", 0) or 0)
            rec_pages = int(group.get("recycling_pages", 0) or 0)
            if fixed > 0:
                fixed_bytes = float(fixed + rec_pages) * page_bytes
                private[stage] += fixed_bytes
                state[stage] += fixed_bytes
                continue
            cap = group.get("resident_tokens")
            base_len = (
                _effective_draft_sequence_tokens(model, active_draft, length, runtime)
                if is_aux else length
            )
            eff_len = min(base_len, int(cap)) if (cap is not None and int(cap) > 0) else base_len
            eff_declared = 0 if is_aux else min(declared, eff_len)
            prefix_pages = eff_declared // tpp if (shared_mode and not is_aux) else 0
            history_pages = _ceil_div(eff_len, tpp) if eff_len > 0 else 0
            if prefix_pages > 0:
                page_key = (key, generation, group_idx)
                seen = shared_page_sets.setdefault(page_key, set())
                new_ids = set(range(prefix_pages)) - seen
                if new_ids:
                    shared_once[stage] += float(len(new_ids)) * page_bytes
                    seen.update(new_ids)
            priv_history_pages = max(0, history_pages - prefix_pages)
            if is_aux:
                draft[stage] += float(priv_history_pages) * page_bytes
            else:
                private[stage] += float(priv_history_pages + rec_pages) * page_bytes
                state[stage] += float(rec_pages) * page_bytes

        # Account for separately allocated components excluded from paired pools
        # (for example RecoverSSM replay_records / state_transient).
        for plan in model.layer_plans:
            if plan.index in covered_layers:
                continue
            comps = layer_cache_components(
                model, plan, parallel, length, capacity_padded=bool(padded),
                interleave=interleave, block_size=block_size, runtime=runtime,
                query_tokens=q_tok, phase=phase,
            )
            stage = layer_stages[int(plan.index)]
            for comp_name, comp_bytes in comps.items():
                if (int(plan.index), str(comp_name)) in covered_members:
                    continue
                val = float(comp_bytes)
                if val <= 0.0:
                    continue
                private[stage] += val
                if comp_name in STATE_COMPONENTS:
                    state[stage] += val

    totals = tuple(
        shared_once[s] + private[s] + draft[s] + boundary[s] for s in range(stages)
    )
    return BatchCacheFootprint(
        tuple(shared_once), tuple(private), tuple(state),
        tuple(draft), tuple(boundary), totals,
        requests=len(requests), stages=stages,
    )


def _union_elementary_spans(spans: Sequence[tuple[int, int, float]]) -> float:
    """Exact piecewise-constant union integral without inflating non-overlapping tails."""
    if not spans:
        return 0.0
    points = sorted({pt for lo, hi, _ in spans for pt in (lo, hi) if hi > lo})
    if len(points) < 2:
        return 0.0
    total = 0.0
    for idx in range(len(points) - 1):
        a, b = points[idx], points[idx + 1]
        if b <= a:
            continue
        active = [rate for lo, hi, rate in spans if lo <= a and hi >= b]
        if active:
            total += float(b - a) * max(active)
    return total


def _stage_shareable_prefix_range(
    contract: Any,
    stage: int,
    *,
    length: int,
    declared_prefix: int,
    eff_block: int,
) -> range:
    """Shareable physical block range on one stage (respects SWA window expiry)."""
    from mindie_sim.catalog.cache_pools import REGIME_ALIASED_WINDOW_COMPRESSED

    if declared_prefix <= 0 or length <= 0 or eff_block <= 0:
        return range(0)
    if (
        contract.pool_regime == REGIME_ALIASED_WINDOW_COMPRESSED
        and stage < len(contract.alias_backing_pitches_by_stage)
        and not contract.alias_backing_pitches_by_stage[stage]
    ):
        win = int(contract.swa_max_window or 0)
        win_start = max(0, length - win) if win > 0 else 0
        shared_hi = min(declared_prefix, length)
        if shared_hi <= win_start:
            return range(0)
        return range(_ceil_div(win_start, eff_block), shared_hi // eff_block)
    max_shareable = min(declared_prefix // eff_block, _ceil_div(length, eff_block))
    return range(max(0, max_shareable))


def _record_hybrid_prefix_blocks(
    req_hybrid: Any,
    *,
    stages: int,
    shared_mode: bool,
    length: int,
    declared_prefix: int,
    key: str,
    generation: int,
    private_pool_bytes: list[int],
    shared_block_units: dict[tuple[str, int, int, int], int],
) -> bool:
    """Accumulate order-invariant private & shared-prefix pool block bytes for one request."""
    contract = req_hybrid.contract
    eff_block = contract.normalized_block_size
    any_shared = False
    for st in range(stages):
        unit_b = int(req_hybrid.pool_unit_bytes_by_stage[st])
        blk_range = (
            _stage_shareable_prefix_range(
                contract, st, length=length, declared_prefix=declared_prefix, eff_block=eff_block,
            )
            if (shared_mode and declared_prefix >= eff_block)
            else range(0)
        )
        if blk_range:
            any_shared = True
            for blk_id in blk_range:
                bkey = (key, generation, st, int(blk_id))
                if unit_b > shared_block_units.get(bkey, 0):
                    shared_block_units[bkey] = unit_b
        private_pool_bytes[st] += max(
            0, int(req_hybrid.total_pool_bytes_by_stage[st]) - len(blk_range) * unit_b,
        )
    return any_shared


def _hybrid_batch_footprint(
    case: Any,
    *,
    requests: Sequence[Any],
    phase: str,
    layer_stages: Mapping[int, int],
    stages: int,
    shared_mode: bool,
    block_size: int,
    interleave: int,
    draft_kind: str,
) -> BatchCacheFootprint:
    """Physical batch footprint for G3 hybrid block-pool regimes with 0 B Memory parity."""
    from mindie_sim.catalog.cache_pools import evaluate_hybrid_pool_receipt

    model, parallel, runtime = case.model, case.runtime.parallel, case.runtime
    is_prefill = str(phase or "decode").strip().lower() == "prefill"
    shared: dict[tuple[int, str, str, int, int], list[tuple[int, int, float]]] = {}
    shared_subsumed: dict[tuple[int, str, str, int, int], list[tuple[int, int, float]]] = {}
    private = [0.0] * stages
    state = [0.0] * stages
    draft = [0.0] * stages
    boundary = [0.0] * stages
    subsumed_priv = [0.0] * stages
    private_pool_bytes = [0] * stages
    shared_block_units: dict[tuple[str, int, int, int], int] = {}
    has_shared_blocks = False
    per_req_shapes: list[tuple[int, int]] = []
    cp_deg = int(getattr(parallel, "cp", 1) or 1)

    for view in requests:
        length = max(0, int(getattr(view, "kv_len", 0) or 0))
        q_tok = 1 if is_prefill else max(1, int(getattr(view, "q_len", 1) or 1))
        declared_prefix = max(0, min(int(getattr(view, "prefix_tokens", 0) or 0), length))
        req_hybrid = evaluate_hybrid_pool_receipt(
            model, parallel, length, num_requests=1, block_size=block_size,
            runtime=runtime, draft_kind=draft_kind, query_tokens=q_tok, phase=phase,
        )
        contract = req_hybrid.contract
        eff_block = contract.normalized_block_size
        eff_interleave = eff_block if cp_deg <= 1 else interleave
        sub_layers = set(contract.subsumed_layers)
        sub_comps = set(contract.subsumed_components)
        generation = _view_generation(view)
        if shared_mode:
            from mindie_sim.schema.runtime_features import canonical_prefix_cache_key
            key = canonical_prefix_cache_key("shared", getattr(view, "prefix_cache_key", ""))
        else:
            key = f"private:{getattr(view, 'request_id', '')}"
        if _record_hybrid_prefix_blocks(
            req_hybrid,
            stages=stages, shared_mode=shared_mode, length=length,
            declared_prefix=declared_prefix, key=key, generation=generation,
            private_pool_bytes=private_pool_bytes, shared_block_units=shared_block_units,
        ):
            has_shared_blocks = True
        per_req_shapes.append((length, q_tok))

        for layer in model.layer_plans:
            ranges = cache_component_ranges(
                model, layer, parallel, sequence_tokens=length,
                prefix_tokens=declared_prefix, interleave=eff_interleave,
                block_size=eff_block, rank=None, capacity_padded=True,
                runtime=runtime, query_tokens=q_tok, generation=generation,
                phase=phase,
            )
            stage = layer_stages[int(layer.index)]
            for item in _prefixed(ranges, key, generation):
                is_sub = int(item.layer) in sub_layers and str(item.component) in sub_comps
                if item.sharing == SHARING_SHARED and item.valid_bytes > 0.0:
                    span_len = max(1, item.end_token - item.start_token)
                    rate = float(item.valid_bytes) / float(span_len)
                    grp_key = (int(item.layer), item.component, key, generation, stage)
                    shared.setdefault(grp_key, []).append((int(item.start_token), int(item.end_token), rate))
                    if is_sub:
                        shared_subsumed.setdefault(grp_key, []).append((int(item.start_token), int(item.end_token), rate))
                    continue
                alloc_val = float(item.allocated_bytes)
                valid_val = float(item.valid_bytes)
                private[stage] += valid_val
                boundary[stage] += max(0.0, alloc_val - valid_val)
                if item.component in STATE_COMPONENTS:
                    state[stage] += valid_val
                if is_sub:
                    subsumed_priv[stage] += alloc_val
        if draft_kind:
            draft_seq = _effective_draft_sequence_tokens(model, draft_kind, length, runtime)
            values = draft_cache_components(
                model, parallel, draft_kind=draft_kind, sequence_tokens=draft_seq,
                runtime=runtime, capacity_padded=True, query_tokens=q_tok, phase=phase,
            )
            d_kv = float(values.get("draft_kv", 0.0))
            d_st = float(values.get("draft_state", 0.0))
            draft[-1] += d_kv
            state[-1] += d_st
            private[-1] += d_st
            if contract.subsumed_draft_kv:
                subsumed_priv[-1] += d_kv

    shared_once = [0.0] * stages
    for (_lyr, _comp, _key, _gen, stage), spans in shared.items():
        shared_once[stage] += _union_elementary_spans(spans)
    shared_sub_once = [0.0] * stages
    for (_lyr, _comp, _key, _gen, stage), spans in shared_subsumed.items():
        shared_sub_once[stage] += _union_elementary_spans(spans)

    uniform_non_shared = (
        bool(per_req_shapes)
        and not has_shared_blocks
        and len(set(per_req_shapes)) == 1
    )
    if uniform_non_shared:
        ln0, qt0 = per_req_shapes[0]
        batch_hybrid = evaluate_hybrid_pool_receipt(
            model, parallel, ln0, num_requests=len(requests), block_size=block_size,
            runtime=runtime, draft_kind=draft_kind, query_tokens=qt0, phase=phase,
        )
        for st in range(stages):
            boundary[st] += float(batch_hybrid.stride_reserve_by_stage[st])
    else:
        shared_pool_bytes = [0] * stages
        for (_k, _g, st, _blk), unit_b in shared_block_units.items():
            shared_pool_bytes[st] += unit_b
        for st in range(stages):
            pool_target = float(private_pool_bytes[st] + shared_pool_bytes[st])
            boundary[st] += max(0.0, pool_target - (shared_sub_once[st] + subsumed_priv[st]))

    total = [
        shared_once[st] + private[st] + draft[st] + boundary[st]
        for st in range(stages)
    ]
    return BatchCacheFootprint(
        tuple(float(value) for value in shared_once),
        tuple(float(value) for value in private),
        tuple(float(value) for value in state),
        tuple(float(value) for value in draft),
        tuple(float(value) for value in boundary),
        tuple(float(value) for value in total),
        requests=len(requests), stages=stages,
    )


def _explicit_group_stage_costs(
    group: Any, stage_map: tuple[int, ...],
) -> list[tuple[int, float]]:
    meta = float(group.metadata_bytes_per_page)
    if group.bytes_per_group_page:
        return [(int(stage_map[group.layers[0]]), float(group.bytes_per_group_page) + meta)]
    per_layer = float(group.bytes_per_layer_page) + meta
    return [(int(stage_map[layer]), per_layer) for layer in group.layers]


def _explicit_group_shareable_pages(
    group: Any, *, length: int, declared: int, total_pages: int, shared_mode: bool,
) -> range:
    if not shared_mode or group.allocation != "live":
        return range(0)
    tpp = max(1, int(group.tokens_per_page))
    if group.resident_tokens is not None and length > group.resident_tokens:
        win_start = length - int(group.resident_tokens)
        sh_lo = _ceil_div(win_start, tpp)
        sh_hi = min(declared, length) // tpp
        return range(sh_lo, sh_hi) if sh_hi > sh_lo else range(0)
    eff_len = min(length, group.resident_tokens) if group.resident_tokens is not None else length
    eff_decl = min(declared, eff_len)
    if eff_decl <= 0:
        return range(0)
    shareable_cap = max(0, total_pages - group.recycling_pages)
    return range(min(shareable_cap, eff_decl // tpp))


def _explicit_page_group_batch_footprint(
    case: Any,
    *,
    requests: Sequence[Any],
    phase: str,
    padded: bool,
    stages: int,
    shared_mode: bool,
) -> BatchCacheFootprint:
    """Physical batch footprint for explicit ``runtime.memory['cache_page_groups']``."""
    from mindie_sim.catalog.cache_pages import CachePages
    from mindie_sim.schema.runtime_features import canonical_prefix_cache_key

    model, runtime = case.model, case.runtime
    parallel = runtime.parallel
    raw_groups = tuple(dict(getattr(runtime, "memory", ()) or ()).get("cache_page_groups") or ())
    stage_map = layer_stage_map(model, parallel, runtime)
    max_q = max((max(1, int(getattr(v, "q_len", 1) or 1)) for v in requests), default=1)
    comp_names = {
        plan.index: set(layer_cache_components(
            model, plan, parallel, runtime.max_model_len,
            capacity_padded=False, runtime=runtime, query_tokens=max_q, phase=phase,
        ))
        for plan in model.layer_plans
    }
    pages_obj = CachePages(raw_groups, comp_names, stage_map)
    shared_once = [0.0] * stages
    private = [0.0] * stages
    state = [0.0] * stages
    draft = [0.0] * stages
    boundary = [0.0] * stages
    seen_shared_pages: dict[tuple[str, int, int], set[int]] = {}
    shared_spans: dict[tuple[int, str, str, int, int], list[tuple[int, int, float]]] = {}
    interleave = cp_interleave(runtime)
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    is_prefill = str(phase or "decode").strip().lower() == "prefill"
    has_aux = any(group.auxiliary for group in pages_obj.groups)
    active_draft = _active_draft_kind(model, runtime)
    draft_kind = "" if has_aux else active_draft

    for view in requests:
        length = max(0, int(getattr(view, "kv_len", 0) or 0))
        q_tok = 1 if is_prefill else max(1, int(getattr(view, "q_len", 1) or 1))
        declared = max(0, min(int(getattr(view, "prefix_tokens", 0) or 0), length))
        generation = _view_generation(view)
        p_key = (
            canonical_prefix_cache_key("shared", getattr(view, "prefix_cache_key", ""))
            if shared_mode else f"private:{getattr(view, 'request_id', '')}"
        )
        for group_idx, group in enumerate(pages_obj.groups):
            eff_len = (
                _effective_draft_sequence_tokens(model, active_draft, length, runtime)
                if group.auxiliary else length
            )
            total_pages = group.pages(eff_len, runtime.max_model_len)
            stage_costs = _explicit_group_stage_costs(group, stage_map)
            if group.fixed_pages:
                for st, cost in stage_costs:
                    val = float(total_pages) * cost
                    if group.auxiliary:
                        draft[st] += val
                    else:
                        private[st] += val
                        state[st] += val
                continue
            shareable_ids = (
                range(0) if group.auxiliary
                else _explicit_group_shareable_pages(
                    group, length=eff_len, declared=declared,
                    total_pages=total_pages, shared_mode=shared_mode,
                )
            )
            if shareable_ids:
                seen = seen_shared_pages.setdefault((p_key, generation, group_idx), set())
                new_ids = set(shareable_ids) - seen
                if new_ids:
                    for st, cost in stage_costs:
                        shared_once[st] += float(len(new_ids)) * cost
                    seen.update(new_ids)
            priv_pages = max(0, total_pages - len(shareable_ids))
            for st, cost in stage_costs:
                if group.auxiliary:
                    draft[st] += float(priv_pages) * cost
                else:
                    private[st] += float(priv_pages) * cost
                    if group.recycling_pages > 0:
                        state[st] += float(min(priv_pages, group.recycling_pages)) * cost
        for plan in model.layer_plans:
            layer_cov = pages_obj.covered.get(plan.index, ())
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
                if item.sharing == SHARING_SHARED and item.valid_bytes > 0.0:
                    span_len = max(1, item.end_token - item.start_token)
                    rate = float(item.valid_bytes) / float(span_len)
                    s_key = (stage, item.component, p_key, generation, plan.index)
                    shared_spans.setdefault(s_key, []).append(
                        (int(item.start_token), int(item.end_token), rate)
                    )
                    continue
                private[stage] += float(item.valid_bytes)
                boundary[stage] += max(0.0, float(item.allocated_bytes) - float(item.valid_bytes))
                if item.component in STATE_COMPONENTS:
                    state[stage] += float(item.valid_bytes)
        if draft_kind:
            draft_seq = _effective_draft_sequence_tokens(model, draft_kind, length, runtime)
            values = draft_cache_components(
                model, parallel, draft_kind=draft_kind, sequence_tokens=draft_seq,
                runtime=runtime, capacity_padded=bool(padded), query_tokens=q_tok,
                phase=phase,
            )
            draft[-1] += float(values.get("draft_kv", 0.0))
            d_st = float(values.get("draft_state", 0.0))
            state[-1] += d_st
            private[-1] += d_st

    for (stage, _comp, _p_key, _gen, _layer), spans in shared_spans.items():
        shared_once[stage] += _union_elementary_spans(spans)
    totals = tuple(
        shared_once[s] + private[s] + draft[s] + boundary[s] for s in range(stages)
    )
    return BatchCacheFootprint(
        tuple(float(v) for v in shared_once),
        tuple(float(v) for v in private),
        tuple(float(v) for v in state),
        tuple(float(v) for v in draft),
        tuple(float(v) for v in boundary),
        totals,
        requests=len(requests), stages=stages,
    )


def batch_cache_footprint(
    case: Any, *, requests: Sequence[Any], replica: int, phase: str,
    padded: bool,
) -> BatchCacheFootprint:
    """Physical batch footprint for one DP replica and phase."""
    del replica
    model, parallel = case.model, case.runtime.parallel
    stages = max(1, int(parallel.pp))
    runtime = case.runtime
    layer_stages = _layer_stages(model, parallel, runtime)
    shared_mode = getattr(runtime, "prefix_cache_mode", "private") == "shared"
    if dict(model.metadata).get("cache_allocator") == "paired_sfa":
        return _paired_sfa_batch_footprint(
            case, requests=requests, phase=phase, padded=padded,
            layer_stages=layer_stages, stages=stages, shared_mode=shared_mode,
        )
    if padded and dict(getattr(runtime, "memory", ()) or ()).get("cache_page_groups"):
        return _explicit_page_group_batch_footprint(
            case, requests=requests, phase=phase, padded=padded,
            stages=stages, shared_mode=shared_mode,
        )
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    interleave = cp_interleave(runtime)
    draft_kind = _active_draft_kind(model, runtime)
    if padded and requests:
        from mindie_sim.catalog.cache_pools import (
            REGIME_UNIFORM_INDEPENDENT,
            resolve_pool_allocator_contract,
        )
        eff_draft = draft_kind if str(phase) == "decode" else ""
        contract = resolve_pool_allocator_contract(
            model, parallel, block_size=block_size, runtime=runtime,
            draft_kind=eff_draft, phase=phase,
        )
        if contract.pool_regime != REGIME_UNIFORM_INDEPENDENT:
            return _hybrid_batch_footprint(
                case, requests=requests, phase=phase, layer_stages=layer_stages,
                stages=stages, shared_mode=shared_mode, block_size=block_size,
                interleave=interleave, draft_kind=eff_draft,
            )
    shared: dict[tuple[int, str, str, int, int], list[tuple[int, int, float]]] = {}
    private = [0.0] * stages
    state = [0.0] * stages
    draft = [0.0] * stages
    boundary = [0.0] * stages
    is_prefill = str(phase or "decode").strip().lower() == "prefill"
    for view in requests:
        length = int(getattr(view, "kv_len", 0) or 0)
        q_tok = 1 if is_prefill else max(1, int(getattr(view, "q_len", 1) or 1))
        declared_prefix = max(0, min(int(getattr(view, "prefix_tokens", 0) or 0), length))
        if shared_mode:
            from mindie_sim.schema.runtime_features import canonical_prefix_cache_key
            key = canonical_prefix_cache_key(
                "shared", getattr(view, "prefix_cache_key", ""))
        else:
            key = f"private:{getattr(view, 'request_id', '')}"
        generation = _view_generation(view)
        for layer in model.layer_plans:
            ranges = cache_component_ranges(
                model, layer, parallel, sequence_tokens=length,
                prefix_tokens=declared_prefix, interleave=interleave,
                block_size=block_size, rank=None, capacity_padded=bool(padded),
                runtime=runtime, query_tokens=q_tok, generation=generation,
                phase=phase,
            )
            stage = layer_stages[int(layer.index)]
            for item in _prefixed(ranges, key, generation):
                if item.sharing == SHARING_SHARED and item.valid_bytes > 0.0:
                    span_len = max(1, item.end_token - item.start_token)
                    rate = float(item.valid_bytes) / float(span_len)
                    grp_key = (int(item.layer), item.component, key, generation, stage)
                    shared.setdefault(grp_key, []).append(
                        (int(item.start_token), int(item.end_token), rate))
                    continue
                private[stage] += float(item.valid_bytes)
                boundary[stage] += max(
                    0.0, float(item.allocated_bytes) - float(item.valid_bytes))
                if item.component in STATE_COMPONENTS:
                    state[stage] += float(item.valid_bytes)
        if draft_kind and (str(phase) == "decode" or bool(getattr(view, "draft_prepare", False))):
            draft_seq = _effective_draft_sequence_tokens(model, draft_kind, length, runtime)
            values = draft_cache_components(
                model, parallel, draft_kind=draft_kind, sequence_tokens=draft_seq,
                runtime=runtime, capacity_padded=bool(padded), query_tokens=q_tok,
                phase=phase)
            draft[-1] += float(values.get("draft_kv", 0.0))
            state[-1] += float(values.get("draft_state", 0.0))
            private[-1] += float(values.get("draft_state", 0.0))
    shared_once = [0.0] * stages
    for (_lyr, _comp, _key, _gen, stage), spans in shared.items():
        shared_once[stage] += _union_elementary_spans(spans)
    total = [
        shared_once[stage] + private[stage] + draft[stage] + boundary[stage]
        for stage in range(stages)
    ]
    return BatchCacheFootprint(
        tuple(float(value) for value in shared_once),
        tuple(float(value) for value in private),
        tuple(float(value) for value in state),
        tuple(float(value) for value in draft),
        tuple(float(value) for value in boundary),
        tuple(float(value) for value in total),
        requests=len(requests), stages=stages)
