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
from dataclasses import dataclass, field
import math
from typing import Any

from mindie_sim.catalog.cache import (
    _LINEAR_ATTENTION_KINDS, layer_cache_components, layer_stage_map,
)
from mindie_sim.catalog.cache_pools import _ceil_div
from mindie_sim.catalog.context_parallel import cp_interleave
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
})
WINDOW_BOUND_COMPONENTS = frozenset({"window", "gqa_kv", "mla_kv"})


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


def cache_component_ranges(
    model: Any, layer: Any, parallel: Any, *, sequence_tokens: int,
    prefix_tokens: int, interleave: int, rank: int | None,
    capacity_padded: bool, owner_spans=None, runtime: Any = None,
    query_tokens: int | None = None, block_size: int | None = None,
    generation: int = 0,
) -> tuple[CacheRange, ...]:
    """Physical shared-prefix and private-suffix ranges for one layer.

    Always computes ``valid_bytes`` from ``capacity_padded=False`` and
    ``allocated_bytes`` from ``capacity_padded=True`` (when requested) so
    padded page tails are never misclassified as valid content.
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
    )
    if prefix <= 0:
        prefix_values = {name: 0.0 for name in unpadded}
    else:
        prefix_values = layer_cache_components(
            model, layer, parallel, prefix, capacity_padded=False,
            interleave=interleave, block_size=page_tokens, rank=rank,
            owner_spans=owner_spans, runtime=runtime, query_tokens=query_tokens,
        )
    if capacity_padded:
        allocated = layer_cache_components(
            model, layer, parallel, length, capacity_padded=True,
            interleave=interleave, block_size=page_tokens, rank=rank,
            owner_spans=owner_spans, runtime=runtime, query_tokens=query_tokens,
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
            live_len = max(0, length - win_start)
            live_shared = max(0, min(prefix, length) - win_start)
            shared = (
                valid * (float(live_shared) / float(live_len))
                if live_len > 0 and live_shared > 0 else 0.0
            )
            sh_start = win_start
            sh_end = max(win_start, min(prefix, length))
        else:
            shared = min(valid, float(prefix_values.get(name, 0.0)))
            sh_start = 0
            sh_end = prefix
        private = max(0.0, valid - shared)
        if shared > 0.0:
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


def draft_cache_components(
    model: Any, parallel: Any, *, draft_kind: str, sequence_tokens: int,
    runtime: Any = None, rank: int | None = None, capacity_padded: bool = True,
    owner_spans=None, query_tokens: int | None = None, phase: str = "decode",
) -> Mapping[str, float]:
    """Inherited draft cache components, keeping the public ``draft_kv`` key."""
    from mindie_sim.catalog.draft_geometry import draft_layer_plans

    components: dict[str, float] = {}
    interleave = max(
        ONE, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    for plan in draft_layer_plans(model, draft_kind):
        values = layer_cache_components(
            model, plan, parallel, sequence_tokens, capacity_padded=capacity_padded,
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


def _active_draft_kind(model: Any, runtime: Any) -> str:
    if getattr(model, "auxiliary", None) is None or runtime is None:
        return ""
    meta = dict(getattr(runtime, "metadata", ()) or ())
    declared = str(meta.get("draft_mode") or "")
    if declared:
        return declared
    if getattr(runtime, "dspark", None):
        return "dspark"
    if (float(getattr(runtime, "mtp_acceptance_length", ONE) or ONE) > ONE
            or any(val > ONE for _, val in getattr(runtime, "mtp_acceptance_by_output_tokens", ()))
            or meta.get("speculative")):
        return "mtp"
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
            page_bytes = float(group["bytes_per_group_page"])
            is_aux = bool(group.get("auxiliary", False))
            fixed = int(group.get("fixed_pages", 0) or 0)
            rec_pages = int(group.get("recycling_pages", 0) or 0)
            if fixed > 0:
                fixed_bytes = float(fixed + rec_pages) * page_bytes
                private[stage] += fixed_bytes
                state[stage] += fixed_bytes
                continue
            cap = group.get("resident_tokens")
            eff_len = min(length, int(cap)) if (cap is not None and int(cap) > 0) else length
            eff_declared = min(declared, eff_len)
            prefix_pages = eff_declared // tpp if shared_mode else 0
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
    shared: dict[tuple[int, str, str, int, int], list[tuple[int, int, float]]] = {}
    private = [0.0] * stages
    state = [0.0] * stages
    draft = [0.0] * stages
    boundary = [0.0] * stages
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    interleave = cp_interleave(runtime)
    draft_kind = _active_draft_kind(model, runtime)
    for view in requests:
        length = int(getattr(view, "kv_len", 0) or 0)
        q_tok = max(1, int(getattr(view, "q_len", 1) or 1))
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
        if str(phase) == "decode" and draft_kind:
            values = draft_cache_components(
                model, parallel, draft_kind=draft_kind, sequence_tokens=length,
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


def physical_union(
    values: Sequence[float], owner_keys: Sequence[str] | None = None,
) -> float:
    """Reference-counted interval union helper for callers with known keys."""
    if owner_keys is not None and len(owner_keys) != len(values):
        raise ModelingError("cache_union_owner_keys_mismatch")
    if owner_keys is None:
        return float(math.fsum(float(value) for value in values))
    union: dict[str, float] = {}
    for key, value in zip(owner_keys, values):
        union[str(key)] = max(union.get(str(key), ZERO), float(value))
    return float(math.fsum(union.values()))
