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
    _LINEAR_ATTENTION_KINDS, layer_cache_components,
)
from mindie_sim.schema.common import ModelingError


ZERO = 0
ONE = 1

DEFAULT_INTERLEAVE = 128
FIRST = 0
SECOND = 1
FIRST = 0
SECOND = 1
SHARING_SHARED = "shared"
SHARING_PRIVATE = "private"
STATE_COMPONENTS = frozenset({
    "recurrent_state", "conv_state", "replay_records", "state_transient",
})


@dataclass(frozen=True)
class CacheRange:
    """One physical cache interval with explicit sharing ownership."""

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


def _layer_stages(model: Any, parallel: Any) -> dict[int, int]:
    pp = max(ONE, int(parallel.pp))
    layers = max(ONE, int(getattr(model, "layers", ONE)))
    result = {}
    for plan in model.layer_plans:
        result[int(plan.index)] = min(pp - ONE, int(plan.index) * pp // layers)
    return result


def _block_tokens(tokens: int, interleave: int) -> int:
    if interleave <= ONE:
        return int(tokens)
    return (int(tokens) // int(interleave)) * int(interleave)


def _component_prefix_tokens(plan: Any, tokens: int) -> int:
    ratio = max(ZERO, int(getattr(plan, "compress_ratio", ZERO) or ZERO))
    if ratio > ONE:
        return (int(tokens) // ratio) * ratio
    return int(tokens)


def cache_component_ranges(
    model: Any, layer: Any, parallel: Any, *, sequence_tokens: int,
    prefix_tokens: int, interleave: int, rank: int | None,
    capacity_padded: bool, owner_spans=None,
) -> tuple[CacheRange, ...]:
    """Physical shared-prefix and private-suffix ranges for one layer."""
    length = max(ZERO, int(sequence_tokens))
    prefix = max(ZERO, min(int(prefix_tokens), length))
    prefix = _component_prefix_tokens(layer, prefix)
    prefix = _block_tokens(prefix, int(interleave))
    full = layer_cache_components(
        model, layer, parallel, length, capacity_padded=capacity_padded,
        interleave=interleave, rank=rank, owner_spans=owner_spans)
    if prefix <= ZERO:
        prefix_values = {name: ZERO for name in full}
    else:
        prefix_values = layer_cache_components(
            model, layer, parallel, prefix, capacity_padded=False,
            interleave=interleave, rank=rank, owner_spans=owner_spans)
    if capacity_padded and int(interleave) > ONE:
        allocated = layer_cache_components(
            model, layer, parallel,
            (length + int(interleave) - ONE) // int(interleave) * int(interleave),
            capacity_padded=True, interleave=interleave, rank=rank,
            owner_spans=owner_spans)
    else:
        allocated = full
    ranges: list[CacheRange] = []
    for name, value in full.items():
        valid = float(value)
        allocated_value = float(allocated.get(name, valid))
        if name in STATE_COMPONENTS or valid <= ZERO:
            component = str(name)
            ranges.append(CacheRange(
                int(layer.index), component, str(component), ZERO, length,
                "", ZERO, SHARING_PRIVATE, rank, allocated_value, valid,
                f"layer{layer.index}:{component}:private:{rank}"))
            continue
        shared = min(valid, float(prefix_values.get(name, ZERO)))
        private = max(ZERO, valid - shared)
        if shared > ZERO:
            ranges.append(CacheRange(
                int(layer.index), str(name), str(name), ZERO, prefix,
                "", ZERO, SHARING_SHARED, rank, shared, shared,
                f"layer{layer.index}:{name}:shared"))
        ranges.append(CacheRange(
            int(layer.index), str(name), str(name), prefix, length,
            "", ZERO, SHARING_PRIVATE, rank, max(ZERO, allocated_value - shared),
            private, f"layer{layer.index}:{name}:private:{rank}"))
    return tuple(ranges)


def _prefixed(ranges: Sequence[CacheRange], prefix_key: str,
              generation: int) -> tuple[CacheRange, ...]:
    return tuple(
        CacheRange(
            item.layer, item.component, item.representation, item.start_token,
            item.end_token, prefix_key, generation, item.sharing, item.rank,
            item.allocated_bytes, item.valid_bytes,
            (f"prefix:{prefix_key}:g{generation}:{item.owner_key}"
             if item.sharing == SHARING_SHARED else item.owner_key))
        for item in ranges)


def draft_cache_components(
    model: Any, parallel: Any, *, draft_kind: str, sequence_tokens: int,
    runtime: Any = None, rank: int | None = None,
) -> Mapping[str, float]:
    """Inherited draft cache components, keeping the public ``draft_kv`` key."""
    from mindie_sim.catalog.draft_geometry import draft_layer_plans

    components: dict[str, float] = {}
    for plan in draft_layer_plans(model, draft_kind):
        values = layer_cache_components(
            model, plan, parallel, sequence_tokens, capacity_padded=True,
            interleave=max(ONE, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE)),
            rank=rank, runtime=runtime)
        for name, value in values.items():
            components[f"draft.{name}"] = components.get(f"draft.{name}", ZERO) + float(value)
    components["draft_kv"] = float(math.fsum(
        value for name, value in components.items()
        if not name.endswith(("recurrent_state", "conv_state", "replay_records",
                              "state_transient"))))
    return components


def batch_cache_footprint(
    case: Any, *, requests: Sequence[Any], replica: int, phase: str,
    padded: bool,
) -> BatchCacheFootprint:
    """Physical batch footprint for one DP replica and phase."""
    model, parallel = case.model, case.runtime.parallel
    stages = max(ONE, int(parallel.pp))
    layer_stages = _layer_stages(model, parallel)
    shared: dict[str, tuple[int, float, int]] = {}
    private = [ZERO] * stages
    state = [ZERO] * stages
    draft = [ZERO] * stages
    boundary = [ZERO] * stages
    runtime = case.runtime
    shared_mode = getattr(runtime, "prefix_cache_mode", "private") == "shared"
    interleave = max(ONE, int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE))
    for view in requests:
        length = int(getattr(view, "kv_len", ZERO) or ZERO)
        declared_prefix = int(getattr(view, "prefix_tokens", ZERO) or ZERO)
        declared_prefix = max(ZERO, min(declared_prefix, length))
        if shared_mode:
            from mindie_sim.schema.runtime_features import canonical_prefix_cache_key
            key = canonical_prefix_cache_key(
                "shared", getattr(view, "prefix_cache_key", ""))
        else:
            key = f"private:{getattr(view, 'request_id', '')}"
        generation = ZERO
        for layer in model.layer_plans:
            ranges = cache_component_ranges(
                model, layer, parallel, sequence_tokens=length,
                prefix_tokens=declared_prefix, interleave=interleave,
                rank=None, capacity_padded=bool(padded))
            stage = layer_stages[int(layer.index)]
            for item in _prefixed(ranges, key, generation):
                if item.sharing == SHARING_SHARED and item.valid_bytes > ZERO:
                    previous = shared.get(item.owner_key)
                    if previous is None or item.end_token > previous[ZERO]:
                        shared[item.owner_key] = (item.end_token, item.valid_bytes,
                                                  stage)
                    continue
                private[stage] += float(item.valid_bytes)
                boundary[stage] += max(
                    ZERO, float(item.allocated_bytes) - float(item.valid_bytes))
                if item.component in STATE_COMPONENTS:
                    state[stage] += float(item.valid_bytes)
        if str(phase) == "decode" and getattr(model, "auxiliary", None) is not None:
            draft_kind = (dict(getattr(runtime, "metadata", ()) or ()).get(
                "draft_mode")
                or ("dspark" if getattr(runtime, "dspark", None) else "mtp"))
            values = draft_cache_components(
                model, parallel, draft_kind=draft_kind, sequence_tokens=length,
                runtime=runtime)
            draft[-ONE] += float(values.get("draft_kv", ZERO))
    shared_once = [ZERO] * stages
    for _end, value, stage in shared.values():
        shared_once[stage] += float(value)
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
