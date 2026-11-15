"""Caller-owned bounded semantic graph cache.

[Role]
  Cache only immutable semantic operators/annotations before hardware-dependent
  recipes, prices, contexts, events or Memory binding.  Physical plans remain
  per-SKU and are never shared through this cache.

[Invariants]
  * LRU eviction by entry count and byte budget; an entry larger than half the
    byte budget is not cached.
  * A cache hit returns a deeply immutable template; every request/rank/state
    identity is rebound by the caller.
  * Keys include every frontend-structural feature field and the implementation
    selection signature; rates and memory capacity are excluded only here.

[Timeouts & Locks]
  Single-threaded bounded LRU; no I/O or locks.

[Side-Effect Policy]
  Mutates only the caller's cache instance.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any, Callable

from mindie_sim.schema.common import ModelingError


ZERO = 0
ONE = 1
FIRST = 0
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ENTRIES = 128
HALF = 2
FREEZE_TAG_LIST = "__list__"
FREEZE_TAG_MAP = "__map__"
FREEZE_TAG_SET = "__set__"
FREEZE_TAG_REPR = "__repr__"


def freeze_value(value: Any) -> Any:
    """Convert nested metadata into a hashable, deeply immutable projection."""
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    if isinstance(value, Mapping):
        return (FREEZE_TAG_MAP, tuple(sorted(
            ((str(key), freeze_value(item)) for key, item in value.items()),
            key=lambda pair: pair[ZERO])))
    if isinstance(value, (list, tuple)):
        return (FREEZE_TAG_LIST, tuple(freeze_value(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (FREEZE_TAG_SET, tuple(sorted(
            (freeze_value(item) for item in value), key=repr)))
    try:
        hash(value)
        return value
    except TypeError:
        return (FREEZE_TAG_REPR, type(value).__name__, repr(value))


@dataclass(frozen=True)
class GraphKey:
    """Hashable semantic identity of one frontend graph template."""

    scope: str
    signature: tuple
    selection: tuple = ()

    def to_dict(self) -> dict[str, Any]:
        return {"scope": self.scope, "signature": self.signature,
                "selection": self.selection}


@dataclass
class SemanticGraphCache:
    """Bounded caller-owned LRU of immutable semantic graph templates."""

    max_bytes: int = DEFAULT_MAX_BYTES
    max_entries: int = DEFAULT_MAX_ENTRIES
    _entries: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _sizes: dict = field(default_factory=dict, repr=False)
    _bytes: int = ZERO
    hits: int = ZERO
    misses: int = ZERO
    evictions: int = ZERO
    oversized: int = ZERO

    def __post_init__(self) -> None:
        if self.max_bytes < ONE or self.max_entries < ONE:
            raise ModelingError("semantic_graph_cache_budget_invalid")

    @property
    def total_bytes(self) -> int:
        return int(self._bytes)

    @property
    def entries(self) -> int:
        return len(self._entries)

    def _size(self, value: Any) -> int:
        frozen = freeze_value(value)
        try:
            return len(json.dumps(frozen, default=repr).encode("utf-8"))
        except (TypeError, ValueError):
            return len(repr(frozen).encode("utf-8"))

    def _evict(self) -> None:
        while (self._entries
               and (len(self._entries) > self.max_entries
                    or self._bytes > self.max_bytes)):
            key, _template = self._entries.popitem(last=False)
            self._bytes -= self._sizes.pop(key, ZERO)
            self.evictions += ONE

    def get_or_build(self, key: GraphKey, builder: Callable[[], Any]) -> Any:
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self.hits += ONE
            return cached
        self.misses += ONE
        template = builder()
        size = self._size(template)
        if size > self.max_bytes // HALF:
            self.oversized += ONE
            return template
        self._entries[key] = template
        self._sizes[key] = size
        self._bytes += size
        self._evict()
        return template

    def clear(self) -> None:
        self._entries.clear()
        self._sizes.clear()
        self._bytes = ZERO
        self.hits = self.misses = self.evictions = self.oversized = ZERO

    def stats(self) -> dict[str, int]:
        return {"entries": self.entries, "bytes": self.total_bytes,
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions, "oversized": self.oversized}


def semantic_graph_key(
    case: Any, plan: Any, *, scope: str, selection_signature: Any = (),
) -> GraphKey:
    """Build the hardware-independent semantic key of one graph template."""
    runtime = case.runtime
    parallel = runtime.parallel
    model = case.model
    signature = freeze_value({
        "schema": "mindie-sim.semantic-graph/1",
        "scope": str(scope),
        "model": {
            "name": model.name, "layers": model.layers,
            "hidden_size": model.hidden_size, "num_heads": model.num_heads,
            "num_kv_heads": model.num_kv_heads, "head_dim": model.head_dim,
            "vocab_size": model.vocab_size, "num_experts": model.num_experts,
            "top_k": model.top_k, "quant": getattr(model.quant, "name", ""),
            "layer_plans": tuple(
                (item.index, item.attention, item.ffn, item.scope,
                 item.cache_kind, item.window, item.compress_ratio,
                 item.num_experts, item.top_k)
                for item in model.layer_plans),
            "auxiliary": freeze_value(getattr(model, "auxiliary", None)),
        },
        "parallel": {
            "dp": parallel.dp, "tp": parallel.tp, "ep": parallel.ep,
            "pp": parallel.pp, "cp": parallel.cp, "pcp": parallel.pcp,
            "o_proj_tp": parallel.o_proj_tp,
            "linear_proj_tp": parallel.linear_proj_tp,
            "lm_head_tp": parallel.lm_head_tp,
            "redundant_experts_per_rank": parallel.redundant_experts_per_rank,
            "cp_plan": parallel.cp_plan,
            "rank_table": parallel.rank_table,
        },
        "features": {
            "collective_startup_mode": runtime.collective_startup_mode,
            "matrix_compute_policy": runtime.matrix_compute_policy,
            "recover_ssm": runtime.recover_ssm,
            "prefix_cache_mode": runtime.prefix_cache_mode,
            "prefix_read_mode": runtime.prefix_read_mode,
            "homogeneous_graph_reuse": runtime.homogeneous_graph_reuse,
            "routing": freeze_value(runtime.routing),
            "metadata": freeze_value(runtime.metadata),
            "attention_policy": freeze_value(runtime.attention_policy),
            "lowering_passes": runtime.lowering_passes,
        },
        "plan": freeze_value(plan),
        "selection": freeze_value(selection_signature),
    })
    return GraphKey(str(scope), signature, freeze_value(selection_signature))
