"""Declared allocator pages, separate from valid cache tensor payloads.

[Role] Price fixed, growing and windowed cache groups and provide the explicit
  alias/pool-aware peak allocation union used by capacity reports.
[Invariants] Each covered cache component has one allocator owner. Group pages
  are charged once; layer pages once per layer. Independent page geometries use
  committed tokens, never a sequence rounded by a different allocator. Page
  allocation is not a read: valid payload accesses and page/scale metadata
  reads are reported separately from padded capacity.
[Timeouts & Locks] Pure arithmetic bounded by declared groups and layers.
[Side-Effect Policy] No I/O or global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True)
class KVPageAllocation:
    """Physical KV pages and their separately charged bytes.

    ``token_spans`` are the live sequence spans, one entry per request/cache
    owner.  Pages are counted per span (two 17/31-token owners at 16 tokens per
    page hold four pages, not ``ceil(48/16)``).  ``element_bytes`` describes
    payload storage; ``scale_bytes_per_page`` and ``metadata_bytes_per_page``
    are explicit page-granular declarations, so no row scale is invented.
    ``access_*`` counts valid logical rows only, while allocation counts the
    padded physical slots.
    """

    token_spans: tuple[int, ...]
    page_tokens: int = 16
    local_kv_heads: int = 1
    head_dim: int = 8
    kv_count: int = 2
    element_bytes: float = 2.0
    scale_bytes_per_page: float = 0.0
    metadata_bytes_per_page: float = 0.0

    def __post_init__(self):
        raw_spans = tuple(self.token_spans)
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in raw_spans):
            raise ModelingError("kv_page_allocation_requires_integer_tokens")
        spans = tuple(raw_spans)
        if any(value < 0 for value in spans):
            raise ModelingError("kv_page_allocation_requires_nonnegative_tokens")
        object.__setattr__(self, "token_spans", spans)
        for name in ("page_tokens", "local_kv_heads", "head_dim", "kv_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelingError("kv_page_allocation_requires_positive_geometry")
        for name in ("element_bytes", "scale_bytes_per_page", "metadata_bytes_per_page"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not math.isfinite(float(value))
                    or float(value) < 0.0):
                raise ModelingError(f"kv_page_allocation_invalid:{name}")
            object.__setattr__(self, name, float(value))

    @property
    def logical_tokens(self) -> int:
        return sum(self.token_spans)

    @property
    def pages(self) -> int:
        if self.token_spans:
            return sum((value + int(self.page_tokens) - 1) // int(self.page_tokens)
                       for value in self.token_spans)
        return 0

    @property
    def physical_token_slots(self) -> int:
        return self.pages * int(self.page_tokens)

    @property
    def elements_per_token(self) -> int:
        return int(self.kv_count) * int(self.local_kv_heads) * int(self.head_dim)

    @property
    def valid_payload_bytes(self) -> float:
        return float(self.logical_tokens * self.elements_per_token) * self.element_bytes

    @property
    def allocated_payload_bytes(self) -> float:
        return float(self.physical_token_slots * self.elements_per_token) * self.element_bytes

    @property
    def page_tail_payload_bytes(self) -> float:
        return self.allocated_payload_bytes - self.valid_payload_bytes

    @property
    def scale_bytes(self) -> float:
        # Every allocated page carries its declared scales, including the final
        # partial page; the payload itself is not rounded into the access view.
        return float(self.pages) * self.scale_bytes_per_page

    @property
    def metadata_bytes(self) -> float:
        return float(self.pages) * self.metadata_bytes_per_page

    @property
    def total_allocation_bytes(self) -> float:
        return self.allocated_payload_bytes + self.scale_bytes + self.metadata_bytes

    @property
    def access_payload_bytes(self) -> float:
        return self.valid_payload_bytes

    @property
    def access_scale_bytes(self) -> float:
        return self.scale_bytes

    @property
    def access_metadata_bytes(self) -> float:
        return self.metadata_bytes

    @property
    def access_bytes(self) -> float:
        return (self.access_payload_bytes + self.access_scale_bytes
                + self.access_metadata_bytes)

    def to_dict(self) -> dict:
        return {
            "token_spans": list(self.token_spans),
            "logical_tokens": self.logical_tokens,
            "pages": self.pages,
            "page_tokens": int(self.page_tokens),
            "physical_token_slots": self.physical_token_slots,
            "elements_per_token": self.elements_per_token,
            "valid_payload_bytes": self.valid_payload_bytes,
            "allocated_payload_bytes": self.allocated_payload_bytes,
            "page_tail_payload_bytes": self.page_tail_payload_bytes,
            "scale_bytes": self.scale_bytes,
            "metadata_bytes": self.metadata_bytes,
            "total_allocation_bytes": self.total_allocation_bytes,
            "access_payload_bytes": self.access_payload_bytes,
            "access_scale_bytes": self.access_scale_bytes,
            "access_metadata_bytes": self.access_metadata_bytes,
            "access_bytes": self.access_bytes,
            "basis": "per_owner_physical_pages; allocation and access are separate",
        }


@dataclass(frozen=True)
class Allocation:
    """One named physical allocation in a capacity plan.

    ``alias_of`` makes this allocation a view of another named allocation, so
    it is not added a second time. ``pool`` places it inside a declared pool
    reservation; the pool's reserved bytes are charged once and suballocations
    are checked against that reservation.  ``live`` is the peak-point liveness
    flag; a sequentially reused buffer is retained in the report but excluded
    from the peak.
    """

    name: str
    bytes: float
    category: str = "workspace"
    alias_of: str | None = None
    pool: str | None = None
    live: bool = True
    note: str = ""
    physical_key: str | None = None
    #: Physical placement facts supplied by the selected program.  They are
    #: optional so existing plans keep their name/bytes/alias/pool semantics.
    rank: int | None = None
    storage: str | None = None
    tier: str = "hbm"
    generation: object | None = None
    ranges: tuple | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("allocation_name_required")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, (int, float)):
            raise ValueError(f"allocation_bytes_invalid:{self.name}")
        value = float(self.bytes)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"allocation_bytes_invalid:{self.name}")
        object.__setattr__(self, "bytes", value)
        if self.alias_of == self.name:
            raise ValueError(f"allocation_self_alias:{self.name}")
        if self.alias_of and self.pool:
            raise ValueError(f"allocation_alias_and_pool_mutually_exclusive:{self.name}")
        if not self.category:
            raise ValueError(f"allocation_category_required:{self.name}")


@dataclass(frozen=True)
class AllocationPlan:
    """A peak-point allocation union with alias and reserved-pool accounting."""

    allocations: tuple[Allocation, ...]
    capacity_bytes: float | None = None
    basis: str = "declared_physical_allocations"

    def __post_init__(self):
        allocations = tuple(self.allocations)
        names = [item.name for item in allocations]
        if len(set(names)) != len(names):
            raise ValueError("allocation_names_must_be_unique")
        object.__setattr__(self, "allocations", allocations)
        if self.capacity_bytes is not None:
            if isinstance(self.capacity_bytes, bool) or not isinstance(
                    self.capacity_bytes, (int, float)):
                raise ValueError("allocation_capacity_invalid")
            value = float(self.capacity_bytes)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("allocation_capacity_invalid")
            object.__setattr__(self, "capacity_bytes", value)

    def _root(self, allocation: Allocation) -> str:
        by_name = {item.name: item for item in self.allocations}
        seen = set()
        current = allocation
        while current.alias_of:
            if current.name in seen:
                raise ValueError(f"allocation_alias_cycle:{current.name}")
            seen.add(current.name)
            target = by_name.get(current.alias_of)
            if target is None:
                raise ValueError(f"allocation_alias_target_missing:{current.name}")
            if not target.live:
                raise ValueError(f"allocation_alias_target_not_live:{current.name}")
            current = target
        return current.name

    @staticmethod
    def _normalise_item_ranges(item: Allocation) -> tuple[tuple[float, float], ...]:
        raw = item.ranges
        if raw is None:
            return ((0.0, item.bytes),) if item.bytes > 0.0 else ()
        pairs: list[tuple[float, float]] = []
        for entry in raw:
            lo, hi = entry
            lo_f, hi_f = float(lo), float(hi)
            if hi_f > lo_f:
                pairs.append((lo_f, hi_f))
        return tuple(pairs)

    @staticmethod
    def _union_interval_bytes(ranges: list[tuple[float, float]]) -> float:
        ordered = sorted((float(lo), float(hi)) for lo, hi in ranges if hi > lo)
        merged: list[tuple[float, float]] = []
        for lo, hi in ordered:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return float(sum(hi - lo for lo, hi in merged))

    def resolve(self) -> dict:
        by_name = {item.name: item for item in self.allocations}
        pools = {item.name for item in self.allocations if item.category == "pool"}
        top: dict[str, Allocation] = {}
        phys_groups: dict[tuple, list[Allocation]] = {}
        aliases = {}
        pool_requests: dict[str, float] = {}
        retired = 0.0
        for item in self.allocations:
            if not item.live:
                retired += item.bytes
                continue
            if item.alias_of:
                root = self._root(item)
                aliases.setdefault(root, []).append({
                    "name": item.name, "bytes": item.bytes, "category": item.category,
                })
                continue
            if item.pool:
                if item.pool not in pools:
                    raise ValueError(f"allocation_pool_target_missing:{item.pool}")
                if not by_name[item.pool].live:
                    raise ValueError(f"allocation_pool_target_not_live:{item.pool}")
                pool_requests[item.pool] = pool_requests.get(item.pool, 0.0) + item.bytes
                continue
            if item.physical_key:
                g_key = (
                    item.physical_key,
                    item.rank,
                    str(item.storage or ""),
                    str(item.tier or "hbm").lower(),
                    item.generation,
                )
                phys_groups.setdefault(g_key, []).append(item)
            else:
                top[item.name] = item
        phys_not_added = 0.0
        for group_items in phys_groups.values():
            winner = max(group_items, key=lambda v: v.bytes)
            if any(it.ranges is not None for it in group_items):
                all_ranges = [
                    span for it in group_items for span in self._normalise_item_ranges(it)
                ]
                union_b = self._union_interval_bytes(all_ranges)
                raw_sum = float(sum(it.bytes for it in group_items))
                phys_not_added += max(0.0, raw_sum - union_b)
                if union_b != winner.bytes:
                    top[winner.name] = Allocation(
                        winner.name, union_b, winner.category,
                        live=True, note=winner.note, physical_key=winner.physical_key,
                        rank=winner.rank, storage=winner.storage, tier=winner.tier,
                        generation=winner.generation, ranges=winner.ranges,
                    )
                else:
                    top[winner.name] = winner
            else:
                top[winner.name] = winner
                phys_not_added += float(sum(it.bytes for it in group_items)) - winner.bytes
        for pool_name, requested in pool_requests.items():
            reserve = by_name[pool_name].bytes
            if requested > reserve + max(1e-9, reserve * 1e-12):
                raise ValueError(
                    f"allocation_pool_overcommitted:{pool_name}:{requested}>{reserve}")
        pool_report = {
            name: {
                "reserved_bytes": by_name[name].bytes,
                "requested_bytes": pool_requests.get(name, 0.0),
                "slack_bytes": by_name[name].bytes - pool_requests.get(name, 0.0),
            }
            for name in sorted(pools)
        }
        categories: dict[str, float] = {}
        for item in top.values():
            categories[item.category] = categories.get(item.category, 0.0) + item.bytes
        peak = float(sum(item.bytes for item in top.values()))
        result = {
            "basis": self.basis,
            "peak_bytes": peak,
            "capacity_bytes": self.capacity_bytes,
            "feasible": (None if self.capacity_bytes is None
                         else peak <= self.capacity_bytes),
            "headroom_bytes": (None if self.capacity_bytes is None
                              else self.capacity_bytes - peak),
            "retired_sequential_bytes": retired,
            "categories": dict(sorted(categories.items())),
            "allocations": [
                {"name": item.name, "bytes": item.bytes, "category": item.category,
                 "note": item.note}
                for item in sorted(top.values(), key=lambda value: value.name)
            ],
            "aliases": {name: values for name, values in sorted(aliases.items())},
            "pools": pool_report,
            "alias_bytes_not_added": float(sum(
                row["bytes"] for values in aliases.values() for row in values)),
            "pool_bytes_not_added": float(sum(pool_requests.values())),
            "physical_bytes_not_added": float(phys_not_added),
        }
        return result

    def to_dict(self) -> dict:
        return self.resolve()


@dataclass(frozen=True)
class CachePageGroup:
    layers: tuple[int, ...]
    tokens_per_page: int = 1
    bytes_per_layer_page: int = 0
    bytes_per_group_page: int = 0
    components: tuple[str, ...] = ()
    members: tuple[tuple[int, str], ...] = ()
    resident_tokens: int | None = None
    max_in_flight_tokens: int = 0
    recycling_pages: int = 0
    fixed_pages: int = 0
    allocation: str = "live"
    metadata_bytes_per_page: int = 0
    metadata_scan: str = "touched"
    auxiliary: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping) -> "CachePageGroup":
        values = dict(raw)
        if "members" in values:
            if "layers" in values or "components" in values:
                raise ModelingError("cache_page_group_members_replace_layers_and_components")
            values["members"] = tuple((member["layer"], member["component"])
                                      for member in values["members"])
            if len(set(values["members"])) != len(values["members"]):
                raise ModelingError("cache_page_group_duplicate_member")
            values["layers"] = tuple(dict.fromkeys(layer for layer, _ in values["members"]))
        else:
            values["layers"] = tuple(values["layers"])
        values["components"] = tuple(values.get("components", ()))
        result = cls(**values)
        if not result.layers or len(set(result.layers)) != len(result.layers):
            raise ModelingError("cache_page_group_requires_unique_layers")
        if (result.tokens_per_page <= 0 or result.fixed_pages < 0
                or result.metadata_bytes_per_page < 0):
            raise ModelingError("cache_page_group_invalid_page_geometry")
        if ((result.bytes_per_layer_page > 0) + (result.bytes_per_group_page > 0)) != 1:
            raise ModelingError("cache_page_group_requires_one_page_price")
        if result.allocation not in {"live", "max_model_len"}:
            raise ModelingError("cache_page_group_unknown_allocation")
        if result.metadata_scan not in {"touched", "all"}:
            raise ModelingError("cache_page_group_unknown_metadata_scan")
        if (result.resident_tokens is not None and result.resident_tokens < 0
                or result.max_in_flight_tokens < 0 or result.recycling_pages < 0):
            raise ModelingError("cache_page_group_negative_window_geometry")
        return result

    def metadata_bytes(self, pages: int) -> float:
        """Allocator page-table/index bytes for this group's physical pages.

        A group-page declaration owns one metadata record per group page.  A
        per-layer page declaration owns one record per covered layer page, so
        the charge is multiplied by the covered layer count.
        """
        if pages <= 0 or not self.metadata_bytes_per_page:
            return 0.0
        per_layer = 0 if self.bytes_per_group_page else len(self.layers)
        return float(pages * self.metadata_bytes_per_page * max(1, per_layer))

    def valid_tokens(self, tokens: int, max_model_len: int) -> int:
        """Valid sequence span this group can actually hold/read."""
        span = min(max(0, int(tokens)), max(0, int(max_model_len)))
        if self.resident_tokens is not None:
            span = min(span, int(self.resident_tokens))
        return span

    def allocated_token_slots(self, tokens: int, max_model_len: int) -> int:
        """Physical token rows in the pages allocated to this group."""
        return self.pages(tokens, max_model_len) * int(self.tokens_per_page)

    def touched_pages(self, tokens: int, max_model_len: int) -> int:
        """Pages whose metadata a valid access may actually touch.

        Allocation can reserve ``max_model_len``, in-flight and recycling
        pages, but a one-token access does not read all reserved page-table
        entries.  The access view uses the live valid/window span, not the
        allocator reservation, unless the declaration is a fixed ring.
        """
        span = self.valid_tokens(tokens, max_model_len)
        if span <= 0:
            return 0
        if self.fixed_pages:
            return min(self.fixed_pages,
                       max(1, math.ceil(span / int(self.tokens_per_page))))
        return max(1, math.ceil(span / int(self.tokens_per_page)))

    def access_metadata_bytes(self, tokens: int, max_model_len: int) -> float:
        if not self.metadata_bytes_per_page:
            return 0.0
        pages = (self.pages(tokens, max_model_len) if self.metadata_scan == "all"
                 else self.touched_pages(tokens, max_model_len))
        per_layer = 0 if self.bytes_per_group_page else len(self.layers)
        return float(pages * self.metadata_bytes_per_page * max(1, per_layer))

    def pages(self, tokens: int, max_model_len: int) -> int:
        if tokens <= 0:
            return 0
        if self.fixed_pages:
            return self.fixed_pages
        span = max_model_len if self.allocation == "max_model_len" else tokens
        if self.resident_tokens is not None:
            span = min(span, self.resident_tokens)
        span = min(span + self.max_in_flight_tokens, max_model_len)
        return (span + self.tokens_per_page - 1) // self.tokens_per_page + self.recycling_pages


class CachePages:
    """A component ownership map and its per-stage allocator geometry."""

    def __init__(self, declarations, component_names, layer_stage):
        self.groups = tuple(CachePageGroup.from_mapping(raw) for raw in declarations)
        self.layer_stage = tuple(layer_stage)
        self.component_names = {
            layer: set(names) for layer, names in component_names.items()
        }
        self.covered: dict[int, set[str]] = {}
        for group in self.groups:
            for layer in group.layers:
                if layer not in component_names:
                    raise ModelingError(f"cache_page_group_unknown_layer:{layer}")
                if group.auxiliary:
                    continue
                components = ({component for member_layer, component in group.members
                               if member_layer == layer} if group.members else
                              set(group.components or self.component_names[layer]))
                unknown = components - self.component_names[layer]
                if unknown:
                    raise ModelingError(f"cache_page_group_unknown_components:{layer}:{sorted(unknown)}")
                owned = self.covered.setdefault(layer, set())
                if owned & components:
                    raise ModelingError(f"cache_page_group_duplicate_owner:{layer}:{sorted(owned & components)}")
                owned.update(components)
            if group.bytes_per_group_page and len({layer_stage[i] for i in group.layers}) != 1:
                raise ModelingError("cache_page_group_spans_pipeline_stages")

    def group_components(self, group: CachePageGroup) -> dict[int, set[str]]:
        """Resolve the components a page group actually owns on each layer."""
        if group.auxiliary:
            return {}
        result = {}
        for layer in group.layers:
            if group.members:
                selected = {component for member_layer, component in group.members
                            if member_layer == layer}
            elif group.components:
                selected = set(group.components)
            else:
                selected = set(self.component_names.get(layer, ()))
            result[layer] = selected
        return result

    def stage_bytes(self, tokens: int, max_model_len: int, stages: int) -> list[float]:
        """Declared page payload bytes; metadata is reported separately."""
        result = [0.0] * stages
        for group in self.groups:
            pages = group.pages(tokens, max_model_len)
            if group.bytes_per_group_page:
                result[self.layer_stage[group.layers[0]]] += pages * group.bytes_per_group_page
            else:
                for layer in group.layers:
                    result[self.layer_stage[layer]] += pages * group.bytes_per_layer_page
        return result

    def metadata_stage_bytes(self, tokens: int, max_model_len: int,
                             stages: int) -> list[float]:
        result = [0.0] * stages
        for group in self.groups:
            pages = group.pages(tokens, max_model_len)
            metadata = group.metadata_bytes(pages)
            if not metadata:
                continue
            if group.bytes_per_group_page:
                # Group pages cannot span stages (validated above).
                result[self.layer_stage[group.layers[0]]] += metadata
            else:
                # Layer pages each own their metadata; splitting by layer keeps
                # a multi-stage declaration on the same stage as its payload.
                per_layer = metadata / len(group.layers)
                for layer in group.layers:
                    result[self.layer_stage[layer]] += per_layer
        return result

    def access_metadata_stage_bytes(self, tokens: int, max_model_len: int,
                                   stages: int) -> list[float]:
        result = [0.0] * stages
        for group in self.groups:
            metadata = group.access_metadata_bytes(tokens, max_model_len)
            if not metadata:
                continue
            if group.bytes_per_group_page:
                result[self.layer_stage[group.layers[0]]] += metadata
            else:
                per_layer = metadata / len(group.layers)
                for layer in group.layers:
                    result[self.layer_stage[layer]] += per_layer
        return result

    def allocation_stage_bytes(self, tokens: int, max_model_len: int,
                               stages: int) -> list[float]:
        """Physical page allocation including declared page metadata."""
        payload = self.stage_bytes(tokens, max_model_len, stages)
        metadata = self.metadata_stage_bytes(tokens, max_model_len, stages)
        return [left + right for left, right in zip(payload, metadata)]


__all__ = [
    "Allocation", "AllocationPlan", "CachePageGroup", "CachePages",
    "KVPageAllocation",
]
