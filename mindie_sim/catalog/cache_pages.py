"""Declared allocator pages, separate from valid cache tensor payloads.

[Role] Price fixed, growing and windowed cache groups on their owning PP stage.
[Invariants] Each covered cache component has one allocator owner. Group pages
  are charged once; layer pages once per layer. Independent page geometries use
  committed tokens, never a sequence rounded by a different allocator.
[Timeouts & Locks] Pure arithmetic bounded by declared groups and layers.
[Side-Effect Policy] No I/O or global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mindie_sim.schema.common import ModelingError


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
        if result.tokens_per_page <= 0 or result.fixed_pages < 0:
            raise ModelingError("cache_page_group_invalid_page_geometry")
        if ((result.bytes_per_layer_page > 0) + (result.bytes_per_group_page > 0)) != 1:
            raise ModelingError("cache_page_group_requires_one_page_price")
        if result.allocation not in {"live", "max_model_len"}:
            raise ModelingError("cache_page_group_unknown_allocation")
        if (result.resident_tokens is not None and result.resident_tokens < 0
                or result.max_in_flight_tokens < 0 or result.recycling_pages < 0):
            raise ModelingError("cache_page_group_negative_window_geometry")
        return result

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
        self.covered: dict[int, set[str]] = {}
        for group in self.groups:
            for layer in group.layers:
                if layer not in component_names:
                    raise ModelingError(f"cache_page_group_unknown_layer:{layer}")
                components = ({component for member_layer, component in group.members
                               if member_layer == layer} if group.members else
                              set(group.components or component_names[layer]))
                unknown = components - component_names[layer]
                if unknown:
                    raise ModelingError(f"cache_page_group_unknown_components:{layer}:{sorted(unknown)}")
                owned = self.covered.setdefault(layer, set())
                if owned & components:
                    raise ModelingError(f"cache_page_group_duplicate_owner:{layer}:{sorted(owned & components)}")
                owned.update(components)
            if group.bytes_per_group_page and len({layer_stage[i] for i in group.layers}) != 1:
                raise ModelingError("cache_page_group_spans_pipeline_stages")

    def stage_bytes(self, tokens: int, max_model_len: int, stages: int) -> list[float]:
        result = [0.0] * stages
        for group in self.groups:
            pages = group.pages(tokens, max_model_len)
            if group.bytes_per_group_page:
                result[self.layer_stage[group.layers[0]]] += pages * group.bytes_per_group_page
            else:
                for layer in group.layers:
                    result[self.layer_stage[layer]] += pages * group.bytes_per_layer_page
        return result
