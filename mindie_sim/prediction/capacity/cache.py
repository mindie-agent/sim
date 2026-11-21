"""Cache geometry, residency and handoff volume.

[Role] Derive stage and rank allocations for target and draft cache components.
[Invariants] Allocated payload, valid bytes and reserves stay distinct. Rank-local
  rows use explicit ownership when present, including uneven placement.
[Timeouts & Locks] Bounded by layers, page groups and ranks. No locks or I/O.
[Side-Effect Policy] Computes views of cache geometry without changing allocation holders.
"""
from __future__ import annotations

import math

from mindie_sim.catalog.cache import (
    cache_component_split,
    cache_row_split,
    layer_cache_components,
    local_kv_heads,
    owned_token_rows,
)
from mindie_sim.schema.common import dtype_bytes


class _MemoryCacheMixin:
    def cache_bytes(self, tokens: int, *, padded=True,
                    query_tokens: int | None = None) -> tuple[float, ...]:
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        declared = int(dict(case.runtime.metadata).get(
            "draft_proposal_tokens", 0) or 0)
        q_tok = (self.query_tokens if query_tokens is None
                 else max(1, int(query_tokens)))
        if (query_tokens is None and self.draft_kind == "dspark"
                and not self.paired_sfa and declared <= 0):
            # A bare DSpark spec widens the draft pool/page geometry; the
            # dense target state keeps its single-row reservation.
            q_tok = 1
        result = (self.pages.allocation_stage_bytes(tokens, case.runtime.max_model_len, p.pp)
                  if padded else [0.0] * p.pp)
        if model.layer_plans:
            for layer in model.layer_plans:
                components = layer_cache_components(
                    model, layer, p, tokens, capacity_padded=padded,
                    interleave=case.runtime.kv_block_size,
                    runtime=case.runtime, query_tokens=q_tok)
                covered = self.pages.covered.get(layer.index, ()) if padded else ()
                result[self.layer_stage[layer.index]] += sum(
                    value for name, value in components.items() if name not in covered)
        from mindie_sim.catalog.draft import optional_draft_spec
        mode = self.draft_kind
        if self.paired_sfa and mode and not padded:
            result[-1] += self._paired_draft_bytes(tokens)
        elif mode and model.auxiliary is not None and not self.paired_sfa:
            if self._uses_inherited_draft_cache(model):
                result[-1] += self._inherited_draft_cache_bytes(
                    model, tokens, padded=padded, query_tokens=q_tok)
            else:
                draft = optional_draft_spec(model, mode)
                row = ((draft.kv_cache_elements or draft.kv_lora_rank + draft.qk_rope_head_dim) if draft.attention_kind == "mla"
                       else 2 * local_kv_heads(draft.num_kv_heads, p.tp) * draft.head_dim)
                draft_tokens = (math.ceil(tokens / case.runtime.kv_block_size) * case.runtime.kv_block_size
                                if padded else tokens)
                result[-1] += draft_tokens * row * dtype_bytes(model.kv_dtype) * draft.layers
        return tuple(result)

    def _uses_inherited_draft_cache(self, model) -> bool:
        """True when draft KV/state follows the declared draft LayerPlans."""
        from mindie_sim.catalog.cache import uses_inherited_draft_cache
        return (not self.paired_sfa
                and uses_inherited_draft_cache(model, self.draft_kind))

    def _inherited_draft_cache_bytes(
        self, model, tokens: int, *, padded: bool, rank: int | None = None,
        query_tokens: int | None = None,
    ) -> float:
        """Draft cache payload from the inherited plans, with T05 ownership."""
        from mindie_sim.catalog.cache_ownership import draft_cache_components

        values = draft_cache_components(
            model, self.case.runtime.parallel, draft_kind=self.draft_kind,
            sequence_tokens=tokens, capacity_padded=padded,
            runtime=self.case.runtime, rank=rank,
            owner_spans=self.rank_token_owners if rank is not None else None,
            query_tokens=(self.query_tokens if query_tokens is None
                          else query_tokens),
        )
        return float(values["draft_kv"])

    def _group_cache_parts(self, tokens: int, rank: int | None = None) -> list[dict]:
        """Known physical and valid components inside declared page groups.

        Token-paged components use the group's allocated token slots for
        capacity and its valid/window span for content.  Compressed/index and
        fixed-state components use their own valid geometry; the undeclared
        remainder of the group quote is kept as ``reserve``, never asserted as
        precise payload.  With ``rank`` the valid components are that physical
        rank's owner rows, while every rank still holds the full page slots.
        """
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        stages = [
            {"alloc_payload": 0.0, "alloc_scale": 0.0,
             "valid_payload": 0.0, "valid_scale": 0.0, "components": {}}
            for _ in range(p.pp)
        ]
        by_index = {plan.index: plan for plan in model.layer_plans}
        for group in self.pages.groups:
            owned = self.pages.group_components(group)
            valid_tokens = group.valid_tokens(tokens, case.runtime.max_model_len)
            alloc_tokens = group.allocated_token_slots(tokens, case.runtime.max_model_len)
            for layer_index in group.layers:
                layer = by_index[layer_index]
                components = layer_cache_components(
                    model, layer, p, valid_tokens, capacity_padded=False,
                    interleave=case.runtime.kv_block_size,
                    rank=rank, owner_spans=self.rank_token_owners,
                    runtime=case.runtime, query_tokens=self.query_tokens,
                )
                stage = stages[self.layer_stage[layer.index]]
                for name in owned.get(layer_index, ()):
                    part = None
                    if not self.paired_sfa and name in {"gqa_kv", "mla_kv", "window"}:
                        part = self._group_token_component(
                            model, layer, p, group, name, valid_tokens,
                            alloc_tokens, components,
                        )
                    if part is None:
                        value = components.get(name, 0.0)
                        if value <= 0.0:
                            continue
                        payload, scale = cache_component_split(
                            model, layer, p, name, value)
                        alloc_payload = valid_payload = payload
                        alloc_scale = valid_scale = scale
                    else:
                        alloc_payload, alloc_scale, valid_payload, valid_scale = part
                    stage["alloc_payload"] += alloc_payload
                    stage["alloc_scale"] += alloc_scale
                    stage["valid_payload"] += valid_payload
                    stage["valid_scale"] += valid_scale
                    detail = stage["components"].setdefault(name, {
                        "alloc_payload": 0.0, "alloc_scale": 0.0,
                        "valid_payload": 0.0, "valid_scale": 0.0,
                    })
                    detail["alloc_payload"] += alloc_payload
                    detail["alloc_scale"] += alloc_scale
                    detail["valid_payload"] += valid_payload
                    detail["valid_scale"] += valid_scale
        if self.paired_sfa and self.draft_kind:
            value = self._paired_draft_bytes(tokens, rank=rank)
            stage = stages[-1]
            for name in ("alloc_payload", "valid_payload"):
                stage[name] += value
            stage["components"]["draft_kv"] = {
                "alloc_payload": value, "alloc_scale": 0,
                "valid_payload": value, "valid_scale": 0}
        return stages

    def _paired_draft_bytes(self, tokens: int, *, rank=None) -> float:
        from mindie_sim.catalog.draft import optional_draft_spec
        from mindie_sim.catalog.paired_sfa import draft_row
        draft = optional_draft_spec(self.case.model, self.draft_kind)
        rows = tokens / self.case.runtime.parallel.cp
        if rank is not None and self.rank_token_owners is not None:
            rows = owned_token_rows(tokens, self.rank_token_owners, rank)
        return rows * draft_row(self.case.model, draft) * draft.layers

    def _group_token_component(
        self, model, layer, parallel, group, name: str, valid_tokens: int,
        alloc_tokens: int, components: dict[str, float],
    ):
        """Rows/bytes for a token-paged component inside one group page."""
        if name == "gqa_kv":
            if (layer.attention not in {"gqa", "projection_only"}
                    or 0 <= layer.kv_owner_layer_id < layer.index):
                return None
            region = "gqa_kv"
            width = 2 * local_kv_heads(layer.num_kv_heads, parallel.tp) * layer.head_dim
            cap = layer.window or 0
        elif name == "mla_kv":
            if layer.attention != "mla" or layer.cache_kind != "dense":
                return None
            region = "mla_kv"
            width = layer.cache_row_elements
            cap = layer.window or 0
        else:  # window
            if layer.attention != "mla" or not layer.window:
                return None
            region = "window"
            width = layer.cache_row_elements
            cap = layer.window
        if name not in components and valid_tokens > 0:
            return None
        # The physical page owns all allocated slots.  Valid content comes from
        # the already CP/TP-local layer_cache_components range, not a second
        # global valid_tokens calculation that would overwrite the local shard.
        alloc_rows = alloc_tokens
        payload, scale = cache_row_split(model, region, width)
        valid_value = components.get(name, 0.0)
        if valid_value > 0.0:
            valid_payload, valid_scale = cache_component_split(
                model, layer, parallel, name, valid_value)
        else:
            valid_payload = valid_scale = 0.0
        return (alloc_rows * payload, alloc_rows * scale,
                valid_payload, valid_scale)

    def _kv_residency(self, tokens: int) -> tuple[list[float], list[float], list[float]]:
        """(gross, HBM-resident, staging) KV bytes, same basis as resize."""
        gross = list(self.cache_bytes(tokens))
        if self.kvc_hbm_ratio == 1.0:
            return gross, list(gross), [0.0] * len(gross)
        resident = [value * self.kvc_hbm_ratio for value in gross]
        staging = [
            value * (1.0 - self.kvc_hbm_ratio)
            for value in self._max_layer_cache_bytes(tokens)
        ]
        return gross, resident, staging

    def _cache_component_breakdown(
        self, tokens: int, *, padded: bool, skip_covered: bool,
        rank: int | None = None,
    ) -> list[dict]:
        """Per-stage component payload/scale split for one liveness view."""
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        stages = [
            {"payload": 0.0, "scale": 0.0, "total": 0.0, "components": {}}
            for _ in range(p.pp)
        ]
        for layer in model.layer_plans:
            components = layer_cache_components(
                model, layer, p, tokens, capacity_padded=padded,
                interleave=case.runtime.kv_block_size,
                rank=rank, owner_spans=self.rank_token_owners,
                runtime=case.runtime, query_tokens=self.query_tokens,
            )
            covered = self.pages.covered.get(layer.index, ()) if skip_covered else ()
            stage = stages[self.layer_stage[layer.index]]
            for name, value in components.items():
                if name in covered:
                    continue
                payload, scale = cache_component_split(
                    model, layer, p, name, value)
                stage["payload"] += payload
                stage["scale"] += scale
                stage["total"] += float(value)
                detail = stage["components"].setdefault(
                    name, {"payload": 0.0, "scale": 0.0, "total": 0.0})
                detail["payload"] += payload
                detail["scale"] += scale
                detail["total"] += float(value)
        from mindie_sim.catalog.draft import optional_draft_spec
        mode = self.draft_kind
        if mode and model.auxiliary is not None and not self.paired_sfa:
            if self._uses_inherited_draft_cache(model):
                value = self._inherited_draft_cache_bytes(
                    model, tokens, padded=padded, rank=rank)
            else:
                draft = optional_draft_spec(model, mode)
                row = ((draft.kv_cache_elements or draft.kv_lora_rank + draft.qk_rope_head_dim)
                       if draft.attention_kind == "mla" else
                       2 * local_kv_heads(draft.num_kv_heads, p.tp) * draft.head_dim)
                draft_tokens = (math.ceil(tokens / case.runtime.kv_block_size)
                                * case.runtime.kv_block_size if padded else tokens)
                if rank is not None and self.rank_token_owners is not None:
                    draft_tokens = owned_token_rows(
                        draft_tokens, self.rank_token_owners, rank)
                value = draft_tokens * row * dtype_bytes(model.kv_dtype) * draft.layers
            stage = stages[-1]
            stage["payload"] += float(value)
            stage["total"] += float(value)
            detail = stage["components"].setdefault(
                "draft_kv", {"payload": 0.0, "scale": 0.0, "total": 0.0})
            detail["payload"] += float(value)
            detail["total"] += float(value)
        return stages

    def _rank_cache_bytes(self, tokens: int, *, padded: bool, rank: int) -> list[float]:
        """Allocation bytes for one physical rank.

        With an explicit T05 owner partition the rank owns its exact rows.
        Without one the legacy stage-level per-rank value is retained, so old
        callers and even CP=2 slices are unchanged.
        """
        if self.paired_sfa or self.rank_token_owners is None:
            return list(self.cache_bytes(tokens, padded=padded))
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        result = (
            self.pages.allocation_stage_bytes(tokens, case.runtime.max_model_len, p.pp)
            if padded else [0.0] * p.pp
        )
        for layer in model.layer_plans:
            components = layer_cache_components(
                model, layer, p, tokens, capacity_padded=padded,
                interleave=case.runtime.kv_block_size,
                rank=int(rank), owner_spans=self.rank_token_owners,
                runtime=case.runtime, query_tokens=self.query_tokens,
            )
            covered = self.pages.covered.get(layer.index, ()) if padded else ()
            result[self.layer_stage[layer.index]] += sum(
                value for name, value in components.items() if name not in covered)
        from mindie_sim.catalog.draft import optional_draft_spec
        mode = self.draft_kind
        if mode and model.auxiliary is not None:
            if self._uses_inherited_draft_cache(model):
                result[-1] += self._inherited_draft_cache_bytes(
                    model, tokens, padded=padded, rank=int(rank))
            else:
                draft = optional_draft_spec(model, mode)
                row = ((draft.kv_cache_elements or draft.kv_lora_rank + draft.qk_rope_head_dim)
                       if draft.attention_kind == "mla"
                       else 2 * local_kv_heads(draft.num_kv_heads, p.tp) * draft.head_dim)
                draft_tokens = (math.ceil(tokens / case.runtime.kv_block_size)
                                * case.runtime.kv_block_size if padded else tokens)
                rows = owned_token_rows(draft_tokens, self.rank_token_owners, int(rank))
                result[-1] += rows * row * dtype_bytes(model.kv_dtype) * draft.layers
        return result

    def _rank_max_layer_cache_bytes(self, tokens: int, rank: int) -> tuple[float, ...]:
        if self.rank_token_owners is None:
            return self._max_layer_cache_bytes(tokens)
        model, p = self.case.model, self.case.runtime.parallel
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            comps = layer_cache_components(
                model, layer, p, tokens, capacity_padded=False,
                interleave=self.case.runtime.kv_block_size,
                rank=int(rank), owner_spans=self.rank_token_owners,
                runtime=self.case.runtime, query_tokens=self.query_tokens,
            )
            st = self.layer_stage[layer.index]
            result[st] = max(result[st], float(sum(comps.values())))
        return tuple(result)

    def _kv_residency_ranks(
        self, tokens: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """(HBM-resident, staging) per stage and physical rank."""
        p = self.case.runtime.parallel
        if self.rank_token_owners is None:
            _gross, resident, staging = self._kv_residency(tokens)
            return (
                [[float(resident[stage])] * self.width for stage in range(p.pp)],
                [[float(staging[stage])] * self.width for stage in range(p.pp)],
            )
        resident = [[0.0] * self.width for _ in range(p.pp)]
        staging = [[0.0] * self.width for _ in range(p.pp)]
        for rank in range(self.width):
            gross = self._rank_cache_bytes(tokens, padded=True, rank=rank)
            layer_max = self._rank_max_layer_cache_bytes(tokens, rank)
            for stage in range(p.pp):
                if self.kvc_hbm_ratio == 1.0:
                    resident[stage][rank] = float(gross[stage])
                    staging[stage][rank] = 0.0
                else:
                    resident[stage][rank] = float(gross[stage]) * self.kvc_hbm_ratio
                    staging[stage][rank] = (
                        float(layer_max[stage]) * (1.0 - self.kvc_hbm_ratio))
        return resident, staging

    def _allocation_view(
        self, tokens: int, *, rank: int | None = None,
        include_components: bool = True,
    ) -> dict:
        """One physical rank's (or the legacy stage's) cache allocation view."""
        case, p = self.case, self.case.runtime.parallel
        allocated = self._cache_component_breakdown(
            tokens, padded=True, skip_covered=True, rank=rank)
        valid = self._cache_component_breakdown(
            tokens, padded=False, skip_covered=True, rank=rank)
        group_parts = self._group_cache_parts(tokens, rank=rank)
        group_payload = self.pages.stage_bytes(
            tokens, case.runtime.max_model_len, p.pp)
        group_metadata = self.pages.metadata_stage_bytes(
            tokens, case.runtime.max_model_len, p.pp)
        group_access_metadata = self.pages.access_metadata_stage_bytes(
            tokens, case.runtime.max_model_len, p.pp)
        result = {
            "sequence_tokens": int(tokens),
            "basis": "padded_cache_pages_separate_from_valid_access",
            "allocated": {
                "payload_bytes_per_stage_rank": [],
                "scale_bytes_per_stage_rank": [],
                "reserve_bytes_per_stage_rank": [],
                "metadata_bytes_per_stage_rank": [],
                "total_bytes_per_stage_rank": [],
            },
            "valid": {
                "payload_bytes_per_stage_rank": [],
                "scale_bytes_per_stage_rank": [],
                "total_bytes_per_stage_rank": [],
            },
            "page_tail": {
                "payload_bytes_per_stage_rank": [],
                "scale_bytes_per_stage_rank": [],
                "reserve_bytes_per_stage_rank": [],
                "metadata_bytes_per_stage_rank": [],
                "total_bytes_per_stage_rank": [],
            },
            "access": {
                "payload_bytes_per_stage_rank": [],
                "scale_bytes_per_stage_rank": [],
                "metadata_bytes_per_stage_rank": [],
                "total_bytes_per_stage_rank": [],
            },
            "components": [],
            "page_groups": {
                "payload_bytes_per_stage_rank": list(group_payload),
                "metadata_bytes_per_stage_rank": list(group_metadata),
                "access_metadata_bytes_per_stage_rank": list(group_access_metadata),
                "component_payload_bytes_per_stage_rank":
                    [stage["alloc_payload"] for stage in group_parts],
                "component_scale_bytes_per_stage_rank":
                    [stage["alloc_scale"] for stage in group_parts],
                "valid_component_payload_bytes_per_stage_rank":
                    [stage["valid_payload"] for stage in group_parts],
                "valid_component_scale_bytes_per_stage_rank":
                    [stage["valid_scale"] for stage in group_parts],
            },
        }
        if rank is not None:
            result["physical_rank"] = int(rank)
        for index in range(p.pp):
            known_payload = group_parts[index]["alloc_payload"]
            known_scale = group_parts[index]["alloc_scale"]
            reserve = max(0.0, group_payload[index] - known_payload - known_scale)
            alloc_payload = allocated[index]["payload"] + known_payload
            alloc_scale = allocated[index]["scale"] + known_scale
            metadata = group_metadata[index]
            access_metadata = group_access_metadata[index]
            alloc_total = alloc_payload + alloc_scale + reserve + metadata
            valid_payload = valid[index]["payload"] + group_parts[index]["valid_payload"]
            valid_scale = valid[index]["scale"] + group_parts[index]["valid_scale"]
            valid_total = valid_payload + valid_scale
            tail_payload = max(0.0, alloc_payload - valid_payload)
            tail_scale = max(0.0, alloc_scale - valid_scale)
            tail_metadata = max(0.0, metadata - access_metadata)
            result["allocated"]["payload_bytes_per_stage_rank"].append(alloc_payload)
            result["allocated"]["scale_bytes_per_stage_rank"].append(alloc_scale)
            result["allocated"]["reserve_bytes_per_stage_rank"].append(reserve)
            result["allocated"]["metadata_bytes_per_stage_rank"].append(metadata)
            result["allocated"]["total_bytes_per_stage_rank"].append(alloc_total)
            result["valid"]["payload_bytes_per_stage_rank"].append(valid_payload)
            result["valid"]["scale_bytes_per_stage_rank"].append(valid_scale)
            result["valid"]["total_bytes_per_stage_rank"].append(valid_total)
            result["page_tail"]["payload_bytes_per_stage_rank"].append(tail_payload)
            result["page_tail"]["scale_bytes_per_stage_rank"].append(tail_scale)
            result["page_tail"]["reserve_bytes_per_stage_rank"].append(reserve)
            result["page_tail"]["metadata_bytes_per_stage_rank"].append(tail_metadata)
            result["page_tail"]["total_bytes_per_stage_rank"].append(
                tail_payload + tail_scale + reserve)
            result["access"]["payload_bytes_per_stage_rank"].append(valid_payload)
            result["access"]["scale_bytes_per_stage_rank"].append(valid_scale)
            result["access"]["metadata_bytes_per_stage_rank"].append(access_metadata)
            result["access"]["total_bytes_per_stage_rank"].append(
                valid_payload + valid_scale + access_metadata)
            if include_components:
                result["components"].append({
                    "stage": index,
                    "allocated": allocated[index]["components"],
                    "group_allocated": group_parts[index]["components"],
                    "valid": valid[index]["components"],
                })
        expected = (list(self.cache_bytes(tokens)) if rank is None else
                    self._rank_cache_bytes(tokens, padded=True, rank=rank))
        if any(abs(left - right) > max(1.0, abs(right) * 1e-9) for left, right in zip(
                result["allocated"]["total_bytes_per_stage_rank"], expected)):
            raise ValueError("memory_cache_allocation_not_conserved")
        return result

    def rank_cache_allocation(self, tokens: int) -> tuple[dict, ...]:
        """Per-physical-rank allocated/valid/page-tail view (same ledger basis)."""
        if self.rank_token_owners is None:
            view = self._allocation_view(tokens, rank=None)
            return tuple(dict(view) for _ in range(self.width))
        return tuple(
            self._allocation_view(tokens, rank=rank)
            for rank in range(self.width)
        )

    def cache_allocation(self, tokens: int, *, include_components: bool = True) -> dict:
        """Physical cache allocation versus valid logical access.

        Page groups keep their known component row scales instead of being
        flattened to opaque payload.  Unknown group internals remain a declared
        group price.  Allocation metadata is charged for every reserved page;
        access metadata counts only pages touched by this valid span or a fixed
        ring's declared touched page.  ``physical_ranks`` carries each rank's
        exact allocated/valid/tail view when a T05 owner partition is supplied.
        """
        result = self._allocation_view(
            tokens, rank=None, include_components=include_components)
        if self.rank_token_owners is not None:
            result["physical_ranks"] = [
                self._allocation_view(tokens, rank=rank,
                                      include_components=include_components)
                for rank in range(self.width)
            ]
        return result

    def _max_layer_cache_bytes(self, tokens: int) -> tuple[float, ...]:
        model, p = self.case.model, self.case.runtime.parallel
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            comps = layer_cache_components(
                model, layer, p, tokens, capacity_padded=False,
                interleave=self.case.runtime.kv_block_size,
                runtime=self.case.runtime, query_tokens=self.query_tokens)
            st = self.layer_stage[layer.index]
            result[st] = max(result[st], float(sum(comps.values())))
        return tuple(result)

    def host_dram_cache_bytes(self, tokens: int, *, padded=True) -> float:
        if not self.sparse_offload:
            return 0.0
        block_size = int(self.sparse_offload["block_size"])
        eff_tokens = math.ceil(tokens / block_size) * block_size if padded else tokens
        offload_layers = len(self.sparse_offload["offload_layers"])
        return float(eff_tokens * int(self.sparse_offload["kv_row_bytes"]) * offload_layers)

    def pd_handoff_volume_and_stage_hbm(self, tokens: int) -> tuple[float, tuple[float, ...]]:
        """Compute SfaRemoteD2HConnector wire volume (D2D index_k + 1x RD2H mla_kv) and stage HBM write."""
        p = self.case.runtime.parallel
        if not self.sparse_offload or self.sparse_offload["options"].get("keep_device_kv_cache"):
            local = self.cache_bytes(tokens, padded=False)
            return sum(local) * p.tp * p.pcp, local
        offload_set = set(self.sparse_offload["offload_layers"])
        topk_buf = int(self.sparse_offload["topk_buffer_size"])
        use_copy_sfa = bool(self.sparse_offload["options"].get("use_fused_copy_sfa", True))
        hot_tokens = min(tokens, topk_buf) if use_copy_sfa else 0
        stage_hbm = [0.0] * p.pp
        rd2h_bytes = 0.0
        for layer in self.case.model.layer_plans:
            components = layer_cache_components(
                self.case.model, layer, p, tokens, capacity_padded=False,
                interleave=self.case.runtime.kv_block_size,
                runtime=self.case.runtime, query_tokens=self.query_tokens,
            )
            st = self.layer_stage[layer.index]
            for name, val in components.items():
                if name == "mla_kv" and layer.index in offload_set:
                    rd2h_bytes += val * p.cp
                    live = min(tokens, layer.window) if layer.window else tokens
                    if hot_tokens and live:
                        stage_hbm[st] += val * min(hot_tokens, live) / live
                else:
                    stage_hbm[st] += val
        wire_volume = sum(stage_hbm) * p.tp * p.pcp + rd2h_bytes
        return wire_volume, tuple(stage_hbm)
