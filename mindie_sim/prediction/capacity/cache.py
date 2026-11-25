"""Cache geometry, residency and handoff volume.

[Role] Derive stage and rank allocations for target and draft cache components.
[Invariants] Allocated payload, valid bytes and reserves stay distinct. Rank-local
  rows use explicit ownership when present, including uneven placement.
[Timeouts & Locks] Bounded by layers, page groups and ranks. No locks or I/O.
[Side-Effect Policy] Computes views of cache geometry without changing allocation holders.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from mindie_sim.catalog.cache import (
    _context_shard_blocks,
    block_round,
    cache_component_split,
    cache_row_split,
    index_row_parts,
    layer_cache_components,
    local_kv_heads,
    owned_token_rows,
)
from mindie_sim.catalog.cache_format import gqa_kv_formats
from mindie_sim.catalog.cache_pools import (
    REGIME_UNIFORM_INDEPENDENT,
    HybridPoolReceipt,
    _ceil_div,
    _draft_is_dcp_sharded,
    _draft_row_bytes_int,
    evaluate_hybrid_pool_receipt,
)
from mindie_sim.catalog.context_parallel import (
    cp_interleave,
    dcp_indexer_sharded,
    layer_dcp_factors,
    owned_positions,
)

_DCP_KV_SHARDED_COMPONENTS = frozenset({
    "gqa_kv", "mla_kv", "window", "compressed_kv",
})
_TARGET_KV_POOL_COMPONENTS = frozenset({
    "gqa_kv", "mla_kv", "window", "compressed_kv",
})
_INDEXER_POOL_COMPONENTS = frozenset({
    "index_k", "index_pool_tail", "index_compressor_state",
})
_RECURRENT_POOL_COMPONENTS = frozenset({
    "recurrent_state", "conv_state", "replay_records", "state_transient",
    "compressor_state", "ple_conv_state",
})
_COMPONENT_POOL_MAP = {
    **{name: "target_kv_bytes" for name in _TARGET_KV_POOL_COMPONENTS},
    **{name: "indexer_k_bytes" for name in _INDEXER_POOL_COMPONENTS},
    **{name: "recurrent_state_bytes" for name in _RECURRENT_POOL_COMPONENTS},
    "draft_kv": "draft_kv_bytes",
    "draft_state": "draft_state_bytes",
    "dcp_dp_cohort_kv": "target_kv_bytes",
    "dcp_dp_cohort_index": "indexer_k_bytes",
    "dcp_dp_cohort_draft": "draft_kv_bytes",
}


class _MemoryCacheMixin:
    def _effective_cache_query_tokens(self, query_tokens: int | None = None) -> int:
        # ``self.query_tokens`` already carries the shared DSpark ``[K+1]`` verify width.
        return self.query_tokens if query_tokens is None else max(1, int(query_tokens))

    def _hybrid_pool_receipt(
        self,
        tokens: int,
        *,
        num_requests: int = 1,
        query_tokens: int | None = None,
        phase: str = "decode",
    ) -> HybridPoolReceipt | None:
        """Resolve exact integer hybrid-pool allocation when a non-uniform regime applies."""
        if (
            self.paired_sfa
            or self.pages.groups
            or not self.case.model.layer_plans
            or int(tokens) <= 0
            or int(num_requests) <= 0
        ):
            return None
        q_tok = self._effective_cache_query_tokens(query_tokens)
        receipt = evaluate_hybrid_pool_receipt(
            self.case.model,
            self.case.runtime.parallel,
            int(tokens),
            num_requests=int(num_requests),
            runtime=self.case.runtime,
            draft_kind=self.draft_kind,
            query_tokens=q_tok,
            phase=phase,
        )
        if receipt.contract.pool_regime == REGIME_UNIFORM_INDEPENDENT:
            return None
        return receipt

    def _dp_cohort_extra_parts_by_stage(
        self,
        tokens: int,
        *,
        padded: bool = True,
        rank: int | None = None,
        query_tokens: int | None = None,
        phase: str = "decode",
        peer_count_override: int | None = None,
    ) -> list[dict[str, float]]:
        """Per-stage DP-cohort extra bytes split into ``(kv, index, draft)``."""
        p = self.case.runtime.parallel
        parts = [{"kv": 0.0, "index": 0.0, "draft": 0.0} for _ in range(p.pp)]
        dp_width = int(getattr(p, "dcp_dp_width", 1) or 1)
        if dp_width <= 1 or not self.case.model.layer_plans:
            return parts
        case, model = self.case, self.case.model
        is_prefill = str(phase or "decode").strip().lower() == "prefill"
        q_tok = 1 if is_prefill else self._effective_cache_query_tokens(query_tokens)
        interleave = cp_interleave(case.runtime)
        sharded_idx = dcp_indexer_sharded(model, case.runtime)
        owner_spans = self.rank_token_owners if rank is not None else None
        for layer in model.layer_plans:
            _, _, layer_dp = layer_dcp_factors(layer, p)
            if layer_dp <= 1:
                continue
            components = layer_cache_components(
                model, layer, p, tokens, capacity_padded=padded,
                interleave=interleave, rank=rank, owner_spans=owner_spans,
                runtime=case.runtime, query_tokens=q_tok, phase=phase,
            )
            covered = self.pages.covered.get(layer.index, ()) if padded else ()
            stage = self.layer_stage[layer.index]
            scale = float(peer_count_override if peer_count_override is not None else (layer_dp - 1))
            for name, value in components.items():
                if name in covered:
                    continue
                if name in _DCP_KV_SHARDED_COMPONENTS:
                    parts[stage]["kv"] += float(value) * scale
                elif name == "index_k" and sharded_idx:
                    parts[stage]["index"] += float(value) * scale
        if self.draft_kind and model.auxiliary is not None and not self.paired_sfa:
            if self._uses_inherited_draft_cache(model):
                from mindie_sim.catalog.draft import optional_draft_spec
                from mindie_sim.catalog.draft_geometry import draft_layer_plans

                draft = optional_draft_spec(model, self.draft_kind)
                draft_cap = self._draft_context_cap(draft)
                eff_tokens = min(int(tokens), draft_cap) if draft_cap > 0 else int(tokens)
                for d_plan in draft_layer_plans(model, self.draft_kind):
                    _, _, d_layer_dp = layer_dcp_factors(d_plan, p)
                    if d_layer_dp <= 1:
                        continue
                    d_comps = layer_cache_components(
                        model, d_plan, p, eff_tokens, capacity_padded=padded,
                        interleave=interleave, rank=rank, owner_spans=owner_spans,
                        runtime=case.runtime, query_tokens=q_tok, phase=phase,
                    )
                    d_scale = float(
                        peer_count_override if peer_count_override is not None else (d_layer_dp - 1)
                    )
                    for name, value in d_comps.items():
                        if name in _DCP_KV_SHARDED_COMPONENTS or (name == "index_k" and sharded_idx):
                            parts[-1]["draft"] += float(value) * d_scale
            else:
                draft_kv, draft_sharded = self._non_inherited_draft_cache_bytes(
                    model, tokens, padded=padded, rank=rank,
                )
                if draft_sharded and draft_kv > 0.0:
                    d_scale = float(
                        peer_count_override if peer_count_override is not None else (dp_width - 1)
                    )
                    parts[-1]["draft"] += draft_kv * d_scale
        return parts

    def _dp_cohort_extra_by_stage(
        self,
        tokens: int,
        *,
        padded: bool = True,
        rank: int | None = None,
        query_tokens: int | None = None,
        phase: str = "decode",
        peer_count_override: int | None = None,
    ) -> list[float]:
        """Extra per-DP-sequence KV bytes hosted on one die from ``dcp_dp_width > 1``."""
        parts = self._dp_cohort_extra_parts_by_stage(
            tokens, padded=padded, rank=rank, query_tokens=query_tokens,
            phase=phase, peer_count_override=peer_count_override,
        )
        return [float(st["kv"] + st["index"] + st["draft"]) for st in parts]

    def _rank_dp_cohort_cache_bytes(
        self,
        tokens: int,
        *,
        padded: bool = True,
        rank: int | None = None,
        query_tokens: int | None = None,
        phase: str = "decode",
    ) -> list[float]:
        """Unified physical-die cache bytes including DP-cohort KV and hybrid pools."""
        base = (
            list(self.cache_bytes(tokens, padded=padded, query_tokens=query_tokens, phase=phase))
            if rank is None
            else self._rank_cache_bytes(
                tokens, padded=padded, rank=int(rank), query_tokens=query_tokens, phase=phase,
            )
        )
        extra = self._dp_cohort_extra_by_stage(
            tokens, padded=padded, rank=rank, query_tokens=query_tokens, phase=phase,
        )
        return [float(b) + float(e) for b, e in zip(base, extra)]

    def dp_cohort_cache_bytes(
        self, tokens: int, *, padded: bool = True,
        query_tokens: int | None = None, phase: str = "decode",
    ) -> tuple[float, ...]:
        """Per-DP-sequence HBM bytes on one die conserving DP-cohort DCP capacity."""
        return tuple(self._rank_dp_cohort_cache_bytes(
            tokens, padded=padded, rank=None, query_tokens=query_tokens, phase=phase,
        ))

    def _sum_unsubsumed_stage_bytes(
        self,
        tokens: int,
        *,
        padded: bool,
        rank: int | None,
        query_tokens: int,
        hybrid: HybridPoolReceipt | None,
        phase: str = "decode",
    ) -> list[float]:
        """Evaluate stage cache bytes via hybrid pool (when active) plus outside allocations."""
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        if hybrid is not None:
            result = [float(v) for v in hybrid.total_pool_bytes_by_stage]
            sub_layers = set(hybrid.contract.subsumed_layers)
            sub_comps = set(hybrid.contract.subsumed_components)
            sub_draft_kv = bool(hybrid.contract.subsumed_draft_kv)
            interleave = int(hybrid.contract.normalized_block_size)
            block_override: int | None = interleave
        else:
            result = (
                self.pages.allocation_stage_bytes(tokens, case.runtime.max_model_len, p.pp)
                if padded else [0.0] * p.pp
            )
            sub_layers = set()
            sub_comps = set()
            sub_draft_kv = False
            interleave = cp_interleave(case.runtime)
            block_override = None
        owner_spans = self.rank_token_owners if rank is not None else None
        if model.layer_plans:
            for layer in model.layer_plans:
                components = layer_cache_components(
                    model, layer, p, tokens, capacity_padded=padded,
                    interleave=interleave, block_size=block_override,
                    rank=rank, owner_spans=owner_spans,
                    runtime=case.runtime, query_tokens=query_tokens, phase=phase,
                )
                covered = self.pages.covered.get(layer.index, ()) if padded else ()
                stage = self.layer_stage[layer.index]
                for name, value in components.items():
                    if name in covered:
                        continue
                    if hybrid is not None and int(layer.index) in sub_layers and name in sub_comps:
                        continue
                    result[stage] += float(value)
        mode = self.draft_kind
        if self.paired_sfa and mode and not padded:
            result[-1] += self._paired_draft_bytes(tokens, rank=rank, padded=False)
        elif mode and model.auxiliary is not None and not self.paired_sfa:
            if self._uses_inherited_draft_cache(model):
                draft_kv, draft_state = self._inherited_draft_cache_parts(
                    model, tokens, padded=padded, rank=rank,
                    query_tokens=query_tokens, phase=phase,
                )
                result[-1] += draft_state + (0.0 if sub_draft_kv else draft_kv)
            elif not sub_draft_kv:
                draft_kv, _ = self._non_inherited_draft_cache_bytes(
                    model, tokens, padded=padded, rank=rank,
                )
                result[-1] += draft_kv
        return result

    def cache_bytes(self, tokens: int, *, padded=True,
                    query_tokens: int | None = None,
                    phase: str = "decode") -> tuple[float, ...]:
        is_prefill = str(phase or "decode").strip().lower() == "prefill"
        q_tok = 1 if is_prefill else self._effective_cache_query_tokens(query_tokens)
        hybrid = (
            self._hybrid_pool_receipt(tokens, num_requests=1, query_tokens=q_tok, phase=phase)
            if padded else None
        )
        return tuple(self._sum_unsubsumed_stage_bytes(
            tokens, padded=bool(padded), rank=None, query_tokens=q_tok,
            hybrid=hybrid, phase=phase,
        ))

    def _draft_context_cap(self, draft) -> int:
        from mindie_sim.catalog.draft_spec import effective_draft_context_cap
        return effective_draft_context_cap(draft, self.case.runtime)

    def _resolve_draft_tp(self) -> int:
        """Resolve draft tensor-parallel degree strictly from contract/runtime metadata."""
        rt = self.case.runtime
        meta = dict(getattr(rt, "metadata", ()) or ())
        contract = getattr(rt, "speculative_contract", None) or meta.get("speculative_contract")
        if contract is not None:
            raw_tp = (
                contract.get("draft_tp")
                if isinstance(contract, Mapping)
                else getattr(contract, "draft_tp", None)
            )
            if raw_tp is not None and not isinstance(raw_tp, bool) and int(raw_tp) > 0:
                return int(raw_tp)
        meta_tp = meta.get("draft_tp")
        if meta_tp is not None and not isinstance(meta_tp, bool) and int(meta_tp) > 0:
            return int(meta_tp)
        spec_cfg = meta.get("speculative_config")
        if isinstance(spec_cfg, Mapping):
            cfg_tp = spec_cfg.get("draft_tensor_parallel_size") or spec_cfg.get("draft_tp")
            if cfg_tp is not None and not isinstance(cfg_tp, bool) and int(cfg_tp) > 0:
                return int(cfg_tp)
        return max(1, int(rt.parallel.tp))

    def _draft_owned_rows(
        self,
        eff_tokens: int,
        block: int,
        draft_cp: int,
        *,
        padded: bool,
        rank: int | None,
    ) -> int:
        """Exact integer token rows owned by one rank for draft KV under CP/T05."""
        tokens_int = max(0, int(eff_tokens))
        blk = max(1, int(block))
        cp = max(1, int(draft_cp))
        if cp <= 1:
            return int(block_round(tokens_int, blk) if padded else tokens_int)
        if rank is not None and self.rank_token_owners is not None:
            base_tokens = block_round(tokens_int, blk) if padded else tokens_int
            owned = owned_token_rows(base_tokens, self.rank_token_owners, int(rank))
            return int(owned) if owned is not None else 0
        if rank is not None:
            cp_rank = int(rank) % cp
            if padded:
                blocks = _ceil_div(tokens_int, blk) if tokens_int else 0
                count = ((blocks - 1 - cp_rank) // cp + 1) if cp_rank < blocks else 0
                return int(count * blk)
            return int(owned_positions(0, tokens_int - 1, cp, blk, cp_rank))
        if padded:
            return int(_context_shard_blocks(tokens_int, blk, cp))
        return int(_ceil_div(tokens_int, cp))

    def _non_inherited_draft_cache_bytes(
        self, model, tokens: int, *, padded: bool, rank: int | None = None,
    ) -> tuple[float, bool]:
        """Return ``(draft_kv_bytes, draft_is_dcp_sharded)`` honoring draft TP/quant/CP/cap."""
        from mindie_sim.catalog.draft import optional_draft_spec

        case, p = self.case, self.case.runtime.parallel
        draft = optional_draft_spec(model, self.draft_kind)
        draft_cap = self._draft_context_cap(draft)
        eff_tokens = min(int(tokens), draft_cap) if draft_cap > 0 else int(tokens)
        block = max(1, int(cp_interleave(case.runtime)))
        draft_tp = self._resolve_draft_tp()
        row_bytes = _draft_row_bytes_int(model, draft, tp=draft_tp)
        draft_sharded = _draft_is_dcp_sharded(
            model, p, self.draft_kind,
            dcp=int(getattr(p, "cp", 1) or 1),
            draft_page=block * row_bytes,
        )
        draft_cp = int(p.cp) if draft_sharded else 1
        rows = self._draft_owned_rows(eff_tokens, block, draft_cp, padded=padded, rank=rank)
        return float(rows * row_bytes * int(draft.layers)), draft_sharded

    def _uses_inherited_draft_cache(self, model) -> bool:
        """True when draft KV/state follows the declared draft LayerPlans."""
        from mindie_sim.catalog.cache import uses_inherited_draft_cache
        return (not self.paired_sfa
                and uses_inherited_draft_cache(model, self.draft_kind))

    def _inherited_draft_cache_parts(
        self, model, tokens: int, *, padded: bool, rank: int | None = None,
        query_tokens: int | None = None, phase: str = "decode",
    ) -> tuple[float, float]:
        """Draft KV and recurrent state from inherited plans, honoring context cap."""
        from mindie_sim.catalog.cache_ownership import draft_cache_components
        from mindie_sim.catalog.draft import optional_draft_spec

        draft = optional_draft_spec(model, self.draft_kind)
        draft_cap = self._draft_context_cap(draft)
        eff_tokens = min(tokens, draft_cap) if draft_cap > 0 else tokens
        is_prefill = str(phase or "decode").strip().lower() == "prefill"
        q_tok = 1 if is_prefill else self._effective_cache_query_tokens(query_tokens)
        values = draft_cache_components(
            model, self.case.runtime.parallel, draft_kind=self.draft_kind,
            sequence_tokens=eff_tokens, capacity_padded=padded,
            runtime=self.case.runtime, rank=rank,
            owner_spans=self.rank_token_owners if rank is not None else None,
            query_tokens=q_tok, phase=phase,
        )
        return float(values.get("draft_kv", 0.0)), float(values.get("draft_state", 0.0))

    def _inherited_draft_cache_bytes(
        self, model, tokens: int, *, padded: bool, rank: int | None = None,
        query_tokens: int | None = None,
    ) -> float:
        """Draft cache payload from the inherited plans, with T05 ownership."""
        draft_kv, draft_state = self._inherited_draft_cache_parts(
            model, tokens, padded=padded, rank=rank, query_tokens=query_tokens,
        )
        return draft_kv + draft_state

    def _group_cache_parts(
        self, tokens: int, rank: int | None = None,
        *, query_tokens: int | None = None, phase: str = "decode",
    ) -> list[dict]:
        """Known physical and valid components inside declared page groups.

        Token-paged components use the group's allocated token slots for
        capacity and its valid/window span for content (both ordinary page
        groups and paired SFA groups).  With ``rank`` the valid components are
        that physical rank's owner rows, while every rank still holds the full
        allocated page slots.
        """
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        stages = [
            {"alloc_payload": 0.0, "alloc_scale": 0.0,
             "valid_payload": 0.0, "valid_scale": 0.0, "components": {}}
            for _ in range(p.pp)
        ]
        is_prefill = str(phase or "decode").strip().lower() == "prefill"
        q_tok = 1 if is_prefill else self._effective_cache_query_tokens(query_tokens)
        by_index = {plan.index: plan for plan in model.layer_plans}
        block = max(1, int(getattr(case.runtime, "kv_block_size", 0) or cp_interleave(case.runtime)))
        for group in self.pages.groups:
            owned = self.pages.group_components(group)
            valid_tokens = group.valid_tokens(tokens, case.runtime.max_model_len)
            alloc_tokens = group.allocated_token_slots(tokens, case.runtime.max_model_len)
            history_pages = max(
                0,
                group.pages(tokens, case.runtime.max_model_len) - int(group.recycling_pages),
            )
            paired_local_alloc = (
                history_pages * block
                if (self.paired_sfa and not group.auxiliary and group.fixed_pages <= 0)
                else 0
            )
            for layer_index in group.layers:
                layer = by_index[layer_index]
                components = layer_cache_components(
                    model, layer, p, valid_tokens, capacity_padded=False,
                    interleave=cp_interleave(case.runtime),
                    rank=rank, owner_spans=self.rank_token_owners,
                    runtime=case.runtime, query_tokens=q_tok, phase=phase,
                )
                stage = stages[self.layer_stage[layer.index]]
                for name in owned.get(layer_index, ()):
                    part = None
                    if not self.paired_sfa and name in {"gqa_kv", "mla_kv", "window"}:
                        part = self._group_token_component(
                            model, layer, p, group, name, valid_tokens,
                            alloc_tokens, components, phase=phase,
                        )
                    elif self.paired_sfa and name in {"mla_kv", "index_k"} and paired_local_alloc > 0:
                        part = self._paired_group_token_component(
                            model, layer, p, name, paired_local_alloc, components, phase=phase,
                        )
                    if part is None:
                        value = components.get(name, 0.0)
                        if value <= 0.0:
                            continue
                        payload, scale = cache_component_split(
                            model, layer, p, name, value, phase=phase)
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
            alloc_draft = self._paired_draft_bytes(tokens, rank=rank, padded=True)
            valid_draft = self._paired_draft_bytes(tokens, rank=rank, padded=False)
            stage = stages[-1]
            stage["alloc_payload"] += alloc_draft
            stage["valid_payload"] += valid_draft
            stage["components"]["draft_kv"] = {
                "alloc_payload": alloc_draft, "alloc_scale": 0.0,
                "valid_payload": valid_draft, "valid_scale": 0.0,
            }
        return stages

    def _paired_group_token_component(
        self, model, layer, parallel, name: str, alloc_local_rows: int,
        components: dict[str, float], *, phase: str = "decode",
    ) -> tuple[float, float, float, float] | None:
        """Separate allocated local slots from rank-local valid rows in paired SFA."""
        if name == "mla_kv":
            if layer.attention != "mla" or layer.cache_kind != "dense":
                return None
            payload_b, scale_b = cache_row_split(
                model, "mla_kv", layer.cache_row_elements, plan=layer, phase=phase,
            )
        elif name == "index_k":
            if not layer.owns_index_k:
                return None
            p_b, s_b, pad_b = index_row_parts(model, layer.index_dim, plan=layer, phase=phase)
            payload_b, scale_b = float(p_b), float(s_b + pad_b)
        else:
            return None
        valid_value = float(components.get(name, 0.0))
        if valid_value > 0.0:
            valid_payload, valid_scale = cache_component_split(
                model, layer, parallel, name, valid_value, phase=phase,
            )
        else:
            valid_payload = valid_scale = 0.0
        return (
            float(alloc_local_rows) * payload_b,
            float(alloc_local_rows) * scale_b,
            valid_payload,
            valid_scale,
        )

    def _paired_draft_bytes(
        self, tokens: int, *, rank: int | None = None, padded: bool = False,
    ) -> float:
        """Draft KV bytes inside paired SFA honoring draft TP, exact integer CP, and cap."""
        from mindie_sim.catalog.draft import optional_draft_spec
        from mindie_sim.catalog.paired_sfa import draft_row

        p = self.case.runtime.parallel
        draft = optional_draft_spec(self.case.model, self.draft_kind)
        draft_cap = self._draft_context_cap(draft)
        max_len = int(self.case.runtime.max_model_len or 0)
        eff_tokens = min(int(tokens), max_len) if max_len > 0 else int(tokens)
        eff_tokens = min(eff_tokens, draft_cap) if draft_cap > 0 else eff_tokens
        draft_tp = self._resolve_draft_tp()
        row_bytes = int(draft_row(self.case.model, draft, tp=draft_tp))
        block = max(1, int(self.case.runtime.kv_block_size or cp_interleave(self.case.runtime)))
        sharded = _draft_is_dcp_sharded(
            self.case.model, p, self.draft_kind,
            dcp=int(getattr(p, "cp", 1) or 1),
            draft_page=block * row_bytes,
        )
        draft_cp = int(p.cp) if sharded else 1
        rows = self._draft_owned_rows(eff_tokens, block, draft_cp, padded=padded, rank=rank)
        return float(rows * row_bytes * int(draft.layers))

    def _group_token_component(
        self, model, layer, parallel, group, name: str, valid_tokens: int,
        alloc_tokens: int, components: dict[str, float], *, phase: str = "decode",
    ):
        """Rows/bytes for a token-paged component inside one group page."""
        del group
        if name == "gqa_kv":
            if (layer.attention not in {"gqa", "projection_only"}
                    or 0 <= layer.kv_owner_layer_id < layer.index):
                return None
            if name not in components and valid_tokens > 0:
                return None
            heads = local_kv_heads(layer.num_kv_heads, parallel.tp)
            key_fmt, val_fmt, is_shared = gqa_kv_formats(model, plan=layer, phase=phase)
            if is_shared:
                payload, scale = cache_row_split(
                    model, "gqa_kv", 2 * heads * layer.head_dim, plan=layer, phase=phase,
                )
            else:
                side = heads * layer.head_dim
                kp, ks, kpad = key_fmt.row_parts(side)
                vp, vs, vpad = val_fmt.row_parts(side)
                payload, scale = float(kp + vp), float(ks + kpad + vs + vpad)
        elif name == "mla_kv":
            if layer.attention != "mla" or layer.cache_kind != "dense":
                return None
            if name not in components and valid_tokens > 0:
                return None
            payload, scale = cache_row_split(
                model, "mla_kv", layer.cache_row_elements, plan=layer, phase=phase,
            )
        else:  # window
            if layer.attention != "mla" or not layer.window:
                return None
            if name not in components and valid_tokens > 0:
                return None
            payload, scale = cache_row_split(
                model, "window", layer.cache_row_elements, plan=layer, phase=phase,
            )
        alloc_rows = alloc_tokens
        valid_value = components.get(name, 0.0)
        if valid_value > 0.0:
            valid_payload, valid_scale = cache_component_split(
                model, layer, parallel, name, valid_value, phase=phase)
        else:
            valid_payload = valid_scale = 0.0
        return (alloc_rows * payload, alloc_rows * scale,
                valid_payload, valid_scale)

    def _kv_residency(
        self, tokens: int, *, query_tokens: int | None = None,
        phase: str = "decode",
    ) -> tuple[list[float], list[float], list[float]]:
        """(gross, HBM-resident, staging) KV bytes, same basis as resize."""
        gross = self._rank_dp_cohort_cache_bytes(
            tokens, padded=True, rank=None, query_tokens=query_tokens, phase=phase,
        )
        if self.kvc_hbm_ratio == 1.0:
            return gross, list(gross), [0.0] * len(gross)
        resident = [value * self.kvc_hbm_ratio for value in gross]
        staging = [
            value * (1.0 - self.kvc_hbm_ratio)
            for value in self._max_layer_cache_bytes(
                tokens, query_tokens=query_tokens, phase=phase,
            )
        ]
        return gross, resident, staging

    def _cache_component_breakdown(
        self, tokens: int, *, padded: bool, skip_covered: bool,
        rank: int | None = None, query_tokens: int | None = None,
        block_size_override: int | None = None, phase: str = "decode",
    ) -> list[dict]:
        """Per-stage component payload/scale split for one liveness view."""
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        is_prefill = str(phase or "decode").strip().lower() == "prefill"
        q_tok = 1 if is_prefill else self._effective_cache_query_tokens(query_tokens)
        interleave = (
            int(block_size_override)
            if block_size_override is not None and int(block_size_override) > 0
            else cp_interleave(case.runtime)
        )
        stages = [
            {"payload": 0.0, "scale": 0.0, "total": 0.0, "components": {}}
            for _ in range(p.pp)
        ]
        for layer in model.layer_plans:
            components = layer_cache_components(
                model, layer, p, tokens, capacity_padded=padded,
                interleave=interleave, block_size=block_size_override,
                rank=rank, owner_spans=self.rank_token_owners,
                runtime=case.runtime, query_tokens=q_tok, phase=phase,
            )
            covered = self.pages.covered.get(layer.index, ()) if skip_covered else ()
            stage = stages[self.layer_stage[layer.index]]
            for name, value in components.items():
                if name in covered:
                    continue
                payload, scale = cache_component_split(
                    model, layer, p, name, value, phase=phase)
                stage["payload"] += payload
                stage["scale"] += scale
                stage["total"] += float(value)
                detail = stage["components"].setdefault(
                    name, {"payload": 0.0, "scale": 0.0, "total": 0.0})
                detail["payload"] += payload
                detail["scale"] += scale
                detail["total"] += float(value)
        mode = self.draft_kind
        if mode and model.auxiliary is not None and not self.paired_sfa:
            draft_state = 0.0
            if self._uses_inherited_draft_cache(model):
                value, draft_state = self._inherited_draft_cache_parts(
                    model, tokens, padded=padded, rank=rank,
                    query_tokens=q_tok, phase=phase,
                )
            else:
                value, _ = self._non_inherited_draft_cache_bytes(
                    model, tokens, padded=padded, rank=rank)
            stage = stages[-1]
            stage["payload"] += float(value) + float(draft_state)
            stage["total"] += float(value) + float(draft_state)
            detail = stage["components"].setdefault(
                "draft_kv", {"payload": 0.0, "scale": 0.0, "total": 0.0})
            detail["payload"] += float(value)
            detail["total"] += float(value)
            if draft_state > 0.0:
                s_detail = stage["components"].setdefault(
                    "draft_state", {"payload": 0.0, "scale": 0.0, "total": 0.0})
                s_detail["payload"] += float(draft_state)
                s_detail["total"] += float(draft_state)
        return stages

    def _rank_cache_bytes(
        self, tokens: int, *, padded: bool, rank: int,
        query_tokens: int | None = None, phase: str = "decode",
    ) -> list[float]:
        """Allocation bytes for one physical rank.

        With an explicit T05 owner partition or CP > 1, each rank owns its exact
        rows.  Paired SFA rank-local unpadded queries resolve per-rank valid
        bytes while padded pool quotes retain per-rank shared group pages.
        """
        if self.paired_sfa and padded:
            return list(self.cache_bytes(
                tokens, padded=True, query_tokens=query_tokens, phase=phase,
            ))
        if not self.paired_sfa and (self.rank_token_owners is None and int(self.case.runtime.parallel.cp) <= 1):
            return list(self.cache_bytes(
                tokens, padded=padded, query_tokens=query_tokens, phase=phase,
            ))
        is_prefill = str(phase or "decode").strip().lower() == "prefill"
        q_tok = 1 if is_prefill else self._effective_cache_query_tokens(query_tokens)
        hybrid = (
            self._hybrid_pool_receipt(tokens, num_requests=1, query_tokens=q_tok, phase=phase)
            if padded else None
        )
        return self._sum_unsubsumed_stage_bytes(
            tokens, padded=bool(padded), rank=int(rank), query_tokens=q_tok,
            hybrid=hybrid, phase=phase,
        )

    def _rank_max_layer_cache_bytes(
        self, tokens: int, rank: int, *, query_tokens: int | None = None,
        phase: str = "decode",
    ) -> tuple[float, ...]:
        if self.rank_token_owners is None and int(self.case.runtime.parallel.cp) <= 1:
            return self._max_layer_cache_bytes(
                tokens, query_tokens=query_tokens, phase=phase,
            )
        model, p = self.case.model, self.case.runtime.parallel
        is_prefill = str(phase or "decode").strip().lower() == "prefill"
        q_tok = 1 if is_prefill else self._effective_cache_query_tokens(query_tokens)
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            comps = layer_cache_components(
                model, layer, p, tokens, capacity_padded=False,
                interleave=cp_interleave(self.case.runtime),
                rank=int(rank), owner_spans=self.rank_token_owners,
                runtime=self.case.runtime, query_tokens=q_tok, phase=phase,
            )
            st = self.layer_stage[layer.index]
            result[st] = max(result[st], float(sum(comps.values())))
        return tuple(result)

    def _kv_residency_ranks(
        self, tokens: int, *, query_tokens: int | None = None,
        phase: str = "decode",
    ) -> tuple[list[list[float]], list[list[float]]]:
        """(HBM-resident, staging) per stage and physical rank."""
        p = self.case.runtime.parallel
        if self.rank_token_owners is None and int(getattr(p, "cp", 1) or 1) <= 1:
            _gross, resident, staging = self._kv_residency(
                tokens, query_tokens=query_tokens, phase=phase,
            )
            return (
                [[float(resident[stage])] * self.width for stage in range(p.pp)],
                [[float(staging[stage])] * self.width for stage in range(p.pp)],
            )
        resident = [[0.0] * self.width for _ in range(p.pp)]
        staging = [[0.0] * self.width for _ in range(p.pp)]
        for rank in range(self.width):
            gross = self._rank_dp_cohort_cache_bytes(
                tokens, padded=True, rank=rank, query_tokens=query_tokens, phase=phase,
            )
            layer_max = self._rank_max_layer_cache_bytes(
                tokens, rank, query_tokens=query_tokens, phase=phase,
            )
            for stage in range(p.pp):
                if self.kvc_hbm_ratio == 1.0:
                    resident[stage][rank] = float(gross[stage])
                    staging[stage][rank] = 0.0
                else:
                    resident[stage][rank] = float(gross[stage]) * self.kvc_hbm_ratio
                    staging[stage][rank] = (
                        float(layer_max[stage]) * (1.0 - self.kvc_hbm_ratio))
        return resident, staging

    def _kv_residency_cohort(
        self, replica: int, tokens: int, *, query_tokens: int | None = None,
        phase: str = "decode",
    ) -> tuple[list[list[list[float]]], list[list[list[float]]], dict[int, tuple[tuple[float, ...], ...]], dict[int, tuple[tuple[float, ...], ...]]]:
        """Single-request physical (HBM-resident, staging) across its cross-DP DCP cohort."""
        p = self.case.runtime.parallel
        dp_width = max(1, int(getattr(p, "dcp_dp_width", 1) or 1))
        cohort_start = (int(replica) // dp_width) * dp_width
        cohort_end = min(int(p.dp), cohort_start + dp_width)
        res_all = [[[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)]
        stg_all = [[[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)]
        res_map: dict[int, tuple[tuple[float, ...], ...]] = {}
        stg_map: dict[int, tuple[tuple[float, ...], ...]] = {}
        for peer_rep in range(cohort_start, cohort_end):
            peer_res = [[0.0] * self.width for _ in range(p.pp)]
            peer_stg = [[0.0] * self.width for _ in range(p.pp)]
            for rank in range(self.width):
                cohort_rk = (peer_rep - cohort_start) * p.pp * self.width + rank
                if peer_rep == int(replica):
                    gross = self._rank_cache_bytes(
                        tokens, padded=True, rank=cohort_rk,
                        query_tokens=query_tokens, phase=phase,
                    )
                else:
                    gross = self._dp_cohort_extra_by_stage(
                        tokens, padded=True, rank=cohort_rk,
                        query_tokens=query_tokens, phase=phase, peer_count_override=1,
                    )
                layer_max = (
                    self._rank_max_layer_cache_bytes(
                        tokens, cohort_rk, query_tokens=query_tokens, phase=phase,
                    )
                    if self.kvc_hbm_ratio < 1.0 else (0.0,) * p.pp
                )
                for stage in range(p.pp):
                    if self.kvc_hbm_ratio == 1.0:
                        peer_res[stage][rank] = float(gross[stage])
                    else:
                        peer_res[stage][rank] = float(gross[stage]) * self.kvc_hbm_ratio
                        peer_stg[stage][rank] = float(layer_max[stage]) * (1.0 - self.kvc_hbm_ratio)
            res_all[peer_rep] = peer_res
            stg_all[peer_rep] = peer_stg
            res_map[peer_rep] = tuple(tuple(row) for row in peer_res)
            stg_map[peer_rep] = tuple(tuple(row) for row in peer_stg)
        return res_all, stg_all, res_map, stg_map

    def _allocation_view(
        self, tokens: int, *, rank: int | None = None,
        include_components: bool = True,
    ) -> dict:
        """One physical rank's (or the legacy stage's) cache allocation view."""
        case, p = self.case, self.case.runtime.parallel
        hybrid = self._hybrid_pool_receipt(tokens, num_requests=1)
        eff_block = hybrid.contract.normalized_block_size if hybrid is not None else None
        allocated = self._cache_component_breakdown(
            tokens, padded=True, skip_covered=True, rank=rank,
            block_size_override=eff_block,
        )
        valid = self._cache_component_breakdown(
            tokens, padded=False, skip_covered=True, rank=rank,
            block_size_override=eff_block,
        )
        group_parts = self._group_cache_parts(tokens, rank=rank)
        group_payload = self.pages.stage_bytes(
            tokens, case.runtime.max_model_len, p.pp)
        group_metadata = self.pages.metadata_stage_bytes(
            tokens, case.runtime.max_model_len, p.pp)
        group_access_metadata = self.pages.access_metadata_stage_bytes(
            tokens, case.runtime.max_model_len, p.pp)
        hybrid_reserve = (
            [float(v) for v in hybrid.stride_reserve_by_stage]
            if hybrid is not None else [0.0] * p.pp
        )
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
            reserve = (
                max(0.0, group_payload[index] - known_payload - known_scale)
                + hybrid_reserve[index]
            )
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
        p = self.case.runtime.parallel
        stage_world = max(1, int(p.dp)) * max(1, int(p.tp)) * max(1, int(p.pcp))
        rank_count = (max(self.width, stage_world)
                      if (self.rank_token_owners is None and p.cp_spans_dp)
                      else self.width)
        if self.rank_token_owners is None and int(p.cp) <= 1:
            view = self._allocation_view(tokens, rank=None)
            return tuple(dict(view) for _ in range(rank_count))
        return tuple(
            self._allocation_view(tokens, rank=rank)
            for rank in range(rank_count)
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

    def cache_pool_breakdown(
        self, tokens: int, *, padded: bool = True, query_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Pool-structured cache breakdown conserving ``cache_bytes(tokens, padded=padded)``."""
        case, model, p = self.case, self.case.model, self.case.runtime.parallel
        q_tok = self._effective_cache_query_tokens(query_tokens)
        hybrid = self._hybrid_pool_receipt(tokens, num_requests=1, query_tokens=q_tok) if padded else None
        eff_block = hybrid.contract.normalized_block_size if hybrid is not None else None
        stages = self._cache_component_breakdown(
            tokens, padded=padded, skip_covered=padded, query_tokens=q_tok,
            block_size_override=eff_block,
        )
        hybrid_reserve = (
            [float(v) for v in hybrid.stride_reserve_by_stage]
            if hybrid is not None else [0.0] * p.pp
        )
        group_parts = (
            self._group_cache_parts(tokens, query_tokens=q_tok)
            if padded and self.pages.groups else None
        )
        group_payload = (
            self.pages.stage_bytes(tokens, case.runtime.max_model_len, p.pp)
            if padded else [0.0] * p.pp
        )
        group_metadata = (
            self.pages.metadata_stage_bytes(tokens, case.runtime.max_model_len, p.pp)
            if padded else [0.0] * p.pp
        )
        by_component: dict[str, float] = {}
        stage_totals: list[float] = []
        pools = {
            "target_kv_bytes": 0.0,
            "indexer_k_bytes": 0.0,
            "recurrent_state_bytes": 0.0,
            "draft_kv_bytes": 0.0,
            "draft_state_bytes": 0.0,
            "page_reserve_bytes": 0.0,
            "page_metadata_bytes": 0.0,
        }
        for index in range(p.pp):
            stage_sum = 0.0
            merged: dict[str, float] = {}
            for name, detail in stages[index]["components"].items():
                merged[name] = merged.get(name, 0.0) + float(detail["total"])
            if group_parts is not None:
                gp = group_parts[index]
                for name, detail in gp["components"].items():
                    amount = float(detail["alloc_payload"]) + float(detail["alloc_scale"])
                    merged[name] = merged.get(name, 0.0) + amount
                reserve = max(
                    0.0,
                    float(group_payload[index]) - float(gp["alloc_payload"]) - float(gp["alloc_scale"]),
                )
            else:
                reserve = (float(group_payload[index]) if padded else 0.0) + hybrid_reserve[index]
            meta_bytes = float(group_metadata[index]) if padded else 0.0
            if self.paired_sfa and self.draft_kind and not padded:
                draft_val = float(self._paired_draft_bytes(tokens))
                if index == p.pp - 1:
                    merged["draft_kv"] = merged.get("draft_kv", 0.0) + draft_val
            for name, val in merged.items():
                by_component[name] = by_component.get(name, 0.0) + val
                stage_sum += val
                pool_key = _COMPONENT_POOL_MAP.get(name, "page_reserve_bytes")
                pools[pool_key] += val
            pools["page_reserve_bytes"] += reserve
            pools["page_metadata_bytes"] += meta_bytes
            stage_totals.append(stage_sum + reserve + meta_bytes)
        expected = list(self.cache_bytes(tokens, padded=padded, query_tokens=q_tok))
        if any(abs(a - b) > max(1.0, abs(b) * 1e-9) for a, b in zip(stage_totals, expected)):
            raise ValueError("memory_cache_pool_breakdown_not_conserved")
        draft_cap = 0
        if self.draft_kind and model.auxiliary is not None:
            from mindie_sim.catalog.draft import optional_draft_spec
            draft_cap = self._draft_context_cap(optional_draft_spec(model, self.draft_kind))
        engram_host = tuple(getattr(self, "engram_host_dram_bytes", (0.0,) * p.pp))
        return {
            "sequence_tokens": int(tokens),
            "padded": bool(padded),
            "query_tokens": int(q_tok),
            "draft_context_cap": int(draft_cap),
            "engram_cpu_offload": bool(case.runtime.engram_cpu_offload),
            "engram_host_dram_bytes": float(max(engram_host, default=0.0)),
            "by_component": by_component,
            "stage_totals_bytes": stage_totals,
            "total_bytes": float(sum(stage_totals)),
            **pools,
        }

    def _max_layer_cache_bytes(
        self, tokens: int, *, query_tokens: int | None = None,
        phase: str = "decode",
    ) -> tuple[float, ...]:
        model, p = self.case.model, self.case.runtime.parallel
        is_prefill = str(phase or "decode").strip().lower() == "prefill"
        q_tok = 1 if is_prefill else self._effective_cache_query_tokens(query_tokens)
        result = [0.0] * p.pp
        for layer in model.layer_plans:
            comps = layer_cache_components(
                model, layer, p, tokens, capacity_padded=False,
                interleave=cp_interleave(self.case.runtime),
                runtime=self.case.runtime, query_tokens=q_tok, phase=phase)
            st = self.layer_stage[layer.index]
            result[st] = max(result[st], float(sum(comps.values())))
        return tuple(result)

    def host_dram_cache_bytes(
        self, tokens: int, *, padded: bool = True, query_tokens: int | None = None,
        phase: str = "decode",
    ) -> float:
        total = 0.0
        if self.sparse_offload:
            block_size = int(self.sparse_offload["block_size"])
            eff_tokens = block_round(tokens, block_size) if padded else tokens
            offload_layers = len(self.sparse_offload["offload_layers"])
            total += float(eff_tokens * int(self.sparse_offload["kv_row_bytes"]) * offload_layers)
        ddr_ratio = float(self.offload.get("kvc_ddr_ratio", 0.0) or 0.0)
        if ddr_ratio > 0.0:
            p = self.case.runtime.parallel
            dp_width = max(1, int(getattr(p, "dcp_dp_width", 1) or 1))
            if dp_width > 1:
                cohort_sum = 0.0
                for peer_idx in range(dp_width):
                    peer_local = 0.0
                    for rk in range(self.width):
                        cohort_rk = peer_idx * p.pp * self.width + rk
                        st_vals = (
                            self._rank_cache_bytes(
                                tokens, padded=padded, rank=cohort_rk,
                                query_tokens=query_tokens, phase=phase,
                            )
                            if peer_idx == 0
                            else self._dp_cohort_extra_by_stage(
                                tokens, padded=padded, rank=cohort_rk,
                                query_tokens=query_tokens, phase=phase,
                                peer_count_override=1,
                            )
                        )
                        peer_local += float(sum(st_vals))
                    cohort_sum += peer_local / float(self.width)
                total += cohort_sum * ddr_ratio
            else:
                gross_stages = self.cache_bytes(
                    tokens, padded=padded, query_tokens=query_tokens, phase=phase,
                )
                total += float(sum(gross_stages)) * ddr_ratio
        return total

    def pd_handoff_volume_and_stage_hbm(self, tokens: int) -> tuple[float, tuple[float, ...]]:
        """Compute SfaRemoteD2HConnector wire volume (D2D index_k + 1x RD2H mla_kv) and stage HBM write."""
        p = self.case.runtime.parallel
        cohort_width = p.tp * p.pcp * max(1, int(getattr(p, "dcp_dp_width", 1) or 1))
        if not self.sparse_offload or self.sparse_offload["options"].get("keep_device_kv_cache"):
            local = self.cache_bytes(tokens, padded=False, query_tokens=1)
            return sum(local) * cohort_width, local
        offload_set = set(self.sparse_offload["offload_layers"])
        topk_buf = int(self.sparse_offload["topk_buffer_size"])
        use_copy_sfa = bool(self.sparse_offload["options"].get("use_fused_copy_sfa", True))
        hot_tokens = min(tokens, topk_buf) if use_copy_sfa else 0
        stage_hbm = [0.0] * p.pp
        rd2h_bytes = 0.0
        for layer in self.case.model.layer_plans:
            components = layer_cache_components(
                self.case.model, layer, p, tokens, capacity_padded=False,
                interleave=cp_interleave(self.case.runtime),
                runtime=self.case.runtime, query_tokens=1, phase="prefill",
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
        wire_volume = sum(stage_hbm) * cohort_width + rd2h_bytes
        return wire_volume, tuple(stage_hbm)
