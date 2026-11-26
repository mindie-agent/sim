"""Placement-aware resident memory and dynamic request ownership.

[Role] Derive weight/cache geometry and account for live allocations.
[Invariants] Stored observations cannot overwrite geometry. Each replica owns
  its cache; PP stages and physical ranks retain separate capacities. A reserved
  pool and the bytes used inside it are not added twice. Cache allocation
  (payload/page tail/scale/page metadata) is reported separately from accesses.
  Every grant/selected/shared/reader holder owns an explicit physical backing
  (rank/stage/storage/tier/generation/range); one batch commits only when all
  physical-union candidates fit, and failed/admitted decisions never leak state.
[Timeouts & Locks] Bounded by layers and active requests, no I/O or locks.
[Side-Effect Policy] Mutates only this prediction's cache ledger.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from mindie_sim.catalog.cache import (
    cache_component_split,
    cache_row_split,
    layer_cache_components,
    local_kv_heads,
    owned_token_rows,
)
from mindie_sim.catalog.cache_pages import Allocation, AllocationPlan, CachePages
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes, term_payload_bytes
from mindie_sim.catalog.weights import (
    layer_weight_terms,
    stage_weight_bytes,
    stage_weight_terms,
)
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import dtype_bytes

from ..runtime.contracts import (
    AllocationBinding,
    AllocationKey,
    BackingKey,
    CapacityDomainState,
    EventKey,
    OccurrenceKey,
    PoolKey,
)
from ..runtime.ledger import Ledger
from .cache import _MemoryCacheMixin
from .contracts import (
    MemoryPolicy,
    _HoldSnapshot,
    _PhysicalRef,
    _normalise_owner_spans,
)
from .grants import _MemoryGrantsMixin
from .ledger import (
    DieResidentReceipt,
    RecoverSsmRingBreakdown,
    ResidentGrantEntry,
    ResidentMemoryReceipt,
    _MemoryLedgerMixin,
    estimate_recover_ssm_snapshot_ring_bytes,
    prepared_weight_bytes,
    recover_ssm_snapshot_ring_breakdown,
)
from .reporting import _MemoryReportingMixin
from .requests import _MemoryRequestsMixin
from .residency import (
    _MemoryWorkspaceMixin,
    _ep_rank_weight_signature,
    _model_has_expert_quant_overrides,
    _resolve_offload_ratios,
    _routed_expert_bytes_by_stage,
    _weight_ledger_by_stage,
)
from .selections import _MemorySelectionsMixin

__all__ = [
    "Allocation",
    "AllocationBinding",
    "AllocationKey",
    "AllocationPlan",
    "BackingKey",
    "CachePages",
    "CapacityDomainState",
    "CaseSpec",
    "DieResidentReceipt",
    "EventKey",
    "Ledger",
    "Mapping",
    "Memory",
    "MemoryPolicy",
    "OccurrenceKey",
    "PoolKey",
    "QuantSpec",
    "RecoverSsmRingBreakdown",
    "ResidentGrantEntry",
    "ResidentMemoryReceipt",
    "annotations",
    "cache_component_split",
    "cache_row_split",
    "dataclass",
    "dtype_bytes",
    "estimate_recover_ssm_snapshot_ring_bytes",
    "layer_cache_components",
    "layer_weight_terms",
    "local_kv_heads",
    "math",
    "owned_token_rows",
    "prepared_weight_bytes",
    "recover_ssm_snapshot_ring_breakdown",
    "replace",
    "stage_weight_bytes",
    "stage_weight_terms",
    "term_bytes",
    "term_payload_bytes",
]


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))


class Memory(
    _MemoryWorkspaceMixin,
    _MemoryCacheMixin,
    _MemoryReportingMixin,
    _MemoryLedgerMixin,
    _MemoryGrantsMixin,
    _MemorySelectionsMixin,
    _MemoryRequestsMixin,
):
    """Placement-aware memory model and allocation ledger for one prediction."""

    def __init__(self, case: CaseSpec, policy: MemoryPolicy | None = None, *,
                 workspace_tokens: int | None = None):
        self.case = case
        self.policy = policy or MemoryPolicy(**case.runtime.memory)
        p = case.runtime.parallel
        partition = case.runtime.pp_layer_partition
        self.layers = tuple(partition or (
            (case.model.layers // p.pp + (stage < case.model.layers % p.pp))
            for stage in range(p.pp)
        ))
        self.layer_stage = tuple(stage for stage, count in enumerate(self.layers) for _ in range(count))
        from mindie_sim.catalog.speculative_state import resolve_speculative_query_tokens
        _draft_meta = dict(case.runtime.metadata)
        _declared = int(_draft_meta.get("draft_proposal_tokens", 0) or 0)
        self.query_tokens = resolve_speculative_query_tokens(case.runtime)
        if dict(case.model.metadata).get("cache_allocator") == "paired_sfa" and _declared > 0:
            self.query_tokens = _declared + 1
        from mindie_sim.catalog.paired_sfa import page_groups
        paired_groups = page_groups(case, self.layer_stage, self.query_tokens)
        self.paired_sfa = bool(paired_groups)
        if paired_groups and self.policy.cache_page_groups:
            raise ValueError("paired_sfa_and_explicit_cache_page_groups_conflict")
        self.pages = CachePages(
            paired_groups or self.policy.cache_page_groups,
            {layer.index: set(layer_cache_components(
                case.model, layer, p, case.runtime.max_model_len,
                capacity_padded=False, runtime=case.runtime,
                query_tokens=self.query_tokens)) for layer in case.model.layer_plans},
            self.layer_stage,
        )
        from mindie_sim.catalog.cache_ownership import _active_draft_kind
        self.draft_kind = _active_draft_kind(case.model, case.runtime)
        draft = bool(self.draft_kind)
        self.offload = _resolve_offload_ratios(case, self.policy)
        self.kvc_hbm_ratio = max(
            0.0, 1.0 - self.offload["kvc_ddr_ratio"] - self.offload["kvc_ssd_ratio"]
        )
        self._init_weights(case, p, draft, workspace_tokens)
        self._init_workspace_capacity(case, p, workspace_tokens)
        self._init_allocation_state(case, p)

    def _init_weights(self, case, p, draft, workspace_tokens):
        meta = dict(case.runtime.metadata)
        extra = dict(meta.get("additional_config") or ()) if isinstance(meta.get("additional_config"), dict) else {}
        shared_expert_dp = bool(meta.get("enable_shared_expert_dp", False))
        pcp_o_proj_shard = bool(
            meta.get("enable_pcp_o_proj_weight_sharding",
                     extra.get("enable_pcp_o_proj_weight_sharding", False)))
        from mindie_sim.catalog.context_parallel import dcp_q_replicate_enabled
        has_sparse_mla = any(
            lp.indexer or lp.owns_topk or lp.topk_owner_layer_id >= 0 or lp.index_k_owner_layer_id >= 0
            for lp in case.model.layer_plans
        )
        dcp_q_rep = (
            int(p.cp) > 1
            and dcp_q_replicate_enabled(
                case.runtime,
                pcp_factor=int(p.dcp_pcp_factor),
                dp_width=int(p.dcp_dp_width),
                sparse=has_sparse_mla,
            )
        )
        self.enable_shared_expert_dp = shared_expert_dp
        self.enable_pcp_o_proj_weight_sharding = pcp_o_proj_shard
        self.dcp_q_replicate = dcp_q_rep
        self.slot_plans = None
        mixed_expert_quant = _model_has_expert_quant_overrides(case.model)
        if (int(getattr(p, "redundant_experts_per_rank", 0) or 0) > 0
                or mixed_expert_quant
                or any(plan.ffn == "moe" and int(plan.num_experts) > 0
                       and int(plan.num_experts) % max(1, int(p.ep)) != 0
                       for plan in case.model.layer_plans)):
            from mindie_sim.catalog.expert_slots import expert_slot_plan
            from mindie_sim.schema.runtime_features import split_routing_features
            _logical, slot_routing = split_routing_features(
                dict(case.runtime.routing or {}))
            moe_plans = [
                plan for plan in case.model.layer_plans
                if plan.ffn == "moe" and int(plan.num_experts) > 0
            ]
            if moe_plans:
                self.slot_plans = {
                    plan.index: expert_slot_plan(
                        case.model, plan, p, routing=slot_routing)
                    for plan in moe_plans
                }
        self.vision_accounting = str(
            meta.get("vision_accounting")
            or (
                "full_checkpoint"
                if (case.model.vision is not None and (
                    case.model.auxiliary is None or case.model.auxiliary.include_vision_weights
                ))
                else "text_backbone"
            )
        )
        include_vision_opt = self.vision_accounting == "full_checkpoint"
        resolved_ws_tokens = self._resolve_workspace_tokens(workspace_tokens)
        self._init_stage_weight_ledgers(
            case, p,
            draft=draft,
            include_vision_opt=include_vision_opt,
            shared_expert_dp=shared_expert_dp,
            pcp_o_proj_shard=pcp_o_proj_shard,
            dcp_q_rep=dcp_q_rep,
            resolved_ws_tokens=resolved_ws_tokens,
            mixed_expert_quant=mixed_expert_quant,
        )
        if case.runtime.engram_cpu_offload and case.model.engram is not None:
            from mindie_sim.catalog.draft_weights import require_model_quant
            from mindie_sim.catalog.weights import (
                engram_table_terms,
                is_ple_engram,
                ple_table_terms,
            )
            quant = require_model_quant(case.model)
            if is_ple_engram(case.model.engram):
                host_by_stage = [0.0] * p.pp
                for table, term in zip(case.model.engram.tables, ple_table_terms(case.model)):
                    stage = (
                        self.layer_stage[table.layer_id]
                        if 0 <= int(table.layer_id) < len(self.layer_stage)
                        else p.pp - 1
                    )
                    host_by_stage[stage] += float(
                        term_bytes(term, quant, p, case.machine.ranks_per_node)
                    )
                self.engram_host_dram_bytes = host_by_stage
            else:
                table_bytes = float(sum(
                    term_bytes(term, quant, p, case.machine.ranks_per_node)
                    for term in engram_table_terms(case.model)
                ))
                self.engram_host_dram_bytes = [table_bytes] * p.pp
        else:
            self.engram_host_dram_bytes = [0.0] * p.pp
        self.c8_static_channel_scale_by_stage = self._resolve_c8_static_channel_scale_by_stage(
            case, p,
        )

    def _compute_single_ep_weight_ledger(
        self,
        case,
        p,
        *,
        ep_rank: int | None,
        draft: bool,
        include_vision_opt: bool,
        shared_expert_dp: bool,
        pcp_o_proj_shard: bool,
        dcp_q_rep: bool,
        resolved_ws_tokens: int,
    ) -> dict[str, list[float]]:
        gross = stage_weight_bytes(
            case.model, p, case.machine.ranks_per_node, include_draft=draft,
            include_engram_tables=not case.runtime.engram_cpu_offload,
            include_vision=include_vision_opt,
            layer_partition=self.layers, draft_kind=self.draft_kind or None,
            enable_shared_expert_dp=shared_expert_dp,
            enable_pcp_o_proj_weight_sharding=pcp_o_proj_shard,
            dcp_q_replicate=dcp_q_rep,
            slot_plans=self.slot_plans,
            ep_rank=ep_rank,
            runtime=case.runtime,
        )
        totals, payloads = _weight_ledger_by_stage(
            case, self.layers, draft_kind=self.draft_kind,
            include_draft=draft,
            include_engram_tables=not case.runtime.engram_cpu_offload,
            include_vision=include_vision_opt,
            enable_shared_expert_dp=shared_expert_dp,
            enable_pcp_o_proj_weight_sharding=pcp_o_proj_shard,
            dcp_q_replicate=dcp_q_rep,
            slot_plans=self.slot_plans,
            ep_rank=ep_rank,
        )
        if any(abs(total - gross_v) > max(1.0, abs(gross_v) * 1e-9)
               for total, gross_v in zip(totals, gross)):
            raise ValueError("memory_weight_decomposition_inconsistent")
        off_ratio = self.offload["expert_ddr_ratio"]
        if off_ratio > 0.0:
            routed, routed_pay, stage_stg = _routed_expert_bytes_by_stage(
                case, self.layer_stage, workspace_tokens=max(1, resolved_ws_tokens),
                slot_plans=self.slot_plans, ep_rank=ep_rank,
            )
            stg_ws = [off_ratio * v for v in stage_stg]
        else:
            routed = [0.0] * p.pp
            routed_pay = [0.0] * p.pp
            stg_ws = [0.0] * p.pp
        saved = [off_ratio * v for v in routed]
        saved_pay = [off_ratio * v for v in routed_pay]
        net = [max(0.0, g - s) for g, s in zip(gross, saved)]
        net_pay = [max(0.0, pay - sp) for pay, sp in zip(payloads, saved_pay)]
        net_meta = [max(0.0, tot - pay) for tot, pay in zip(net, net_pay)]
        return {
            "gross": gross, "net": net, "payload": net_pay, "meta": net_meta,
            "routed": routed, "routed_payload": routed_pay,
            "saved": saved, "saved_payload": saved_pay, "staging": stg_ws,
        }

    def _init_stage_weight_ledgers(
        self,
        case,
        p,
        *,
        draft: bool,
        include_vision_opt: bool,
        shared_expert_dp: bool,
        pcp_o_proj_shard: bool,
        dcp_q_rep: bool,
        resolved_ws_tokens: int,
        mixed_expert_quant: bool,
    ) -> None:
        ep = max(1, int(p.ep))
        if self.slot_plans is not None and ep > 1:
            from mindie_sim.catalog.weight_layer_terms import _layer_resolved_quant
            layer_quants = (
                {plan.index: _layer_resolved_quant(case.model, plan)
                 for plan in case.model.layer_plans if plan.index in self.slot_plans}
                if mixed_expert_quant else None
            )
            sig_cache: dict[tuple, dict[str, list[float]]] = {}
            per_ep: list[dict[str, list[float]]] = []
            for rank_i in range(ep):
                sig = _ep_rank_weight_signature(
                    case.model, self.slot_plans, rank_i,
                    mixed_quant=mixed_expert_quant, layer_quants=layer_quants,
                )
                entry = sig_cache.get(sig)
                if entry is None:
                    entry = self._compute_single_ep_weight_ledger(
                        case, p, ep_rank=rank_i, draft=draft,
                        include_vision_opt=include_vision_opt,
                        shared_expert_dp=shared_expert_dp,
                        pcp_o_proj_shard=pcp_o_proj_shard,
                        dcp_q_rep=dcp_q_rep,
                        resolved_ws_tokens=resolved_ws_tokens,
                    )
                    sig_cache[sig] = entry
                per_ep.append(entry)
            derive_ws = bool(self.policy.derive_workspace)
            worst_ep = tuple(
                max(
                    range(ep),
                    key=lambda r: (
                        per_ep[r]["net"][st] + (per_ep[r]["staging"][st] if derive_ws else 0.0),
                        per_ep[r]["net"][st],
                    ),
                )
                for st in range(p.pp)
            )
            self.bottleneck_ep_rank_by_stage = worst_ep
            self.gross_weights_by_ep_rank = tuple(list(row["gross"]) for row in per_ep)
            self.weights_by_ep_rank = tuple(list(row["net"]) for row in per_ep)
            self.expert_staging_by_ep_rank = tuple(list(row["staging"]) for row in per_ep)
            self.gross_weights = [per_ep[worst_ep[st]]["gross"][st] for st in range(p.pp)]
            self.weights = [per_ep[worst_ep[st]]["net"][st] for st in range(p.pp)]
            self.weights_payload = [per_ep[worst_ep[st]]["payload"][st] for st in range(p.pp)]
            self.weights_quant_metadata = [per_ep[worst_ep[st]]["meta"][st] for st in range(p.pp)]
            self.routed_expert_bytes = [per_ep[worst_ep[st]]["routed"][st] for st in range(p.pp)]
            self.routed_expert_payload_bytes = [per_ep[worst_ep[st]]["routed_payload"][st] for st in range(p.pp)]
            self.expert_hbm_saved = [per_ep[worst_ep[st]]["saved"][st] for st in range(p.pp)]
            self.expert_payload_hbm_saved = [per_ep[worst_ep[st]]["saved_payload"][st] for st in range(p.pp)]
            self.expert_staging_workspace = [
                per_ep[worst_ep[st]]["staging"][st] for st in range(p.pp)
            ]
            return

        single = self._compute_single_ep_weight_ledger(
            case, p, ep_rank=(0 if self.slot_plans is not None else None), draft=draft,
            include_vision_opt=include_vision_opt,
            shared_expert_dp=shared_expert_dp,
            pcp_o_proj_shard=pcp_o_proj_shard,
            dcp_q_rep=dcp_q_rep,
            resolved_ws_tokens=resolved_ws_tokens,
        )
        self.bottleneck_ep_rank_by_stage = (0,) * p.pp
        self.gross_weights_by_ep_rank = (list(single["gross"]),)
        self.weights_by_ep_rank = (list(single["net"]),)
        self.expert_staging_by_ep_rank = (list(single["staging"]),)
        self.gross_weights = single["gross"]
        self.weights = single["net"]
        self.weights_payload = single["payload"]
        self.weights_quant_metadata = single["meta"]
        self.routed_expert_bytes = single["routed"]
        self.routed_expert_payload_bytes = single["routed_payload"]
        self.expert_hbm_saved = single["saved"]
        self.expert_payload_hbm_saved = single["saved_payload"]
        self.expert_staging_workspace = single["staging"]

    def _ep_rank_for_die(self, replica: int, stage: int, local_rank: int, placement=None) -> int:
        from mindie_sim.schema.common import ModelingError
        from mindie_sim.schema.parallel import ReplicaLocalRank

        p = self.case.runtime.parallel
        ep = max(1, int(p.ep))
        if ep <= 1 or len(self.weights_by_ep_rank) <= 1:
            return 0
        rep_i = int(replica)
        stg_i = int(stage)
        lane_i = int(local_rank)
        lane = ReplicaLocalRank(lane_i)
        if placement is not None:
            try:
                return int(placement.coord_for(rep_i, stg_i, lane).ep_rank)
            except ModelingError:
                pass
        if p.rank_table is None:
            coords = p.ep_group_coords(rep_i, stg_i, lane)
            return int(coords.index((rep_i, stg_i, lane)))
        logical_rank = (rep_i * int(p.pp) + stg_i) * int(p.tp) * int(p.pcp) + lane_i
        for group in p.rank_table:
            if logical_rank in group:
                return int(tuple(int(item) for item in group).index(logical_rank))
        return lane_i % ep

    def _resolve_c8_static_channel_scale_by_stage(self, case, p) -> tuple[int, ...]:
        """Resolve fixed per-die load-time GQA static C8/FAQuant scale/offset planes."""
        from mindie_sim.catalog.cache import DEFAULT_CACHE_BLOCK_SIZE
        from mindie_sim.catalog.cache_format import (
            FAQUANT_KV_TAG,
            _declared_kv_tag,
            dense_c8_kv_planes,
            gqa_kv_formats,
            resolve_layer_cache_abis,
        )

        model = case.model
        pp = max(1, int(p.pp))
        stage_bytes = [0] * pp
        if not model.layer_plans:
            return tuple(stage_bytes)
        is_faquant = _declared_kv_tag(model).upper().replace("-", "_") == FAQUANT_KV_TAG
        allowed_freq = {"checkpoint", "layer"} if is_faquant else {"checkpoint"}
        block = max(1, int(getattr(case.runtime, "kv_block_size", 0) or DEFAULT_CACHE_BLOCK_SIZE))
        meta = dict(getattr(model, "metadata", ()) or ())
        quant = getattr(model, "quant", None)
        scale_dtype = str(
            meta.get("kv_scale_dtype")
            or (getattr(quant, "kv_scale_dtype", "") if quant is not None else "")
            or "fp32"
        ).strip().lower()
        for plan in model.layer_plans:
            if plan.attention not in {"gqa", "projection_only"}:
                continue
            key_fmt, val_fmt, _shared = gqa_kv_formats(model, plan=plan)
            is_static_c8 = (
                (key_fmt.dtype == "int8" and key_fmt.scale_dtype == "none")
                or (val_fmt.dtype == "int8" and val_fmt.scale_dtype == "none")
            )
            if not is_static_c8:
                continue
            stage = self.layer_stage[plan.index] if plan.index < len(self.layer_stage) else pp - 1
            if 0 <= int(plan.kv_owner_layer_id) < int(plan.index):
                if is_faquant:
                    heads = local_kv_heads(int(plan.num_kv_heads or model.kv_heads), int(p.tp))
                    q_heads = max(1, int(plan.num_heads or model.heads) // max(1, int(p.tp)))
                    head_dim = int(plan.head_dim or model.head_dim)
                    if heads > 0 and head_dim > 0:
                        q_planes = dense_c8_kv_planes(
                            heads, block, head_dim,
                            scale_dtype=scale_dtype, faquant=True, num_q_heads=q_heads,
                        )
                        stage_bytes[stage] += sum(
                            int(pl.bytes_per_unit)
                            for pl in q_planes
                            if pl.allocation_frequency == "layer" and pl.write_trigger == "load"
                        )
                continue
            abis = resolve_layer_cache_abis(
                model, plan, p, runtime=case.runtime, stage=stage,
            )
            per_layer = sum(
                int(pl.bytes_per_unit)
                for abi in abis
                for pl in abi.planes
                if pl.allocation_frequency in allowed_freq and pl.write_trigger == "load"
            )
            stage_bytes[stage] += per_layer
        return tuple(stage_bytes)

    def _init_workspace_capacity(self, case, p, workspace_tokens):
        async_enabled = bool(case.runtime.async_scheduling)
        self.async_delayed_release_steps = (
            (1 if async_enabled else 0) if self.policy.async_delayed_release_steps is None
            else max(0, int(self.policy.async_delayed_release_steps))
        )
        self.async_in_flight_lookahead_steps = (
            (1 if async_enabled else 0) if self.policy.async_in_flight_lookahead_steps is None
            else max(0, int(self.policy.async_in_flight_lookahead_steps))
        )
        self.async_workspace_buffer_multiplier = (
            (1.0 if async_enabled else 0.0) if self.policy.async_workspace_buffer_multiplier is None
            else max(0.0, float(self.policy.async_workspace_buffer_multiplier))
        )
        copies = self.policy.concurrent_workspace_copies
        if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
            raise ValueError("memory_concurrent_workspace_copies_requires_positive_integer")
        self.concurrent_workspace_copies = copies
        temporary = self.policy.temporary_dequant_bytes
        if (isinstance(temporary, bool) or not isinstance(temporary, (int, float))
                or not math.isfinite(float(temporary)) or float(temporary) < 0.0):
            raise ValueError("memory_temporary_dequant_bytes_requires_nonnegative")
        self.temporary_dequant_bytes = float(temporary)
        width = p.tp * p.pcp
        self.async_workspace = (
            self._async_inflight_workspace(workspace_tokens)
            if self.policy.derive_workspace and self.async_workspace_buffer_multiplier > 0
            else [0.0] * p.pp
        )
        base_workspace = (self._workspace(workspace_tokens) if self.policy.derive_workspace
                          else [0.0] * p.pp)
        self.workspace_base = [value * copies for value in base_workspace]
        staging_workspace = (self.expert_staging_workspace if self.policy.derive_workspace
                             else [0.0] * p.pp)
        self.workspace_expert_staging = list(staging_workspace)
        self.sparse_offload = dict(case.runtime.metadata).get("sparse_kv_offload")
        static_offload = (
            list(self.sparse_offload.get("static_npu_bytes_per_stage_rank", (0.0,) * p.pp))
            if self.sparse_offload else [0.0] * p.pp
        )
        self.workspace_static_offload = list(static_offload)
        self.workspace = [
            base + extra + stage_buf + offload
            for base, extra, stage_buf, offload in zip(
                self.workspace_base, self.async_workspace, staging_workspace, static_offload
            )
        ]
        self.temporary_dequant = [self.temporary_dequant_bytes] * p.pp
        self.width = max(1, int(width))
        owner_placement = self.policy.rank_token_owners
        if owner_placement is None:
            metadata = dict(case.runtime.metadata)
            owner_placement = (metadata.get("rank_token_owners")
                               or metadata.get("token_owner_partition"))
        self.rank_token_owners = _normalise_owner_spans(
            owner_placement, self.width)
        fixed_per_rank = [
            self.weights[stage] + self.policy.reserved + self.workspace[stage]
            + self.temporary_dequant[stage]
            + float(self.c8_static_channel_scale_by_stage[stage])
            for stage in range(p.pp)
        ]
        self.fixed_per_rank = fixed_per_rank
        placement = None
        if len(self.weights_by_ep_rank) > 1:
            from mindie_sim.schema.common import ModelingError
            try:
                placement = case.machine.validate_placement(p)
            except ModelingError:
                placement = None
        rank_ep_ids = [
            [
                [
                    self._ep_rank_for_die(replica, stage, local_rank, placement)
                    for local_rank in range(len(self._stage_dies(replica, stage)))
                ]
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
        self.rank_weights = [
            [
                [
                    self.weights_by_ep_rank[rank_ep_ids[replica][stage][local_rank]][stage]
                    for local_rank in range(len(rank_ep_ids[replica][stage]))
                ]
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
        self.rank_workspace = [
            [
                [
                    self.workspace_base[stage] + self.async_workspace[stage]
                    + (
                        self.expert_staging_by_ep_rank[rank_ep_ids[replica][stage][local_rank]][stage]
                        if self.policy.derive_workspace else 0.0
                    )
                    + self.workspace_static_offload[stage]
                    for local_rank in range(len(rank_ep_ids[replica][stage]))
                ]
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
        self.rank_capacity = [
            [
                [
                    die.hbm_bytes * case.runtime.gpu_memory_utilization
                    - (
                        self.rank_weights[replica][stage][local_rank]
                        + self.policy.reserved
                        + self.rank_workspace[replica][stage][local_rank]
                        + self.temporary_dequant[stage]
                        + float(self.c8_static_channel_scale_by_stage[stage])
                    )
                    for local_rank, die in enumerate(self._stage_dies(replica, stage))
                ]
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
        self._base_rank_capacity = [
            [list(row) for row in replica_ranks]
            for replica_ranks in self.rank_capacity
        ]
        self.capacity = [
            [min(row) for row in replica_ranks]
            for replica_ranks in self.rank_capacity
        ]

    def _init_allocation_state(self, case, p):
        self.peak = [[0.0] * p.pp for _ in range(p.dp)]
        self.peak_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.live = [[0.0] * p.pp for _ in range(p.dp)]
        self.request_tokens: dict[tuple[int, str], int] = {}
        self.request_valid_tokens: dict[tuple[int, str], int] = {}
        self.request_phases: dict[tuple[int, str], str] = {}
        self.request_query_tokens: dict[tuple[int, str], int] = {}
        self._request_hold_tokens: dict[tuple[int, str], dict[str, tuple[int, int, str]]] = {}
        self._prepared_weights: dict[str, object] = {}
        self._prepared_persistent_by_stage = [0.0] * p.pp
        self._prepared_load_peak_by_stage = [0.0] * p.pp
        self.live_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.staging_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.selected_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.shared_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.granted_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.reader_rank = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        self.owners: dict[tuple[int, str], tuple[float, ...]] = {}
        self.owner_ranks: dict[
            tuple[int, str], tuple[tuple[float, ...], ...]] = {}
        self.logical_owners: dict[tuple[int, str], tuple[float, ...]] = {}
        self.logical_owner_ranks: dict[
            tuple[int, str], tuple[tuple[float, ...], ...]] = {}
        self.logical_host_owners: dict[tuple[int, str], float] = {}
        self.logical_staging_owners: dict[tuple[int, str], tuple[float, ...]] = {}
        self.logical_staging_rank: dict[
            tuple[int, str], tuple[tuple[float, ...], ...]] = {}
        self.staging_owners: dict[tuple[int, str], tuple[float, ...]] = {}
        self.staging_owner_ranks: dict[
            tuple[int, str], tuple[tuple[float, ...], ...]] = {}
        self._cohort_owner_ranks: dict[
            tuple[int, str], dict[int, tuple[tuple[float, ...], ...]]] = {}
        self._cohort_staging_ranks: dict[
            tuple[int, str], dict[int, tuple[tuple[float, ...], ...]]] = {}
        self.staging_live = [0.0] * p.dp
        self.async_consumer_holds: dict[
            tuple[int, str], dict[str, _HoldSnapshot]] = {}
        self.retired_requests: set[tuple[int, str]] = set()
        self.request_versions: dict[tuple[int, str], int] = {}
        self.grants: dict[tuple[int, str], tuple[float, ...]] = {}
        self.granted = [[0.0] * p.pp for _ in range(p.dp)]
        self.grant_bindings: dict[tuple[int, str], dict] = {}
        self.effect_bindings: dict[tuple[int, str], dict] = {}
        self.onchip_effects: dict[tuple[int, str], dict] = {}
        self.granted_host = [0.0] * p.dp
        self.grant_hosts: dict[tuple[int, str], float] = {}
        self.grant_records: dict[tuple[int, str], dict] = {}
        self.grant_backings: dict[tuple, dict] = {}
        self.grant_readers: dict[
            tuple[int, str], dict[str, dict]] = {}
        self.budget_version = 0
        self.release_events: list[dict] = []
        self.selected_plans: dict[tuple[int, int], AllocationPlan] = {}
        self.selected_items: dict[tuple[int, int], dict[str, Allocation]] = {}
        self.selected_reserved = [[0.0] * p.pp for _ in range(p.dp)]
        self.selected_released: set[tuple[int, int, str]] = set()
        self.selected_physical: dict[tuple[int, int, str], float] = {}
        self.peak_combined = [[0.0] * p.pp for _ in range(p.dp)]
        self.shared_pages: dict[tuple[int, str], dict] = {}
        self.shared_reserved = [[0.0] * p.pp for _ in range(p.dp)]
        self.delayed_owners: dict[tuple[int, str], tuple[int, tuple[float, ...]]] = {}
        self.peak_delayed_release = [[0.0] * p.pp for _ in range(p.dp)]
        self.peak_kv_resident = [[0.0] * p.pp for _ in range(p.dp)]
        self.peak_kv_staging = [[0.0] * p.pp for _ in range(p.dp)]
        self.host_dram_capacity = [
            float(self.sparse_offload["usable_dram_bytes_per_dp"]) if self.sparse_offload else math.inf
            for _ in range(p.dp)
        ]
        self.host_dram_live = [0.0 for _ in range(p.dp)]
        self.host_dram_peak = [0.0 for _ in range(p.dp)]
        self.host_dram_owners: dict[tuple[int, str], float] = {}
        #: Unified physical holder ledger.  Key: (replica, kind, name).
        self._refs: dict[
            tuple[int, str, str], tuple[_PhysicalRef, ...]] = {}
        #: Canonical RuntimeState capacity domain once adopted.  The detached
        #: domain is constructed here and adopted without copying its payload.
        self.runtime_state = None
        self.ledger_domain: CapacityDomainState | None = None
        self._typed_grants: dict[
            tuple[int, str], tuple[AllocationKey, ...]] = {}
        self._ledger_serial = 0
        self._ledger_generations: dict[object, int] = {}
