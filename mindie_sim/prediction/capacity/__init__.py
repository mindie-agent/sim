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
        _raw_draft_mode = dict(case.runtime.metadata).get("draft_mode")
        if not _raw_draft_mode and (
            case.runtime.dspark
            or case.runtime.mtp_acceptance_length > 1
            or any(value > 1 for _, value in case.runtime.mtp_acceptance_by_output_tokens)
            or dict(case.runtime.metadata).get("speculative")
        ):
            raise NotImplementedError(
                "LEGACY_PURGED: read draft_kind strictly from SpeculativeExecutionContract "
                "(catalog/draft_spec.py); zero silent ngram/speculative->mtp coercion"
            )
        self.draft_kind = str(_raw_draft_mode or "")
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
        self.slot_plans = None
        if (int(getattr(p, "redundant_experts_per_rank", 0) or 0) > 0
                or any(plan.ffn == "moe" and int(plan.num_experts) > 0
                       and int(plan.num_experts) % max(1, int(p.ep)) != 0
                       for plan in case.model.layer_plans)):
            from mindie_sim.catalog.expert_slots import expert_slot_plan
            from mindie_sim.schema.runtime_features import split_routing_features
            _logical, slot_routing = split_routing_features(
                dict(case.runtime.routing or {}))
            self.slot_plans = {
                plan.index: expert_slot_plan(
                    case.model, plan, p, routing=slot_routing)
                for plan in case.model.layer_plans
                if plan.ffn == "moe" and int(plan.num_experts) > 0
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
        self.gross_weights = stage_weight_bytes(
            case.model, p, case.machine.ranks_per_node, include_draft=draft,
            include_engram_tables=not case.runtime.engram_cpu_offload,
            include_vision=include_vision_opt,
            layer_partition=self.layers, draft_kind=self.draft_kind or None,
            enable_shared_expert_dp=shared_expert_dp,
            enable_pcp_o_proj_weight_sharding=pcp_o_proj_shard,
            dcp_q_replicate=dcp_q_rep,
            slot_plans=self.slot_plans,
        )
        weight_totals, weight_payloads = _weight_ledger_by_stage(
            case, self.layers, draft_kind=self.draft_kind,
            include_draft=draft,
            include_engram_tables=not case.runtime.engram_cpu_offload,
            include_vision=include_vision_opt,
            enable_shared_expert_dp=shared_expert_dp,
            enable_pcp_o_proj_weight_sharding=pcp_o_proj_shard,
            dcp_q_replicate=dcp_q_rep,
            slot_plans=self.slot_plans,
        )
        if any(abs(total - gross) > max(1.0, abs(gross) * 1e-9)
               for total, gross in zip(weight_totals, self.gross_weights)):
            raise ValueError("memory_weight_decomposition_inconsistent")
        resolved_ws_tokens = self._resolve_workspace_tokens(workspace_tokens)
        if self.offload["expert_ddr_ratio"] > 0.0:
            (self.routed_expert_bytes, self.routed_expert_payload_bytes,
             stage_staging_base) = _routed_expert_bytes_by_stage(
                case, self.layer_stage, workspace_tokens=max(1, resolved_ws_tokens),
                slot_plans=self.slot_plans,
            )
            self.expert_staging_workspace = [
                self.offload["expert_ddr_ratio"] * value for value in stage_staging_base
            ]
        else:
            self.routed_expert_bytes = [0.0] * p.pp
            self.routed_expert_payload_bytes = [0.0] * p.pp
            self.expert_staging_workspace = [0.0] * p.pp
        self.expert_hbm_saved = [
            self.offload["expert_ddr_ratio"] * value for value in self.routed_expert_bytes
        ]
        self.expert_payload_hbm_saved = [
            self.offload["expert_ddr_ratio"] * value
            for value in self.routed_expert_payload_bytes
        ]
        self.weights = [
            max(0.0, gross - saved)
            for gross, saved in zip(self.gross_weights, self.expert_hbm_saved)
        ]
        self.weights_payload = [
            max(0.0, payload - saved)
            for payload, saved in zip(weight_payloads, self.expert_payload_hbm_saved)
        ]
        self.weights_quant_metadata = [
            max(0.0, total - payload)
            for total, payload in zip(self.weights, self.weights_payload)
        ]
        if case.runtime.engram_cpu_offload and case.model.engram is not None:
            from mindie_sim.catalog.weights import engram_table_terms
            if case.model.quant is None:
                raise NotImplementedError(
                    "LEGACY_PURGED: require normalized case.model.quant instead of silent QuantSpec('bf16')"
                )
            quant = case.model.quant
            table_bytes = float(sum(
                term_bytes(term, quant, p, case.machine.ranks_per_node)
                for term in engram_table_terms(case.model)
            ))
            self.engram_host_dram_bytes = [table_bytes] * p.pp
        else:
            self.engram_host_dram_bytes = [0.0] * p.pp

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
            for stage in range(p.pp)
        ]
        self.fixed_per_rank = fixed_per_rank
        self.rank_capacity = [
            [
                [math.floor(die.hbm_bytes * case.runtime.gpu_memory_utilization)
                 - fixed_per_rank[stage] for die in self._stage_dies(replica, stage)]
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
        self.request_query_tokens: dict[tuple[int, str], int] = {}
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
