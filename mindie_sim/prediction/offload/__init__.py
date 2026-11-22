"""Hierarchical expert and KV cache offload with prefetch masking.

[Role] Expose validated policies, state access records and engine transfer scheduling.
[Invariants] Tier traffic preserves rank ownership, transfer latency and state access identity.
[Timeouts & Locks] Bounded by model layers and active iteration work. No I/O, waits or locks.
[Side-Effect Policy] Analysis is in-memory. Scheduling submits work to the shared engine.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any
import weakref

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import analyze_network_efficiency
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec

from .policy import (
    OffloadPolicy,
    missing_scenario_error,
    offload_resources,
    resolve_case_offload_policy,
    resolve_sku_host_dma_rates,
    _PREFETCH_VERSIONS,
    _nonnegative,
    _offload_memory_for,
    _positive,
    _ratio,
    _resolve_policy,
    _submit_offload_transfer,
    _utilization,
)
from .expert import (
    is_routed_expert_entry_op,
    schedule_expert_layer_offload,
    _active_experts_per_rank,
    _expected_active_experts,
    _moe_geometry,
    _resolve_active_experts_from_op,
)
from .accesses import (
    StateAccess,
    project_state_tiers,
    _access_bytes,
    _accesses_from_ops,
    _as_access,
    _as_accesses,
    _normalize_span,
    _physical_producer_output,
    _unique_accesses,
)
from .events import (
    bind_offload_access_events,
    release_offload_access,
    _access_registry,
    _accesses_for_work,
    _event_for_access,
    _event_table,
    _has_event_bindings,
    _merge_event_tables,
    _register_access_events,
    _registered_events,
    _work_event_table,
)
from .kv import (
    is_kv_or_state_entry_op,
    schedule_draft_kv_writeback,
    schedule_kv_iteration_offload,
    schedule_kv_layer_offload,
    schedule_kv_layer_writeback,
    _STATE_WRITER_FAMILIES,
    _submit_kv_tier_transfers,
    _submit_writeback_groups,
    _writeback_groups,
    _writer_occurrences,
)


_ATTENTION_FAMILIES = frozenset({
    "attention", "qkv_projection", "output_projection", "rotary", "cache_store",
    "gdn_gating", "short_convolution", "recurrent_vector",
})
_FFN_FAMILIES = frozenset({
    "grouped_gemm", "moe_ffn", "gate_up_projection", "down_projection", "swiglu",
    "situ", "router", "topk", "routing", "alltoall", "alltoallv", "projection",
})
_ATTENTION_RECIPES = frozenset({
    "q_a", "q_b", "kv_a", "kv_b", "qkv", "qkv_projection", "linear_qkvz",
    "linear_output", "q_proj", "kv_proj", "q_absorb", "v_expand", "q_flat",
    "k_flat", "v_flat", "qkv_concatenate", "qkv_split_materialize",
})
_ATTENTION_NORMS = frozenset({"q_norm", "kv_norm", "qk_norm", "q_a_norm", "kv_a_norm"})
_ATTENTION_COMMS = frozenset({"linear_reduce", "catalog_tp_allreduce", "output_reduce"})
_FFN_TOKENS = (
    "expert", "grouped", "moe", "ffn", "router", "route_", "topk", "gate_up",
    "dense_down", "dense_gate_up", "swiglu", "situ", "latent", "shared",
)


__all__ = [
    "OffloadPolicy",
    "missing_scenario_error", "resolve_sku_host_dma_rates",
    "resolve_case_offload_policy", "offload_resources", "schedule_expert_layer_offload",
    "schedule_kv_layer_offload", "schedule_kv_layer_writeback",
    "schedule_draft_kv_writeback", "schedule_kv_iteration_offload",
    "StateAccess", "project_state_tiers", "release_offload_access",
    "bind_offload_access_events",
]
