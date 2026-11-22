"""Hierarchical expert and KV cache offload with prefetch masking.

[Role] Expose validated policies, state access records and engine transfer scheduling.
[Invariants] Tier traffic preserves rank ownership, transfer latency and state access identity.
[Timeouts & Locks] Bounded by model layers and active iteration work. No I/O, waits or locks.
[Side-Effect Policy] Analysis is in-memory. Scheduling submits work to the shared engine.
"""
from __future__ import annotations

from .policy import (
    OffloadPolicy,
    missing_scenario_error,
    offload_resources,
    resolve_case_offload_policy,
    resolve_sku_host_dma_rates,
    _PREFETCH_VERSIONS as _PREFETCH_VERSIONS,
    _nonnegative as _nonnegative,
    _offload_memory_for as _offload_memory_for,
    _positive as _positive,
    _ratio as _ratio,
    _resolve_policy as _resolve_policy,
    _submit_offload_transfer as _submit_offload_transfer,
    _utilization as _utilization,
)
from .expert import (
    is_routed_expert_entry_op as is_routed_expert_entry_op,
    schedule_expert_layer_offload,
    _active_experts_per_rank as _active_experts_per_rank,
    _expected_active_experts as _expected_active_experts,
    _moe_geometry as _moe_geometry,
    _resolve_active_experts_from_op as _resolve_active_experts_from_op,
)
from .accesses import (
    StateAccess,
    project_state_tiers,
    _access_bytes as _access_bytes,
    _accesses_from_ops as _accesses_from_ops,
    _as_access as _as_access,
    _as_accesses as _as_accesses,
    _normalize_span as _normalize_span,
    _physical_producer_output as _physical_producer_output,
    _unique_accesses as _unique_accesses,
)
from .events import (
    bind_offload_access_events,
    release_offload_access,
    _access_registry as _access_registry,
    _accesses_for_work as _accesses_for_work,
    _event_for_access as _event_for_access,
    _event_table as _event_table,
    _has_event_bindings as _has_event_bindings,
    _merge_event_tables as _merge_event_tables,
    _register_access_events as _register_access_events,
    _registered_events as _registered_events,
    _work_event_table as _work_event_table,
)
from .kv import (
    is_kv_or_state_entry_op as is_kv_or_state_entry_op,
    schedule_draft_kv_writeback,
    schedule_kv_iteration_offload,
    schedule_kv_layer_offload,
    schedule_kv_layer_writeback,
    _STATE_WRITER_FAMILIES as _STATE_WRITER_FAMILIES,
    _submit_kv_tier_transfers as _submit_kv_tier_transfers,
    _submit_writeback_groups as _submit_writeback_groups,
    _writeback_groups as _writeback_groups,
    _writer_occurrences as _writer_occurrences,
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
