"""Governed Context Parallelism operator package.

[Role]
  Public package boundary for 1:1 vLLM-Ascend DCP, PCP, and PCP+DCP
  computation-graph lowering and paired-SFA recipe registration.

[Invariants]
  * Exports graph lowering helpers and ``recipes()`` from ``graph.py`` without
    introducing duplicate parallel structs or hidden state.
  * Useful attention FLOPs and KV bytes are conserved across participating ranks.

[Timeouts & Locks]
  Import-time re-exports only; zero I/O or locking.

[Side-Effect Policy]
  Pure module exports.
"""
from __future__ import annotations

from .graph import (
    ATTENTION_UPDATE_MAX_EFFECTIVE_SP,
    CP_CHUNKED_PREFILL_STREAM,
    CP_DECODE_COMM_STREAM,
    CP_WEIGHT_STREAM,
    DCP_LSE_MERGE_EAGER_RECIPE,
    DCP_LSE_MERGE_RECIPE,
    FP32_BYTES,
    PACK_LSE_OPS,
    PAIRED_FP8_QSFA_IMPLEMENTATION,
    active_layer_dcp,
    active_layer_dcp_factors,
    dcp_q_replicate_enabled,
    emit_dcp_output_exchange_combine,
    emit_dcp_query_gather,
    emit_mtp_current_attention,
    emit_pcp_o_proj_weight_allgather,
    emit_pcp_restore_unpermute,
    emit_sfa_index_remap,
    emit_sfa_prefill_kv_gather,
    emit_v41_pcp_c2_ring_gather,
    emit_v41_pcp_c2_ring_write,
    expand_dcp_lse_eager_kernels,
    is_fast_sfa_dcp_exchange,
    packed_sfa_geometry,
    pcp_activation_gather_scale,
    pcp_o_proj_weight_sharding_enabled,
    pcp_quantized_allgather_enabled,
    recipes,
    should_split_mtp_dcp,
)

__all__ = (
    "ATTENTION_UPDATE_MAX_EFFECTIVE_SP",
    "CP_CHUNKED_PREFILL_STREAM",
    "CP_DECODE_COMM_STREAM",
    "CP_WEIGHT_STREAM",
    "DCP_LSE_MERGE_EAGER_RECIPE",
    "DCP_LSE_MERGE_RECIPE",
    "FP32_BYTES",
    "PACK_LSE_OPS",
    "PAIRED_FP8_QSFA_IMPLEMENTATION",
    "active_layer_dcp",
    "active_layer_dcp_factors",
    "dcp_q_replicate_enabled",
    "emit_dcp_output_exchange_combine",
    "emit_dcp_query_gather",
    "emit_mtp_current_attention",
    "emit_pcp_o_proj_weight_allgather",
    "emit_pcp_restore_unpermute",
    "emit_sfa_index_remap",
    "emit_sfa_prefill_kv_gather",
    "emit_v41_pcp_c2_ring_gather",
    "emit_v41_pcp_c2_ring_write",
    "expand_dcp_lse_eager_kernels",
    "is_fast_sfa_dcp_exchange",
    "packed_sfa_geometry",
    "pcp_activation_gather_scale",
    "pcp_o_proj_weight_sharding_enabled",
    "pcp_quantized_allgather_enabled",
    "recipes",
    "should_split_mtp_dcp",
)
