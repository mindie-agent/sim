"""Deterministic HBM KV/state capacity and PD/EPD disaggregation planner.

[Role]
  Budget per-rank HBM capacity across TP/EP/PP/DCP shards, EPLB redundant
  experts, runtime workspace, ACLGraph bucket pools, HCCL communicator buffers,
  O(1)-in-seq_len recurrent/conv states (GDN/KDA/Mamba), and owner-deduplicated
  KV/Index blocks.  Optimize Prefill:Decode (P:D) and Encoder-Prefill-Decode
  (EPD) instance ratios under TTFT/TPOT SLOs and KV transfer bandwidth limits.

[Invariants]
  * Recurrent/conv state (GDN/KDA/Mamba) scales strictly with ``max_num_seqs``
    and NEVER with ``seq_len`` or ``max_model_len``.
  * Cross-layer KV/Index sharing allocates physical blocks ONLY for declared
    owner layers; consumer layers reference owner blocks with zero duplication.
  * Negative remaining HBM is reported explicitly as ``status="oom"``; it is
    never clamped to a positive block count.
  * Missing latency/bandwidth inputs return ``status="unmodeled"`` with an
    explicit ``unmodeled_reason``; no latency is fabricated.

[Timeouts & Locks]
  Pure deterministic arithmetic; zero subprocesses, network calls, or locks.

[Side-Effect Policy]
  Reads caller-provided mappings and catalog profiles only; never writes files.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .a3_types import A3InputError, dtype_bytes
from .model_catalog import (
    ModelArchitectureProfile,
    build_architecture_profile,
    load_architecture_profile,
)


class ServingPlannerError(ValueError):
    """Explicit validation failure in serving capacity or PD ratio planning."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _pos_float(value: Any, reason: str) -> float:
    if not _finite(value) or float(value) <= 0:
        raise ServingPlannerError(reason)
    return float(value)


def _nonneg_float(value: Any, reason: str) -> float:
    if not _finite(value) or float(value) < 0:
        raise ServingPlannerError(reason)
    return float(value)


def _pos_int(value: Any, reason: str) -> int:
    if isinstance(value, bool):
        raise ServingPlannerError(reason)
    try:
        num = int(value)
    except (TypeError, ValueError) as exc:
        raise ServingPlannerError(reason) from exc
    if num <= 0:
        raise ServingPlannerError(reason)
    return num


def _nonneg_int(value: Any, reason: str) -> int:
    if isinstance(value, bool):
        raise ServingPlannerError(reason)
    try:
        num = int(value)
    except (TypeError, ValueError) as exc:
        raise ServingPlannerError(reason) from exc
    if num < 0:
        raise ServingPlannerError(reason)
    return num


def _resolve_profile(spec: Mapping[str, Any]) -> ModelArchitectureProfile | None:
    profile = spec.get("architecture_profile") or spec.get("model_profile")
    if isinstance(profile, ModelArchitectureProfile):
        return profile
    if isinstance(profile, Mapping):
        try:
            return build_architecture_profile(profile)
        except A3InputError as exc:
            raise ServingPlannerError(str(exc)) from exc
    model_key = spec.get("model_key") or spec.get("model_preset")
    if isinstance(model_key, str) and model_key.strip():
        try:
            return load_architecture_profile(model_key.strip())
        except A3InputError as exc:
            raise ServingPlannerError(str(exc)) from exc
    return None


def plan_hbm_kv_capacity(
    spec: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Deterministically budget per-rank HBM and allocatable KV cache blocks.

    Subtracts TP/EP/PP sharded weights, EPLB redundant expert weights, runtime
    workspace, ACLGraph capture bucket pool overhead, HCCL communicator buffers,
    and O(1)-in-seq_len recurrent/conv states (GDN/KDA/Mamba), then divides
    remaining HBM by owner-deduplicated per-block KV bytes.
    """
    merged: dict[str, Any] = {}
    if spec is not None:
        if not isinstance(spec, Mapping):
            raise ServingPlannerError("capacity_spec_invalid")
        merged.update(spec)
    merged.update(kwargs)

    profile = _resolve_profile(merged)

    # 1. Usable HBM per rank
    if merged.get("hbm_capacity_bytes") is not None:
        hbm_capacity_bytes = _pos_float(merged["hbm_capacity_bytes"], "hbm_capacity_bytes_invalid")
    else:
        hbm_gib = _pos_float(merged.get("hbm_gib", 64.0), "hbm_gib_invalid")
        hbm_capacity_bytes = hbm_gib * (1024.0 ** 3)

    util = _pos_float(merged.get("gpu_memory_utilization", 0.90), "gpu_memory_utilization_invalid")
    if util > 1.0:
        raise ServingPlannerError("gpu_memory_utilization_exceeds_one")
    usable_hbm_bytes = hbm_capacity_bytes * util

    # 2. Parallel degrees
    parallel_raw = merged.get("parallel")
    p_map = dict(parallel_raw) if isinstance(parallel_raw, Mapping) else {}
    tp_size = _pos_int(merged.get("tp_size", p_map.get("tp", 1)), "tp_size_invalid")
    ep_size = _pos_int(merged.get("ep_size", p_map.get("ep", 1)), "ep_size_invalid")
    pp_size = _pos_int(merged.get("pp_size", p_map.get("pp", 1)), "pp_size_invalid")
    dp_size = _pos_int(merged.get("dp_size", p_map.get("dp", 1)), "dp_size_invalid")
    dcp_size = _pos_int(merged.get("dcp_size", p_map.get("dcp", 1)), "dcp_size_invalid")
    moe_tp_size = _pos_int(merged.get("moe_tp_size", p_map.get("moe_tp", 1 if ep_size > 1 else tp_size)), "moe_tp_size_invalid")

    if profile is not None:
        try:
            profile.validate_tp(tp_size)
        except A3InputError as exc:
            raise ServingPlannerError(str(exc)) from exc

    # 3. Model weights per rank (including EPLB redundant experts)
    eplb_redundant = _nonneg_int(merged.get("eplb_redundant_experts", 0), "eplb_redundant_experts_invalid")
    eplb_scope = str(merged.get("eplb_scope", "auto")).strip().lower()
    if eplb_redundant == 0:
        eplb_redundant_per_rank = 0
    elif eplb_scope == "per_rank":
        eplb_redundant_per_rank = eplb_redundant
    elif eplb_scope == "global":
        eplb_redundant_per_rank = math.ceil(eplb_redundant / ep_size)
    else:
        # Auto: if >= ep_size and ep_size > 1, treat as global redundant count across EP group;
        # if < ep_size, treat as per-rank redundant experts.
        eplb_redundant_per_rank = (
            math.ceil(eplb_redundant / ep_size)
            if (ep_size > 1 and eplb_redundant >= ep_size)
            else eplb_redundant
        )

    if merged.get("weight_bytes_per_rank") is not None:
        base_weight_bytes_per_rank = _nonneg_float(merged["weight_bytes_per_rank"], "weight_bytes_per_rank_invalid")
        single_expert_bytes = _nonneg_float(
            merged.get("single_expert_weight_bytes", 0.0), "single_expert_weight_bytes_invalid"
        )
        per_rank_expert_bytes = single_expert_bytes / (moe_tp_size * pp_size) if single_expert_bytes > 0 else 0.0
    elif profile is not None:
        dense_per_rank = profile.dense_weight_bytes / (tp_size * pp_size)
        local_routed = math.ceil(profile.num_routed_experts / ep_size) if profile.num_routed_experts > 0 else 0
        per_rank_expert_bytes = profile.single_expert_weight_bytes / (moe_tp_size * pp_size)
        routed_per_rank = local_routed * per_rank_expert_bytes
        base_weight_bytes_per_rank = dense_per_rank + routed_per_rank
    else:
        dense_total = _nonneg_float(merged.get("dense_weight_bytes", 0.0), "dense_weight_bytes_invalid")
        routed_total = _nonneg_float(merged.get("routed_expert_weight_bytes", 0.0), "routed_expert_weight_bytes_invalid")
        num_routed = _nonneg_int(merged.get("num_routed_experts", 0), "num_routed_experts_invalid")
        single_expert_bytes = _nonneg_float(
            merged.get(
                "single_expert_weight_bytes",
                (routed_total / num_routed) if (routed_total > 0 and num_routed > 0) else 0.0,
            ),
            "single_expert_weight_bytes_invalid",
        )
        if dense_total == 0.0 and routed_total == 0.0 and single_expert_bytes == 0.0:
            if merged.get("total_weight_bytes") is not None:
                total_w = _nonneg_float(merged["total_weight_bytes"], "total_weight_bytes_invalid")
                base_weight_bytes_per_rank = total_w / (tp_size * pp_size)
                per_rank_expert_bytes = 0.0
            else:
                return {
                    "status": "unmodeled",
                    "unmodeled_reason": "missing_weight_specification",
                    "allocatable_kv_blocks": None,
                }
        else:
            per_rank_expert_bytes = single_expert_bytes / (moe_tp_size * pp_size)
            if num_routed > 0 and single_expert_bytes > 0:
                routed_per_rank = math.ceil(num_routed / ep_size) * per_rank_expert_bytes
            else:
                routed_per_rank = routed_total / (ep_size * moe_tp_size * pp_size)
            base_weight_bytes_per_rank = dense_total / (tp_size * pp_size) + routed_per_rank

    if eplb_redundant_per_rank > 0 and per_rank_expert_bytes <= 0.0:
        raise ServingPlannerError("eplb_redundant_experts_requires_expert_weight_geometry")
    eplb_weight_bytes_per_rank = float(eplb_redundant_per_rank * per_rank_expert_bytes)
    total_weight_bytes_per_rank = base_weight_bytes_per_rank + eplb_weight_bytes_per_rank

    # 4. Runtime workspace, ACLGraph bucket pool, and HCCL communicator buffers
    max_num_seqs = _pos_int(merged.get("max_num_seqs", 64), "max_num_seqs_invalid")
    max_batched_tokens = _pos_int(merged.get("max_num_batched_tokens", 4096), "max_num_batched_tokens_invalid")
    seq_len = _pos_int(merged.get("seq_len", merged.get("max_model_len", 4096)), "seq_len_invalid")

    if merged.get("workspace_bytes") is not None:
        workspace_bytes_per_rank = _nonneg_float(merged["workspace_bytes"], "workspace_bytes_invalid")
    elif merged.get("workspace_mib") is not None:
        workspace_bytes_per_rank = _nonneg_float(merged["workspace_mib"], "workspace_mib_invalid") * (1024.0 ** 2)
    else:
        hidden_size = int(merged.get("hidden_size") or (profile.hidden_size if profile else 4096))
        act_dtype = str(merged.get("activation_dtype") or (profile.activation_dtype if profile else "bf16"))
        act_b = dtype_bytes(act_dtype)
        workspace_bytes_per_rank = float(max_batched_tokens * hidden_size * act_b * 2.0 + 64.0 * (1024.0 ** 2))

    aclgraph_bucket_count = _nonneg_int(merged.get("aclgraph_bucket_count", 0), "aclgraph_bucket_count_invalid")
    if merged.get("aclgraph_per_bucket_bytes") is not None:
        per_bucket_bytes = _nonneg_float(merged["aclgraph_per_bucket_bytes"], "aclgraph_per_bucket_bytes_invalid")
    else:
        per_bucket_mib = _nonneg_float(merged.get("aclgraph_per_bucket_mib", 32.0), "aclgraph_per_bucket_mib_invalid")
        per_bucket_bytes = per_bucket_mib * (1024.0 ** 2)
    aclgraph_pool_bytes_per_rank = float(aclgraph_bucket_count * per_bucket_bytes)

    default_hccl_mib = 200.0 if (tp_size * ep_size * pp_size * dp_size > 1) else 0.0
    hccl_buffsize_mib = _nonneg_float(
        merged.get("hccl_buffsize_mib", default_hccl_mib), "hccl_buffsize_mib_invalid"
    )
    hccl_comm_groups = _pos_int(merged.get("hccl_comm_groups", 1), "hccl_comm_groups_invalid")
    hccl_buffer_bytes_per_rank = float(hccl_buffsize_mib * hccl_comm_groups * (1024.0 ** 2))

    # 5. Recurrent / conv states (GDN / KDA / Mamba) — strictly O(max_num_seqs), O(1) in seq_len
    if merged.get("state_bytes_per_seq_per_rank") is not None:
        state_bytes_per_seq = _nonneg_float(
            merged["state_bytes_per_seq_per_rank"], "state_bytes_per_seq_per_rank_invalid"
        )
        recurrent_layer_count = _nonneg_int(
            merged.get("num_recurrent_layers", 1 if state_bytes_per_seq > 0 else 0),
            "num_recurrent_layers_invalid",
        )
    elif profile is not None:
        state_bytes_per_seq = profile.total_state_bytes_per_seq(tp_size=tp_size, pp_size=pp_size)
        recurrent_layer_count = len(profile.recurrent_layer_indices)
    else:
        recurrent_cfg = merged.get("recurrent_state_config")
        if isinstance(recurrent_cfg, Mapping):
            rec_layers = _nonneg_int(recurrent_cfg.get("num_layers", 0), "recurrent_layers_invalid")
            v_heads = _pos_int(recurrent_cfg.get("num_v_heads", 32), "recurrent_v_heads_invalid")
            k_heads = _pos_int(recurrent_cfg.get("num_k_heads", v_heads), "recurrent_k_heads_invalid")
            dk = _pos_int(recurrent_cfg.get("head_k_dim", 128), "recurrent_dk_invalid")
            dv = _pos_int(recurrent_cfg.get("head_v_dim", 128), "recurrent_dv_invalid")
            conv_k = _nonneg_int(recurrent_cfg.get("conv_kernel_size", 4), "conv_kernel_size_invalid")
            variant = str(recurrent_cfg.get("variant", "gdn")).strip().lower()
            local_v = max(1, math.ceil(v_heads / tp_size))
            rec_mat_bytes = float(local_v * dk * dv * 4.0)
            channels = (2 * k_heads * dk + v_heads * dv) if variant == "gdn" else (3 * v_heads * dv)
            conv_bytes = float(math.ceil(channels / tp_size) * max(conv_k - 1, 0) * 2.0)
            layers_on_stage = math.ceil(rec_layers / pp_size) if rec_layers > 0 else 0
            state_bytes_per_seq = float(layers_on_stage * (rec_mat_bytes + conv_bytes))
            recurrent_layer_count = rec_layers
        else:
            state_bytes_per_seq = 0.0
            recurrent_layer_count = 0

    recurrent_state_bytes_per_rank = float(max_num_seqs * state_bytes_per_seq)

    # 6. Cross-layer KV / Index owner deduplication
    kv_block_size = _pos_int(merged.get("kv_block_size", 128), "kv_block_size_invalid")
    if profile is not None and merged.get("kv_owner_layers") is None and merged.get("kv_bytes_per_owner_layer_per_token") is None:
        owner_layers = list(profile.physical_kv_owner_layers)
        consumer_layers = list(profile.consumer_shared_kv_layers)
        per_owner_layer_token_bytes = profile.kv_bytes_per_owner_layer_per_token(tp_size=tp_size, dcp_size=dcp_size)
        owners_on_stage = math.ceil(len(owner_layers) / pp_size) if owner_layers else 0
        kv_bytes_per_token_per_rank = float(owners_on_stage * per_owner_layer_token_bytes)
    else:
        num_total_layers = _pos_int(
            merged.get("num_layers", profile.num_layers if profile else 32), "num_layers_invalid"
        )
        raw_recurrent_indices = merged.get(
            "recurrent_layer_indices",
            list(profile.recurrent_layer_indices) if profile else [],
        )
        recurrent_set = {int(i) for i in raw_recurrent_indices}
        raw_owners = merged.get("kv_owner_layers")
        if raw_owners is not None:
            if not isinstance(raw_owners, (list, tuple)):
                raise ServingPlannerError("kv_owner_layers_invalid")
            owner_layers = []
            seen_owners: set[int] = set()
            for idx_raw in raw_owners:
                idx = int(idx_raw)
                if idx < 0 or idx >= num_total_layers:
                    raise ServingPlannerError(f"kv_owner_layer_out_of_range:{idx}")
                if idx in recurrent_set:
                    raise ServingPlannerError(f"kv_owner_layer_is_recurrent:{idx}")
                if idx not in seen_owners:
                    seen_owners.add(idx)
                    owner_layers.append(idx)
        else:
            owner_layers = [i for i in range(num_total_layers) if i not in recurrent_set]

        owner_set = set(owner_layers)
        consumer_layers = [
            i for i in range(num_total_layers) if i not in owner_set and i not in recurrent_set
        ]
        # Validate every consumer layer has a preceding physical owner layer
        for c_idx in consumer_layers:
            if not any(o_idx < c_idx for o_idx in owner_layers):
                raise ServingPlannerError(f"kv_consumer_missing_earlier_owner:{c_idx}")

        if merged.get("kv_bytes_per_owner_layer_per_token") is not None:
            per_owner_layer_token_bytes = _nonneg_float(
                merged["kv_bytes_per_owner_layer_per_token"], "kv_bytes_per_owner_layer_per_token_invalid"
            ) / dcp_size
        elif profile is not None:
            per_owner_layer_token_bytes = profile.kv_bytes_per_owner_layer_per_token(tp_size=tp_size, dcp_size=dcp_size)
        else:
            kv_dtype = str(merged.get("kv_dtype", "bf16"))
            kv_b = dtype_bytes(kv_dtype)
            kv_lora_rank = _nonneg_int(merged.get("kv_lora_rank", 0), "kv_lora_rank_invalid")
            qk_rope_dim = _nonneg_int(merged.get("qk_rope_head_dim", 64), "qk_rope_head_dim_invalid")
            index_dim = _nonneg_int(merged.get("index_head_dim", 0), "index_head_dim_invalid")
            if kv_lora_rank > 0:
                row_bytes = float((kv_lora_rank + qk_rope_dim) * kv_b)
            else:
                num_kv_heads = _pos_int(merged.get("num_key_value_heads", 8), "num_key_value_heads_invalid")
                head_dim = _pos_int(merged.get("head_dim", 128), "head_dim_invalid")
                local_kv = max(1, math.ceil(num_kv_heads / tp_size))
                row_bytes = float(2 * local_kv * head_dim * kv_b)
            if index_dim > 0:
                row_bytes += float(index_dim * kv_b)
            per_owner_layer_token_bytes = row_bytes / dcp_size

        owners_on_stage = math.ceil(len(owner_layers) / pp_size) if owner_layers else 0
        kv_bytes_per_token_per_rank = float(owners_on_stage * per_owner_layer_token_bytes)

    unshared_token_kv_layers = len(owner_layers) + len(consumer_layers)
    unshared_on_stage = math.ceil(unshared_token_kv_layers / pp_size) if unshared_token_kv_layers else 0
    unshared_kv_bytes_per_token = float(unshared_on_stage * per_owner_layer_token_bytes)
    deduplicated_kv_bytes_saved_per_token = max(0.0, unshared_kv_bytes_per_token - kv_bytes_per_token_per_rank)

    # 7. Remaining HBM & block allocation
    non_kv_deductions = (
        total_weight_bytes_per_rank
        + workspace_bytes_per_rank
        + aclgraph_pool_bytes_per_rank
        + hccl_buffer_bytes_per_rank
        + recurrent_state_bytes_per_rank
    )
    available_kv_hbm_bytes = usable_hbm_bytes - non_kv_deductions
    kv_block_bytes = kv_bytes_per_token_per_rank * kv_block_size

    oom_reasons: list[str] = []
    if available_kv_hbm_bytes <= 0:
        oom_reasons.append("non_kv_deductions_exceed_usable_hbm")
        allocatable_kv_blocks = 0
    elif kv_block_bytes <= 0:
        allocatable_kv_blocks = 0
    else:
        allocatable_kv_blocks = int(available_kv_hbm_bytes // kv_block_bytes)
        if allocatable_kv_blocks <= 0:
            oom_reasons.append("insufficient_hbm_for_single_kv_block")

    allocatable_kv_tokens = allocatable_kv_blocks * kv_block_size
    blocks_per_seq = math.ceil(seq_len / kv_block_size)
    max_seqs_by_kv = (allocatable_kv_blocks // blocks_per_seq) if blocks_per_seq > 0 else 0
    effective_max_concurrency = min(max_num_seqs, max_seqs_by_kv) if kv_block_bytes > 0 else max_num_seqs
    if not oom_reasons and kv_block_bytes > 0 and max_seqs_by_kv <= 0:
        oom_reasons.append("kv_blocks_cannot_fit_single_sequence_at_seq_len")

    return {
        "status": "oom" if oom_reasons else "fits",
        "oom_reasons": oom_reasons,
        "hbm_capacity_bytes": hbm_capacity_bytes,
        "gpu_memory_utilization": util,
        "usable_hbm_bytes": usable_hbm_bytes,
        "parallel": {
            "tp": tp_size,
            "ep": ep_size,
            "pp": pp_size,
            "dp": dp_size,
            "dcp": dcp_size,
            "moe_tp": moe_tp_size,
        },
        "weight_budget": {
            "base_weight_bytes_per_rank": base_weight_bytes_per_rank,
            "eplb_redundant_experts_input": eplb_redundant,
            "eplb_redundant_experts_per_rank": eplb_redundant_per_rank,
            "eplb_weight_bytes_per_rank": eplb_weight_bytes_per_rank,
            "total_weight_bytes_per_rank": total_weight_bytes_per_rank,
        },
        "runtime_overhead_budget": {
            "workspace_bytes_per_rank": workspace_bytes_per_rank,
            "aclgraph_bucket_count": aclgraph_bucket_count,
            "aclgraph_per_bucket_bytes": per_bucket_bytes,
            "aclgraph_pool_bytes_per_rank": aclgraph_pool_bytes_per_rank,
            "hccl_buffsize_mib": hccl_buffsize_mib,
            "hccl_comm_groups": hccl_comm_groups,
            "hccl_buffer_bytes_per_rank": hccl_buffer_bytes_per_rank,
        },
        "recurrent_state_budget": {
            "num_recurrent_layers": recurrent_layer_count,
            "max_num_seqs": max_num_seqs,
            "seq_len_passed": seq_len,
            "state_scales_with_seq_len": False,
            "state_bytes_per_seq_per_rank": state_bytes_per_seq,
            "recurrent_state_bytes_per_rank": recurrent_state_bytes_per_rank,
        },
        "kv_owner_deduplication": {
            "physical_kv_owner_layers": owner_layers,
            "consumer_shared_kv_layers": consumer_layers,
            "num_owner_layers": len(owner_layers),
            "num_consumer_shared_layers": len(consumer_layers),
            "kv_bytes_per_owner_layer_per_token": per_owner_layer_token_bytes,
            "kv_bytes_per_token_per_rank": kv_bytes_per_token_per_rank,
            "unshared_kv_bytes_per_token_per_rank": unshared_kv_bytes_per_token,
            "deduplicated_kv_bytes_saved_per_token": deduplicated_kv_bytes_saved_per_token,
        },
        "available_kv_hbm_bytes": available_kv_hbm_bytes,
        "kv_block_size": kv_block_size,
        "kv_block_bytes": kv_block_bytes,
        "allocatable_kv_blocks": allocatable_kv_blocks,
        "allocatable_kv_tokens": allocatable_kv_tokens,
        "max_concurrent_seqs_by_kv": max_seqs_by_kv,
        "effective_max_concurrency": effective_max_concurrency,
    }


def optimize_pd_disaggregation_ratio(
    spec: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Deterministically optimize P:D and EPD instance ratios and compare serving modes.

    Models prefill QPS, decode QPS, KV transfer time and link bottleneck
    utilization across ``direct_rdma``, ``mooncake_layerwise``/``hixl_layerwise``,
    and ``host_staged`` connectors, enforces TTFT/TPOT SLO gates and vLLM-Ascend
    asymmetric TP constraints, and compares colocated vs PD disaggregated vs EPD
    per-chip throughput.
    """
    merged: dict[str, Any] = {}
    if spec is not None:
        if not isinstance(spec, Mapping):
            raise ServingPlannerError("pd_spec_invalid")
        merged.update(spec)
    merged.update(kwargs)

    raw_p_ms = merged.get("prefill_step_ms", merged.get("prefill_makespan_ms", merged.get("prefill_ttft_ms")))
    raw_d_ms = merged.get("decode_step_ms", merged.get("decode_tpot_ms"))
    if raw_p_ms is None or raw_d_ms is None:
        return {
            "status": "unmodeled",
            "unmodeled_reason": "missing_prefill_or_decode_step_ms",
            "prefill_instances": None,
            "decode_instances": None,
        }
    prefill_step_ms = _pos_float(raw_p_ms, "prefill_step_ms_invalid")
    decode_step_ms = _pos_float(raw_d_ms, "decode_step_ms_invalid")

    p_conc = _pos_int(merged.get("prefill_concurrency", merged.get("prefill_batch_size", 1)), "prefill_concurrency_invalid")
    d_conc = _pos_int(merged.get("decode_concurrency", merged.get("decode_batch_size", 32)), "decode_concurrency_invalid")
    p_devs = _pos_int(merged.get("prefill_devices_per_instance", merged.get("num_devices_p", 8)), "prefill_devices_invalid")
    d_devs = _pos_int(merged.get("decode_devices_per_instance", merged.get("num_devices_d", 8)), "decode_devices_invalid")
    prompt_len = _pos_int(merged.get("prompt_len", merged.get("input_length", 4096)), "prompt_len_invalid")
    output_len = _pos_int(merged.get("output_len", merged.get("output_length", 512)), "output_len_invalid")

    tp_p = _pos_int(merged.get("tp_p", p_devs), "tp_p_invalid")
    tp_d = _pos_int(merged.get("tp_d", d_devs), "tp_d_invalid")
    tp_topology_valid = (tp_p == tp_d) or (tp_p > tp_d and tp_p % tp_d == 0)

    spec_accept = _nonneg_float(merged.get("speculative_acceptance_length", 0.0), "speculative_acceptance_length_invalid")
    effective_tpot_ms = decode_step_ms / (1.0 + spec_accept)

    # KV Transfer Volume & Mode
    transfer_mode = str(merged.get("transfer_mode", "direct_rdma")).strip().lower()
    valid_modes = {"direct_rdma", "mooncake_layerwise", "hixl_layerwise", "host_staged"}
    if transfer_mode not in valid_modes:
        raise ServingPlannerError(f"transfer_mode_unsupported:{transfer_mode}")

    if merged.get("kv_bytes_per_request") is not None:
        kv_bytes_per_req = _nonneg_float(merged["kv_bytes_per_request"], "kv_bytes_per_request_invalid")
    elif merged.get("kv_bytes_per_token") is not None:
        kv_bytes_per_req = float(prompt_len) * _nonneg_float(merged["kv_bytes_per_token"], "kv_bytes_per_token_invalid")
    else:
        kv_bytes_per_req = 0.0

    if merged.get("transfer_bw_bytes_per_s") is not None:
        link_bw_Bps = _pos_float(merged["transfer_bw_bytes_per_s"], "transfer_bw_bytes_per_s_invalid")
    else:
        bw_gbps = _pos_float(merged.get("transfer_bw_gbps", 25.0), "transfer_bw_gbps_invalid")
        link_bw_Bps = bw_gbps * 1e9

    launch_overhead_ms = _nonneg_float(merged.get("transfer_launch_overhead_ms", 0.0), "transfer_launch_overhead_ms_invalid")

    if kv_bytes_per_req > 0.0:
        wire_ms = (kv_bytes_per_req / link_bw_Bps) * 1000.0
        if transfer_mode == "host_staged":
            host_bw_gbps = _pos_float(merged.get("host_staging_bw_gbps", min(16.0, link_bw_Bps / 1e9)), "host_staging_bw_gbps_invalid")
            host_bw_Bps = host_bw_gbps * 1e9
            # Host staging incurs D2H + network/copy + H2D hops
            stage_ms = 2.0 * (kv_bytes_per_req / host_bw_Bps) * 1000.0
            kv_transfer_ms = wire_ms + stage_ms + launch_overhead_ms
            exposed_kv_transfer_ms = kv_transfer_ms
            effective_link_Bps = 1.0 / ((1.0 / link_bw_Bps) + (2.0 / host_bw_Bps))
        elif transfer_mode in {"mooncake_layerwise", "hixl_layerwise"}:
            overlap_frac = _nonneg_float(merged.get("layerwise_overlap_fraction", 0.85), "layerwise_overlap_fraction_invalid")
            overlap_frac = min(1.0, overlap_frac)
            kv_transfer_ms = wire_ms + launch_overhead_ms
            overlapped_ms = overlap_frac * prefill_step_ms
            exposed_kv_transfer_ms = max(0.0, wire_ms - overlapped_ms) + launch_overhead_ms
            effective_link_Bps = link_bw_Bps
        else:
            # direct_rdma
            kv_transfer_ms = wire_ms + launch_overhead_ms
            exposed_kv_transfer_ms = kv_transfer_ms
            effective_link_Bps = link_bw_Bps
        kv_link_max_qps = effective_link_Bps / kv_bytes_per_req
    else:
        kv_transfer_ms = 0.0
        exposed_kv_transfer_ms = 0.0
        kv_link_max_qps = float("inf")

    # Multimodal Encoder (for EPD vs PD vs Colocated)
    encoder_step_ms = _nonneg_float(merged.get("encoder_step_ms", 0.0), "encoder_step_ms_invalid")
    encoder_conc = _pos_int(merged.get("encoder_concurrency", merged.get("encoder_batch_size", max(1, p_conc))), "encoder_concurrency_invalid")
    encoder_devs = _pos_int(merged.get("encoder_devices_per_instance", max(1, p_devs // 2)), "encoder_devices_invalid")
    mm_fraction = _nonneg_float(merged.get("multimodal_request_fraction", 1.0 if encoder_step_ms > 0 else 0.0), "multimodal_request_fraction_invalid")
    if mm_fraction > 1.0:
        raise ServingPlannerError("multimodal_request_fraction_exceeds_one")
    ec_cache_hit_rate = _nonneg_float(merged.get("ec_cache_hit_rate", 0.0), "ec_cache_hit_rate_invalid")
    if ec_cache_hit_rate > 1.0:
        raise ServingPlannerError("ec_cache_hit_rate_exceeds_one")
    ec_transfer_ms = _nonneg_float(merged.get("ec_transfer_ms", 0.0), "ec_transfer_ms_invalid")

    # In standard PD disaggregation, if encoder_step_ms > 0 and not disaggregated into E,
    # P runs inline vision encoding for multimodal requests.
    inline_encoder_ms = encoder_step_ms * mm_fraction
    pd_prefill_makespan_ms = prefill_step_ms + inline_encoder_ms
    compute_p_qps = p_conc / (pd_prefill_makespan_ms / 1000.0)
    kv_transfer_utilization = (compute_p_qps / kv_link_max_qps) if math.isfinite(kv_link_max_qps) else 0.0
    kv_transfer_bottleneck = kv_transfer_utilization > 1.0
    effective_p_qps = min(compute_p_qps, kv_link_max_qps)

    decode_tokens = max(output_len - 1, 1)
    d_qps = d_conc / (effective_tpot_ms * decode_tokens) * 1000.0

    # Continuous ratios
    p_to_d_ratio = d_qps / effective_p_qps
    d_to_p_ratio = effective_p_qps / d_qps

    # Disaggregated TTFT & SLO checks
    effective_ttft_ms = pd_prefill_makespan_ms + exposed_kv_transfer_ms
    ttft_slo_ms = merged.get("ttft_slo_ms")
    tpot_slo_ms = merged.get("tpot_slo_ms")
    slo_violations: list[str] = []
    if ttft_slo_ms is not None:
        ttft_limit = _pos_float(ttft_slo_ms, "ttft_slo_ms_invalid")
        if effective_ttft_ms > ttft_limit:
            slo_violations.append("ttft_exceeds_slo")
    else:
        ttft_limit = None
    if tpot_slo_ms is not None:
        tpot_limit = _pos_float(tpot_slo_ms, "tpot_slo_ms_invalid")
        if effective_tpot_ms > tpot_limit:
            slo_violations.append("tpot_exceeds_slo")
    else:
        tpot_limit = None
    if not tp_topology_valid:
        slo_violations.append("unsupported_asymmetric_tp_ratio")

    slo_compliant = len(slo_violations) == 0

    # Integer instance counts (target_qps or chip_budget or canonical search)
    target_qps_raw = merged.get("target_qps")
    chip_budget_raw = merged.get("total_chip_budget", merged.get("num_devices"))
    if target_qps_raw is not None:
        target_qps = _pos_float(target_qps_raw, "target_qps_invalid")
        p_instances = math.ceil(target_qps / effective_p_qps)
        d_instances = math.ceil(target_qps / d_qps)
    elif chip_budget_raw is not None:
        chip_budget = _pos_int(chip_budget_raw, "total_chip_budget_invalid")
        best_pair = (1, 1)
        best_qps_per_chip = -1.0
        max_p = max(1, chip_budget // p_devs)
        max_d = max(1, chip_budget // d_devs)
        for cand_p in range(1, max_p + 1):
            for cand_d in range(1, max_d + 1):
                chips = cand_p * p_devs + cand_d * d_devs
                if chips > chip_budget:
                    continue
                bal_qps = min(cand_p * effective_p_qps, cand_d * d_qps)
                qps_chip = bal_qps / chips
                if qps_chip > best_qps_per_chip:
                    best_qps_per_chip = qps_chip
                    best_pair = (cand_p, cand_d)
        p_instances, d_instances = best_pair
    else:
        max_inst = _pos_int(merged.get("max_search_instances", 16), "max_search_instances_invalid")
        best_pair = (1, 1)
        best_qps_per_chip = -1.0
        for cand_p in range(1, max_inst + 1):
            for cand_d in range(1, max_inst + 1):
                chips = cand_p * p_devs + cand_d * d_devs
                bal_qps = min(cand_p * effective_p_qps, cand_d * d_qps)
                qps_chip = bal_qps / chips
                if qps_chip > best_qps_per_chip:
                    best_qps_per_chip = qps_chip
                    best_pair = (cand_p, cand_d)
        p_instances, d_instances = best_pair

    disagg_total_devices = p_instances * p_devs + d_instances * d_devs
    disagg_balanced_qps = min(p_instances * effective_p_qps, d_instances * d_qps)
    disagg_qps_per_chip = disagg_balanced_qps / disagg_total_devices

    # Mode comparison: Colocated vs PD Disaggregated vs EPD Disaggregated
    colocated_devs = max(p_devs, d_devs)
    colocated_interference = _nonneg_float(merged.get("colocated_prefill_interference_factor", 1.25), "colocated_interference_invalid")
    colocated_tpot_ms = effective_tpot_ms * colocated_interference + (pd_prefill_makespan_ms / max(1, decode_tokens))
    colocated_ttft_ms = pd_prefill_makespan_ms + effective_tpot_ms
    # Colocated concurrency is bounded by P/D shared batch capacity
    colocated_conc = min(d_conc, max(p_conc, d_conc // 2))
    colocated_cycle_ms = pd_prefill_makespan_ms + decode_tokens * effective_tpot_ms * colocated_interference
    colocated_instance_qps = colocated_conc / (colocated_cycle_ms / 1000.0)
    colocated_replicas = (
        math.ceil( float(target_qps_raw) / colocated_instance_qps )
        if target_qps_raw is not None
        else max(1, disagg_total_devices // colocated_devs)
    )
    colocated_total_devices = colocated_replicas * colocated_devs
    colocated_achieved_qps = colocated_replicas * colocated_instance_qps
    colocated_qps_per_chip = colocated_instance_qps / colocated_devs
    colocated_slo_ok = (
        (ttft_limit is None or colocated_ttft_ms <= ttft_limit)
        and (tpot_limit is None or colocated_tpot_ms <= tpot_limit)
    )

    # EPD (Encoder-Prefill-Decode) mode evaluation
    if encoder_step_ms > 0.0:
        effective_enc_ms = encoder_step_ms * (1.0 - ec_cache_hit_rate)
        e_instance_qps = (
            encoder_conc / (effective_enc_ms / 1000.0)
            if effective_enc_ms > 0.0
            else float("inf")
        )
        # In EPD, P no longer runs the vision encoder inline
        epd_p_compute_qps = p_conc / (prefill_step_ms / 1000.0)
        epd_p_qps = min(epd_p_compute_qps, kv_link_max_qps)
        epd_ttft_ms = mm_fraction * (effective_enc_ms + ec_transfer_ms) + prefill_step_ms + exposed_kv_transfer_ms
        ref_qps = float(target_qps_raw) if target_qps_raw is not None else disagg_balanced_qps
        mm_qps = ref_qps * mm_fraction
        e_instances = math.ceil(mm_qps / e_instance_qps) if (mm_qps > 0 and math.isfinite(e_instance_qps)) else 1
        epd_p_instances = math.ceil(ref_qps / epd_p_qps)
        epd_d_instances = math.ceil(ref_qps / d_qps)
        epd_total_devices = e_instances * encoder_devs + epd_p_instances * p_devs + epd_d_instances * d_devs
        epd_balanced_qps = min(
            (e_instances * e_instance_qps / mm_fraction) if mm_fraction > 0 else float("inf"),
            epd_p_instances * epd_p_qps,
            epd_d_instances * d_qps,
        )
        epd_qps_per_chip = epd_balanced_qps / epd_total_devices
        epd_slo_ok = (
            (ttft_limit is None or epd_ttft_ms <= ttft_limit)
            and (tpot_limit is None or effective_tpot_ms <= tpot_limit)
            and tp_topology_valid
        )
    else:
        e_instances = 0
        epd_p_instances = p_instances
        epd_d_instances = d_instances
        epd_total_devices = disagg_total_devices
        epd_balanced_qps = disagg_balanced_qps
        epd_qps_per_chip = disagg_qps_per_chip
        epd_ttft_ms = effective_ttft_ms
        epd_slo_ok = slo_compliant

    mode_candidates = [
        ("colocated", colocated_slo_ok, colocated_qps_per_chip),
        ("pd_disaggregated", slo_compliant, disagg_qps_per_chip),
    ]
    if encoder_step_ms > 0.0:
        mode_candidates.append(("epd_disaggregated", epd_slo_ok, epd_qps_per_chip))

    compliant_modes = [m for m in mode_candidates if m[1]]
    if compliant_modes:
        recommended_mode = max(compliant_modes, key=lambda item: item[2])[0]
    else:
        recommended_mode = max(mode_candidates, key=lambda item: item[2])[0]

    return {
        "status": "modeled",
        "slo_compliant": slo_compliant,
        "slo_violations": slo_violations,
        "tp_topology_valid": tp_topology_valid,
        "prefill_instances": p_instances,
        "decode_instances": d_instances,
        "encoder_instances": e_instances,
        "p_to_d_instance_ratio": p_to_d_ratio,
        "d_to_p_instance_ratio": d_to_p_ratio,
        "msmodeling_pd_ratio": p_to_d_ratio,
        "prefill_instance_qps": effective_p_qps,
        "compute_prefill_instance_qps": compute_p_qps,
        "decode_instance_qps": d_qps,
        "balanced_qps": disagg_balanced_qps,
        "total_devices": disagg_total_devices,
        "per_chip_qps": disagg_qps_per_chip,
        "effective_ttft_ms": effective_ttft_ms,
        "effective_tpot_ms": effective_tpot_ms,
        "kv_transfer": {
            "transfer_mode": transfer_mode,
            "kv_bytes_per_request": kv_bytes_per_req,
            "kv_transfer_ms": kv_transfer_ms,
            "exposed_kv_transfer_ms": exposed_kv_transfer_ms,
            "kv_link_max_qps": kv_link_max_qps if math.isfinite(kv_link_max_qps) else None,
            "kv_transfer_utilization": kv_transfer_utilization,
            "kv_transfer_bottleneck": kv_transfer_bottleneck,
        },
        "mode_comparison": {
            "colocated": {
                "replicas": colocated_replicas,
                "total_devices": colocated_total_devices,
                "achieved_qps": colocated_achieved_qps,
                "per_chip_qps": colocated_qps_per_chip,
                "ttft_ms": colocated_ttft_ms,
                "tpot_ms": colocated_tpot_ms,
                "slo_compliant": colocated_slo_ok,
            },
            "pd_disaggregated": {
                "prefill_instances": p_instances,
                "decode_instances": d_instances,
                "total_devices": disagg_total_devices,
                "achieved_qps": disagg_balanced_qps,
                "per_chip_qps": disagg_qps_per_chip,
                "ttft_ms": effective_ttft_ms,
                "tpot_ms": effective_tpot_ms,
                "slo_compliant": slo_compliant,
            },
            "epd_disaggregated": {
                "encoder_instances": e_instances,
                "prefill_instances": epd_p_instances,
                "decode_instances": epd_d_instances,
                "total_devices": epd_total_devices,
                "achieved_qps": epd_balanced_qps,
                "per_chip_qps": epd_qps_per_chip,
                "ttft_ms": epd_ttft_ms,
                "tpot_ms": effective_tpot_ms,
                "slo_compliant": epd_slo_ok,
            },
            "recommended_mode": recommended_mode,
        },
    }


__all__ = [
    "ServingPlannerError",
    "plan_hbm_kv_capacity",
    "optimize_pd_disaggregation_ratio",
]
