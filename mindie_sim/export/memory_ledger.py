"""Closed-form per-die Weight/Quant/Cache/Engram-PLE/PP HBM memory calculator.

[Role] Compute deterministic, Excel-verifiable per-die memory ledgers honoring
  ``QuantSpec.exclusions``, post-TP/EP/PP/DCP/PCP/Local-TP tensor sharding,
  MXFP4/MXFP8 microscaling and W8A8/W4A8 per-channel aux overhead, KV/Indexer/
  KeyPool/KDA recurrent cache sizing, Engram/PLE Host DRAM offload, and closed-form
  per-DP Max HBM Concurrency.
[Invariants] Strictly pure analytical computation over ``Case`` and ``Memory``;
  zero device contexts, zero weight loading, and exact parity with Sheet 3's
  live OOXML ``VLOOKUP``/``ROUNDUP`` formulas.
[Timeouts & Locks] Single-pass O(Unique Weight Terms + Layers) arithmetic; no
  locks or I/O.
[Side-Effect Policy] Read-only over caller ``Case`` instances.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

from mindie_sim.catalog.cache import (
    DEFAULT_CACHE_BLOCK_SIZE,
    INDEX_INTERLEAVE_TOKENS,
    layer_cache_components,
    local_kv_heads,
)
from mindie_sim.catalog.cache_format import dense_c8_kv_planes
from mindie_sim.catalog.cache_pools import exact_element_bytes
from mindie_sim.catalog.context_parallel import dcp_q_replicate_enabled
from mindie_sim.catalog.npu_checkpoints import (
    NpuCheckpointSpec,
    classify_weight_tensor_quant,
    normalize_hw_family,
    resolve_default_checkpoint,
    validate_quant_for_hardware,
)
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import (
    WeightTerm,
    _sharded_shape,
    term_bytes,
    term_payload_bytes,
)
from mindie_sim.catalog.weights import (
    _expand_resident_terms,
    engram_table_terms,
    stage_weight_terms,
)
from mindie_sim.prediction.capacity import Memory, prepared_weight_bytes
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.stack import PreparedWeightSpec

from .models import (
    BF16_BYTES,
    FP32_BYTES,
    KV_QUANT_GROUP_SIZE,
    MAX_EXPLICIT_LAYER_LABELS,
    MIN_DENOM_EPS,
    MXFP_GROUP_SIZE,
    QUANT_RULE_CHECKPOINT_IGNORE,
    QUANT_RULE_FIXED_BF16,
    QUANT_RULE_FIXED_FP32,
    QUANT_RULE_FOLLOW_PRESET,
    QUANT_RULE_PRESET_LINEAR_FP8,
    SUPPORTED_CACHE_DTYPES,
    WEIGHT_QUANT_FORMATS,
    CacheLedgerRow,
    MemoryCalculatorLedger,
    StageBudgetRow,
    WeightLedgerRow,
)

_ENGRAM_CATEGORIES = frozenset({"engram_table", "ple_table"})
_VISION_CATEGORIES = frozenset({"vision", "projector", "vision_pos", "image_token"})
_VISION_NAME_PREFIXES = ("vision_", "aligner_", "merger_", "projector_", "image_")
_W4A8_EXPERT_ONLY_CATEGORIES = frozenset({"expert", "routed_expert", "moe.routed"})
_MXFP4_EXPERT_CATEGORIES = frozenset({"expert", "routed_expert", "moe.routed"})
_FIXED_FP32_CATEGORIES = frozenset({
    "param_fp32", "conv_weight_f32", "a_log", "dt_bias", "norm_f32",
    "compressor_fp32_runtime", "mhc", "ape",
})
_FIXED_BF16_CATEGORIES = frozenset({
    "norm", "embedding", "lm_head", "router", "router_bias",
    "engram_table", "ple_table", "vision", "projector",
    "vision_pos", "image_token", "conv_weight_bf16", "linear_param_bf16",
    "compressor_bf16_runtime", "dspark_input",
})
#: Distinct KV-cache bytes/element factors:
#: MLA C8 uses FP32/128 scales; GQA static C8 stores INT8 payload (1.0 B/elem)
#: plus static per-channel checkpoint scales/offsets (no block-128 inflation);
#: FP8 uses FP16/128 scales; MXFP8 uses E8M0/32 scales (33/32 B/elem).
C8_CACHE_WIDTH_BYTES = 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE)
GQA_STATIC_C8_CACHE_WIDTH_BYTES = 1.0
FP8_CACHE_WIDTH_BYTES = 1.0 + float(BF16_BYTES) / float(KV_QUANT_GROUP_SIZE)
MXFP8_CACHE_WIDTH_BYTES = 1.0 + 1.0 / float(MXFP_GROUP_SIZE)
_FLOAT16_FORMATS = frozenset({"bf16", "fp16", "bfloat16", "float16"})
_GQA_CACHE_COMPONENTS = frozenset({"gqa_kv", "window"})
#: Cache components whose byte width scales with the target KV cache dtype override.
KV_DTYPE_SENSITIVE_COMPONENTS = frozenset({
    "gqa_kv", "mla_kv", "window", "compressed_kv", "dcp_dp_cohort_kv",
})
_PREPARED_WEIGHT_GROUP = "Prepared Runtime Weights"


def _is_ple_category(cat: str) -> bool:
    low = str(cat or "").lower()
    return low == "ple" or low.endswith("_ple")
_WEIGHT_QUANT_ALIASES: dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "fp32": "FP32",
    "float32": "FP32",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
    "w8a8": "W8A8",
    "int8": "INT8",
    "w4a16": "W4A16",
    "compressed_tensors_w4a16": "W4A16",
    "w4a16_compressed": "W4A16",
    "w4a16_awq": "W4A16",
    "w4a16_gptq": "W4A16",
    "w4a8": "W4A8",
    "int4": "INT4",
    "mxfp8": "MXFP8",
    "w8a8_mxfp8": "MXFP8",
    "w8a8-mxfp8": "MXFP8",
    "mxfp8_mxfp4": "MXFP4",
    "mxfp8+mxfp4": "MXFP4",
    "quarot_mxfp8_mxfp4": "MXFP4",
    "mxfp4": "MXFP4",
    "mxfp4_routed_only": "MXFP4",
    "w4a8_mxfp": "MXFP4",
    "w4a8-mxfp": "MXFP4",
    "w4a8_mxfp4": "MXFP4",
    "w4a8-mxfp4": "MXFP4",
}
_CACHE_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "c8": "C8",
    "int8": "C8",
    "faquant": "C8",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
    "mxfp8": "MXFP8",
    "mx_fp8": "MXFP8",
    "qk_mxfp8_dynamic_v_mxfp8_per_channel": "MXFP8",
    "qk-mxfp8-dynamic-v-mxfp8-per-channel": "MXFP8",
}
_QUANT_FORMAT_DISPLAY: dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "fp32": "FP32",
    "float32": "FP32",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
    "int8": "W8A8",
    "w8a8": "W8A8",
    "w4a16": "W4A16",
    "int4": "W4A8",
    "w4a8": "W4A8",
    "mxfp8": "MXFP8",
    "mx_fp8": "MXFP8",
    "mxfp4": "MXFP4",
    "mx_fp4": "MXFP4",
    "fp4": "MXFP4",
    "fp4_e2m1": "MXFP4",
}


def evaluate_weight_storage_bytes(
    local_rows: int, local_cols: int, quant_format: str,
) -> tuple[int, int, int, int]:
    """Closed-form ``(payload, scale, aux, total)`` bytes for one post-TP local matrix."""
    key = str(quant_format or "BF16").strip().upper()
    spec = WEIGHT_QUANT_FORMATS.get(key)
    if spec is None:
        raise ModelingError(f"unsupported_weight_quant_format:{quant_format}")
    rows = int(local_rows)
    cols = int(local_cols)
    if rows < 1 or cols < 1:
        raise ModelingError(f"invalid_local_weight_shape:({rows},{cols})")
    payload = rows * math.ceil(cols * float(spec["bytes"]))
    group = int(spec["group"])
    scale = (
        rows * ((cols + group - 1) // group) * int(spec["scale"])
        if group > 0 else 0
    )
    aux = rows * int(spec["aux"]) if int(spec["aux"]) > 0 else 0
    return payload, scale, aux, payload + scale + aux


def _normalize_quant_preset(quant: QuantSpec | None, fallback: str = "BF16") -> str:
    if quant is None:
        return fallback
    name = str(getattr(quant, "name", "") or "").strip().lower()
    exp_fmt = str(getattr(quant, "expert_weight_format", "") or "").strip().lower()
    w_fmt = str(getattr(quant, "weight_format", "") or "").strip().lower()
    act_fmt = str(getattr(quant, "activation_format", "") or "").strip().lower()
    combined = f"{name} {exp_fmt} {w_fmt}"
    if "mxfp4" in combined or "mx_fp4" in combined:
        return "MXFP4"
    if "mxfp8" in combined or "mx_fp8" in combined:
        return "MXFP8"
    if "w4a16" in combined or (
        (exp_fmt == "int4" or w_fmt == "int4")
        and act_fmt in _FLOAT16_FORMATS
        and "w4a8" not in name
    ):
        return "W4A16"
    if "w4a8" in combined or "int4" in combined or "fp4" in combined:
        return "W4A8"
    if "w8a8" in combined or "int8" in combined:
        return "W8A8"
    if "fp8" in combined:
        return "FP8"
    return "BF16"


def _describe_quant_display(
    quant: QuantSpec | None,
    resolved_wq: str,
    *,
    explicit_override: bool,
    apply_preset_to_ignored: bool = False,
) -> tuple[str, tuple[str, ...]]:
    exclusions = tuple(getattr(quant, "exclusions", ()) or ()) if quant is not None else ()
    if apply_preset_to_ignored:
        return f"{resolved_wq} (Full Quant / QuaRot Override)", exclusions
    if quant is None or not exclusions:
        return resolved_wq, ()
    excl_set = set(exclusions)
    routed_only = (
        resolved_wq == "MXFP4"
        and ({"attn", "mla", "kda"} & excl_set)
        and "shared_expert" in excl_set
    )
    if routed_only:
        overrides = dict(getattr(quant, "category_overrides", ()) or ())
        if "latent_moe" in overrides or "attn_res_proj" in overrides:
            return "MXFP4 (Routed+Latent+AttnRes, Attn/Shared BF16)", exclusions
        return "MXFP4 (Routed-Only, Attn/Shared BF16)", exclusions
    w_fmt = str(getattr(quant, "weight_format", "bf16") or "bf16").lower()
    exp_fmt = str(getattr(quant, "expert_weight_format", "bf16") or "bf16").lower()
    if not explicit_override and w_fmt != exp_fmt and "fp8" in w_fmt and "mxfp4" in exp_fmt:
        q_name = str(getattr(quant, "name", "") or "").lower()
        if "mxfp8" in q_name:
            return "MXFP8 Linear + MXFP4 Routed/Latent (QuaRot / msmodelslim)", exclusions
        return "FP8 Linear + MXFP4 Routed Experts (Router/Norm/Head BF16)", exclusions
    if resolved_wq == "W4A16":
        if ({"attn", "mla", "kda"} & excl_set) and "shared_expert" in excl_set:
            return "W4A16 (INT4 Group-32 Routed Experts, Attn/Shared BF16)", exclusions
        return "W4A16 (Compressed INT4 Group-32, BF16 Activations)", exclusions
    if resolved_wq == "W4A8":
        return "W4A8_DYNAMIC (INT8 Linear + INT4 Routed Experts, Norm/Router/Head BF16)", exclusions
    if resolved_wq == "W8A8":
        return "W8A8_DYNAMIC (INT8 Linear & Experts, Norm/Router/Head BF16)", exclusions
    return f"{resolved_wq} (Excl: {','.join(exclusions[:MAX_EXPLICIT_LAYER_LABELS])})", exclusions


def _canonicalize_weight_quant_override(raw: str) -> str:
    token = str(raw).strip().lower()
    return _WEIGHT_QUANT_ALIASES.get(token, str(raw).strip().upper())


def _canonicalize_cache_dtype_override(raw: str) -> str:
    token = str(raw).strip().lower()
    return _CACHE_DTYPE_ALIASES.get(token, str(raw).strip().upper())


def _normalize_cache_dtype(model: Any) -> str:
    meta = dict(getattr(model, "metadata", ()) or ())
    quant = getattr(model, "quant", None)
    tokens = (
        str(meta.get("kv_cache_tag") or "").strip().lower(),
        str(meta.get("kv_cache_type") or "").strip().lower(),
        str(getattr(quant, "kv_dtype", "") if quant is not None else "").strip().lower(),
        str(getattr(model, "kv_dtype", "") or "").strip().lower(),
    )
    joined = " ".join(tok for tok in tokens if tok)
    if "mxfp8" in joined or "mx_fp8" in joined:
        return "MXFP8"
    if "int8" in joined or "faquant" in joined or any(tok == "c8" for tok in tokens):
        return "C8"
    if "fp8" in joined:
        return "FP8"
    return "BF16"


def _resolve_shard_axis_and_rule(
    term: WeightTerm,
    placed_parallel: Any = None,
    base_parallel: Any = None,
) -> tuple[str, int]:
    shard = str(term.shard or "replicated")
    role = str(term.placement_role or "")
    axis = 1 if (role == "attention_output" or shard in {"in_tp", "dense_down", "expert_down", "group_in"}) else 0
    if shard in {"expert_gate_up", "expert_down"}:
        return "EP", axis
    if shard == "replicated":
        return "Replicated", axis
    if shard == "node":
        return "Node", axis
    if role == "attention_output":
        return "Local-TP (o_proj)", axis
    if role == "attention_query":
        return "Local-TP (q_proj)", axis
    if term.category == "shared_expert":
        if (
            placed_parallel is not None
            and base_parallel is not None
            and int(getattr(placed_parallel, "tp", 1) or 1) == 1
            and int(getattr(base_parallel, "tp", 1) or 1) > 1
        ):
            return "Shared-DP (Replicated)", axis
        return "Shared-TP", axis
    return f"TP ({shard})", axis


def stage_layer_counts(case: Case) -> tuple[int, ...]:
    """Return decoder layer counts owned by each pipeline stage."""
    p = case.runtime.parallel
    if case.runtime.pp_layer_partition:
        return tuple(int(x) for x in case.runtime.pp_layer_partition)
    return tuple(
        case.model.layers // p.pp + (1 if stage < case.model.layers % p.pp else 0)
        for stage in range(p.pp)
    )


def _model_has_mla(model: Any) -> bool:
    return any(getattr(lp, "attention", "") == "mla" for lp in getattr(model, "layer_plans", ()) or ())


def _cache_dtype_width_bytes(dtype: str, *, is_gqa: bool) -> float:
    if dtype == "MXFP8":
        return MXFP8_CACHE_WIDTH_BYTES
    if dtype == "C8":
        return GQA_STATIC_C8_CACHE_WIDTH_BYTES if is_gqa else C8_CACHE_WIDTH_BYTES
    if dtype == "FP8":
        return FP8_CACHE_WIDTH_BYTES
    return float(BF16_BYTES)


def _cache_dtype_override_ratio(model: Any, target_dtype: str, *, component: str = "") -> float:
    base = _normalize_cache_dtype(model)
    target = str(target_dtype or base).strip().upper()
    if target == base:
        return 1.0
    is_gqa = (
        component in _GQA_CACHE_COMPONENTS
        or (component == "" and not _model_has_mla(model))
    )
    base_w = _cache_dtype_width_bytes(base, is_gqa=is_gqa)
    target_w = _cache_dtype_width_bytes(target, is_gqa=is_gqa)
    return target_w / max(base_w, MIN_DENOM_EPS)


def _gqa_c8_static_scale_bytes_by_stage(
    case: Case,
    stage_layers: Sequence[int],
    block_size: int,
) -> tuple[list[float], list[int]]:
    """Per-stage static checkpoint C8 channel scale + offset bytes for GQA layers."""
    n_stages = len(stage_layers)
    by_stage = [0.0] * n_stages
    layers_by_stage = [0] * n_stages
    layer_stage = [s for s, count in enumerate(stage_layers) for _ in range(count)]
    tp = max(1, int(case.runtime.parallel.tp))
    blk = max(1, int(block_size))
    for lp in getattr(case.model, "layer_plans", ()) or ():
        if getattr(lp, "attention", "") != "gqa":
            continue
        k_dim = int(getattr(lp, "head_dim", 0) or getattr(case.model, "head_dim", 0) or 0)
        v_dim = int(getattr(lp, "v_head_dim", 0) or k_dim)
        n_kv = int(getattr(lp, "num_kv_heads", 0) or getattr(case.model, "kv_heads", 0) or 0)
        if k_dim <= 0 or n_kv <= 0:
            continue
        local_kv = local_kv_heads(n_kv, tp)
        k_planes = dense_c8_kv_planes(local_kv, blk, k_dim)
        if v_dim == k_dim:
            static_b = float(sum(
                pl.bytes_per_unit for pl in k_planes if pl.allocation_frequency == "checkpoint"
            ))
        else:
            v_planes = dense_c8_kv_planes(local_kv, blk, v_dim)
            static_b = float(
                sum(
                    pl.bytes_per_unit for pl in k_planes
                    if pl.allocation_frequency == "checkpoint" and pl.plane_id.startswith("k_")
                )
                + sum(
                    pl.bytes_per_unit for pl in v_planes
                    if pl.allocation_frequency == "checkpoint" and pl.plane_id.startswith("v_")
                )
            )
        st = layer_stage[lp.index] if lp.index < len(layer_stage) else 0
        if 0 <= st < n_stages:
            by_stage[st] += static_b
            layers_by_stage[st] += 1
    return by_stage, layers_by_stage


_FIXED_STATE_CACHE_COMPONENTS = frozenset({
    "recurrent_state",
    "conv_state",
    "compressor_state",
    "index_compressor_state",
    "index_pool_tail",
    "ple_conv_state",
    "draft_state",
    "replay_records",
    "state_transient",
    "page_group_reserve",
    "page_metadata",
    "c8_static_channel_scale",
})


def _add_fixed_cache_stage_values(
    comp_name: str,
    values: Sequence[float],
    stage_layers: Sequence[int],
    comp_stage_bytes: dict[str, list[float]],
    comp_stage_layers: dict[str, list[int]],
    comp_scalable: dict[str, bool],
    comp_kv_scaled: dict[str, bool],
    *,
    layer_counts: Sequence[int] | None = None,
) -> None:
    for st_idx in range(min(len(stage_layers), len(values))):
        val = float(values[st_idx] or 0.0)
        if val <= 0.0:
            continue
        comp_scalable[comp_name] = False
        comp_kv_scaled[comp_name] = False
        comp_stage_bytes[comp_name][st_idx] += val
        layers_val = (
            int(layer_counts[st_idx])
            if layer_counts is not None and st_idx < len(layer_counts)
            else int(stage_layers[st_idx])
        )
        comp_stage_layers[comp_name][st_idx] = max(1, layers_val)


def _compute_cache_ledger_rows(
    case: Case,
    mem: Memory,
    alloc_cache: Mapping[str, Any],
    *,
    stage_layers: Sequence[int],
    cache_dtype: str,
    seq_len: int,
    concurrency: int,
    block_size: int,
    spec_tokens: int,
) -> tuple[int, tuple[CacheLedgerRow, ...], list[float], list[float]]:
    """Unify cache ledger rows directly onto ``Memory(case)`` (including Draft KV & page groups)."""
    model = case.model
    p = case.runtime.parallel
    dcp = max(1, int(p.cp))
    n_stages = len(stage_layers)
    local_tokens = math.ceil(max(1, seq_len) / (max(1, block_size) * dcp)) * max(1, block_size)
    layer_stage = [s for s, count in enumerate(stage_layers) for _ in range(count)]
    comp_stage_bytes: dict[str, list[float]] = defaultdict(lambda: [0.0] * n_stages)
    comp_stage_layers: dict[str, list[int]] = defaultdict(lambda: [0] * n_stages)
    comp_scalable: dict[str, bool] = {}

    fixed_state_comps = _FIXED_STATE_CACHE_COMPONENTS
    # Count active target layers per component so per-token-per-layer rates stay exact.
    for lp in model.layer_plans:
        stage = layer_stage[lp.index] if lp.index < len(layer_stage) else 0
        raw_comps = layer_cache_components(
            model, lp, p, max(1, seq_len),
            capacity_padded=True,
            interleave=INDEX_INTERLEAVE_TOKENS,
            runtime=case.runtime,
            query_tokens=max(1, 1 + spec_tokens),
        )
        for comp_name, val in raw_comps.items():
            if val > 0.0:
                comp_stage_layers[comp_name][stage] += 1

    # Read authoritative per-stage component bytes directly from Memory.allocation_report().
    alloc_summary = alloc_cache.get("allocated") or {}
    reserves = alloc_summary.get("reserve_bytes_per_stage_rank") or ()
    metadatas = alloc_summary.get("metadata_bytes_per_stage_rank") or ()
    comp_entries = alloc_cache.get("components") or ()
    comp_kv_scaled: dict[str, bool] = {}
    for st_idx in range(n_stages):
        entry = comp_entries[st_idx] if st_idx < len(comp_entries) else {}
        for comp_name, detail in (entry.get("allocated") or {}).items():
            val = float(detail.get("total", 0.0) or 0.0)
            if val <= 0.0:
                continue
            scalable = comp_name not in fixed_state_comps
            kv_scaled = scalable and (comp_name in KV_DTYPE_SENSITIVE_COMPONENTS)
            comp_scalable[comp_name] = scalable
            comp_kv_scaled[comp_name] = kv_scaled
            ratio = _cache_dtype_override_ratio(model, cache_dtype, component=comp_name) if kv_scaled else 1.0
            comp_stage_bytes[comp_name][st_idx] += val * ratio
        for comp_name, detail in (entry.get("group_allocated") or {}).items():
            val = float(detail.get("alloc_payload", 0.0) or 0.0) + float(detail.get("alloc_scale", 0.0) or 0.0)
            if val <= 0.0:
                continue
            scalable = comp_name not in fixed_state_comps
            kv_scaled = scalable and (comp_name in KV_DTYPE_SENSITIVE_COMPONENTS)
            comp_scalable[comp_name] = scalable
            comp_kv_scaled[comp_name] = kv_scaled
            ratio = _cache_dtype_override_ratio(model, cache_dtype, component=comp_name) if kv_scaled else 1.0
            comp_stage_bytes[comp_name][st_idx] += val * ratio
    _add_fixed_cache_stage_values(
        "page_group_reserve", reserves, stage_layers,
        comp_stage_bytes, comp_stage_layers, comp_scalable, comp_kv_scaled,
    )
    stride_reserves = mem._unified_pool_stride_reserve_by_stage(max(1, seq_len))
    _add_fixed_cache_stage_values(
        "page_group_reserve", stride_reserves, stage_layers,
        comp_stage_bytes, comp_stage_layers, comp_scalable, comp_kv_scaled,
    )
    _add_fixed_cache_stage_values(
        "page_metadata", metadatas, stage_layers,
        comp_stage_bytes, comp_stage_layers, comp_scalable, comp_kv_scaled,
    )
    if cache_dtype == "C8" and (_GQA_CACHE_COMPONENTS & set(comp_stage_bytes)):
        c8_static_b, c8_gqa_layers = _gqa_c8_static_scale_bytes_by_stage(case, stage_layers, block_size)
        amortized_static = [v / float(max(1, concurrency)) for v in c8_static_b]
        _add_fixed_cache_stage_values(
            "c8_static_channel_scale", amortized_static, stage_layers,
            comp_stage_bytes, comp_stage_layers, comp_scalable, comp_kv_scaled,
            layer_counts=c8_gqa_layers,
        )

    # Account for DP-cohort DCP amplification when dcp_dp_width > 1 so per-DP cache matches Memory;
    # the cohort's extra sharded KV bytes take the same cache-dtype width ratio as the local KV.
    if int(getattr(p, "dcp_dp_width", 1) or 1) > 1:
        cohort_ratio = _cache_dtype_override_ratio(model, cache_dtype, component="dcp_dp_cohort_kv")
        cohort_totals = mem.dp_cohort_cache_bytes(max(1, seq_len), padded=True)
        base_totals = mem.cache_bytes(max(1, seq_len), padded=True)
        for st_idx in range(min(n_stages, len(cohort_totals), len(base_totals))):
            extra = (float(cohort_totals[st_idx]) - float(base_totals[st_idx])) * cohort_ratio
            if extra > MIN_DENOM_EPS:
                comp_scalable["dcp_dp_cohort_kv"] = True
                comp_kv_scaled["dcp_dp_cohort_kv"] = True
                comp_stage_bytes["dcp_dp_cohort_kv"][st_idx] += extra
                comp_stage_layers["dcp_dp_cohort_kv"][st_idx] = max(1, int(stage_layers[st_idx]))

    if "draft_kv" in comp_stage_bytes and sum(comp_stage_layers["draft_kv"]) == 0:
        aux = getattr(model, "auxiliary", None)
        draft_obj = getattr(aux, mem.draft_kind, None) if (aux is not None and mem.draft_kind) else None
        draft_layers = max(1, int(getattr(draft_obj, "layers", 1) or 1))
        comp_stage_layers["draft_kv"][-1] = draft_layers

    stage_per_req_totals = [0.0] * n_stages
    stage_totals = [0.0] * n_stages
    rows: list[CacheLedgerRow] = []
    index_dtype_lbl = str(getattr(getattr(model, "quant", None), "index_dtype", "int8") or "int8").upper()
    for comp_name in sorted(comp_stage_bytes):
        per_stage = comp_stage_bytes[comp_name]
        layers_per_stage = tuple(comp_stage_layers[comp_name])
        max_layers = max(layers_per_stage, default=0)
        max_req_bytes = max(per_stage, default=0.0)
        scalable = comp_scalable.get(comp_name, True) and (max_layers > 0)
        kv_scaled = comp_kv_scaled.get(comp_name, False) and scalable
        b_per_tok = (
            max_req_bytes / float(max(1, max_layers * local_tokens))
            if scalable else 0.0
        )
        for st_idx, st_val in enumerate(per_stage):
            stage_per_req_totals[st_idx] += st_val
            stage_totals[st_idx] += st_val * float(concurrency)
        if not scalable:
            c_kind = "State (Fixed)"
        elif kv_scaled:
            c_kind = cache_dtype
        elif comp_name == "index_k":
            c_kind = f"Index ({index_dtype_lbl})"
        elif comp_name == "draft_kv":
            c_kind = "Draft KV (Fixed)"
        else:
            c_kind = "Independent (Fixed)"
        rows.append(CacheLedgerRow(
            component=comp_name,
            cache_kind=c_kind,
            active_layers_per_stage=layers_per_stage,
            max_stage_layers=max_layers,
            bytes_per_token_per_layer=b_per_tok,
            per_request_bytes=max_req_bytes,
            batch_total_bytes=max_req_bytes * float(concurrency),
            scales_with_tokens=scalable,
            per_stage_bytes=tuple(float(v) for v in per_stage),
        ))
    return local_tokens, tuple(rows), stage_totals, stage_per_req_totals


def _is_vision_term(term: WeightTerm) -> bool:
    cat = str(term.category or "").lower()
    fam = cat.split(".", 1)[0]
    if cat in _VISION_CATEGORIES or fam in _VISION_CATEGORIES:
        return True
    return str(term.name or "").lower().startswith(_VISION_NAME_PREFIXES)


def _linear_format_for_preset(resolved_wq: str) -> str:
    if resolved_wq == "W4A8":
        return "W8A8"
    if resolved_wq == "MXFP4":
        return "MXFP8"
    return resolved_wq


def _resolve_term_quant_format(
    term: WeightTerm,
    model_quant: QuantSpec | None,
    resolved_wq: str,
    *,
    explicit_override: bool,
    apply_preset_to_ignored: bool = False,
    checkpoint: NpuCheckpointSpec | None = None,
) -> tuple[str, bool, str, str]:
    cat = str(term.category or "")
    fam = cat.split(".", 1)[0]
    term_quant = term.runtime_quant or term.quant or model_quant
    if not explicit_override and not apply_preset_to_ignored and term.runtime_dtype:
        rt_fmt = _QUANT_FORMAT_DISPLAY.get(str(term.runtime_dtype).strip().lower(), "BF16")
        rt_rule = QUANT_RULE_FIXED_FP32 if rt_fmt == "FP32" else QUANT_RULE_FIXED_BF16
        return rt_fmt, (rt_fmt in {"BF16", "FP32"}), f"Runtime {rt_fmt} param", rt_rule
    if cat == "engram_table" and not explicit_override and not apply_preset_to_ignored and term_quant is not None:
        e_mode = str(getattr(term_quant, "engram_table_mode", "device_bf16") or "device_bf16").lower()
        if e_mode in {"host_fp8_e8m0", "source_fp8_e8m0"}:
            return "MXFP8", False, f"Engram table ({e_mode})", QUANT_RULE_FIXED_BF16
        if e_mode == "device_int8_fp32":
            return "W8A8", False, f"Engram table ({e_mode})", QUANT_RULE_FIXED_BF16
        return "BF16", True, f"Engram table ({e_mode})", QUANT_RULE_FIXED_BF16
    if fam in _FIXED_FP32_CATEGORIES or cat in _FIXED_FP32_CATEGORIES:
        return "FP32", True, "FP32 param", QUANT_RULE_FIXED_FP32
    if fam in _FIXED_BF16_CATEGORIES or cat in _FIXED_BF16_CATEGORIES or _is_ple_category(cat) or _is_ple_category(fam):
        note = "Vision ViT BF16" if _is_vision_term(term) else "Native BF16 structural/head param"
        return "BF16", True, note, QUANT_RULE_FIXED_BF16

    is_w4_expert = cat in _W4A8_EXPERT_ONLY_CATEGORIES or fam in _W4A8_EXPERT_ONLY_CATEGORIES
    is_mxfp4_expert = cat in _MXFP4_EXPERT_CATEGORIES or fam in _MXFP4_EXPERT_CATEGORIES
    if term_quant is not None:
        exclusions = set(getattr(term_quant, "exclusions", ()) or ())
        if cat in exclusions or fam in exclusions:
            if apply_preset_to_ignored:
                over_fmt = resolved_wq if is_mxfp4_expert else _linear_format_for_preset(resolved_wq)
                return over_fmt, False, f"Override ignore -> {over_fmt}", QUANT_RULE_CHECKPOINT_IGNORE
            native = _QUANT_FORMAT_DISPLAY.get(str(term_quant.native_format or "bf16").lower(), "BF16")
            return native, True, "quantization_config.ignore -> BF16", QUANT_RULE_CHECKPOINT_IGNORE
        if not explicit_override:
            resolved_q = term_quant.for_category(cat)
            eff_raw = str(resolved_q.format_for(cat) or "bf16").lower()
            lin_method = str(resolved_q.linear_method(cat) or "").lower()
            act_fmt = str(getattr(resolved_q, "activation_format", "") or "").lower()
            w_blk = tuple(getattr(resolved_q, "weight_block", (0, 0)) or (0, 0))
            if lin_method == "mx_fp8" or (
                eff_raw in {"fp8", "fp8_e4m3", "fp8_e5m2"} and w_blk == (1, MXFP_GROUP_SIZE)
            ):
                mapped = "MXFP8"
            elif lin_method == "mx_fp4":
                mapped = "MXFP4"
            elif lin_method == "w4a16" or (eff_raw == "int4" and act_fmt in _FLOAT16_FORMATS):
                mapped = "W4A16"
            else:
                mapped = _QUANT_FORMAT_DISPLAY.get(eff_raw, resolved_wq)
            if resolved_wq == "W4A8" and not is_w4_expert and mapped == "W4A8":
                mapped = "W8A8"
            note = (
                "MXFP4 (group=32)" if mapped == "MXFP4"
                else ("MXFP8 (group=32 E8M0)" if mapped == "MXFP8"
                      else ("W4A16 (INT4 Group-32, BF16 Act)" if mapped == "W4A16"
                            else ("W4A8_DYNAMIC (INT4 Packed Expert)" if mapped == "W4A8"
                                  else ("W8A8_DYNAMIC (INT8 Linear)" if mapped == "W8A8" else mapped))))
            )
            rule = QUANT_RULE_FOLLOW_PRESET if (is_mxfp4_expert or is_w4_expert) else QUANT_RULE_PRESET_LINEAR_FP8
            return mapped, (mapped == "BF16"), note, rule
    elif checkpoint is not None and not explicit_override:
        wt_q = classify_weight_tensor_quant(term.name, checkpoint)
        if wt_q.ledger_format in WEIGHT_QUANT_FORMATS:
            mapped = wt_q.ledger_format
            rule = (
                QUANT_RULE_CHECKPOINT_IGNORE
                if not wt_q.is_quantized
                else (QUANT_RULE_FOLLOW_PRESET if (is_mxfp4_expert or is_w4_expert) else QUANT_RULE_PRESET_LINEAR_FP8)
            )
            return mapped, (not wt_q.is_quantized), f"{checkpoint.repo_id} ({wt_q.raw_tag})", rule

    if resolved_wq == "W4A8" and not is_w4_expert:
        return "W8A8", False, "W8A8_DYNAMIC (INT8 Linear under W4A8)", QUANT_RULE_PRESET_LINEAR_FP8
    if resolved_wq == "MXFP4" and not is_mxfp4_expert and explicit_override:
        return "MXFP8", False, "MXFP8 Linear under MXFP4", QUANT_RULE_PRESET_LINEAR_FP8
    note = (
        "MXFP4 (group=32)" if resolved_wq == "MXFP4"
        else ("MXFP8 (group=32 E8M0)" if resolved_wq == "MXFP8"
              else ("W4A16 (INT4 Group-32, BF16 Act)" if resolved_wq == "W4A16"
                    else ("W4A8_DYNAMIC (INT4 Packed Expert)" if resolved_wq == "W4A8"
                          else ("W8A8_DYNAMIC (INT8)" if resolved_wq == "W8A8" else resolved_wq))))
    )
    rule = QUANT_RULE_FOLLOW_PRESET if (is_mxfp4_expert or is_w4_expert) else QUANT_RULE_PRESET_LINEAR_FP8
    return resolved_wq, (resolved_wq == "BF16"), note, rule


def _resolve_weight_placement_flags(case: Case) -> tuple[bool, bool, bool]:
    """Match ``Memory._init_weights`` resident weight placement flags."""
    meta = dict(case.runtime.metadata)
    raw_extra = meta.get("additional_config")
    extra = dict(raw_extra) if isinstance(raw_extra, Mapping) else {}
    shared_expert_dp = bool(meta.get("enable_shared_expert_dp", False))
    pcp_o_proj_shard = bool(
        meta.get(
            "enable_pcp_o_proj_weight_sharding",
            extra.get("enable_pcp_o_proj_weight_sharding", False),
        )
    )
    p = case.runtime.parallel
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
    return shared_expert_dp, pcp_o_proj_shard, dcp_q_rep


def _price_evaluated_term(
    term: WeightTerm,
    quant_spec: QuantSpec,
    placed_parallel: Any,
    ranks_per_node: int,
    quant_fmt: str,
    local_cnt: int,
) -> tuple[int, int, int, int]:
    """Compute exact per-instance ``(payload, scale, aux, total)`` bytes from canonical term math."""
    cnt = max(1, int(local_cnt))
    total_all = term_bytes(term, quant_spec, placed_parallel, ranks_per_node)
    payload_all = term_payload_bytes(term, quant_spec, placed_parallel, ranks_per_node)
    total_1 = int(round(total_all / float(cnt)))
    payload_1 = int(round(payload_all / float(cnt)))
    meta_1 = max(0, total_1 - payload_1)
    if quant_fmt in {"W8A8", "W4A8", "INT8", "INT4"} and term.category != "engram_table":
        rep_q = (term.runtime_quant or term.quant or quant_spec).for_category(term.category)
        if quant_fmt in {"W4A8", "INT4"} and int(getattr(rep_q, "expert_block", 1) or 1) > 1:
            scale_1, aux_1 = meta_1, 0
        else:
            scale_1, aux_1 = 0, meta_1
    else:
        scale_1, aux_1 = meta_1, 0
    return payload_1, scale_1, aux_1, total_1


def _append_prepared_weight_entries(
    grouped: dict[tuple[Any, ...], dict[str, Any]],
    mem: Memory,
    pp: int,
    prepared_weights: Sequence[PreparedWeightSpec],
) -> None:
    """Include persistent non-aliased ``PreparedWeightSpec`` runtime HBM copies."""
    for spec in prepared_weights:
        if not isinstance(spec, PreparedWeightSpec):
            continue
        if spec.alias_of or spec.lifetime == "load_peak_only":
            continue
        total_1 = int(prepared_weight_bytes(spec))
        if total_1 <= 0:
            continue
        shape = tuple(int(d) for d in spec.local_shape) if spec.local_shape else (1, 1)
        local_r = shape[0]
        local_c = math.prod(shape[1:]) if len(shape) > 1 else 1
        payload_1 = int(exact_element_bytes(math.prod(shape), spec.dtype)) if spec.local_shape else 0
        aux_1 = max(0, total_1 - payload_1)
        dtype_low = str(spec.dtype or "bf16").strip().lower()
        q_fmt = _QUANT_FORMAT_DISPLAY.get(dtype_low, "FP32" if "64" in dtype_low or "32" in dtype_low else "BF16")
        q_rule = QUANT_RULE_FIXED_FP32 if q_fmt == "FP32" else QUANT_RULE_FIXED_BF16
        stages: list[int] = []
        if spec.layer_indices:
            for layer_idx in spec.layer_indices:
                st = mem.layer_stage[layer_idx] if 0 <= layer_idx < len(mem.layer_stage) else pp - 1
                stages.append(st)
        else:
            stages.append(pp - 1 if spec.scope == "draft" else 0)
        key = (
            "prepared_weight", spec.allocation_id, local_r, local_c, local_r, local_c,
            "Prepared", False, False, q_fmt, payload_1, 0, aux_1,
        )
        entry = grouped.get(key)
        if entry is None:
            entry = {
                "category": "prepared_weight",
                "name": spec.allocation_id,
                "global_rows": local_r,
                "global_cols": local_c,
                "shard_rule": "Prepared",
                "axis": 0,
                "local_rows": local_r,
                "local_cols": local_c,
                "stage_repeats": [0] * pp,
                "is_engram": False,
                "is_vision": False,
                "quant": q_fmt,
                "is_excluded": True,
                "quant_note": f"Prepared runtime copy ({spec.transform}, {spec.dtype})",
                "quant_rule": q_rule,
                "payload_1": payload_1,
                "scale_1": 0,
                "aux_1": aux_1,
                "total_1": total_1,
                "group_override": _PREPARED_WEIGHT_GROUP,
            }
            grouped[key] = entry
        for st_idx in stages:
            if 0 <= st_idx < pp:
                entry["stage_repeats"][st_idx] += 1


def _weight_group_label(category: str, name: str = "") -> str:
    lower = category.lower()
    n_low = name.lower()
    if lower in _ENGRAM_CATEGORIES or "engram" in lower or "ple" in lower:
        return "Engram / PLE"
    if "vision" in lower or "projector" in lower or "image" in lower or n_low.startswith(_VISION_NAME_PREFIXES):
        return "Vision"
    if "expert" in lower or "router" in lower or "moe" in lower:
        return "MoE"
    if "mla" in lower or "indexer" in lower or ("attn" in lower and "attn_res" not in lower) or "compressor" in lower:
        return "Attention / MLA"
    if "kda" in lower or "gdn" in lower or "conv" in lower or "linear" in lower:
        return "LinearAttn"
    if "mlp" in lower or "ffn" in lower:
        return "Dense FFN"
    if lower in {"embedding", "lm_head"}:
        return "Head / Embedding"
    return "Norm / Residual"


def _collect_weight_ledger_rows(
    case: Case,
    mem: Memory,
    stage_layers: Sequence[int],
    resolved_wq: str,
    *,
    explicit_override: bool,
    include_vision_tower: bool = False,
    apply_preset_to_ignored: bool = False,
    engram_host_offload: bool = False,
    prepared_weights: Sequence[PreparedWeightSpec] = (),
) -> tuple[
    tuple[WeightLedgerRow, ...],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    int,
]:
    p = case.runtime.parallel
    shared_expert_dp, pcp_o_proj_shard, dcp_q_rep = _resolve_weight_placement_flags(case)
    stage_terms = stage_weight_terms(
        case.model, p,
        include_draft=bool(mem.draft_kind),
        include_engram=True,
        include_engram_tables=True,
        include_vision=(True if case.model.vision is not None else None),
        layer_partition=stage_layers,
        draft_kind=mem.draft_kind or None,
        slot_plans=mem.slot_plans,
    )
    engram_names = {t.name for t in engram_table_terms(case.model)}
    hw_fam = normalize_hw_family(str(case.machine.sku or case.machine.name or "a5"))
    default_ckpt = resolve_default_checkpoint(str(case.model.name or ""), hw_family=hw_fam)
    quant_spec = case.model.quant or QuantSpec("bf16")
    use_canonical_bytes = (not explicit_override) and (not apply_preset_to_ignored)
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for stage_idx, terms in enumerate(stage_terms):
        for term in terms:
            if not term.runtime_owned:
                continue
            for t_sub, placed in _expand_resident_terms(
                term, p, shared_expert_dp,
                pcp_o_proj_weight_sharding=pcp_o_proj_shard,
                dcp_q_replicate=dcp_q_rep,
                model=case.model,
            ):
                local_r, local_c, local_cnt = _sharded_shape(t_sub, placed, case.machine.ranks_per_node)
                shard_rule, axis = _resolve_shard_axis_and_rule(t_sub, placed, p)
                is_engram = (
                    t_sub.category in _ENGRAM_CATEGORIES
                    or _is_ple_category(t_sub.category)
                    or t_sub.name in engram_names
                    or "engram_table" in t_sub.name.lower()
                    or "ple_table" in t_sub.name.lower()
                )
                is_vis = _is_vision_term(t_sub)
                quant_fmt, is_excl, q_note, q_rule = _resolve_term_quant_format(
                    t_sub, case.model.quant, resolved_wq,
                    explicit_override=explicit_override,
                    apply_preset_to_ignored=apply_preset_to_ignored,
                    checkpoint=default_ckpt,
                )
                if use_canonical_bytes:
                    payload_1, scale_1, aux_1, total_1 = _price_evaluated_term(
                        t_sub, quant_spec, placed, case.machine.ranks_per_node, quant_fmt, local_cnt,
                    )
                else:
                    payload_1, scale_1, aux_1, total_1 = evaluate_weight_storage_bytes(
                        local_r, local_c, quant_fmt,
                    )
                key = (
                    t_sub.category, t_sub.name, int(t_sub.rows), int(t_sub.cols),
                    int(local_r), int(local_c), shard_rule, is_engram, is_vis,
                    quant_fmt, payload_1, scale_1, aux_1,
                )
                entry = grouped.get(key)
                if entry is None:
                    entry = {
                        "category": t_sub.category,
                        "name": t_sub.name,
                        "global_rows": int(t_sub.rows),
                        "global_cols": int(t_sub.cols),
                        "shard_rule": shard_rule,
                        "axis": axis,
                        "local_rows": int(local_r),
                        "local_cols": int(local_c),
                        "stage_repeats": [0] * p.pp,
                        "is_engram": is_engram,
                        "is_vision": is_vis,
                        "quant": quant_fmt,
                        "is_excluded": is_excl,
                        "quant_note": q_note,
                        "quant_rule": q_rule,
                        "payload_1": payload_1,
                        "scale_1": scale_1,
                        "aux_1": aux_1,
                        "total_1": total_1,
                    }
                    grouped[key] = entry
                entry["stage_repeats"][stage_idx] += max(1, int(local_cnt))

    if prepared_weights:
        _append_prepared_weight_entries(grouped, mem, int(p.pp), prepared_weights)

    return _finalize_weight_ledger_rows(
        grouped, p.pp,
        include_vision_tower=include_vision_tower,
        engram_host_offload=engram_host_offload,
    )


def _finalize_weight_ledger_rows(
    grouped: Mapping[tuple[Any, ...], dict[str, Any]],
    pp: int,
    *,
    include_vision_tower: bool,
    engram_host_offload: bool,
) -> tuple[
    tuple[WeightLedgerRow, ...],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    int,
]:
    stage_w_payload = [0.0] * pp
    stage_w_scale = [0.0] * pp
    stage_engram_total = [0.0] * pp
    stage_native_bf16 = [0.0] * pp
    stage_vision_total = [0.0] * pp
    evaluated: list[tuple[dict[str, Any], int, int, int, int, int, tuple[int, ...], int]] = []
    for entry in grouped.values():
        payload_1 = int(entry["payload_1"])
        scale_1 = int(entry["scale_1"])
        aux_1 = int(entry["aux_1"])
        total_1 = int(entry["total_1"])
        native_bf16_1 = int(entry["local_rows"]) * int(entry["local_cols"]) * BF16_BYTES
        repeats = tuple(entry["stage_repeats"])
        max_rep = max(repeats, default=0)
        for st_idx, rep in enumerate(repeats):
            if rep <= 0:
                continue
            if entry["is_engram"]:
                stage_engram_total[st_idx] += float(total_1 * rep)
            elif entry["is_vision"]:
                stage_vision_total[st_idx] += float(total_1 * rep)
                if include_vision_tower:
                    stage_w_payload[st_idx] += float(payload_1 * rep)
                    stage_w_scale[st_idx] += float((scale_1 + aux_1) * rep)
                    stage_native_bf16[st_idx] += float(native_bf16_1 * rep)
            else:
                stage_w_payload[st_idx] += float(payload_1 * rep)
                stage_w_scale[st_idx] += float((scale_1 + aux_1) * rep)
                stage_native_bf16[st_idx] += float(native_bf16_1 * rep)
        evaluated.append((entry, payload_1, scale_1, aux_1, total_1, native_bf16_1, repeats, max_rep))

    peak_stage = max(
        range(pp),
        key=lambda st: (
            stage_w_payload[st]
            + stage_w_scale[st]
            + (0.0 if engram_host_offload else stage_engram_total[st])
        ),
        default=0,
    )
    weight_rows: list[WeightLedgerRow] = []
    for entry, payload_1, scale_1, aux_1, total_1, native_bf16_1, repeats, max_rep in evaluated:
        is_hbm_row = (
            not (entry["is_engram"] and engram_host_offload)
            and not (entry["is_vision"] and not include_vision_tower)
        )
        peak_rep = repeats[peak_stage] if peak_stage < len(repeats) else 0
        row_rep = peak_rep if is_hbm_row else max_rep
        grp_lbl = str(entry.get("group_override") or _weight_group_label(entry["category"], entry["name"]))
        weight_rows.append(WeightLedgerRow(
            group=grp_lbl,
            name=entry["name"],
            category=entry["category"],
            global_rows=entry["global_rows"],
            global_cols=entry["global_cols"],
            shard_rule=entry["shard_rule"],
            shard_axis=entry["axis"],
            local_rows=entry["local_rows"],
            local_cols=entry["local_cols"],
            stage_repeats=repeats,
            max_stage_repeats=max_rep,
            quant_format=entry["quant"],
            payload_bytes=payload_1 * row_rep,
            scale_bytes=scale_1 * row_rep,
            aux_bytes=aux_1 * row_rep,
            per_die_bytes=total_1 * row_rep,
            is_engram_or_ple=entry["is_engram"],
            is_quant_excluded=entry["is_excluded"],
            quant_note=entry["quant_note"],
            native_bf16_bytes=native_bf16_1 * row_rep,
            quant_rule=entry["quant_rule"],
            is_vision=entry["is_vision"],
            peak_stage_repeats=row_rep,
        ))
    weight_rows.sort(key=lambda r: (r.group, r.name))
    return (
        tuple(weight_rows),
        stage_w_payload,
        stage_w_scale,
        stage_engram_total,
        stage_native_bf16,
        stage_vision_total,
        peak_stage,
    )


def _compute_max_hbm_concurrency(
    case: Case,
    stage_budgets: Sequence[StageBudgetRow],
    stage_per_req_cache: Sequence[float],
    fallback_conc: int,
    *,
    phase: str,
    resolved_seq: int,
) -> int:
    """Binary-search exact per-DP HBM concurrency honoring per-batch workspace growth."""
    static_fixed = [
        st.weight_payload_bytes
        + st.weight_scale_aux_bytes
        + st.engram_ple_hbm_bytes
        + st.reserve_bytes
        for st in stage_budgets
    ]
    ws_1_tok = _resolve_ledger_workspace_tokens(
        case, phase=phase, resolved_seq=resolved_seq, resolved_conc=1,
    )
    mem_1 = Memory(case, workspace_tokens=ws_1_tok)
    ws_1 = [
        float(mem_1.workspace[st.stage] if st.stage < len(mem_1.workspace) else 0.0)
        + float(mem_1.temporary_dequant[st.stage] if st.stage < len(mem_1.temporary_dequant) else 0.0)
        for st in stage_budgets
    ]
    caps: list[int] = []
    for st, base_fixed, ws_b, per_req in zip(stage_budgets, static_fixed, ws_1, stage_per_req_cache):
        avail = st.usable_hbm_budget_bytes - (base_fixed + ws_b)
        if avail <= 0.0:
            return 0
        if per_req > 0.0:
            caps.append(int(avail // per_req))
    if not caps:
        return max(0, int(fallback_conc))
    hi = min(caps)
    if hi <= 1:
        return max(0, hi)

    def _fits(cand_b: int) -> bool:
        cand_ws_tok = _resolve_ledger_workspace_tokens(
            case, phase=phase, resolved_seq=resolved_seq, resolved_conc=cand_b,
        )
        if cand_ws_tok <= ws_1_tok:
            cand_ws = ws_1
        else:
            cand_mem = Memory(case, workspace_tokens=cand_ws_tok)
            cand_ws = [
                float(cand_mem.workspace[st.stage] if st.stage < len(cand_mem.workspace) else 0.0)
                + float(cand_mem.temporary_dequant[st.stage] if st.stage < len(cand_mem.temporary_dequant) else 0.0)
                for st in stage_budgets
            ]
        return all(
            base_fixed + ws_b + float(cand_b) * per_req <= st.usable_hbm_budget_bytes
            for st, base_fixed, ws_b, per_req in zip(stage_budgets, static_fixed, cand_ws, stage_per_req_cache)
        )

    if _fits(hi):
        return hi
    lo = 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _build_stage_budget_rows(
    *,
    pp: int,
    stage_layers: list[int],
    stage_w_payload: list[float],
    stage_w_scale: list[float],
    stage_engram_total: list[float],
    stage_cache_bytes: list[float],
    stage_per_req_cache: list[float],
    ws_by_stage: list[float],
    deq_by_stage: list[float],
    reserve_bytes: float,
    usable_hbm: float,
    engram_host_offload: bool,
) -> tuple[StageBudgetRow, ...]:
    rows: list[StageBudgetRow] = []
    for st in range(pp):
        engram_hbm = 0.0 if engram_host_offload else stage_engram_total[st]
        engram_host = stage_engram_total[st] if engram_host_offload else 0.0
        ws_peak = (ws_by_stage[st] if st < len(ws_by_stage) else 0.0) + (
            deq_by_stage[st] if st < len(deq_by_stage) else 0.0
        )
        total_hbm = (
            stage_w_payload[st] + stage_w_scale[st] + engram_hbm
            + stage_cache_bytes[st] + ws_peak + reserve_bytes
        )
        rows.append(StageBudgetRow(
            stage=st,
            owned_layers=stage_layers[st],
            weight_payload_bytes=stage_w_payload[st],
            weight_scale_aux_bytes=stage_w_scale[st],
            engram_ple_hbm_bytes=engram_hbm,
            engram_ple_host_bytes=engram_host,
            cache_bytes=stage_cache_bytes[st],
            workspace_peak_bytes=ws_peak,
            reserve_bytes=reserve_bytes,
            total_hbm_bytes=total_hbm,
            usable_hbm_budget_bytes=usable_hbm,
            headroom_bytes=usable_hbm - total_hbm,
            fits_hbm=total_hbm <= usable_hbm,
            per_request_cache_bytes=stage_per_req_cache[st] if st < len(stage_per_req_cache) else 0.0,
        ))
    return tuple(rows)


def _resolve_ledger_workspace_tokens(
    case: Case, *, phase: str, resolved_seq: int, resolved_conc: int,
) -> int:
    rt_meta = dict(case.runtime.metadata)
    spec_tokens = int(rt_meta.get("draft_proposal_tokens", 0) or 0)
    role_norm = str(rt_meta.get("deployment_role") or rt_meta.get("role") or "").strip().lower()
    prompts = [int(getattr(req, "prompt_tokens", 0) or 0) for req in (getattr(case.workload, "requests", ()) or ())]
    raw_prompt = max(prompts) if prompts else resolved_seq
    prefill_ws = (
        min(raw_prompt, int(case.runtime.max_num_batched_tokens))
        if case.runtime.enable_chunked_prefill and int(case.runtime.max_num_batched_tokens) > 0
        else raw_prompt
    )
    decode_ws = max(1, resolved_conc * max(1, 1 + spec_tokens))
    if role_norm == "pd_prefill" or str(phase or "").strip().lower() == "prefill":
        return max(1, prefill_ws)
    if role_norm == "pd_decode":
        return decode_ws
    return max(1, prefill_ws, decode_ws)


def _resolve_ledger_parameters(
    case: Case,
    *,
    phase: str,
    seq_len: int | None,
    concurrency: int | None,
    weight_quant: str | None,
    cache_dtype: str | None,
    engram_host_offload: bool | None,
    include_vision_tower: bool | None,
    apply_preset_to_ignored: bool,
    prepared_weights: Sequence[PreparedWeightSpec],
) -> dict[str, Any]:
    """Validate and resolve normalized quant, cache, vision, and workspace parameters."""
    resolved_seq = max(1, int(seq_len if seq_len is not None else case.runtime.max_model_len))
    resolved_conc = max(1, int(concurrency if concurrency is not None else max(1, len(case.workload.requests))))
    explicit_override = weight_quant is not None
    resolved_wq = (
        _canonicalize_weight_quant_override(weight_quant)
        if weight_quant is not None else _normalize_quant_preset(case.model.quant)
    )
    if resolved_wq not in WEIGHT_QUANT_FORMATS:
        raise ModelingError(f"unsupported_weight_quant:{resolved_wq}")
    wq_display, q_exclusions = _describe_quant_display(
        case.model.quant, resolved_wq,
        explicit_override=explicit_override,
        apply_preset_to_ignored=bool(apply_preset_to_ignored),
    )
    resolved_cd = (
        _canonicalize_cache_dtype_override(cache_dtype)
        if cache_dtype is not None else _normalize_cache_dtype(case.model)
    )
    if resolved_cd not in SUPPORTED_CACHE_DTYPES:
        raise ModelingError(f"unsupported_cache_dtype:{resolved_cd}")
    hw_fam = normalize_hw_family(str(case.machine.sku or case.machine.name or "a5"))
    if weight_quant is not None or cache_dtype is not None:
        validate_quant_for_hardware(
            resolved_wq if weight_quant is not None else "BF16",
            hw_fam,
            kv_cache_dtype=(resolved_cd if cache_dtype is not None else "BF16"),
            strict=True,
        )
    resolved_offload = (
        bool(case.runtime.engram_cpu_offload)
        if engram_host_offload is None else bool(engram_host_offload)
    )
    rt_meta = dict(case.runtime.metadata)
    explicit_vit = None if include_vision_tower is None else bool(include_vision_tower)
    block_size = int(
        getattr(case.runtime, "kv_block_size", None)
        or getattr(case.runtime, "block_size", None)
        or DEFAULT_CACHE_BLOCK_SIZE
    )
    spec_tokens = int(rt_meta.get("draft_proposal_tokens", 0) or 0)
    ws_tokens = _resolve_ledger_workspace_tokens(
        case, phase=phase, resolved_seq=resolved_seq, resolved_conc=resolved_conc,
    )
    meta_prepared = [
        item for item in (rt_meta.get("prepared_weights") or ())
        if isinstance(item, PreparedWeightSpec)
    ]
    resolved_prepared = tuple(dict.fromkeys((*meta_prepared, *prepared_weights)))
    return {
        "rt_meta": rt_meta,
        "resolved_seq": resolved_seq,
        "resolved_conc": resolved_conc,
        "explicit_override": explicit_override,
        "resolved_wq": resolved_wq,
        "wq_display": wq_display,
        "q_exclusions": q_exclusions,
        "resolved_cd": resolved_cd,
        "resolved_offload": resolved_offload,
        "explicit_vit": explicit_vit,
        "block_size": block_size,
        "spec_tokens": spec_tokens,
        "ws_tokens": ws_tokens,
        "resolved_prepared": resolved_prepared,
    }


def _extract_receipt_weight_totals(
    *,
    pp: int,
    receipt: Any,
    alloc_weights_by_stage: Sequence[float],
    stage_budgets: Sequence[StageHbmBudgetRow],
    stage_native_bf16: Sequence[float],
    stage_engram_total: Sequence[float],
    stage_vision_total: Sequence[float],
    peak_weight_stage: int,
    resolved_offload: bool,
) -> tuple[tuple[float, ...], float, float, float, float]:
    """Extract per-stage resident weight receipt totals and peak compression metrics."""
    receipt_weights_by_stage: list[float] = []
    for st in range(pp):
        st_die = next(
            (d for d in receipt.dies if d.replica == 0 and d.stage == st and d.rank == 0),
            None,
        )
        if st_die is not None:
            receipt_weights_by_stage.append(
                float(st_die.checkpoint_weight_bytes + st_die.vision_weight_bytes + st_die.prepared_weight_bytes)
            )
        else:
            receipt_weights_by_stage.append(float(alloc_weights_by_stage[st]))
    peak_budget = stage_budgets[peak_weight_stage] if peak_weight_stage < len(stage_budgets) else None
    weight_footprint = (
        peak_budget.weight_payload_bytes + peak_budget.weight_scale_aux_bytes + peak_budget.engram_ple_hbm_bytes
        if peak_budget is not None else 0.0
    )
    native_bf16_footprint = (
        stage_native_bf16[peak_weight_stage] + (0.0 if resolved_offload else stage_engram_total[peak_weight_stage])
        if peak_weight_stage < len(stage_native_bf16) else 0.0
    )
    vision_bytes = max(stage_vision_total, default=0.0)
    comp_ratio = (native_bf16_footprint / weight_footprint) if weight_footprint > 0.0 else 1.0
    return tuple(receipt_weights_by_stage), weight_footprint, native_bf16_footprint, vision_bytes, comp_ratio


def compute_memory_quant_ledger(
    case: Case,
    *,
    phase: str = "decode",
    seq_len: int | None = None,
    concurrency: int | None = None,
    weight_quant: str | None = None,
    cache_dtype: str | None = None,
    engram_host_offload: bool | None = None,
    include_vision_tower: bool | None = None,
    apply_preset_to_ignored: bool = False,
    prepared_weights: Sequence[PreparedWeightSpec] = (),
) -> MemoryCalculatorLedger:
    """Derive closed-form per-die Weight/Quant/Cache/Engram-PLE/PP HBM budget ledger."""
    p = case.runtime.parallel
    stage_layers = stage_layer_counts(case)
    cfg = _resolve_ledger_parameters(
        case,
        phase=phase,
        seq_len=seq_len,
        concurrency=concurrency,
        weight_quant=weight_quant,
        cache_dtype=cache_dtype,
        engram_host_offload=engram_host_offload,
        include_vision_tower=include_vision_tower,
        apply_preset_to_ignored=apply_preset_to_ignored,
        prepared_weights=prepared_weights,
    )
    resolved_seq, resolved_conc = cfg["resolved_seq"], cfg["resolved_conc"]
    resolved_wq, resolved_cd = cfg["resolved_wq"], cfg["resolved_cd"]
    resolved_offload = cfg["resolved_offload"]
    resolved_prepared = cfg["resolved_prepared"]

    mem = Memory(case, workspace_tokens=cfg["ws_tokens"])
    resolved_vit = (
        bool(mem.vision_accounting == "full_checkpoint")
        if cfg["explicit_vit"] is None else bool(cfg["explicit_vit"])
    )
    if resolved_prepared:
        mem.register_prepared_weights(resolved_prepared)
    alloc = mem.allocation_report(resolved_seq)
    receipt = mem.resident_memory_receipt(
        phase=phase,
        sequence_tokens=resolved_seq,
        batch_per_dp=resolved_conc,
        vision_accounting=("full_checkpoint" if resolved_vit else "text_backbone"),
        prepared_weights=resolved_prepared,
    )
    (
        weight_rows, stage_w_payload, stage_w_scale, stage_engram_total,
        stage_native_bf16, stage_vision_total, peak_weight_stage,
    ) = _collect_weight_ledger_rows(
        case, mem, stage_layers, resolved_wq,
        explicit_override=cfg["explicit_override"],
        include_vision_tower=resolved_vit,
        apply_preset_to_ignored=bool(apply_preset_to_ignored),
        engram_host_offload=resolved_offload,
        prepared_weights=resolved_prepared,
    )
    local_tokens, cache_rows, stage_cache_bytes, stage_per_req_cache = _compute_cache_ledger_rows(
        case, mem, alloc["cache"],
        stage_layers=stage_layers,
        cache_dtype=resolved_cd,
        seq_len=resolved_seq,
        concurrency=resolved_conc,
        block_size=cfg["block_size"],
        spec_tokens=cfg["spec_tokens"],
    )
    die_hbm = float(min(d.hbm_bytes for d in case.machine.dies))
    util = float(case.runtime.gpu_memory_utilization)
    usable_hbm = die_hbm * util
    stage_budgets = _build_stage_budget_rows(
        pp=int(p.pp),
        stage_layers=stage_layers,
        stage_w_payload=stage_w_payload,
        stage_w_scale=stage_w_scale,
        stage_engram_total=stage_engram_total,
        stage_cache_bytes=stage_cache_bytes,
        stage_per_req_cache=stage_per_req_cache,
        ws_by_stage=[float(x) for x in alloc["workspace"]["total_bytes_per_stage_rank"]],
        deq_by_stage=[float(x) for x in alloc["temporary_dequant"]["bytes_per_stage_rank"]],
        reserve_bytes=float(alloc["reserve"]["total_bytes_per_rank"]),
        usable_hbm=usable_hbm,
        engram_host_offload=resolved_offload,
    )

    local_tp = max(
        int(getattr(p, "effective_o_proj_tp", p.tp) or p.tp),
        int(getattr(p, "q_proj_tp", 1) or 1),
        int(getattr(p, "dcp_linear_tp", 1) or 1),
    )
    shared_tp = 1 if bool(cfg["rt_meta"].get("enable_shared_expert_dp", False)) else int(p.tp)
    stride_reserves = mem._unified_pool_stride_reserve_by_stage(resolved_seq)
    sim_cache_by_stage = tuple(
        float(v) + (float(stride_reserves[idx]) if idx < len(stride_reserves) else 0.0)
        for idx, v in enumerate(mem.dp_cohort_cache_bytes(resolved_seq, padded=True))
    )
    receipt_weights_by_stage, weight_footprint, native_bf16_footprint, vision_bytes, comp_ratio = (
        _extract_receipt_weight_totals(
            pp=int(p.pp),
            receipt=receipt,
            alloc_weights_by_stage=alloc["weights"]["total_bytes_per_stage_rank"],
            stage_budgets=stage_budgets,
            stage_native_bf16=stage_native_bf16,
            stage_engram_total=stage_engram_total,
            stage_vision_total=stage_vision_total,
            peak_weight_stage=peak_weight_stage,
            resolved_offload=resolved_offload,
        )
    )
    max_conc = _compute_max_hbm_concurrency(
        case, stage_budgets, stage_per_req_cache, resolved_conc,
        phase=phase, resolved_seq=resolved_seq,
    )
    return MemoryCalculatorLedger(
        model_name=str(case.model.name),
        hardware_sku=str(case.machine.sku or case.machine.name),
        phase=phase,
        weight_quant=resolved_wq,
        cache_dtype=resolved_cd,
        engram_host_offload=resolved_offload,
        dp=int(p.dp), tp=int(p.tp), ep=int(p.ep), pp=int(p.pp),
        dcp=int(p.cp), pcp=int(p.pcp), local_tp=int(local_tp), shared_tp=int(shared_tp),
        total_dies=len(case.machine.dies),
        seq_len=resolved_seq, concurrency=resolved_conc,
        block_size=cfg["block_size"], spec_tokens=cfg["spec_tokens"], local_cache_tokens=local_tokens,
        die_hbm_capacity_bytes=die_hbm, gpu_memory_utilization=util, usable_hbm_bytes=usable_hbm,
        weight_rows=weight_rows,
        cache_rows=cache_rows,
        stage_budgets=tuple(stage_budgets),
        max_stage_hbm_bytes=max((row.total_hbm_bytes for row in stage_budgets), default=0.0),
        max_stage_host_offload_bytes=max((row.engram_ple_host_bytes for row in stage_budgets), default=0.0),
        fits_hbm=all(row.fits_hbm for row in stage_budgets),
        sim_memory_weight_bytes_by_stage=receipt_weights_by_stage,
        sim_memory_cache_bytes_by_stage=sim_cache_by_stage,
        weight_quant_display=cfg["wq_display"],
        quant_exclusions=cfg["q_exclusions"],
        weight_footprint_per_die_bytes=weight_footprint,
        per_request_cache_bytes_max_stage=max(stage_per_req_cache, default=0.0),
        max_hbm_concurrency_per_dp=max_conc,
        include_vision_tower=resolved_vit,
        apply_preset_to_ignored=bool(apply_preset_to_ignored),
        native_bf16_weight_footprint_per_die_bytes=native_bf16_footprint,
        vision_weight_bytes_per_die=vision_bytes,
        weight_compression_ratio=comp_ratio,
        peak_weight_stage=peak_weight_stage,
    )
