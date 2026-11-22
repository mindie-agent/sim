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
)
from mindie_sim.catalog.npu_checkpoints import (
    NpuCheckpointSpec,
    classify_weight_tensor_quant,
    normalize_hw_family,
    resolve_default_checkpoint,
)
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import WeightTerm, _sharded_shape
from mindie_sim.catalog.weights import engram_table_terms, stage_weight_terms
from mindie_sim.prediction.capacity import Memory
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError

from .models import (
    BF16_BYTES,
    FP32_BYTES,
    KV_QUANT_GROUP_SIZE,
    MAX_EXPLICIT_LAYER_LABELS,
    MIN_DENOM_EPS,
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
_MXFP4_EXPERT_CATEGORIES = frozenset({"expert", "routed_expert", "moe.routed", "latent_moe", "attn_res_proj"})
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
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
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
    combined = f"{name} {exp_fmt} {w_fmt}"
    if "mxfp4" in combined or "mx_fp4" in combined:
        return "MXFP4"
    if "mxfp8" in combined or "mx_fp8" in combined:
        return "MXFP8"
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
    quant = getattr(model, "quant", None)
    raw = str(
        (getattr(quant, "kv_dtype", "") if quant is not None else "")
        or getattr(model, "kv_dtype", "")
        or "bf16"
    ).strip().lower()
    if "int8" in raw or raw == "c8":
        return "C8"
    if "fp8" in raw:
        return "FP8"
    return "BF16"


def _resolve_shard_axis_and_rule(term: WeightTerm) -> tuple[str, int]:
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


def _cache_dtype_override_ratio(model: Any, target_dtype: str) -> float:
    base = _normalize_cache_dtype(model)
    target = str(target_dtype or base).strip().upper()
    if target == base:
        return 1.0
    width_by_dtype = {
        "BF16": float(BF16_BYTES),
        "C8": 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE),
        "FP8": 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE),
    }
    base_w = width_by_dtype.get(base, float(BF16_BYTES))
    target_w = width_by_dtype.get(target, float(BF16_BYTES))
    return target_w / max(base_w, MIN_DENOM_EPS)


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
})


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

    scale_override = _cache_dtype_override_ratio(model, cache_dtype)
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
    for st_idx in range(n_stages):
        entry = comp_entries[st_idx] if st_idx < len(comp_entries) else {}
        for comp_name, detail in (entry.get("allocated") or {}).items():
            val = float(detail.get("total", 0.0) or 0.0)
            if val <= 0.0:
                continue
            scalable = comp_name not in fixed_state_comps
            comp_scalable[comp_name] = scalable
            comp_stage_bytes[comp_name][st_idx] += val * (scale_override if scalable else 1.0)
        for comp_name, detail in (entry.get("group_allocated") or {}).items():
            val = float(detail.get("alloc_payload", 0.0) or 0.0) + float(detail.get("alloc_scale", 0.0) or 0.0)
            if val <= 0.0:
                continue
            scalable = comp_name not in fixed_state_comps
            comp_scalable[comp_name] = scalable
            comp_stage_bytes[comp_name][st_idx] += val * (scale_override if scalable else 1.0)
        if st_idx < len(reserves) and float(reserves[st_idx] or 0.0) > 0.0:
            comp_scalable["page_group_reserve"] = False
            comp_stage_bytes["page_group_reserve"][st_idx] += float(reserves[st_idx])
            comp_stage_layers["page_group_reserve"][st_idx] = max(1, int(stage_layers[st_idx]))
        if st_idx < len(metadatas) and float(metadatas[st_idx] or 0.0) > 0.0:
            comp_scalable["page_metadata"] = False
            comp_stage_bytes["page_metadata"][st_idx] += float(metadatas[st_idx])
            comp_stage_layers["page_metadata"][st_idx] = max(1, int(stage_layers[st_idx]))

    # Account for DP-cohort DCP amplification when dcp_dp_width > 1 so per-DP cache matches Memory;
    # the cohort's extra sharded KV bytes take the same cache-dtype width ratio as the local KV.
    if int(getattr(p, "dcp_dp_width", 1) or 1) > 1:
        cohort_totals = mem.dp_cohort_cache_bytes(max(1, seq_len), padded=True)
        base_totals = mem.cache_bytes(max(1, seq_len), padded=True)
        for st_idx in range(min(n_stages, len(cohort_totals), len(base_totals))):
            extra = (float(cohort_totals[st_idx]) - float(base_totals[st_idx])) * scale_override
            if extra > MIN_DENOM_EPS:
                comp_scalable["dcp_dp_cohort_kv"] = True
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
    for comp_name in sorted(comp_stage_bytes):
        per_stage = comp_stage_bytes[comp_name]
        layers_per_stage = tuple(comp_stage_layers[comp_name])
        max_layers = max(layers_per_stage, default=0)
        max_req_bytes = max(per_stage, default=0.0)
        scalable = comp_scalable.get(comp_name, True) and (max_layers > 0)
        b_per_tok = (
            max_req_bytes / float(max(1, max_layers * local_tokens))
            if scalable else 0.0
        )
        for st_idx, st_val in enumerate(per_stage):
            stage_per_req_totals[st_idx] += st_val
            stage_totals[st_idx] += st_val * float(concurrency)
        rows.append(CacheLedgerRow(
            component=comp_name,
            cache_kind=cache_dtype if scalable else "State (Fixed)",
            active_layers_per_stage=layers_per_stage,
            max_stage_layers=max_layers,
            bytes_per_token_per_layer=b_per_tok,
            per_request_bytes=max_req_bytes,
            batch_total_bytes=max_req_bytes * float(concurrency),
            scales_with_tokens=scalable,
            per_stage_bytes=tuple(float(v) for v in per_stage),
        ))
    return local_tokens, tuple(rows), stage_totals, stage_per_req_totals


def _workload_has_media(case: Case) -> bool:
    for req in getattr(case.workload, "requests", ()) or ():
        if (
            int(getattr(req, "image_tokens", 0) or 0) > 0
            or int(getattr(req, "media_tokens", 0) or 0) > 0
            or int(getattr(req, "images", 0) or 0) > 0
            or bool(getattr(req, "media", None))
        ):
            return True
    return False


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
    if fam in _FIXED_FP32_CATEGORIES or cat in _FIXED_FP32_CATEGORIES:
        return "FP32", True, "FP32 param", QUANT_RULE_FIXED_FP32
    if fam in _FIXED_BF16_CATEGORIES or cat in _FIXED_BF16_CATEGORIES or _is_ple_category(cat) or _is_ple_category(fam):
        note = "Vision ViT BF16" if _is_vision_term(term) else "Native BF16 structural/head param"
        return "BF16", True, note, QUANT_RULE_FIXED_BF16

    is_w4_expert = cat in _W4A8_EXPERT_ONLY_CATEGORIES or fam in _W4A8_EXPERT_ONLY_CATEGORIES
    is_mxfp4_expert = cat in _MXFP4_EXPERT_CATEGORIES or fam in _MXFP4_EXPERT_CATEGORIES
    term_quant = term.runtime_quant or term.quant or model_quant
    if term_quant is not None:
        exclusions = set(getattr(term_quant, "exclusions", ()) or ())
        if cat in exclusions or fam in exclusions:
            if apply_preset_to_ignored:
                over_fmt = resolved_wq if is_mxfp4_expert else _linear_format_for_preset(resolved_wq)
                return over_fmt, False, f"Override ignore -> {over_fmt}", QUANT_RULE_CHECKPOINT_IGNORE
            native = _QUANT_FORMAT_DISPLAY.get(str(term_quant.native_format or "bf16").lower(), "BF16")
            return native, True, "quantization_config.ignore -> BF16", QUANT_RULE_CHECKPOINT_IGNORE
        if not explicit_override:
            eff_raw = str(term_quant.format_for(cat) or "bf16").lower()
            mapped = _QUANT_FORMAT_DISPLAY.get(eff_raw, resolved_wq)
            if resolved_wq == "W4A8" and not is_w4_expert and mapped == "W4A8":
                mapped = "W8A8"
            note = (
                "MXFP4 (group=32)" if mapped == "MXFP4"
                else ("W4A8_DYNAMIC (INT4 Packed Expert)" if mapped == "W4A8"
                      else ("W8A8_DYNAMIC (INT8 Linear)" if mapped == "W8A8" else mapped))
            )
            rule = (
                QUANT_RULE_PRESET_LINEAR_FP8
                if (mapped in {"FP8", "MXFP8", "W8A8"} and resolved_wq in {"MXFP4", "W4A8"} and not is_w4_expert)
                else QUANT_RULE_FOLLOW_PRESET
            )
            return mapped, (mapped == "BF16"), note, rule
    elif checkpoint is not None and not explicit_override:
        wt_q = classify_weight_tensor_quant(term.name, checkpoint)
        if wt_q.ledger_format in WEIGHT_QUANT_FORMATS:
            mapped = wt_q.ledger_format
            rule = (
                QUANT_RULE_PRESET_LINEAR_FP8
                if (mapped in {"FP8", "MXFP8", "W8A8"} and resolved_wq in {"MXFP4", "W4A8"} and not is_w4_expert)
                else (QUANT_RULE_CHECKPOINT_IGNORE if not wt_q.is_quantized else QUANT_RULE_FOLLOW_PRESET)
            )
            return mapped, (not wt_q.is_quantized), f"{checkpoint.repo_id} ({wt_q.raw_tag})", rule

    if resolved_wq == "W4A8" and not is_w4_expert:
        return "W8A8", False, "W8A8_DYNAMIC (INT8 Linear under W4A8)", QUANT_RULE_PRESET_LINEAR_FP8
    if resolved_wq == "MXFP4" and not is_mxfp4_expert and explicit_override:
        return "MXFP8", False, "MXFP8 Linear under MXFP4", QUANT_RULE_PRESET_LINEAR_FP8
    note = (
        "MXFP4 (group=32)" if resolved_wq == "MXFP4"
        else ("W4A8_DYNAMIC (INT4 Packed Expert)" if resolved_wq == "W4A8"
              else ("W8A8_DYNAMIC (INT8)" if resolved_wq == "W8A8" else resolved_wq))
    )
    return resolved_wq, (resolved_wq == "BF16"), note, QUANT_RULE_FOLLOW_PRESET


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
    stage_terms = stage_weight_terms(
        case.model, p,
        include_draft=bool(mem.draft_kind),
        include_engram=True,
        include_engram_tables=True,
        layer_partition=stage_layers,
        draft_kind=mem.draft_kind or None,
        slot_plans=mem.slot_plans,
    )
    engram_names = {t.name for t in engram_table_terms(case.model)}
    hw_fam = normalize_hw_family(str(case.machine.sku or case.machine.name or "a5"))
    default_ckpt = resolve_default_checkpoint(str(case.model.name or ""), hw_family=hw_fam)
    grouped: dict[tuple[str, str, int, int, str, bool, bool], dict[str, Any]] = {}
    for stage_idx, terms in enumerate(stage_terms):
        for term in terms:
            if not term.runtime_owned:
                continue
            local_r, local_c, local_cnt = _sharded_shape(term, p, case.machine.ranks_per_node)
            shard_rule, axis = _resolve_shard_axis_and_rule(term)
            is_engram = (
                term.category in _ENGRAM_CATEGORIES
                or _is_ple_category(term.category)
                or term.name in engram_names
                or "engram_table" in term.name.lower()
                or "ple_table" in term.name.lower()
            )
            is_vis = _is_vision_term(term)
            key = (term.category, term.name, int(term.rows), int(term.cols), shard_rule, is_engram, is_vis)
            entry = grouped.get(key)
            if entry is None:
                quant_fmt, is_excl, q_note, q_rule = _resolve_term_quant_format(
                    term, case.model.quant, resolved_wq,
                    explicit_override=explicit_override,
                    apply_preset_to_ignored=apply_preset_to_ignored,
                    checkpoint=default_ckpt,
                )
                entry = {
                    "category": term.category,
                    "name": term.name,
                    "global_rows": int(term.rows),
                    "global_cols": int(term.cols),
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
                }
                grouped[key] = entry
            entry["stage_repeats"][stage_idx] += max(1, int(local_cnt))

    stage_w_payload = [0.0] * p.pp
    stage_w_scale = [0.0] * p.pp
    stage_engram_total = [0.0] * p.pp
    stage_native_bf16 = [0.0] * p.pp
    stage_vision_total = [0.0] * p.pp
    evaluated: list[tuple[dict[str, Any], int, int, int, int, int, tuple[int, ...], int]] = []
    for entry in grouped.values():
        payload_1, scale_1, aux_1, total_1 = evaluate_weight_storage_bytes(
            entry["local_rows"], entry["local_cols"], entry["quant"],
        )
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
        range(p.pp),
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
        weight_rows.append(WeightLedgerRow(
            group=_weight_group_label(entry["category"], entry["name"]),
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
) -> MemoryCalculatorLedger:
    """Derive closed-form per-die Weight/Quant/Cache/Engram-PLE/PP HBM budget ledger."""
    p = case.runtime.parallel
    stage_layers = stage_layer_counts(case)
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
    resolved_offload = (
        bool(case.runtime.engram_cpu_offload)
        if engram_host_offload is None else bool(engram_host_offload)
    )
    resolved_vit = (
        _workload_has_media(case)
        if include_vision_tower is None else bool(include_vision_tower)
    )
    block_size = int(
        getattr(case.runtime, "kv_block_size", None)
        or getattr(case.runtime, "block_size", None)
        or DEFAULT_CACHE_BLOCK_SIZE
    )
    spec_tokens = int(dict(case.runtime.metadata).get("draft_proposal_tokens", 0) or 0)
    ws_tokens = _resolve_ledger_workspace_tokens(
        case, phase=phase, resolved_seq=resolved_seq, resolved_conc=resolved_conc,
    )
    mem = Memory(case, workspace_tokens=ws_tokens)
    alloc = mem.allocation_report(resolved_seq)
    (
        weight_rows,
        stage_w_payload,
        stage_w_scale,
        stage_engram_total,
        stage_native_bf16,
        stage_vision_total,
        peak_weight_stage,
    ) = _collect_weight_ledger_rows(
        case, mem, stage_layers, resolved_wq,
        explicit_override=explicit_override,
        include_vision_tower=resolved_vit,
        apply_preset_to_ignored=bool(apply_preset_to_ignored),
        engram_host_offload=resolved_offload,
    )
    local_tokens, cache_rows, stage_cache_bytes, stage_per_req_cache = _compute_cache_ledger_rows(
        case,
        mem,
        alloc["cache"],
        stage_layers=stage_layers,
        cache_dtype=resolved_cd,
        seq_len=resolved_seq,
        concurrency=resolved_conc,
        block_size=block_size,
        spec_tokens=spec_tokens,
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
    shared_dp = bool(dict(case.runtime.metadata).get("enable_shared_expert_dp", False))
    shared_tp = 1 if shared_dp else int(p.tp)
    sim_cache_by_stage = tuple(
        float(v) for v in mem.dp_cohort_cache_bytes(resolved_seq, padded=True)
    )
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
        block_size=block_size, spec_tokens=spec_tokens, local_cache_tokens=local_tokens,
        die_hbm_capacity_bytes=die_hbm, gpu_memory_utilization=util, usable_hbm_bytes=usable_hbm,
        weight_rows=weight_rows,
        cache_rows=cache_rows,
        stage_budgets=tuple(stage_budgets),
        max_stage_hbm_bytes=max((row.total_hbm_bytes for row in stage_budgets), default=0.0),
        max_stage_host_offload_bytes=max((row.engram_ple_host_bytes for row in stage_budgets), default=0.0),
        fits_hbm=all(row.fits_hbm for row in stage_budgets),
        sim_memory_weight_bytes_by_stage=tuple(float(x) for x in alloc["weights"]["total_bytes_per_stage_rank"]),
        sim_memory_cache_bytes_by_stage=sim_cache_by_stage,
        weight_quant_display=wq_display,
        quant_exclusions=q_exclusions,
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
