"""Pure per-die memory ledger projection from canonical ``DieResidentReceipt``.

[Role] Project ``Memory.cache_ledger_projection`` (``DieResidentReceipt`` +
  ``HybridPoolReceipt``) and canonical ``stage_weight_terms`` into the frozen
  ``MemoryCalculatorLedger`` consumed by ``ExportBundle``, ``agent_brief``, and
  Sheet 3 (``Memory_Quant_Calculator``).
[Invariants] Strictly pure analytical projection over ``Case`` and ``Memory``:
  every stage weight, cache, workspace, reserve, usable HBM, and total HBM
  field satisfies exact 0 B multi-key bijection with the bottleneck die's
  ``DieResidentReceipt``. Fixed per-die static C8 channel scales are never
  divided by or multiplied by concurrency; hardware ISA legality is enforced
  upfront via ``validate_quant_for_hardware``; zero shadow HBM calculators and
  zero AG014 model-name branches.
[Timeouts & Locks] Single-pass O(Unique Weight Terms + Legal Dtypes x Layers)
  in-memory arithmetic; no locks or I/O.
[Side-Effect Policy] Read-only over caller ``Case`` instances.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

from mindie_sim.catalog.cache import DEFAULT_CACHE_BLOCK_SIZE
from mindie_sim.catalog.draft_weights import require_model_quant
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
from mindie_sim.catalog.weights import _expand_resident_terms, stage_weight_terms
from mindie_sim.prediction.capacity import Memory, prepared_weight_bytes
from mindie_sim.prediction.capacity.ledger import CacheLedgerProjection
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.stack import PreparedWeightSpec

from .models import (
    BF16_BYTES,
    FP32_BYTES,
    KV_QUANT_GROUP_SIZE,
    MIN_DENOM_EPS,
    MXFP_GROUP_SIZE,
    QUANT_RULE_CHECKPOINT_IGNORE,
    QUANT_RULE_FIXED_BF16,
    QUANT_RULE_FIXED_FP32,
    QUANT_RULE_FOLLOW_PRESET,
    QUANT_RULE_PRESET_LINEAR_FP8,
    QUANT_RULE_PRESET_LINEAR_W8A8,
    WEIGHT_QUANT_FORMATS,
    CacheLedgerRow,
    MemoryCalculatorLedger,
    StageBudgetRow,
    WeightLedgerRow,
)

#: Canonical KV-cache effective element widths (bytes/element) for format queries.
C8_CACHE_WIDTH_BYTES = 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE)
GQA_STATIC_C8_CACHE_WIDTH_BYTES = 1.0
FP8_CACHE_WIDTH_BYTES = 1.0 + float(BF16_BYTES) / float(KV_QUANT_GROUP_SIZE)
MXFP8_CACHE_WIDTH_BYTES = 1.0 + 1.0 / float(MXFP_GROUP_SIZE)

#: Cache components whose byte width responds to target KV cache dtype overrides.
KV_DTYPE_SENSITIVE_COMPONENTS = frozenset({
    "gqa_kv", "mla_kv", "window", "compressed_kv", "dcp_dp_cohort_kv",
})
_GQA_CACHE_COMPONENTS = frozenset({"gqa_kv", "window"})
_PREPARED_WEIGHT_GROUP = "Prepared Runtime Weights"
_FIXED_PER_DIE_COMPONENTS = frozenset({"c8_static_channel_scale"})
_NON_TOKEN_SCALED_COMPONENTS = frozenset({
    "recurrent_state", "conv_state", "ple_conv_state", "replay_records",
    "state_transient", "compressor_state", "index_compressor_state",
    "index_pool_tail", "draft_state", "page_group_reserve", "page_metadata",
    "c8_static_channel_scale",
})
_COMPONENT_DISPLAY_ORDER = (
    "gqa_kv", "mla_kv", "window", "compressed_kv",
    "index_k", "index_pool_tail", "compressor_state", "index_compressor_state",
    "recurrent_state", "conv_state", "ple_conv_state",
    "replay_records", "state_transient",
    "draft_kv", "draft_state",
    "dcp_dp_cohort_kv", "page_group_reserve", "page_metadata",
    "c8_static_channel_scale",
)
_ENGRAM_CATEGORIES = frozenset({"engram_table", "ple_table"})
_VISION_CATEGORIES = frozenset({"vision", "projector", "vision_pos", "image_token"})
_VISION_NAME_PREFIXES = ("vision_", "aligner_", "merger_", "projector_", "image_")
_EXPERT_CATEGORIES = frozenset({"expert", "routed_expert", "moe.routed"})
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
_FLOAT16_FORMATS = frozenset({"bf16", "fp16", "bfloat16", "float16"})
_MXFP8_KV_TAG = "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"
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
    "w8a8_dynamic": "W8A8",
    "w8a8_mix": "W8A8",
    "int8": "INT8",
    "w4a16": "W4A16",
    "compressed_tensors_w4a16": "W4A16",
    "w4a16_compressed": "W4A16",
    "w4a16_awq": "W4A16",
    "w4a16_gptq": "W4A16",
    "w4a8": "W4A8",
    "w4a8_dynamic": "W4A8",
    "int4": "INT4",
    "mxfp8": "MXFP8",
    "w8a8_mxfp8": "MXFP8",
    "mxfp4": "MXFP4",
    "w4a8_mxfp4": "MXFP4",
    "w4a4_mxfp4": "MXFP4",
    "fp4_e2m1": "MXFP4",
}
_CACHE_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "int8": "C8",
    "c8": "C8",
    "kv_c8": "C8",
    "fa3_c8": "C8",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
    "mxfp8": "MXFP8",
    "qk_mxfp8_dynamic_v_mxfp8_per_channel": "MXFP8",
}
_CANONICAL_CACHE_TO_KV_DTYPE: dict[str, str] = {
    "BF16": "bf16",
    "C8": "int8",
    "FP8": "fp8_e4m3",
    "MXFP8": "fp8_e4m3",
}
_W4A16_SCALE_BYTES = 2 * BF16_BYTES
_HW_FAMILY_A5 = "a5"
_HW_FAMILY_A23 = frozenset({"a2", "a3"})
_DEFAULT_CONCURRENCY = 1


def _is_ple_category(cat: str) -> bool:
    low = str(cat or "").lower()
    return low == "ple" or low.endswith("_ple")


def _is_vision_term(term: WeightTerm) -> bool:
    cat = str(term.category or "").lower()
    name = str(term.name or "").lower()
    return cat in _VISION_CATEGORIES or name.startswith(_VISION_NAME_PREFIXES)


def evaluate_weight_storage_bytes(
    local_rows: int,
    local_cols: int,
    quant_format: str,
    *,
    repeats: int = 1,
    scale_bytes_per_group: int | None = None,
) -> tuple[int, int, int, int]:
    """Exact closed-form integer ``(payload, mx_scale, channel_aux, total)`` bytes."""
    fmt = _canonicalize_weight_quant_override(quant_format)
    if fmt not in WEIGHT_QUANT_FORMATS:
        raise ModelingError(f"unsupported_weight_quant_format:{quant_format!r}")
    spec = WEIGHT_QUANT_FORMATS[fmt]
    rows = max(0, int(local_rows))
    cols = max(0, int(local_cols))
    rep = max(0, int(repeats))
    if rows == 0 or cols == 0 or rep == 0:
        return 0, 0, 0, 0
    elem_bytes = float(spec["bytes"])
    payload_row = int(math.ceil(cols * elem_bytes))
    payload = rows * payload_row * rep
    group = int(spec["group"])
    scale_unit = int(spec["scale"]) if scale_bytes_per_group is None else int(scale_bytes_per_group)
    scale = (
        rows * int(math.ceil(cols / float(group))) * scale_unit * rep
        if group > 0 and scale_unit > 0 else 0
    )
    aux = rows * int(spec["aux"]) * rep
    return payload, scale, aux, payload + scale + aux


def _canonicalize_weight_quant_override(raw: str) -> str:
    low = str(raw or "BF16").strip().lower()
    return _WEIGHT_QUANT_ALIASES.get(low, low.upper())


def _canonicalize_cache_dtype_override(raw: str) -> str:
    low = str(raw or "BF16").strip().lower()
    if low in _CACHE_DTYPE_ALIASES:
        return _CACHE_DTYPE_ALIASES[low]
    if "mxfp8" in low:
        return "MXFP8"
    if "fp8" in low:
        return "FP8"
    if "int8" in low or "c8" in low:
        return "C8"
    return "BF16"


def _normalize_quant_preset(quant: QuantSpec | None, fallback: str = "BF16") -> str:
    if quant is None:
        return fallback
    name = str(quant.name or "").strip().lower()
    exp_fmt = str(quant.expert_weight_format or "").strip().lower()
    w_fmt = str(quant.weight_format or "").strip().lower()
    act_fmt = str(quant.activation_format or "").strip().lower()
    scale_fmt = str(quant.scale_format or "").strip().lower()
    block = tuple(quant.weight_block or ())
    if "w4a16" in name or (exp_fmt == "int4" and act_fmt in _FLOAT16_FORMATS) or (
        w_fmt == "int4" and act_fmt in _FLOAT16_FORMATS
    ):
        return "W4A16"
    if "mxfp4" in name or exp_fmt in {"fp4_e2m1", "mxfp4"} or w_fmt in {"fp4_e2m1", "mxfp4"}:
        return "MXFP4"
    if "mxfp8" in name or (scale_fmt == "fp8_e8m0" and block == (1, MXFP_GROUP_SIZE)):
        return "MXFP8"
    if "w4a8" in name or exp_fmt == "int4" or w_fmt == "int4":
        return "W4A8"
    if "w8a8" in name or exp_fmt == "int8" or w_fmt == "int8":
        return "W8A8"
    if "fp8" in name or exp_fmt.startswith("fp8") or w_fmt.startswith("fp8"):
        return "FP8"
    return _canonicalize_weight_quant_override(w_fmt or fallback)


def _normalize_cache_dtype(model: Any) -> str:
    meta = dict(getattr(model, "metadata", ()) or ())
    tag = str(meta.get("kv_cache_tag") or "").strip().lower()
    if "mxfp8" in tag:
        return "MXFP8"
    quant = getattr(model, "quant", None)
    raw = str(
        (getattr(quant, "kv_dtype", "") if quant is not None else "")
        or getattr(model, "kv_dtype", "")
        or "bf16"
    ).strip().lower()
    return _canonicalize_cache_dtype_override(raw)


def _model_has_mla(model: Any) -> bool:
    return any(
        getattr(lp, "attention", "") == "mla"
        for lp in getattr(model, "layer_plans", ())
    )


def _cache_dtype_width_bytes(dtype: str, *, is_gqa: bool) -> float:
    norm = _canonicalize_cache_dtype_override(dtype)
    if norm == "BF16":
        return float(BF16_BYTES)
    if norm == "MXFP8":
        return MXFP8_CACHE_WIDTH_BYTES
    if norm == "FP8":
        return FP8_CACHE_WIDTH_BYTES
    return GQA_STATIC_C8_CACHE_WIDTH_BYTES if is_gqa else C8_CACHE_WIDTH_BYTES


def _cache_dtype_override_ratio(
    model: Any, target_dtype: str, *, component: str = "",
) -> float:
    base_dtype = _normalize_cache_dtype(model)
    is_gqa = (
        component in _GQA_CACHE_COMPONENTS
        if component else (not _model_has_mla(model))
    )
    base_w = _cache_dtype_width_bytes(base_dtype, is_gqa=is_gqa)
    target_w = _cache_dtype_width_bytes(target_dtype, is_gqa=is_gqa)
    return target_w / max(base_w, MIN_DENOM_EPS)


def stage_layer_counts(case: Case) -> tuple[int, ...]:
    p = case.runtime.parallel
    part = case.runtime.pp_layer_partition
    if part:
        return tuple(int(v) for v in part)
    total = int(case.model.layers)
    return tuple(total // p.pp + (1 if st < total % p.pp else 0) for st in range(p.pp))


def _resolve_ledger_workspace_tokens(
    case: Case, *, phase: str, resolved_seq: int, resolved_conc: int,
) -> int:
    rt = case.runtime
    if str(phase) == "prefill":
        return max(1, int(rt.max_num_batched_tokens or resolved_seq))
    meta = dict(rt.metadata)
    prop = int(meta.get("draft_proposal_tokens", 0) or 0)
    if rt.dspark is not None:
        prop = max(prop, int(rt.dspark.speculative_tokens))
    return max(1, int(resolved_conc) * (1 + max(0, prop)))


def _case_with_weight_quant(
    case: Case,
    resolved_wq: str,
    *,
    apply_preset_to_ignored: bool,
) -> Case:
    base_q = require_model_quant(case.model)
    exclusions = () if apply_preset_to_ignored else tuple(base_q.exclusions)
    if resolved_wq == "BF16":
        new_q = QuantSpec(
            "bf16", weight_format="bf16", activation_format="bf16",
            expert_weight_format="bf16", kv_dtype=base_q.kv_dtype,
            index_dtype=base_q.index_dtype, native_format=base_q.native_format,
        )
    elif resolved_wq == "W8A8":
        new_q = QuantSpec(
            "w8a8", weight_format="int8", activation_format="int8",
            expert_weight_format="int8", kv_dtype=base_q.kv_dtype,
            index_dtype=base_q.index_dtype, native_format=base_q.native_format,
            exclusions=exclusions,
        )
    elif resolved_wq == "W4A8":
        new_q = QuantSpec(
            "w4a8_dynamic", weight_format="int8", activation_format="int8",
            expert_weight_format="int4", kv_dtype=base_q.kv_dtype,
            index_dtype=base_q.index_dtype, native_format=base_q.native_format,
            exclusions=exclusions,
        )
    elif resolved_wq == "W4A16":
        new_q = QuantSpec(
            "w4a16", weight_format="bf16", activation_format="bf16",
            expert_weight_format="int4", kv_dtype=base_q.kv_dtype,
            index_dtype=base_q.index_dtype, native_format=base_q.native_format,
            weight_block=(1, MXFP_GROUP_SIZE), scale_format="bf16",
            scale_bytes=float(_W4A16_SCALE_BYTES), exclusions=exclusions,
        )
    elif resolved_wq == "MXFP8":
        new_q = QuantSpec(
            "w8a8_mxfp8", weight_format="fp8_e4m3", activation_format="fp8_e4m3",
            expert_weight_format="fp8_e4m3", kv_dtype=base_q.kv_dtype,
            index_dtype=base_q.index_dtype, native_format=base_q.native_format,
            weight_block=(1, MXFP_GROUP_SIZE), scale_format="fp8_e8m0",
            scale_bytes=1.0, activation_group_size=MXFP_GROUP_SIZE,
            exclusions=exclusions,
        )
    elif resolved_wq == "MXFP4":
        new_q = QuantSpec(
            "w4a8_mxfp4", weight_format="fp8_e4m3", activation_format="fp8_e4m3",
            expert_weight_format="fp4_e2m1", kv_dtype=base_q.kv_dtype,
            index_dtype=base_q.index_dtype, native_format=base_q.native_format,
            weight_block=(1, MXFP_GROUP_SIZE), scale_format="fp8_e8m0",
            scale_bytes=1.0, activation_group_size=MXFP_GROUP_SIZE,
            exclusions=exclusions,
        )
    elif resolved_wq == "FP8":
        new_q = QuantSpec(
            "fp8", weight_format="fp8_e4m3", activation_format="fp8_e4m3",
            expert_weight_format="fp8_e4m3", kv_dtype=base_q.kv_dtype,
            index_dtype=base_q.index_dtype, native_format=base_q.native_format,
            exclusions=exclusions,
        )
    else:
        return case
    return replace(case, model=replace(case.model, quant=new_q))


def _case_with_cache_dtype(case: Case, target_dtype: str) -> Case:
    """Re-resolve full model cache ABI metadata and ``QuantSpec.kv_dtype`` for ``target_dtype``."""
    canon = _canonicalize_cache_dtype_override(target_dtype)
    kv_dt = _CANONICAL_CACHE_TO_KV_DTYPE.get(canon, "bf16")
    base_q = require_model_quant(case.model)
    new_q = replace(base_q, kv_dtype=kv_dt)
    meta = dict(case.model.metadata)
    if canon == "MXFP8":
        meta["kv_cache_tag"] = _MXFP8_KV_TAG
    else:
        meta.pop("kv_cache_tag", None)
    if canon != _normalize_cache_dtype(case.model):
        # Clear pinned static row-width overrides that belong only to the original dtype.
        meta.pop("cache_representations", None)
    new_model = replace(
        case.model,
        quant=new_q,
        kv_dtype=kv_dt,
        metadata=tuple(sorted(meta.items())),
    )
    return replace(case, model=new_model)


def _hw_cache_dtype_candidates(case: Case, active_dtype: str) -> tuple[str, ...]:
    hw_family = normalize_hw_family(str(getattr(case.machine, "sku", "") or getattr(case.machine, "arch_family", "")))
    if hw_family in _HW_FAMILY_A23:
        base = ["BF16", "C8"]
    elif hw_family == _HW_FAMILY_A5:
        base = ["BF16", "FP8"]
    else:
        base = ["BF16", "C8", "FP8"]
    if active_dtype not in base:
        base.append(active_dtype)
    return tuple(base)


def _resolve_shard_axis_and_rule(
    term: WeightTerm,
    placed_parallel: Any,
    global_rows: int,
    global_cols: int,
    local_rows: int,
    local_cols: int,
) -> tuple[int, str]:
    del placed_parallel
    shard = str(term.shard or "replicated")
    if local_rows < global_rows and local_cols == global_cols:
        return 0, f"{shard} (row/{max(1, global_rows // max(1, local_rows))})"
    if local_cols < global_cols and local_rows == global_rows:
        return 1, f"{shard} (col/{max(1, global_cols // max(1, local_cols))})"
    if local_rows < global_rows and local_cols < global_cols:
        return 0, f"{shard} (2d)"
    return -1, shard


def _weight_group_label(category: str, name: str = "") -> str:
    low_c = str(category or "").lower()
    low_n = str(name or "").lower()
    if low_c in _VISION_CATEGORIES or low_n.startswith(_VISION_NAME_PREFIXES):
        return "Vision / Multimodal"
    if low_c in _ENGRAM_CATEGORIES or _is_ple_category(low_c):
        return "Engram / PLE Memory"
    if low_c in {"embedding", "lm_head", "norm", "norm_f32", "mhc", "ape"}:
        return "Embedding / Norm / Head"
    if low_c in _EXPERT_CATEGORIES:
        return "MoE Routed Experts"
    if low_c in {"shared_expert", "moe.shared"}:
        return "MoE Shared Experts"
    if low_c in {"router", "router_bias"}:
        return "MoE Router"
    if low_c in {"dense_mlp", "dense_ffn", "ffn"}:
        return "Dense FFN"
    if "draft" in low_c or "dspark" in low_c or "mtp" in low_c:
        return "Speculative Draft"
    if "linear" in low_c or "kda" in low_c or "gdn" in low_c or "conv" in low_c or "a_log" in low_c or "dt_bias" in low_c:
        return "Linear Attention (KDA / GDN)"
    return "Attention / MLA"


def _classify_term_format_and_rule(
    term: WeightTerm,
    quant: QuantSpec,
    resolved_wq: str,
    hw_family: str,
    checkpoint: NpuCheckpointSpec | None,
    apply_preset_to_ignored: bool,
) -> tuple[str, str, bool, str]:
    """Return ``(quant_format, quant_rule, is_excluded, note)`` for one term."""
    cat = str(term.category or "").lower()
    eff_quant = term.runtime_quant or term.quant or quant
    linear_rule = (
        QUANT_RULE_PRESET_LINEAR_W8A8
        if hw_family in _HW_FAMILY_A23 else QUANT_RULE_PRESET_LINEAR_FP8
    )
    rt_dtype = str(term.runtime_dtype or "").strip().lower()
    if cat in _FIXED_FP32_CATEGORIES or rt_dtype == "fp32":
        return "FP32", QUANT_RULE_FIXED_FP32, False, "FP32 fixed"
    if cat in _FIXED_BF16_CATEGORIES or _is_vision_term(term) or rt_dtype in _FLOAT16_FORMATS:
        return "BF16", QUANT_RULE_FIXED_BF16, False, "BF16 fixed"

    raw_fmt = str(eff_quant.format_for(term.category)).strip().lower()
    is_excluded = (not apply_preset_to_ignored) and (
        str(term.category) in eff_quant.exclusions or raw_fmt in _FLOAT16_FORMATS
    )
    if checkpoint is not None and not apply_preset_to_ignored:
        ckpt_cls = classify_weight_tensor_quant(term.category, checkpoint)
        if str(getattr(ckpt_cls, "weight_format", "") or "").lower() in _FLOAT16_FORMATS and raw_fmt in _FLOAT16_FORMATS:
            is_excluded = True

    if raw_fmt in _FLOAT16_FORMATS:
        if resolved_wq != "BF16" and cat not in _FIXED_BF16_CATEGORIES:
            return "BF16", QUANT_RULE_CHECKPOINT_IGNORE, True, "checkpoint ignore"
        return "BF16", QUANT_RULE_FOLLOW_PRESET, False, ""

    if raw_fmt == "fp4_e2m1":
        return "MXFP4", QUANT_RULE_FOLLOW_PRESET, False, ""
    if raw_fmt == "int4":
        if str(eff_quant.activation_format or "").strip().lower() in _FLOAT16_FORMATS or resolved_wq == "W4A16":
            return "W4A16", QUANT_RULE_FOLLOW_PRESET, False, ""
        return "W4A8", QUANT_RULE_FOLLOW_PRESET, False, ""
    if raw_fmt == "int8":
        rule = QUANT_RULE_FOLLOW_PRESET if cat in _EXPERT_CATEGORIES else linear_rule
        return "W8A8", rule, is_excluded, ""
    if raw_fmt.startswith("fp8"):
        is_mx = str(eff_quant.scale_format or "").lower() == "fp8_e8m0" and tuple(eff_quant.weight_block or ()) == (1, MXFP_GROUP_SIZE)
        fmt = "MXFP8" if is_mx else "FP8"
        rule = QUANT_RULE_FOLLOW_PRESET if cat in _EXPERT_CATEGORIES else linear_rule
        return fmt, rule, is_excluded, ""
    return _canonicalize_weight_quant_override(raw_fmt), QUANT_RULE_FOLLOW_PRESET, is_excluded, ""


def _describe_quant_display(
    resolved_wq: str,
    weight_rows: Sequence[WeightLedgerRow],
) -> str:
    if resolved_wq == "BF16":
        return "BF16 (Full Precision)"
    routed_fmts = {r.quant_format for r in weight_rows if r.category in _EXPERT_CATEGORIES}
    non_routed_linear = [
        r for r in weight_rows
        if r.category not in _EXPERT_CATEGORIES
        and r.category not in _FIXED_BF16_CATEGORIES
        and r.category not in _FIXED_FP32_CATEGORIES
        and not r.is_vision
        and not r.is_engram_or_ple
        and r.group != _PREPARED_WEIGHT_GROUP
    ]
    all_non_routed_bf16 = bool(non_routed_linear) and all(r.quant_format == "BF16" for r in non_routed_linear)
    if resolved_wq == "MXFP4" and all_non_routed_bf16 and "MXFP4" in routed_fmts:
        return "MXFP4 (Routed-Only, Attn/Shared BF16)"
    if resolved_wq == "W4A16":
        return "W4A16 (Routed INT4 Group-32, Activations BF16)"
    if resolved_wq == "W4A8":
        return "W4A8 (Routed INT4, Linear W8A8)"
    if resolved_wq == "MXFP8":
        return "MXFP8 (FP8 E4M3 + E8M0/32)"
    return resolved_wq


def _append_prepared_weight_entries(
    grouped: dict[tuple[Any, ...], dict[str, Any]],
    order: list[tuple[Any, ...]],
    mem: Memory,
    prepared_weights: Sequence[PreparedWeightSpec],
    pp: int,
) -> None:
    """Append deduplicated persistent ``PreparedWeightSpec`` entries into weight table."""
    seen_backings: set[tuple[int, str]] = set()
    for spec_p in prepared_weights:
        if not isinstance(spec_p, PreparedWeightSpec):
            continue
        if spec_p.alias_of or spec_p.lifetime == "load_peak_only":
            continue
        raw_b = prepared_weight_bytes(spec_p)
        target_stages = (
            [mem.layer_stage[i] if 0 <= i < len(mem.layer_stage) else pp - 1 for i in spec_p.layer_indices]
            if spec_p.layer_indices else [pp - 1 if spec_p.scope == "draft" else 0]
        )
        st_reps = [0] * pp
        for idx, st in enumerate(target_stages):
            backing = f"{spec_p.allocation_id}:s{st}:i{idx}"
            if (st, backing) in seen_backings:
                continue
            seen_backings.add((st, backing))
            st_reps[st] += 1
        if sum(st_reps) <= 0:
            continue
        l_rows = int(spec_p.local_shape[0]) if spec_p.local_shape else 1
        l_cols = int(math.prod(spec_p.local_shape[1:])) if len(spec_p.local_shape) > 1 else 1
        fmt_p = _canonicalize_weight_quant_override(spec_p.dtype)
        key_p = (_PREPARED_WEIGHT_GROUP, spec_p.allocation_id, "prepared_weight", l_rows, l_cols, fmt_p)
        grouped[key_p] = {
            "group": _PREPARED_WEIGHT_GROUP, "name": spec_p.allocation_id,
            "category": "prepared_weight", "g_rows": l_rows, "g_cols": l_cols,
            "rule": f"prepared({spec_p.layout})", "axis": -1, "l_rows": l_rows, "l_cols": l_cols,
            "q_fmt": fmt_p, "q_rule": QUANT_RULE_FIXED_BF16,
            "is_excl": False, "q_note": spec_p.transform,
            "is_engram": False, "is_vis": False,
            "unit_p": raw_b, "unit_s": 0, "unit_a": 0, "unit_bf16": raw_b,
            "stage_reps": st_reps,
        }
        order.append(key_p)


def _finalize_weight_ledger_rows(
    grouped: Mapping[tuple[Any, ...], Mapping[str, Any]],
    order: Sequence[tuple[Any, ...]],
    mem: Memory,
    pp: int,
) -> tuple[tuple[WeightLedgerRow, ...], list[int], list[int], list[int], int]:
    """Materialize ``WeightLedgerRow`` tuples and per-stage payload/scale/engram sums."""
    peak_stage = max(range(pp), key=lambda st: float(mem.weights[st]))
    stage_payload = [0] * pp
    stage_scale_aux = [0] * pp
    stage_engram_raw = [0] * pp
    rows_out: list[WeightLedgerRow] = []
    include_vit = mem.vision_accounting == "full_checkpoint"

    for key in order:
        rec = grouped[key]
        reps = tuple(int(v) for v in rec["stage_reps"])
        pk_rep = reps[peak_stage]
        p_b = int(rec["unit_p"]) * pk_rep
        s_b = int(rec["unit_s"]) * pk_rep
        a_b = int(rec["unit_a"]) * pk_rep
        rows_out.append(WeightLedgerRow(
            group=rec["group"], name=rec["name"], category=rec["category"],
            global_rows=rec["g_rows"], global_cols=rec["g_cols"],
            shard_rule=rec["rule"], shard_axis=rec["axis"],
            local_rows=rec["l_rows"], local_cols=rec["l_cols"],
            stage_repeats=reps, max_stage_repeats=max(reps, default=0),
            quant_format=rec["q_fmt"],
            payload_bytes=p_b, scale_bytes=s_b, aux_bytes=a_b,
            per_die_bytes=p_b + s_b + a_b,
            is_engram_or_ple=rec["is_engram"],
            is_quant_excluded=rec["is_excl"],
            quant_note=rec["q_note"],
            native_bf16_bytes=int(rec["unit_bf16"]) * pk_rep,
            quant_rule=rec["q_rule"],
            is_vision=rec["is_vis"],
            peak_stage_repeats=pk_rep,
        ))
        for st in range(pp):
            st_r = reps[st]
            if st_r <= 0 or (rec["is_vis"] and not include_vit):
                continue
            if rec["is_engram"]:
                stage_engram_raw[st] += (int(rec["unit_p"]) + int(rec["unit_s"]) + int(rec["unit_a"])) * st_r
                continue
            stage_payload[st] += int(rec["unit_p"]) * st_r
            stage_scale_aux[st] += (int(rec["unit_s"]) + int(rec["unit_a"])) * st_r

    return tuple(rows_out), stage_payload, stage_scale_aux, stage_engram_raw, peak_stage


def _project_weight_ledger_rows(
    case: Case,
    mem: Memory,
    *,
    resolved_wq: str,
    hw_family: str,
    apply_preset_to_ignored: bool,
    prepared_weights: Sequence[PreparedWeightSpec],
) -> tuple[tuple[WeightLedgerRow, ...], list[int], list[int], list[int], int]:
    """Project canonical ``Memory`` resident weight terms into ``WeightLedgerRow`` records."""
    model = case.model
    p = case.runtime.parallel
    pp = max(1, int(p.pp))
    quant = require_model_quant(model)
    checkpoint = resolve_default_checkpoint(model.name, hw_family=hw_family)
    all_stage_terms = stage_weight_terms(
        model, p,
        include_draft=bool(mem.draft_kind),
        draft_kind=mem.draft_kind or None,
        include_engram_tables=True,
        include_vision=True,
        layer_partition=mem.layers,
        slot_plans=mem.slot_plans,
    )
    shared_dp = bool(getattr(mem, "enable_shared_expert_dp", False))
    pcp_o_proj = bool(getattr(mem, "enable_pcp_o_proj_weight_sharding", False))
    dcp_q_rep = bool(getattr(mem, "dcp_q_replicate", False))
    node_ranks = int(case.machine.ranks_per_node)

    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for st in range(pp):
        for term in all_stage_terms[st]:
            if not term.runtime_owned:
                continue
            expanded = _expand_resident_terms(
                term, p, shared_dp,
                pcp_o_proj_weight_sharding=pcp_o_proj,
                dcp_q_replicate=dcp_q_rep,
                model=model,
            )
            for t_sub, placed in expanded:
                if not t_sub.runtime_owned:
                    continue
                g_rows, g_cols = int(t_sub.rows), int(t_sub.cols)
                l_rows, l_cols, l_count = _sharded_shape(t_sub, placed, node_ranks)
                if l_count <= 0:
                    continue
                q_fmt, q_rule, is_excl, q_note = _classify_term_format_and_rule(
                    t_sub, quant, resolved_wq, hw_family, checkpoint, apply_preset_to_ignored,
                )
                tot_b = int(round(term_bytes(t_sub, quant, placed, node_ranks)))
                pay_b = int(round(term_payload_bytes(t_sub, quant, placed, node_ranks)))
                unit_total = tot_b // l_count
                unit_payload = pay_b // l_count
                unit_meta = max(0, unit_total - unit_payload)
                spec = WEIGHT_QUANT_FORMATS.get(q_fmt, WEIGHT_QUANT_FORMATS["BF16"])
                unit_scale, unit_aux = (unit_meta, 0) if int(spec["group"]) > 0 else (0, unit_meta)
                axis, rule_str = _resolve_shard_axis_and_rule(
                    t_sub, placed, g_rows, g_cols, l_rows, l_cols,
                )
                is_engram = str(t_sub.category or "").lower() in _ENGRAM_CATEGORIES
                is_vis = _is_vision_term(t_sub)
                key = (
                    _weight_group_label(t_sub.category, t_sub.name),
                    str(t_sub.name), str(t_sub.category),
                    g_rows, g_cols, rule_str, axis, l_rows, l_cols,
                    q_fmt, q_rule, is_engram, is_vis, unit_payload, unit_scale, unit_aux,
                )
                rec = grouped.get(key)
                if rec is None:
                    rec = {
                        "group": key[0], "name": key[1], "category": key[2],
                        "g_rows": g_rows, "g_cols": g_cols, "rule": rule_str,
                        "axis": axis, "l_rows": l_rows, "l_cols": l_cols,
                        "q_fmt": q_fmt, "q_rule": q_rule,
                        "is_excl": is_excl, "q_note": q_note,
                        "is_engram": is_engram, "is_vis": is_vis,
                        "unit_p": unit_payload, "unit_s": unit_scale, "unit_a": unit_aux,
                        "unit_bf16": l_rows * l_cols * BF16_BYTES,
                        "stage_reps": [0] * pp,
                    }
                    grouped[key] = rec
                    order.append(key)
                rec["stage_reps"][st] += int(l_count)

    _append_prepared_weight_entries(grouped, order, mem, prepared_weights, pp)
    return _finalize_weight_ledger_rows(grouped, order, mem, pp)


def _project_cache_ledger_rows(
    case: Case,
    active_proj: CacheLedgerProjection,
    alt_projections: Mapping[str, CacheLedgerProjection],
    *,
    active_dtype: str,
    local_tokens: int,
    concurrency: int,
) -> tuple[CacheLedgerRow, ...]:
    """Project exact per-component ``CacheLedgerRow`` rows directly from ``CacheLedgerProjection``."""
    p = case.runtime.parallel
    pp = max(1, int(p.pp))
    conc = max(1, int(concurrency))
    all_projs = {active_dtype: active_proj, **alt_projections}
    legal_dtypes = tuple(all_projs.keys())

    # Union of all components present across any legal dtype & stage.
    comp_set: set[str] = set()
    for proj in all_projs.values():
        for st_map in proj.per_stage_component_batch_bytes:
            comp_set.update(st_map.keys())

    ordered_comps = [c for c in _COMPONENT_DISPLAY_ORDER if c in comp_set]
    ordered_comps.extend(sorted(comp_set.difference(ordered_comps)))

    # Select bottleneck cache stage on active projection.
    stage_totals = [sum(st_map.values()) for st_map in active_proj.per_stage_component_batch_bytes]
    peak_cache_stage = max(range(pp), key=lambda st: stage_totals[st]) if stage_totals else 0

    rows: list[CacheLedgerRow] = []
    for comp in ordered_comps:
        is_fixed_die = comp in _FIXED_PER_DIE_COMPONENTS
        scales_tok = comp not in _NON_TOKEN_SCALED_COMPONENTS
        layer_counts = active_proj.active_layers_by_component.get(comp, (0,) * pp)
        if not any(layer_counts):
            for proj in all_projs.values():
                cand = proj.active_layers_by_component.get(comp)
                if cand and any(cand):
                    layer_counts = cand
                    break
        max_layers = max(layer_counts, default=0)
        kind_label = active_proj.cache_kind_by_component.get(comp, comp)
        if kind_label == comp:
            for proj in all_projs.values():
                if comp in proj.cache_kind_by_component:
                    kind_label = proj.cache_kind_by_component[comp]
                    break

        active_batch_by_st = tuple(
            float(active_proj.per_stage_component_batch_bytes[st].get(comp, 0))
            for st in range(pp)
        )
        active_per_req_by_st = tuple(
            0.0 if is_fixed_die else float(active_proj.per_stage_component_per_req_bytes[st].get(comp, 0.0))
            for st in range(pp)
        )
        batch_b = active_batch_by_st[peak_cache_stage]
        per_req_b = 0.0 if is_fixed_die else active_per_req_by_st[peak_cache_stage]
        denom = float(max_layers * max(1, local_tokens)) if (scales_tok and max_layers > 0) else 0.0
        rate_b = (per_req_b / denom) if denom > 0.0 else 0.0

        rate_by_dt: list[tuple[str, float]] = []
        req_by_dt: list[tuple[str, float]] = []
        for dt in legal_dtypes:
            dt_proj = all_projs[dt]
            dt_batch = float(dt_proj.per_stage_component_batch_bytes[peak_cache_stage].get(comp, 0))
            if is_fixed_die:
                # Fixed per-die bytes stored directly (never divided or scaled by concurrency).
                req_by_dt.append((dt, dt_batch))
                rate_by_dt.append((dt, 0.0))
            else:
                dt_req = float(dt_proj.per_stage_component_per_req_bytes[peak_cache_stage].get(comp, 0.0))
                if dt_req == 0.0 and dt_batch > 0.0:
                    dt_req = dt_batch / float(conc)
                req_by_dt.append((dt, dt_req))
                dt_rate = (dt_req / denom) if denom > 0.0 else 0.0
                rate_by_dt.append((dt, dt_rate))

        per_stage_out = active_batch_by_st if is_fixed_die else active_per_req_by_st
        rows.append(CacheLedgerRow(
            component=comp,
            cache_kind=kind_label,
            active_layers_per_stage=tuple(int(v) for v in layer_counts),
            max_stage_layers=int(max_layers),
            bytes_per_token_per_layer=float(rate_b),
            per_request_bytes=float(per_req_b),
            batch_total_bytes=float(batch_b),
            scales_with_tokens=bool(scales_tok),
            per_stage_bytes=per_stage_out,
            bytes_per_token_by_dtype=tuple(rate_by_dt),
            per_request_bytes_by_dtype=tuple(req_by_dt),
        ))
    return tuple(rows)


def _build_stage_budgets_from_receipt(
    case: Case,
    mem: Memory,
    proj: CacheLedgerProjection,
    *,
    stage_payload: Sequence[int],
    stage_scale_aux: Sequence[int],
    stage_engram_raw: Sequence[int],
) -> tuple[StageBudgetRow, ...]:
    """Project exact per-PP-stage ``StageBudgetRow`` with 0 B multi-key bijection to ``DieResidentReceipt``."""
    p = case.runtime.parallel
    pp = max(1, int(p.pp))
    layer_counts = stage_layer_counts(case)
    offload_engram = bool(case.runtime.engram_cpu_offload)
    budgets: list[StageBudgetRow] = []

    for st in range(pp):
        rk = proj.bottleneck_rank_by_stage[st]
        die = proj.receipt.die_receipt(0, st, rk)
        receipt_weight_hbm = int(
            die.checkpoint_weight_bytes + die.prepared_weight_bytes + die.vision_weight_bytes
        )
        engram_raw = int(stage_engram_raw[st])
        engram_hbm = 0 if offload_engram else engram_raw
        engram_host = int(round(mem.engram_host_dram_bytes[st])) if offload_engram else 0
        non_engram_hbm = max(0, receipt_weight_hbm - engram_hbm)
        w_scale = min(int(stage_scale_aux[st]), non_engram_hbm)
        w_payload = non_engram_hbm - w_scale
        if int(stage_payload[st]) + int(stage_scale_aux[st]) == non_engram_hbm:
            w_payload = int(stage_payload[st])
            w_scale = int(stage_scale_aux[st])
        cache_b = int(
            die.kv_cache_allocated_bytes
            + die.index_cache_allocated_bytes
            + die.recurrent_state_bytes
            + die.draft_cache_bytes
        )
        ws_b = int(die.workspace_bytes)
        res_b = int(die.reserve_bytes)
        total_hbm = int(die.simultaneous_live_hbm_bytes)
        usable_b = int(die.usable_hbm_bytes)
        headroom_b = usable_b - total_hbm
        per_req_cache = float(sum(
            v for k, v in proj.per_stage_component_per_req_bytes[st].items()
            if k not in _FIXED_PER_DIE_COMPONENTS
        ))
        budgets.append(StageBudgetRow(
            stage=st,
            owned_layers=int(layer_counts[st]),
            weight_payload_bytes=float(w_payload),
            weight_scale_aux_bytes=float(w_scale),
            engram_ple_hbm_bytes=float(engram_hbm),
            engram_ple_host_bytes=float(engram_host),
            cache_bytes=float(cache_b),
            workspace_peak_bytes=float(ws_b),
            reserve_bytes=float(res_b),
            total_hbm_bytes=float(total_hbm),
            usable_hbm_budget_bytes=float(usable_b),
            headroom_bytes=float(headroom_b),
            fits_hbm=bool(total_hbm <= usable_b),
            per_request_cache_bytes=per_req_cache,
        ))
    return tuple(budgets)


def _compute_max_hbm_concurrency(
    stage_budgets: Sequence[StageBudgetRow],
    proj: CacheLedgerProjection,
) -> int:
    """Closed-form max per-DP concurrency honoring fixed per-die overhead."""
    limits: list[int] = []
    for st in stage_budgets:
        fixed_c8 = int(proj.per_stage_component_batch_bytes[st.stage].get("c8_static_channel_scale", 0))
        fixed_hbm = (
            st.weight_payload_bytes
            + st.weight_scale_aux_bytes
            + st.engram_ple_hbm_bytes
            + st.workspace_peak_bytes
            + st.reserve_bytes
            + float(fixed_c8)
        )
        avail = st.usable_hbm_budget_bytes - fixed_hbm
        if avail <= 0.0:
            return 0
        per_req = float(st.per_request_cache_bytes)
        if per_req > MIN_DENOM_EPS:
            limits.append(int(math.floor(avail / per_req)))
    return min(limits) if limits else 0


def _resolve_case_overrides(
    case: Case,
    *,
    hw_family: str,
    weight_quant: str | None,
    cache_dtype: str | None,
    engram_host_offload: bool | None,
    include_vision_tower: bool | None,
    apply_preset_to_ignored: bool,
) -> tuple[Case, str, str]:
    """Apply explicit ledger overrides while enforcing hardware ISA legality."""
    resolved = case
    if weight_quant is not None:
        canon_wq = _canonicalize_weight_quant_override(weight_quant)
        validate_quant_for_hardware(canon_wq, hw_family, strict=True)
        resolved = _case_with_weight_quant(
            resolved, canon_wq, apply_preset_to_ignored=apply_preset_to_ignored,
        )
    elif apply_preset_to_ignored:
        canon_wq = _normalize_quant_preset(resolved.model.quant)
        resolved = _case_with_weight_quant(
            resolved, canon_wq, apply_preset_to_ignored=True,
        )
    if cache_dtype is not None:
        canon_cd = _canonicalize_cache_dtype_override(cache_dtype)
        validate_quant_for_hardware(
            "bf16", hw_family,
            kv_cache_dtype=_CANONICAL_CACHE_TO_KV_DTYPE.get(canon_cd, "bf16"),
            strict=True,
        )
        resolved = _case_with_cache_dtype(resolved, canon_cd)
    if engram_host_offload is not None:
        resolved = replace(
            resolved,
            runtime=replace(resolved.runtime, engram_cpu_offload=bool(engram_host_offload)),
        )
    if include_vision_tower is not None:
        vis_scope = "full_checkpoint" if bool(include_vision_tower) else "text_backbone"
        rt_meta = {**dict(resolved.runtime.metadata), "vision_accounting": vis_scope}
        resolved = replace(
            resolved,
            runtime=replace(resolved.runtime, metadata=tuple(sorted(rt_meta.items()))),
        )
    resolved_wq = _normalize_quant_preset(resolved.model.quant)
    resolved_cd = _normalize_cache_dtype(resolved.model)
    validate_quant_for_hardware(
        resolved_wq, hw_family,
        kv_cache_dtype=_CANONICAL_CACHE_TO_KV_DTYPE.get(resolved_cd, "bf16"),
        strict=True,
    )
    return resolved, resolved_wq, resolved_cd


def _probe_legal_cache_projections(
    resolved_case: Case,
    *,
    resolved_wq: str,
    resolved_cd: str,
    hw_family: str,
    phase: str,
    resolved_seq: int,
    resolved_conc: int,
    ws_tok: int,
    prepared_weights: Sequence[PreparedWeightSpec],
) -> dict[str, CacheLedgerProjection]:
    """Project all hardware-legal alternate cache dtypes via canonical ``Memory``."""
    alt_projections: dict[str, CacheLedgerProjection] = {}
    for cand_dt in _hw_cache_dtype_candidates(resolved_case, resolved_cd):
        if cand_dt == resolved_cd:
            continue
        try:
            validate_quant_for_hardware(
                resolved_wq, hw_family,
                kv_cache_dtype=_CANONICAL_CACHE_TO_KV_DTYPE.get(cand_dt, "bf16"),
                strict=True,
            )
            alt_case = _case_with_cache_dtype(resolved_case, cand_dt)
            alt_mem = Memory(alt_case, workspace_tokens=ws_tok)
            if prepared_weights:
                alt_mem.register_prepared_weights(prepared_weights)
            alt_projections[cand_dt] = alt_mem.cache_ledger_projection(
                scenario=str(alt_case.model.name),
                phase=str(phase),
                sequence_tokens=resolved_seq,
                batch_per_dp=resolved_conc,
                vision_accounting=alt_mem.vision_accounting,
                prepared_weights=prepared_weights,
            )
        except ModelingError:
            continue
    return alt_projections


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
    memory_receipt: Any = None,
) -> MemoryCalculatorLedger:
    """Project canonical ``Memory`` / ``DieResidentReceipt`` into ``MemoryCalculatorLedger``."""
    del memory_receipt
    hw_sku = str(getattr(case.machine, "sku", "") or getattr(case.machine, "name", ""))
    hw_family = normalize_hw_family(hw_sku or str(getattr(case.machine, "arch_family", "")))
    resolved_case, resolved_wq, resolved_cd = _resolve_case_overrides(
        case,
        hw_family=hw_family,
        weight_quant=weight_quant,
        cache_dtype=cache_dtype,
        engram_host_offload=engram_host_offload,
        include_vision_tower=include_vision_tower,
        apply_preset_to_ignored=apply_preset_to_ignored,
    )
    rt = resolved_case.runtime
    p = rt.parallel
    resolved_seq = int(
        seq_len if seq_len is not None else (
            max((int(r.prompt_tokens) + int(r.output_tokens) for r in resolved_case.workload.requests), default=0)
            or int(rt.max_model_len or DEFAULT_CACHE_BLOCK_SIZE)
        )
    )
    resolved_conc = max(
        _DEFAULT_CONCURRENCY,
        int(concurrency if concurrency is not None else (len(resolved_case.workload.requests) or _DEFAULT_CONCURRENCY)),
    )
    ws_tok = _resolve_ledger_workspace_tokens(
        resolved_case, phase=phase, resolved_seq=resolved_seq, resolved_conc=resolved_conc,
    )
    mem = Memory(resolved_case, workspace_tokens=ws_tok)
    if prepared_weights:
        mem.register_prepared_weights(prepared_weights)
    proj = mem.cache_ledger_projection(
        scenario=str(resolved_case.model.name),
        phase=str(phase),
        sequence_tokens=resolved_seq,
        batch_per_dp=resolved_conc,
        vision_accounting=mem.vision_accounting,
        prepared_weights=prepared_weights,
    )
    alt_projections = _probe_legal_cache_projections(
        resolved_case,
        resolved_wq=resolved_wq, resolved_cd=resolved_cd, hw_family=hw_family,
        phase=str(phase), resolved_seq=resolved_seq, resolved_conc=resolved_conc,
        ws_tok=ws_tok, prepared_weights=prepared_weights,
    )
    weight_rows, st_payload, st_scale, st_engram, peak_w_stage = _project_weight_ledger_rows(
        resolved_case, mem,
        resolved_wq=resolved_wq, hw_family=hw_family,
        apply_preset_to_ignored=apply_preset_to_ignored,
        prepared_weights=prepared_weights,
    )
    block_size = int(proj.normalized_block_size)
    cp_deg = max(1, int(getattr(p, "cp", 1) or 1))
    local_tok = int(math.ceil(resolved_seq / float(max(1, block_size * cp_deg)))) * block_size if resolved_seq > 0 else 0
    cache_rows = _project_cache_ledger_rows(
        resolved_case, proj, alt_projections,
        active_dtype=resolved_cd, local_tokens=local_tok, concurrency=resolved_conc,
    )
    stage_budgets = _build_stage_budgets_from_receipt(
        resolved_case, mem, proj,
        stage_payload=st_payload, stage_scale_aux=st_scale, stage_engram_raw=st_engram,
    )
    die0 = proj.receipt.die_receipt(0, 0, proj.bottleneck_rank_by_stage[0])
    die_hbm = float(resolved_case.machine.dies[0].hbm_bytes)
    include_vit = mem.vision_accounting == "full_checkpoint"
    offload_engram = bool(rt.engram_cpu_offload)
    hbm_w_rows = [
        r for r in weight_rows
        if not (r.is_engram_or_ple and offload_engram)
        and not (r.is_vision and not include_vit)
    ]
    w_footprint = float(sum(r.per_die_bytes for r in hbm_w_rows))
    native_bf16_w = float(sum(r.native_bf16_bytes for r in hbm_w_rows))
    vis_w = float(sum(r.per_die_bytes for r in weight_rows if r.is_vision)) if include_vit else 0.0
    pcp_deg = max(1, int(getattr(p, "pcp", 1) or 1))
    prep_persist = tuple(getattr(mem, "_prepared_persistent_by_stage", (0.0,) * p.pp))
    return MemoryCalculatorLedger(
        model_name=str(resolved_case.model.name),
        hardware_sku=hw_sku,
        phase=str(phase),
        weight_quant=resolved_wq,
        cache_dtype=resolved_cd,
        engram_host_offload=offload_engram,
        dp=int(p.dp), tp=int(p.tp), ep=int(p.ep), pp=int(p.pp),
        dcp=max(1, int(getattr(p, "dcp", p.cp) or 1)), pcp=pcp_deg,
        local_tp=int(getattr(p, "o_proj_tp", 0) or getattr(p, "local_tp", 0) or p.tp),
        shared_tp=1 if bool(getattr(mem, "enable_shared_expert_dp", False)) else int(p.tp),
        total_dies=int(p.dp * p.tp * p.pp * pcp_deg),
        seq_len=resolved_seq, concurrency=resolved_conc,
        block_size=block_size, spec_tokens=max(0, int(mem.query_tokens) - 1),
        local_cache_tokens=local_tok,
        die_hbm_capacity_bytes=die_hbm,
        gpu_memory_utilization=float(rt.gpu_memory_utilization),
        usable_hbm_bytes=float(die0.usable_hbm_bytes),
        weight_rows=weight_rows,
        cache_rows=cache_rows,
        stage_budgets=stage_budgets,
        max_stage_hbm_bytes=max((sb.total_hbm_bytes for sb in stage_budgets), default=0.0),
        max_stage_host_offload_bytes=max((sb.engram_ple_host_bytes for sb in stage_budgets), default=0.0),
        fits_hbm=all(sb.fits_hbm for sb in stage_budgets),
        sim_memory_weight_bytes_by_stage=tuple(float(mem.weights[st]) + float(prep_persist[st]) for st in range(p.pp)),
        sim_memory_cache_bytes_by_stage=tuple(float(sb.cache_bytes) for sb in stage_budgets),
        weight_quant_display=_describe_quant_display(resolved_wq, weight_rows),
        quant_exclusions=tuple(sorted(set(require_model_quant(resolved_case.model).exclusions))),
        weight_footprint_per_die_bytes=w_footprint,
        per_request_cache_bytes_max_stage=max((sb.per_request_cache_bytes for sb in stage_budgets), default=0.0),
        max_hbm_concurrency_per_dp=_compute_max_hbm_concurrency(stage_budgets, proj),
        include_vision_tower=include_vit,
        apply_preset_to_ignored=bool(apply_preset_to_ignored),
        native_bf16_weight_footprint_per_die_bytes=native_bf16_w,
        vision_weight_bytes_per_die=vis_w,
        weight_compression_ratio=(native_bf16_w / w_footprint) if w_footprint > MIN_DENOM_EPS else 1.0,
        peak_weight_stage=peak_w_stage,
    )


__all__ = [
    "C8_CACHE_WIDTH_BYTES",
    "FP8_CACHE_WIDTH_BYTES",
    "GQA_STATIC_C8_CACHE_WIDTH_BYTES",
    "KV_DTYPE_SENSITIVE_COMPONENTS",
    "MXFP8_CACHE_WIDTH_BYTES",
    "_PREPARED_WEIGHT_GROUP",
    "_cache_dtype_override_ratio",
    "_normalize_cache_dtype",
    "_resolve_ledger_workspace_tokens",
    "compute_memory_quant_ledger",
    "evaluate_weight_storage_bytes",
    "stage_layer_counts",
]
