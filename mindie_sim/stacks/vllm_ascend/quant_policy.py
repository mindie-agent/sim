"""1:1 vLLM-Ascend quantization instantiation, shard packing, and CacheStateABI builder.

[Role] Mirror upstream ``vllm-ascend`` quantization instantiation strictly via
  structural ``ModelSpec`` / ``LayerPlan`` fields and ``CheckpointQuantMap``:
  - strict Homogeneous Shard-Packing Law across all experts,
  - runtime-only ``PreparedWeightSpec`` materializations,
  - exact per-native-layer typed ``CacheStateABI`` for KV, Indexer, and
    recurrent/PLE/Engram state.
[Invariants] Zero per-model-name executable branches (AG014). Zero silent BF16
  or FLOAT fallbacks, zero ``BF16 -> fp16`` coercions, zero expert-0-only
  sampling, zero heterogeneous gate/up ``FLOAT`` downgrades. Unknown non-empty
  checkpoint tags raise ``ModelingError("unknown_checkpoint_quant_tag:...")``.
  A2/A3 never admit MXFP/FP8 weights or FP8 KV cache, and reject INT8 KV on
  window-compressed MLA. Zero imports from ``mindie_sim.export`` or
  ``mindie_sim.compose`` (AG010).
[Timeouts & Locks] Pure in-memory resolution over immutable descriptors; no I/O
  or locks.
[Side-Effect Policy] Returns frozen ``RuntimeQuantPlan`` records; zero input
  mutation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import fnmatch
import math
import re
from typing import Any

from mindie_sim.catalog.cache import layer_stage_map
from mindie_sim.catalog.cache_format import (
    _lookup_region_decl,
    _resolve_index_cache_format,
    _row_planes_from_format,
    cache_format,
)
from mindie_sim.catalog.cache_pools import (
    REGIME_ALIASED_WINDOW_COMPRESSED,
    resolve_pool_allocator_contract,
)
from mindie_sim.catalog.context_parallel import dcp_indexer_sharded, layer_dcp_factors
from mindie_sim.catalog.draft import optional_draft_spec
from mindie_sim.catalog.draft_geometry import draft_layer_plans
from mindie_sim.catalog.draft_spec import effective_draft_context_cap
from mindie_sim.catalog.mqsmla_layout import (
    EXPONENT_SCALE_TYPES as MQSMLA_EXPONENT_SCALES,
    HEAD_WIDTH as MQSMLA_HEAD_WIDTH,
    MODE_PADDING as MQSMLA_MODE_PADDING,
    NATIVE_GROUP as MQSMLA_NATIVE_GROUP,
    ROPE_WIDTH as MQSMLA_ROPE_WIDTH,
)
from mindie_sim.catalog.speculative_state import speculative_state_plan
from mindie_sim.schema.checkpoint import (
    CacheQuantRule,
    CheckpointOrigin,
    CheckpointQuantMap,
    ModelScope,
    TensorQuantRule,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import TensorPrecisionABI
from mindie_sim.schema.parallel import ParallelSpec, local_kv_heads
from mindie_sim.schema.stack import (
    CachePlacementRule,
    CachePlaneABI,
    CacheStateABI,
    PreparedWeightSpec,
    ResolvedStackOptions,
    RuntimeQuantPlan,
    resolve_checkpoint_cache_abis,
)

from .checkpoint_registry import (
    QUANT_MODEL_PREFIX_MAPPINGS,
    QUANT_MODEL_SUBSTR_MAPPINGS,
    expand_checkpoint_tensor_paths,
)
from .profiles import (
    A2_A3_INT_ONLY_FAMILIES,
    VLLM_ASCEND_STACK_ID,
    normalize_hw_family,
    resolve_model_type,
)
from .speculative import ZERO_NEURAL_DRAFT_METHODS, resolve_speculative_companion

#: Upstream ``packed_modules_mapping`` + ``UPDATED_PACKED_MODULES_MAPPING``.
UPDATED_PACKED_MODULES_MAPPING: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "fused_qkvz_proj": ("in_proj_qkv", "in_proj_z"),
    "fused_ba_proj": ("in_proj_b", "in_proj_a"),
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
    "in_proj_qkv": ("q_proj", "k_proj", "v_proj"),
    "fused_bfg_proj": ("b_proj", "f_a_proj", "g_proj"),
    "fused_bfg_a_proj": ("b_proj", "f_a_proj", "g_a_proj"),
    "fused_qkvbfg_a_proj": ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"),
    "fused_qkv_a_proj": ("q_a_proj", "kv_a_proj_with_mqa"),
    "wk_weights_proj": ("wk", "weights_proj"),
    "index_qk_proj": ("index_q_proj", "index_k_proj"),
}

#: Constructor ``quant_config=None`` unquantized modules in ``vllm-ascend``.
CONSTRUCTOR_UNQUANTIZED_ROLES: frozenset[str] = frozenset({
    "rmsnorm",
    "layernorm",
    "input_layernorm",
    "post_attention_layernorm",
    "embed_tokens",
    "vocab_parallel_embedding",
    "lm_head",
    "mlp.gate",
    "gate",
    "self_attention_res_proj",
    "mlp_res_proj",
    "output_attn_res_proj",
    "conv1d",
    "q_conv1d",
    "k_conv1d",
    "v_conv1d",
    "hyper_connection.pre_proj",
    "hyper_connection.post_proj",
    "ple.conv1d",
    "engram.embed",
    "mla.kv_b_proj",
})

MXFP_GROUP_SIZE = 32
W4A8_DEFAULT_GROUP_SIZE = 128
FP8_DEFAULT_BLOCK_SIZE = 128
DEFAULT_KV_BLOCK_SIZE = 128
DEFAULT_CONV_KERNEL_SIZE = 4

NZ_K_ALIGN = 16
NZ_N_ALIGN = 64
INT4_PACKED_PER_BYTE = 2
SCALE_BYTES_E8M0 = 1
SCALE_BYTES_FP16 = 2
SCALE_BYTES_FP32 = 4

MXFP_KV_SCALE_GROUP_SIZE = 64
MXFP_KV_SCALE_VALUES_PER_GROUP = 2
MXFP_KV_NZ_DIM_FRAG = 32
MXFP_K_SCALE_NZ_TOKEN_FRAG = 16
MXFP_V_SCALE_NZ_DIM_FRAG = 16

W8A16_WEIGHT_BITS = 8
_EXPERT_ID_RE = re.compile(r"\bexperts\.(\d+)\.")

_MXFP8_TAGS = frozenset({"W8A8_MXFP8", "MXFP8", "MX_FP8", "W8A8_MXFP", "MXFP8_DYNAMIC"})
_MXFP4_W4A8_TAGS = frozenset({
    "W4A8_MXFP", "W4A8_MXFP4", "W4A8_MXFP4_DYNAMIC",
    "MXFP4", "MX_FP4", "FP4", "NVFP4", "QUARK_MXFP4",
})
_MXFP4_W4A4_TAGS = frozenset({"W4A4_MXFP4", "W4A4_MXFP"})
_FP8_TAGS = frozenset({"FP8", "FP8_DYNAMIC", "W8A8_FP8", "FP8_E4M3", "FP8_E5M2", "QUARK_FP8"})
_W8A8_DYNAMIC_TAGS = frozenset({
    "W8A8_DYNAMIC", "W8A8D", "W8A8_INT8", "INT8_DYNAMIC", "INT8_SYM", "QUARK_INT8",
})
_W8A8_STATIC_TAGS = frozenset({"W8A8", "W8A8_STATIC", "INT8"})
_W8A8_MIX_TAGS = frozenset({"W8A8_MIX", "W8A8_PDMIX"})
_W4A8_TAGS = frozenset({
    "W4A8_DYNAMIC", "W4A8", "W4A8_COMPRESSED", "INT4", "W4A4_DYNAMIC", "W4A4",
})
_W4A16_TAGS = frozenset({
    "W4A16", "W4A16_DYNAMIC", "W4A16_AWQ", "W4A16_GPTQ", "W8A16_GPTQ", "W4A16_COMPRESSED",
})

_MXFP8_KV_TAGS = frozenset({
    "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL",
    "MXFP8_KV",
    "MXFP8_PER_CHANNEL",
})
_MLA_FP8_KV_TAGS = frozenset({"FP8", "FP8_E4M3", "FP8_DYNAMIC", "MXFP8", "MXFP8_DYNAMIC", "W8A8_MXFP8"})
_INT8_KV_TAGS = frozenset({"INT8", "C8", "KV_C8", "INT8_DYNAMIC", "INT8_SYM", "FAQUANT", "FAKQUANT"})
_NATIVE_KV_TAGS = frozenset({"", "AUTO", "NONE", "FLOAT", "BF16", "BFLOAT16", "FP16", "FLOAT16"})
_FP8_INDEX_TAGS = frozenset({"fp8", "fp8_e4m3", "fp8_dynamic", "mxfp8", "mxfp8_dynamic"})
_INT8_INDEX_TAGS = frozenset({"int8", "c8", "int8_dynamic", "int8_sym"})
_FAQUANT_BUFFER_TOKENS = ("fa_q", "fa_k", "fa_v", "faquant", "fakquant")

_FLOAT_DTYPE_CANON: dict[str, str] = {
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float16": "fp16",
    "fp16": "fp16",
    "float32": "fp32",
    "fp32": "fp32",
}

_PROJECTION_PRESET_TAGS: dict[str, tuple[str, str]] = {
    "mxfp8": ("W8A8_MXFP8", "W8A8_MXFP8"),
    "w8a8_mxfp8": ("W8A8_MXFP8", "W8A8_MXFP8"),
    "w8a8_mxfp": ("W8A8_MXFP8", "W8A8_MXFP8"),
    "mxfp8_mxfp4": ("W8A8_MXFP8", "W4A8_MXFP"),
    "mxfp4_routed_only": ("FLOAT", "W4A8_MXFP"),
    "mxfp4_all": ("W4A8_MXFP", "W4A8_MXFP"),
    "mxfp4": ("W4A8_MXFP", "W4A8_MXFP"),
    "w4a8_mxfp": ("W8A8_MXFP8", "W4A8_MXFP"),
    "w4a4_mxfp4": ("W4A4_MXFP4", "W4A4_MXFP4"),
    "fp8": ("FP8_DYNAMIC", "FP8_DYNAMIC"),
    "fp8_dynamic": ("FP8_DYNAMIC", "FP8_DYNAMIC"),
    "w8a8": ("W8A8_DYNAMIC", "W8A8_DYNAMIC"),
    "int8": ("W8A8_DYNAMIC", "W8A8_DYNAMIC"),
    "w8a8_dynamic": ("W8A8_DYNAMIC", "W8A8_DYNAMIC"),
    "w8a8_static": ("W8A8", "W8A8"),
    "w8a8_mix": ("W8A8_MIX", "W8A8_DYNAMIC"),
    "w4a8": ("W8A8_DYNAMIC", "W4A8_DYNAMIC"),
    "w4a8_dynamic": ("W8A8_DYNAMIC", "W4A8_DYNAMIC"),
    "w4a16": ("W4A16", "W4A16"),
    "bf16": ("FLOAT", "FLOAT"),
    "fp16": ("FLOAT", "FLOAT"),
    "fp32": ("FLOAT", "FLOAT"),
    "none": ("FLOAT", "FLOAT"),
}


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ModelingError(f"invalid_scale_group_extent:{denominator}")
    return max(1, (int(numerator) + int(denominator) - 1) // int(denominator))


def _validate_native_float_dtype(raw: str) -> str:
    norm = str(raw or "").strip().lower()
    canon = _FLOAT_DTYPE_CANON.get(norm)
    if canon is None:
        raise ModelingError(f"native_float_dtype_invalid:{raw!r}")
    return canon


def _normalize_native_float(checkpoint: CheckpointQuantMap, model: ModelSpec) -> str:
    if str(checkpoint.native_float_dtype or "").strip():
        return _validate_native_float_dtype(checkpoint.native_float_dtype)
    qspec = getattr(model, "quantization", None)
    for cand in (
        getattr(qspec, "native_format", "") if qspec is not None else "",
        getattr(model, "activation_dtype", ""),
        getattr(model, "weight_dtype", ""),
    ):
        text = str(cand or "").strip().lower()
        if text in _FLOAT_DTYPE_CANON:
            return _FLOAT_DTYPE_CANON[text]
    raise ModelingError(
        f"native_float_dtype_missing:checkpoint={checkpoint.provenance.checkpoint_id}:model={model.name}"
    )


def _normalize_scale_dtype(raw: str, *, default_dtype: str = "fp32") -> tuple[str, int]:
    norm = str(raw or "").strip().lower()
    if not norm:
        norm = default_dtype
    if norm in {"e8m0", "fp8_e8m0"}:
        return "e8m0", SCALE_BYTES_E8M0
    if norm in {"bf16", "bfloat16"}:
        return "bf16", SCALE_BYTES_FP16
    if norm in {"fp16", "float16"}:
        return "fp16", SCALE_BYTES_FP16
    if norm in {"fp32", "float32"}:
        return "fp32", SCALE_BYTES_FP32
    raise ModelingError(f"unknown_cache_scale_dtype:{raw!r}")


def _effective_option_map(options: ResolvedStackOptions) -> dict[str, Any]:
    return {str(k): v for k, v in options.effective_mechanisms}


def _is_ple_engram(model: ModelSpec, plan: LayerPlan) -> bool:
    if not plan.engram:
        return False
    if int(plan.conv_kernel or 0) > 0:
        return True
    src = getattr(model.engram, "source", ()) or ()
    return any(str(k).startswith("ple_") for k, _ in src)


def _is_window_compressed_mla(plan: LayerPlan) -> bool:
    return bool(
        str(plan.attention or "").lower() == "mla"
        and (
            str(plan.mla_variant or "").lower() == "latent_mqa"
            or (int(plan.window or 0) > 0 and (bool(plan.owns_long_kv) or int(plan.compress_ratio or 1) > 1))
        )
    )


def _guard_a5_only(hw: str, method: str, tag: str) -> None:
    if hw in A2_A3_INT_ONLY_FAMILIES:
        raise ModelingError(
            f"hw_isa_incompatible:{hw}:linear_method={method}:tag={tag}:"
            "A2/A3 supports INT8/INT4 quantization only"
        )


def _w4a8_nz_eligible(*, is_moe: bool, use_nz: bool, k_dim: int, n_dim: int) -> bool:
    if is_moe:
        return True
    if not use_nz:
        return False
    if k_dim > 0 and n_dim > 0:
        return (k_dim % NZ_K_ALIGN == 0) and (n_dim % NZ_N_ALIGN == 0)
    return True


def _dense_float_abi(dtype: str) -> TensorPrecisionABI:
    return TensorPrecisionABI(
        linear_method="dense",
        input_dtype=dtype,
        weight_dtype=dtype,
        output_dtype=dtype,
        compute_dtype=dtype,
        accum_dtype="fp32",
        weight_layout="ND",
    )


def _resolve_a5_fp_tag_to_abi(
    tag: str, *, hw: str, nat: str, group_size: tuple[int, ...],
) -> TensorPrecisionABI | None:
    if tag in _MXFP8_TAGS:
        _guard_a5_only(hw, "mx_fp8", tag)
        blk = group_size if group_size else (1, MXFP_GROUP_SIZE)
        return TensorPrecisionABI(
            linear_method="mx_fp8",
            input_dtype="fp8_e4m3",
            weight_dtype="fp8_e4m3",
            output_dtype=nat,
            compute_dtype="fp8_e4m3",
            accum_dtype="fp32",
            weight_layout="ND",
            weight_block=blk,
            weight_scale_dtype="e8m0",
            weight_scale_bytes=SCALE_BYTES_E8M0,
            act_group_size=MXFP_GROUP_SIZE,
            act_scale_dtype="e8m0",
            act_scale_pack="paired_e8m0",
        )
    if tag in _MXFP4_W4A8_TAGS or tag in _MXFP4_W4A4_TAGS:
        _guard_a5_only(hw, "mx_fp4", tag)
        blk = group_size if group_size else (1, MXFP_GROUP_SIZE)
        in_dtype = "fp4_e2m1" if tag in _MXFP4_W4A4_TAGS else "fp8_e4m3"
        return TensorPrecisionABI(
            linear_method="mx_fp4",
            input_dtype=in_dtype,
            weight_dtype="fp4_e2m1",
            output_dtype=nat,
            compute_dtype="fp4_e2m1",
            accum_dtype="fp32",
            weight_layout="ND",
            weight_block=blk,
            weight_scale_dtype="e8m0",
            weight_scale_bytes=SCALE_BYTES_E8M0,
            weight_metadata=(("packed_fp4_per_byte", INT4_PACKED_PER_BYTE),),
            act_group_size=MXFP_GROUP_SIZE,
            act_scale_dtype="e8m0",
            act_scale_pack="paired_e8m0",
        )
    if tag in _FP8_TAGS:
        _guard_a5_only(hw, "fp8", tag)
        blk = group_size if group_size else (FP8_DEFAULT_BLOCK_SIZE, FP8_DEFAULT_BLOCK_SIZE)
        fp8_dt = "fp8_e5m2" if tag == "FP8_E5M2" else "fp8_e4m3"
        return TensorPrecisionABI(
            linear_method="fp8",
            input_dtype=fp8_dt,
            weight_dtype=fp8_dt,
            output_dtype=nat,
            compute_dtype=fp8_dt,
            accum_dtype="fp32",
            weight_layout="ND",
            weight_block=blk,
            weight_scale_dtype="fp32",
            weight_scale_bytes=SCALE_BYTES_FP32,
            act_group_size=FP8_DEFAULT_BLOCK_SIZE,
            act_scale_dtype="fp32",
        )
    return None


def _resolve_int_tag_to_abi(
    tag: str,
    *,
    nat: str,
    phase: str,
    role: str,
    is_moe: bool,
    use_nz: bool,
    k_dim: int,
    n_dim: int,
    group_size: tuple[int, ...],
) -> TensorPrecisionABI | None:
    if tag in _W8A8_DYNAMIC_TAGS or tag in _W8A8_STATIC_TAGS or tag in _W8A8_MIX_TAGS:
        if tag in _W8A8_MIX_TAGS:
            dyn = (str(phase).lower() == "prefill") or (str(role).lower() == "pd_prefill")
            meta: tuple[tuple[str, Any], ...] = (
                ("dynamic_act", dyn),
                ("quant_mode", "pdmix_prefill_dynamic" if dyn else "pdmix_decode_static"),
            )
        elif tag in _W8A8_STATIC_TAGS:
            meta = (("dynamic_act", False), ("quant_mode", "static"))
        else:
            meta = (("dynamic_act", True),)
        return TensorPrecisionABI(
            linear_method="w8a8",
            input_dtype="int8",
            weight_dtype="int8",
            output_dtype=nat,
            compute_dtype="int8",
            accum_dtype="int32",
            weight_layout="FRACTAL_NZ" if use_nz else "ND",
            weight_scale_dtype="fp32",
            weight_scale_bytes=SCALE_BYTES_FP32,
            weight_metadata=meta,
            act_scale_dtype="fp32",
        )
    if tag in _W4A8_TAGS:
        nz_active = _w4a8_nz_eligible(is_moe=is_moe, use_nz=use_nz, k_dim=k_dim, n_dim=n_dim)
        blk = group_size if group_size else (1, W4A8_DEFAULT_GROUP_SIZE)
        return TensorPrecisionABI(
            linear_method="w4a8",
            input_dtype="int8",
            weight_dtype="int4",
            output_dtype=nat,
            compute_dtype="int4",
            accum_dtype="int32",
            weight_layout="FRACTAL_NZ" if nz_active else "ND",
            weight_block=blk,
            weight_scale_dtype="fp32",
            weight_scale_bytes=SCALE_BYTES_FP32,
            weight_metadata=(("packed_int4_per_byte", INT4_PACKED_PER_BYTE), ("msd_scale_bias", True)),
            act_scale_dtype="fp32",
        )
    if tag in _W4A16_TAGS:
        blk = group_size if group_size else (1, W4A8_DEFAULT_GROUP_SIZE)
        is_w8 = tag.startswith("W8A16")
        return TensorPrecisionABI(
            linear_method="w4a16",
            input_dtype=nat,
            weight_dtype="int8" if is_w8 else "int4",
            output_dtype=nat,
            compute_dtype=nat,
            accum_dtype="fp32",
            weight_layout="FRACTAL_NZ" if use_nz else "ND",
            weight_block=blk,
            weight_scale_dtype=nat,
            weight_scale_bytes=SCALE_BYTES_FP16,
            weight_metadata=(
                (("weight_only_bits", W8A16_WEIGHT_BITS),)
                if is_w8 else (("packed_int4_per_byte", INT4_PACKED_PER_BYTE), ("weight_offset_dtype", nat))
            ),
        )
    return None


def map_quant_prefix(model_type: str, tensor_path: str) -> tuple[str, ...]:
    """Generate checkpoint lookup candidates via declarative ``checkpoint_registry`` tables."""
    return expand_checkpoint_tensor_paths(tensor_path, model_type)


def _apply_typed_rule_overrides(
    abi: TensorPrecisionABI, rule: TensorQuantRule | None,
) -> TensorPrecisionABI:
    """Honor explicit typed weight/activation/scale dtypes declared on a ``TensorQuantRule``."""
    if rule is None:
        return abi
    w_dt = str(rule.weight_dtype or "").strip().lower()
    a_dt = str(rule.activation_dtype or "").strip().lower()
    if abi.linear_method == "fp8":
        if w_dt in {"fp8_e4m3", "fp8_e5m2"}:
            in_dt = a_dt if a_dt in {"fp8_e4m3", "fp8_e5m2"} else w_dt
            abi = replace(abi, weight_dtype=w_dt, input_dtype=in_dt, compute_dtype=w_dt)
        elif a_dt in {"fp8_e4m3", "fp8_e5m2"}:
            abi = replace(abi, input_dtype=a_dt)
    elif abi.linear_method in {"mx_fp4", "mx_fp8"} and a_dt in {"fp4_e2m1", "mxfp4", "fp8_e4m3", "fp8_e5m2"}:
        in_dt = "fp4_e2m1" if a_dt == "mxfp4" else a_dt
        abi = replace(abi, input_dtype=in_dt)
    sc_raw = str(rule.scale_dtype or "").strip().lower()
    if sc_raw and sc_raw != "none" and abi.weight_scale_dtype:
        sc_dt, sc_bytes = _normalize_scale_dtype(sc_raw, default_dtype=abi.weight_scale_dtype)
        abi = replace(abi, weight_scale_dtype=sc_dt, weight_scale_bytes=sc_bytes)
    return abi


def resolve_raw_tag_to_abi(
    raw_tag: str,
    *,
    hw_family: str,
    native_float_dtype: str = "",
    phase: str = "decode",
    role: str = "colocated",
    is_moe: bool = False,
    use_nz: bool = False,
    k_dim: int = 0,
    n_dim: int = 0,
    group_size: tuple[int, ...] = (),
    rule: TensorQuantRule | None = None,
) -> TensorPrecisionABI:
    """Resolve one raw checkpoint quantization tag into a strict ``TensorPrecisionABI``."""
    tag = str(raw_tag or "").strip().upper()
    hw = normalize_hw_family(hw_family)

    # Explicit floating-point tags are preserved verbatim (never coerced to native_float_dtype).
    if tag in {"BF16", "BFLOAT16"}:
        return _dense_float_abi("bf16")
    if tag in {"FP16", "FLOAT16"}:
        return _dense_float_abi("fp16")
    if tag in {"FP32", "FLOAT32"}:
        return _dense_float_abi("fp32")

    nat = _validate_native_float_dtype(native_float_dtype)
    if tag in {"", "NONE", "FLOAT"}:
        return _dense_float_abi(nat)

    a5_abi = _resolve_a5_fp_tag_to_abi(tag, hw=hw, nat=nat, group_size=group_size)
    if a5_abi is not None:
        return _apply_typed_rule_overrides(a5_abi, rule)

    int_abi = _resolve_int_tag_to_abi(
        tag, nat=nat, phase=phase, role=role, is_moe=is_moe,
        use_nz=use_nz, k_dim=k_dim, n_dim=n_dim, group_size=group_size,
    )
    if int_abi is not None:
        return _apply_typed_rule_overrides(int_abi, rule)

    raise ModelingError(f"unknown_checkpoint_quant_tag:{raw_tag}")


def _hf_quant_tag_for_role(
    hf_cfg: Mapping[str, Any], *, is_routed_expert: bool,
) -> tuple[str, tuple[int, ...]] | None:
    method = str(hf_cfg.get("quant_method") or "").strip().lower()
    scale_fmt = str(hf_cfg.get("scale_fmt") or "").strip().lower()
    fmt = str(hf_cfg.get("fmt") or "").strip().lower()
    expert_dtype = str(hf_cfg.get("expert_dtype") or "").strip().lower()
    raw_block = hf_cfg.get("weight_block_size")
    blk: tuple[int, ...] = ()
    if isinstance(raw_block, (tuple, list)) and all(isinstance(x, int) and x > 0 for x in raw_block):
        blk = tuple(int(x) for x in raw_block)

    if is_routed_expert and expert_dtype in {"fp4", "mxfp4", "nvfp4"}:
        return "W4A8_MXFP", (1, MXFP_GROUP_SIZE)
    if method in {"mxfp8", "mx_fp8"} or (method == "fp8" and scale_fmt == "ue8m0"):
        return "W8A8_MXFP8", blk or (1, MXFP_GROUP_SIZE)
    if method in {"mxfp4", "mx_fp4"}:
        return "W4A8_MXFP", blk or (1, MXFP_GROUP_SIZE)
    if method == "fp8":
        fp8_tag = "FP8_E5M2" if fmt == "e5m2" else "FP8_DYNAMIC"
        return fp8_tag, blk or (FP8_DEFAULT_BLOCK_SIZE, FP8_DEFAULT_BLOCK_SIZE)
    if method in {"w8a8", "w8a8_dynamic"}:
        return "W8A8_DYNAMIC", blk
    if method in {"w4a8", "w4a8_dynamic"}:
        return ("W4A8_DYNAMIC" if is_routed_expert else "W8A8_DYNAMIC"), blk
    return None


def _project_blind_box_tag(
    projection_preset: str, *, role_name: str, is_routed_expert: bool,
) -> str:
    if role_name in CONSTRUCTOR_UNQUANTIZED_ROLES:
        return "FLOAT"
    norm = str(projection_preset or "").strip().lower()
    if not norm or norm in {"none", "bf16", "fp16", "float"}:
        return "FLOAT"
    pair = _PROJECTION_PRESET_TAGS.get(norm)
    if pair is None:
        raise ModelingError(f"unknown_target_quant_projection:{projection_preset}")
    return pair[1] if is_routed_expert else pair[0]


def _substitute_expert_paths(candidate_paths: Sequence[str], expert_id: int | None) -> tuple[str, ...]:
    if expert_id is None or expert_id == 0:
        return tuple(candidate_paths)
    tok = f".experts.{expert_id}."
    return tuple(p.replace(".experts.0.", tok) if ".experts.0." in p else p for p in candidate_paths)


_EXPERT_FULL_ENUM_LIMIT = 16


def _expert_boundary_ids(checkpoint: CheckpointQuantMap, num_experts: int) -> tuple[int, ...]:
    """Collect every expert ID that can differ across rules/exceptions/ignores plus endpoints."""
    count = max(1, int(num_experts or 1))
    if count <= 1:
        return (0,)
    expert_rules = [r for r in checkpoint.rules if ".experts." in r.pattern]
    presence_domains = [
        tuple(sorted(set(r.presence.expert_indices)))
        for r in expert_rules
        if r.presence is not None and r.presence.expert_indices
    ]
    has_unconstrained_expert_rule = any(
        r.presence is None or not r.presence.expert_indices for r in expert_rules
    )
    eff_count = count
    if (
        presence_domains
        and len(set(presence_domains)) == 1
        and not has_unconstrained_expert_rule
    ):
        dom = presence_domains[0]
        if dom[0] == 0 and dom[-1] == len(dom) - 1 and 0 < len(dom) < count:
            has_beyond = any(
                int(m) >= len(dom)
                for src in (*checkpoint.ignored_modules, *(e.pattern for e in checkpoint.layer_exceptions))
                for m in _EXPERT_ID_RE.findall(src)
            ) or any(
                val >= len(dom)
                for exc in checkpoint.layer_exceptions
                for val in exc.expert_indices
            )
            if not has_beyond:
                eff_count = len(dom)
    if eff_count <= _EXPERT_FULL_ENUM_LIMIT:
        return tuple(range(eff_count))
    ids: set[int] = {0, 1, eff_count - 1}
    for exc in checkpoint.layer_exceptions:
        for m in _EXPERT_ID_RE.findall(exc.pattern):
            val = int(m)
            if 0 <= val < eff_count:
                ids.add(val)
        for val in exc.expert_indices:
            if 0 <= val < eff_count:
                ids.add(val)
    for entry in checkpoint.ignored_modules:
        for m in _EXPERT_ID_RE.findall(entry):
            val = int(m)
            if 0 <= val < eff_count:
                ids.add(val)
    for rule in checkpoint.rules:
        for m in _EXPERT_ID_RE.findall(rule.pattern):
            val = int(m)
            if 0 <= val < eff_count:
                ids.add(val)
        pres = rule.presence
        if pres is not None and pres.expert_indices:
            present = frozenset(v for v in pres.expert_indices if 0 <= v < eff_count)
            if present and len(present) < eff_count:
                ids.add(min(present))
                ids.add(max(present))
                for missing in range(eff_count):
                    if missing not in present:
                        ids.add(missing)
                        break
    return tuple(sorted(ids))


def _lookup_modelslim_tag(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    expert_id: int | None = None,
) -> str | None:
    paths = _substitute_expert_paths(candidate_paths, expert_id)
    for path in paths:
        mapped = map_quant_prefix(model_type, path)
        for lid in layer_ids:
            for cand in mapped:
                tag = checkpoint.resolve_raw_tag(
                    cand, scope=scope, source_native_layer_id=lid, expert_id=expert_id,
                )
                if tag is not None:
                    return tag
    return None


def _lookup_exception_or_ignored(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    expert_id: int | None = None,
) -> str | None:
    paths = _substitute_expert_paths(candidate_paths, expert_id)
    for path in paths:
        mapped = map_quant_prefix(model_type, path)
        for lid in layer_ids:
            for cand in mapped:
                tag = checkpoint.resolve_exception_or_ignored(
                    cand, scope=scope, source_native_layer_id=lid, expert_id=expert_id,
                )
                if tag is not None:
                    return tag
    return None


def _rule_matches_candidate(
    rule: TensorQuantRule,
    cand: str,
    *,
    scope: ModelScope,
    lid: int,
    expert_id: int | None,
) -> bool:
    if rule.matches(cand, scope=scope, source_native_layer_id=lid, expert_id=expert_id):
        return True
    if not (rule.pattern.startswith("*") or rule.pattern.endswith("*")):
        return False
    if rule.presence is not None and not rule.presence.contains(
        scope=scope, source_native_layer_id=lid, expert_id=expert_id,
    ):
        return False
    return fnmatch.fnmatchcase(cand, rule.pattern)


def _lookup_non_wildcard_rule(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    expert_id: int | None = None,
) -> TensorQuantRule | None:
    specific_rules = tuple(r for r in checkpoint.rules if r.pattern != "*")
    if not specific_rules:
        return None
    paths = _substitute_expert_paths(candidate_paths, expert_id)
    for path in paths:
        mapped = map_quant_prefix(model_type, path)
        for lid in layer_ids:
            for cand in mapped:
                for rule in specific_rules:
                    if _rule_matches_candidate(
                        rule, cand, scope=scope, lid=lid, expert_id=expert_id,
                    ):
                        return rule
    return None


def _lookup_hf_rule_tag(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    expert_id: int | None = None,
) -> TensorQuantRule | None:
    specific = _lookup_non_wildcard_rule(
        checkpoint, model_type=model_type, candidate_paths=candidate_paths,
        scope=scope, layer_ids=layer_ids, expert_id=expert_id,
    )
    if specific is not None:
        return specific
    for rule in checkpoint.rules:
        if rule.pattern != "*":
            continue
        if rule.presence is None or any(
            rule.presence.contains(scope=scope, source_native_layer_id=lid, expert_id=expert_id)
            for lid in layer_ids
        ):
            return rule
    return None


def _lookup_modelslim_w13_tag(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    expert_id: int | None,
) -> str | None:
    """Resolve ``mlp.experts.w13`` and fail closed if gate/up tags are heterogeneous."""
    gate_paths = [p for p in candidate_paths if any(tok in p for tok in ("gate_proj", ".w1.", "w13"))]
    up_paths = [p for p in candidate_paths if any(tok in p for tok in ("up_proj", ".w3."))]
    gate_tag = _lookup_modelslim_tag(
        checkpoint, model_type=model_type, candidate_paths=gate_paths or candidate_paths,
        scope=scope, layer_ids=layer_ids, expert_id=expert_id,
    )
    up_tag = (
        _lookup_modelslim_tag(
            checkpoint, model_type=model_type, candidate_paths=up_paths,
            scope=scope, layer_ids=layer_ids, expert_id=expert_id,
        )
        if up_paths else None
    )
    if gate_tag is not None and up_tag is not None and gate_tag.upper() != up_tag.upper():
        raise ModelingError(
            f"heterogeneous_expert_gate_up_quantization:scope={scope}:layers={tuple(layer_ids)}:"
            f"expert={expert_id}:gate={gate_tag}:up={up_tag}"
        )
    return gate_tag if gate_tag is not None else up_tag


def _rule_signature(rule: TensorQuantRule | None) -> tuple[str, tuple[int, ...], str, str, str]:
    if rule is None:
        return ("", (), "", "", "")
    return (
        rule.raw_tag.upper(),
        rule.group_size,
        rule.weight_dtype,
        rule.activation_dtype,
        rule.scale_dtype,
    )


def _lookup_non_wildcard_w13_rule(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    expert_id: int | None,
) -> TensorQuantRule | None:
    """Resolve HF ``mlp.experts.w13`` specific rule and fail closed on gate/up mismatch."""
    gate_paths = [p for p in candidate_paths if any(tok in p for tok in ("gate_proj", ".w1.", "w13"))]
    up_paths = [p for p in candidate_paths if any(tok in p for tok in ("up_proj", ".w3."))]
    gate_rule = _lookup_non_wildcard_rule(
        checkpoint, model_type=model_type, candidate_paths=gate_paths or candidate_paths,
        scope=scope, layer_ids=layer_ids, expert_id=expert_id,
    )
    up_rule = (
        _lookup_non_wildcard_rule(
            checkpoint, model_type=model_type, candidate_paths=up_paths,
            scope=scope, layer_ids=layer_ids, expert_id=expert_id,
        )
        if up_paths else None
    )
    if gate_rule is not None and up_rule is not None and _rule_signature(gate_rule) != _rule_signature(up_rule):
        raise ModelingError(
            f"heterogeneous_expert_gate_up_quantization:scope={scope}:layers={tuple(layer_ids)}:"
            f"expert={expert_id}:gate={gate_rule.raw_tag}:up={up_rule.raw_tag}"
        )
    return gate_rule if gate_rule is not None else up_rule


def _resolve_single_expert_role_tag(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    role_name: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    is_routed_expert: bool,
    expert_id: int | None,
    target_quant_projection: str,
    hf_cfg: Mapping[str, Any],
) -> tuple[str, tuple[int, ...], bool, TensorQuantRule | None]:
    if checkpoint.source_format == "modelslim_description":
        if role_name == "mlp.experts.w13":
            explicit = _lookup_modelslim_w13_tag(
                checkpoint, model_type=model_type, candidate_paths=candidate_paths,
                scope=scope, layer_ids=layer_ids, expert_id=expert_id,
            )
        else:
            explicit = _lookup_modelslim_tag(
                checkpoint, model_type=model_type, candidate_paths=candidate_paths,
                scope=scope, layer_ids=layer_ids, expert_id=expert_id,
            )
        return (explicit, (), False, None) if explicit is not None else ("FLOAT", (), False, None)

    exc_or_ignored = _lookup_exception_or_ignored(
        checkpoint, model_type=model_type, candidate_paths=candidate_paths,
        scope=scope, layer_ids=layer_ids, expert_id=expert_id,
    )
    if exc_or_ignored is not None:
        return exc_or_ignored, (), False, None

    if role_name in CONSTRUCTOR_UNQUANTIZED_ROLES:
        return "FLOAT", (), False, None

    if checkpoint.source_format == "hf_quantization_config":
        if role_name == "mlp.experts.w13":
            specific_rule = _lookup_non_wildcard_w13_rule(
                checkpoint, model_type=model_type, candidate_paths=candidate_paths,
                scope=scope, layer_ids=layer_ids, expert_id=expert_id,
            )
        else:
            specific_rule = _lookup_non_wildcard_rule(
                checkpoint, model_type=model_type, candidate_paths=candidate_paths,
                scope=scope, layer_ids=layer_ids, expert_id=expert_id,
            )
        if specific_rule is not None:
            return specific_rule.raw_tag, specific_rule.group_size, False, specific_rule
        wildcard_rule = _lookup_hf_rule_tag(
            checkpoint, model_type=model_type, candidate_paths=candidate_paths,
            scope=scope, layer_ids=layer_ids, expert_id=expert_id,
        )
        if hf_cfg:
            hf_hit = _hf_quant_tag_for_role(hf_cfg, is_routed_expert=is_routed_expert)
            if hf_hit is not None:
                if hf_hit[0] in _MXFP8_TAGS or hf_hit[0] in _MXFP4_W4A8_TAGS:
                    mx_rule = (
                        replace(wildcard_rule, weight_dtype="", scale_dtype="")
                        if (wildcard_rule is not None and str(wildcard_rule.activation_dtype or "").strip())
                        else None
                    )
                    return hf_hit[0], hf_hit[1], False, mx_rule
                return hf_hit[0], hf_hit[1], False, wildcard_rule
        if wildcard_rule is not None:
            return wildcard_rule.raw_tag, wildcard_rule.group_size, False, wildcard_rule
        if hf_cfg and str(hf_cfg.get("quant_method") or "").strip().lower() == "ascend":
            raise ModelingError(
                f"unresolved_ascend_hf_quant_role:scope={scope}:layers={tuple(layer_ids)}:role={role_name}"
            )
        return "FLOAT", (), False, None

    if checkpoint.source_format == "unquantized_native":
        if target_quant_projection:
            proj_tag = _project_blind_box_tag(
                target_quant_projection, role_name=role_name, is_routed_expert=is_routed_expert,
            )
            return proj_tag, (), (proj_tag != "FLOAT"), None
        return "FLOAT", (), False, None

    if checkpoint.source_format == "explicit_synthetic_preset" and checkpoint.model_quant_type:
        proj_tag = _project_blind_box_tag(
            checkpoint.model_quant_type, role_name=role_name, is_routed_expert=is_routed_expert,
        )
        return proj_tag, (), False, None

    return "FLOAT", (), False, None


def _resolve_role_tag(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    role_name: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    is_routed_expert: bool,
    expert_ids: Sequence[int],
    target_quant_projection: str,
    hf_cfg: Mapping[str, Any],
) -> tuple[str, tuple[int, ...], bool, TensorQuantRule | None]:
    """Return ``(raw_tag, group_size, used_projection, rule)`` enforcing per-expert homogeneity."""
    if not is_routed_expert:
        return _resolve_single_expert_role_tag(
            checkpoint, model_type=model_type, role_name=role_name,
            candidate_paths=candidate_paths, scope=scope, layer_ids=layer_ids,
            is_routed_expert=False, expert_id=None,
            target_quant_projection=target_quant_projection, hf_cfg=hf_cfg,
        )
    check_ids = tuple(expert_ids) if expert_ids else (0,)
    first = _resolve_single_expert_role_tag(
        checkpoint, model_type=model_type, role_name=role_name,
        candidate_paths=candidate_paths, scope=scope, layer_ids=layer_ids,
        is_routed_expert=True, expert_id=check_ids[0],
        target_quant_projection=target_quant_projection, hf_cfg=hf_cfg,
    )
    first_key = (
        first[0].upper(),
        first[1],
        first[3].weight_dtype if first[3] is not None else "",
        first[3].activation_dtype if first[3] is not None else "",
        first[3].scale_dtype if first[3] is not None else "",
    )
    for eid in check_ids[1:]:
        cur = _resolve_single_expert_role_tag(
            checkpoint, model_type=model_type, role_name=role_name,
            candidate_paths=candidate_paths, scope=scope, layer_ids=layer_ids,
            is_routed_expert=True, expert_id=eid,
            target_quant_projection=target_quant_projection, hf_cfg=hf_cfg,
        )
        cur_key = (
            cur[0].upper(),
            cur[1],
            cur[3].weight_dtype if cur[3] is not None else "",
            cur[3].activation_dtype if cur[3] is not None else "",
            cur[3].scale_dtype if cur[3] is not None else "",
        )
        if cur_key != first_key:
            raise ModelingError(
                f"heterogeneous_expert_quantization:scope={scope}:layers={tuple(layer_ids)}:"
                f"role={role_name}:expert_{check_ids[0]}={first[0]}:expert_{eid}={cur[0]}"
            )
    return first


def _layer_role_descriptors(
    plan: LayerPlan, *, model: ModelSpec,
) -> tuple[list[tuple[str, tuple[str, ...], bool]], list[tuple[str, tuple[str, ...]]]]:
    """Return structural ``(role_specs, candidate_pack_groups)`` for one target layer."""
    idx = int(plan.index)
    attn = str(plan.attention or model.attention).lower()
    roles: list[tuple[str, tuple[str, ...], bool]] = []
    packs: list[tuple[str, tuple[str, ...]]] = []

    if attn == "kda":
        prefix_a, prefix_b = f"model.layers.{idx}.self_attn", f"model.layers.{idx}.linear_attn"
        if int(plan.gate_rank or 0) > 0:
            for r in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj", "f_b_proj", "g_b_proj", "o_proj"):
                roles.append((r, (f"{prefix_b}.{r}.weight", f"{prefix_a}.{r}.weight"), False))
            packs.append(("fused_qkvbfg_a_proj", ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")))
        else:
            for r in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "f_b_proj", "g_proj", "o_proj"):
                roles.append((r, (f"{prefix_a}.{r}.weight", f"{prefix_b}.{r}.weight"), False))
            packs.append(("in_proj_qkv", ("q_proj", "k_proj", "v_proj")))
            packs.append(("fused_bfg_proj", ("b_proj", "f_a_proj", "g_proj")))
        roles.append(("conv1d", (f"{prefix_a}.conv1d.weight", f"{prefix_b}.conv1d.weight"), False))
    elif attn == "gdn":
        prefix = f"model.layers.{idx}.linear_attn"
        for r in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            roles.append((f"linear_attn.{r}", (f"{prefix}.{r}.weight",), False))
        roles.append(("conv1d", (f"{prefix}.conv1d.weight",), False))
        packs.append(("fused_qkvz_proj", ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z")))
        packs.append(("fused_ba_proj", ("linear_attn.in_proj_b", "linear_attn.in_proj_a")))
    elif attn == "mla":
        prefix = f"model.layers.{idx}.self_attn"
        attn_short = f"layers.{idx}.attn"
        mla_alt: dict[str, tuple[str, ...]] = {
            "q_a_proj": (f"{attn_short}.wq_a.weight",),
            "q_b_proj": (f"{attn_short}.wq_b.weight",),
            "kv_a_proj_with_mqa": (f"{attn_short}.wkv.weight",),
            "kv_b_proj": (),
            "o_proj": (f"{attn_short}.wo_a.weight", f"{attn_short}.wo_b.weight"),
        }
        for r in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
            roles.append((f"mla.{r}", (f"{prefix}.{r}.weight", *mla_alt[r]), False))
        if plan.mla_output_gate:
            roles.append(("mla.g_proj", (f"{prefix}.g_proj.weight",), False))
        packs.append(("fused_qkv_a_proj", ("mla.q_a_proj", "mla.kv_a_proj_with_mqa")))
        if plan.indexer:
            idx_p, idx_short = f"{prefix}.indexer", f"{attn_short}.indexer"
            roles.append(("indexer.wk", (f"{idx_p}.wk.weight", f"{idx_short}.wk.weight"), False))
            roles.append(("indexer.weights_proj", (f"{idx_p}.weights_proj.weight", f"{idx_short}.weights_proj.weight"), False))
            packs.append(("wk_weights_proj", ("indexer.wk", "indexer.weights_proj")))
    else:
        prefix = f"model.layers.{idx}.self_attn"
        for r in ("q_proj", "k_proj", "v_proj", "o_proj"):
            roles.append((r, (f"{prefix}.{r}.weight",), False))
        packs.append(("qkv_proj", ("q_proj", "k_proj", "v_proj")))
        if plan.indexer:
            for r in ("index_q_proj", "index_k_proj"):
                roles.append((
                    r,
                    (
                        f"{prefix}.{r}.weight",
                        f"{prefix}.indexer.{r}.weight",
                        f"{prefix}.indexer.index_qk_proj.weight",
                        f"{prefix}.index_qk_proj.weight",
                    ),
                    False,
                ))
            packs.append(("index_qk_proj", ("index_q_proj", "index_k_proj")))

    if int(plan.residual_block_size or 0) > 0:
        roles.append(("self_attention_res_proj", (f"model.layers.{idx}.self_attention_res_proj.weight",), False))
        roles.append(("mlp_res_proj", (f"model.layers.{idx}.mlp_res_proj.weight",), False))
    if plan.hyper_connections or int(plan.hc_mult or 1) > 1:
        roles.append(("hyper_connection.pre_proj", (f"model.layers.{idx}.hyper_connection.pre_proj.weight",), False))
        roles.append(("hyper_connection.post_proj", (f"model.layers.{idx}.hyper_connection.post_proj.weight",), False))
    if _is_ple_engram(model, plan):
        roles.append(("ple.embed_proj", (f"model.layers.{idx}.ple.embed_proj.weight",), False))
        roles.append(("ple.key_proj", (f"model.layers.{idx}.ple.key_proj.weight",), False))
        roles.append(("ple.value_proj", (f"model.layers.{idx}.ple.value_proj.weight",), False))
        roles.append(("ple.conv1d", (f"model.layers.{idx}.ple.conv1d.weight",), False))

    mlp_p, bsm_p = f"model.layers.{idx}.mlp", f"model.layers.{idx}.block_sparse_moe"
    moe_p, ffn_p = f"model.layers.{idx}.moe", f"layers.{idx}.ffn"
    if str(plan.ffn).lower() == "moe":
        if int(plan.expert_latent_size or 0) > 0:
            roles.append((
                "mlp.latent_moe",
                (
                    f"{mlp_p}.latent_proj.weight",
                    f"{mlp_p}.latent_moe.weight",
                    f"{bsm_p}.routed_expert_down_proj.weight",
                    f"{bsm_p}.routed_expert_up_proj.weight",
                ),
                False,
            ))
        roles.append((
            "mlp.gate",
            (f"{mlp_p}.gate.weight", f"{bsm_p}.gate.weight", f"{moe_p}.gate.weight", f"{ffn_p}.gate.weight"),
            False,
        ))
        roles.append((
            "mlp.experts.w13",
            (
                f"{mlp_p}.experts.0.gate_proj.weight",
                f"{mlp_p}.experts.0.up_proj.weight",
                f"{mlp_p}.experts.w13.weight",
                f"{bsm_p}.experts.0.w1.weight",
                f"{bsm_p}.experts.0.w3.weight",
                f"{moe_p}.experts.w13_weight",
                f"{ffn_p}.experts.0.w1.weight",
                f"{ffn_p}.experts.0.w3.weight",
                f"{moe_p}.gate_proj.weight",
                f"{moe_p}.up_proj.weight",
            ),
            True,
        ))
        roles.append((
            "mlp.experts.w2",
            (
                f"{mlp_p}.experts.0.down_proj.weight",
                f"{mlp_p}.experts.w2.weight",
                f"{bsm_p}.experts.0.w2.weight",
                f"{moe_p}.experts.w2_weight",
                f"{ffn_p}.experts.0.w2.weight",
                f"{moe_p}.down_proj.weight",
            ),
            True,
        ))
        if int(plan.num_shared_experts or 0) > 0 or int(model.num_shared_experts or 0) > 0:
            se_p = f"model.layers.{idx}.share_expert"
            roles.append(("mlp.shared_expert.gate_proj", (f"{mlp_p}.shared_experts.gate_proj.weight", f"{mlp_p}.shared_expert.gate_proj.weight", f"{bsm_p}.shared_experts.gate_proj.weight", f"{se_p}.gate_proj.weight", f"{ffn_p}.shared_experts.w1.weight"), False))
            roles.append(("mlp.shared_expert.up_proj", (f"{mlp_p}.shared_experts.up_proj.weight", f"{mlp_p}.shared_expert.up_proj.weight", f"{bsm_p}.shared_experts.up_proj.weight", f"{se_p}.up_proj.weight", f"{ffn_p}.shared_experts.w3.weight"), False))
            roles.append(("mlp.shared_expert.down_proj", (f"{mlp_p}.shared_experts.down_proj.weight", f"{mlp_p}.shared_expert.down_proj.weight", f"{bsm_p}.shared_experts.down_proj.weight", f"{se_p}.down_proj.weight", f"{ffn_p}.shared_experts.w2.weight"), False))
            packs.append(("shared_expert.gate_up_proj", ("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj")))
    else:
        roles.append(("mlp.gate_proj", (f"{mlp_p}.gate_proj.weight",), False))
        roles.append(("mlp.up_proj", (f"{mlp_p}.up_proj.weight",), False))
        roles.append(("mlp.down_proj", (f"{mlp_p}.down_proj.weight",), False))
        packs.append(("gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")))

    return roles, packs


def _apply_homogeneous_packs(
    *,
    scope: str,
    native_layer: int,
    layer_abis: Mapping[str, TensorPrecisionABI],
    candidate_packs: Sequence[tuple[str, tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...], str]]:
    """Enforce Homogeneous Shard-Packing Law: pack iff every shard has identical ABI."""
    packed: list[tuple[str, tuple[str, ...], str]] = []
    for pack_name, shards in candidate_packs:
        if not all(role in layer_abis for role in shards):
            continue
        sigs = {layer_abis[role].abi_signature for role in shards}
        if len(sigs) == 1:
            packed.append((f"{scope}.L{native_layer}.{pack_name}", tuple(shards), next(iter(sigs))))
        elif pack_name == "fused_qkvbfg_a_proj":
            for sub_name, sub_shards in (
                ("in_proj_qkv", ("q_proj", "k_proj", "v_proj")),
                ("fused_bfg_a_proj", ("b_proj", "f_a_proj", "g_a_proj")),
            ):
                if all(r in layer_abis for r in sub_shards):
                    sub_sigs = {layer_abis[r].abi_signature for r in sub_shards}
                    if len(sub_sigs) == 1:
                        packed.append((f"{scope}.L{native_layer}.{sub_name}", sub_shards, next(iter(sub_sigs))))
    return packed


def _resolve_draft_ports(
    model: ModelSpec,
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    hw: str,
    native_float: str,
    phase: str,
    role: str,
    use_nz: bool,
    expert_ids: Sequence[int],
    target_quant_projection: str,
    hf_cfg: Mapping[str, Any],
) -> tuple[list[tuple[tuple[str, int, str], TensorPrecisionABI]], bool]:
    """Resolve optional draft (MTP / DSpark) ports structurally."""
    aux = model.auxiliary
    if aux is None or (aux.mtp is None and aux.dspark is None):
        return [], False
    draft_spec = aux.mtp if aux.mtp is not None else aux.dspark
    draft_layers = max(1, int(getattr(draft_spec, "layers", 1) or 1))
    start_id = int(model.body_layers or model.layers)
    has_moe_draft = bool(model.num_experts > 0 and aux.mtp is not None)
    draft_is_ple = any(_is_ple_engram(model, p) for p in model.layer_plans)
    ports: list[tuple[tuple[str, int, str], TensorPrecisionABI]] = []
    used_proj = False

    for local_idx in range(draft_layers):
        offset_id = start_id + local_idx
        layer_ids = (local_idx, offset_id)
        role_defs: list[tuple[str, tuple[str, ...], bool]] = [
            ("self_attn.q_proj", (f"mtp.layers.{local_idx}.self_attn.q_proj.weight", f"mtp.layers.{offset_id}.self_attn.q_proj.weight", f"model.layers.{offset_id}.self_attn.q_proj.weight"), False),
            ("self_attn.o_proj", (f"mtp.layers.{local_idx}.self_attn.o_proj.weight", f"mtp.layers.{offset_id}.self_attn.o_proj.weight", f"model.layers.{offset_id}.self_attn.o_proj.weight"), False),
        ]
        if has_moe_draft:
            role_defs.extend([
                (
                    "mlp.gate",
                    (
                        f"mtp.layers.{local_idx}.mlp.gate.weight",
                        f"mtp.layers.{offset_id}.mlp.gate.weight",
                        f"model.layers.{offset_id}.mlp.gate.weight",
                    ),
                    False,
                ),
                (
                    "mlp.experts.w13",
                    (
                        f"mtp.layers.{local_idx}.mlp.experts.0.gate_proj.weight",
                        f"mtp.layers.{local_idx}.mlp.experts.0.up_proj.weight",
                        f"mtp.layers.{offset_id}.mlp.experts.0.gate_proj.weight",
                        f"mtp.layers.{offset_id}.mlp.experts.0.up_proj.weight",
                        f"mtp.layers.{local_idx}.mlp.experts.w13.weight",
                        f"mtp.{local_idx}.ffn.experts.0.w1.weight",
                        f"mtp.{local_idx}.ffn.experts.0.w3.weight",
                        f"model.layers.{offset_id}.mlp.experts.0.gate_proj.weight",
                        f"model.layers.{offset_id}.mlp.experts.0.up_proj.weight",
                    ),
                    True,
                ),
                (
                    "mlp.experts.w2",
                    (
                        f"mtp.layers.{local_idx}.mlp.experts.0.down_proj.weight",
                        f"mtp.layers.{offset_id}.mlp.experts.0.down_proj.weight",
                        f"mtp.layers.{local_idx}.mlp.experts.w2.weight",
                        f"mtp.{local_idx}.ffn.experts.0.w2.weight",
                        f"model.layers.{offset_id}.mlp.experts.0.down_proj.weight",
                    ),
                    True,
                ),
            ])
        else:
            role_defs.append(("mlp.down_proj", (f"mtp.layers.{local_idx}.mlp.down_proj.weight", f"dspark.layers.{local_idx}.mlp.down_proj.weight"), False))

        for role_name, cands, is_expert in role_defs:
            if checkpoint.source_format == "modelslim_description":
                check_ids = tuple(expert_ids) if is_expert else (None,)
                found_tags: list[str] = []
                for eid in check_ids:
                    if role_name == "mlp.experts.w13":
                        explicit = _lookup_modelslim_w13_tag(
                            checkpoint, model_type=model_type, candidate_paths=cands,
                            scope="draft", layer_ids=layer_ids, expert_id=eid,
                        )
                        if explicit is None:
                            explicit = _lookup_modelslim_w13_tag(
                                checkpoint, model_type=model_type, candidate_paths=cands,
                                scope="target", layer_ids=(offset_id,), expert_id=eid,
                            )
                    else:
                        explicit = _lookup_modelslim_tag(
                            checkpoint, model_type=model_type, candidate_paths=cands,
                            scope="draft", layer_ids=layer_ids, expert_id=eid,
                        )
                        if explicit is None:
                            explicit = _lookup_modelslim_tag(
                                checkpoint, model_type=model_type, candidate_paths=cands,
                                scope="target", layer_ids=(offset_id,), expert_id=eid,
                            )
                    found_tags.append(explicit if explicit is not None else "FLOAT")
                if len({t.upper() for t in found_tags}) > 1:
                    raise ModelingError(
                        f"heterogeneous_draft_expert_quantization:layer={local_idx}:role={role_name}:tags={sorted(set(found_tags))}"
                    )
                tag, blk, did_proj, matched_rule = found_tags[0], (), False, None
            else:
                tag, blk, did_proj, matched_rule = _resolve_role_tag(
                    checkpoint,
                    model_type=model_type,
                    role_name=role_name,
                    candidate_paths=cands,
                    scope="draft",
                    layer_ids=layer_ids,
                    is_routed_expert=is_expert,
                    expert_ids=expert_ids,
                    target_quant_projection="" if draft_is_ple else target_quant_projection,
                    hf_cfg={} if draft_is_ple else hf_cfg,
                )
            used_proj = used_proj or did_proj
            abi = resolve_raw_tag_to_abi(
                tag, hw_family=hw, native_float_dtype=native_float,
                phase=phase, role=role, is_moe=is_expert, use_nz=use_nz,
                group_size=blk, rule=matched_rule,
            )
            ports.append((("draft", local_idx, role_name), abi))
    return ports, used_proj


def _detect_kv_mode(
    checkpoint: CheckpointQuantMap,
    *,
    hw: str,
    native_float: str,
    effective_opts: Mapping[str, Any],
    is_mla: bool,
    is_window_compressed_mla: bool = False,
    kv_tag_override: str | None = None,
) -> tuple[str, str, bool, list[tuple[str, str, str]]]:
    """Resolve effective KV cache mode, scale dtype, FAQuant active flag, and dispositions."""
    disps: list[tuple[str, str, str]] = []
    explicit_kv = bool(effective_opts.get("kv_cache_dtype_explicit", False))
    opt_kv_raw = str(effective_opts.get("kv_cache_dtype") or "").strip().upper()
    use_explicit_kv = explicit_kv and bool(opt_kv_raw) and opt_kv_raw not in {"AUTO", "NONE"}
    if kv_tag_override is not None:
        raw_source = kv_tag_override
    elif use_explicit_kv:
        raw_source = opt_kv_raw
    else:
        raw_source = checkpoint.cache_rule.kv_tag or effective_opts.get("kv_cache_dtype") or ""
    raw_kv = str(raw_source or effective_opts.get("kv_cache_dtype") or "").strip().upper()
    aux_low = " ".join(checkpoint.auxiliary_buffers).lower()
    has_faquant = (
        ("FAQUANT" in raw_kv)
        or ("FAKQUANT" in raw_kv)
        or ((not use_explicit_kv) and any(tok in aux_low for tok in _FAQUANT_BUFFER_TOKENS))
    )
    is_mxfp_gqa = (not is_mla) and (
        (raw_kv in _MXFP8_KV_TAGS)
        or ("v_cache_scale" in aux_low and "mxfp" in raw_kv.lower())
    )

    if is_mxfp_gqa and has_faquant:
        has_faquant = False
        disps.append(("fa_quant", "downgraded", "mxfp8_kv_cache_supersedes_faquant"))

    if is_mxfp_gqa:
        if hw in A2_A3_INT_ONLY_FAMILIES:
            raise ModelingError(f"hw_isa_incompatible:{hw}:kv_cache=QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL")
        return "qk_mxfp8_dynamic_v_mxfp8_per_channel", "e8m0", False, disps

    if raw_kv in _MLA_FP8_KV_TAGS:
        if hw in A2_A3_INT_ONLY_FAMILIES:
            raise ModelingError(f"hw_isa_incompatible:{hw}:kv_cache={raw_kv}:A2/A3_requires_bf16_or_c8")
        scale_dtype, _ = _normalize_scale_dtype(checkpoint.cache_rule.kv_scale_dtype, default_dtype="fp32")
        return "fp8_e4m3", scale_dtype, has_faquant, disps

    if raw_kv in _INT8_KV_TAGS:
        if is_window_compressed_mla and hw in A2_A3_INT_ONLY_FAMILIES:
            raise ModelingError(
                f"kv_cache_int8_unsupported_on_window_compressed_mla:{hw}:kv_cache={raw_kv}"
            )
        scale_dtype, _ = _normalize_scale_dtype(checkpoint.cache_rule.kv_scale_dtype, default_dtype="fp32")
        return "int8", scale_dtype, has_faquant, disps

    if raw_kv in _NATIVE_KV_TAGS:
        if raw_kv in {"BF16", "BFLOAT16"}:
            return "bf16", "", has_faquant, disps
        if raw_kv in {"FP16", "FLOAT16"}:
            return "fp16", "", has_faquant, disps
        return native_float, "", has_faquant, disps

    raise ModelingError(f"unknown_kv_cache_dtype:{raw_kv}")


def _resolve_index_dtype(
    raw_tag: str | None,
    *,
    hw: str,
    native_float: str,
    fallback_kv_mode: str,
) -> str:
    text = str(raw_tag or "").strip().lower()
    if not text or text in {"auto", "none"}:
        if fallback_kv_mode == "fp8_e4m3":
            text = "fp8_e4m3"
        elif fallback_kv_mode == "int8":
            text = "int8"
        else:
            return native_float
    if text in _FP8_INDEX_TAGS:
        if hw in A2_A3_INT_ONLY_FAMILIES:
            raise ModelingError(f"hw_isa_incompatible:{hw}:indexer_kv_dtype={text}")
        return "fp8_e4m3"
    if text in _INT8_INDEX_TAGS:
        return "int8"
    if text in {"bf16", "bfloat16"}:
        return "bf16"
    if text in {"fp16", "float16"}:
        return "fp16"
    if text == "float":
        return native_float
    raise ModelingError(f"unknown_indexer_kv_dtype:{raw_tag!r}")


def _materialize_plane_bytes(planes: tuple[CachePlaneABI, ...]) -> tuple[CachePlaneABI, ...]:
    return tuple(
        plane if plane.bytes_per_unit > 0 else replace(plane, bytes_per_unit=plane.exact_bytes())
        for plane in planes
    )


def _append_mxfp_page_padding(planes: tuple[CachePlaneABI, ...]) -> tuple[CachePlaneABI, ...]:
    materialized = _materialize_plane_bytes(planes)
    payload_bytes = sum(p.bytes_per_unit for p in materialized)
    pitch = max(payload_bytes, max((p.page_pitch for p in materialized), default=payload_bytes))
    pad_bytes = pitch - payload_bytes
    pad_plane = CachePlaneABI(
        plane_id="page_padding",
        plane_kind="padding",
        logical_axes=("pad_byte",),
        logical_shape=(pad_bytes,),
        physical_shape=(pad_bytes,),
        dtype="uint8",
        layout="PA_NZ",
        offset_in_page=payload_bytes,
        alignment=1,
        page_pitch=pitch,
        allocation_frequency="page",
        write_trigger="page_assign",
        bytes_per_unit=pad_bytes,
    )
    return materialized + (pad_plane,)


def _build_unaligned_mxfp_gqa_planes(
    *,
    local_heads: int,
    head_dim: int,
    v_head_dim: int,
    kv_block_size: int,
) -> tuple[CachePlaneABI, ...]:
    """PA_NZ MXFP8 GQA planes when ``kv_block_size`` is smaller than 64 tokens."""
    k_phys = (local_heads, _ceil_div(head_dim, MXFP_KV_NZ_DIM_FRAG), kv_block_size, MXFP_KV_NZ_DIM_FRAG)
    k_bytes = math.prod(k_phys)
    ks_groups = _ceil_div(head_dim, MXFP_GROUP_SIZE)
    ks_phys = (
        local_heads,
        _ceil_div(kv_block_size, MXFP_K_SCALE_NZ_TOKEN_FRAG),
        _ceil_div(head_dim, MXFP_KV_SCALE_GROUP_SIZE),
        MXFP_K_SCALE_NZ_TOKEN_FRAG,
        MXFP_KV_SCALE_VALUES_PER_GROUP,
    )
    ks_bytes = math.prod(ks_phys)
    v_phys = (local_heads, _ceil_div(v_head_dim, MXFP_KV_NZ_DIM_FRAG), kv_block_size, MXFP_KV_NZ_DIM_FRAG)
    v_bytes = math.prod(v_phys)
    vs_groups = _ceil_div(kv_block_size, MXFP_GROUP_SIZE)
    vs_phys = (
        local_heads,
        _ceil_div(v_head_dim, MXFP_V_SCALE_NZ_DIM_FRAG),
        _ceil_div(kv_block_size, MXFP_KV_SCALE_GROUP_SIZE),
        MXFP_V_SCALE_NZ_DIM_FRAG,
        MXFP_KV_SCALE_VALUES_PER_GROUP,
    )
    vs_bytes = math.prod(vs_phys)
    pitch = k_bytes + ks_bytes + v_bytes + vs_bytes
    planes = (
        CachePlaneABI(
            plane_id="k_payload", plane_kind="payload",
            logical_axes=("head", "token", "dim"), logical_shape=(local_heads, kv_block_size, head_dim),
            physical_shape=k_phys, dtype="fp8_e4m3", layout="PA_NZ", offset_in_page=0,
            alignment=MXFP_KV_NZ_DIM_FRAG, page_pitch=pitch,
            allocation_frequency="page", write_trigger="append", bytes_per_unit=k_bytes,
        ),
        CachePlaneABI(
            plane_id="k_scale", plane_kind="scale",
            logical_axes=("head", "token", "group"), logical_shape=(local_heads, kv_block_size, ks_groups),
            physical_shape=ks_phys, dtype="e8m0", layout="PA_NZ",
            group_axes=(2,), group_extents=(MXFP_GROUP_SIZE,), scale_dtype="e8m0",
            offset_in_page=k_bytes, alignment=MXFP_K_SCALE_NZ_TOKEN_FRAG, page_pitch=pitch,
            allocation_frequency="page", write_trigger="append", bytes_per_unit=ks_bytes,
        ),
        CachePlaneABI(
            plane_id="v_payload", plane_kind="payload",
            logical_axes=("head", "token", "dim"), logical_shape=(local_heads, kv_block_size, v_head_dim),
            physical_shape=v_phys, dtype="fp8_e4m3", layout="PA_NZ", offset_in_page=k_bytes + ks_bytes,
            alignment=MXFP_KV_NZ_DIM_FRAG, page_pitch=pitch,
            allocation_frequency="page", write_trigger="append", bytes_per_unit=v_bytes,
        ),
        CachePlaneABI(
            plane_id="v_scale", plane_kind="scale",
            logical_axes=("head", "dim", "block_group"), logical_shape=(local_heads, v_head_dim, vs_groups),
            physical_shape=vs_phys, dtype="e8m0", layout="PA_NZ",
            group_axes=(1,), group_extents=(MXFP_GROUP_SIZE,), scale_dtype="e8m0",
            offset_in_page=k_bytes + ks_bytes + v_bytes, alignment=MXFP_V_SCALE_NZ_DIM_FRAG,
            page_pitch=pitch, allocation_frequency="page", write_trigger="page_reassign",
            bytes_per_unit=vs_bytes,
        ),
    )
    return _append_mxfp_page_padding(planes)


def _mqsmla_stack_planes(plan: LayerPlan, scale_dtype: str) -> tuple[CachePlaneABI, ...] | None:
    """Canonical 608/584 B MQSMLA row planes for window/compressed FP8 MLA layers."""
    if (
        int(plan.cache_row_elements) != MQSMLA_HEAD_WIDTH
        or int(plan.qk_rope_head_dim) != MQSMLA_ROPE_WIDTH
    ):
        return None
    norm_scale = str(scale_dtype or "").strip().lower()
    if norm_scale in MQSMLA_EXPONENT_SCALES:
        mode = "uint8"
        out_scale = "e8m0" if norm_scale in {"e8m0", "fp8_e8m0"} else "uint8"
    elif norm_scale == "bf16":
        mode = "bf16"
        out_scale = "bf16"
    else:
        return None
    nope = MQSMLA_HEAD_WIDTH - MQSMLA_ROPE_WIDTH
    rope_b = MQSMLA_ROPE_WIDTH * int(dtype_bytes("bf16"))
    scale_groups = nope // MQSMLA_NATIVE_GROUP
    scale_b = scale_groups * int(dtype_bytes(out_scale))
    pad_b = MQSMLA_MODE_PADDING[mode]
    data_b = nope + rope_b
    pitch = data_b + scale_b + pad_b
    return (
        CachePlaneABI(
            plane_id="mla_latent", plane_kind="payload",
            logical_axes=("tokens", "dim"), logical_shape=(1, nope), physical_shape=(1, nope),
            dtype="fp8_e4m3", layout="packed_row", group_axes=(1,), group_extents=(MQSMLA_NATIVE_GROUP,),
            scale_dtype=out_scale, suffix_shape=(MQSMLA_ROPE_WIDTH,), suffix_dtype="bf16",
            offset_in_page=0, alignment=1, page_pitch=pitch,
            allocation_frequency="token", write_trigger="append", bytes_per_unit=nope,
        ),
        CachePlaneABI(
            plane_id="mla_rotary", plane_kind="suffix",
            logical_axes=("tokens", "dim"), logical_shape=(1, MQSMLA_ROPE_WIDTH),
            physical_shape=(1, MQSMLA_ROPE_WIDTH), dtype="bf16", layout="ND",
            offset_in_page=nope, alignment=1, page_pitch=pitch,
            allocation_frequency="token", write_trigger="append", bytes_per_unit=rope_b,
        ),
        CachePlaneABI(
            plane_id="mla_latent_scale", plane_kind="scale",
            logical_axes=("tokens", "scale_groups"), logical_shape=(1, scale_groups),
            physical_shape=(1, scale_groups), dtype=out_scale, layout="ND",
            group_axes=(1,), group_extents=(MQSMLA_NATIVE_GROUP,),
            offset_in_page=data_b, alignment=1, page_pitch=pitch,
            allocation_frequency="token", write_trigger="append", bytes_per_unit=scale_b,
        ),
        CachePlaneABI(
            plane_id="mla_padding", plane_kind="padding",
            logical_axes=("padding_bytes",), logical_shape=(pad_b,), physical_shape=(pad_b,),
            dtype="uint8", layout="ND", offset_in_page=data_b + scale_b, alignment=1,
            page_pitch=pitch, allocation_frequency="token", write_trigger="append",
            bytes_per_unit=pad_b,
        ),
    )


def _apply_speculative_recurrent_overlay(
    model: ModelSpec,
    plan: LayerPlan,
    abi: CacheStateABI,
    *,
    parallel: ParallelSpec,
    checkpoint_digest: str,
    speculative_k: int,
    recoverssm_active: bool,
) -> CacheStateABI:
    """Apply vLLM-Ascend speculative rollback/replay residency onto delegated recurrent ABI."""
    if speculative_k <= 0 and not recoverssm_active:
        return replace(abi, planes=_materialize_plane_bytes(abi.planes), checkpoint_digest=checkpoint_digest)
    rollback_slots = max(0, int(speculative_k))
    spec_plan = speculative_state_plan(
        model, plan, parallel,
        query_tokens=max(1, 1 + rollback_slots),
        recover_ssm=recoverssm_active,
        capacity_padded=False,
    )
    rec_bytes = int(round(spec_plan.recurrent_bytes))
    conv_bytes = int(round(spec_plan.conv_bytes))
    if rec_bytes <= 0 or conv_bytes <= 0:
        raise ModelingError(
            f"recurrent_state_plan_empty:layer={abi.native_layer}:attn={plan.attention}:"
            f"rec_bytes={rec_bytes}:conv_bytes={conv_bytes}"
        )
    rec_plane, conv_plane = abi.planes[0], abi.planes[1]
    conv_total_slots = max(1, abi.history_slots) + rollback_slots
    planes: list[CachePlaneABI] = [
        replace(
            rec_plane,
            write_trigger="accepted_commit" if recoverssm_active else "append",
            bytes_per_unit=rec_bytes,
        ),
        replace(
            conv_plane,
            logical_shape=(conv_plane.logical_shape[0], conv_total_slots),
            physical_shape=(conv_plane.physical_shape[0], conv_total_slots),
            offset_in_page=rec_bytes,
            bytes_per_unit=conv_bytes,
        ),
    ]
    replay_bytes = int(round(spec_plan.replay_bytes))
    if replay_bytes > 0:
        q_slots = max(1, int(spec_plan.query_tokens))
        planes.append(CachePlaneABI(
            plane_id="replay_records",
            plane_kind="payload",
            logical_axes=("verify_slot", "record_byte"),
            logical_shape=(q_slots, max(1, replay_bytes // q_slots)),
            physical_shape=(q_slots, max(1, replay_bytes // q_slots)),
            dtype="uint8",
            layout="ND",
            offset_in_page=rec_bytes + conv_bytes,
            allocation_frequency="request_slot",
            write_trigger="verify",
            bytes_per_unit=replay_bytes,
        ))
    return replace(
        abi,
        planes=tuple(planes),
        checkpoint_digest=checkpoint_digest,
        rollback_slots=rollback_slots,
        checkpoint_slots=1 if recoverssm_active else 0,
        record_shape=rec_plane.logical_shape if recoverssm_active else (),
        record_dtype=rec_plane.dtype if recoverssm_active else "",
    )


def _apply_vllm_index_overlay(
    abi: CacheStateABI,
    *,
    model: ModelSpec,
    plan: LayerPlan,
    native_float: str,
    checkpoint_digest: str,
    placement: CachePlacementRule,
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE,
    phase: str = "decode",
) -> CacheStateABI:
    """Apply explicit ``cache_representations``, E8M0 scale groups, and exact KPool ``index_tail``."""
    model_meta = dict(getattr(model, "metadata", ()) or ())
    if _lookup_region_decl(model_meta, "index_k", plan=plan, phase=phase) is not None:
        planes = list(_row_planes_from_format(
            _resolve_index_cache_format(model, plan, phase=phase),
            int(plan.index_dim),
            prefix=f"l{abi.native_layer}.index_k",
            block_size=kv_block_size,
        ))
    else:
        planes = list(_materialize_plane_bytes(abi.planes))
    idx_dim = int(plan.index_dim or planes[0].logical_shape[-1])
    for i, plane in enumerate(planes):
        if plane.plane_kind == "scale" and plane.dtype == "e8m0":
            groups = _ceil_div(idx_dim, MXFP_GROUP_SIZE)
            target_shape = (1, groups) if len(plane.logical_axes) == 2 else (groups,)
            if plane.logical_shape != target_shape:
                planes[i] = replace(
                    plane,
                    logical_shape=target_shape,
                    physical_shape=target_shape,
                    group_axes=(len(target_shape) - 1,),
                    group_extents=(MXFP_GROUP_SIZE,),
                    bytes_per_unit=groups,
                )
    pool = max(1, int(abi.pooling_ratio))
    tail_slots = 0
    if pool > 1:
        planes = [replace(p, allocation_frequency="token_group") for p in planes]
        if str(plan.attention or "").lower() == "gqa":
            tail_slots = 2 * pool
            tail_bytes = tail_slots * idx_dim * int(dtype_bytes(native_float))
            tail_plane = CachePlaneABI(
                plane_id="index_tail",
                plane_kind="suffix",
                logical_axes=("tail_slot", "index_dim"),
                logical_shape=(tail_slots, idx_dim),
                physical_shape=(tail_slots, idx_dim),
                dtype=native_float,
                layout="ND",
                allocation_frequency="request_slot",
                write_trigger="append",
                bytes_per_unit=tail_bytes,
            )
        else:
            raw_cap = (model_meta.get("native_keypool_caller") or {}).get("tail_capacity", pool)
            tail_slots = max(1, int(raw_cap))
            tail_bytes = tail_slots * 2 * idx_dim * SCALE_BYTES_FP32
            tail_plane = CachePlaneABI(
                plane_id="index_tail",
                plane_kind="suffix",
                logical_axes=("tail_slot", "vector", "index_dim"),
                logical_shape=(tail_slots, 2, idx_dim),
                physical_shape=(tail_slots, 2, idx_dim),
                dtype="fp32",
                layout="ND",
                allocation_frequency="request_slot",
                write_trigger="append",
                bytes_per_unit=tail_bytes,
            )
        planes.append(tail_plane)
    return replace(
        abi,
        planes=tuple(planes),
        checkpoint_digest=checkpoint_digest,
        recipe_abi_id=f"index_k_{planes[0].dtype}_pool{pool}",
        tail_slots=tail_slots,
        placement=placement,
    )


def _schema_kv_tag(
    kv_mode: str,
    kv_scale_dtype: str,
    *,
    is_mla: bool,
    has_faquant: bool,
    native_float: str,
) -> str:
    if kv_mode == "qk_mxfp8_dynamic_v_mxfp8_per_channel":
        return "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"
    if kv_mode == "fp8_e4m3":
        return "MXFP8" if (is_mla and kv_scale_dtype == "e8m0") else "FP8_E4M3"
    if kv_mode == "int8":
        if is_mla:
            return "INT8_DYNAMIC"
        return "FAQUANT" if has_faquant else "C8"
    if kv_mode == "fp32" or kv_mode == native_float:
        return "FLOAT"
    return kv_mode.upper()


def _schema_index_tag(index_dtype: str, index_scale_dtype: str, *, native_float: str) -> str:
    if index_dtype == "fp8_e4m3":
        return "MXFP8" if index_scale_dtype == "e8m0" else "FP8_E4M3"
    if index_dtype == "int8":
        return "INT8"
    if index_dtype == "fp32" or index_dtype == native_float:
        return "FLOAT"
    return index_dtype.upper()


def _build_engram_or_ple_abi(
    model: ModelSpec,
    plan: LayerPlan,
    *,
    native_layer: int,
    native_float: str,
    engram_offload: bool,
    checkpoint_digest: str,
    phase: str,
    role: str,
    speculative_k: int = 0,
) -> CacheStateABI:
    if _is_ple_engram(model, plan):
        engram_spec = model.engram
        engram_src = dict(getattr(engram_spec, "source", ()) or ()) if engram_spec is not None else {}
        ple_conv_k = int(engram_src.get("ple_conv_kernel_size", DEFAULT_CONV_KERNEL_SIZE))
        max_ngram = max(1, int(getattr(engram_spec, "max_ngram", 1) or 1))
        ple_hist = max(1, (ple_conv_k - 1) * max_ngram) + max(0, int(speculative_k))
        hc_mult = max((int(getattr(p, "hc_mult", 1) or 1) for p in model.layer_plans), default=1)
        ple_dim = int(model.hidden_size)
        return CacheStateABI(
            scope="target",
            native_layer=native_layer,
            component_role="ple_conv_state",
            producer_key=f"target.l{native_layer}.ple_conv_state",
            planes=(CachePlaneABI(
                plane_id="ple_conv_state",
                plane_kind="payload",
                logical_axes=("history_slot", "hc_mult", "ple_dim"),
                logical_shape=(ple_hist, hc_mult, ple_dim),
                physical_shape=(ple_hist, hc_mult, ple_dim),
                dtype=native_float,
                layout="ND",
                allocation_frequency="request_slot",
                write_trigger="append",
                bytes_per_unit=ple_hist * hc_mult * ple_dim * int(dtype_bytes(native_float)),
            ),),
            phase=phase,
            deployment_role=role,
            checkpoint_digest=checkpoint_digest,
            recipe_abi_id="ple_conv_state",
            history_slots=ple_hist,
            placement=CachePlacementRule(token_ownership_axis="none", memory_tier="hbm"),
        )
    engram_spec = model.engram
    matched_tables = tuple(
        t for t in (getattr(engram_spec, "tables", ()) or ())
        if int(getattr(t, "layer_id", -1)) == int(native_layer)
    )
    if matched_tables:
        table_slots = sum(int(t.rows) for t in matched_tables)
        table_dim = int(matched_tables[0].width)
        row_b = float(getattr(engram_spec, "row_bytes", 0.0) or (table_dim * int(dtype_bytes(native_float))))
        storage_mode = str(getattr(engram_spec, "storage_mode", "device_bf16") or "device_bf16").lower()
        table_dtype = "fp8_e4m3" if "fp8" in storage_mode else ("int8" if "int8" in storage_mode else native_float)
        total_bytes = int(round(float(table_slots) * row_b))
    else:
        declared_slots = int(getattr(engram_spec, "vocab_size", 0) or 0)
        if declared_slots <= 0:
            raise ModelingError(f"engram_table_spec_missing:layer={native_layer}")
        table_slots = declared_slots
        table_dim = int(getattr(engram_spec, "embed_dim", model.hidden_size) or model.hidden_size)
        table_dtype = native_float
        total_bytes = table_slots * table_dim * int(dtype_bytes(native_float))
    return CacheStateABI(
        scope="target",
        native_layer=native_layer,
        component_role="engram_table",
        producer_key=f"target.l{native_layer}.engram_table",
        planes=(CachePlaneABI(
            plane_id="engram_table",
            plane_kind="payload",
            logical_axes=("slot", "embed_dim"),
            logical_shape=(table_slots, table_dim),
            physical_shape=(table_slots, table_dim),
            dtype=table_dtype,
            layout="ND",
            allocation_frequency="checkpoint",
            write_trigger="load",
            bytes_per_unit=total_bytes,
        ),),
        phase=phase,
        deployment_role=role,
        checkpoint_digest=checkpoint_digest,
        recipe_abi_id="engram_host_or_hbm_table",
        placement=CachePlacementRule(
            token_ownership_axis="none",
            memory_tier="dram" if engram_offload else "hbm",
        ),
    )


def _collect_delegated_cache_rules(
    model: ModelSpec,
    checkpoint: CheckpointQuantMap,
    *,
    hw: str,
    native_float: str,
    effective_opts: Mapping[str, Any],
    role: str,
    kv_block_size: int,
    kv_mode: str,
    kv_scale_dtype: str,
    has_faquant: bool,
    default_idx_dtype: str,
    index_scale_dtype: str,
    disps: list[tuple[str, str, str]],
) -> tuple[
    CacheQuantRule,
    dict[int, tuple[str, str, bool]],
    set[int],
    dict[int, LayerPlan],
    str,
    bool,
    bool,
]:
    plans = model.layer_plans
    has_mla = any(str(p.attention).lower() == "mla" for p in plans)
    explicit_kv = bool(effective_opts.get("kv_cache_dtype_explicit", False))
    opt_kv_raw = str(effective_opts.get("kv_cache_dtype") or "").strip().upper()
    use_explicit_kv = explicit_kv and bool(opt_kv_raw) and opt_kv_raw not in {"AUTO", "NONE"}
    explicit_idx = bool(effective_opts.get("indexer_kv_dtype_explicit", False))
    opt_idx_raw = str(effective_opts.get("indexer_kv_dtype") or "").strip().upper()
    use_explicit_idx = explicit_idx and bool(opt_idx_raw) and opt_idx_raw not in {"AUTO", "NONE"}
    has_layer_kv = bool(checkpoint.cache_rule.layer_scoped_kv) and not use_explicit_kv
    has_layer_idx = bool(checkpoint.cache_rule.layer_scoped_index) and not use_explicit_idx
    c8_producer_recorded = False
    any_faquant = has_faquant
    any_mxfp_gqa = (kv_mode == "qk_mxfp8_dynamic_v_mxfp8_per_channel")
    layer_kv_tags: list[tuple[int, str]] = []
    layer_idx_tags: list[tuple[int, str]] = []
    layer_kv_meta: dict[int, tuple[str, str, bool]] = {}
    unaligned_mxfp_layers: set[int] = set()
    plan_by_native: dict[int, LayerPlan] = {}
    eff_kv_scale = "" if kv_mode == "qk_mxfp8_dynamic_v_mxfp8_per_channel" else kv_scale_dtype

    for p in plans:
        n_id = p.native_layer_id if p.native_layer_id >= 0 else p.index
        plan_by_native[n_id] = p
        attn = str(p.attention or model.attention).lower()
        l_mode, l_scale, l_faq = kv_mode, kv_scale_dtype, has_faquant
        if has_layer_kv and attn not in {"kda", "gdn"}:
            l_mode, l_scale, l_faq, l_disps = _detect_kv_mode(
                checkpoint, hw=hw, native_float=native_float, effective_opts=effective_opts,
                is_mla=(attn == "mla"), is_window_compressed_mla=_is_window_compressed_mla(p),
                kv_tag_override=checkpoint.cache_rule.kv_tag_for_layer(n_id),
            )
            for item in l_disps:
                if item not in disps:
                    disps.append(item)
            any_faquant = any_faquant or l_faq
            any_mxfp_gqa = any_mxfp_gqa or (l_mode == "qk_mxfp8_dynamic_v_mxfp8_per_channel")
            if not eff_kv_scale and l_scale and l_mode != "qk_mxfp8_dynamic_v_mxfp8_per_channel":
                eff_kv_scale = l_scale
        if attn not in {"kda", "gdn"} and p.cache_kind != "none":
            if attn != "mla" and l_mode == "int8" and str(role).lower() == "pd_prefill":
                l_mode = native_float
                disps.append((
                    "kv_cache_c8_producer",
                    "downgraded",
                    f"scope=target:layer={n_id}:kv_c8_producer_role_keeps_native_{native_float}",
                ))
                c8_producer_recorded = True
            layer_kv_meta[n_id] = (l_mode, l_scale, l_faq)
            h_dim = int(p.head_dim or model.head_dim)
            v_dim = int(p.v_head_dim or h_dim)
            if (
                attn != "mla"
                and l_mode == "qk_mxfp8_dynamic_v_mxfp8_per_channel"
                and (
                    kv_block_size % MXFP_KV_SCALE_GROUP_SIZE != 0
                    or h_dim % MXFP_KV_SCALE_GROUP_SIZE != 0
                    or v_dim % MXFP_KV_NZ_DIM_FRAG != 0
                )
            ):
                unaligned_mxfp_layers.add(n_id)
                layer_kv_tags.append((n_id, "FLOAT"))
            else:
                layer_kv_tags.append((
                    n_id,
                    _schema_kv_tag(l_mode, l_scale, is_mla=(attn == "mla"), has_faquant=l_faq, native_float=native_float),
                ))
        if p.indexer and (p.owns_index_k or p.index_k_owner_layer_id < 0 or p.index_k_owner_layer_id == n_id):
            l_idx = (
                _resolve_index_dtype(
                    checkpoint.cache_rule.index_tag_for_layer(n_id) or effective_opts.get("indexer_kv_dtype"),
                    hw=hw, native_float=native_float, fallback_kv_mode=l_mode,
                )
                if has_layer_idx else default_idx_dtype
            )
            layer_idx_tags.append((n_id, _schema_index_tag(l_idx, index_scale_dtype, native_float=native_float)))

    rule = CacheQuantRule(
        kv_tag=_schema_kv_tag(kv_mode, eff_kv_scale, is_mla=has_mla, has_faquant=has_faquant, native_float=native_float),
        index_tag=_schema_index_tag(default_idx_dtype, index_scale_dtype, native_float=native_float),
        kv_scale_dtype=eff_kv_scale,
        index_scale_dtype=index_scale_dtype,
        kv_offset_dtype=checkpoint.cache_rule.kv_offset_dtype,
        layer_scoped_kv=tuple(layer_kv_tags),
        layer_scoped_index=tuple(layer_idx_tags),
        provenance=checkpoint.cache_rule.provenance,
    )
    return rule, layer_kv_meta, unaligned_mxfp_layers, plan_by_native, eff_kv_scale, any_faquant, any_mxfp_gqa


def _cache_placement_for_layer(
    model: ModelSpec,
    plan: LayerPlan,
    parallel: ParallelSpec,
    *,
    stage: int,
    base: CachePlacementRule,
) -> CachePlacementRule:
    is_mla_p = str(plan.attention or model.attention).lower() == "mla"
    _, tp_gather, _ = layer_dcp_factors(plan, parallel)
    pcp = max(1, int(parallel.pcp or 1))
    cp_shard = max(1, tp_gather * pcp)
    ranks = tuple(range(cp_shard)) if cp_shard > 1 else ()
    if is_mla_p:
        tok_ax = "dcp_head" if parallel.cp > 1 else ("cp_striped" if pcp > 1 else "replicated")
        return CachePlacementRule(
            pp_stage=stage,
            rank_ids=ranks,
            token_ownership_axis=tok_ax,
            replication_factor=max(1, parallel.tp // max(1, tp_gather)),
        )
    kv_heads = max(1, int(plan.num_kv_heads or plan.num_heads or model.num_kv_heads or model.num_heads or 1))
    raw_rep = max(1, parallel.tp // kv_heads)
    tok_ax = "dcp_head" if tp_gather > 1 else ("cp_striped" if pcp > 1 else base.token_ownership_axis)
    return replace(
        base,
        pp_stage=stage,
        rank_ids=ranks,
        token_ownership_axis=tok_ax,
        replication_factor=max(1, raw_rep // max(1, tp_gather)),
    )


def _index_placement_for_layer(
    model: ModelSpec,
    parallel: ParallelSpec,
    *,
    stage: int,
) -> CachePlacementRule:
    """Resolve index-K placement using ``dcp_indexer_sharded`` and shared 1-head replication."""
    idx_sharded = bool(parallel.cp > 1 and dcp_indexer_sharded(model))
    idx_dcp = min(max(1, parallel.tp), max(1, parallel.cp)) if idx_sharded else 1
    pcp = max(1, int(parallel.pcp or 1))
    cp_shard = max(1, idx_dcp * pcp)
    tok_ax = "dcp_head" if idx_dcp > 1 else ("cp_striped" if pcp > 1 else "replicated")
    return CachePlacementRule(
        pp_stage=stage,
        rank_ids=tuple(range(cp_shard)) if cp_shard > 1 else (),
        token_ownership_axis=tok_ax,
        replication_factor=max(1, parallel.tp // idx_dcp),
    )


def _finalize_kv_cache_abi(
    abi: CacheStateABI,
    model: ModelSpec,
    plan: LayerPlan,
    *,
    parallel: ParallelSpec,
    model_meta: Mapping[str, Any],
    stage: int,
    kv_block_size: int,
    phase: str,
    native_float: str,
    ckpt_digest: str,
    eff_kv_scale: str,
    l_mode: str,
    l_scale: str,
    l_faq: bool,
    unaligned_mxfp: bool,
    draft_cap: int | None = None,
) -> CacheStateABI:
    is_mla = str(plan.attention or model.attention).lower() == "mla"
    comp_ratio = 1 if abi.component_role == "window_kv" else abi.compression_ratio
    place = _cache_placement_for_layer(model, plan, parallel, stage=stage, base=abi.placement)
    if unaligned_mxfp:
        n_kv = int(plan.num_kv_heads or model.num_kv_heads or model.num_heads)
        h_dim = int(plan.head_dim or model.head_dim)
        v_dim = int(plan.v_head_dim or h_dim)
        planes = _build_unaligned_mxfp_gqa_planes(
            local_heads=local_kv_heads(n_kv, parallel.tp), head_dim=h_dim,
            v_head_dim=v_dim, kv_block_size=kv_block_size,
        )
        recipe = "qk_mxfp8_dynamic_v_mxfp8_per_channel"
    elif (not is_mla) and l_mode == "qk_mxfp8_dynamic_v_mxfp8_per_channel":
        planes = _append_mxfp_page_padding(abi.planes)
        recipe = "qk_mxfp8_dynamic_v_mxfp8_per_channel"
    else:
        region = (
            "window" if abi.component_role == "window_kv"
            else ("compressed_kv" if abi.component_role == "compressed_kv" else "mla_kv")
        )
        has_region_decl = is_mla and _lookup_region_decl(model_meta, region, plan=plan, phase=phase) is not None
        if has_region_decl:
            planes = _row_planes_from_format(
                cache_format(model, region, plan=plan, phase=phase),
                int(plan.cache_row_elements),
                prefix=f"{abi.scope}.l{abi.native_layer}.{region}",
                block_size=kv_block_size,
            )
            p_dt = planes[0].dtype
            s_dt = next((pl.dtype for pl in planes if pl.plane_kind == "scale"), native_float)
            recipe = f"mla_{p_dt}_{s_dt}"
        else:
            mqsmla = (
                _mqsmla_stack_planes(plan, l_scale)
                if (is_mla and abi.component_role in {"window_kv", "compressed_kv"} and l_mode == "fp8_e4m3")
                else None
            )
            if mqsmla is not None:
                planes = mqsmla
            else:
                planes = _materialize_plane_bytes(abi.planes)
                if l_scale and l_scale != eff_kv_scale:
                    planes = tuple(
                        replace(pl, dtype=l_scale, scale_dtype=l_scale, bytes_per_unit=replace(pl, dtype=l_scale).exact_bytes())
                        if pl.plane_kind == "scale" else pl
                        for pl in planes
                    )
            recipe = (
                f"mla_{l_mode}_{l_scale or native_float}" if is_mla
                else ("gqa_faquant" if l_faq else f"gqa_{l_mode}")
            )
    cp_interleave = kv_block_size if len(place.rank_ids) > 1 else abi.cp_interleave_size
    return replace(
        abi, planes=planes, checkpoint_digest=ckpt_digest, recipe_abi_id=recipe,
        compression_ratio=comp_ratio, placement=place, cp_interleave_size=cp_interleave,
        draft_context_cap=(draft_cap if draft_cap is not None else abi.draft_context_cap),
    )


def _build_compressor_state_abis(
    plans: Sequence[LayerPlan],
    *,
    stage_for: Any,
    ckpt_digest: str,
    role: str,
    phase: str,
) -> list[CacheStateABI]:
    """Emit FP32 ``compressor_state`` / ``index_compressor_state`` ABIs matching ``catalog.cache``."""
    out: list[CacheStateABI] = []
    for p in plans:
        n_id = p.native_layer_id if p.native_layer_id >= 0 else p.index
        rows = int(p.compressor_state_rows or 0)
        if rows <= 0:
            continue
        stage = stage_for(p)
        width = max(1, int(p.compressor_width or 1))
        if p.owns_long_kv:
            h_dim = int(p.head_dim or 1)
            shape = (rows, 2, width, h_dim)
            n_bytes = math.prod(shape) * SCALE_BYTES_FP32
            out.append(CacheStateABI(
                scope="target", native_layer=n_id, component_role="compressor_state",
                producer_key=f"target.l{n_id}.compressor_state",
                planes=(CachePlaneABI(
                    plane_id="compressor_state", plane_kind="payload",
                    logical_axes=("state_row", "kv", "width", "dim"),
                    logical_shape=shape, physical_shape=shape, dtype="fp32", layout="ND",
                    allocation_frequency="request_slot", write_trigger="append", bytes_per_unit=n_bytes,
                ),),
                phase=phase, deployment_role=role, checkpoint_digest=ckpt_digest,
                recipe_abi_id="compressor_state_fp32",
                placement=CachePlacementRule(pp_stage=stage, token_ownership_axis="replicated"),
            ))
        if p.owns_index_k and p.index_input == "hidden" and int(p.compress_ratio or 0) > 0:
            idx_dim = int(p.index_dim or 1)
            shape = (rows, 2, width, idx_dim)
            n_bytes = math.prod(shape) * SCALE_BYTES_FP32
            out.append(CacheStateABI(
                scope="target", native_layer=n_id, component_role="compressor_state",
                producer_key=f"target.l{n_id}.index_compressor_state",
                planes=(CachePlaneABI(
                    plane_id="index_compressor_state", plane_kind="payload",
                    logical_axes=("state_row", "kv", "width", "dim"),
                    logical_shape=shape, physical_shape=shape, dtype="fp32", layout="ND",
                    allocation_frequency="request_slot", write_trigger="append", bytes_per_unit=n_bytes,
                ),),
                phase=phase, deployment_role=role, checkpoint_digest=ckpt_digest,
                recipe_abi_id="index_compressor_state_fp32",
                placement=CachePlacementRule(pp_stage=stage, token_ownership_axis="replicated"),
            ))
    return out


def _apply_aliased_window_backing_tags(
    model: ModelSpec,
    parallel: ParallelSpec,
    *,
    stage_map: Sequence[int],
    kv_block_size: int,
    draft_kind: str,
    speculative_k: int,
    kv_abis: list[CacheStateABI],
    idx_abis: list[CacheStateABI],
    rec_abis: list[CacheStateABI],
) -> None:
    """Stamp ``page_group`` / ``overlay_region`` / ``alias_of`` matching ``cache_pools`` backings."""
    contract = resolve_pool_allocator_contract(
        model, parallel, block_size=kv_block_size,
        draft_kind=(draft_kind if (speculative_k > 0 and draft_kind) else None),
        query_tokens=(1 + speculative_k if speculative_k > 0 else 1),
    )
    if contract.pool_regime != REGIME_ALIASED_WINDOW_COMPRESSED:
        return
    pp = max(1, int(parallel.pp))
    last_stage = pp - 1
    owner_pg: dict[int, tuple[str, str]] = {}
    win_alias: dict[int, tuple[str, str]] = {}
    draft_alias: dict[int, tuple[str, str]] = {}
    for st in range(pp):
        st_plans = [p for p in model.layer_plans if (stage_map[p.index] if p.index < len(stage_map) else last_stage) == st]
        owners = sorted((p for p in st_plans if p.owns_long_kv), key=lambda p: int(p.index))
        swas = sorted((p for p in st_plans if int(p.window or 0) > 0), key=lambda p: int(p.index))
        if not owners:
            continue
        n_back = len(owners)
        for slot_idx, own in enumerate(owners):
            o_nid = own.native_layer_id if own.native_layer_id >= 0 else own.index
            pg = f"s{st}.alias_backing_{slot_idx}"
            o_key = f"target.l{o_nid}.compressed_kv"
            owner_pg[o_nid] = (pg, o_key)
            for sp in swas[slot_idx::n_back]:
                s_nid = sp.native_layer_id if sp.native_layer_id >= 0 else sp.index
                win_alias[s_nid] = (pg, o_key)
            if st == last_stage:
                draft_alias[slot_idx] = (pg, o_key)
    for i, a in enumerate(kv_abis):
        if a.scope == "target" and a.component_role == "compressed_kv" and a.native_layer in owner_pg:
            pg, _ = owner_pg[a.native_layer]
            kv_abis[i] = replace(a, page_group=pg, overlay_region="compressed_kv")
        elif a.scope == "target" and a.component_role == "window_kv" and a.native_layer in win_alias:
            pg, o_key = win_alias[a.native_layer]
            kv_abis[i] = replace(a, page_group=pg, overlay_region="window_kv", alias_of=o_key)
        elif a.scope == "draft" and a.component_role == "window_kv" and a.native_layer in draft_alias:
            pg, o_key = draft_alias[a.native_layer]
            kv_abis[i] = replace(a, page_group=pg, overlay_region="draft_window_kv", alias_of=o_key)
    for i, a in enumerate(idx_abis):
        if a.scope == "target" and a.native_layer in owner_pg:
            pg, _ = owner_pg[a.native_layer]
            idx_abis[i] = replace(a, page_group=pg, overlay_region="index_k")
    for i, a in enumerate(rec_abis):
        if a.scope == "target" and a.component_role == "compressor_state" and a.native_layer in owner_pg:
            pg, _ = owner_pg[a.native_layer]
            rec_abis[i] = replace(a, page_group=pg, overlay_region="compressor_state")


def _resolve_draft_kv_tag(
    model: ModelSpec,
    *,
    draft_kind: str,
    draft_spec: Any,
    hw: str,
    parallel: ParallelSpec,
    kv_mode: str,
    speculative_k: int,
    checkpoint: CheckpointQuantMap,
    effective_opts: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve draft-specific KV dtype override before inheriting target cache rules."""
    dq = getattr(draft_spec, "quant", None)
    if dq is not None and str(getattr(dq, "kv_dtype", "") or "").strip():
        return str(dq.kv_dtype).strip().upper()
    try:
        comp = resolve_speculative_companion(
            model.name, target_tp=parallel.tp, target_kv_dtype=kv_mode,
            hw_family=hw, method=draft_kind, num_speculative_tokens=speculative_k,
        )
        if comp.draft_kv_cache_dtype:
            return str(comp.draft_kv_cache_dtype).strip().upper()
    except ModelingError:
        pass
    if effective_opts is not None and bool(effective_opts.get("kv_cache_dtype_explicit", False)):
        opt_kv = str(effective_opts.get("kv_cache_dtype") or "").strip().upper()
        if opt_kv and opt_kv not in {"AUTO", "NONE"}:
            return opt_kv
    return checkpoint.cache_rule.kv_tag or None


def _append_draft_cache_abis(
    model: ModelSpec,
    checkpoint: CheckpointQuantMap,
    *,
    hw: str,
    delegated_rule: CacheQuantRule,
    parallel: ParallelSpec,
    effective_opts: Mapping[str, Any],
    model_meta: Mapping[str, Any],
    last_stage: int,
    kv_block_size: int,
    role: str,
    phase: str,
    native_float: str,
    ckpt_digest: str,
    kv_mode: str,
    eff_kv_scale: str,
    has_faquant: bool,
    speculative_k: int,
    recoverssm_active: bool,
    disps: list[tuple[str, str, str]],
    kv_abis: list[CacheStateABI],
    idx_abis: list[CacheStateABI],
    rec_abis: list[CacheStateABI],
) -> str:
    """Append active neural speculative draft (MTP / DSpark) cache ABIs on the final PP stage."""
    spec_method = str(effective_opts.get("spec_method") or "").strip().lower()
    aux = getattr(model, "auxiliary", None)
    if speculative_k <= 0 or aux is None or spec_method in ZERO_NEURAL_DRAFT_METHODS:
        return ""
    draft_kind = (
        "dspark" if (spec_method in {"dspark", "dflash"} and aux.dspark is not None)
        else ("mtp" if aux.mtp is not None else ("dspark" if aux.dspark is not None else ""))
    )
    d_plans = draft_layer_plans(model, draft_kind) if draft_kind else ()
    if not d_plans:
        return draft_kind
    draft_spec = optional_draft_spec(model, draft_kind)
    raw_dml = int(effective_opts.get("draft_max_model_len") or 0)
    cap_hint = type("_DraftCapHint", (), {"metadata": (("draft_max_model_len", raw_dml),)})() if raw_dml > 0 else None
    d_cap_val = effective_draft_context_cap(draft_spec, cap_hint)
    d_cap = d_cap_val if d_cap_val > 0 else None

    d_tag_override = _resolve_draft_kv_tag(
        model, draft_kind=draft_kind, draft_spec=draft_spec, hw=hw,
        parallel=parallel, kv_mode=kv_mode, speculative_k=speculative_k,
        checkpoint=checkpoint, effective_opts=effective_opts,
    )
    d_has_mla = any(str(p.attention or model.attention).lower() == "mla" for p in d_plans)
    d_kv_mode, d_kv_scale, d_has_faq, _ = _detect_kv_mode(
        checkpoint, hw=hw, native_float=native_float, effective_opts=effective_opts,
        is_mla=d_has_mla, kv_tag_override=d_tag_override,
    )
    d_eff_scale = "" if d_kv_mode == "qk_mxfp8_dynamic_v_mxfp8_per_channel" else d_kv_scale
    d_layer_tags: list[tuple[int, str]] = []
    d_layer_meta: dict[int, tuple[str, str, bool]] = {}
    d_unaligned: set[int] = set()
    for p in d_plans:
        dn = p.native_layer_id if p.native_layer_id >= 0 else p.index
        d_is_mla = str(p.attention or model.attention).lower() == "mla"
        dm, ds, df = d_kv_mode, d_eff_scale, d_has_faq
        if (not d_is_mla) and dm == "int8" and str(role).lower() == "pd_prefill":
            dm = native_float
            disps.append(("kv_cache_c8_producer", "downgraded", f"scope=draft:layer={dn}:kv_c8_producer_role_keeps_native_{native_float}"))
        d_layer_meta[dn] = (dm, ds, df)
        h_dim = int(p.head_dim or model.head_dim)
        v_dim = int(p.v_head_dim or h_dim)
        if (not d_is_mla) and dm == "qk_mxfp8_dynamic_v_mxfp8_per_channel" and (
            kv_block_size % MXFP_KV_SCALE_GROUP_SIZE != 0 or h_dim % MXFP_KV_SCALE_GROUP_SIZE != 0 or v_dim % MXFP_KV_NZ_DIM_FRAG != 0
        ):
            d_unaligned.add(dn)
            d_layer_tags.append((dn, "FLOAT"))
        else:
            d_layer_tags.append((dn, _schema_kv_tag(dm, ds, is_mla=d_is_mla, has_faquant=df, native_float=native_float)))
    draft_rule = replace(
        delegated_rule,
        kv_tag=_schema_kv_tag(d_kv_mode, d_eff_scale, is_mla=d_has_mla, has_faquant=d_has_faq, native_float=native_float),
        kv_scale_dtype=d_eff_scale,
        layer_scoped_kv=tuple(d_layer_tags),
        layer_scoped_index=(),
    )
    d_raw_kv, d_raw_idx, d_raw_rec = resolve_checkpoint_cache_abis(
        d_plans, replace(checkpoint, cache_rule=draft_rule), parallel, native_float,
        role=role, phase=phase, kv_block_size=kv_block_size,
        kda_parallel_axis="tp_x_pcp", stage=last_stage,
    )
    d_plan_by_native = {(p.native_layer_id if p.native_layer_id >= 0 else p.index): p for p in d_plans}
    for d_abi in d_raw_kv:
        d_plan = d_plan_by_native[d_abi.native_layer]
        dm, ds, df = d_layer_meta.get(d_abi.native_layer, (d_kv_mode, d_eff_scale, d_has_faq))
        kv_abis.append(_finalize_kv_cache_abi(
            d_abi, model, d_plan, parallel=parallel, model_meta=model_meta, stage=last_stage,
            kv_block_size=kv_block_size, phase=phase, native_float=native_float,
            ckpt_digest=ckpt_digest, eff_kv_scale=d_eff_scale,
            l_mode=dm, l_scale=ds, l_faq=df,
            unaligned_mxfp=(d_abi.native_layer in d_unaligned), draft_cap=d_cap,
        ))
    for d_abi in d_raw_idx:
        d_plan = d_plan_by_native[d_abi.native_layer]
        idx_place = _index_placement_for_layer(model, parallel, stage=last_stage)
        idx_abis.append(replace(
            _apply_vllm_index_overlay(
                d_abi, model=model, plan=d_plan, native_float=native_float,
                checkpoint_digest=ckpt_digest, placement=idx_place,
                kv_block_size=kv_block_size, phase=phase,
            ),
            cp_interleave_size=(kv_block_size if len(idx_place.rank_ids) > 1 else d_abi.cp_interleave_size),
            draft_context_cap=d_cap,
        ))
    for d_abi in d_raw_rec:
        d_plan = d_plan_by_native[d_abi.native_layer]
        d_rec = _apply_speculative_recurrent_overlay(
            model, d_plan, d_abi,
            parallel=parallel, checkpoint_digest=ckpt_digest,
            speculative_k=speculative_k, recoverssm_active=recoverssm_active,
        )
        rec_abis.append(replace(
            d_rec, placement=replace(d_rec.placement, pp_stage=last_stage), draft_context_cap=d_cap,
        ))
    return draft_kind


def build_cache_state_abis(
    model: ModelSpec,
    checkpoint: CheckpointQuantMap,
    *,
    hw: str,
    model_type: str = "",
    parallel: ParallelSpec,
    effective_opts: Mapping[str, Any],
    role: str,
    phase: str,
    speculative_k: int,
    recoverssm_active: bool,
) -> tuple[
    tuple[CacheStateABI, ...],
    tuple[CacheStateABI, ...],
    tuple[CacheStateABI, ...],
    tuple[tuple[str, str, str], ...],
    bool,
    bool,
]:
    """Validate vLLM-Ascend cache policy and delegate ABI instantiation to L0 ``schema.stack``."""
    del model_type
    plans = model.layer_plans
    native_float = _normalize_native_float(checkpoint, model)
    scoped_kv_layers = {int(lid) for lid, _ in checkpoint.cache_rule.layer_scoped_kv}
    has_mla = any(str(p.attention).lower() == "mla" for p in plans)
    has_unscoped_wc_mla = any(
        _is_window_compressed_mla(p)
        and (p.native_layer_id if p.native_layer_id >= 0 else p.index) not in scoped_kv_layers
        for p in plans
    )
    kv_mode, kv_scale_dtype, has_faquant, disps = _detect_kv_mode(
        checkpoint, hw=hw, native_float=native_float, effective_opts=effective_opts,
        is_mla=has_mla, is_window_compressed_mla=has_unscoped_wc_mla,
    )
    kv_block_size = int(effective_opts.get("kv_block_size") or DEFAULT_KV_BLOCK_SIZE)
    ckpt_digest = checkpoint.digest()
    engram_offload = bool(effective_opts.get("engram_cpu_offload", False))
    explicit_idx = bool(effective_opts.get("indexer_kv_dtype_explicit", False))
    opt_idx_val = str(effective_opts.get("indexer_kv_dtype") or "").strip()
    idx_source = (
        opt_idx_val
        if (explicit_idx and opt_idx_val and opt_idx_val.lower() not in {"auto", "none"})
        else (checkpoint.cache_rule.index_tag or effective_opts.get("indexer_kv_dtype"))
    )
    default_idx_dtype = _resolve_index_dtype(
        idx_source,
        hw=hw, native_float=native_float, fallback_kv_mode=kv_mode,
    )
    index_scale_dtype, _ = _normalize_scale_dtype(
        checkpoint.cache_rule.index_scale_dtype, default_dtype="fp32",
    )
    (
        delegated_rule, layer_kv_meta, unaligned_mxfp_layers, plan_by_native,
        eff_kv_scale, any_faquant, any_mxfp_gqa,
    ) = _collect_delegated_cache_rules(
        model, checkpoint, hw=hw, native_float=native_float, effective_opts=effective_opts,
        role=role, kv_block_size=kv_block_size, kv_mode=kv_mode, kv_scale_dtype=kv_scale_dtype,
        has_faquant=has_faquant, default_idx_dtype=default_idx_dtype,
        index_scale_dtype=index_scale_dtype, disps=disps,
    )
    raw_kv, raw_idx, raw_rec = resolve_checkpoint_cache_abis(
        plans, replace(checkpoint, cache_rule=delegated_rule), parallel, native_float,
        role=role, phase=phase, kv_block_size=kv_block_size, kda_parallel_axis="tp_x_pcp",
    )
    pp_part = tuple(int(x) for x in (effective_opts.get("pp_layer_partition") or ()))
    runtime_pp_hint = type("_PpHint", (), {"pp_layer_partition": pp_part})() if pp_part else None
    stage_map = layer_stage_map(model, parallel, runtime_pp_hint)
    last_stage = max(0, int(parallel.pp) - 1)
    model_meta = dict(getattr(model, "metadata", ()) or ())

    def _stage_for(plan: LayerPlan) -> int:
        return stage_map[plan.index] if 0 <= int(plan.index) < len(stage_map) else last_stage

    kv_abis: list[CacheStateABI] = []
    for abi in raw_kv:
        l_mode, l_scale, l_faq = layer_kv_meta.get(abi.native_layer, (kv_mode, eff_kv_scale, has_faquant))
        plan = plan_by_native[abi.native_layer]
        kv_abis.append(_finalize_kv_cache_abi(
            abi, model, plan, parallel=parallel, model_meta=model_meta, stage=_stage_for(plan),
            kv_block_size=kv_block_size, phase=phase, native_float=native_float,
            ckpt_digest=ckpt_digest, eff_kv_scale=eff_kv_scale,
            l_mode=l_mode, l_scale=l_scale, l_faq=l_faq,
            unaligned_mxfp=(abi.native_layer in unaligned_mxfp_layers),
        ))
    idx_abis: list[CacheStateABI] = []
    for abi in raw_idx:
        plan = plan_by_native[abi.native_layer]
        idx_place = _index_placement_for_layer(model, parallel, stage=_stage_for(plan))
        idx_abis.append(replace(
            _apply_vllm_index_overlay(
                abi, model=model, plan=plan, native_float=native_float,
                checkpoint_digest=ckpt_digest, placement=idx_place,
                kv_block_size=kv_block_size, phase=phase,
            ),
            cp_interleave_size=(kv_block_size if len(idx_place.rank_ids) > 1 else abi.cp_interleave_size),
        ))
    rec_abis: list[CacheStateABI] = []
    for abi in raw_rec:
        plan = plan_by_native[abi.native_layer]
        rec_abi = _apply_speculative_recurrent_overlay(
            model, plan, abi,
            parallel=parallel, checkpoint_digest=ckpt_digest,
            speculative_k=speculative_k, recoverssm_active=recoverssm_active,
        )
        rec_abis.append(replace(rec_abi, placement=replace(rec_abi.placement, pp_stage=_stage_for(plan))))
    rec_abis.extend(_build_compressor_state_abis(plans, stage_for=_stage_for, ckpt_digest=ckpt_digest, role=role, phase=phase))
    for p in plans:
        if p.engram:
            n_id = p.native_layer_id if p.native_layer_id >= 0 else p.index
            engram_abi = _build_engram_or_ple_abi(
                model, p, native_layer=n_id, native_float=native_float,
                engram_offload=engram_offload, checkpoint_digest=ckpt_digest,
                phase=phase, role=role, speculative_k=speculative_k,
            )
            rec_abis.append(replace(engram_abi, placement=replace(engram_abi.placement, pp_stage=_stage_for(p))))

    draft_kind = _append_draft_cache_abis(
        model, checkpoint, hw=hw, delegated_rule=delegated_rule, parallel=parallel,
        effective_opts=effective_opts, model_meta=model_meta, last_stage=last_stage,
        kv_block_size=kv_block_size, role=role, phase=phase, native_float=native_float,
        ckpt_digest=ckpt_digest, kv_mode=kv_mode, eff_kv_scale=eff_kv_scale,
        has_faquant=has_faquant, speculative_k=speculative_k, recoverssm_active=recoverssm_active,
        disps=disps, kv_abis=kv_abis, idx_abis=idx_abis, rec_abis=rec_abis,
    )
    _apply_aliased_window_backing_tags(
        model, parallel, stage_map=stage_map, kv_block_size=kv_block_size,
        draft_kind=draft_kind, speculative_k=speculative_k,
        kv_abis=kv_abis, idx_abis=idx_abis, rec_abis=rec_abis,
    )
    return tuple(kv_abis), tuple(idx_abis), tuple(rec_abis), tuple(disps), any_faquant, any_mxfp_gqa


def _build_prepared_weights(
    model: ModelSpec,
    *,
    parallel: ParallelSpec,
    port_abis: Sequence[tuple[tuple[str, int, str], TensorPrecisionABI]],
    has_faquant: bool,
    is_mxfp_gqa: bool,
) -> tuple[PreparedWeightSpec, ...]:
    """Emit runtime-only ``PreparedWeightSpec`` allocations structurally."""
    prepared: list[PreparedWeightSpec] = []
    plans = model.layer_plans

    kda_plans = [
        p for p in plans
        if str(p.attention).lower() == "kda" and int(p.gate_rank or 0) <= 0 and bool(p.packed_input_projection)
    ]
    if kda_plans:
        kda_layers = tuple(p.native_layer_id if p.native_layer_id >= 0 else p.index for p in kda_plans)
        lin_heads = int(kda_plans[0].linear_heads or model.num_heads)
        lin_dim = int(kda_plans[0].linear_head_dim or model.head_dim)
        local_proj = max(1, (lin_heads * lin_dim) // max(1, parallel.tp))
        prepared.append(PreparedWeightSpec(
            allocation_id="kda.prepared_f_matmul",
            source_tensors=("f_a_proj.weight", "f_b_proj.weight"),
            transform="matmul_f_b_f_a_bf16",
            scope="target",
            layer_indices=kda_layers,
            local_shape=(local_proj, int(model.hidden_size)),
            dtype="bf16",
            layout="ND",
            retains_source_tensors=True,
            lifetime="persistent",
        ))

    w4a8_layers = sorted({
        coord[1] for coord, abi in port_abis
        if coord[0] == "target" and abi.linear_method == "w4a8"
    })
    if w4a8_layers:
        n_experts_local = max(1, _ceil_div(int(model.num_experts or 1), max(1, parallel.ep)))
        moe_inter = int(model.moe_intermediate_size or model.intermediate_size)
        moe_inter_local = max(1, _ceil_div(moe_inter, max(1, int(parallel.etp or 1))))
        w4a8_out_channels = 2 * moe_inter_local + int(model.hidden_size)
        prepared.append(PreparedWeightSpec(
            allocation_id="w4a8.post_load_int64_scale_and_bias",
            source_tensors=("weight_scale", "weight_scale_second"),
            transform="w4a8_msd_pack_int64_scale_and_fp32_bias",
            scope="target",
            layer_indices=tuple(w4a8_layers),
            local_shape=(n_experts_local, w4a8_out_channels),
            dtype="int64",
            layout="ND",
            extra_metadata_bytes=n_experts_local * w4a8_out_channels * SCALE_BYTES_FP32,
            retains_source_tensors=True,
            lifetime="persistent",
        ))

    gqa_layers = tuple(
        p.native_layer_id if p.native_layer_id >= 0 else p.index
        for p in plans if str(p.attention).lower() not in {"kda", "gdn", "mla"}
    )
    if has_faquant and gqa_layers:
        l_kv = local_kv_heads(int(model.num_kv_heads or model.num_heads), parallel.tp)
        h_dim = int(model.head_dim)
        for name, xform in (
            ("faquant.fak_descale_float", "faquant_derive_fak_descale_float"),
            ("faquant.fak_descale_reciprocal", "faquant_derive_fak_descale_reciprocal"),
            ("faquant.quant_kscale", "faquant_derive_quant_kscale"),
        ):
            prepared.append(PreparedWeightSpec(
                allocation_id=name,
                source_tensors=("fa_k_scale", "fa_q_scale", "fa_v_scale"),
                transform=xform,
                scope="target",
                layer_indices=gqa_layers,
                local_shape=(l_kv, h_dim),
                dtype="fp32",
                layout="ND",
                retains_source_tensors=True,
                lifetime="persistent",
            ))

    if is_mxfp_gqa and gqa_layers:
        l_kv = local_kv_heads(int(model.num_kv_heads or model.num_heads), parallel.tp)
        h_dim = int(model.head_dim)
        prepared.append(PreparedWeightSpec(
            allocation_id="mxfp_c8.v_cache_scale_float_reciprocal",
            source_tensors=("v_cache_scale",),
            transform="reciprocal_fp32_from_v_cache_scale",
            scope="target",
            layer_indices=gqa_layers,
            local_shape=(l_kv, _ceil_div(h_dim, MXFP_GROUP_SIZE)),
            dtype="fp32",
            layout="ND",
            retains_source_tensors=True,
            lifetime="persistent",
        ))

    return tuple(prepared)


def _resolve_port_abis_and_packs(
    model: ModelSpec,
    checkpoint: CheckpointQuantMap,
    *,
    hw: str,
    model_type: str,
    native_float: str,
    use_nz: bool,
    target_proj: str,
    phase: str,
    role: str,
) -> tuple[
    tuple[tuple[tuple[str, int, str], TensorPrecisionABI], ...],
    tuple[tuple[str, tuple[str, ...], str], ...],
    bool,
]:
    """Resolve port ABIs & homogeneous packs purely structurally (zero global mutable cache)."""
    hf_cfg = dict(checkpoint.hf_quant_config)
    expert_ids = _expert_boundary_ids(checkpoint, int(model.num_experts or 0))
    port_abis: list[tuple[tuple[str, int, str], TensorPrecisionABI]] = []
    packed_groups: list[tuple[str, tuple[str, ...], str]] = []
    used_projection = False

    for top_role, top_paths in (
        ("embed_tokens", ("model.embed_tokens.weight",)),
        ("lm_head", ("lm_head.weight", "language_model.lm_head.weight")),
    ):
        tag, blk, did_proj, matched_rule = _resolve_role_tag(
            checkpoint, model_type=model_type, role_name=top_role, candidate_paths=top_paths,
            scope="target", layer_ids=(0,), is_routed_expert=False, expert_ids=expert_ids,
            target_quant_projection=target_proj, hf_cfg=hf_cfg,
        )
        used_projection = used_projection or did_proj
        port_abis.append((("target", 0, top_role), resolve_raw_tag_to_abi(
            tag, hw_family=hw, native_float_dtype=native_float, phase=phase, role=role,
            use_nz=use_nz, group_size=blk, rule=matched_rule,
        )))

    if any(int(p.residual_block_size or 0) > 0 for p in model.layer_plans):
        port_abis.append((
            ("target", 0, "output_attn_res_proj"),
            resolve_raw_tag_to_abi("FLOAT", hw_family=hw, native_float_dtype=native_float, phase=phase, role=role),
        ))

    for plan in model.layer_plans:
        native_layer = plan.native_layer_id if plan.native_layer_id >= 0 else plan.index
        layer_ids = (plan.index, native_layer) if native_layer != plan.index else (plan.index,)
        role_specs, candidate_packs = _layer_role_descriptors(plan, model=model)
        layer_abi_map: dict[str, TensorPrecisionABI] = {}
        for role_name, cands, is_expert in role_specs:
            tag, blk, did_proj, matched_rule = _resolve_role_tag(
                checkpoint, model_type=model_type, role_name=role_name, candidate_paths=cands,
                scope="target", layer_ids=layer_ids, is_routed_expert=is_expert, expert_ids=expert_ids,
                target_quant_projection=target_proj, hf_cfg=hf_cfg,
            )
            used_projection = used_projection or did_proj
            abi = resolve_raw_tag_to_abi(
                tag, hw_family=hw, native_float_dtype=native_float, phase=phase, role=role,
                is_moe=is_expert, use_nz=use_nz, k_dim=int(model.hidden_size),
                n_dim=int(model.moe_intermediate_size if is_expert else model.intermediate_size),
                group_size=blk, rule=matched_rule,
            )
            layer_abi_map[role_name] = abi
            port_abis.append((("target", native_layer, role_name), abi))

        packed_groups.extend(_apply_homogeneous_packs(
            scope="target", native_layer=native_layer, layer_abis=layer_abi_map,
            candidate_packs=candidate_packs,
        ))

    draft_ports, draft_proj = _resolve_draft_ports(
        model, checkpoint, model_type=model_type, hw=hw, native_float=native_float,
        phase=phase, role=role, use_nz=use_nz, expert_ids=expert_ids,
        target_quant_projection=target_proj, hf_cfg=hf_cfg,
    )
    port_abis.extend(draft_ports)
    used_projection = used_projection or draft_proj
    return tuple(port_abis), tuple(packed_groups), used_projection


def instantiate_runtime_quant_plan(
    model: ModelSpec,
    checkpoint: CheckpointQuantMap,
    hardware: Hardware,
    parallel: ParallelSpec,
    options: ResolvedStackOptions,
    *,
    role: str,
    phase: str,
) -> RuntimeQuantPlan:
    """Instantiate exact ``RuntimeQuantPlan`` purely from structural model/checkpoint specs."""
    hw = normalize_hw_family(hardware)
    model_type = resolve_model_type(model)
    native_float = _normalize_native_float(checkpoint, model)
    eff_opts = _effective_option_map(options)
    use_nz = bool(eff_opts.get("enable_weight_nz_layout", hw in A2_A3_INT_ONLY_FAMILIES))
    target_proj = str(eff_opts.get("target_quant_projection") or "").strip()

    port_abis, packed_groups, used_projection = _resolve_port_abis_and_packs(
        model, checkpoint, hw=hw, model_type=model_type,
        native_float=native_float, use_nz=use_nz, target_proj=target_proj,
        phase=phase, role=role,
    )

    spec_k = int(options.speculative.draft_len if options.speculative is not None else eff_opts.get("speculative_k", 0))
    recoverssm_active = bool(eff_opts.get("recoverssm_enabled", False))
    kv_abis, idx_abis, rec_abis, cache_disps, has_faquant, is_mxfp_gqa = build_cache_state_abis(
        model, checkpoint, hw=hw, parallel=parallel,
        effective_opts=eff_opts, role=role, phase=phase,
        speculative_k=spec_k, recoverssm_active=recoverssm_active,
    )
    prepared = _build_prepared_weights(
        model, parallel=parallel, port_abis=port_abis,
        has_faquant=has_faquant, is_mxfp_gqa=is_mxfp_gqa,
    )

    provenance: CheckpointOrigin = (
        "stack_default_quant_projection"
        if (checkpoint.source_format == "unquantized_native" and used_projection)
        else checkpoint.provenance.origin
    )
    dispositions = list(cache_disps)
    if provenance == "stack_default_quant_projection":
        dispositions.append((
            "blind_box_quant_projection",
            "applied",
            f"projected={target_proj}:quant_config_none_preserved",
        ))

    return RuntimeQuantPlan(
        stack_id=VLLM_ASCEND_STACK_ID,
        checkpoint_digest=checkpoint.digest(),
        provenance=provenance,
        port_abis=port_abis,
        packed_groups=packed_groups,
        prepared_weights=prepared,
        kv_cache_abi=kv_abis,
        index_cache_abi=idx_abis,
        recurrent_state_abi=rec_abis,
        dispositions=tuple(dispositions),
    )


__all__ = [
    "CONSTRUCTOR_UNQUANTIZED_ROLES",
    "DEFAULT_KV_BLOCK_SIZE",
    "MXFP_GROUP_SIZE",
    "QUANT_MODEL_PREFIX_MAPPINGS",
    "QUANT_MODEL_SUBSTR_MAPPINGS",
    "UPDATED_PACKED_MODULES_MAPPING",
    "build_cache_state_abis",
    "instantiate_runtime_quant_plan",
    "map_quant_prefix",
    "resolve_raw_tag_to_abi",
]
