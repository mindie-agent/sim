"""1:1 vLLM-Ascend quantization instantiation, shard packing, and CacheStateABI builder.

[Role] Mirror upstream ``vllm-ascend`` quantization instantiation verbatim:
  - ``modelslim_config.py`` prefix/substring mappings and ``packed_modules_mapping``,
  - constructor ``quant_config=None`` unquantized module enforcement,
  - strict Homogeneous Shard-Packing Law (shards merge iff ``vllm-ascend`` packs
    them AND every shard resolves to identical ``TensorPrecisionABI``),
  - runtime-only ``PreparedWeightSpec`` materializations (Kimi-K3.1 BF16
    ``F = F_b @ F_a``, W4A8 INT64 scale + FP32 ``scale_bias``, FAQuant derived
    scales, MXFP8 ``v_cache_scale_float_reciprocal``),
  - exact per-native-layer typed ``CacheStateABI`` for KV, Indexer, and
    recurrent/PLE/Engram state across Kimi-K3.1, Qwen3.8-Flash-Next,
    GLM-5.3-Flash, DeepSeek-V4.1, and Qwen3-32B (BF16 / C8 / FAQuant /
    ``QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL``).
[Invariants] Zero invented quantization heuristics. Unknown non-empty checkpoint
  tags raise ``ModelingError("unknown_checkpoint_quant_tag:...")``. A2/A3 never
  admit MXFP/FP8 weights or FP8 KV cache. Blind-box ``unquantized_native`` with
  production target projection quantizes ONLY modules receiving ``quant_config``
  and stamps ``provenance="stack_default_quant_projection"``. Zero imports from
  ``mindie_sim.export`` or ``mindie_sim.compose`` (AG010).
[Timeouts & Locks] Pure in-memory resolution over immutable descriptors; no I/O
  or locks.
[Side-Effect Policy] Returns frozen ``RuntimeQuantPlan`` records; zero input
  mutation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mindie_sim.catalog.speculative_state import speculative_state_plan
from mindie_sim.schema.checkpoint import (
    CheckpointOrigin,
    CheckpointQuantMap,
    ModelScope,
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
)

from .profiles import (
    A2_A3_INT_ONLY_FAMILIES,
    VLLM_ASCEND_STACK_ID,
    normalize_canonical_model,
    normalize_hw_family,
    resolve_model_type,
)

#: Verbatim upstream ``modelslim_config.py`` prefix mappings.
QUANT_MODEL_PREFIX_MAPPINGS: dict[str, tuple[tuple[str, str], ...]] = {
    "qwen3_vl": (("model.language_model.", "model."),),
    "qwen3_vl_moe": (("model.language_model.", "model."),),
    "qwen3_5": (("model.language_model.", "model."),),
    "qwen3_5_moe": (("model.language_model.", "model."),),
    "qwen3_5_text": (("model.language_model.", "model."),),
    "qwen3_5_moe_text": (("model.language_model.", "model."),),
    "qwen4_exp": (("model.language_model.", "model."),),
    "qwen4_exp_text": (("model.language_model.", "model."),),
    "kimi_k25": (("language_model.model.", "model."),),
    "kimi_k3": (("language_model.model.", "model."), ("model.language_model.", "model.")),
    "kimi_k31": (("language_model.model.", "model."), ("model.language_model.", "model.")),
    "glm5_next": (("model.language_model.", "model."),),
    "glm5_next_text": (("model.language_model.", "model."),),
}

#: Verbatim upstream ``modelslim_config.py`` substring mappings.
QUANT_MODEL_SUBSTR_MAPPINGS: dict[str, tuple[tuple[str, str], ...]] = {
    "qwen3_5": ((".mlp.shared_expert.", ".mlp.shared_experts."),),
    "qwen3_5_moe": ((".mlp.shared_expert.", ".mlp.shared_experts."),),
    "qwen4_exp": ((".mlp.shared_expert.", ".mlp.shared_experts."),),
    "qwen4_exp_text": ((".mlp.shared_expert.", ".mlp.shared_experts."),),
}

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
MLA_SCALE_GROUP_SIZE = 128
INDEX_SCALE_GROUP_SIZE = 128
DEFAULT_KV_BLOCK_SIZE = 128

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

DEFAULT_CONV_KERNEL_SIZE = 4
DEFAULT_MLA_LATENT_DIM = 512
DEFAULT_INDEX_DIM = 128
DEFAULT_ENGRAM_SLOTS = 65536
W4A8_SCALE_BIAS_DIM = 16
W8A16_WEIGHT_BITS = 8

_FLOAT_TAGS = frozenset({"", "NONE", "FLOAT", "BF16", "BFLOAT16", "FP16", "FLOAT16", "FP32", "FLOAT32"})
_MXFP8_TAGS = frozenset({"W8A8_MXFP8", "MXFP8", "MX_FP8", "W8A8_MXFP", "MXFP8_DYNAMIC"})
_MXFP4_W4A8_TAGS = frozenset({
    "W4A8_MXFP", "W4A8_MXFP4", "W4A8_MXFP4_DYNAMIC",
    "MXFP4", "MX_FP4", "FP4", "NVFP4", "QUARK_MXFP4",
})
_MXFP4_W4A4_TAGS = frozenset({"W4A4_MXFP4", "W4A4_MXFP"})
_FP8_TAGS = frozenset({"FP8", "FP8_DYNAMIC", "W8A8_FP8", "FP8_E4M3", "QUARK_FP8"})
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
    "MXFP8_C8",
    "C8_MXFP8",
    "C8_MXFP",
    "MXFP8",
})
_MLA_FP8_KV_TAGS = frozenset({"FP8", "FP8_E4M3", "FP8_DYNAMIC", "MXFP8", "MXFP8_DYNAMIC", "W8A8_MXFP8"})
_INT8_KV_TAGS = frozenset({"INT8", "C8", "KV_C8", "INT8_DYNAMIC", "INT8_SYM", "FAQUANT", "FAKQUANT"})
_FP8_INDEX_TAGS = frozenset({"fp8", "fp8_e4m3", "fp8_dynamic", "mxfp8", "mxfp8_dynamic"})
_INT8_INDEX_TAGS = frozenset({"int8", "c8", "int8_dynamic", "int8_sym"})
_FAQUANT_BUFFER_TOKENS = ("fa_q", "fa_k", "fa_v", "faquant", "fakquant")
_KIMI_K3_TYPES = frozenset({"kimi_k3", "kimi_k31"})
_GLM5_NEXT_TYPES = frozenset({"glm5_next", "glm5_next_text"})
_QWEN4_EXP_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text"})
_DEFAULT_DRAFT_KV_CAP = 0

_PROJECTION_PRESET_TAGS: dict[str, tuple[str, str]] = {
    "mxfp8": ("W8A8_MXFP8", "W8A8_MXFP8"),
    "w8a8_mxfp8": ("W8A8_MXFP8", "W8A8_MXFP8"),
    "w8a8_mxfp": ("W8A8_MXFP8", "W8A8_MXFP8"),
    "mxfp8_mxfp4": ("W8A8_MXFP8", "W4A8_MXFP"),
    "mxfp4_routed_only": ("W8A8_MXFP8", "W4A8_MXFP"),
    "mxfp4_all": ("W4A8_MXFP", "W4A8_MXFP"),
    "mxfp4": ("W8A8_MXFP8", "W4A8_MXFP"),
    "w4a8_mxfp": ("W8A8_MXFP8", "W4A8_MXFP"),
    "w4a4_mxfp4": ("W8A8_MXFP8", "W4A4_MXFP4"),
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


def map_quant_prefix(model_type: str, tensor_path: str) -> tuple[str, ...]:
    """Generate upstream ``modelslim_config.py`` prefix/substring lookup candidates."""
    mtype = str(model_type or "").strip().lower()
    base = str(tensor_path or "").strip()
    if not base:
        return ()
    candidates: list[str] = [base]
    for src, dst in QUANT_MODEL_PREFIX_MAPPINGS.get(mtype, ()):
        if base.startswith(src):
            candidates.append(dst + base[len(src):])
        elif base.startswith(dst):
            candidates.append(src + base[len(dst):])
    expanded: list[str] = list(candidates)
    for cand in candidates:
        for src_sub, dst_sub in QUANT_MODEL_SUBSTR_MAPPINGS.get(mtype, ()):
            if src_sub in cand:
                expanded.append(cand.replace(src_sub, dst_sub))
            if dst_sub in cand:
                expanded.append(cand.replace(dst_sub, src_sub))
    with_unweighted: list[str] = []
    for cand in expanded:
        with_unweighted.append(cand)
        if cand.endswith(".weight"):
            with_unweighted.append(cand[: -len(".weight")])
    return tuple(dict.fromkeys(with_unweighted))


def _normalize_native_float(checkpoint: CheckpointQuantMap, model: ModelSpec) -> str:
    raw = str(checkpoint.native_float_dtype or model.weight_dtype or model.activation_dtype or "bf16").strip().lower()
    if raw in {"bfloat16", "bf16"}:
        return "bf16"
    if raw in {"float16", "fp16"}:
        return "fp16"
    if raw in {"float32", "fp32"}:
        return "fp32"
    return "bf16"


def _effective_option_map(options: ResolvedStackOptions) -> dict[str, Any]:
    return {str(k): v for k, v in options.effective_mechanisms}


def _w4a8_nz_eligible(*, is_moe: bool, use_nz: bool, k_dim: int, n_dim: int) -> bool:
    if is_moe:
        return True
    if not use_nz:
        return False
    if k_dim > 0 and n_dim > 0:
        return (k_dim % NZ_K_ALIGN == 0) and (n_dim % NZ_N_ALIGN == 0)
    return True


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
        return TensorPrecisionABI(
            linear_method="fp8",
            input_dtype="fp8_e4m3",
            weight_dtype="fp8_e4m3",
            output_dtype=nat,
            compute_dtype="fp8_e4m3",
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


def resolve_raw_tag_to_abi(
    raw_tag: str,
    *,
    hw_family: str,
    native_float_dtype: str = "bf16",
    phase: str = "decode",
    role: str = "colocated",
    is_moe: bool = False,
    use_nz: bool = False,
    k_dim: int = 0,
    n_dim: int = 0,
    group_size: tuple[int, ...] = (),
) -> TensorPrecisionABI:
    """Resolve one raw checkpoint quantization tag into exact ``TensorPrecisionABI``."""
    tag = str(raw_tag or "").strip().upper()
    hw = normalize_hw_family(hw_family)
    nat = native_float_dtype or "bf16"

    if tag in _FLOAT_TAGS:
        dtype_out = "fp16" if tag in {"FP16", "FLOAT16"} else ("fp32" if tag in {"FP32", "FLOAT32"} else nat)
        return TensorPrecisionABI(
            linear_method="dense",
            input_dtype=dtype_out,
            weight_dtype=dtype_out,
            output_dtype=dtype_out,
            compute_dtype=dtype_out,
            accum_dtype="fp32",
            weight_layout="ND",
        )

    a5_abi = _resolve_a5_fp_tag_to_abi(tag, hw=hw, nat=nat, group_size=group_size)
    if a5_abi is not None:
        return a5_abi

    int_abi = _resolve_int_tag_to_abi(
        tag, nat=nat, phase=phase, role=role, is_moe=is_moe,
        use_nz=use_nz, k_dim=k_dim, n_dim=n_dim, group_size=group_size,
    )
    if int_abi is not None:
        return int_abi

    raise ModelingError(f"unknown_checkpoint_quant_tag:{raw_tag}")


def _guard_a5_only(hw: str, method: str, tag: str) -> None:
    if hw in A2_A3_INT_ONLY_FAMILIES:
        raise ModelingError(
            f"hw_isa_incompatible:{hw}:linear_method={method}:tag={tag}:"
            "A2/A3 supports INT8/INT4 quantization only"
        )


def _hf_quant_tag_for_role(
    hf_cfg: Mapping[str, Any], *, is_routed_expert: bool,
) -> tuple[str, tuple[int, ...]] | None:
    method = str(hf_cfg.get("quant_method") or "").strip().lower()
    scale_fmt = str(hf_cfg.get("scale_fmt") or "").strip().lower()
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
        return "FP8_DYNAMIC", blk or (FP8_DEFAULT_BLOCK_SIZE, FP8_DEFAULT_BLOCK_SIZE)
    if method in {"w8a8", "w8a8_dynamic"}:
        return "W8A8_DYNAMIC", blk
    if method in {"w4a8", "w4a8_dynamic"}:
        return ("W4A8_DYNAMIC" if is_routed_expert else "W8A8_DYNAMIC"), blk
    return None


def _lookup_modelslim_tag(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    expert_id: int | None = None,
) -> str | None:
    for path in candidate_paths:
        mapped = map_quant_prefix(model_type, path)
        for lid in layer_ids:
            for cand in mapped:
                tag = checkpoint.resolve_raw_tag(
                    cand, scope=scope, source_native_layer_id=lid, expert_id=expert_id,
                )
                if tag is not None:
                    return tag
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


def _lookup_exception_or_ignored(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    expert_id: int | None = None,
) -> str | None:
    for path in candidate_paths:
        mapped = map_quant_prefix(model_type, path)
        for lid in layer_ids:
            for cand in mapped:
                tag = checkpoint.resolve_exception_or_ignored(
                    cand, scope=scope, source_native_layer_id=lid, expert_id=expert_id,
                )
                if tag is not None:
                    return tag
    return None


def _lookup_non_wildcard_rule(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    expert_id: int | None = None,
) -> tuple[str, tuple[int, ...]] | None:
    specific_rules = tuple(r for r in checkpoint.rules if r.pattern != "*")
    if not specific_rules:
        return None
    for path in candidate_paths:
        mapped = map_quant_prefix(model_type, path)
        for lid in layer_ids:
            for cand in mapped:
                for rule in specific_rules:
                    if rule.matches(
                        cand, scope=scope, source_native_layer_id=lid, expert_id=expert_id,
                    ):
                        return rule.raw_tag, rule.group_size
    return None


def _lookup_hf_rule_tag(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    expert_id: int | None = None,
) -> tuple[str, tuple[int, ...]] | None:
    specific = _lookup_non_wildcard_rule(
        checkpoint,
        model_type=model_type,
        candidate_paths=candidate_paths,
        scope=scope,
        layer_ids=layer_ids,
        expert_id=expert_id,
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
            return rule.raw_tag, rule.group_size
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
    """Resolve ``mlp.experts.w13`` while enforcing gate/up homogeneity when both exist."""
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
        return "FLOAT"
    return gate_tag if gate_tag is not None else up_tag


def _resolve_role_tag(
    checkpoint: CheckpointQuantMap,
    *,
    model_type: str,
    role_name: str,
    candidate_paths: Sequence[str],
    scope: ModelScope,
    layer_ids: Sequence[int],
    is_routed_expert: bool,
    target_quant_projection: str,
    hf_cfg: Mapping[str, Any],
) -> tuple[str, tuple[int, ...], bool]:
    """Return ``(raw_tag, group_size, used_projection)`` for one role."""
    expert_id = 0 if is_routed_expert else None
    if checkpoint.source_format == "modelslim_description":
        if role_name == "mlp.experts.w13":
            explicit = _lookup_modelslim_w13_tag(
                checkpoint,
                model_type=model_type,
                candidate_paths=candidate_paths,
                scope=scope,
                layer_ids=layer_ids,
                expert_id=expert_id,
            )
        else:
            explicit = _lookup_modelslim_tag(
                checkpoint,
                model_type=model_type,
                candidate_paths=candidate_paths,
                scope=scope,
                layer_ids=layer_ids,
                expert_id=expert_id,
            )
        if explicit is not None:
            return explicit, (), False
        return "FLOAT", (), False

    exc_or_ignored = _lookup_exception_or_ignored(
        checkpoint,
        model_type=model_type,
        candidate_paths=candidate_paths,
        scope=scope,
        layer_ids=layer_ids,
        expert_id=expert_id,
    )
    if exc_or_ignored is not None:
        return exc_or_ignored, (), False

    if role_name in CONSTRUCTOR_UNQUANTIZED_ROLES:
        return "FLOAT", (), False

    if checkpoint.source_format == "hf_quantization_config":
        if hf_cfg:
            hf_hit = _hf_quant_tag_for_role(hf_cfg, is_routed_expert=is_routed_expert)
            if hf_hit is not None:
                return hf_hit[0], hf_hit[1], False
        rule_hit = _lookup_hf_rule_tag(
            checkpoint,
            model_type=model_type,
            candidate_paths=candidate_paths,
            scope=scope,
            layer_ids=layer_ids,
            expert_id=expert_id,
        )
        if rule_hit is not None:
            return rule_hit[0], rule_hit[1], False
        if hf_cfg and str(hf_cfg.get("quant_method") or "").strip().lower() == "ascend":
            return "W8A8_DYNAMIC", (), False
        return "FLOAT", (), False

    if checkpoint.source_format == "unquantized_native":
        if target_quant_projection:
            proj_tag = _project_blind_box_tag(
                target_quant_projection, role_name=role_name, is_routed_expert=is_routed_expert,
            )
            return proj_tag, (), (proj_tag != "FLOAT")
        return "FLOAT", (), False

    if checkpoint.source_format == "explicit_synthetic_preset" and checkpoint.model_quant_type:
        proj_tag = _project_blind_box_tag(
            checkpoint.model_quant_type, role_name=role_name, is_routed_expert=is_routed_expert,
        )
        return proj_tag, (), False

    return "FLOAT", (), False


def _layer_role_descriptors(
    plan: LayerPlan, *, model: ModelSpec, model_type: str,
) -> tuple[list[tuple[str, tuple[str, ...], bool]], list[tuple[str, tuple[str, ...]]]]:
    """Return ``(role_specs, candidate_pack_groups)`` for one target layer."""
    idx = int(plan.index)
    attn = str(plan.attention or model.attention).lower()
    roles: list[tuple[str, tuple[str, ...], bool]] = []
    packs: list[tuple[str, tuple[str, ...]]] = []

    if attn == "kda":
        prefix_a, prefix_b = f"model.layers.{idx}.self_attn", f"model.layers.{idx}.linear_attn"
        if model_type in _GLM5_NEXT_TYPES:
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
        if plan.mla_output_gate or model_type in _KIMI_K3_TYPES:
            roles.append(("mla.g_proj", (f"{prefix}.g_proj.weight",), False))
        packs.append(("fused_qkv_a_proj", ("mla.q_a_proj", "mla.kv_a_proj_with_mqa")))
        if plan.indexer:
            idx_p = f"{prefix}.indexer"
            idx_short = f"{attn_short}.indexer"
            roles.append(("indexer.wk", (f"{idx_p}.wk.weight", f"{idx_short}.wk.weight"), False))
            roles.append(("indexer.weights_proj", (f"{idx_p}.weights_proj.weight", f"{idx_short}.weights_proj.weight"), False))
            if model_type in _KIMI_K3_TYPES:
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

    if plan.residual_block_size > 0 or model_type in _KIMI_K3_TYPES:
        roles.append(("self_attention_res_proj", (f"model.layers.{idx}.self_attention_res_proj.weight",), False))
        roles.append(("mlp_res_proj", (f"model.layers.{idx}.mlp_res_proj.weight",), False))
    if plan.hyper_connections or model_type in _QWEN4_EXP_TYPES:
        roles.append(("hyper_connection.pre_proj", (f"model.layers.{idx}.hyper_connection.pre_proj.weight",), False))
        roles.append(("hyper_connection.post_proj", (f"model.layers.{idx}.hyper_connection.post_proj.weight",), False))
    if plan.engram and model_type in _QWEN4_EXP_TYPES:
        roles.append(("ple.embed_proj", (f"model.layers.{idx}.ple.embed_proj.weight",), False))
        roles.append(("ple.key_proj", (f"model.layers.{idx}.ple.key_proj.weight",), False))
        roles.append(("ple.value_proj", (f"model.layers.{idx}.ple.value_proj.weight",), False))
        roles.append(("ple.conv1d", (f"model.layers.{idx}.ple.conv1d.weight",), False))

    mlp_p = f"model.layers.{idx}.mlp"
    bsm_p = f"model.layers.{idx}.block_sparse_moe"
    moe_p = f"model.layers.{idx}.moe"
    ffn_p = f"layers.{idx}.ffn"
    if str(plan.ffn).lower() == "moe":
        if plan.expert_latent_size > 0:
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
                f"{mlp_p}.experts.w13.weight",
                f"{bsm_p}.experts.0.w1.weight",
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
        if plan.num_shared_experts > 0 or model.num_shared_experts > 0:
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
    model_type: str,
) -> list[tuple[str, tuple[str, ...], str]]:
    """Enforce Homogeneous Shard-Packing Law: pack iff every shard has identical ABI."""
    packed: list[tuple[str, tuple[str, ...], str]] = []
    for pack_name, shards in candidate_packs:
        if not all(role in layer_abis for role in shards):
            continue
        sigs = {layer_abis[role].abi_signature for role in shards}
        if len(sigs) == 1:
            packed.append((f"{scope}.L{native_layer}.{pack_name}", tuple(shards), next(iter(sigs))))
        elif pack_name == "fused_qkvbfg_a_proj" and model_type in _GLM5_NEXT_TYPES:
            # Heterogeneous GLM-5.3-Flash KDA falls back to homogeneous QKV + BFG sub-packs.
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
    target_quant_projection: str,
    hf_cfg: Mapping[str, Any],
) -> tuple[list[tuple[tuple[str, int, str], TensorPrecisionABI]], bool]:
    """Resolve optional draft (MTP / DSpark) ports preserving Qwen4Exp MTP BF16."""
    aux = model.auxiliary
    if aux is None or (aux.mtp is None and aux.dspark is None):
        return [], False
    draft_spec = aux.mtp if aux.mtp is not None else aux.dspark
    draft_layers = max(1, int(getattr(draft_spec, "layers", 1) or 1))
    start_id = int(model.body_layers or model.layers)
    has_moe_draft = bool(model.num_experts > 0 and aux.mtp is not None)
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
                        f"mtp.layers.{offset_id}.mlp.experts.0.gate_proj.weight",
                        f"mtp.layers.{local_idx}.mlp.experts.w13.weight",
                        f"mtp.{local_idx}.ffn.experts.0.w1.weight",
                        f"model.layers.{offset_id}.mlp.experts.0.gate_proj.weight",
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
                exp_id = 0 if is_expert else None
                explicit = _lookup_modelslim_tag(
                    checkpoint, model_type=model_type, candidate_paths=cands,
                    scope="draft", layer_ids=layer_ids, expert_id=exp_id,
                )
                if explicit is None:
                    # Appended native MTP layers (e.g. DeepSeek-R1 L61) live in target scope.
                    explicit = _lookup_modelslim_tag(
                        checkpoint, model_type=model_type, candidate_paths=cands,
                        scope="target", layer_ids=(offset_id,), expert_id=exp_id,
                    )
                tag, blk, did_proj = (explicit if explicit is not None else "FLOAT"), (), False
            else:
                tag, blk, did_proj = _resolve_role_tag(
                    checkpoint,
                    model_type=model_type,
                    role_name=role_name,
                    candidate_paths=cands,
                    scope="draft",
                    layer_ids=layer_ids,
                    is_routed_expert=is_expert,
                    target_quant_projection="" if model_type in _QWEN4_EXP_TYPES else target_quant_projection,
                    hf_cfg=hf_cfg if model_type not in _QWEN4_EXP_TYPES else {},
                )
            used_proj = used_proj or did_proj
            abi = resolve_raw_tag_to_abi(
                tag, hw_family=hw, native_float_dtype=native_float,
                phase=phase, role=role, is_moe=is_expert, use_nz=use_nz, group_size=blk,
            )
            ports.append((("draft", local_idx, role_name), abi))
    return ports, used_proj


def _detect_kv_mode(
    checkpoint: CheckpointQuantMap,
    *,
    hw: str,
    model_type: str,
    effective_opts: Mapping[str, Any],
    is_mla: bool,
    kv_tag_override: str | None = None,
) -> tuple[str, str, bool, list[tuple[str, str, str]]]:
    """Resolve effective KV cache mode, scale dtype, FAQuant active flag, and dispositions."""
    disps: list[tuple[str, str, str]] = []
    raw_source = (
        kv_tag_override
        if kv_tag_override is not None
        else (checkpoint.cache_rule.kv_tag or effective_opts.get("kv_cache_dtype") or "")
    )
    raw_kv = str(raw_source or effective_opts.get("kv_cache_dtype") or "").strip().upper()
    aux_low = " ".join(checkpoint.auxiliary_buffers).lower()
    has_faquant = (
        ("FAQUANT" in raw_kv)
        or ("FAKQUANT" in raw_kv)
        or any(tok in aux_low for tok in _FAQUANT_BUFFER_TOKENS)
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
        default_scale = "fp16" if model_type in _GLM5_NEXT_TYPES else "fp32"
        scale_dtype = str(checkpoint.cache_rule.kv_scale_dtype or default_scale).strip().lower()
        return "fp8_e4m3", scale_dtype, has_faquant, disps

    if raw_kv in _INT8_KV_TAGS:
        scale_dtype = str(checkpoint.cache_rule.kv_scale_dtype or "fp32").strip().lower()
        return "int8", scale_dtype, has_faquant, disps

    return "bf16", "", has_faquant, disps


def _resolve_index_dtype(
    raw_tag: str | None,
    *,
    hw: str,
    fallback_kv_mode: str,
) -> str:
    norm = str(
        raw_tag
        or ("fp8_e4m3" if fallback_kv_mode == "fp8_e4m3" else ("int8" if fallback_kv_mode == "int8" else "bf16"))
    ).strip().lower()
    if norm in _FP8_INDEX_TAGS:
        if hw in A2_A3_INT_ONLY_FAMILIES:
            raise ModelingError(f"hw_isa_incompatible:{hw}:indexer_kv_dtype={norm}")
        return "fp8_e4m3"
    if norm in _INT8_INDEX_TAGS:
        return "int8"
    return "bf16"


def _build_mxfp_gqa_cache_abi(
    *,
    native_layer: int,
    local_heads: int,
    head_dim: int,
    kv_block_size: int,
    placement: CachePlacementRule,
    checkpoint_digest: str,
    phase: str,
    role: str,
    page_pitch_override: int = 0,
) -> CacheStateABI:
    """Build exact ``QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL`` PA_NZ ``CacheStateABI``."""
    k_bytes = local_heads * kv_block_size * head_dim
    ks_bytes = local_heads * kv_block_size * head_dim // MXFP_GROUP_SIZE
    v_bytes = local_heads * kv_block_size * head_dim
    vs_bytes = local_heads * kv_block_size * head_dim // MXFP_GROUP_SIZE
    payload_bytes = k_bytes + ks_bytes + v_bytes + vs_bytes
    pitch = max(payload_bytes, int(page_pitch_override or payload_bytes))
    pad_bytes = pitch - payload_bytes

    k_plane = CachePlaneABI(
        plane_id="k_payload",
        plane_kind="payload",
        logical_axes=("head", "token", "dim"),
        logical_shape=(local_heads, kv_block_size, head_dim),
        physical_shape=(local_heads, head_dim // MXFP_KV_NZ_DIM_FRAG, kv_block_size, MXFP_KV_NZ_DIM_FRAG),
        dtype="fp8_e4m3",
        layout="PA_NZ",
        offset_in_page=0,
        alignment=MXFP_KV_NZ_DIM_FRAG,
        page_pitch=pitch,
        allocation_frequency="page",
        write_trigger="append",
        bytes_per_unit=k_bytes,
    )
    k_scale_plane = CachePlaneABI(
        plane_id="k_scale",
        plane_kind="scale",
        logical_axes=("head", "token", "group"),
        logical_shape=(local_heads, kv_block_size, head_dim // MXFP_GROUP_SIZE),
        physical_shape=(
            local_heads,
            max(1, kv_block_size // MXFP_K_SCALE_NZ_TOKEN_FRAG),
            max(1, head_dim // MXFP_KV_SCALE_GROUP_SIZE),
            MXFP_K_SCALE_NZ_TOKEN_FRAG,
            MXFP_KV_SCALE_VALUES_PER_GROUP,
        ),
        dtype="e8m0",
        layout="PA_NZ",
        group_axes=(2,),
        group_extents=(MXFP_GROUP_SIZE,),
        scale_dtype="e8m0",
        offset_in_page=k_bytes,
        alignment=MXFP_K_SCALE_NZ_TOKEN_FRAG,
        page_pitch=pitch,
        allocation_frequency="token",
        write_trigger="append",
        bytes_per_unit=ks_bytes,
    )
    v_plane = CachePlaneABI(
        plane_id="v_payload",
        plane_kind="payload",
        logical_axes=("head", "token", "dim"),
        logical_shape=(local_heads, kv_block_size, head_dim),
        physical_shape=(local_heads, head_dim // MXFP_KV_NZ_DIM_FRAG, kv_block_size, MXFP_KV_NZ_DIM_FRAG),
        dtype="fp8_e4m3",
        layout="PA_NZ",
        offset_in_page=k_bytes + ks_bytes,
        alignment=MXFP_KV_NZ_DIM_FRAG,
        page_pitch=pitch,
        allocation_frequency="page",
        write_trigger="append",
        bytes_per_unit=v_bytes,
    )
    v_scale_plane = CachePlaneABI(
        plane_id="v_scale",
        plane_kind="scale",
        logical_axes=("head", "dim", "block_group"),
        logical_shape=(local_heads, head_dim, max(1, kv_block_size // MXFP_GROUP_SIZE)),
        physical_shape=(
            local_heads,
            max(1, head_dim // MXFP_V_SCALE_NZ_DIM_FRAG),
            max(1, kv_block_size // MXFP_KV_SCALE_GROUP_SIZE),
            MXFP_V_SCALE_NZ_DIM_FRAG,
            MXFP_KV_SCALE_VALUES_PER_GROUP,
        ),
        dtype="e8m0",
        layout="PA_NZ",
        group_axes=(1,),
        group_extents=(MXFP_GROUP_SIZE,),
        scale_dtype="e8m0",
        offset_in_page=k_bytes + ks_bytes + v_bytes,
        alignment=MXFP_V_SCALE_NZ_DIM_FRAG,
        page_pitch=pitch,
        allocation_frequency="page",
        write_trigger="page_reassign",
        bytes_per_unit=vs_bytes,
    )
    pad_plane = CachePlaneABI(
        plane_id="page_padding",
        plane_kind="padding",
        logical_axes=("byte",),
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
    return CacheStateABI(
        scope="target",
        native_layer=native_layer,
        component_role="kv_combined",
        producer_key=f"target.L{native_layer}.kv_combined",
        planes=(k_plane, k_scale_plane, v_plane, v_scale_plane, pad_plane),
        phase=phase,
        deployment_role=role,
        checkpoint_digest=checkpoint_digest,
        recipe_abi_id="qk_mxfp8_dynamic_v_mxfp8_per_channel",
        kv_block_size=kv_block_size,
        kernel_block_size=kv_block_size,
        placement=placement,
    )


def _build_mla_cache_abi(
    plan: LayerPlan,
    *,
    native_layer: int,
    kv_mode: str,
    kv_scale_dtype: str,
    kv_block_size: int,
    placement: CachePlacementRule,
    checkpoint_digest: str,
    phase: str,
    role: str,
    component_role: str = "mla_latent",
    consumer_keys: tuple[str, ...] = (),
) -> CacheStateABI:
    latent_dim = int(plan.kv_lora_rank or DEFAULT_MLA_LATENT_DIM)
    rope_dim = int(plan.qk_rope_head_dim or 0)
    planes: list[CachePlaneABI] = []
    if kv_mode in {"fp8_e4m3", "int8"}:
        planes.append(CachePlaneABI(
            plane_id="mla_latent",
            plane_kind="payload",
            logical_axes=("latent",),
            logical_shape=(latent_dim,),
            physical_shape=(latent_dim,),
            dtype=kv_mode,
            layout="ND",
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=latent_dim,
        ))
        if kv_scale_dtype in {"e8m0", "fp8_e8m0"}:
            scale_group_ext = MXFP_GROUP_SIZE
            scale_groups = max(1, latent_dim // MXFP_GROUP_SIZE)
            s_dtype = "e8m0"
            s_bytes = scale_groups * SCALE_BYTES_E8M0
        else:
            scale_group_ext = MLA_SCALE_GROUP_SIZE
            scale_groups = max(1, latent_dim // MLA_SCALE_GROUP_SIZE)
            s_dtype = "fp16" if kv_scale_dtype in {"fp16", "bf16", "float16", "bfloat16"} else "fp32"
            s_bytes = scale_groups * (SCALE_BYTES_FP16 if s_dtype == "fp16" else SCALE_BYTES_FP32)
        planes.append(CachePlaneABI(
            plane_id="mla_latent_scale",
            plane_kind="scale",
            logical_axes=("scale_group",),
            logical_shape=(scale_groups,),
            physical_shape=(scale_groups,),
            dtype=s_dtype,
            layout="ND",
            group_axes=(0,),
            group_extents=(scale_group_ext,),
            scale_dtype=s_dtype,
            offset_in_page=latent_dim,
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=s_bytes,
        ))
        offset_next = latent_dim + s_bytes
    else:
        lat_bytes = latent_dim * int(dtype_bytes("bf16"))
        planes.append(CachePlaneABI(
            plane_id="mla_latent",
            plane_kind="payload",
            logical_axes=("latent",),
            logical_shape=(latent_dim,),
            physical_shape=(latent_dim,),
            dtype="bf16",
            layout="ND",
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=lat_bytes,
        ))
        offset_next = lat_bytes

    if rope_dim > 0:
        rope_bytes = rope_dim * int(dtype_bytes("bf16"))
        planes.append(CachePlaneABI(
            plane_id="mla_rotary",
            plane_kind="suffix",
            logical_axes=("rope",),
            logical_shape=(rope_dim,),
            physical_shape=(rope_dim,),
            dtype="bf16",
            layout="ND",
            suffix_shape=(rope_dim,),
            suffix_dtype="bf16",
            offset_in_page=offset_next,
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=rope_bytes,
        ))

    return CacheStateABI(
        scope="target",
        native_layer=native_layer,
        component_role=component_role,  # type: ignore[arg-type]
        producer_key=f"target.L{native_layer}.{component_role}",
        consumer_keys=consumer_keys,
        planes=tuple(planes),
        phase=phase,
        deployment_role=role,
        checkpoint_digest=checkpoint_digest,
        recipe_abi_id=f"mla_{kv_mode}_{kv_scale_dtype or 'bf16'}",
        kv_block_size=kv_block_size,
        kernel_block_size=kv_block_size,
        window=int(plan.window) if plan.window > 0 and component_role == "window_kv" else None,
        compression_ratio=max(1, int(plan.compress_ratio)) if component_role == "compressed_kv" else 1,
        placement=placement,
    )


def _build_index_cache_abi(
    plan: LayerPlan,
    *,
    native_layer: int,
    index_dtype: str,
    index_scale_dtype: str,
    kv_block_size: int,
    placement: CachePlacementRule,
    checkpoint_digest: str,
    phase: str,
    role: str,
    consumer_keys: tuple[str, ...] = (),
) -> CacheStateABI:
    """Build exact DSA/KPool Index-K CacheStateABI (zero phantom per-token weights row)."""
    idx_dim = int(plan.index_dim or DEFAULT_INDEX_DIM)
    pool_ratio = max(1, int(plan.index_kpool or 1))
    tail_slots = max(0, pool_ratio - 1) if pool_ratio > 1 else 0
    planes: list[CachePlaneABI] = []
    if index_dtype in {"fp8_e4m3", "int8"}:
        planes.append(CachePlaneABI(
            plane_id="index_k",
            plane_kind="payload",
            logical_axes=("index_dim",),
            logical_shape=(idx_dim,),
            physical_shape=(idx_dim,),
            dtype=index_dtype,
            layout="ND",
            allocation_frequency="token_group" if pool_ratio > 1 else "token",
            write_trigger="append",
            bytes_per_unit=idx_dim,
        ))
        s_dtype = "fp16" if index_scale_dtype in {"fp16", "bf16"} else "fp32"
        s_bytes = SCALE_BYTES_FP16 if s_dtype == "fp16" else SCALE_BYTES_FP32
        planes.append(CachePlaneABI(
            plane_id="index_k_scale",
            plane_kind="scale",
            logical_axes=("scale_group",),
            logical_shape=(1,),
            physical_shape=(1,),
            dtype=s_dtype,
            layout="ND",
            group_axes=(0,),
            group_extents=(min(idx_dim, INDEX_SCALE_GROUP_SIZE),),
            scale_dtype=s_dtype,
            offset_in_page=idx_dim,
            allocation_frequency="token_group" if pool_ratio > 1 else "token",
            write_trigger="append",
            bytes_per_unit=s_bytes,
        ))
    else:
        planes.append(CachePlaneABI(
            plane_id="index_k",
            plane_kind="payload",
            logical_axes=("index_dim",),
            logical_shape=(idx_dim,),
            physical_shape=(idx_dim,),
            dtype="bf16",
            layout="ND",
            allocation_frequency="token_group" if pool_ratio > 1 else "token",
            write_trigger="append",
            bytes_per_unit=idx_dim * int(dtype_bytes("bf16")),
        ))

    if tail_slots > 0:
        tail_bytes = tail_slots * idx_dim * int(dtype_bytes("bf16"))
        planes.append(CachePlaneABI(
            plane_id="index_tail",
            plane_kind="suffix",
            logical_axes=("tail_slot", "index_dim"),
            logical_shape=(tail_slots, idx_dim),
            physical_shape=(tail_slots, idx_dim),
            dtype="bf16",
            layout="ND",
            allocation_frequency="request_slot",
            write_trigger="append",
            bytes_per_unit=tail_bytes,
        ))

    return CacheStateABI(
        scope="target",
        native_layer=native_layer,
        component_role="index_k",
        producer_key=f"target.L{native_layer}.index_k",
        consumer_keys=consumer_keys,
        planes=tuple(planes),
        phase=phase,
        deployment_role=role,
        checkpoint_digest=checkpoint_digest,
        recipe_abi_id=f"index_k_{index_dtype}_pool{pool_ratio}",
        kv_block_size=kv_block_size,
        kernel_block_size=kv_block_size,
        pooling_ratio=pool_ratio,
        tail_slots=tail_slots,
        placement=placement,
    )


def _build_recurrent_cache_abi(
    model: ModelSpec,
    plan: LayerPlan,
    *,
    native_layer: int,
    parallel: ParallelSpec,
    checkpoint_digest: str,
    phase: str,
    role: str,
    speculative_k: int,
    recoverssm_active: bool,
) -> CacheStateABI:
    """Build exact KDA/GDN conv+recurrent+replay CacheStateABI via ``speculative_state_plan``."""
    attn = str(plan.attention or plan.architecture).lower()
    shard_div = max(1, parallel.tp * parallel.pcp) if (attn == "kda" and parallel.pcp > 1) else max(1, parallel.tp)
    shard_axis = "tp*pcp" if (attn == "kda" and parallel.pcp > 1) else "tp"
    v_heads = max(1, int(plan.linear_value_heads or plan.linear_heads or 1))
    local_heads = max(1, v_heads // shard_div)
    h_dim = max(1, int(plan.linear_head_dim or DEFAULT_INDEX_DIM))
    rec_dtype = str(plan.recurrent_state_dtype or "fp32").lower()

    conv_k = max(2, int(plan.conv_kernel or DEFAULT_CONV_KERNEL_SIZE))
    history_slots = conv_k - 1
    rollback_slots = max(0, int(speculative_k))
    conv_w = int(plan.conv_width or (plan.conv_state_elements // max(1, history_slots)) or 0)
    local_conv_w = max(1, conv_w // shard_div) if conv_w > 0 else max(1, local_heads * h_dim)
    conv_dtype = str(plan.conv_state_dtype or "bf16").lower()

    spec_plan = speculative_state_plan(
        model, plan, parallel,
        query_tokens=max(1, 1 + rollback_slots),
        recover_ssm=recoverssm_active,
        capacity_padded=False,
    )
    rec_bytes = int(round(spec_plan.recurrent_bytes)) or (
        local_heads * h_dim * h_dim * int(dtype_bytes(rec_dtype))
    )
    conv_bytes = int(round(spec_plan.conv_bytes)) or (
        local_conv_w * (history_slots + rollback_slots) * int(dtype_bytes(conv_dtype))
    )
    conv_total_slots = history_slots + rollback_slots

    planes: list[CachePlaneABI] = [
        CachePlaneABI(
            plane_id="recurrent_state",
            plane_kind="payload",
            logical_axes=("head", "k_dim", "v_dim"),
            logical_shape=(local_heads, h_dim, h_dim),
            physical_shape=(local_heads, h_dim, h_dim),
            dtype=rec_dtype,
            layout="ND",
            allocation_frequency="request_slot",
            write_trigger="accepted_commit" if recoverssm_active else "append",
            bytes_per_unit=rec_bytes,
        ),
        CachePlaneABI(
            plane_id="conv_history",
            plane_kind="payload",
            logical_axes=("channel", "history_slot"),
            logical_shape=(local_conv_w, conv_total_slots),
            physical_shape=(local_conv_w, conv_total_slots),
            dtype=conv_dtype,
            layout="ND",
            offset_in_page=rec_bytes,
            allocation_frequency="request_slot",
            write_trigger="append",
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
            dtype="fp32",
            layout="ND",
            offset_in_page=rec_bytes + conv_bytes,
            allocation_frequency="request_slot",
            write_trigger="verify",
            bytes_per_unit=replay_bytes,
        ))
    return CacheStateABI(
        scope="target",
        native_layer=native_layer,
        component_role="recurrent_state",
        producer_key=f"target.L{native_layer}.recurrent_state",
        planes=tuple(planes),
        phase=phase,
        deployment_role=role,
        checkpoint_digest=checkpoint_digest,
        recipe_abi_id=f"{attn}_recurrent_state",
        history_slots=history_slots,
        rollback_slots=rollback_slots,
        checkpoint_slots=1 if recoverssm_active else 0,
        record_shape=(local_heads, h_dim, h_dim) if recoverssm_active else (),
        record_dtype=rec_dtype if recoverssm_active else "",
        placement=CachePlacementRule(
            token_ownership_axis="none",
            replication_factor=1,
            state_shard_axis=shard_axis,
        ),
    )


def _build_engram_or_ple_abi(
    model: ModelSpec,
    *,
    native_layer: int,
    model_type: str,
    engram_offload: bool,
    checkpoint_digest: str,
    phase: str,
    role: str,
) -> CacheStateABI:
    if model_type in _QWEN4_EXP_TYPES:
        ple_dim = int(model.hidden_size)
        ple_hist = DEFAULT_CONV_KERNEL_SIZE - 1
        return CacheStateABI(
            scope="target",
            native_layer=native_layer,
            component_role="ple_conv_state",
            producer_key=f"target.L{native_layer}.ple_conv_state",
            planes=(CachePlaneABI(
                plane_id="ple_conv_state",
                plane_kind="payload",
                logical_axes=("ple_dim", "history_slot"),
                logical_shape=(ple_dim, ple_hist),
                physical_shape=(ple_dim, ple_hist),
                dtype="bf16",
                layout="ND",
                allocation_frequency="request_slot",
                write_trigger="append",
                bytes_per_unit=ple_dim * ple_hist * int(dtype_bytes("bf16")),
            ),),
            phase=phase,
            deployment_role=role,
            checkpoint_digest=checkpoint_digest,
            recipe_abi_id="qwen4_exp_ple_conv_state",
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
        row_b = float(getattr(engram_spec, "row_bytes", 0.0) or (table_dim * int(dtype_bytes("bf16"))))
        storage_mode = str(getattr(engram_spec, "storage_mode", "device_bf16") or "device_bf16").lower()
        table_dtype = "fp8_e4m3" if "fp8" in storage_mode else ("int8" if "int8" in storage_mode else "bf16")
        total_bytes = int(round(float(table_slots) * row_b))
    else:
        table_slots = int(getattr(engram_spec, "vocab_size", DEFAULT_ENGRAM_SLOTS) or DEFAULT_ENGRAM_SLOTS)
        table_dim = int(getattr(engram_spec, "embed_dim", model.hidden_size) or model.hidden_size)
        table_dtype = "bf16"
        total_bytes = table_slots * table_dim * int(dtype_bytes("bf16"))
    return CacheStateABI(
        scope="target",
        native_layer=native_layer,
        component_role="engram_table",
        producer_key=f"target.L{native_layer}.engram_table",
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


def build_cache_state_abis(
    model: ModelSpec,
    checkpoint: CheckpointQuantMap,
    *,
    hw: str,
    model_type: str,
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
    """Build exact per-native-layer typed KV, Indexer, and recurrent/PLE/Engram ABIs."""
    plans = model.layer_plans
    has_mla = any(str(p.attention).lower() == "mla" for p in plans)
    kv_mode, kv_scale_dtype, has_faquant, disps = _detect_kv_mode(
        checkpoint, hw=hw, model_type=model_type, effective_opts=effective_opts, is_mla=has_mla,
    )
    kv_block_size = int(effective_opts.get("kv_block_size") or DEFAULT_KV_BLOCK_SIZE)
    ckpt_digest = checkpoint.digest()
    engram_offload = bool(effective_opts.get("engram_cpu_offload", False))

    default_idx_dtype = _resolve_index_dtype(
        checkpoint.cache_rule.index_tag or effective_opts.get("indexer_kv_dtype"),
        hw=hw, fallback_kv_mode=kv_mode,
    )
    index_scale_dtype = str(checkpoint.cache_rule.index_scale_dtype or "fp32").strip().lower()

    long_kv_consumers: dict[int, list[str]] = {}
    index_consumers: dict[int, list[str]] = {}
    for p in plans:
        n_id = p.native_layer_id if p.native_layer_id >= 0 else p.index
        if p.kv_owner_layer_id >= 0 and p.kv_owner_layer_id != n_id:
            long_kv_consumers.setdefault(p.kv_owner_layer_id, []).append(f"target.L{n_id}.mla_consumer")
        if p.index_k_owner_layer_id >= 0 and p.index_k_owner_layer_id != n_id:
            index_consumers.setdefault(p.index_k_owner_layer_id, []).append(f"target.L{n_id}.index_consumer")

    kv_abis: list[CacheStateABI] = []
    idx_abis: list[CacheStateABI] = []
    rec_abis: list[CacheStateABI] = []
    c8_producer_recorded = False
    any_faquant = has_faquant
    any_mxfp_gqa = (kv_mode == "qk_mxfp8_dynamic_v_mxfp8_per_channel")
    token_axis = "dcp_head" if parallel.cp > 1 else ("cp_striped" if parallel.pcp > 1 else "replicated")
    mla_rep = max(1, parallel.tp // max(1, parallel.cp))
    has_layer_kv = bool(checkpoint.cache_rule.layer_scoped_kv)
    has_layer_idx = bool(checkpoint.cache_rule.layer_scoped_index)

    for p in plans:
        n_id = p.native_layer_id if p.native_layer_id >= 0 else p.index
        attn = str(p.attention or model.attention).lower()
        layer_kv_mode, layer_kv_scale, layer_faquant = kv_mode, kv_scale_dtype, has_faquant
        if has_layer_kv and attn not in {"kda", "gdn"}:
            l_tag = checkpoint.cache_rule.kv_tag_for_layer(n_id)
            layer_kv_mode, layer_kv_scale, layer_faquant, l_disps = _detect_kv_mode(
                checkpoint, hw=hw, model_type=model_type, effective_opts=effective_opts,
                is_mla=(attn == "mla"), kv_tag_override=l_tag,
            )
            for item in l_disps:
                if item not in disps:
                    disps.append(item)
            any_faquant = any_faquant or layer_faquant
            any_mxfp_gqa = any_mxfp_gqa or (layer_kv_mode == "qk_mxfp8_dynamic_v_mxfp8_per_channel")
        if attn in {"kda", "gdn"}:
            rec_abis.append(_build_recurrent_cache_abi(
                model, p, native_layer=n_id, parallel=parallel, checkpoint_digest=ckpt_digest,
                phase=phase, role=role, speculative_k=speculative_k, recoverssm_active=recoverssm_active,
            ))
        elif attn == "mla":
            mla_place = CachePlacementRule(token_ownership_axis=token_axis, replication_factor=mla_rep)
            role_name = "window_kv" if p.window > 0 else "mla_latent"
            kv_abis.append(_build_mla_cache_abi(
                p, native_layer=n_id, kv_mode=layer_kv_mode, kv_scale_dtype=layer_kv_scale,
                kv_block_size=kv_block_size, placement=mla_place, checkpoint_digest=ckpt_digest,
                phase=phase, role=role, component_role=role_name,
            ))
            if p.owns_long_kv:
                kv_abis.append(_build_mla_cache_abi(
                    p, native_layer=n_id, kv_mode=layer_kv_mode, kv_scale_dtype=layer_kv_scale,
                    kv_block_size=kv_block_size, placement=mla_place, checkpoint_digest=ckpt_digest,
                    phase=phase, role=role, component_role="compressed_kv",
                    consumer_keys=tuple(long_kv_consumers.get(n_id, ())),
                ))
        else:
            n_kv = int(p.num_kv_heads or model.num_kv_heads or model.num_heads)
            h_dim = int(p.head_dim or model.head_dim)
            l_kv = local_kv_heads(n_kv, parallel.tp)
            rep = max(1, parallel.tp // n_kv) if n_kv < parallel.tp else 1
            gqa_place = CachePlacementRule(
                head_range=(0, l_kv), token_ownership_axis=token_axis, replication_factor=rep,
            )
            if layer_kv_mode == "qk_mxfp8_dynamic_v_mxfp8_per_channel":
                kv_abis.append(_build_mxfp_gqa_cache_abi(
                    native_layer=n_id, local_heads=l_kv, head_dim=h_dim, kv_block_size=kv_block_size,
                    placement=gqa_place, checkpoint_digest=ckpt_digest, phase=phase, role=role,
                ))
            else:
                eff_gqa_mode = layer_kv_mode
                if layer_kv_mode == "int8" and str(role).lower() == "pd_prefill":
                    eff_gqa_mode = "bf16"
                    if not c8_producer_recorded:
                        disps.append(("kv_cache_c8_producer", "downgraded", "kv_c8_producer_role_keeps_native_bf16"))
                        c8_producer_recorded = True
                kv_abis.append(_build_standard_gqa_cache_abi(
                    native_layer=n_id, local_heads=l_kv, head_dim=h_dim, kv_mode=eff_gqa_mode,
                    kv_block_size=kv_block_size, placement=gqa_place, checkpoint_digest=ckpt_digest,
                    phase=phase, role=role, has_faquant=layer_faquant,
                ))

        if p.indexer and (p.owns_index_k or p.index_k_owner_layer_id < 0 or p.index_k_owner_layer_id == n_id):
            layer_idx_dtype = (
                _resolve_index_dtype(
                    checkpoint.cache_rule.index_tag_for_layer(n_id) or effective_opts.get("indexer_kv_dtype"),
                    hw=hw, fallback_kv_mode=layer_kv_mode,
                )
                if has_layer_idx else default_idx_dtype
            )
            idx_place = CachePlacementRule(token_ownership_axis=token_axis, replication_factor=mla_rep)
            idx_abis.append(_build_index_cache_abi(
                p, native_layer=n_id, index_dtype=layer_idx_dtype, index_scale_dtype=index_scale_dtype,
                kv_block_size=kv_block_size, placement=idx_place, checkpoint_digest=ckpt_digest,
                phase=phase, role=role, consumer_keys=tuple(index_consumers.get(n_id, ())),
            ))
        if p.engram:
            rec_abis.append(_build_engram_or_ple_abi(
                model, native_layer=n_id, model_type=model_type, engram_offload=engram_offload,
                checkpoint_digest=ckpt_digest, phase=phase, role=role,
            ))

    return tuple(kv_abis), tuple(idx_abis), tuple(rec_abis), tuple(disps), any_faquant, any_mxfp_gqa


def _build_standard_gqa_cache_abi(
    *,
    native_layer: int,
    local_heads: int,
    head_dim: int,
    kv_mode: str,
    kv_block_size: int,
    placement: CachePlacementRule,
    checkpoint_digest: str,
    phase: str,
    role: str,
    has_faquant: bool,
) -> CacheStateABI:
    elem_dtype = kv_mode if kv_mode in {"int8", "fp8_e4m3"} else "bf16"
    elem_bytes = int(dtype_bytes(elem_dtype))
    half_bytes = local_heads * head_dim * elem_bytes
    planes: list[CachePlaneABI] = [
        CachePlaneABI(
            plane_id="kv_k",
            plane_kind="payload",
            logical_axes=("head", "dim"),
            logical_shape=(local_heads, head_dim),
            physical_shape=(local_heads, head_dim),
            dtype=elem_dtype,
            layout="ND",
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=half_bytes,
        ),
        CachePlaneABI(
            plane_id="kv_v",
            plane_kind="payload",
            logical_axes=("head", "dim"),
            logical_shape=(local_heads, head_dim),
            physical_shape=(local_heads, head_dim),
            dtype=elem_dtype,
            layout="ND",
            offset_in_page=half_bytes,
            allocation_frequency="token",
            write_trigger="append",
            bytes_per_unit=half_bytes,
        ),
    ]
    if elem_dtype == "int8":
        planes.extend([
            CachePlaneABI(
                plane_id="kv_k_scale",
                plane_kind="scale",
                logical_axes=("head", "dim"),
                logical_shape=(local_heads, head_dim),
                physical_shape=(local_heads, head_dim),
                dtype="fp32",
                layout="ND",
                scale_dtype="fp32",
                allocation_frequency="layer",
                write_trigger="load",
                bytes_per_unit=local_heads * head_dim * SCALE_BYTES_FP32,
            ),
            CachePlaneABI(
                plane_id="kv_k_offset",
                plane_kind="offset",
                logical_axes=("head", "dim"),
                logical_shape=(local_heads, head_dim),
                physical_shape=(local_heads, head_dim),
                dtype="int8",
                layout="ND",
                offset_dtype="int8",
                allocation_frequency="layer",
                write_trigger="load",
                bytes_per_unit=local_heads * head_dim,
            ),
            CachePlaneABI(
                plane_id="kv_v_scale",
                plane_kind="scale",
                logical_axes=("head", "dim"),
                logical_shape=(local_heads, head_dim),
                physical_shape=(local_heads, head_dim),
                dtype="fp32",
                layout="ND",
                scale_dtype="fp32",
                allocation_frequency="layer",
                write_trigger="load",
                bytes_per_unit=local_heads * head_dim * SCALE_BYTES_FP32,
            ),
            CachePlaneABI(
                plane_id="kv_v_offset",
                plane_kind="offset",
                logical_axes=("head", "dim"),
                logical_shape=(local_heads, head_dim),
                physical_shape=(local_heads, head_dim),
                dtype="int8",
                layout="ND",
                offset_dtype="int8",
                allocation_frequency="layer",
                write_trigger="load",
                bytes_per_unit=local_heads * head_dim,
            ),
        ])
    return CacheStateABI(
        scope="target",
        native_layer=native_layer,
        component_role="kv_combined",
        producer_key=f"target.L{native_layer}.kv_combined",
        planes=tuple(planes),
        phase=phase,
        deployment_role=role,
        checkpoint_digest=checkpoint_digest,
        recipe_abi_id="gqa_faquant" if has_faquant else f"gqa_{elem_dtype}",
        kv_block_size=kv_block_size,
        kernel_block_size=kv_block_size,
        placement=placement,
    )


def _build_prepared_weights(
    model: ModelSpec,
    *,
    model_type: str,
    parallel: ParallelSpec,
    port_abis: Sequence[tuple[tuple[str, int, str], TensorPrecisionABI]],
    has_faquant: bool,
    is_mxfp_gqa: bool,
) -> tuple[PreparedWeightSpec, ...]:
    """Emit runtime-only ``PreparedWeightSpec`` allocations (0 checkpoint bytes, real HBM)."""
    prepared: list[PreparedWeightSpec] = []
    plans = model.layer_plans

    if model_type in _KIMI_K3_TYPES:
        kda_plans = [p for p in plans if str(p.attention).lower() == "kda"]
        if kda_plans:
            kda_layers = tuple(p.native_layer_id if p.native_layer_id >= 0 else p.index for p in kda_plans)
            lin_heads = int(kda_plans[0].linear_heads or model.num_heads)
            lin_dim = int(kda_plans[0].linear_head_dim or model.head_dim)
            local_proj = max(1, (lin_heads * lin_dim) // max(1, parallel.tp))
            prepared.append(PreparedWeightSpec(
                allocation_id="kimi_k3.kda.prepared_f_matmul",
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
        n_experts_local = max(1, int(model.num_experts or 1) // max(1, parallel.ep))
        n_dim = int(model.moe_intermediate_size or model.intermediate_size)
        prepared.append(PreparedWeightSpec(
            allocation_id="w4a8.post_load_int64_scale_and_bias",
            source_tensors=("weight_scale", "weight_scale_second"),
            transform="w4a8_msd_pack_int64_scale_and_fp32_bias",
            scope="target",
            layer_indices=tuple(w4a8_layers),
            local_shape=(n_experts_local, n_dim),
            dtype="int64",
            layout="ND",
            extra_metadata_bytes=n_experts_local * W4A8_SCALE_BIAS_DIM * SCALE_BYTES_FP32,
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
            local_shape=(l_kv, max(1, h_dim // MXFP_GROUP_SIZE)),
            dtype="fp32",
            layout="ND",
            retains_source_tensors=True,
            lifetime="persistent",
        ))

    return tuple(prepared)


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
    """Instantiate exact ``RuntimeQuantPlan`` mirroring ``vllm-ascend`` load/pack logic."""
    hw = normalize_hw_family(hardware)
    model_type = resolve_model_type(model)
    native_float = _normalize_native_float(checkpoint, model)
    eff_opts = _effective_option_map(options)
    use_nz = bool(eff_opts.get("enable_weight_nz_layout", hw in A2_A3_INT_ONLY_FAMILIES))
    target_proj = str(eff_opts.get("target_quant_projection") or "").strip()
    hf_cfg = dict(checkpoint.hf_quant_config)

    port_abis: list[tuple[tuple[str, int, str], TensorPrecisionABI]] = []
    packed_groups: list[tuple[str, tuple[str, ...], str]] = []
    used_projection = False

    # Model-level embedding & LM head (quant_config=None unless explicitly quantized in checkpoint).
    for top_role, top_paths in (
        ("embed_tokens", ("model.embed_tokens.weight",)),
        ("lm_head", ("lm_head.weight", "language_model.lm_head.weight")),
    ):
        tag, blk, did_proj = _resolve_role_tag(
            checkpoint, model_type=model_type, role_name=top_role, candidate_paths=top_paths,
            scope="target", layer_ids=(0,), is_routed_expert=False,
            target_quant_projection=target_proj, hf_cfg=hf_cfg,
        )
        used_projection = used_projection or did_proj
        port_abis.append((("target", 0, top_role), resolve_raw_tag_to_abi(
            tag, hw_family=hw, native_float_dtype=native_float, phase=phase, role=role, use_nz=use_nz, group_size=blk,
        )))

    if model_type in _KIMI_K3_TYPES:
        port_abis.append((
            ("target", 0, "output_attn_res_proj"),
            resolve_raw_tag_to_abi("FLOAT", hw_family=hw, native_float_dtype=native_float, phase=phase, role=role),
        ))

    for plan in model.layer_plans:
        native_layer = plan.native_layer_id if plan.native_layer_id >= 0 else plan.index
        layer_ids = (plan.index, native_layer) if native_layer != plan.index else (plan.index,)
        role_specs, candidate_packs = _layer_role_descriptors(plan, model=model, model_type=model_type)
        layer_abi_map: dict[str, TensorPrecisionABI] = {}
        for role_name, cands, is_expert in role_specs:
            tag, blk, did_proj = _resolve_role_tag(
                checkpoint, model_type=model_type, role_name=role_name, candidate_paths=cands,
                scope="target", layer_ids=layer_ids, is_routed_expert=is_expert,
                target_quant_projection=target_proj, hf_cfg=hf_cfg,
            )
            used_projection = used_projection or did_proj
            abi = resolve_raw_tag_to_abi(
                tag, hw_family=hw, native_float_dtype=native_float, phase=phase, role=role,
                is_moe=is_expert, use_nz=use_nz, k_dim=int(model.hidden_size),
                n_dim=int(model.moe_intermediate_size if is_expert else model.intermediate_size),
                group_size=blk,
            )
            layer_abi_map[role_name] = abi
            port_abis.append((("target", native_layer, role_name), abi))

        packed_groups.extend(_apply_homogeneous_packs(
            scope="target", native_layer=native_layer, layer_abis=layer_abi_map,
            candidate_packs=candidate_packs, model_type=model_type,
        ))

    draft_ports, draft_proj = _resolve_draft_ports(
        model, checkpoint, model_type=model_type, hw=hw, native_float=native_float,
        phase=phase, role=role, use_nz=use_nz, target_quant_projection=target_proj, hf_cfg=hf_cfg,
    )
    port_abis.extend(draft_ports)
    used_projection = used_projection or draft_proj

    spec_k = int(options.speculative.draft_len if options.speculative is not None else eff_opts.get("speculative_k", 0))
    recoverssm_active = bool(eff_opts.get("recoverssm_enabled", False))
    kv_abis, idx_abis, rec_abis, cache_disps, has_faquant, is_mxfp_gqa = build_cache_state_abis(
        model, checkpoint, hw=hw, model_type=model_type, parallel=parallel,
        effective_opts=eff_opts, role=role, phase=phase,
        speculative_k=spec_k, recoverssm_active=recoverssm_active,
    )
    prepared = _build_prepared_weights(
        model, model_type=model_type, parallel=parallel, port_abis=port_abis,
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
            f"projected={target_proj}:canonical={normalize_canonical_model(model)}:quant_config_none_preserved",
        ))

    return RuntimeQuantPlan(
        checkpoint_digest=checkpoint.digest(),
        stack_id=VLLM_ASCEND_STACK_ID,
        port_abis=tuple(port_abis),
        packed_groups=tuple(packed_groups),
        kv_cache_abi=kv_abis,
        index_cache_abi=idx_abis,
        recurrent_state_abi=rec_abis,
        prepared_weights=prepared,
        dispositions=tuple(dispositions),
        provenance=provenance,
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
