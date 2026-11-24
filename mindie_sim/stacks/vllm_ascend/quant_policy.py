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
  A2/A3 never admit MXFP/FP8 weights or FP8 KV cache. Zero imports from
  ``mindie_sim.export`` or ``mindie_sim.compose`` (AG010).
[Timeouts & Locks] Pure in-memory resolution over immutable descriptors; no I/O
  or locks.
[Side-Effect Policy] Returns frozen ``RuntimeQuantPlan`` records; zero input
  mutation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mindie_sim.schema.checkpoint import CheckpointQuantMap
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.operators import TensorPrecisionABI
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.stack import (
    CacheStateABI,
    ResolvedStackOptions,
    RuntimeQuantPlan,
)

from .profiles import (
    A2_A3_INT_ONLY_FAMILIES,
    normalize_hw_family,
)

# Purged model_type prefix/substring tables: Commit #2 must move checkpoint
# path normalization to stacks/vllm_ascend/checkpoint_registry.py or structural
# CheckpointQuantMap predicates with zero AG014 model-name tables here.
QUANT_MODEL_PREFIX_MAPPINGS: dict[str, tuple[tuple[str, str], ...]] = {}
QUANT_MODEL_SUBSTR_MAPPINGS: dict[str, tuple[tuple[str, str], ...]] = {}

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

W4A8_SCALE_BIAS_DIM = 16
W8A16_WEIGHT_BITS = 8

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


def map_quant_prefix(model_type: str, tensor_path: str) -> tuple[str, ...]:
    """Unplugged socket: model_type prefix/substr mappings purged per AG014."""
    raise NotImplementedError(
        "LEGACY_PURGED: replace model_type prefix/substr tables with "
        "stacks/vllm_ascend/checkpoint_registry.py or structural CheckpointQuantMap path resolution"
    )


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
) -> TensorPrecisionABI:
    """Unplugged socket: silent BF16 fallback and BF16->fp16 coercion purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: wire resolve_raw_tag_to_abi with strict native_float_dtype "
        "validation (zero silent bf16 fallback) and verbatim BF16/FP16/FP32 preservation"
    )


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
    """Unplugged socket: model_type branches, silent BF16 fallbacks, and floor truncation purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: wire structural CacheStateABI builder (zero model_type branches, "
        "fail closed on unknown/unsupported KV/index tags including A3 DS-V4.1 INT8 KV, "
        "exact scale-group accounting, zero BF16->FP16 scale coercion, zero DEFAULT_ENGRAM_SLOTS)"
    )


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
    """Unplugged socket: model_type branches, expert-0 sampling, and FLOAT downgrades purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: dispatch quant instantiation purely on structural LayerPlan fields "
        "(kda_projection_layout, mla_output_gate, indexer_fused_wk_weights, residual_block_size, "
        "hyper_connections, engram) + stacks/vllm_ascend/checkpoint_registry.py, validate all "
        "experts homogeneously without expert-0 sampling or FLOAT downgrade"
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
