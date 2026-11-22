"""Curated Ascend NPU quantized checkpoints, msmodelslim parser, and metadata reader.

[Role]
  Provide ground-truth ``(canonical_model, hw_family) -> NpuCheckpointSpec``
  defaults mined from the live 165-repo Eco-Tech scan and vLLM Ascend tutorials
  with strict A2/A3 (INT8/INT4/C8) vs A5 (MXFP8/MXFP4/FP8) ISA separation and
  QuaRot-first priority, parse ``msmodelslim`` ``quant_model_description.json``
  maps into exact ``QuantSpec`` and per-tensor classifications, and read local
  or remote checkpoint JSON headers without downloading weights.

[Invariants]
  * Zero weight downloads and zero NPU device contexts.
  * Strict hardware-ISA separation: A2 (910B, 64G) and A3 (910C dual-Die,
    64G/die) have NO hardware FP8/MXFP — never select or recommend MXFP8/MXFP4/
    FP8 weights or FP8 KV cache on A2 or A3 (INT8 W8A8_DYNAMIC, INT4 routed
    W4A8_DYNAMIC, and C8 INT8 KV cache only).
  * QuaRot-first ranking: whenever multiple Eco-Tech checkpoints exist for the
    same ``(model, hw_family, bitwidth)``, ``*-QuaRot*`` ranks first.
  * Remote metadata reads enforce ``<= 256 KiB`` and bounded ``urllib.request``
    timeouts (``except TimeoutError: raise`` strictly before ``except OSError:``).

[Timeouts & Locks]
  Bounded HTTP fetches (default ``5.0s``) and atomic ``os.replace()`` cache
  writes under ``${XDG_CACHE_HOME:-~/.cache}/mindie-sim/checkpoints/``.

[Side-Effect Policy]
  Pure in-memory lookups except optional cache persistence when resolving a
  remote checkpoint URL or repo ID with ``allow_network=True``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from mindie_sim.catalog.fleet import (
    ARCHITECTURE_PRESET_ALIASES,
    ARCHITECTURE_PRESET_CATALOG,
    CORE_CATALOG_ENTRIES,
    CORE_MODEL_ALIASES,
)
from mindie_sim.catalog.types import QuantSpec, merge_source
from mindie_sim.schema.common import ModelingError

DEFAULT_HTTP_TIMEOUT_S = 5.0
MAX_REMOTE_METADATA_BYTES = 256 * 1024
MAX_LOCAL_METADATA_BYTES = 50 * 1024 * 1024
MXFP_GROUP_SIZE = 32
MXFP_SCALE_PACK = 2
MXFP_SCALE_BYTES = 1.0
FP8_BLOCK_SIZE = 128
FP32_SCALE_BYTES = 4.0
BF16_SCALE_BYTES = 2.0
CACHE_HASH_HEX_LEN = 16
CACHE_STEM_MAX_CHARS = 64
MIN_REPO_PATH_SEGMENTS = 2

MODELERS_PAGE_BASE = "https://modelers.cn/models"
MODELERS_RAW_BASE = "https://modelers.cn/coderepo/web/v1/file"
MODELSCOPE_PAGE_BASE = "https://www.modelscope.cn/models"
MODELSCOPE_RAW_BASE = "https://modelscope.cn/models"
HF_PAGE_BASE = "https://huggingface.co"

METADATA_FILENAMES: tuple[str, ...] = (
    "config.json", "quant_model_description.json", "quantize_config.json",
)
_METADATA_KEYS_TO_SKIP = frozenset({
    "model_quant_type", "version", "group_size", "kv_cache_type",
    "fa_quant_type", "metadata", "quant_method", "format",
})
_NUMERIC_SEGMENT_RE = re.compile(r"(?<=\.)\d+(?=\.)")
_SAFE_CACHE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_KNOWN_DESC_TAGS = frozenset({
    "W8A8_MXFP8", "W4A8_MXFP", "W4A8_MXFP4", "W4A4_MXFP4", "W4A4_MXFP",
    "W8A8_DYNAMIC", "W8A8", "W4A8_DYNAMIC", "W4A8", "W4A4_DYNAMIC", "W4A4",
    "FP8_DYNAMIC", "FP8", "W8A8_FP8", "MXFP8", "MXFP4",
    "FLOAT", "BF16", "FP16", "NONE", "C8",
})
_SUPPORTED_HW_FAMILIES = frozenset({"a2", "a3", "a5"})
A2_A3_INT_ONLY_FAMILIES = frozenset({"a2", "a3"})
A2_A3_ALLOWED_WEIGHT_PRESETS = frozenset({
    "w8a8_dynamic", "w8a8", "w4a8_dynamic", "w4a8", "w4a4_dynamic", "w4a4",
    "int8", "int4", "bf16", "fp16", "float",
})
A2_A3_ALLOWED_KV_DTYPES = frozenset({"bf16", "fp16", "int8", "c8", "auto", ""})
A5_ONLY_QUANT_TOKENS = ("mxfp", "fp8", "fp4")

#: Complete 165-repository snapshot from the live Eco-Tech scan on modelers.cn.
ECO_TECH_REPOS: tuple[str, ...] = tuple(f"Eco-Tech/{name}" for name in """
DeepSeek-Math-V2-w8a8-mtp-QuaRot DeepSeek-R1-0528-w4a8-mtp-QuaRot DeepSeek-R1-0528-w4a8c8-mtp-QuaRot DeepSeek-R1-0528-w8a8-mtp-QuaRot DeepSeek-R1-0528-w8a8c8-mtp-QuaRot
DeepSeek-V3-0324-w4a8-mtp-QuaRot DeepSeek-V3-0324-w4a8c8-mtp-QuaRot DeepSeek-V3-0324-w8a8-mtp-QuaRot DeepSeek-V3-0324-w8a8c8-mtp-QuaRot DeepSeek-V3-w4a8-mtp-QuaRot
DeepSeek-V3-w4a8c8-mtp-QuaRot DeepSeek-V3-w8a8-mtp-QuaRot DeepSeek-V3-w8a8c8-mtp-QuaRot DeepSeek-V3.1-Terminus-w4a8-mtp-QuaRot DeepSeek-V3.1-Terminus-w4a8c8-mtp-QuaRot
DeepSeek-V3.1-Terminus-w8a8-mtp-QuaRot DeepSeek-V3.1-Terminus-w8a8c8-mtp-QuaRot DeepSeek-V3.1-w4a8-mtp-QuaRot DeepSeek-V3.1-w4a8c8-mtp-QuaRot DeepSeek-V3.1-w8a8-mtp-QuaRot
DeepSeek-V3.1-w8a8c8-mtp-QuaRot DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot DeepSeek-V3.2-Speciale-w8a8-mtp-QuaRot DeepSeek-V3.2-w8a8-mtp-QuaRot DeepSeek-V4-Flash-0731-w8a8
DeepSeek-V4-Flash-DSpark-w8a8 DeepSeek-V4-Flash-Vision-Exp-w8a8-QuaRot DeepSeek-V4-Flash-w8a8-mtp DeepSeek-V4-Pro-0813-w4a8 DeepSeek-V4-Pro-w4a8-mtp
DeepSeek-V4.1-Flash-1017-w4a8 DeepSeek-V4.1-Flash-1017-w8a8 DeepSeek-V4.1-Flash-w4a8 DeepSeek-V4.1-Flash-w8a8 GLM-4.1V-9B-Thinking-w8a8s-310
GLM-4.7-W8A8 GLM-4.7-W8A8-floatmtp GLM-5-w4a8 GLM-5-w4a8-mtp-QuaRot GLM-5-w8a8
GLM-5.1-W8A8-xLLM GLM-5.1-w4a4c8-mxfp4 GLM-5.1-w4a8 GLM-5.1-w8a8 GLM-5.1-w8a8c8
GLM-5.1-w8a8c8-MTP GLM-5.1-w8a8c8-mxfp8 GLM-5.2-W8A8C8-mxfp8 GLM-5.2-w4a4c8-mxfp4 GLM-5.2-w4a8
GLM-5.2-w4a8c8 GLM-5.2-w8a8 GLM-5.2-w8a8c8 GLM-5.3-Flash-w8a8 GLM-5.3-Flash-w8a8-mxfp8
GLM-5.3-W8A8C8-mxfp8 GLM-5.3-w8a8 GLM-5.3-w8a8c8 Hy3-w8a8 Hy4-preview-w8a8
InternVL3_5-241B-A28B-w8a8 InternVL3_5-38B-w8a8 KIMI-k2-Instruct-0905-W4A8-QuaRot KIMI-k2-Instruct-0905-W8A8-QuaRot KIMI-k2-Thinking-W8A8-QuaRot
Kimi-K2.5-W8A8-xLLM Kimi-K2.5-w4a8 Kimi-K2.6-w4a8 Kimi-K2.6-w8a8-xllm Kimi-K2.7-Code-w4a8
Kimi-K3-w4a8 Kimi-K3.1-Pro-w4a8 Kimi-K3.1-Pro-w4a8-mxfp Ling-2.5-1T-w4a8 Ling-2.6-1T-w4a8
Ling-3.0-flash-w4a8 Ling-3.1-flash-w4a8 MiniMax-M2.5-w8a8-QuaRot MiniMax-M2.7-eagle-model-short MiniMax-M2.7-w8a8-QuaRot
MiniMax-M3-w8a8 MiniMax-M3-w8a8-0626 MiniMax-M3.1-Flash-w8a8 MiniMax-M3.1-Flash-w8a8-mxfp8 Minimax2.7-BF16-xLLM
Qwen2.5-72B-Instruct-w8a8c8-pdmix Qwen2.5-VL-72B-Instruct-w8a8 Qwen2.5-VL-7B-Instruct-w8a8 Qwen3-14B-w8a8s-310 Qwen3-14B-w8a8sc-310-vllm
Qwen3-235B-A22B-Instruct-2507-w8a8-QuaRot Qwen3-235B-A22B-Instruct-2507-w8a8c8-QuaRot Qwen3-235B-A22B-w8a8-QuaRot Qwen3-30B-A3B-Thinking-2507-w8a8 Qwen3-30B-A3B-w4a4-LAOS
Qwen3-30B-A3B-w8a8 Qwen3-30B-A3B-w8a8-QuaRot-310 Qwen3-32B-w16a16s-310 Qwen3-32B-w16a16sc-310-mindie Qwen3-32B-w4a4-LAOS
Qwen3-32B-w8a8-nopdmix Qwen3-32B-w8a8-pdmix Qwen3-32B-w8a8c8 Qwen3-32B-w8a8s-310 Qwen3-32B-w8a8sc-310-vllm
Qwen3-8B-w8a8s-310 Qwen3-8B-w8a8sc-310-vllm Qwen3-Coder-30B-A3B-Instruct-w8a8 Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot Qwen3-Next-80B-A3B-Instruct-w8a8-mtp
Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot Qwen3-VL-30B-A3B-Instruct-w8a8-QuaRot Qwen3-VL-30B-A3B-Instruct-w8a8-QuaRot-310 Qwen3-VL-32B-Instruct-w8a8-QuaRot Qwen3-VL-32B-Instruct-w8a8s-310
Qwen3-VL-32B-Instruct-w8a8sc-310-mindie Qwen3-VL-32B-Instruct-w8a8sc-310-vllm Qwen3-VL-4B-Instruct-w8a8s-310 Qwen3-VL-4B-Instruct-w8a8sc-310-mindie Qwen3-VL-4B-Instruct-w8a8sc-310-vllm
Qwen3-VL-8B-Instruct-w8a8-QuaRot Qwen3-VL-8B-Instruct-w8a8s-310 Qwen3-VL-8B-Instruct-w8a8sc-310-mindie Qwen3-VL-8B-Instruct-w8a8sc-310-vllm Qwen3.5-122B-A10B-w8a8-mtp
Qwen3.5-27B-w8a8-mtp Qwen3.5-2B-W8A8-310P Qwen3.5-35B-A3B-w8a8-mtp Qwen3.5-397B-A17B-w4a4-mxfp4 Qwen3.5-397B-A17B-w4a8-mtp
Qwen3.5-397B-A17B-w4a8c8-mxfp Qwen3.5-397B-A17B-w8a8-mtp Qwen3.5-397B-A17B-w8a8-mxfp8 Qwen3.5-4B-W8A8-310P Qwen3.5-4B-w8a8
Qwen3.5-9B-W8A8-310P Qwen3.6-27B-W8A8-310P Qwen3.6-27B-w8a8 Qwen3.6-27B-w8a8-mxfp8 Qwen3.6-27B-w8a8c8-mxfp8
Qwen3.6-35B-A3B-w4a8-mxfp Qwen3.6-35B-A3B-w8a8 Qwen3.6-35B-A3B-w8a8-mxfp8 Qwen3.6-35B-A3B-w8a8c8-mxfp8 Qwen3.8-2.4T-A95B-w4a8
Qwen3.8-2.4T-A95B-w8a8 Qwen3.8-27B-w8a8 Qwen3.8-27B-w8a8-310p Qwen3.8-27B-w8a8-mxfp8 Qwen3.8-27B-w8a8-mxfp8-mtp
Qwen3.8-27B-w8a8c8-mxfp8 Qwen3.8-Flash-Next-w4a8 Qwen3.8-Flash-Next-w4a8-mxfp Qwen3.8-Flash-Next-w8a8-mtp Qwen3.8-Flash-Next-w8a8-mxfp8-mtp
Ring-2.5-1T-w4a8 Step-3.5-Flash-w8a8-mtp Step-3.7-Flash-w8a8-mtp Step-3.7-Flash-w8a8-mxfp8 Step-5-Preview-w8a8-mxfp8
Wan2.2-T2V-A14B-Diffusers-MXFP-W4A4F4-DAOS Wan2.2-T2V-A14B-Diffusers-w8a8 Wan2.2-T2V-A14B-MXFP-W4A4F4-DAOS gemma-4-26B-A4B-it-w4a4-mxfp4 gemma-4-31B-it-w4a4-mxfp4
""".split())
_ECO_TECH_LOWER_INDEX: dict[str, str] = {repo.lower(): repo for repo in ECO_TECH_REPOS}

# Exact scanned quant_model_description.json pattern tables.
_KL = "language_model.model.layers.*."
_KIMI_K31_FLOAT_PATTERNS: tuple[str, ...] = (
    "language_model.lm_head.weight",
    "language_model.model.embed_tokens.weight",
    *(f"{_KL}{s}.weight" for s in (
        "block_sparse_moe.gate", "block_sparse_moe.routed_expert_norm",
        "input_layernorm", "mlp_res_norm", "mlp_res_proj",
        "post_attention_layernorm", "self_attention_res_norm",
        "self_attention_res_proj", "self_attn.b_proj", "self_attn.f_a_proj",
        "self_attn.f_b_proj", "self_attn.g_proj", "self_attn.indexer.k_norm",
        "self_attn.indexer.weights_proj", "self_attn.indexer.wk",
        "self_attn.k_conv1d", "self_attn.kv_a_layernorm", "self_attn.kv_b_proj",
        "self_attn.o_norm", "self_attn.o_proj", "self_attn.q_a_layernorm",
        "self_attn.q_conv1d", "self_attn.v_conv1d",
    )),
    "language_model.model.norm.weight",
    "language_model.model.output_attn_res_norm.weight",
    "language_model.model.output_attn_res_proj.weight",
    "mm_projector.post_norm.weight",
    "mm_projector.proj.*.weight",
    *(f"vision_tower.{s}.weight" for s in (
        "encoder.blocks.*.mlp.fc0", "encoder.blocks.*.mlp.fc1",
        "encoder.blocks.*.norm0", "encoder.blocks.*.norm1",
        "encoder.blocks.*.wo", "encoder.blocks.*.wqkv",
        "encoder.final_layernorm", "patch_embed.pos_emb", "patch_embed.proj",
    )),
)
_KIMI_K31_LINEAR_PATTERNS: tuple[str, ...] = tuple(f"{_KL}{s}.weight" for s in (
    "block_sparse_moe.routed_expert_down_proj", "block_sparse_moe.routed_expert_up_proj",
    "block_sparse_moe.shared_experts.down_proj", "block_sparse_moe.shared_experts.gate_proj",
    "block_sparse_moe.shared_experts.up_proj",
    "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
    "self_attn.indexer.wq_b", "self_attn.k_proj", "self_attn.kv_a_proj_with_mqa",
    "self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.q_proj", "self_attn.v_proj",
))
_KIMI_K31_EXPERT_PATTERNS: tuple[str, ...] = tuple(
    f"{_KL}block_sparse_moe.experts.*.{w}.weight" for w in ("w1", "w2", "w3")
)

_QL = "model.language_model.layers.*."
_ML = "mtp.layers.*."
_QWEN38_FLASH_FLOAT_PATTERNS: tuple[str, ...] = (
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.hyper_connection_mixer.hc_norm.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    *(f"{_QL}{s}.weight" for s in (
        "attn_hyper_connection.block_inject_weight", "attn_hyper_connection.hc_norm",
        "attn_hyper_connection.input_mix_weight_down", "attn_hyper_connection.input_mix_weight_up",
        "linear_attn.conv1d", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.norm",
        "linear_attn.out_proj", "mlp.gate", "mlp.shared_expert.down_proj",
        "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert_gate",
        "mlp_hyper_connection.block_inject_weight", "mlp_hyper_connection.hc_norm",
        "mlp_hyper_connection.input_mix_weight_down", "mlp_hyper_connection.input_mix_weight_up",
        "ple.conv1d", "ple.key_proj", "ple.norm_conv", "ple.norm_key", "ple.norm_query",
        "ple.ple_embedding.ngram_embedding.shard_*", "ple.value_proj",
        "self_attn.indexer.k_layernorm", "self_attn.indexer.q_layernorm",
        "self_attn.k_norm", "self_attn.q_norm",
    )),
    *(f"model.visual.{s}.weight" for s in (
        "blocks.*.attn.proj", "blocks.*.attn.qkv", "blocks.*.mlp.linear_fc1",
        "blocks.*.mlp.linear_fc2", "blocks.*.norm1", "blocks.*.norm2",
        "merger.linear_fc1", "merger.linear_fc2", "merger.norm",
        "patch_embed.proj", "pos_embed",
    )),
    "mtp.fc_embedding.weight", "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    *(f"{_ML}{s}.weight" for s in (
        "attn_hyper_connection.block_inject_weight", "attn_hyper_connection.hc_norm",
        "attn_hyper_connection.input_mix_weight_down", "attn_hyper_connection.input_mix_weight_up",
        "mlp.experts.*.down_proj", "mlp.experts.*.gate_proj", "mlp.experts.*.up_proj",
        "mlp.gate", "mlp.shared_expert.down_proj", "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj", "mlp.shared_expert_gate",
        "mlp_hyper_connection.block_inject_weight", "mlp_hyper_connection.hc_norm",
        "mlp_hyper_connection.input_mix_weight_down", "mlp_hyper_connection.input_mix_weight_up",
        "self_attn.indexer.index_qk_proj", "self_attn.indexer.k_layernorm",
        "self_attn.indexer.q_layernorm", "self_attn.k_norm", "self_attn.k_proj",
        "self_attn.o_proj", "self_attn.q_norm", "self_attn.q_proj", "self_attn.v_proj",
    )),
    "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
)
_QWEN38_FLASH_ATTN_PATTERNS: tuple[str, ...] = tuple(f"{_QL}self_attn.{s}.weight" for s in (
    "indexer.index_qk_proj", "k_proj", "o_proj", "q_proj", "v_proj",
))
_QWEN38_FLASH_EXPERT_PATTERNS: tuple[str, ...] = tuple(
    f"{_QL}mlp.experts.*.{s}.weight" for s in ("down_proj", "gate_proj", "up_proj")
)

SCANNED_QUANT_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp": {
        "W8A8_MXFP8": _KIMI_K31_LINEAR_PATTERNS,
        "W4A8_MXFP": _KIMI_K31_EXPERT_PATTERNS,
        "FLOAT": _KIMI_K31_FLOAT_PATTERNS,
    },
    "Eco-Tech/Kimi-K3.1-Pro-w4a8": {
        "W8A8_DYNAMIC": _KIMI_K31_LINEAR_PATTERNS,
        "W4A8_DYNAMIC": _KIMI_K31_EXPERT_PATTERNS,
        "FLOAT": _KIMI_K31_FLOAT_PATTERNS,
    },
    "Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp": {
        "W8A8_MXFP8": _QWEN38_FLASH_ATTN_PATTERNS,
        "W4A8_MXFP": _QWEN38_FLASH_EXPERT_PATTERNS,
        "FLOAT": _QWEN38_FLASH_FLOAT_PATTERNS,
    },
    "Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp": {
        "W8A8_DYNAMIC": _QWEN38_FLASH_EXPERT_PATTERNS + _QWEN38_FLASH_ATTN_PATTERNS,
        "FLOAT": _QWEN38_FLASH_FLOAT_PATTERNS,
    },
}

_MX_BASE = {
    "weight_block": (1, MXFP_GROUP_SIZE), "expert_block": MXFP_GROUP_SIZE,
    "scale_format": "fp8_e8m0", "scale_bytes": MXFP_SCALE_BYTES,
    "activation_group_size": MXFP_GROUP_SIZE, "activation_scale_format": "fp8_e8m0",
    "activation_scale_pack": MXFP_SCALE_PACK, "is_quantized": True,
}
_INT_BASE = {
    "weight_block": (1, 1), "expert_block": 1, "scale_format": "fp32",
    "scale_bytes": FP32_SCALE_BYTES, "activation_group_size": 0,
    "activation_scale_format": "fp32", "activation_scale_pack": 1, "is_quantized": True,
}
_INT8_RT = {
    "runtime_weight_scale_format": "bf16", "runtime_weight_scale_bytes": BF16_SCALE_BYTES,
    "runtime_activation_scale_format": "fp32", "runtime_int8_weight_metadata": "scale_only",
}
_TAG_SPEC_TABLE: dict[str, dict[str, Any]] = {
    "W8A8_MXFP8": {"weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3", "ledger_format": "MXFP8", **_MX_BASE},
    "W4A8_MXFP": {"weight_format": "mxfp4", "activation_format": "fp8_e4m3", "ledger_format": "MXFP4", **_MX_BASE},
    "W4A4_MXFP4": {"weight_format": "mxfp4", "activation_format": "fp8_e4m3", "ledger_format": "MXFP4", **_MX_BASE},
    "W8A8_DYNAMIC": {"weight_format": "int8", "activation_format": "int8", "ledger_format": "W8A8", **_INT_BASE, **_INT8_RT},
    "W8A8": {"weight_format": "int8", "activation_format": "int8", "ledger_format": "W8A8", **_INT_BASE, **_INT8_RT},
    "W4A8_DYNAMIC": {"weight_format": "int4", "activation_format": "int8", "ledger_format": "W4A8", **_INT_BASE},
    "W4A4_DYNAMIC": {"weight_format": "int4", "activation_format": "int8", "ledger_format": "W4A8", **_INT_BASE},
    "FP8_DYNAMIC": {
        "weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3", "ledger_format": "FP8",
        "weight_block": (FP8_BLOCK_SIZE, FP8_BLOCK_SIZE), "expert_block": MXFP_GROUP_SIZE,
        "scale_format": "fp32", "scale_bytes": FP32_SCALE_BYTES,
        "activation_group_size": FP8_BLOCK_SIZE, "activation_scale_format": "fp32",
        "activation_scale_pack": 1, "is_quantized": True,
    },
    "FLOAT": {
        "weight_format": "bf16", "activation_format": "bf16", "ledger_format": "BF16",
        "weight_block": (0, 0), "expert_block": 0, "scale_format": "none",
        "scale_bytes": 0.0, "activation_group_size": 0, "activation_scale_format": "",
        "activation_scale_pack": 1, "is_quantized": False,
    },
}
_TAG_ALIASES: dict[str, str] = {
    "W8A8_MXFP8": "W8A8_MXFP8", "MXFP8": "W8A8_MXFP8", "W8A8_MXFP": "W8A8_MXFP8",
    "W4A8_MXFP": "W4A8_MXFP", "W4A8_MXFP4": "W4A8_MXFP", "MXFP4": "W4A8_MXFP",
    "W4A8_MXFP4_DYNAMIC": "W4A8_MXFP",
    "W4A4_MXFP4": "W4A4_MXFP4", "W4A4_MXFP": "W4A4_MXFP4", "MXFP4_W4A4": "W4A4_MXFP4",
    "W8A8_DYNAMIC": "W8A8_DYNAMIC", "W8A8_INT8": "W8A8_DYNAMIC", "W8A8": "W8A8", "INT8": "W8A8",
    "W4A8_DYNAMIC": "W4A8_DYNAMIC", "W4A8": "W4A8_DYNAMIC", "INT4": "W4A8_DYNAMIC",
    "W4A4_DYNAMIC": "W4A4_DYNAMIC", "W4A4": "W4A4_DYNAMIC",
    "FP8_DYNAMIC": "FP8_DYNAMIC", "FP8": "FP8_DYNAMIC", "W8A8_FP8": "FP8_DYNAMIC",
    "FLOAT": "FLOAT", "BF16": "FLOAT", "FP16": "FLOAT", "NONE": "FLOAT",
}
_STANDARD_UNQUANTIZED_EXCLUSIONS: tuple[str, ...] = (
    "router", "lm_head", "embedding", "norm", "mhc", "vision",
    "projector", "conv_weight_f32", "conv_weight_bf16",
)


@dataclass(frozen=True)
class WeightTensorQuant:
    """Exact per-tensor or per-role quantization classification."""

    raw_tag: str
    weight_format: str
    activation_format: str
    ledger_format: str
    scale_format: str
    group_size: int
    is_quantized: bool
    matched_pattern: str
    category: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_tag": self.raw_tag, "weight_format": self.weight_format,
            "activation_format": self.activation_format, "ledger_format": self.ledger_format,
            "scale_format": self.scale_format, "group_size": self.group_size,
            "is_quantized": self.is_quantized, "matched_pattern": self.matched_pattern,
            "category": self.category,
        }


@dataclass(frozen=True)
class NpuCheckpointSpec:
    """Curated open-source Ascend NPU checkpoint for one ``(model, hw_family)``."""

    canonical_model: str
    hw_family: str
    repo_id: str
    modelers_url: str
    modelscope_url: str
    quant_preset: str
    kv_cache_dtype: str = "bf16"
    indexer_kv_dtype: str = "bf16"
    hf_url: str = ""
    has_mtp_weights: bool = False
    has_dspark_weights: bool = False
    w8a8_repo_id: str = ""
    w4a8_repo_id: str = ""
    weight_patterns: tuple[tuple[str, tuple[str, ...]], ...] = ()
    alternatives: tuple[str, ...] = ()
    source_origin: str = "eco_tech_scan"
    notes: str = ""

    @property
    def uses_quarot(self) -> bool:
        return "quarot" in self.repo_id.lower()

    def weight_pattern_map(self) -> dict[str, tuple[str, ...]]:
        return {tag: patterns for tag, patterns in self.weight_patterns}

    def build_quant_spec(
        self, *, base_config: Mapping[str, Any] | None = None,
    ) -> QuantSpec:
        """Build an exact ``QuantSpec`` honoring baked patterns and KV dtypes."""
        primary_url = self.modelers_url or self.modelscope_url or self.hf_url
        if self.weight_patterns:
            parsed = parse_quant_model_description(
                self.weight_pattern_map(), base_config=base_config,
                checkpoint_repo=self.repo_id, checkpoint_url=primary_url,
            )
            return replace(
                parsed, kv_dtype=self.kv_cache_dtype or parsed.kv_dtype,
                index_dtype=self.indexer_kv_dtype or parsed.index_dtype,
            )
        return _quant_spec_from_preset(
            self.quant_preset, kv_dtype=self.kv_cache_dtype,
            index_dtype=self.indexer_kv_dtype, checkpoint_repo=self.repo_id,
            checkpoint_url=primary_url, base_config=base_config,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_model": self.canonical_model, "hw_family": self.hw_family,
            "repo_id": self.repo_id, "modelers_url": self.modelers_url,
            "modelscope_url": self.modelscope_url, "hf_url": self.hf_url,
            "quant_preset": self.quant_preset, "kv_cache_dtype": self.kv_cache_dtype,
            "indexer_kv_dtype": self.indexer_kv_dtype, "has_mtp_weights": self.has_mtp_weights,
            "has_dspark_weights": self.has_dspark_weights, "uses_quarot": self.uses_quarot,
            "w8a8_repo_id": self.w8a8_repo_id, "w4a8_repo_id": self.w4a8_repo_id,
            "weight_pattern_counts": {tag: len(pats) for tag, pats in self.weight_patterns},
            "alternatives": list(self.alternatives), "source_origin": self.source_origin,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CheckpointMetadataResult:
    """Zero-weight-download metadata resolution payload for local or remote specs."""

    spec_or_url: str
    source_kind: str
    repo_id: str
    canonical_model: str
    hw_family: str
    modelers_url: str
    modelscope_url: str
    hf_url: str
    config: Mapping[str, Any]
    quant_model_description: Mapping[str, Any]
    quantize_config: Mapping[str, Any]
    quant_spec: QuantSpec | None
    checkpoint: NpuCheckpointSpec | None
    from_cache: bool = False
    offline_fallback: bool = False
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_or_url": self.spec_or_url, "source_kind": self.source_kind,
            "repo_id": self.repo_id, "canonical_model": self.canonical_model,
            "hw_family": self.hw_family, "modelers_url": self.modelers_url,
            "modelscope_url": self.modelscope_url, "hf_url": self.hf_url,
            "has_config": bool(self.config),
            "has_quant_model_description": bool(self.quant_model_description),
            "has_quantize_config": bool(self.quantize_config),
            "quant_spec": self.quant_spec.to_dict() if self.quant_spec else None,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "from_cache": self.from_cache, "offline_fallback": self.offline_fallback,
            "warnings": list(self.warnings),
        }


def normalize_hw_family(hardware_or_family: str) -> str:
    """Normalize any Ascend SKU, topology alias, or family label to ``a2``/``a3``/``a5``."""
    raw = str(hardware_or_family or "a5").strip().lower()
    if raw in _SUPPORTED_HW_FAMILIES:
        return raw
    if any(tok in raw for tok in ("950", "a5", "910_95", "9582", "9581", "9579", "958b", "9599")):
        return "a5"
    if any(tok in raw for tok in ("910_93", "910c", "a3", "9392", "9382")):
        return "a3"
    if any(tok in raw for tok in ("910b", "a2", "310p", "300i")):
        return "a2"
    return "a5"


def _repo_urls(repo_id: str) -> tuple[str, str, str]:
    owner = repo_id.split("/", 1)[0].lower() if "/" in repo_id else ""
    modelers = f"{MODELERS_PAGE_BASE}/{repo_id}" if owner == "eco-tech" else ""
    modelscope = f"{MODELSCOPE_PAGE_BASE}/{repo_id}" if repo_id else ""
    hf_owners = {
        "deepseek-ai", "google", "stepfun-ai", "qwen", "zai-org",
        "moonshotai", "minimaxai", "meta-llama", "tencent", "meituan-longcat",
    }
    return modelers, modelscope, (f"{HF_PAGE_BASE}/{repo_id}" if owner in hf_owners else "")


def validate_quant_for_hardware(
    quant_preset_or_tag: str,
    hw_family: str,
    *,
    kv_cache_dtype: str = "bf16",
    strict: bool = False,
) -> tuple[str, ...]:
    """Enforce A2/A3 INT8/INT4/C8-only vs A5 MXFP/FP8 hardware-ISA separation."""
    hw = normalize_hw_family(hw_family)
    warnings: list[str] = []
    if hw in A2_A3_INT_ONLY_FAMILIES:
        q_lower = str(quant_preset_or_tag or "").strip().lower()
        kv_lower = str(kv_cache_dtype or "bf16").strip().lower()
        if any(tok in q_lower for tok in A5_ONLY_QUANT_TOKENS):
            msg = (
                f"hw_isa_incompatible:{hw}:weight_quant={quant_preset_or_tag}:"
                "A2/A3 lacks hardware FP8/MXFP; use W8A8_DYNAMIC or W4A8_DYNAMIC"
            )
            if strict:
                raise ModelingError(msg)
            warnings.append(msg)
        if kv_lower and kv_lower not in A2_A3_ALLOWED_KV_DTYPES:
            msg = (
                f"hw_isa_incompatible:{hw}:kv_cache_dtype={kv_cache_dtype}:"
                "A2/A3 supports BF16 or C8 (INT8) KV cache only"
            )
            if strict:
                raise ModelingError(msg)
            warnings.append(msg)
    return tuple(warnings)


def rank_checkpoint_candidates(
    candidates: Sequence[str],
    *,
    hw_family: str = "a5",
    prefer_quarot: bool = True,
    preferred_quant: str | None = None,
) -> tuple[str, ...]:
    """Rank checkpoint repo IDs for one ``(model, hw_family, bitwidth)`` with QuaRot first."""
    hw = normalize_hw_family(hw_family)
    pref = str(preferred_quant or "").strip().lower()

    def _score(repo: str) -> tuple[int, int, int, int, str]:
        lower = repo.lower()
        isa_pen = 1 if (hw in A2_A3_INT_ONLY_FAMILIES and any(t in lower for t in ("mxfp", "fp8"))) else 0
        quant_pen = 0 if (not pref or pref in lower) else 1
        quarot_pen = 0 if (prefer_quarot and "quarot" in lower) else 1
        c8_pen = 0 if "c8" in lower else 1
        return (isa_pen, quant_pen, quarot_pen, c8_pen, lower)

    unique = list(dict.fromkeys(str(c).strip() for c in candidates if str(c).strip()))
    return tuple(sorted(unique, key=_score))


def msmodelslim_quarot_guidance(
    hw_family: str = "a5", *, quant_preset: str = "",
) -> dict[str, Any]:
    """Return structured ``msmodelslim`` QuaRot Hadamard outlier-rotation guidance."""
    hw = normalize_hw_family(hw_family)
    if hw in A2_A3_INT_ONLY_FAMILIES:
        recipe = quant_preset or "W4A8_DYNAMIC + C8 (INT8 Attn/Linear + INT4 Experts + INT8 KV)"
        algo = "QuaRot (Hadamard rotation) + W8A8_DYNAMIC/W4A8_DYNAMIC + C8 KV"
    else:
        recipe = quant_preset or "W4A8_MXFP / W8A8_MXFP8 (MXFP8 Attn + MXFP4 Experts)"
        algo = "QuaRot (Hadamard rotation) + MXFP8/MXFP4 microscaling (group=32, E8M0)"
    return {
        "hw_family": hw, "prefer_quarot": True,
        "recommended_algorithm": algo, "target_recipe": recipe,
        "rationale": (
            "QuaRot Hadamard rotation suppresses outlier channels and yields higher "
            "accuracy at the same bitwidth; prefer *-QuaRot Eco-Tech checkpoints "
            "or enable QuaRot in msmodelslim for uncataloged models."
        ),
    }


def _infer_pair_repo_ids(
    repo_id: str, quant_preset: str, alternatives: Sequence[str],
    w8a8_repo_id: str, w4a8_repo_id: str,
) -> tuple[str, str]:
    pool = rank_checkpoint_candidates((repo_id, *alternatives), hw_family="a3", prefer_quarot=True)
    w8, w4 = w8a8_repo_id, w4a8_repo_id
    for cand in pool:
        lower = cand.lower()
        if any(tok in lower for tok in ("mxfp", "310")):
            continue
        if not w4 and "w4a8" in lower:
            w4 = cand
        if not w8 and "w8a8" in lower:
            w8 = cand
    if not w8 and quant_preset in {"w8a8_dynamic", "w8a8"} and "mxfp" not in repo_id.lower():
        w8 = repo_id
    if not w4 and quant_preset in {"w4a8_dynamic", "w4a8"} and "mxfp" not in repo_id.lower():
        w4 = repo_id
    return w8, w4


def _ckpt(
    canonical_model: str, hw_family: str, repo_id: str, quant_preset: str, **opts: Any,
) -> NpuCheckpointSpec:
    kv = str(opts.get("kv_cache_dtype", "bf16"))
    validate_quant_for_hardware(quant_preset, hw_family, kv_cache_dtype=kv, strict=True)
    m_url, ms_url, hf_url = _repo_urls(repo_id)
    raw_pats = SCANNED_QUANT_PATTERNS.get(repo_id)
    alts = tuple(opts.get("alternatives", ()))
    w8, w4 = _infer_pair_repo_ids(
        repo_id, quant_preset, alts, str(opts.get("w8a8_repo_id", "")), str(opts.get("w4a8_repo_id", "")),
    )
    return NpuCheckpointSpec(
        canonical_model=canonical_model, hw_family=hw_family, repo_id=repo_id,
        modelers_url=m_url, modelscope_url=ms_url, hf_url=hf_url, quant_preset=quant_preset,
        kv_cache_dtype=kv, indexer_kv_dtype=str(opts.get("indexer_kv_dtype", "bf16")),
        has_mtp_weights=bool(opts.get("has_mtp_weights", False)),
        has_dspark_weights=bool(opts.get("has_dspark_weights", False)),
        w8a8_repo_id=w8, w4a8_repo_id=w4,
        weight_patterns=tuple(sorted(raw_pats.items())) if raw_pats else (),
        alternatives=alts,
        source_origin=str(opts.get("source_origin", "eco_tech_scan+vllm_ascend_tutorial")),
        notes=str(opts.get("notes", "")),
    )


# Declarative (model, a5_spec, a3_spec_or_None, a2_spec_or_None) checkpoint catalog.
# Each hw spec is (repo_id, quant_preset, opts_dict).
_CHECKPOINT_TABLE: tuple[tuple[str, tuple[str, str, dict[str, Any]], Any, Any], ...] = (
    ("kimi-k3.1",
     ("Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp", "w4a8_mxfp", {"kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8", "w4a8_repo_id": "Eco-Tech/Kimi-K3.1-Pro-w4a8", "alternatives": ("Eco-Tech/Kimi-K3.1-Pro-w4a8",)}),
     ("Eco-Tech/Kimi-K3.1-Pro-w4a8", "w4a8_dynamic", {"kv_cache_dtype": "int8", "indexer_kv_dtype": "int8", "w4a8_repo_id": "Eco-Tech/Kimi-K3.1-Pro-w4a8"}),
     None),
    ("kimi-k3",
     ("Eco-Tech/Kimi-K3-w4a8", "w4a8_dynamic", {}),
     ("Eco-Tech/Kimi-K3-w4a8", "w4a8_dynamic", {"kv_cache_dtype": "int8"}),
     None),
    ("kimi-k3-dspark",
     ("Inferact/Kimi-K3-DSpark-Block5", "bf16", {"has_dspark_weights": True, "alternatives": ("Eco-Tech/Kimi-K3-DSpark-MLA-NPU",), "source_origin": "vllm_ascend_tutorial"}),
     ("Eco-Tech/Kimi-K3-DSpark-MLA-NPU", "bf16", {"has_dspark_weights": True, "alternatives": ("Inferact/Kimi-K3-DSpark-Block5",), "source_origin": "vllm_ascend_tutorial"}),
     None),
    ("kimi-k3-dspark-block5",
     ("Inferact/Kimi-K3-DSpark-Block5", "bf16", {"has_dspark_weights": True, "alternatives": ("Eco-Tech/Kimi-K3-DSpark-MLA-NPU",), "source_origin": "vllm_ascend_tutorial"}),
     None, None),
    ("kimi-k2.6",
     ("Eco-Tech/Kimi-K2.6-w4a8", "w4a8_dynamic", {"w8a8_repo_id": "Eco-Tech/Kimi-K2.6-w8a8-xllm", "w4a8_repo_id": "Eco-Tech/Kimi-K2.6-w4a8", "alternatives": ("Eco-Tech/Kimi-K2.6-w8a8-xllm",)}),
     None, None),
    ("kimi-k2.5",
     ("Eco-Tech/Kimi-K2.5-w4a8", "w4a8_dynamic", {"w8a8_repo_id": "Eco-Tech/Kimi-K2.5-W8A8-xLLM", "w4a8_repo_id": "Eco-Tech/Kimi-K2.5-w4a8", "alternatives": ("Eco-Tech/Kimi-K2.5-W8A8-xLLM",)}),
     None, None),
    ("kimi-k2",
     ("Eco-Tech/KIMI-k2-Instruct-0905-W8A8-QuaRot", "w8a8_dynamic", {"w8a8_repo_id": "Eco-Tech/KIMI-k2-Instruct-0905-W8A8-QuaRot", "w4a8_repo_id": "Eco-Tech/KIMI-k2-Instruct-0905-W4A8-QuaRot", "alternatives": ("Eco-Tech/KIMI-k2-Instruct-0905-W4A8-QuaRot",)}),
     ("Eco-Tech/KIMI-k2-Instruct-0905-W4A8-QuaRot", "w4a8_dynamic", {"w8a8_repo_id": "Eco-Tech/KIMI-k2-Instruct-0905-W8A8-QuaRot", "w4a8_repo_id": "Eco-Tech/KIMI-k2-Instruct-0905-W4A8-QuaRot", "alternatives": ("Eco-Tech/KIMI-k2-Instruct-0905-W8A8-QuaRot",)}),
     None),
    ("kimi-k2-thinking", ("Eco-Tech/KIMI-k2-Thinking-W8A8-QuaRot", "w8a8_dynamic", {}), None, None),
    ("qwen3.8-flash-next",
     ("Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp", "w4a8_mxfp", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp", "w4a8_repo_id": "Eco-Tech/Qwen3.8-Flash-Next-w4a8", "alternatives": ("Eco-Tech/Qwen3.8-Flash-Next-w8a8-mxfp8-mtp",)}),
     ("Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp", "w8a8_dynamic", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp", "w4a8_repo_id": "Eco-Tech/Qwen3.8-Flash-Next-w4a8", "alternatives": ("Eco-Tech/Qwen3.8-Flash-Next-w4a8",)}),
     ("Eco-Tech/Qwen3.8-Flash-Next-w4a8", "w4a8_dynamic", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp", "w4a8_repo_id": "Eco-Tech/Qwen3.8-Flash-Next-w4a8", "alternatives": ("Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp",)})),
    ("qwen3.8-27b",
     ("Eco-Tech/Qwen3.8-27B-w8a8c8-mxfp8", "w8a8_mxfp8", {"kv_cache_dtype": "fp8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Qwen3.8-27B-w8a8", "alternatives": ("Eco-Tech/Qwen3.8-27B-w8a8-mxfp8", "Eco-Tech/Qwen3.8-27B-w8a8-mxfp8-mtp")}),
     ("Eco-Tech/Qwen3.8-27B-w8a8", "w8a8_dynamic", {"has_mtp_weights": True}),
     None),
    ("qwen3.8-2.4t-a95b",
     ("Eco-Tech/Qwen3.8-2.4T-A95B-w4a8", "w4a8_dynamic", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Qwen3.8-2.4T-A95B-w8a8", "w4a8_repo_id": "Eco-Tech/Qwen3.8-2.4T-A95B-w4a8", "alternatives": ("Eco-Tech/Qwen3.8-2.4T-A95B-w8a8",)}),
     None, None),
    ("qwen3.6-35b-a3b",
     ("Eco-Tech/Qwen3.6-35B-A3B-w4a8-mxfp", "w4a8_mxfp", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Qwen3.6-35B-A3B-w8a8", "alternatives": ("Eco-Tech/Qwen3.6-35B-A3B-w8a8c8-mxfp8", "Eco-Tech/Qwen3.6-35B-A3B-w8a8-mxfp8")}),
     ("Eco-Tech/Qwen3.6-35B-A3B-w8a8", "w8a8_dynamic", {"has_mtp_weights": True}),
     None),
    ("qwen3.6-27b",
     ("Eco-Tech/Qwen3.6-27B-w8a8c8-mxfp8", "w8a8_mxfp8", {"kv_cache_dtype": "fp8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Qwen3.6-27B-w8a8", "alternatives": ("Eco-Tech/Qwen3.6-27B-w8a8-mxfp8",)}),
     ("Eco-Tech/Qwen3.6-27B-w8a8", "w8a8_dynamic", {"has_mtp_weights": True}),
     None),
    ("qwen3.5-397b-a17b",
     ("Eco-Tech/Qwen3.5-397B-A17B-w4a8c8-mxfp", "w4a8_mxfp", {"kv_cache_dtype": "fp8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp", "w4a8_repo_id": "Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp", "alternatives": ("Eco-Tech/Qwen3.5-397B-A17B-w4a4-mxfp4", "Eco-Tech/Qwen3.5-397B-A17B-w8a8-mxfp8", "Eco-Tech/Qwen3.5-397B-A17B-w8a8c8-mxfp8")}),
     ("Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp", "w8a8_dynamic", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp", "w4a8_repo_id": "Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp", "alternatives": ("Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp",)}),
     ("Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp", "w4a8_dynamic", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp", "w4a8_repo_id": "Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp", "alternatives": ("Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp",)})),
    ("qwen3.5-122b-a10b", ("Eco-Tech/Qwen3.5-122B-A10B-w8a8-mtp", "w8a8_dynamic", {"has_mtp_weights": True}), None, None),
    ("qwen3.5-35b-a3b", ("Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp", "w8a8_dynamic", {"has_mtp_weights": True}), None, None),
    ("qwen3.5-moe", ("Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp", "w8a8_dynamic", {"has_mtp_weights": True}), None, None),
    ("qwen3.5-27b", ("Eco-Tech/Qwen3.5-27B-w8a8-mtp", "w8a8_dynamic", {"has_mtp_weights": True}), None, None),
    ("qwen3-next-80b-a3b", ("Eco-Tech/Qwen3-Next-80B-A3B-Instruct-w8a8-mtp", "w8a8_dynamic", {"has_mtp_weights": True}), None, None),
    ("qwen3-235b-a22b",
     ("Eco-Tech/Qwen3-235B-A22B-Instruct-2507-w8a8c8-QuaRot", "w8a8_dynamic", {"kv_cache_dtype": "int8", "alternatives": ("Eco-Tech/Qwen3-235B-A22B-w8a8-QuaRot", "Eco-Tech/Qwen3-235B-A22B-Instruct-2507-w8a8-QuaRot")}),
     None, None),
    ("qwen3-30b-a3b",
     ("Eco-Tech/Qwen3-30B-A3B-w8a8", "w8a8_dynamic", {"alternatives": ("Eco-Tech/Qwen3-30B-A3B-Thinking-2507-w8a8", "Eco-Tech/Qwen3-30B-A3B-w4a4-LAOS")}),
     None, None),
    ("qwen3-32b",
     ("Eco-Tech/Qwen3-32B-w8a8c8", "w8a8_dynamic", {"kv_cache_dtype": "int8", "alternatives": ("vllm-ascend/Qwen3-32B-W8A8", "Eco-Tech/Qwen3-32B-w8a8-pdmix", "Eco-Tech/Qwen3-32B-w4a4-LAOS")}),
     None, None),
    ("deepseek-v4.1-flash",
     ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "fp8_dynamic+mxfp4_experts", {"indexer_kv_dtype": "int8", "has_mtp_weights": True, "has_dspark_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8", "w4a8_repo_id": "Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8", "alternatives": ("Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8", "Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8"), "source_origin": "vllm_ascend_tutorial"}),
     ("Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8", "w4a8_dynamic", {"has_dspark_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8", "w4a8_repo_id": "Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8", "alternatives": ("Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8", "Eco-Tech/DeepSeek-V4.1-Flash-w4a8", "Eco-Tech/DeepSeek-V4.1-Flash-w8a8")}),
     None),
    ("deepseek-v4-pro-0813",
     ("deepseek-ai/DeepSeek-V4-Pro-0813", "fp8_dynamic+mxfp4_experts", {"indexer_kv_dtype": "int8", "has_mtp_weights": True, "has_dspark_weights": True, "w4a8_repo_id": "Eco-Tech/DeepSeek-V4-Pro-0813-w4a8", "alternatives": ("Eco-Tech/DeepSeek-V4-Pro-0813-w4a8", "Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp"), "source_origin": "vllm_ascend_tutorial"}),
     ("Eco-Tech/DeepSeek-V4-Pro-0813-w4a8", "w4a8_dynamic", {"has_dspark_weights": True, "has_mtp_weights": True, "w4a8_repo_id": "Eco-Tech/DeepSeek-V4-Pro-0813-w4a8", "alternatives": ("Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp",)}),
     None),
    ("deepseek-v4-flash-0731",
     ("deepseek-ai/DeepSeek-V4-Flash-0731", "fp8_dynamic+mxfp4_experts", {"indexer_kv_dtype": "int8", "has_mtp_weights": True, "has_dspark_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", "alternatives": ("Eco-Tech/DeepSeek-V4-Flash-0731-w8a8", "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp"), "source_origin": "vllm_ascend_tutorial"}),
     ("Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", "w8a8_dynamic", {"has_mtp_weights": True, "has_dspark_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", "alternatives": ("Eco-Tech/DeepSeek-V4-Flash-0731-w8a8", "Eco-Tech/DeepSeek-V4-Flash-DSpark-w8a8")}),
     None),
    ("deepseek-v4-10t",
     ("deepseek-ai/DeepSeek-V4-Pro-0813", "fp8_dynamic+mxfp4_experts", {"indexer_kv_dtype": "int8", "has_mtp_weights": True, "w4a8_repo_id": "Eco-Tech/DeepSeek-V4-Pro-0813-w4a8", "alternatives": ("Eco-Tech/DeepSeek-V4-Pro-0813-w4a8",), "source_origin": "vllm_ascend_tutorial"}),
     ("Eco-Tech/DeepSeek-V4-Pro-0813-w4a8", "w4a8_dynamic", {"alternatives": ("Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp",)}),
     None),
    ("deepseek-v3.2",
     ("Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot", "w8a8_dynamic", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot", "w4a8_repo_id": "Eco-Tech/DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot", "alternatives": ("Eco-Tech/DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot", "Eco-Tech/DeepSeek-V3.2-Speciale-w8a8-mtp-QuaRot")}),
     None,
     ("Eco-Tech/DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot", "w4a8_dynamic", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot", "w4a8_repo_id": "Eco-Tech/DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot", "alternatives": ("Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot",)})),
    ("deepseek-v3.2-prune-3l",
     ("Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot", "w8a8_dynamic", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot", "w4a8_repo_id": "Eco-Tech/DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot"}),
     None, None),
    ("deepseek-v3.1",
     ("Eco-Tech/DeepSeek-V3.1-Terminus-w8a8c8-mtp-QuaRot", "w8a8_dynamic", {"kv_cache_dtype": "int8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V3.1-Terminus-w8a8c8-mtp-QuaRot", "w4a8_repo_id": "Eco-Tech/DeepSeek-V3.1-Terminus-w4a8c8-mtp-QuaRot", "alternatives": ("Eco-Tech/DeepSeek-V3.1-Terminus-w4a8c8-mtp-QuaRot", "Eco-Tech/DeepSeek-V3.1-w4a8-mtp-QuaRot")}),
     None,
     ("Eco-Tech/DeepSeek-V3.1-Terminus-w4a8c8-mtp-QuaRot", "w4a8_dynamic", {"kv_cache_dtype": "int8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V3.1-Terminus-w8a8c8-mtp-QuaRot", "w4a8_repo_id": "Eco-Tech/DeepSeek-V3.1-Terminus-w4a8c8-mtp-QuaRot", "alternatives": ("Eco-Tech/DeepSeek-V3.1-Terminus-w8a8c8-mtp-QuaRot",)})),
    ("deepseek-r1",
     ("Eco-Tech/DeepSeek-R1-0528-w8a8c8-mtp-QuaRot", "w8a8_dynamic", {"kv_cache_dtype": "int8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-R1-0528-w8a8c8-mtp-QuaRot", "w4a8_repo_id": "Eco-Tech/DeepSeek-R1-0528-w4a8c8-mtp-QuaRot", "alternatives": ("vllm-ascend/DeepSeek-R1-W8A8", "Eco-Tech/DeepSeek-R1-0528-w4a8c8-mtp-QuaRot")}),
     None,
     ("Eco-Tech/DeepSeek-R1-0528-w4a8c8-mtp-QuaRot", "w4a8_dynamic", {"kv_cache_dtype": "int8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-R1-0528-w8a8c8-mtp-QuaRot", "w4a8_repo_id": "Eco-Tech/DeepSeek-R1-0528-w4a8c8-mtp-QuaRot", "alternatives": ("vllm-ascend/DeepSeek-R1-W8A8", "Eco-Tech/DeepSeek-R1-0528-w8a8-mtp-QuaRot")})),
    ("deepseek-v3",
     ("Eco-Tech/DeepSeek-V3-0324-w8a8c8-mtp-QuaRot", "w8a8_dynamic", {"kv_cache_dtype": "int8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V3-0324-w8a8c8-mtp-QuaRot", "w4a8_repo_id": "Eco-Tech/DeepSeek-V3-0324-w4a8c8-mtp-QuaRot", "alternatives": ("Eco-Tech/DeepSeek-V3-0324-w4a8c8-mtp-QuaRot", "Eco-Tech/DeepSeek-V3-0324-w8a8-mtp-QuaRot")}),
     None,
     ("Eco-Tech/DeepSeek-V3-0324-w4a8c8-mtp-QuaRot", "w4a8_dynamic", {"kv_cache_dtype": "int8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/DeepSeek-V3-0324-w8a8c8-mtp-QuaRot", "w4a8_repo_id": "Eco-Tech/DeepSeek-V3-0324-w4a8c8-mtp-QuaRot", "alternatives": ("Eco-Tech/DeepSeek-V3-0324-w8a8c8-mtp-QuaRot",)})),
    ("deepseek-v2-lite", ("deepseek-ai/DeepSeek-V2-Lite", "bf16", {"source_origin": "vllm_ascend_tutorial"}), None, None),
    ("glm-5.3",
     ("Eco-Tech/GLM-5.3-W8A8C8-mxfp8", "w8a8_mxfp8", {"kv_cache_dtype": "fp8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/GLM-5.3-w8a8c8", "alternatives": ("Eco-Tech/GLM-5.3-w4a4c8-mxfp4",)}),
     ("Eco-Tech/GLM-5.3-w8a8c8", "w8a8_dynamic", {"kv_cache_dtype": "int8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/GLM-5.3-w8a8c8", "alternatives": ("Eco-Tech/GLM-5.3-w8a8",)}),
     None),
    ("glm-5.3-flash",
     ("Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8", "w8a8_mxfp8", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/GLM-5.3-Flash-w8a8"}),
     ("Eco-Tech/GLM-5.3-Flash-w8a8", "w8a8_dynamic", {"has_mtp_weights": True}),
     None),
    ("glm-5.2",
     ("Eco-Tech/GLM-5.2-w4a4c8-mxfp4", "w4a4_mxfp4", {"kv_cache_dtype": "fp8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/GLM-5.2-w8a8c8", "w4a8_repo_id": "Eco-Tech/GLM-5.2-w4a8c8", "alternatives": ("Eco-Tech/GLM-5.2-W8A8C8-mxfp8",)}),
     ("Eco-Tech/GLM-5.2-w8a8c8", "w8a8_dynamic", {"kv_cache_dtype": "int8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/GLM-5.2-w8a8c8", "w4a8_repo_id": "Eco-Tech/GLM-5.2-w4a8c8", "alternatives": ("Eco-Tech/GLM-5.2-w4a8c8", "Eco-Tech/GLM-5.2-w8a8")}),
     ("Eco-Tech/GLM-5.2-w4a8c8", "w4a8_dynamic", {"kv_cache_dtype": "int8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/GLM-5.2-w8a8c8", "w4a8_repo_id": "Eco-Tech/GLM-5.2-w4a8c8", "alternatives": ("Eco-Tech/GLM-5.2-w8a8c8", "Eco-Tech/GLM-5.2-w4a8")})),
    ("glm-5.1",
     ("Eco-Tech/GLM-5.1-w4a4c8-mxfp4", "w4a4_mxfp4", {"kv_cache_dtype": "fp8", "has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/GLM-5.1-w8a8c8-MTP", "w4a8_repo_id": "Eco-Tech/GLM-5.1-w4a8", "alternatives": ("Eco-Tech/GLM-5.1-w8a8c8-mxfp8",)}),
     ("Eco-Tech/GLM-5.1-w4a8", "w4a8_dynamic", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/GLM-5.1-w8a8c8-MTP", "w4a8_repo_id": "Eco-Tech/GLM-5.1-w4a8", "alternatives": ("Eco-Tech/GLM-5.1-w8a8c8-MTP", "Eco-Tech/GLM-5.1-w8a8", "Eco-Tech/GLM-5-w4a8-mtp-QuaRot")}),
     None),
    ("glm-5",
     ("Eco-Tech/GLM-5-w4a8-mtp-QuaRot", "w4a8_dynamic", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/GLM-5-w8a8", "w4a8_repo_id": "Eco-Tech/GLM-5-w4a8-mtp-QuaRot", "alternatives": ("Eco-Tech/GLM-5-w4a8", "Eco-Tech/GLM-5-w8a8"), "notes": "QuaRot-first priority over non-QuaRot GLM-5-w4a8"}),
     None, None),
    ("minimax-m3",
     ("MiniMax/MiniMax-M3-MXFP8", "w8a8_mxfp8", {"kv_cache_dtype": "fp8", "w8a8_repo_id": "Eco-Tech/MiniMax-M3-w8a8-0626", "alternatives": ("Eco-Tech/MiniMax-M3-w8a8-0626", "Eco-Tech/MiniMax-M3-w8a8"), "source_origin": "vllm_ascend_tutorial"}),
     ("Eco-Tech/MiniMax-M3-w8a8-0626", "w8a8_dynamic", {"w8a8_repo_id": "Eco-Tech/MiniMax-M3-w8a8-0626", "alternatives": ("Eco-Tech/MiniMax-M3-w8a8",)}),
     None),
    ("minimax-m3.1-flash",
     ("Eco-Tech/MiniMax-M3.1-Flash-w8a8-mxfp8", "w8a8_mxfp8", {"kv_cache_dtype": "fp8", "has_dspark_weights": True, "w8a8_repo_id": "Eco-Tech/MiniMax-M3.1-Flash-w8a8"}),
     ("Eco-Tech/MiniMax-M3.1-Flash-w8a8", "w8a8_dynamic", {"has_dspark_weights": True}),
     None),
    ("minimax-m2.7", ("Eco-Tech/MiniMax-M2.7-w8a8-QuaRot", "w8a8_dynamic", {}), None, None),
    ("minimax-m2.5", ("Eco-Tech/MiniMax-M2.5-w8a8-QuaRot", "w8a8_dynamic", {}), None, None),
    ("step-5-preview",
     ("Eco-Tech/Step-5-Preview-w8a8-mxfp8", "w8a8_mxfp8", {"has_mtp_weights": True}),
     ("stepfun-ai/Step-5-Preview", "bf16", {"has_mtp_weights": True, "source_origin": "vllm_ascend_tutorial"}),
     None),
    ("step-3.7-flash",
     ("Eco-Tech/Step-3.7-Flash-w8a8-mxfp8", "w8a8_mxfp8", {"has_mtp_weights": True, "w8a8_repo_id": "Eco-Tech/Step-3.7-Flash-w8a8-mtp"}),
     ("Eco-Tech/Step-3.7-Flash-w8a8-mtp", "w8a8_dynamic", {"has_mtp_weights": True}),
     None),
    ("step-3.5-flash", ("Eco-Tech/Step-3.5-Flash-w8a8-mtp", "w8a8_dynamic", {"has_mtp_weights": True}), None, None),
    ("gemma-4-31b",
     ("Eco-Tech/gemma-4-31B-it-w4a4-mxfp4", "w4a4_mxfp4", {}),
     ("google/gemma-4-31B-it", "bf16", {"source_origin": "vllm_ascend_tutorial"}),
     None),
    ("gemma-4-26b-a4b",
     ("Eco-Tech/gemma-4-26B-A4B-it-w4a4-mxfp4", "w4a4_mxfp4", {}),
     ("google/gemma-4-26B-A4B-it", "bf16", {"source_origin": "vllm_ascend_tutorial"}),
     None),
    ("bailing-v3",
     ("Eco-Tech/Ling-3.1-flash-w4a8", "w4a8_dynamic", {"alternatives": ("Eco-Tech/Ling-3.0-flash-w4a8", "Eco-Tech/Ling-2.6-1T-w4a8", "Eco-Tech/Ring-2.5-1T-w4a8")}),
     None, None),
    ("qwen3-coder-480b-a35b", ("Eco-Tech/Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot", "w8a8_dynamic", {}), None, None),
    ("qwen3-vl-235b-a22b", ("Eco-Tech/Qwen3-VL-235B-A22B-Instruct-w8a8-QuaRot", "w8a8_dynamic", {}), None, None),
    ("qwen3-vl-32b", ("Eco-Tech/Qwen3-VL-32B-Instruct-w8a8-QuaRot", "w8a8_dynamic", {}), None, None),
    ("qwen3-vl-30b-a3b", ("Eco-Tech/Qwen3-VL-30B-A3B-Instruct-w8a8-QuaRot", "w8a8_dynamic", {}), None, None),
    ("qwen3-vl-8b", ("Eco-Tech/Qwen3-VL-8B-Instruct-w8a8-QuaRot", "w8a8_dynamic", {}), None, None),
    ("longcat-flash-lite", ("meituan-longcat/LongCat-Flash-Lite", "bf16", {"source_origin": "vllm_ascend_tutorial"}), None, None),
    ("hunyuan-video-dit", ("tencent/HunyuanVideo", "bf16", {"source_origin": "vllm_ascend_tutorial"}), None, None),
)


def _build_default_npu_checkpoints() -> dict[tuple[str, str], NpuCheckpointSpec]:
    reg: dict[tuple[str, str], NpuCheckpointSpec] = {}
    for model, a5, a3, a2 in _CHECKPOINT_TABLE:
        a3_eff = a3 if a3 is not None else a5
        a2_eff = a2 if a2 is not None else a3_eff
        for hw, (repo, preset, opts) in (("a5", a5), ("a3", a3_eff), ("a2", a2_eff)):
            reg[(model, hw)] = _ckpt(model, hw, repo, preset, **opts)
    return reg


DEFAULT_NPU_CHECKPOINTS: dict[tuple[str, str], NpuCheckpointSpec] = _build_default_npu_checkpoints()

_EXTRA_MODEL_ALIASES: dict[str, str] = {
    "kimi-k3.1-pro": "kimi-k3.1", "kimi-k3.1-pro-w4a8": "kimi-k3.1",
    "kimi-k3.1-pro-w4a8-mxfp": "kimi-k3.1", "kimi-k3-w4a8": "kimi-k3",
    "deepseek-v4-pro": "deepseek-v4-pro-0813", "deepseek-v4-flash": "deepseek-v4-flash-0731",
    "deepseek-v4-flash-0731-w8a8": "deepseek-v4-flash-0731",
    "deepseek-v4-flash-w8a8-mtp": "deepseek-v4-flash-0731",
    "deepseek-v4-pro-0813-w4a8": "deepseek-v4-pro-0813",
    "deepseek-v4-pro-w4a8-mtp": "deepseek-v4-pro-0813",
    "deepseek-v4.1-flash-1017-w4a8": "deepseek-v4.1-flash",
    "deepseek-v4.1-flash-1017-w8a8": "deepseek-v4.1-flash",
    "glm-5": "glm-5", "glm-5-w4a8": "glm-5", "glm-5-w4a8-mtp-quarot": "glm-5",
    "glm-5.1-w4a8": "glm-5.1", "glm-5.2-w8a8c8": "glm-5.2", "glm-5.3-w8a8c8": "glm-5.3",
    "qwen3-30b": "qwen3-30b-a3b", "qwen3.8-2.4t": "qwen3.8-2.4t-a95b",
    "step-3.7-flash-w8a8-mtp": "step-3.7-flash", "step-3.5-flash-w8a8-mtp": "step-3.5-flash",
}


def normalize_model_key(model_id: str) -> str:
    """Resolve a catalog key, alias, or ``owner/repo`` checkpoint ID to a canonical model key."""
    text = str(model_id or "").strip()
    if not text:
        return ""
    repo_tail = text.rsplit("/", 1)[-1] if "/" in text else text
    aliases = {
        **{k.lower(): v for k, v in CORE_MODEL_ALIASES.items()},
        **{k.lower(): v for k, v in ARCHITECTURE_PRESET_ALIASES.items()},
        **_EXTRA_MODEL_ALIASES,
    }
    known = set(CORE_CATALOG_ENTRIES) | set(ARCHITECTURE_PRESET_CATALOG) | {
        m for (m, _hw) in DEFAULT_NPU_CHECKPOINTS
    }
    for candidate in (text, repo_tail):
        lower = candidate.lower()
        if lower in known:
            return lower
        if lower in aliases:
            return aliases[lower]
    stripped = re.sub(
        r"(?:[-_](?:w\d+a\d+(?:c\d+)?(?:s|sc)?|mxfp\d*|mxfp|fp8|bf16|mtp|floatmtp|quarot|laos|pdmix|nopdmix|xllm|310p?|vllm|mindie|exp|terminus|speciale|0626|0731|0813|1017|0528|0324|0905|2507|it|instruct|pro))+$",
        "",
        repo_tail.lower(),
    )
    if stripped in known:
        return stripped
    if stripped in aliases:
        return aliases[stripped]
    for (canon, _hw), spec in DEFAULT_NPU_CHECKPOINTS.items():
        if spec.repo_id.lower() == text.lower() or any(alt.lower() == text.lower() for alt in spec.alternatives):
            return canon
    return repo_tail.lower()


def resolve_default_checkpoint(
    model_id: str, hw_family: str = "a5",
) -> NpuCheckpointSpec | None:
    """Look up the curated default NPU checkpoint for ``(model_id, hw_family)``."""
    canon = normalize_model_key(model_id)
    hw = normalize_hw_family(hw_family)
    direct = DEFAULT_NPU_CHECKPOINTS.get((canon, hw))
    if direct is not None:
        return direct
    order = ("a3", "a2") if hw in A2_A3_INT_ONLY_FAMILIES else ("a5", "a3", "a2")
    for fallback_hw in order:
        found = DEFAULT_NPU_CHECKPOINTS.get((canon, fallback_hw))
        if found is not None:
            return found
    return None


def select_checkpoint_for_capacity(
    model_id: str,
    hw_family: str,
    *,
    w8a8_fits_hbm: bool = True,
    hbm_constrained: bool = False,
    long_context: bool = False,
) -> tuple[NpuCheckpointSpec | None, tuple[str, ...]]:
    """Select an ISA-valid checkpoint with capacity-driven ``w8a8 -> w4a8 + c8`` upgrade on A2/A3."""
    hw = normalize_hw_family(hw_family)
    base = resolve_default_checkpoint(model_id, hw_family=hw)
    if base is None:
        return None, ()
    notes: list[str] = []
    selected = base
    needs_w4a8 = hw in A2_A3_INT_ONLY_FAMILIES and (not w8a8_fits_hbm or hbm_constrained)
    if needs_w4a8 and selected.quant_preset in {"w8a8_dynamic", "w8a8"}:
        if selected.w4a8_repo_id:
            target_repo = selected.w4a8_repo_id
            m_url, ms_url, hf_url = _repo_urls(target_repo)
            raw_pats = SCANNED_QUANT_PATTERNS.get(target_repo)
            kv = "int8" if "c8" in target_repo.lower() else selected.kv_cache_dtype
            selected = replace(
                selected, repo_id=target_repo, modelers_url=m_url, modelscope_url=ms_url,
                hf_url=hf_url, quant_preset="w4a8_dynamic", kv_cache_dtype=kv,
                weight_patterns=tuple(sorted(raw_pats.items())) if raw_pats else (),
                notes=f"Capacity-upgraded {hw.upper()} from W8A8 ({selected.w8a8_repo_id}) to W4A8 ({target_repo})",
            )
            notes.append(f"capacity_upgrade_{hw}_w8a8_to_w4a8:{selected.w8a8_repo_id}->{target_repo}")
        else:
            notes.append(f"capacity_hint_{hw}_msmodelslim_quarot_w4a8")
    if hw in A2_A3_INT_ONLY_FAMILIES and long_context and selected.kv_cache_dtype == "bf16":
        selected = replace(selected, kv_cache_dtype="int8")
        notes.append(f"long_context_{hw}_kv_promoted_to_c8_int8")
    return selected, tuple(notes)


def list_model_checkpoints(model_id: str) -> dict[str, NpuCheckpointSpec]:
    """Return all curated ``hw_family -> NpuCheckpointSpec`` entries for one model."""
    canon = normalize_model_key(model_id)
    return {
        hw: DEFAULT_NPU_CHECKPOINTS[(canon, hw)]
        for hw in ("a2", "a3", "a5")
        if (canon, hw) in DEFAULT_NPU_CHECKPOINTS
    }


def lookup_eco_tech_repo(repo_or_name: str) -> str | None:
    """Return the exact canonical ``Eco-Tech/<repo>`` casing if present in the 165-repo scan."""
    text = str(repo_or_name or "").strip()
    if not text:
        return None
    lower = text.lower()
    if lower in _ECO_TECH_LOWER_INDEX:
        return _ECO_TECH_LOWER_INDEX[lower]
    return _ECO_TECH_LOWER_INDEX.get(f"eco-tech/{lower}") if "/" not in text else None


def _normalize_quant_tag(raw_tag: str) -> str:
    cleaned = str(raw_tag or "FLOAT").strip().upper().replace("-", "_")
    return _TAG_ALIASES.get(cleaned, cleaned if cleaned in _TAG_SPEC_TABLE else "FLOAT")


def _extract_native_format(base_config: Mapping[str, Any] | None) -> str:
    if not isinstance(base_config, Mapping):
        return "bf16"
    text_cfg = base_config.get("text_config")
    cfg = text_cfg if isinstance(text_cfg, Mapping) else base_config
    raw = str(cfg.get("torch_dtype") or cfg.get("dtype") or base_config.get("torch_dtype") or "bf16").lower()
    if "16" in raw and "bf" not in raw and "bfloat" not in raw:
        return "fp16"
    return "fp32" if "32" in raw else "bf16"


def _flatten_desc_mapping(desc_mapping: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    pattern_to_tag: dict[str, str] = {}
    kv_dtype = ""
    raw_kv = str(desc_mapping.get("kv_cache_type") or desc_mapping.get("fa_quant_type") or "").strip().upper()
    if raw_kv == "C8" or "INT8" in raw_kv:
        kv_dtype = "int8"
    elif "FP8" in raw_kv or "MXFP8" in raw_kv:
        kv_dtype = "fp8"
    for key, val in desc_mapping.items():
        key_str = str(key)
        if key_str.lower() in _METADATA_KEYS_TO_SKIP:
            continue
        upper_key = key_str.upper().replace("-", "_")
        if isinstance(val, (list, tuple)) and upper_key in _KNOWN_DESC_TAGS:
            for pattern in val:
                pattern_to_tag[str(pattern)] = upper_key
        elif isinstance(val, str):
            upper_val = val.strip().upper().replace("-", "_")
            if upper_val == "C8" or "kv_cache" in key_str.lower():
                kv_dtype = kv_dtype or "int8"
            else:
                pattern_to_tag[key_str] = upper_val
    return pattern_to_tag, kv_dtype


def _infer_arch_flags(keys: Sequence[str]) -> tuple[bool, bool, bool]:
    has_mla = any(any(t in k for t in ("kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj")) for k in keys)
    has_kda = any(any(t in k for t in ("f_a_proj", "f_b_proj", "b_proj", "q_conv1d", "k_conv1d", "v_conv1d")) for k in keys)
    has_gdn = any("linear_attn" in k for k in keys)
    return has_mla, has_kda, has_gdn


def _tensor_categories(tensor_name: str, *, has_mla: bool, has_kda: bool) -> tuple[str, ...]:
    """Map one checkpoint tensor key or pattern to simulator QuantSpec category keys."""
    lower = str(tensor_name or "").lower()
    if lower.startswith("mtp.") or ".mtp." in lower:
        return ("mtp",)
    if any(t in lower for t in ("vision_tower", "model.visual", "visual.")):
        return ("vision",)
    if "mm_projector" in lower or "merger" in lower:
        return ("projector",)
    if "lm_head" in lower:
        return ("lm_head",)
    if "embed_tokens" in lower or "word_embeddings" in lower:
        return ("embedding",)
    if "hyper_connection" in lower or "_res_proj" in lower:
        return ("mhc",)
    if ".ple." in lower or "ple_embedding" in lower:
        return ("ple",)
    if "conv1d" in lower:
        return ("conv_weight_f32",) if has_kda else ("conv_weight_bf16",)
    if any(t in lower for t in ("layernorm", "_norm.", ".norm.", "k_norm", "q_norm", "o_norm")):
        return ("norm",)
    if "shared_expert_gate" in lower or lower.endswith(".gate.weight") or ".router." in lower:
        return ("router",)
    if "routed_expert_down_proj" in lower or "routed_expert_up_proj" in lower:
        return ("latent_moe",)
    if ".experts." in lower:
        return ("routed_expert",)
    if "shared_experts" in lower or "shared_expert." in lower:
        return ("shared_expert",)
    if "linear_attn." in lower:
        if "out_proj" in lower:
            return ("gdn.o", "gdn")
        return ("gdn.beta_alpha", "gdn") if ("in_proj_a" in lower or "in_proj_b" in lower) else ("gdn.qkvz", "gdn")
    if ".indexer." in lower:
        if "weights_proj" in lower:
            return ("indexer.weights",)
        return ("indexer.wk",) if ".wk." in lower else ("indexer_fp8", "indexer.wq")
    return _attn_mlp_categories(lower, has_mla=has_mla, has_kda=has_kda)


def _attn_mlp_categories(lower: str, *, has_mla: bool, has_kda: bool) -> tuple[str, ...]:
    for token, cat in (
        ("kv_b_proj", "mla_absorb"), ("kv_a_proj_with_mqa", "mla.kv_a"),
        ("q_a_proj", "mla.q_a"), ("q_b_proj", "mla.q_b"),
        (".f_a_proj.", "kda.f_a"), (".f_b_proj.", "kda.f_b"), (".b_proj.", "kda.b"),
    ):
        if token in lower:
            return (cat,)
    if ".g_proj." in lower:
        return ("kda.g", "mla.g") if (has_mla and has_kda) else (("kda.g",) if has_kda else ("mla.g",))
    if ".self_attn.o_proj" in lower or ".attn.o_proj" in lower:
        return ("mla.o", "kda.o") if (has_mla and has_kda) else (("mla.o",) if has_mla else ("attn.o",))
    if any(t in lower for t in (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")):
        if has_kda:
            return ("kda.q",) if ".q_proj" in lower else (("kda.k",) if ".k_proj" in lower else ("kda.v",))
        return ("attn.qkv",)
    if ".mlp." in lower and any(t in lower for t in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")):
        return ("dense_mlp",)
    return ()


def _override_fields_for_tag(tag: str, native_format: str) -> dict[str, Any]:
    spec = _TAG_SPEC_TABLE[_normalize_quant_tag(tag)]
    if not spec["is_quantized"]:
        return {"weight_format": native_format, "activation_format": native_format}
    return {
        k: spec[k] for k in (
            "weight_format", "activation_format", "weight_block", "expert_block",
            "scale_format", "scale_bytes", "activation_group_size",
            "activation_scale_format", "activation_scale_pack",
        )
    }


def _pick_dominant_tag(tags: Sequence[str], default: str) -> str:
    counts: dict[str, int] = {}
    for tag in tags:
        norm = _normalize_quant_tag(tag)
        if norm != "FLOAT":
            counts[norm] = counts.get(norm, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if counts else default


def parse_quant_model_description(
    desc_mapping: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any] | None = None,
    checkpoint_repo: str = "",
    checkpoint_url: str = "",
) -> QuantSpec:
    """Parse any ``msmodelslim`` / Eco-Tech ``quant_model_description.json`` into ``QuantSpec``."""
    if not isinstance(desc_mapping, Mapping):
        raise ModelingError("quant_model_description_requires_mapping")
    native_format = _extract_native_format(base_config)
    pattern_to_tag, detected_kv = _flatten_desc_mapping(desc_mapping)
    has_mla, has_kda, _ = _infer_arch_flags(tuple(pattern_to_tag.keys()))
    cat_tags: dict[str, list[str]] = {}
    for pattern, raw_tag in pattern_to_tag.items():
        for cat in _tensor_categories(pattern, has_mla=has_mla, has_kda=has_kda):
            cat_tags.setdefault(cat, []).append(raw_tag)

    linear_pool = [
        t for c in ("attn.qkv", "attn.o", "mla.q_a", "mla.q_b", "mla.kv_a", "mla.o",
                    "kda.q", "kda.k", "kda.v", "dense_mlp", "shared_expert", "latent_moe", "indexer_fp8")
        for t in cat_tags.get(c, ())
    ]
    expert_pool = cat_tags.get("routed_expert", [])
    linear_tag = _pick_dominant_tag(linear_pool, _pick_dominant_tag(expert_pool, "FLOAT"))
    expert_tag = _pick_dominant_tag(expert_pool, linear_tag)
    lin_spec, exp_spec = _TAG_SPEC_TABLE[linear_tag], _TAG_SPEC_TABLE[expert_tag]

    exclusions = list(_STANDARD_UNQUANTIZED_EXCLUSIONS)
    for opt in ("mtp", "gdn", "shared_expert", "ple"):
        if cat_tags.get(opt) and all(_normalize_quant_tag(t) == "FLOAT" for t in cat_tags[opt]):
            if opt not in exclusions:
                exclusions.append(opt)

    overrides: dict[str, dict[str, Any]] = {}
    skip = set(exclusions) | {"norm", "router", "lm_head", "embedding", "vision", "projector"}
    for cat, tags_here in sorted(cat_tags.items()):
        if cat in skip:
            continue
        target = "FLOAT" if all(_normalize_quant_tag(t) == "FLOAT" for t in tags_here) else _pick_dominant_tag(tags_here, "FLOAT")
        if (cat == "routed_expert" and (target != linear_tag or exp_spec["weight_format"] == "mxfp4")) or target != linear_tag:
            overrides[cat] = _override_fields_for_tag(target, native_format)

    name = f"{linear_tag}+{expert_tag}" if (expert_tag != linear_tag and expert_tag != "FLOAT") else linear_tag
    kv_dtype = detected_kv or "bf16"
    source_meta = {
        "quant_method": name.lower(), "linear_tag": linear_tag,
        "expert_tag": expert_tag, "pattern_entries": str(len(pattern_to_tag)),
        **({"checkpoint": checkpoint_repo} if checkpoint_repo else {}),
        **({"checkpoint_url": checkpoint_url} if checkpoint_url else {}),
    }
    return QuantSpec(
        name=name,
        weight_format=lin_spec["weight_format"] if lin_spec["is_quantized"] else native_format,
        activation_format=lin_spec["activation_format"] if lin_spec["is_quantized"] else native_format,
        expert_weight_format=exp_spec["weight_format"] if exp_spec["is_quantized"] else native_format,
        weight_block=lin_spec["weight_block"],
        expert_block=exp_spec["expert_block"] or lin_spec["expert_block"] or MXFP_GROUP_SIZE,
        scale_format=lin_spec["scale_format"] if lin_spec["is_quantized"] else "fp32",
        scale_bytes=float(lin_spec["scale_bytes"] if lin_spec["is_quantized"] else FP32_SCALE_BYTES),
        kv_dtype=kv_dtype, index_dtype=kv_dtype if kv_dtype in {"int8", "fp8"} else "bf16",
        index_scale_dtype="fp16" if kv_dtype in {"int8", "fp8"} else "none",
        exclusions=tuple(exclusions),
        activation_group_size=int(lin_spec["activation_group_size"]),
        activation_scale_format=str(lin_spec["activation_scale_format"]),
        activation_scale_pack=int(lin_spec["activation_scale_pack"]),
        category_overrides=overrides, native_format=native_format,
        runtime_weight_scale_format=str(lin_spec.get("runtime_weight_scale_format", "")),
        runtime_weight_scale_bytes=float(lin_spec.get("runtime_weight_scale_bytes", 0.0)),
        runtime_activation_scale_format=str(lin_spec.get("runtime_activation_scale_format", "")),
        runtime_int8_weight_metadata=str(lin_spec.get("runtime_int8_weight_metadata", "")),
        source=merge_source(source_meta),
        notes=(f"parsed from quant_model_description ({len(pattern_to_tag)} entries): linear={linear_tag}, expert={expert_tag}, kv={kv_dtype}",),
    )


_PRESET_RECIPES: dict[str, dict[str, Any]] = {
    "w4a8_mxfp": {"name": "W8A8_MXFP8+W4A8_MXFP", "wf": "fp8_e4m3", "af": "fp8_e4m3", "ef": "mxfp4", "wb": (1, MXFP_GROUP_SIZE), "eb": MXFP_GROUP_SIZE, "sf": "fp8_e8m0", "sb": MXFP_SCALE_BYTES, "ags": MXFP_GROUP_SIZE, "asf": "fp8_e8m0", "asp": MXFP_SCALE_PACK, "excl": _STANDARD_UNQUANTIZED_EXCLUSIONS + ("mtp",)},
    "w8a8_mxfp8": {"name": "W8A8_MXFP8", "wf": "fp8_e4m3", "af": "fp8_e4m3", "ef": "fp8_e4m3", "wb": (1, MXFP_GROUP_SIZE), "eb": MXFP_GROUP_SIZE, "sf": "fp8_e8m0", "sb": MXFP_SCALE_BYTES, "ags": MXFP_GROUP_SIZE, "asf": "fp8_e8m0", "asp": MXFP_SCALE_PACK, "excl": _STANDARD_UNQUANTIZED_EXCLUSIONS + ("gdn", "mtp")},
    "w4a4_mxfp4": {"name": "W4A4_MXFP4", "wf": "mxfp4", "af": "fp8_e4m3", "ef": "mxfp4", "wb": (1, MXFP_GROUP_SIZE), "eb": MXFP_GROUP_SIZE, "sf": "fp8_e8m0", "sb": MXFP_SCALE_BYTES, "ags": MXFP_GROUP_SIZE, "asf": "fp8_e8m0", "asp": MXFP_SCALE_PACK, "excl": _STANDARD_UNQUANTIZED_EXCLUSIONS + ("mtp",)},
    "fp8_dynamic+mxfp4_experts": {"name": "fp8_dynamic+mxfp4_experts", "wf": "fp8_e4m3", "af": "fp8_e4m3", "ef": "mxfp4", "wb": (MXFP_GROUP_SIZE, MXFP_GROUP_SIZE), "eb": MXFP_GROUP_SIZE, "sf": "fp8_e8m0", "sb": MXFP_SCALE_BYTES, "ags": MXFP_GROUP_SIZE, "asf": "fp8_e8m0", "asp": MXFP_SCALE_PACK, "excl": _STANDARD_UNQUANTIZED_EXCLUSIONS},
    "w4a8_dynamic": {"name": "W4A8_DYNAMIC", "wf": "int8", "af": "int8", "ef": "int4", "wb": (1, 1), "eb": 1, "sf": "fp32", "sb": FP32_SCALE_BYTES, "ags": 0, "asf": "fp32", "asp": 1, "excl": _STANDARD_UNQUANTIZED_EXCLUSIONS + ("mtp",), **_INT8_RT},
    "w8a8_dynamic": {"name": "W8A8_DYNAMIC", "wf": "int8", "af": "int8", "ef": "int8", "wb": (1, 1), "eb": 1, "sf": "fp32", "sb": FP32_SCALE_BYTES, "ags": 0, "asf": "fp32", "asp": 1, "excl": _STANDARD_UNQUANTIZED_EXCLUSIONS + ("mtp",), **_INT8_RT},
}
_PRESET_KEY_ALIAS = {
    "w4a8_mxfp4": "w4a8_mxfp", "mxfp8": "w8a8_mxfp8", "mxfp4": "w4a4_mxfp4",
    "w4a8": "w4a8_dynamic", "w8a8": "w8a8_dynamic",
}


def _quant_spec_from_preset(
    preset: str, *, kv_dtype: str = "bf16", index_dtype: str = "bf16",
    checkpoint_repo: str = "", checkpoint_url: str = "", base_config: Mapping[str, Any] | None = None,
) -> QuantSpec:
    native = _extract_native_format(base_config)
    key = str(preset or "bf16").strip().lower().replace("-", "_")
    key = _PRESET_KEY_ALIAS.get(key, key)
    src = merge_source({
        "quant_method": key,
        **({"checkpoint": checkpoint_repo} if checkpoint_repo else {}),
        **({"checkpoint_url": checkpoint_url} if checkpoint_url else {}),
    })
    rec = _PRESET_RECIPES.get(key)
    if rec is None:
        return QuantSpec(
            name=native, weight_format=native, activation_format=native,
            expert_weight_format=native, kv_dtype=kv_dtype or native,
            index_dtype=index_dtype or native, index_scale_dtype="none",
            exclusions=(), native_format=native, source=src,
        )
    idx_dt = index_dtype or ("int8" if key == "fp8_dynamic+mxfp4_experts" else "bf16")
    return QuantSpec(
        name=rec["name"], weight_format=rec["wf"], activation_format=rec["af"],
        expert_weight_format=rec["ef"], weight_block=rec["wb"], expert_block=rec["eb"],
        scale_format=rec["sf"], scale_bytes=float(rec["sb"]),
        kv_dtype=kv_dtype, index_dtype=idx_dt,
        index_scale_dtype="fp16" if idx_dt in {"int8", "fp8"} else "none",
        activation_group_size=int(rec["ags"]), activation_scale_format=str(rec["asf"]),
        activation_scale_pack=int(rec["asp"]), exclusions=rec["excl"],
        native_format=native,
        runtime_weight_scale_format=str(rec.get("runtime_weight_scale_format", "")),
        runtime_weight_scale_bytes=float(rec.get("runtime_weight_scale_bytes", 0.0)),
        runtime_activation_scale_format=str(rec.get("runtime_activation_scale_format", "")),
        runtime_int8_weight_metadata=str(rec.get("runtime_int8_weight_metadata", "")),
        source=src,
    )


def _match_weight_pattern(
    tensor_name: str, pattern_to_tag: Mapping[str, str],
) -> tuple[str, str] | None:
    if tensor_name in pattern_to_tag:
        return tensor_name, pattern_to_tag[tensor_name]
    normalized = _NUMERIC_SEGMENT_RE.sub("*", tensor_name)
    if normalized in pattern_to_tag:
        return normalized, pattern_to_tag[normalized]
    best_pattern, best_tag, best_score = "", "", -1
    for pattern, tag in pattern_to_tag.items():
        if "*" in pattern and fnmatchcase(tensor_name, pattern):
            score = len(pattern.replace("*", ""))
            if score > best_score:
                best_pattern, best_tag, best_score = pattern, tag, score
    return (best_pattern, best_tag) if best_score >= 0 else None


def _weight_quant_from_tag(
    raw_tag: str, *, matched_pattern: str, category: str, native_format: str,
) -> WeightTensorQuant:
    spec = _TAG_SPEC_TABLE[_normalize_quant_tag(raw_tag)]
    is_q = bool(spec["is_quantized"])
    wf = str(spec["weight_format"]) if is_q else native_format
    af = str(spec["activation_format"]) if is_q else native_format
    ledger = str(spec["ledger_format"]) if is_q else ("FP32" if native_format == "fp32" else "BF16")
    group = int(spec["expert_block"] if wf in {"mxfp4", "int4"} else spec["weight_block"][1])
    return WeightTensorQuant(
        raw_tag=str(raw_tag).upper(), weight_format=wf, activation_format=af,
        ledger_format=ledger, scale_format=str(spec["scale_format"]),
        group_size=group if is_q else 0, is_quantized=is_q,
        matched_pattern=matched_pattern, category=category,
    )


def _classify_from_quant_spec(
    role_name: str, quant: QuantSpec, *, native_format: str,
) -> WeightTensorQuant:
    resolved = quant.for_category(role_name)
    wf = str(resolved.format_for(role_name) or native_format).lower()
    af = str(resolved.activation_format or native_format).lower()
    sf = str(resolved.scale_format or "none").lower()
    mx_group = int(resolved.expert_block or resolved.weight_block[1] or 0)
    if wf in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        return WeightTensorQuant("FLOAT", wf, wf, "FP32" if "32" in wf else "BF16", "none", 0, False, f"quant_spec:{role_name}", role_name)
    if wf in {"mxfp4", "fp4", "fp4_e2m1"}:
        return WeightTensorQuant("W4A8_MXFP", "mxfp4", af, "MXFP4", "fp8_e8m0", mx_group or MXFP_GROUP_SIZE, True, f"quant_spec:{role_name}", role_name)
    if wf in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
        is_mx = sf in {"fp8_e8m0", "ue8m0"} and mx_group == MXFP_GROUP_SIZE
        return WeightTensorQuant("W8A8_MXFP8" if is_mx else "FP8_DYNAMIC", "fp8_e4m3", af, "MXFP8" if is_mx else "FP8", sf, mx_group or (MXFP_GROUP_SIZE if is_mx else FP8_BLOCK_SIZE), True, f"quant_spec:{role_name}", role_name)
    if wf in {"int4", "w4a8"}:
        return WeightTensorQuant("W4A8_DYNAMIC", "int4", "int8", "W4A8", "fp32", 1, True, f"quant_spec:{role_name}", role_name)
    return WeightTensorQuant("W8A8_DYNAMIC", "int8", "int8", "W8A8", "fp32", 1, True, f"quant_spec:{role_name}", role_name)


def classify_weight_tensor_quant(
    tensor_or_role_name: str,
    desc_or_quant_spec: Mapping[str, Any] | NpuCheckpointSpec | QuantSpec,
    *,
    native_format: str = "bf16",
) -> WeightTensorQuant:
    """Resolve the exact weight/activation/ledger dtype for one tensor key or role name."""
    name = str(tensor_or_role_name or "").strip()
    if isinstance(desc_or_quant_spec, NpuCheckpointSpec):
        if desc_or_quant_spec.weight_patterns:
            return classify_weight_tensor_quant(name, desc_or_quant_spec.weight_pattern_map(), native_format=native_format)
        return _classify_from_quant_spec(name, desc_or_quant_spec.build_quant_spec(), native_format=native_format)
    if isinstance(desc_or_quant_spec, QuantSpec):
        cats = _tensor_categories(name, has_mla=True, has_kda=True)
        return _classify_from_quant_spec(cats[0] if cats else name, desc_or_quant_spec, native_format=native_format)
    pattern_to_tag, _ = _flatten_desc_mapping(desc_or_quant_spec)
    has_mla, has_kda, _ = _infer_arch_flags(tuple(pattern_to_tag.keys()))
    matched = _match_weight_pattern(name, pattern_to_tag)
    if matched is not None:
        pat, raw_tag = matched
        cats = _tensor_categories(name, has_mla=has_mla, has_kda=has_kda)
        return _weight_quant_from_tag(raw_tag, matched_pattern=pat, category=cats[0] if cats else name, native_format=native_format)
    parsed = parse_quant_model_description(desc_or_quant_spec)
    cats = _tensor_categories(name, has_mla=has_mla, has_kda=has_kda)
    return _classify_from_quant_spec(cats[0] if cats else name, parsed, native_format=native_format)


def checkpoint_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Return isolated checkpoint metadata cache root honoring XDG/MINDIE/HOME env vars."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "mindie-sim" / "checkpoints"
    mindie_data = os.environ.get("MINDIE_DATA_HOME", "").strip()
    if mindie_data:
        return Path(mindie_data).expanduser() / "cache" / "mindie-sim" / "checkpoints"
    home = os.environ.get("HOME", "").strip()
    return (Path(home).expanduser() if home else Path.home()) / ".cache" / "mindie-sim" / "checkpoints"


def _cache_file_for(identifier: str, cache_root: Path) -> Path:
    safe = _SAFE_CACHE_RE.sub("_", identifier.strip().strip("/"))
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:CACHE_HASH_HEX_LEN]
    stem = f"{safe[:CACHE_STEM_MAX_CHARS]}_{digest}" if safe else digest
    return cache_root / f"{stem}.json"


def _read_local_json_bounded(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    if path.stat().st_size > MAX_LOCAL_METADATA_BYTES:
        raise ModelingError(f"checkpoint_metadata_exceeds_50MiB:{path.name}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except (OSError, ValueError) as exc:
        raise ModelingError(f"checkpoint_metadata_unreadable:{path.name}:{type(exc).__name__}") from exc
    return dict(raw) if isinstance(raw, Mapping) else {}


def _write_cache_atomic(target: Path, payload: Mapping[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(target.parent),
            prefix=f".{target.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, target)
    except TimeoutError:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    except OSError:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _fetch_json_url(
    url: str, *, timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch one JSON header with strict <= 256 KiB cap and TimeoutError propagation."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "mindie-sim-checkpoint-reader/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as response:
            blob = response.read(MAX_REMOTE_METADATA_BYTES + 1)
    except TimeoutError:
        raise
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise exc.reason from exc
        return None, f"fetch_failed:{url}:{type(exc).__name__}"
    except OSError as exc:
        return None, f"fetch_failed:{url}:{type(exc).__name__}"
    if len(blob) > MAX_REMOTE_METADATA_BYTES:
        return None, f"metadata_response_exceeds_256KiB:{url}"
    try:
        parsed = json.loads(blob.decode("utf-8"))
    except ValueError as exc:
        return None, f"metadata_json_invalid:{url}:{type(exc).__name__}"
    return (dict(parsed) if isinstance(parsed, Mapping) else None), None


def _parse_remote_target(spec_or_url: str) -> tuple[str, str, tuple[str, ...]]:
    raw = spec_or_url.strip()
    if raw.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(raw)
        host = (parsed.netloc or "").lower()
        parts = [p for p in parsed.path.split("/") if p]
        if "modelers.cn" in host:
            idx = 1 if parts and parts[0] == "models" else 0
            repo = "/".join(parts[idx : idx + MIN_REPO_PATH_SEGMENTS])
            return "modelers", repo, (f"{MODELERS_RAW_BASE}/{repo}/main/media",)
        if "modelscope.cn" in host:
            idx = 1 if parts and parts[0] == "models" else 0
            repo = "/".join(parts[idx : idx + MIN_REPO_PATH_SEGMENTS])
            return "modelscope", repo, (f"{MODELSCOPE_RAW_BASE}/{repo}/resolve/master",)
        repo = "/".join(parts[:MIN_REPO_PATH_SEGMENTS])
        return "huggingface", repo, (f"{HF_PAGE_BASE}/{repo}/resolve/main",)
    repo = lookup_eco_tech_repo(raw) or raw
    owner = repo.split("/", 1)[0].lower()
    if owner == "eco-tech":
        return "modelers", repo, (
            f"{MODELERS_RAW_BASE}/{repo}/main/media",
            f"{MODELSCOPE_RAW_BASE}/{repo}/resolve/master",
        )
    if owner in {"vllm-ascend", "inferact", "minimax"}:
        return "modelscope", repo, (
            f"{MODELSCOPE_RAW_BASE}/{repo}/resolve/master",
            f"{HF_PAGE_BASE}/{repo}/resolve/main",
        )
    return "huggingface", repo, (
        f"{MODELSCOPE_RAW_BASE}/{repo}/resolve/master",
        f"{HF_PAGE_BASE}/{repo}/resolve/main",
    )


def _infer_offline_spec_for_repo(
    repo_id: str, hw_family: str,
) -> tuple[NpuCheckpointSpec | None, QuantSpec | None, tuple[str, ...]]:
    hw = normalize_hw_family(hw_family)
    for (_m, _hw), spec in DEFAULT_NPU_CHECKPOINTS.items():
        if spec.repo_id.lower() == repo_id.lower():
            return spec, spec.build_quant_spec(), validate_quant_for_hardware(spec.quant_preset, hw, kv_cache_dtype=spec.kv_cache_dtype)
    if repo_id in SCANNED_QUANT_PATTERNS:
        m_url, ms_url, hf_url = _repo_urls(repo_id)
        qspec = parse_quant_model_description(SCANNED_QUANT_PATTERNS[repo_id], checkpoint_repo=repo_id, checkpoint_url=m_url or ms_url or hf_url)
        return None, qspec, validate_quant_for_hardware(qspec.name, hw, kv_cache_dtype=qspec.kv_dtype)
    default_ckpt = resolve_default_checkpoint(normalize_model_key(repo_id), hw_family=hw)
    if default_ckpt is not None:
        return default_ckpt, default_ckpt.build_quant_spec(), ()
    lower = repo_id.lower()
    kv = "fp8" if ("c8" in lower and "mxfp" in lower) else ("int8" if "c8" in lower else "bf16")
    preset = (
        "w4a4_mxfp4" if ("w4a4" in lower and "mxfp" in lower) else
        "w4a8_mxfp" if ("w4a8" in lower and "mxfp" in lower) else
        "w8a8_mxfp8" if "mxfp8" in lower else
        "w4a8_dynamic" if "w4a8" in lower else
        "w8a8_dynamic" if "w8a8" in lower else "bf16"
    )
    m_url, ms_url, hf_url = _repo_urls(repo_id)
    qspec = _quant_spec_from_preset(preset, kv_dtype=kv, checkpoint_repo=repo_id, checkpoint_url=m_url or ms_url or hf_url)
    return None, qspec, validate_quant_for_hardware(preset, hw, kv_cache_dtype=kv)


def _resolve_local_checkpoint(path: Path, hw_family: str) -> CheckpointMetadataResult:
    hw = normalize_hw_family(hw_family)
    base_dir = path if path.is_dir() else path.parent
    cfg = _read_local_json_bounded(path if path.is_file() and path.name == "config.json" else base_dir / "config.json")
    desc = _read_local_json_bounded(path if path.is_file() and "quant_model_description" in path.name else base_dir / "quant_model_description.json")
    qcfg = _read_local_json_bounded(path if path.is_file() and path.name == "quantize_config.json" else base_dir / "quantize_config.json")
    repo_label = base_dir.name
    canon = normalize_model_key(repo_label)
    ckpt = resolve_default_checkpoint(canon, hw_family=hw)
    qspec = parse_quant_model_description(desc, base_config=cfg or None, checkpoint_repo=repo_label) if desc else (ckpt.build_quant_spec(base_config=cfg or None) if ckpt else None)
    warns = validate_quant_for_hardware(qspec.name, hw, kv_cache_dtype=qspec.kv_dtype) if qspec is not None else ()
    return CheckpointMetadataResult(
        spec_or_url=str(path), source_kind="local_dir" if path.is_dir() else "local_file",
        repo_id=repo_label, canonical_model=canon, hw_family=hw,
        modelers_url="", modelscope_url="", hf_url="", config=cfg,
        quant_model_description=desc, quantize_config=qcfg,
        quant_spec=qspec, checkpoint=ckpt, warnings=warns,
    )


def _build_remote_result(
    raw: str, source_kind: str, repo_id: str, hw: str,
    cfg: Mapping[str, Any], desc: Mapping[str, Any], qcfg: Mapping[str, Any],
    *, from_cache: bool = False, offline_fallback: bool = False, warnings: Sequence[str] = (),
) -> CheckpointMetadataResult:
    m_url, ms_url, hf_url = _repo_urls(repo_id)
    ckpt, fallback_q, isa_warns = _infer_offline_spec_for_repo(repo_id, hw)
    qspec = (
        parse_quant_model_description(desc, base_config=cfg or None, checkpoint_repo=repo_id, checkpoint_url=m_url or ms_url or hf_url)
        if desc else fallback_q
    )
    all_warns = list(warnings)
    all_warns.extend(validate_quant_for_hardware(qspec.name, hw, kv_cache_dtype=qspec.kv_dtype) if qspec is not None else isa_warns)
    return CheckpointMetadataResult(
        spec_or_url=raw, source_kind=source_kind, repo_id=repo_id,
        canonical_model=normalize_model_key(repo_id), hw_family=hw,
        modelers_url=m_url, modelscope_url=ms_url, hf_url=hf_url,
        config=dict(cfg), quant_model_description=dict(desc), quantize_config=dict(qcfg),
        quant_spec=qspec, checkpoint=ckpt, from_cache=from_cache,
        offline_fallback=offline_fallback, warnings=tuple(dict.fromkeys(all_warns)),
    )


def resolve_checkpoint_metadata(
    spec_or_url: str | Path,
    *,
    hw_family: str = "a5",
    timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
    allow_network: bool = True,
    cache_dir: str | Path | None = None,
) -> CheckpointMetadataResult:
    """Resolve checkpoint JSON metadata from local path, URL, repo ID, cache, or baked defaults."""
    hw = normalize_hw_family(hw_family)
    if isinstance(spec_or_url, Path):
        return _resolve_local_checkpoint(spec_or_url.expanduser(), hw)
    raw = str(spec_or_url or "").strip()
    if not raw:
        raise ModelingError("checkpoint_spec_empty")
    local_probe = Path(raw).expanduser()
    if local_probe.exists():
        return _resolve_local_checkpoint(local_probe, hw)
    if not (raw.startswith(("http://", "https://")) or "/" in raw):
        ckpt = resolve_default_checkpoint(raw, hw_family=hw)
        if ckpt is None:
            raise ModelingError(f"checkpoint_model_unknown:{raw}")
        return CheckpointMetadataResult(
            spec_or_url=raw, source_kind="builtin_registry", repo_id=ckpt.repo_id,
            canonical_model=ckpt.canonical_model, hw_family=ckpt.hw_family,
            modelers_url=ckpt.modelers_url, modelscope_url=ckpt.modelscope_url, hf_url=ckpt.hf_url,
            config={}, quant_model_description=ckpt.weight_pattern_map(), quantize_config={},
            quant_spec=ckpt.build_quant_spec(), checkpoint=ckpt,
        )

    source_kind, repo_id, raw_bases = _parse_remote_target(raw)
    cache_path = _cache_file_for(repo_id or raw, checkpoint_cache_dir(cache_dir))
    if cache_path.is_file():
        cached = _read_local_json_bounded(cache_path)
        if cached:
            return _build_remote_result(
                raw, "cache", repo_id, hw, cached.get("config") or {},
                cached.get("quant_model_description") or {}, cached.get("quantize_config") or {},
                from_cache=True,
            )

    warnings: list[str] = []
    fetched: dict[str, dict[str, Any]] = {}
    if allow_network:
        for base in raw_bases:
            for fname in METADATA_FILENAMES:
                if fname not in fetched:
                    data, err = _fetch_json_url(f"{base}/{fname}", timeout_s=timeout_s)
                    if data is not None:
                        fetched[fname] = data
                    elif err is not None:
                        warnings.append(err)
            if fetched:
                break

    if fetched:
        cfg, desc, qcfg = fetched.get("config.json", {}), fetched.get("quant_model_description.json", {}), fetched.get("quantize_config.json", {})
        _write_cache_atomic(cache_path, {"repo_id": repo_id, "config": cfg, "quant_model_description": desc, "quantize_config": qcfg})
        return _build_remote_result(raw, source_kind, repo_id, hw, cfg, desc, qcfg, warnings=warnings)

    warnings.append("network_disabled_using_builtin_registry" if not allow_network else "remote_metadata_unreachable_using_builtin_registry")
    ckpt, _q, _w = _infer_offline_spec_for_repo(repo_id, hw)
    return _build_remote_result(
        raw, "builtin_registry", repo_id, hw, {},
        ckpt.weight_pattern_map() if ckpt else {}, {},
        offline_fallback=True, warnings=warnings,
    )


__all__ = [
    "DEFAULT_HTTP_TIMEOUT_S", "MAX_REMOTE_METADATA_BYTES", "MAX_LOCAL_METADATA_BYTES",
    "A2_A3_INT_ONLY_FAMILIES", "A2_A3_ALLOWED_WEIGHT_PRESETS", "A2_A3_ALLOWED_KV_DTYPES",
    "ECO_TECH_REPOS", "SCANNED_QUANT_PATTERNS", "DEFAULT_NPU_CHECKPOINTS",
    "WeightTensorQuant", "NpuCheckpointSpec", "CheckpointMetadataResult",
    "normalize_hw_family", "normalize_model_key", "validate_quant_for_hardware",
    "rank_checkpoint_candidates", "msmodelslim_quarot_guidance",
    "resolve_default_checkpoint", "select_checkpoint_for_capacity",
    "list_model_checkpoints", "lookup_eco_tech_repo",
    "parse_quant_model_description", "classify_weight_tensor_quant",
    "checkpoint_cache_dir", "resolve_checkpoint_metadata",
]
