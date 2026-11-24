"""vLLM-Ascend runtime profiles, manifest loader, ISA guards, and MC2/MSO policy.

[Role] Own the packaged ``vllm-ascend`` plugin manifest (``manifest.json``),
  model/hardware profile defaults, curated default checkpoints, strict
  A2/A3 (INT8/INT4/C8) vs A5 (MXFP8/MXFP4/FP8) ISA guards, shipping collective
  policies, workload-domain classification, and MC2-vs-MSO mutual exclusion.
[Invariants] Strictly LLM/VLM scope (Diffusion/DiT models classify as
  ``"diffusion"`` and are rejected at the plugin boundary). Zero imports from
  ``mindie_sim.export`` or ``mindie_sim.compose`` (AG010). A2/A3 never admit
  MXFP/FP8 weights or FP8 KV cache; A5 never emits legacy PCIe/RoCE env flags.
[Timeouts & Locks] Single bounded read (< 50 MiB) of packaged ``manifest.json``
  cached via ``lru_cache``; no network I/O or locks.
[Side-Effect Policy] Read-only over packaged manifest; returns fresh dicts and
  immutable tuples without mutating caller inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.stack import ManifestBinding, StackTrack, WorkloadDomain

VLLM_ASCEND_STACK_ID = "vllm-ascend"
VLLM_ASCEND_REPO = "vllm-project/vllm-ascend"
MANIFEST_SCHEMA = "mindie-sim.stack-manifest/1"
MANIFEST_PATH = Path(__file__).resolve().with_name("manifest.json")
MAX_MANIFEST_BYTES = 50 * 1024 * 1024

_DOC_ROOT = "docs/source/tutorials/models/"
_BASE_ENV = {"OMP_PROC_BIND": "false"}
_MLAPO_BASE_ENV = {**_BASE_ENV, "VLLM_ASCEND_ENABLE_MLAPO": "1"}

FUSED_MC2_DISABLED = 0
FUSED_MC2_DISPATCH_FFN_COMBINE = 1
FUSED_MC2_MEGAMOE_ROUTED = 2
FUSED_MC2_MEGAMOE_SHARED = 3

NO_MTP_PROPOSAL_TOKENS = 0
SINGLE_MTP_PROPOSAL_TOKEN = 1
MTP_STANDARD_PROPOSAL_TOKENS = 3
MTP_DSPARK_PROPOSAL_TOKENS = 5
MTP_KIMI_K3_PROPOSAL_TOKENS = 7
PAIRED_SFA_DOC_BLOCK_SIZE = 544

_A5_CCU_SCHED_MODELS = frozenset({"kimi-k3.1", "deepseek-v4.1-flash"})
_A3_PREFILL_UNSET_HCCL_MODELS = frozenset({"kimi-k3.1"})
_A5_FAMILY_TOKENS = ("a5", "950", "9582", "9581", "9579", "958b", "9599", "3510", "arch35")
_A3_FAMILY_TOKENS = ("a3", "910c", "910_93", "9392", "9391", "9382", "9362")
_A2_FAMILY_TOKENS = ("a2", "910b", "310p", "300i")
_NVIDIA_TOKENS = ("nvidia", "cuda", "h100", "h200", "b200", "gb200", "a100", "hopper", "blackwell")

A2_A3_INT_ONLY_FAMILIES = frozenset({"a2", "a3"})
A2_A3_ALLOWED_KV_DTYPES = frozenset({"bf16", "fp16", "int8", "c8", "auto", ""})
A5_ONLY_QUANT_TOKENS = ("mxfp", "fp8", "fp4")

_DIFFUSION_ARCH_TOKENS = ("dit", "unet", "diffusion", "hunyuan_video", "wan2", "flux", "sdxl", "sd3")
_ROLE_SCOPED_TUTORIAL_FLAGS = ("kv_cache_dtype", "indexer_kv_dtype", "enable_fused_mc2")
_SUFFIX_STRIP_RE = re.compile(
    r"(?:[-_](?:w\d+a\d+(?:c\d+)?(?:s|sc)?|mxfp\d*|mxfp|fp8|bf16|mtp|floatmtp"
    r"|quarot|laos|pdmix|nopdmix|xllm|310p?|vllm|mindie|exp|terminus|speciale"
    r"|0626|0731|0813|1017|0528|0324|0905|2507|it|instruct|pro))+$"
)

_MODEL_ALIASES: dict[str, str] = {
    "deepseek-v3.2-prune-3l": "deepseek-v3.2",
    "deepseek-v4-pro": "deepseek-v4-pro-0813",
    "deepseek-v4-flash": "deepseek-v4-flash-0731",
    "deepseek-v4.1": "deepseek-v4.1-flash",
    "kimi-k3.1-pro": "kimi-k3.1",
    "kimi-k3.1-pro-w4a8": "kimi-k3.1",
    "kimi-k3.1-pro-w4a8-mxfp": "kimi-k3.1",
    "kimi-k3-w4a8": "kimi-k3",
    "qwen3-30b": "qwen3-30b-a3b",
    "qwen3.8-2.4t": "qwen3.8-2.4t-a95b",
    "meta-llama/meta-llama-3.1-8b-instruct": "llama-3.1-8b",
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b",
    "llama-3.1-8b-instruct": "llama-3.1-8b",
    "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b",
    "llama3": "llama-3",
    "qwen/qwen3-8b": "qwen3-8b",
    "qwen/qwen3-14b": "qwen3-14b",
    "qwen/qwen3-32b": "qwen3-32b",
    "qwen/qwen3-30b-a3b": "qwen3-30b-a3b",
}

#: Curated default checkpoint repository per (canonical_model, hw_family).
#: Strict A2/A3 INT8/INT4/C8 vs A5 MXFP8/MXFP4/FP8 ISA separation with QuaRot priority.
DEFAULT_CHECKPOINTS_BY_MODEL_HW: dict[tuple[str, str], str] = {
    ("kimi-k3.1", "a5"): "Eco-Tech/Kimi-K3.1-Pro-w4a8-mxfp",
    ("kimi-k3.1", "a3"): "Eco-Tech/Kimi-K3.1-Pro-w4a8",
    ("kimi-k3.1", "a2"): "Eco-Tech/Kimi-K3.1-Pro-w4a8",
    ("kimi-k3", "a5"): "Eco-Tech/Kimi-K3-w4a8",
    ("kimi-k3", "a3"): "Eco-Tech/Kimi-K3-w4a8",
    ("kimi-k3", "a2"): "Eco-Tech/Kimi-K3-w4a8",
    ("kimi-k3-dspark", "a5"): "Inferact/Kimi-K3-DSpark-Block5",
    ("kimi-k3-dspark", "a3"): "Eco-Tech/Kimi-K3-DSpark-MLA-NPU",
    ("kimi-k3-dspark", "a2"): "Eco-Tech/Kimi-K3-DSpark-MLA-NPU",
    ("kimi-k2.6", "a5"): "Eco-Tech/Kimi-K2.6-w4a8",
    ("kimi-k2.6", "a3"): "Eco-Tech/Kimi-K2.6-w4a8",
    ("kimi-k2.6", "a2"): "Eco-Tech/Kimi-K2.6-w4a8",
    ("kimi-k2.5", "a5"): "Eco-Tech/Kimi-K2.5-w4a8",
    ("kimi-k2.5", "a3"): "Eco-Tech/Kimi-K2.5-w4a8",
    ("kimi-k2.5", "a2"): "Eco-Tech/Kimi-K2.5-w4a8",
    ("kimi-k2", "a5"): "Eco-Tech/KIMI-k2-Instruct-0905-W8A8-QuaRot",
    ("kimi-k2", "a3"): "Eco-Tech/KIMI-k2-Instruct-0905-W4A8-QuaRot",
    ("kimi-k2", "a2"): "Eco-Tech/KIMI-k2-Instruct-0905-W4A8-QuaRot",
    ("qwen3.8-flash-next", "a5"): "Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp",
    ("qwen3.8-flash-next", "a3"): "Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp",
    ("qwen3.8-flash-next", "a2"): "Eco-Tech/Qwen3.8-Flash-Next-w4a8",
    ("qwen3.8-27b", "a5"): "Eco-Tech/Qwen3.8-27B-w8a8c8-mxfp8",
    ("qwen3.8-27b", "a3"): "Eco-Tech/Qwen3.8-27B-w8a8",
    ("qwen3.8-27b", "a2"): "Eco-Tech/Qwen3.8-27B-w8a8",
    ("qwen3.8-2.4t-a95b", "a5"): "Eco-Tech/Qwen3.8-2.4T-A95B-w4a8",
    ("qwen3.8-2.4t-a95b", "a3"): "Eco-Tech/Qwen3.8-2.4T-A95B-w4a8",
    ("qwen3.8-2.4t-a95b", "a2"): "Eco-Tech/Qwen3.8-2.4T-A95B-w4a8",
    ("qwen3.6-35b-a3b", "a5"): "Eco-Tech/Qwen3.6-35B-A3B-w4a8-mxfp",
    ("qwen3.6-35b-a3b", "a3"): "Eco-Tech/Qwen3.6-35B-A3B-w8a8",
    ("qwen3.6-35b-a3b", "a2"): "Eco-Tech/Qwen3.6-35B-A3B-w8a8",
    ("qwen3.6-27b", "a5"): "Eco-Tech/Qwen3.6-27B-w8a8c8-mxfp8",
    ("qwen3.6-27b", "a3"): "Eco-Tech/Qwen3.6-27B-w8a8",
    ("qwen3.6-27b", "a2"): "Eco-Tech/Qwen3.6-27B-w8a8",
    ("qwen3.5-397b-a17b", "a5"): "Eco-Tech/Qwen3.5-397B-A17B-w4a8c8-mxfp",
    ("qwen3.5-397b-a17b", "a3"): "Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp",
    ("qwen3.5-397b-a17b", "a2"): "Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp",
    ("qwen3.5-122b-a10b", "a5"): "Eco-Tech/Qwen3.5-122B-A10B-w8a8-mtp",
    ("qwen3.5-122b-a10b", "a3"): "Eco-Tech/Qwen3.5-122B-A10B-w8a8-mtp",
    ("qwen3.5-122b-a10b", "a2"): "Eco-Tech/Qwen3.5-122B-A10B-w8a8-mtp",
    ("qwen3.5-35b-a3b", "a5"): "Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp",
    ("qwen3.5-35b-a3b", "a3"): "Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp",
    ("qwen3.5-35b-a3b", "a2"): "Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp",
    ("qwen3.5-27b", "a5"): "Eco-Tech/Qwen3.5-27B-w8a8-mtp",
    ("qwen3.5-27b", "a3"): "Eco-Tech/Qwen3.5-27B-w8a8-mtp",
    ("qwen3.5-27b", "a2"): "Eco-Tech/Qwen3.5-27B-w8a8-mtp",
    ("qwen3-next-80b-a3b", "a5"): "Eco-Tech/Qwen3-Next-80B-A3B-Instruct-w8a8-mtp",
    ("qwen3-next-80b-a3b", "a3"): "Eco-Tech/Qwen3-Next-80B-A3B-Instruct-w8a8-mtp",
    ("qwen3-next-80b-a3b", "a2"): "Eco-Tech/Qwen3-Next-80B-A3B-Instruct-w8a8-mtp",
    ("qwen3-235b-a22b", "a5"): "Eco-Tech/Qwen3-235B-A22B-Instruct-2507-w8a8c8-QuaRot",
    ("qwen3-235b-a22b", "a3"): "Eco-Tech/Qwen3-235B-A22B-Instruct-2507-w8a8c8-QuaRot",
    ("qwen3-235b-a22b", "a2"): "Eco-Tech/Qwen3-235B-A22B-Instruct-2507-w8a8c8-QuaRot",
    ("qwen3-30b-a3b", "a5"): "Eco-Tech/Qwen3-30B-A3B-w8a8",
    ("qwen3-30b-a3b", "a3"): "Eco-Tech/Qwen3-30B-A3B-w8a8",
    ("qwen3-30b-a3b", "a2"): "Eco-Tech/Qwen3-30B-A3B-w8a8",
    ("qwen3-32b", "a5"): "Eco-Tech/Qwen3-32B-w8a8c8",
    ("qwen3-32b", "a3"): "Eco-Tech/Qwen3-32B-w8a8c8",
    ("qwen3-32b", "a2"): "Eco-Tech/Qwen3-32B-w8a8c8",
    ("deepseek-v4.1-flash", "a5"): "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    ("deepseek-v4.1-flash", "a3"): "Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8",
    ("deepseek-v4.1-flash", "a2"): "Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8",
    ("deepseek-v4-pro-0813", "a5"): "deepseek-ai/DeepSeek-V4-Pro-0813",
    ("deepseek-v4-pro-0813", "a3"): "Eco-Tech/DeepSeek-V4-Pro-0813-w4a8",
    ("deepseek-v4-pro-0813", "a2"): "Eco-Tech/DeepSeek-V4-Pro-0813-w4a8",
    ("deepseek-v4-flash-0731", "a5"): "deepseek-ai/DeepSeek-V4-Flash-0731",
    ("deepseek-v4-flash-0731", "a3"): "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
    ("deepseek-v4-flash-0731", "a2"): "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp",
    ("deepseek-v3.2", "a5"): "Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot",
    ("deepseek-v3.2", "a3"): "Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot",
    ("deepseek-v3.2", "a2"): "Eco-Tech/DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot",
    ("deepseek-v3.1", "a5"): "Eco-Tech/DeepSeek-V3.1-Terminus-w8a8c8-mtp-QuaRot",
    ("deepseek-v3.1", "a3"): "Eco-Tech/DeepSeek-V3.1-Terminus-w8a8c8-mtp-QuaRot",
    ("deepseek-v3.1", "a2"): "Eco-Tech/DeepSeek-V3.1-Terminus-w4a8c8-mtp-QuaRot",
    ("deepseek-r1", "a5"): "Eco-Tech/DeepSeek-R1-0528-w8a8c8-mtp-QuaRot",
    ("deepseek-r1", "a3"): "Eco-Tech/DeepSeek-R1-0528-w8a8c8-mtp-QuaRot",
    ("deepseek-r1", "a2"): "Eco-Tech/DeepSeek-R1-0528-w4a8c8-mtp-QuaRot",
    ("deepseek-v3", "a5"): "Eco-Tech/DeepSeek-V3-0324-w8a8c8-mtp-QuaRot",
    ("deepseek-v3", "a3"): "Eco-Tech/DeepSeek-V3-0324-w8a8c8-mtp-QuaRot",
    ("deepseek-v3", "a2"): "Eco-Tech/DeepSeek-V3-0324-w4a8c8-mtp-QuaRot",
    ("glm-5.3", "a5"): "Eco-Tech/GLM-5.3-W8A8C8-mxfp8",
    ("glm-5.3", "a3"): "Eco-Tech/GLM-5.3-w8a8c8",
    ("glm-5.3", "a2"): "Eco-Tech/GLM-5.3-w8a8c8",
    ("glm-5.3-flash", "a5"): "Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8",
    ("glm-5.3-flash", "a3"): "Eco-Tech/GLM-5.3-Flash-w8a8",
    ("glm-5.3-flash", "a2"): "Eco-Tech/GLM-5.3-Flash-w8a8",
    ("glm-5.2", "a5"): "Eco-Tech/GLM-5.2-w4a4c8-mxfp4",
    ("glm-5.2", "a3"): "Eco-Tech/GLM-5.2-w8a8c8",
    ("glm-5.2", "a2"): "Eco-Tech/GLM-5.2-w4a8c8",
    ("glm-5.1", "a5"): "Eco-Tech/GLM-5.1-w4a4c8-mxfp4",
    ("glm-5.1", "a3"): "Eco-Tech/GLM-5.1-w4a8",
    ("glm-5.1", "a2"): "Eco-Tech/GLM-5.1-w4a8",
    ("glm-5", "a5"): "Eco-Tech/GLM-5-w4a8-mtp-QuaRot",
    ("glm-5", "a3"): "Eco-Tech/GLM-5-w4a8-mtp-QuaRot",
    ("glm-5", "a2"): "Eco-Tech/GLM-5-w4a8-mtp-QuaRot",
    ("minimax-m3", "a5"): "MiniMax/MiniMax-M3-MXFP8",
    ("minimax-m3", "a3"): "Eco-Tech/MiniMax-M3-w8a8-0626",
    ("minimax-m3", "a2"): "Eco-Tech/MiniMax-M3-w8a8-0626",
    ("minimax-m3.1-flash", "a5"): "Eco-Tech/MiniMax-M3.1-Flash-w8a8-mxfp8",
    ("minimax-m3.1-flash", "a3"): "Eco-Tech/MiniMax-M3.1-Flash-w8a8",
    ("minimax-m3.1-flash", "a2"): "Eco-Tech/MiniMax-M3.1-Flash-w8a8",
    ("step-5-preview", "a5"): "Eco-Tech/Step-5-Preview-w8a8-mxfp8",
    ("step-5-preview", "a3"): "stepfun-ai/Step-5-Preview",
    ("step-5-preview", "a2"): "stepfun-ai/Step-5-Preview",
    ("gemma-4-31b", "a5"): "Eco-Tech/gemma-4-31B-it-w4a4-mxfp4",
    ("gemma-4-31b", "a3"): "google/gemma-4-31B-it",
    ("gemma-4-31b", "a2"): "google/gemma-4-31B-it",
    ("gemma-4-26b-a4b", "a5"): "Eco-Tech/gemma-4-26B-A4B-it-w4a4-mxfp4",
    ("gemma-4-26b-a4b", "a3"): "google/gemma-4-26B-A4B-it",
    ("gemma-4-26b-a4b", "a2"): "google/gemma-4-26B-A4B-it",
    ("bailing-v3", "a5"): "Eco-Tech/Ling-3.1-flash-w4a8",
    ("bailing-v3", "a3"): "Eco-Tech/Ling-3.1-flash-w4a8",
    ("bailing-v3", "a2"): "Eco-Tech/Ling-3.1-flash-w4a8",
}


def _p(
    quant_by_arch: Mapping[str, str],
    mtp: int,
    doc: str,
    *,
    mlapo: bool = False,
    sp: bool = True,
    mso: bool = False,
    engram_offload: bool = False,
    support: str = "vllm_ascend_main",
    kv_block_size: int | None = None,
    capture: str = "full",
    topologies: Mapping[str, Any] | None = None,
    tutorial_flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "quant": dict(quant_by_arch),
        "graph_capture_mode": capture,
        "sp": sp,
        "mlapo": bool(mlapo),
        "mtp_proposal_tokens": int(mtp),
        "multistream_overlap_shared_expert": bool(mso),
        "engram_cpu_offload": bool(engram_offload),
        "framework_support": support,
        "kv_block_size": kv_block_size,
        "env": dict(_MLAPO_BASE_ENV if mlapo else _BASE_ENV),
        "topologies": dict(topologies or {}),
        "tutorial_flags": dict(tutorial_flags or {}),
        "source": {
            "repository": VLLM_ASCEND_REPO,
            "doc": _DOC_ROOT + doc,
            "basis": "structural_runtime_capability",
        },
    }
    if capture == "full":
        payload["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    return payload


def _topo(
    *,
    colocated: Mapping[str, Any] | None = None,
    pd_prefill: Mapping[str, Any] | None = None,
    pd_decode: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if colocated:
        out["colocated"] = dict(colocated)
    if pd_prefill:
        out["pd_prefill"] = dict(pd_prefill)
    if pd_decode:
        out["pd_decode"] = dict(pd_decode)
    return out


#: Documented vLLM-Ascend LLM/VLM serving profiles (strictly excludes Diffusion/DiT).
VA_RECOMMENDED_PROFILES: dict[str, dict[str, Any]] = {
    "deepseek-r1": _p(
        {"a2": "w8a8", "a3": "w8a8"}, MTP_STANDARD_PROPOSAL_TOKENS, "DeepSeek-V3.2.md", mlapo=True,
    ),
    "deepseek-v3": _p(
        {"a2": "w8a8", "a3": "w8a8"}, MTP_STANDARD_PROPOSAL_TOKENS, "DeepSeek-V3.2.md", mlapo=True,
    ),
    "deepseek-v3.2": _p(
        {"a2": "w8a8", "a3": "w8a8"}, MTP_STANDARD_PROPOSAL_TOKENS, "DeepSeek-V3.2.md", mlapo=True,
    ),
    "deepseek-v2-lite": _p({}, NO_MTP_PROPOSAL_TOKENS, "DeepSeek-V2.md", mlapo=True),
    "deepseek-v4-pro-0813": _p(
        {"a2": "w4a8", "a3": "w4a8"}, NO_MTP_PROPOSAL_TOKENS, "DeepSeek-V4-Pro.md", mso=True,
        topologies={
            "a5": _topo(
                colocated={"dp": 2, "tp": 1, "pcp": 8, "ep": 16, "block_size": 32, "kv_cache_dtype": "fp8"},
                pd_prefill={"dp": 2, "tp": 1, "pcp": 8, "ep": 16, "block_size": 32},
                pd_decode={"dp": 16, "tp": 1, "ep": 16, "block_size": 32},
            ),
            "a3": _topo(colocated={"dp": 4, "tp": 8, "ep": 32, "block_size": 128}),
        },
    ),
    "deepseek-v4-10t": _p(
        {"a2": "w4a8", "a3": "w4a8"}, MTP_DSPARK_PROPOSAL_TOKENS, "DeepSeek-V4-Pro.md", mso=True,
        topologies={"a5": _topo(colocated={"dp": 2, "tp": 1, "pcp": 8, "ep": 16, "block_size": 32})},
    ),
    "deepseek-v4-flash-0731": _p(
        {"a2": "w4a8", "a3": "w4a8"}, SINGLE_MTP_PROPOSAL_TOKEN, "DeepSeek-V4-Flash.md", mso=True,
        topologies={
            "a5": _topo(
                colocated={"dp": 8, "tp": 1, "ep": 8, "block_size": 32, "kv_cache_dtype": "fp8"},
                pd_prefill={"dp": 4, "tp": 1, "pcp": 4, "ep": 16, "block_size": 32},
                pd_decode={"dp": 8, "tp": 1, "ep": 8, "block_size": 32},
            ),
            "a3": _topo(colocated={"dp": 1, "tp": 8, "ep": 8, "block_size": 128}),
        },
        tutorial_flags={"enable_shared_expert_dp": True},
    ),
    "deepseek-v4.1-flash": _p(
        {"a2": "w4a8", "a3": "w4a8"}, MTP_DSPARK_PROPOSAL_TOKENS, "DeepSeek-V4.1-Flash.md",
        mso=True, engram_offload=True,
        topologies={
            "a5": _topo(
                colocated={"dp": 8, "tp": 1, "ep": 8, "block_size": 128},
                pd_prefill={"dp": 2, "tp": 1, "pcp": 4, "ep": 8, "block_size": 128},
                pd_decode={"dp": 8, "tp": 1, "ep": 8, "block_size": 128},
            ),
            "a3": _topo(colocated={"dp": 4, "tp": 4, "ep": 16, "block_size": 128}),
        },
    ),
    "glm-5": _p(
        {"a2": "w8a8", "a3": "w8a8"}, MTP_STANDARD_PROPOSAL_TOKENS, "GLM5.md", mlapo=True, mso=True,
        topologies={
            "a5": _topo(colocated={"dp": 1, "tp": 1, "pcp": 8, "ep": 8, "kv_cache_dtype": "fp8"}),
            "a3": _topo(colocated={"dp": 1, "tp": 16, "ep": 16}),
        },
        tutorial_flags={"enable_flashcomm1": True},
    ),
    "glm-5.1": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8_mxfp4"}, MTP_STANDARD_PROPOSAL_TOKENS, "GLM5.md",
        mlapo=True, mso=True,
        topologies={"a5": _topo(colocated={"dp": 1, "tp": 1, "pcp": 8, "ep": 8})},
        tutorial_flags={"enable_flashcomm1": True},
    ),
    "glm-5.2": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8_mxfp4"}, MTP_STANDARD_PROPOSAL_TOKENS, "GLM5.2.md",
        mlapo=True, mso=True,
        topologies={
            "a5": _topo(
                colocated={"dp": 1, "tp": 8, "ep": 8, "kv_cache_dtype": "fp8"},
                pd_prefill={"dp": 1, "tp": 8, "pp": 2, "ep": 8, "kv_cache_dtype": "fp8"},
                pd_decode={"dp": 8, "tp": 1, "ep": 8, "o_proj_tp": 8, "lm_head_tp": 8},
            ),
            "a3": _topo(colocated={"dp": 1, "tp": 16, "ep": 16, "kv_cache_dtype": "int8"}),
        },
        tutorial_flags={"enable_flashcomm1": True, "enable_fused_mc2": 1},
    ),
    "glm-5.3": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, MTP_STANDARD_PROPOSAL_TOKENS, "GLM5.2.md", mlapo=True, mso=True,
        topologies={"a5": _topo(colocated={"dp": 8, "tp": 4, "ep": 32, "kv_cache_dtype": "fp8"})},
        tutorial_flags={"enable_flashcomm1": True, "enable_fused_mc2": 1},
    ),
    "glm-5.3-flash": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, MTP_STANDARD_PROPOSAL_TOKENS, "GLM5.3-Flash.md", mso=True,
        topologies={
            "a5": _topo(
                colocated={"dp": 1, "tp": 8, "ep": 8, "kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8"},
                pd_prefill={"dp": 1, "tp": 1, "pcp": 8, "ep": 8, "kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8"},
                pd_decode={
                    "dp": 8, "tp": 1, "ep": 8, "linear_proj_tp": 8, "o_proj_tp": 8, "lm_head_tp": 8,
                    "kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8",
                },
            ),
            "a3": _topo(colocated={"dp": 1, "tp": 16, "ep": 16, "kv_cache_dtype": "int8"}),
        },
    ),
    "kimi-k2": _p({"a2": "w8a8", "a3": "w8a8"}, NO_MTP_PROPOSAL_TOKENS, "Kimi-K2.md"),
    "kimi-k2-thinking": _p({}, NO_MTP_PROPOSAL_TOKENS, "Kimi-K2.md"),
    "kimi-k2.5": _p({"a2": "w8a8", "a3": "w8a8"}, MTP_STANDARD_PROPOSAL_TOKENS, "Kimi-K2.5.md"),
    "kimi-k2.6": _p({"a2": "w4a8", "a3": "w4a8"}, NO_MTP_PROPOSAL_TOKENS, "Kimi-K2.6.md"),
    "kimi-k3": _p(
        {"a2": "w4a8", "a3": "w4a8"}, MTP_KIMI_K3_PROPOSAL_TOKENS, "Kimi-K3.md", mlapo=True,
        topologies={
            "a5": _topo(colocated={"dp": 4, "tp": 8, "ep": 32}),
            "a3": _topo(colocated={"dp": 4, "tp": 16, "ep": 64}),
        },
        tutorial_flags={"enable_flashcomm1": True},
    ),
    "kimi-k3.1": _p(
        {"a2": "w4a8", "a3": "w4a8", "a5": "mxfp4_routed_only"}, MTP_DSPARK_PROPOSAL_TOKENS,
        "Kimi-K3.1-Pro.md", mlapo=True, mso=True, kv_block_size=PAIRED_SFA_DOC_BLOCK_SIZE,
        topologies={
            "a5": _topo(
                pd_prefill={
                    "dp": 1, "tp": 1, "pp": 4, "pcp": 8, "ep": 8, "block_size": 1536,
                    "kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8",
                },
                pd_decode={
                    "dp": 1, "tp": 8, "dcp": 8, "ep": 8, "block_size": 1536,
                    "kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8",
                },
            ),
            "a3": _topo(
                pd_prefill={"dp": 1, "tp": 1, "pp": 4, "pcp": 4, "ep": 4, "block_size": 768, "kv_cache_dtype": "int8"},
                pd_decode={"dp": 1, "tp": 16, "dcp": 16, "ep": 16, "block_size": 768, "kv_cache_dtype": "int8"},
            ),
            "a2": _topo(colocated={"dp": 1, "tp": 8, "ep": 8, "block_size": 768, "kv_cache_dtype": "int8"}),
        },
        tutorial_flags={"enable_fused_mc2": 0},
    ),
    "kimi-k3-dspark": _p({}, NO_MTP_PROPOSAL_TOKENS, "Kimi-K3.md", sp=False),
    "kimi-k3-dspark-block5": _p({}, NO_MTP_PROPOSAL_TOKENS, "Kimi-K3.1-Pro.md", mlapo=True, sp=False),
    "qwen3-235b-a22b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, NO_MTP_PROPOSAL_TOKENS, "Qwen3-MoE.md"),
    "qwen3-30b-a3b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, NO_MTP_PROPOSAL_TOKENS, "Qwen3-MoE.md"),
    "qwen3-32b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, NO_MTP_PROPOSAL_TOKENS, "Qwen3.md", sp=False),
    "qwen3-next-80b-a3b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, SINGLE_MTP_PROPOSAL_TOKEN, "Qwen3-Next.md", mso=True,
    ),
    "qwen3.5-27b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, SINGLE_MTP_PROPOSAL_TOKEN, "Qwen3.5-27B-Qwen3.6-27B.md", sp=True,
    ),
    "qwen3.6-27b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, SINGLE_MTP_PROPOSAL_TOKEN, "Qwen3.5-27B-Qwen3.6-27B.md", sp=True,
    ),
    "qwen3.8-27b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.8-27B.md", sp=True,
    ),
    "qwen3.5-35b-a3b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, SINGLE_MTP_PROPOSAL_TOKEN, "Qwen3.6-35B-A3B.md", mso=True,
    ),
    "qwen3.6-35b-a3b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.6-35B-A3B.md", mso=True,
    ),
    "qwen3.5-122b-a10b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, SINGLE_MTP_PROPOSAL_TOKEN, "Qwen3.5-397B-A17B.md", mso=True,
        topologies={"a5": _topo(colocated={"dp": 1, "tp": 4, "ep": 4})},
        tutorial_flags={"enable_flashcomm1": True},
    ),
    "qwen3.5-397b-a17b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.5-397B-A17B.md", mso=True,
        topologies={
            "a5": _topo(
                colocated={"dp": 1, "tp": 8, "ep": 8, "enable_fused_mc2": 2},
                pd_decode={"dp": 2, "tp": 4, "ep": 8, "enable_fused_mc2": 2},
            ),
            "a3": _topo(colocated={"dp": 1, "tp": 16, "ep": 16}),
        },
        tutorial_flags={"enable_flashcomm1": True, "enable_fused_mc2": 1},
    ),
    "qwen3.8-2.4t-a95b": _p(
        {"a2": "w4a8", "a3": "w4a8", "a5": "mxfp8"}, MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.8-2.4T.md", mso=True,
        topologies={
            "a5": _topo(colocated={"dp": 8, "tp": 8, "ep": 64}),
            "a3": _topo(colocated={"dp": 4, "tp": 16, "ep": 64}),
        },
    ),
    "qwen3.8-flash-next": _p(
        {"a2": "w4a8", "a3": "w4a8", "a5": "mxfp4"}, MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.8-Flash-Next.md",
        mso=True, engram_offload=True,
        topologies={
            "a5": _topo(
                colocated={"dp": 1, "tp": 8, "ep": 8, "kv_cache_dtype": "fp8"},
                pd_prefill={"dp": 1, "tp": 4, "pp": 2, "ep": 4, "kv_cache_dtype": "fp8"},
                pd_decode={"dp": 4, "tp": 2, "ep": 8, "kv_cache_dtype": "fp8"},
            ),
            "a3": _topo(colocated={"dp": 1, "tp": 8, "ep": 8, "kv_cache_dtype": "int8"}),
            "a2": _topo(colocated={"dp": 1, "tp": 8, "ep": 8, "kv_cache_dtype": "int8"}),
        },
        tutorial_flags={
            "enable_flashcomm1": True, "enable_shared_expert_dp": True,
            "enable_fused_mc2": 0, "enable_prefill_mc2": True,
        },
    ),
    "minimax-m3": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, MTP_KIMI_K3_PROPOSAL_TOKENS, "MiniMax-M3.1-Flash.md", mso=True,
        topologies={
            "a5": _topo(colocated={"dp": 2, "tp": 4, "ep": 8, "kv_cache_dtype": "fp8"}),
            "a3": _topo(colocated={"dp": 4, "tp": 4, "ep": 16}),
        },
        tutorial_flags={"enable_flashcomm1": True, "enable_shared_expert_dp": True},
    ),
    "minimax-m3.1-flash": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, MTP_KIMI_K3_PROPOSAL_TOKENS, "MiniMax-M3.1-Flash.md", mso=True,
        topologies={"a5": _topo(colocated={"dp": 4, "tp": 4, "ep": 16, "kv_cache_dtype": "fp8"})},
    ),
    "step-5-preview": _p(
        {"a5": "mxfp8"}, MTP_STANDARD_PROPOSAL_TOKENS, "Step-5-Preview.md", mso=True,
        topologies={"a5": _topo(colocated={"dp": 4, "tp": 4, "ep": 16})},
        tutorial_flags={"enable_shared_expert_dp": True},
    ),
    "gemma-4-31b": _p(
        {"a5": "mxfp4"}, NO_MTP_PROPOSAL_TOKENS, "Gemma4.md", sp=False,
        topologies={"a5": _topo(colocated={"dp": 2, "tp": 4, "ep": 1})},
    ),
    "gemma-4-26b-a4b": _p(
        {"a5": "mxfp4"}, NO_MTP_PROPOSAL_TOKENS, "Gemma4.md", sp=True,
        topologies={"a5": _topo(colocated={"dp": 2, "tp": 4, "ep": 8})},
    ),
    "longcat-flash-lite": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, NO_MTP_PROPOSAL_TOKENS, "LongCat-Flash.md", mlapo=True, mso=True,
    ),
    "bailing-v3": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, NO_MTP_PROPOSAL_TOKENS, "Bailing-V3.md", mso=True,
    ),
    "qwen3.5-moe": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, NO_MTP_PROPOSAL_TOKENS, "Qwen3.6-35B-A3B.md", mso=True,
    ),
}


def normalize_hw_family(hw: Hardware | str) -> str:
    """Resolve hardware object or SKU/family string to ``a2`` / ``a3`` / ``a5`` / ``other``."""
    if isinstance(hw, Hardware):
        probes = (
            hw.arch_family,
            hw.sku,
            hw.name,
            hw.dies[0].arch_family if hw.dies else "",
            hw.dies[0].sku if hw.dies else "",
            hw.dies[0].compute_arch.preset if hw.dies else "",
        )
        raw = " ".join(str(p or "") for p in probes).lower()
    else:
        raw = str(hw or "").strip().lower()
    if any(tok in raw for tok in _A5_FAMILY_TOKENS):
        return "a5"
    if any(tok in raw for tok in _A3_FAMILY_TOKENS):
        return "a3"
    if any(tok in raw for tok in _A2_FAMILY_TOKENS):
        return "a2"
    return "other"


def hardware_backend_kind(hardware: Hardware) -> str:
    """Classify a ``Hardware`` object into ``ascend`` / ``nvidia`` / ``generic``."""
    if normalize_hw_family(hardware) in {"a2", "a3", "a5"}:
        return "ascend"
    probes = " ".join(
        str(x or "") for x in (
            hardware.arch_family,
            hardware.sku,
            hardware.name,
            hardware.dies[0].arch_family if hardware.dies else "",
            hardware.dies[0].compute_arch.preset if hardware.dies else "",
        )
    ).lower()
    if "ascend" in probes or "npu" in probes or "dav_" in probes:
        return "ascend"
    if any(tok in probes for tok in _NVIDIA_TOKENS):
        return "nvidia"
    return "generic"


def normalize_canonical_model(model_or_name: ModelSpec | str) -> str:
    """Normalize a ``ModelSpec``, catalog key, alias, or repo ID to canonical model name."""
    raw = model_or_name.name if isinstance(model_or_name, ModelSpec) else str(model_or_name or "")
    text = raw.strip()
    if not text:
        return ""
    tail = text.rsplit("/", 1)[-1] if "/" in text else text
    for cand in (text.lower(), tail.lower()):
        if cand in VA_RECOMMENDED_PROFILES:
            return cand
        if cand in _MODEL_ALIASES:
            return _MODEL_ALIASES[cand]
    stripped = _SUFFIX_STRIP_RE.sub("", tail.lower())
    if stripped in VA_RECOMMENDED_PROFILES:
        return stripped
    if stripped in _MODEL_ALIASES:
        return _MODEL_ALIASES[stripped]
    return tail.lower()


def resolve_model_type(model: ModelSpec) -> str:
    """Extract normalized HF ``model_type`` from ``ModelSpec`` metadata/catalog facts."""
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    meta = dict(getattr(model, "metadata", ()) or ())
    raw = str(facts.get("model_type") or meta.get("source_model_type") or meta.get("model_type") or "").strip().lower()
    if raw:
        return raw
    canon = normalize_canonical_model(model)
    if canon.startswith("kimi-k3"):
        return "kimi_k3"
    if canon == "glm-5.3-flash":
        return "glm5_next_text"
    if canon == "qwen3.8-flash-next":
        return "qwen4_exp_text"
    if canon.startswith("deepseek-v4.1"):
        return "deepseek_v41"
    return canon.replace("-", "_")


def infer_model_workload_domain(model: ModelSpec) -> WorkloadDomain:
    """Classify ``ModelSpec`` structurally into ``llm`` / ``vlm`` / ``diffusion``."""
    meta = dict(getattr(model, "metadata", ()) or ())
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    explicit_domain = str(meta.get("domain") or facts.get("workload_domain") or "").strip().lower()
    if explicit_domain in {"llm", "vlm", "diffusion", "custom"}:
        return explicit_domain  # type: ignore[return-value]
    for plan in model.layer_plans:
        arch = str(getattr(plan, "architecture", "") or "").lower()
        if any(tok in arch for tok in _DIFFUSION_ARCH_TOKENS):
            return "diffusion"
    model_type = resolve_model_type(model)
    name_low = str(model.name or "").lower()
    if any(tok in model_type or tok in name_low for tok in _DIFFUSION_ARCH_TOKENS):
        return "diffusion"
    if model.vision is not None or bool(facts.get("vision_present")):
        return "vlm"
    return "llm"


def default_checkpoint_repo(canonical_model: str, hw_family: str) -> str | None:
    """Return curated default checkpoint repository for ``(canonical_model, hw_family)``."""
    canon = normalize_canonical_model(canonical_model)
    if any(tok in canon for tok in _DIFFUSION_ARCH_TOKENS):
        return None
    hw = normalize_hw_family(hw_family)
    if hw not in {"a2", "a3", "a5"}:
        return None
    direct = DEFAULT_CHECKPOINTS_BY_MODEL_HW.get((canon, hw))
    if direct is not None:
        return direct
    order = ("a3", "a2") if hw in A2_A3_INT_ONLY_FAMILIES else ("a5", "a3", "a2")
    for fallback_hw in order:
        found = DEFAULT_CHECKPOINTS_BY_MODEL_HW.get((canon, fallback_hw))
        if found is not None:
            return found
    return None


def validate_vllm_ascend_quant_isa(
    quant_preset_or_tag: str,
    hw_family: str,
    *,
    kv_cache_dtype: str = "bf16",
    strict: bool = False,
) -> tuple[str, ...]:
    """Enforce strict A2/A3 INT-only vs A5 MXFP/FP8 ISA separation."""
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


@lru_cache(maxsize=1)
def load_vllm_ascend_manifest() -> dict[str, Any]:
    """Load and validate packaged ``vllm-ascend`` plugin ``manifest.json``."""
    if not MANIFEST_PATH.is_file():
        raise ModelingError(f"vllm_ascend_manifest_missing:{MANIFEST_PATH}")
    if MANIFEST_PATH.stat().st_size > MAX_MANIFEST_BYTES:
        raise ModelingError(f"vllm_ascend_manifest_exceeds_50_mib:{MANIFEST_PATH}")
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema") != MANIFEST_SCHEMA:
        raise ModelingError("vllm_ascend_manifest_schema_invalid")
    return dict(raw)


@lru_cache(maxsize=1)
def load_manifest_bindings() -> tuple[ManifestBinding, ...]:
    """Return immutable ``ManifestBinding`` records from ``manifest.json``."""
    raw = load_vllm_ascend_manifest()
    bindings: list[ManifestBinding] = []
    for item in raw.get("bindings") or ():
        prov_raw = item.get("provenance") or {}
        prov = SourceRef(
            repository=str(prov_raw.get("repository") or VLLM_ASCEND_REPO),
            revision=str(prov_raw.get("revision") or raw.get("manifest_revision") or ""),
            relative_path=str(prov_raw.get("relative_path") or "manifest.json"),
            source_sha256=str(prov_raw.get("source_sha256") or ""),
        )
        mech_id = str(item["mechanism_id"])
        bindings.append(
            ManifestBinding(
                mechanism_id=mech_id,
                state=str(item["state"]),  # type: ignore[arg-type]
                ref=str(item.get("ref") or ""),
                lowering_pass_id=str(item.get("lowering_pass_id") or ""),
                provenance=prov,
                models=tuple(str(m) for m in (item.get("models") or ())),
                archs=tuple(str(a).lower() for a in (item.get("archs") or ())),
                phase_scope=tuple(str(p).lower() for p in (item.get("phases") or item.get("phase_scope") or ())),
                options=tuple(sorted((str(k), v) for k, v in (item.get("options") or {}).items())),
                capability_predicates=tuple(str(p) for p in (item.get("capability_predicates") or ())),
                notes=str(item.get("notes") or ""),
            )
        )
    return tuple(bindings)


def resolve_manifest_bindings(
    track: StackTrack,
    canonical_model: str,
    arch: str,
    *,
    phase: str | None = None,
) -> tuple[ManifestBinding, ...]:
    """Select active manifest mechanism bindings for ``(track, model, arch, phase)``."""
    if track == "eager":
        return ()
    allowed_states = {"merged"} if track == "shipping" else {"merged", "unmerged_pr"}
    canon = normalize_canonical_model(canonical_model)
    hw = normalize_hw_family(arch)
    return tuple(
        b for b in load_manifest_bindings()
        if b.state in allowed_states and b.matches(canon, hw, phase=phase)
    )


def shipping_collective_policy(
    arch: str, *, canonical: str = "", role: str | None = None,
) -> dict[str, Any] | None:
    """Resolve arch-disjoint shipping collective policy without cross-gen leaks."""
    arch_norm = normalize_hw_family(arch)
    role_norm = str(role or "").strip().lower() or "colocated"
    canon = normalize_canonical_model(canonical)
    if arch_norm == "a5":
        env_mode = "CCU_SCHED" if canon in _A5_CCU_SCHED_MODELS else "CCU_MS"
        return {
            "arch": "a5",
            "role": role_norm,
            "process_env_mode": env_mode,
            "executor": "ccu_ms",
            "fabric": "ub_direct",
            "oneshot_active": True,
        }
    if arch_norm == "a3":
        if canon in _A3_PREFILL_UNSET_HCCL_MODELS and role_norm == "pd_prefill":
            return {
                "arch": "a3",
                "role": role_norm,
                "process_env_mode": "",
                "executor": "default",
                "fabric": "hccs_sw",
                "oneshot_active": True,
            }
        return {
            "arch": "a3",
            "role": role_norm,
            "process_env_mode": "AIV",
            "executor": "aiv",
            "fabric": "hccs_sw",
            "oneshot_active": True,
        }
    if arch_norm == "a2":
        return {
            "arch": "a2",
            "role": role_norm,
            "process_env_mode": "AIV",
            "executor": "aiv",
            "fabric": "hccs_8p_roce",
            "oneshot_active": True,
        }
    return None


def resolve_fused_mc2_and_mso(
    *,
    arch: str,
    model: ModelSpec,
    parallel: ParallelSpec,
    requested_fused_mc2: int | None,
    requested_mso: bool,
) -> tuple[int, bool, tuple[tuple[str, str, str], ...]]:
    """Resolve upstream ``enable_fused_mc2`` level vs MSO mutex and ISA legality."""
    hw = normalize_hw_family(arch)
    dispositions: list[tuple[str, str, str]] = []
    has_moe = bool(model.num_experts > 0 or any(p.ffn == "moe" for p in model.layer_plans))
    has_shared = bool(model.num_shared_experts > 0 or any(p.num_shared_experts > 0 for p in model.layer_plans))
    eff_mc2 = FUSED_MC2_DISABLED
    if requested_fused_mc2 is not None:
        lvl = int(requested_fused_mc2)
        if not has_moe and lvl > FUSED_MC2_DISABLED:
            dispositions.append(("enable_fused_mc2", "rejected", "dense_model_has_no_moe_mc2"))
        elif parallel.ep <= 1 and lvl > FUSED_MC2_DISABLED:
            dispositions.append(("enable_fused_mc2", "downgraded", "mc2_requires_ep_gt_1"))
        elif lvl == FUSED_MC2_DISPATCH_FFN_COMBINE and hw != "a3":
            dispositions.append(("enable_fused_mc2", "rejected", f"level_1_dispatch_ffn_combine_a3_only:hw={hw}"))
        elif lvl == FUSED_MC2_MEGAMOE_ROUTED and hw not in {"a3", "a5"}:
            dispositions.append(("enable_fused_mc2", "rejected", f"level_2_megamoe_requires_a3_or_a5:hw={hw}"))
        elif lvl == FUSED_MC2_MEGAMOE_SHARED and hw != "a5":
            dispositions.append(("enable_fused_mc2", "downgraded", f"level_3_megamoe_shared_a5_only:hw={hw}"))
            if hw == "a3" and has_moe and parallel.ep > 1:
                eff_mc2 = FUSED_MC2_MEGAMOE_ROUTED
        elif lvl == FUSED_MC2_MEGAMOE_SHARED and not has_shared:
            eff_mc2 = FUSED_MC2_MEGAMOE_ROUTED
            dispositions.append(("enable_fused_mc2", "downgraded", "level_3_downgraded_no_shared_experts"))
        elif FUSED_MC2_DISABLED <= lvl <= FUSED_MC2_MEGAMOE_SHARED:
            eff_mc2 = lvl
            dispositions.append(("enable_fused_mc2", "accepted", f"arch={hw}:level={lvl}"))
        else:
            dispositions.append(("enable_fused_mc2", "rejected", f"unsupported_mc2_level:{lvl}"))

    eff_mso = bool(requested_mso)
    if eff_mso and not has_shared:
        eff_mso = False
        dispositions.append(("multistream_overlap_shared_expert", "rejected", "model_has_no_shared_experts"))
    elif eff_mso and eff_mc2 > FUSED_MC2_DISABLED:
        eff_mso = False
        dispositions.append((
            "multistream_overlap_shared_expert",
            "downgraded",
            f"mutex_with_enable_fused_mc2={eff_mc2}",
        ))
    elif eff_mso:
        dispositions.append(("multistream_overlap_shared_expert", "accepted", "aux_stream_shared_expert_overlap"))
    return eff_mc2, eff_mso, tuple(dispositions)


def _select_role_topology(arch_topos: Mapping[str, Any], *, role: str | None = None) -> dict[str, Any]:
    if not isinstance(arch_topos, Mapping) or not arch_topos:
        return {}
    if role and role in arch_topos:
        return dict(arch_topos[role])
    if "colocated" in arch_topos:
        return dict(arch_topos["colocated"])
    first_role = next(iter(arch_topos))
    return dict(arch_topos[first_role])


def _filter_tutorial_fused_mc2(flags: dict[str, Any], arch: str) -> None:
    if "enable_fused_mc2" not in flags or flags["enable_fused_mc2"] is None:
        return
    lvl = int(flags["enable_fused_mc2"])
    hw = normalize_hw_family(arch)
    if lvl == FUSED_MC2_DISABLED:
        flags["enable_fused_mc2"] = FUSED_MC2_DISABLED
    elif lvl == FUSED_MC2_DISPATCH_FFN_COMBINE and hw != "a3":
        flags.pop("enable_fused_mc2", None)
    elif lvl == FUSED_MC2_MEGAMOE_ROUTED and hw not in {"a3", "a5"}:
        flags.pop("enable_fused_mc2", None)
    elif lvl == FUSED_MC2_MEGAMOE_SHARED and hw != "a5":
        flags.pop("enable_fused_mc2", None)
    elif lvl < FUSED_MC2_DISABLED or lvl > FUSED_MC2_MEGAMOE_SHARED:
        flags.pop("enable_fused_mc2", None)


def simulation_config(
    canonical: str, sku: str, arch: str, *, role: str | None = None,
) -> dict[str, Any]:
    """Resolve documented vLLM-Ascend runtime defaults for ``(canonical, sku, arch)``."""
    resolved = normalize_canonical_model(canonical)
    hw = normalize_hw_family(arch)
    profile = VA_RECOMMENDED_PROFILES.get(resolved, {})
    quant = profile.get("quant", {}).get(hw)
    documented_capture = profile.get("graph_capture_mode")
    if documented_capture not in {"full", "piecewise", "none"}:
        documented_capture = None
    arch_topos = (profile.get("topologies") or {}).get(hw) or {}
    selected_topo = _select_role_topology(arch_topos, role=role)
    resolved_flags = deepcopy(profile.get("tutorial_flags") or {})
    for flag in _ROLE_SCOPED_TUTORIAL_FLAGS:
        if flag in selected_topo:
            resolved_flags[flag] = selected_topo[flag]
        elif flag != "enable_fused_mc2":
            resolved_flags.pop(flag, None)
    _filter_tutorial_fused_mc2(resolved_flags, hw)
    requested_mso = bool(profile.get("multistream_overlap_shared_expert", False))
    eff_mc2 = resolved_flags.get("enable_fused_mc2")
    effective_mso = False if (eff_mc2 is not None and int(eff_mc2) > FUSED_MC2_DISABLED) else requested_mso
    prof_mlapo = bool(profile.get("mlapo", False))
    role_norm = str(role or "").strip().lower()
    env = dict(_BASE_ENV)
    coll_policy = shipping_collective_policy(hw, canonical=resolved, role=role)
    if coll_policy and coll_policy.get("process_env_mode"):
        env["HCCL_OP_EXPANSION_MODE"] = str(coll_policy["process_env_mode"])
    if prof_mlapo and role_norm != "pd_prefill":
        env["VLLM_ASCEND_ENABLE_MLAPO"] = "1"
    resolved_kv_block = selected_topo.get("block_size") or profile.get("kv_block_size")
    return {
        "model": canonical,
        "sku": sku,
        "hardware_family": hw,
        "canonical_model": resolved,
        "hardware_arch": hw,
        "quant": quant,
        "graph_capture_mode": documented_capture,
        "compilation_config": dict(profile["compilation_config"]) if "compilation_config" in profile else None,
        "sp": profile.get("sp"),
        "mlapo": bool(prof_mlapo and role_norm != "pd_prefill"),
        "mtp_proposal_tokens": int(profile.get("mtp_proposal_tokens", NO_MTP_PROPOSAL_TOKENS)),
        **({"kv_block_size": int(resolved_kv_block)} if resolved_kv_block else {}),
        "multistream_overlap_shared_expert": effective_mso,
        "requested_multistream_overlap_shared_expert": requested_mso,
        "engram_cpu_offload": bool(profile.get("engram_cpu_offload", False)),
        "framework_support": str(profile.get("framework_support", "analytical_projection")),
        "env": env,
        "shipping_collective_policy": coll_policy,
        "topologies": deepcopy(profile.get("topologies") or {}),
        "recommended_topology": deepcopy((profile.get("topologies") or {}).get(hw)),
        "active_role_topology": deepcopy(selected_topo) if selected_topo else None,
        "tutorial_flags": resolved_flags,
        "source": deepcopy(profile.get("source")),
        "coverage": "supported_simulation_subset",
        "quantization_basis": (
            "documented_format_family_with_simulation_recipe"
            if quant
            else "preserve_caller_no_documented_hardware_quantization"
        ),
        "execution_controls_basis": "simulation_choices_not_complete_launch_configuration",
        "limitations": [
            "checkpoint_category_quantization_and_cache_formats_require_explicit_declaration",
            "mtp_is_advisory_workload_dependent_and_not_enabled_automatically",
            "environment_metadata_is_not_applied_to_execution",
            "source_lowerings_have_independent_supported_domains",
        ],
    }


__all__ = [
    "A2_A3_ALLOWED_KV_DTYPES",
    "A2_A3_INT_ONLY_FAMILIES",
    "A5_ONLY_QUANT_TOKENS",
    "DEFAULT_CHECKPOINTS_BY_MODEL_HW",
    "FUSED_MC2_DISABLED",
    "FUSED_MC2_DISPATCH_FFN_COMBINE",
    "FUSED_MC2_MEGAMOE_ROUTED",
    "FUSED_MC2_MEGAMOE_SHARED",
    "MANIFEST_PATH",
    "MANIFEST_SCHEMA",
    "MTP_DSPARK_PROPOSAL_TOKENS",
    "MTP_KIMI_K3_PROPOSAL_TOKENS",
    "MTP_STANDARD_PROPOSAL_TOKENS",
    "NO_MTP_PROPOSAL_TOKENS",
    "PAIRED_SFA_DOC_BLOCK_SIZE",
    "SINGLE_MTP_PROPOSAL_TOKEN",
    "VA_RECOMMENDED_PROFILES",
    "VLLM_ASCEND_REPO",
    "VLLM_ASCEND_STACK_ID",
    "ManifestBinding",
    "default_checkpoint_repo",
    "hardware_backend_kind",
    "infer_model_workload_domain",
    "load_manifest_bindings",
    "load_vllm_ascend_manifest",
    "normalize_canonical_model",
    "normalize_hw_family",
    "resolve_fused_mc2_and_mso",
    "resolve_manifest_bindings",
    "resolve_model_type",
    "shipping_collective_policy",
    "simulation_config",
    "validate_vllm_ascend_quant_isa",
]
