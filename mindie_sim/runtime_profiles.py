"""Documented runtime profile defaults and simulation configuration helpers.

[Role] Keep structural model capability references separate from supported
  simulation choices, without binding to repository or container revisions.
[Invariants] A hardware family without a cited quantized checkpoint keeps its
  caller's quantization. Framework merge status is informational provenance,
  never a gate on a mathematically valid parallel composition. ``mtp_proposal_tokens``
  is gated on structural draft availability so models without a draft module
  never advertise a non-zero proposal count.
[Timeouts & Locks] Bounded preset lookups and local model catalog reads, with no locks.
[Side-Effect Policy] Returns new dictionaries; never launches or configures vLLM.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


_REPO = "vllm-project/vllm-ascend"
_DOC_ROOT = "docs/source/tutorials/models/"
_ENV = {"HCCL_OP_EXPANSION_MODE": "AIV", "OMP_PROC_BIND": "false"}
_MLAPO_ENV = {**_ENV, "VLLM_ASCEND_ENABLE_MLAPO": "1"}

NO_MTP_PROPOSAL_TOKENS = 0
SINGLE_MTP_PROPOSAL_TOKEN = 1
MTP_STANDARD_PROPOSAL_TOKENS = 3
MTP_DSPARK_PROPOSAL_TOKENS = 5
MTP_KIMI_K3_PROPOSAL_TOKENS = 7
PAIRED_SFA_DOC_BLOCK_SIZE = 544


def _p(quant_by_arch, mtp, doc, *, mlapo=False, sp=True, mso=False,
       engram_offload=False, support="vllm_ascend_main", kv_block_size=None,
       capture="full", topologies=None, tutorial_flags=None):
    payload: dict[str, Any] = {
        "quant": quant_by_arch,
        "graph_capture_mode": capture,
        "sp": sp,
        "mtp_proposal_tokens": mtp,
        "multistream_overlap_shared_expert": mso,
        "engram_cpu_offload": engram_offload,
        "framework_support": support,
        "kv_block_size": kv_block_size,
        "env": _MLAPO_ENV if mlapo else _ENV,
        "topologies": dict(topologies or {}),
        "tutorial_flags": dict(tutorial_flags or {}),
        "source": {
            "repository": _REPO,
            "doc": _DOC_ROOT + doc,
            "basis": "structural_runtime_capability",
        },
    }
    if capture == "full":
        payload["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    return payload


def _topo(
    *,
    colocated: dict[str, Any] | None = None,
    pd_prefill: dict[str, Any] | None = None,
    pd_decode: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Informational tutorial topology subset (never restricts caller overrides)."""
    out: dict[str, dict[str, Any]] = {}
    if colocated:
        out["colocated"] = dict(colocated)
    if pd_prefill:
        out["pd_prefill"] = dict(pd_prefill)
    if pd_decode:
        out["pd_decode"] = dict(pd_decode)
    return out


# Quantized checkpoint alternatives explicitly listed by each document. Missing
# hardware families intentionally have no override; support on A2/A3 does not
# establish an A5 checkpoint format when the catalog config already declares
# its own native quantization. MTP and topology remain caller decisions.
VA_RECOMMENDED_PROFILES = {
    "deepseek-r1": _p({"a2": "w8a8", "a3": "w8a8"},
                      MTP_STANDARD_PROPOSAL_TOKENS, "DeepSeek-V3.2.md", mlapo=True),
    "deepseek-v3": _p({"a2": "w8a8", "a3": "w8a8"},
                      MTP_STANDARD_PROPOSAL_TOKENS, "DeepSeek-V3.2.md", mlapo=True),
    "deepseek-v3.2": _p({"a2": "w8a8", "a3": "w8a8"},
                        MTP_STANDARD_PROPOSAL_TOKENS, "DeepSeek-V3.2.md", mlapo=True),
    "deepseek-v2-lite": _p({}, NO_MTP_PROPOSAL_TOKENS, "DeepSeek-V2.md", mlapo=True),
    "deepseek-v4-pro-0813": _p(
        {"a2": "w4a8", "a3": "w4a8"}, NO_MTP_PROPOSAL_TOKENS,
        "DeepSeek-V4-Pro.md", mso=True,
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
        {"a2": "w4a8", "a3": "w4a8"}, MTP_DSPARK_PROPOSAL_TOKENS,
        "DeepSeek-V4-Pro.md", mso=True,
        topologies={"a5": _topo(colocated={"dp": 2, "tp": 1, "pcp": 8, "ep": 16, "block_size": 32})},
    ),
    "deepseek-v4-flash-0731": _p(
        {"a2": "w8a8", "a3": "w8a8"}, SINGLE_MTP_PROPOSAL_TOKEN,
        "DeepSeek-V4-Flash.md", mso=True,
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
        {"a2": "w4a8", "a3": "w4a8"}, MTP_DSPARK_PROPOSAL_TOKENS,
        "DeepSeek-V4.1-Flash.md", mso=True, engram_offload=True,
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
        {"a2": "w8a8", "a3": "w8a8"}, MTP_STANDARD_PROPOSAL_TOKENS,
        "GLM5.md", mlapo=True, mso=True,
        topologies={
            "a5": _topo(colocated={"dp": 1, "tp": 1, "pcp": 8, "ep": 8, "kv_cache_dtype": "fp8"}),
            "a3": _topo(colocated={"dp": 1, "tp": 16, "ep": 16}),
        },
        tutorial_flags={"enable_flashcomm1": True},
    ),
    "glm-5.1": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8_mxfp4"}, MTP_STANDARD_PROPOSAL_TOKENS,
        "GLM5.md", mlapo=True, mso=True,
        topologies={"a5": _topo(colocated={"dp": 1, "tp": 1, "pcp": 8, "ep": 8})},
        tutorial_flags={"enable_flashcomm1": True},
    ),
    "glm-5.2": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8_mxfp4"}, MTP_STANDARD_PROPOSAL_TOKENS,
        "GLM5.2.md", mlapo=True, mso=True,
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
        {"a2": "w8a8", "a3": "w8a8"}, MTP_STANDARD_PROPOSAL_TOKENS,
        "GLM5.2.md", mlapo=True, mso=True,
        topologies={"a5": _topo(colocated={"dp": 8, "tp": 4, "ep": 32, "kv_cache_dtype": "int8"})},
        tutorial_flags={"enable_flashcomm1": True, "enable_fused_mc2": 1},
    ),
    "glm-5.3-flash": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, MTP_STANDARD_PROPOSAL_TOKENS,
        "GLM5.3-Flash.md", mso=True,
        topologies={
            "a5": _topo(
                colocated={"dp": 1, "tp": 8, "ep": 8, "kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8"},
                pd_prefill={"dp": 1, "tp": 1, "pcp": 8, "ep": 8, "kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8"},
                pd_decode={"dp": 8, "tp": 1, "ep": 8, "linear_proj_tp": 8, "o_proj_tp": 8, "lm_head_tp": 8,
                           "kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8"},
            ),
            "a3": _topo(colocated={"dp": 1, "tp": 16, "ep": 16, "kv_cache_dtype": "int8"}),
        },
    ),
    "kimi-k2": _p({"a2": "w8a8", "a3": "w8a8"}, NO_MTP_PROPOSAL_TOKENS, "Kimi-K2.md"),
    "kimi-k2-thinking": _p({}, NO_MTP_PROPOSAL_TOKENS, "Kimi-K2.md"),
    "kimi-k2.5": _p({"a2": "w8a8", "a3": "w8a8"},
                    MTP_STANDARD_PROPOSAL_TOKENS, "Kimi-K2.5.md"),
    "kimi-k2.6": _p({"a2": "w4a8", "a3": "w4a8"}, NO_MTP_PROPOSAL_TOKENS, "Kimi-K2.6.md"),
    "kimi-k3": _p(
        {"a2": "w4a8", "a3": "w4a8"}, MTP_KIMI_K3_PROPOSAL_TOKENS,
        "Kimi-K3.md", mlapo=True,
        topologies={
            "a5": _topo(colocated={"dp": 4, "tp": 8, "ep": 32}),
            "a3": _topo(colocated={"dp": 4, "tp": 16, "ep": 64}),
        },
        tutorial_flags={"enable_flashcomm1": True},
    ),
    "kimi-k3.1": _p(
        {"a3": "w4a8", "a5": "mxfp4_routed_only"}, MTP_STANDARD_PROPOSAL_TOKENS,
        "Kimi-K3.1-Pro.md", mlapo=True, mso=True,
        kv_block_size=PAIRED_SFA_DOC_BLOCK_SIZE,
        topologies={
            "a5": _topo(
                pd_prefill={"dp": 1, "tp": 1, "pp": 4, "pcp": 8, "ep": 8, "block_size": 1536,
                            "kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8"},
                pd_decode={"dp": 1, "tp": 8, "dcp": 8, "ep": 8, "block_size": 1536,
                           "kv_cache_dtype": "fp8", "indexer_kv_dtype": "fp8"},
            ),
            "a3": _topo(
                pd_prefill={"dp": 1, "tp": 1, "pp": 4, "pcp": 4, "ep": 4, "block_size": 768, "kv_cache_dtype": "int8"},
                pd_decode={"dp": 1, "tp": 16, "dcp": 16, "ep": 16, "block_size": 768, "kv_cache_dtype": "int8"},
            ),
        },
        tutorial_flags={"enable_fused_mc2": 0},
    ),
    "kimi-k3-dspark": _p({}, NO_MTP_PROPOSAL_TOKENS, "Kimi-K3.md", sp=False),
    "kimi-k3-dspark-block5": _p({}, NO_MTP_PROPOSAL_TOKENS, "Kimi-K3.1-Pro.md",
                                mlapo=True, sp=False),
    "qwen3-235b-a22b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                          NO_MTP_PROPOSAL_TOKENS, "Qwen3-MoE.md"),
    "qwen3-30b-a3b": _p({"a2": "w8a8", "a3": "w8a8"},
                        NO_MTP_PROPOSAL_TOKENS, "Qwen3-MoE.md"),
    "qwen3-32b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                    NO_MTP_PROPOSAL_TOKENS, "Qwen3.md", sp=False),
    "qwen3-next-80b-a3b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                             SINGLE_MTP_PROPOSAL_TOKEN, "Qwen3-Next.md", mso=True),
    "qwen3.5-27b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                      SINGLE_MTP_PROPOSAL_TOKEN, "Qwen3.5-27B-Qwen3.6-27B.md", sp=True),
    "qwen3.6-27b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                      SINGLE_MTP_PROPOSAL_TOKEN, "Qwen3.5-27B-Qwen3.6-27B.md", sp=True),
    "qwen3.8-27b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                      MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.8-27B.md", sp=True),
    "qwen3.5-35b-a3b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                          SINGLE_MTP_PROPOSAL_TOKEN, "Qwen3.6-35B-A3B.md", mso=True),
    "qwen3.6-35b-a3b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                          MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.6-35B-A3B.md", mso=True),
    "qwen3.5-122b-a10b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
        SINGLE_MTP_PROPOSAL_TOKEN, "Qwen3.5-397B-A17B.md", mso=True,
        topologies={"a5": _topo(colocated={"dp": 1, "tp": 4, "ep": 4})},
        tutorial_flags={"enable_flashcomm1": True},
    ),
    "qwen3.5-397b-a17b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
        MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.5-397B-A17B.md", mso=True,
        topologies={
            "a5": _topo(
                colocated={"dp": 1, "tp": 8, "ep": 8},
                pd_decode={"dp": 2, "tp": 4, "ep": 8},
            ),
            "a3": _topo(colocated={"dp": 1, "tp": 16, "ep": 16}),
        },
        tutorial_flags={"enable_flashcomm1": True, "enable_fused_mc2": 1},
    ),
    "qwen3.8-2.4t-a95b": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
        MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.8-2.4T.md", mso=True,
        topologies={
            "a5": _topo(colocated={"dp": 8, "tp": 8, "ep": 64}),
            "a3": _topo(colocated={"dp": 4, "tp": 16, "ep": 64}),
        },
    ),
    "qwen3.8-flash-next": _p(
        {"a2": "w4a8", "a3": "w4a8", "a5": "mxfp4"},
        MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.8-Flash-Next.md",
        mso=True, engram_offload=True,
        topologies={
            "a5": _topo(
                colocated={"dp": 1, "tp": 8, "ep": 8},
                pd_prefill={"dp": 1, "tp": 4, "pp": 2, "ep": 4},
                pd_decode={"dp": 4, "tp": 2, "ep": 8},
            ),
            "a3": _topo(colocated={"dp": 1, "tp": 8, "ep": 8}),
        },
        tutorial_flags={
            "enable_flashcomm1": True, "enable_shared_expert_dp": True,
            "enable_fused_mc2": 0, "enable_prefill_mc2": True,
        },
    ),
    "minimax-m3": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
        MTP_KIMI_K3_PROPOSAL_TOKENS, "MiniMax-M3.1-Flash.md", mso=True,
        topologies={
            "a5": _topo(colocated={"dp": 2, "tp": 4, "ep": 8, "kv_cache_dtype": "fp8"}),
            "a3": _topo(colocated={"dp": 4, "tp": 4, "ep": 16}),
        },
        tutorial_flags={"enable_flashcomm1": True, "enable_shared_expert_dp": True},
    ),
    "minimax-m3.1-flash": _p(
        {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
        MTP_KIMI_K3_PROPOSAL_TOKENS, "MiniMax-M3.1-Flash.md", mso=True,
        topologies={"a5": _topo(colocated={"dp": 4, "tp": 4, "ep": 16})},
    ),
    "step-5-preview": _p(
        {"a5": "mxfp8"}, MTP_STANDARD_PROPOSAL_TOKENS,
        "Step-5-Preview.md", mso=True,
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
    "longcat-flash-lite": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                             NO_MTP_PROPOSAL_TOKENS, "LongCat-Flash.md",
                             mlapo=True, mso=True),
    "bailing-v3": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                     NO_MTP_PROPOSAL_TOKENS, "Bailing-V3.md", mso=True),
    "qwen3.5-moe": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                      NO_MTP_PROPOSAL_TOKENS, "Qwen3.6-35B-A3B.md", mso=True),
    "hunyuan-video-dit": _p({}, NO_MTP_PROPOSAL_TOKENS, "HunyuanVideo.md",
                            sp=False, capture="piecewise"),
}

_MODEL_ALIASES = {
    "deepseek-v3.2-prune-3l": "deepseek-v3.2",
    "deepseek-v4-pro": "deepseek-v4-pro-0813",
    "deepseek-v4-flash": "deepseek-v4-flash-0731",
    "deepseek-v4.1": "deepseek-v4.1-flash",
}


def _supports_structural_draft(canonical: str) -> bool:
    """True when the catalog model declares native/external draft or MLA synthesis."""
    from mindie_sim.schema.common import ModelingError
    try:
        from mindie_sim.model_catalog import load_model
        spec = load_model(canonical)
    except (ModelingError, KeyError, ValueError):
        return True
    aux = spec.auxiliary
    if aux is not None and aux.standalone_draft:
        return False
    if aux is not None and (aux.mtp is not None or aux.dspark is not None):
        return True
    return spec.attention == "mla"


_ROLE_SCOPED_TUTORIAL_FLAGS = ("kv_cache_dtype", "indexer_kv_dtype")


def _select_role_topology(
    arch_topos: dict[str, Any], *, role: str | None = None,
) -> dict[str, Any]:
    """Select the deployment-role topology dict for one hardware family."""
    if not isinstance(arch_topos, dict) or not arch_topos:
        return {}
    if role and role in arch_topos:
        return dict(arch_topos[role])
    if "colocated" in arch_topos:
        return dict(arch_topos["colocated"])
    first_role = next(iter(arch_topos))
    return dict(arch_topos[first_role])


def simulation_config(canonical, sku, arch, *, role: str | None = None):
    """Resolve only documented quant alternatives, with explicit coverage limits."""
    resolved = _MODEL_ALIASES.get(canonical, canonical)
    profile = VA_RECOMMENDED_PROFILES.get(resolved, {})
    quant = profile.get("quant", {}).get(arch)
    # A documented profile may declare its execution controls; an undocumented
    # model gets no fabricated capture/SP policy.  ``compose_case`` resolves the
    # final value as ``explicit caller > documented profile > structural default``.
    documented_capture = profile.get("graph_capture_mode")
    if documented_capture not in {"full", "piecewise", "none"}:
        documented_capture = None
    documented_sp = profile.get("sp")
    raw_mtp = int(profile.get("mtp_proposal_tokens", NO_MTP_PROPOSAL_TOKENS))
    mtp_tokens = (
        raw_mtp
        if raw_mtp > NO_MTP_PROPOSAL_TOKENS and _supports_structural_draft(resolved)
        else NO_MTP_PROPOSAL_TOKENS
    )
    arch_topos = (profile.get("topologies") or {}).get(arch) or {}
    selected_topo = _select_role_topology(arch_topos, role=role)
    resolved_flags = deepcopy(profile.get("tutorial_flags") or {})
    for flag in _ROLE_SCOPED_TUTORIAL_FLAGS:
        if flag in selected_topo:
            resolved_flags[flag] = selected_topo[flag]
        else:
            resolved_flags.pop(flag, None)
    return {
        "model": canonical, "sku": sku, "hardware_family": arch,
        "canonical_model": resolved, "hardware_arch": arch, "quant": quant,
        "graph_capture_mode": documented_capture,
        "compilation_config": (dict(profile["compilation_config"])
                               if "compilation_config" in profile else None),
        "sp": documented_sp,
        "mtp_proposal_tokens": mtp_tokens,
        **({"kv_block_size": profile["kv_block_size"]}
           if profile.get("kv_block_size") else {}),
        "multistream_overlap_shared_expert": bool(
            profile.get("multistream_overlap_shared_expert", False)),
        "engram_cpu_offload": bool(profile.get("engram_cpu_offload", False)),
        "framework_support": str(profile.get("framework_support", "analytical_projection")),
        "env": dict(profile.get("env", {})),
        "topologies": deepcopy(profile.get("topologies") or {}),
        "recommended_topology": deepcopy((profile.get("topologies") or {}).get(arch)),
        "tutorial_flags": resolved_flags,
        "source": deepcopy(profile.get("source")),
        "coverage": "supported_simulation_subset",
        "quantization_basis": ("documented_format_family_with_simulation_recipe" if quant else
                               "preserve_caller_no_documented_hardware_quantization"),
        "execution_controls_basis": "simulation_choices_not_complete_launch_configuration",
        "limitations": [
            "checkpoint_category_quantization_and_cache_formats_require_explicit_declaration",
            "mtp_is_advisory_workload_dependent_and_not_enabled_automatically",
            "environment_metadata_is_not_applied_to_execution",
            "source_lowerings_have_independent_supported_domains",
        ],
    }
