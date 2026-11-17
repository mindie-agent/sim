"""Source-scoped vLLM-Ascend simulation presets.

[Role] Keep structural model capability references separate from supported
  simulation choices, without binding to repository or container revisions.
[Invariants] A hardware family without a cited quantized checkpoint keeps its
  caller's quantization. Framework merge status is informational provenance,
  never a gate on a mathematically valid parallel composition. These subsets
  are not complete production launch files.
[Timeouts & Locks] Bounded in-memory lookups; no I/O, network, devices or locks.
[Side-Effect Policy] Returns new dictionaries; never launches or configures vLLM.
"""
from copy import deepcopy


_REPO = "vllm-project/vllm-ascend"
_DOC_ROOT = "docs/source/tutorials/models/"
_ENV = {"HCCL_OP_EXPANSION_MODE": "AIV", "OMP_PROC_BIND": "false"}
_MLAPO_ENV = {**_ENV, "VLLM_ASCEND_ENABLE_MLAPO": "1"}

MTP_STANDARD_PROPOSAL_TOKENS = 3
MTP_DSPARK_PROPOSAL_TOKENS = 5
MTP_KIMI_K3_PROPOSAL_TOKENS = 7
PAIRED_SFA_DOC_BLOCK_SIZE = 544


def _p(quant_by_arch, mtp, doc, *, mlapo=False, sp=True, mso=False,
       engram_offload=False, support="vllm_ascend_main", kv_block_size=None):
    return {
        "quant": quant_by_arch,
        "graph_capture_mode": "full",
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "sp": sp,
        "mtp_proposal_tokens": mtp,
        "multistream_overlap_shared_expert": mso,
        "engram_cpu_offload": engram_offload,
        "framework_support": support,
        "kv_block_size": kv_block_size,
        "env": _MLAPO_ENV if mlapo else _ENV,
        "source": {
            "repository": _REPO,
            "doc": _DOC_ROOT + doc,
            "basis": "structural_runtime_capability",
        },
    }


# Quantized checkpoint alternatives explicitly listed by each document. Missing
# hardware families intentionally have no override; support on A2/A3 does not
# establish an A5 checkpoint format. MTP and topology remain caller decisions.
VA_RECOMMENDED_PROFILES = {
    "deepseek-r1": _p({"a2": "w8a8", "a3": "w8a8"},
                      MTP_STANDARD_PROPOSAL_TOKENS, "DeepSeek-V3.2.md", mlapo=True),
    "deepseek-v3": _p({"a2": "w8a8", "a3": "w8a8"},
                      MTP_STANDARD_PROPOSAL_TOKENS, "DeepSeek-V3.2.md", mlapo=True),
    "deepseek-v3.2": _p({"a2": "w8a8", "a3": "w8a8"},
                        MTP_STANDARD_PROPOSAL_TOKENS, "DeepSeek-V3.2.md", mlapo=True),
    "deepseek-v4-pro-0813": _p({"a2": "w4a8", "a3": "w4a8"}, 0,
                               "DeepSeek-V4-Pro.md", mso=True),
    "deepseek-v4-10t": _p({"a2": "w4a8", "a3": "w4a8"},
                          MTP_DSPARK_PROPOSAL_TOKENS, "DeepSeek-V4-Pro.md", mso=True),
    "deepseek-v4-flash-0731": _p({"a2": "w8a8", "a3": "w8a8"}, 1,
                                 "DeepSeek-V4-Flash.md", mso=True),
    "deepseek-v4.1-flash": _p({"a2": "w4a8", "a3": "w4a8"},
                              MTP_DSPARK_PROPOSAL_TOKENS,
                              "DeepSeek-V4.1-Flash.md", mso=True,
                              engram_offload=True),
    "glm-5": _p({"a2": "w8a8", "a3": "w8a8"},
                MTP_STANDARD_PROPOSAL_TOKENS, "GLM5.md", mlapo=True, mso=True),
    "glm-5.1": _p({"a2": "w8a8", "a3": "w8a8"},
                  MTP_STANDARD_PROPOSAL_TOKENS, "GLM5.md", mlapo=True, mso=True),
    "glm-5.2": _p({"a2": "w8a8", "a3": "w8a8"},
                  MTP_STANDARD_PROPOSAL_TOKENS, "GLM5.2.md", mlapo=True, mso=True),
    "glm-5.3": _p({"a2": "w8a8", "a3": "w8a8"},
                  MTP_STANDARD_PROPOSAL_TOKENS, "GLM5.2.md", mlapo=True, mso=True),
    "glm-5.3-flash": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                        MTP_STANDARD_PROPOSAL_TOKENS, "GLM5.3-Flash.md", mso=True),
    "kimi-k2": _p({"a2": "w8a8", "a3": "w8a8"}, 0, "Kimi-K2.md"),
    "kimi-k2.5": _p({"a2": "w8a8", "a3": "w8a8"},
                    MTP_STANDARD_PROPOSAL_TOKENS, "Kimi-K2.5.md"),
    "kimi-k2.6": _p({"a2": "w4a8", "a3": "w4a8"}, 0, "Kimi-K2.6.md"),
    "kimi-k3": _p({"a2": "w4a8", "a3": "w4a8"},
                  MTP_KIMI_K3_PROPOSAL_TOKENS, "Kimi-K3.md", mlapo=True),
    "kimi-k3.1": _p({"a3": "w4a8"}, MTP_STANDARD_PROPOSAL_TOKENS,
                    "Kimi-K3.1-Pro.md", mlapo=True, mso=True,
                    kv_block_size=PAIRED_SFA_DOC_BLOCK_SIZE),
    "qwen3-235b-a22b": _p({"a2": "w8a8", "a3": "w8a8"}, 0, "Qwen3-MoE.md"),
    "qwen3-30b-a3b": _p({"a2": "w8a8", "a3": "w8a8"}, 0, "Qwen3-MoE.md"),
    "qwen3-next-80b-a3b": _p({"a2": "w8a8", "a3": "w8a8"}, 1, "Qwen3-Next.md"),
    "qwen3.5-27b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, 1,
                      "Qwen3.5-27B-Qwen3.6-27B.md", sp=False),
    "qwen3.6-27b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, 1,
                      "Qwen3.5-27B-Qwen3.6-27B.md", sp=False),
    "qwen3.8-27b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                      MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.8-27B.md", sp=False),
    "qwen3.5-35b-a3b": _p({"a2": "w8a8", "a3": "w8a8"}, 1,
                          "Qwen3.6-35B-A3B.md", mso=True),
    "qwen3.6-35b-a3b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                          MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.6-35B-A3B.md", mso=True),
    "qwen3.5-397b-a17b": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
                            MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.5-397B-A17B.md",
                            mso=True),
    "qwen3.8-flash-next": _p({"a2": "w4a8", "a3": "w4a8", "a5": "mxfp4"},
                             MTP_STANDARD_PROPOSAL_TOKENS, "Qwen3.8-Flash-Next.md",
                             mso=True, engram_offload=True),
    "minimax-m3": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, 0,
                     "MiniMax-M3.1-Flash.md", mso=True),
    "minimax-m3.1-flash": _p({"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"}, 0,
                             "MiniMax-M3.1-Flash.md", mso=True),
    "step-5-preview": _p({"a5": "mxfp8"}, MTP_STANDARD_PROPOSAL_TOKENS,
                         "Step-5-Preview.md", mso=True),
    "gemma-4-31b": _p({}, 0, "Gemma4.md", sp=False),
    "gemma-4-26b-a4b": _p({}, 0, "Gemma4.md", sp=True),
}

_MODEL_ALIASES = {
    "deepseek-v3.2-prune-3l": "deepseek-v3.2",
    "deepseek-v4-pro": "deepseek-v4-pro-0813",
    "deepseek-v4-flash": "deepseek-v4-flash-0731",
}


def simulation_config(canonical, sku, arch):
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
    return {
        "model": canonical, "sku": sku, "hardware_family": arch,
        "canonical_model": resolved, "hardware_arch": arch, "quant": quant,
        "graph_capture_mode": documented_capture,
        "compilation_config": (dict(profile["compilation_config"])
                               if "compilation_config" in profile else None),
        "sp": documented_sp,
        "mtp_proposal_tokens": int(profile.get("mtp_proposal_tokens", 0)),
        **({"kv_block_size": profile["kv_block_size"]}
           if profile.get("kv_block_size") else {}),
        "multistream_overlap_shared_expert": bool(
            profile.get("multistream_overlap_shared_expert", False)),
        "engram_cpu_offload": bool(profile.get("engram_cpu_offload", False)),
        "framework_support": str(profile.get("framework_support", "analytical_projection")),
        "env": dict(profile.get("env", {})),
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
