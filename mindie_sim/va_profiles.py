"""Source-scoped vLLM-Ascend simulation presets.

[Role] Keep model-document references separate from supported simulation choices.
[Invariants] A hardware family without a cited quantized checkpoint keeps its
  caller's quantization. These subsets are not complete production launch files.
[Timeouts & Locks] Bounded in-memory lookups; no I/O, network, devices or locks.
[Side-Effect Policy] Returns new dictionaries; never launches or configures vLLM.
"""
from copy import deepcopy


_MAIN_REV = "1223fea314cb6172900bd615c9940fcd2c658925"
_MODEL_REV = "8c30e4f491f73cba737a1dacc522894e437cdf3d"
_DOC_ROOT = "docs/source/tutorials/models/"


def _profile(document, quant, *, revision=_MAIN_REV, lines, mtp=0, env=None):
    return {
        "quant": quant, "graph_capture_mode": "full", "sp": True,
        "mtp_proposal_tokens": mtp, "env": env or {},
        "source": {"repository": "vllm-project/vllm-ascend", "revision": revision,
                   "path": _DOC_ROOT + document, "lines": lines},
    }


# Quantized checkpoint alternatives explicitly listed by each document. Missing
# hardware families intentionally have no override; support on A2/A3 does not
# establish an A5 checkpoint format. MTP and topology remain caller decisions.
VA_RECOMMENDED_PROFILES = {
    "deepseek-r1": _profile("DeepSeek-R1.md", {"a2": "w8a8", "a3": "w8a8"},
                            lines="20-22,160", mtp=3),
    "deepseek-v3.2": _profile("DeepSeek-V3.2.md", {"a2": "w8a8", "a3": "w8a8"},
                              lines="18-20,134-153", mtp=3,
                              env={"VLLM_ASCEND_ENABLE_MLAPO": "1"}),
    "deepseek-v4-flash-0731": _profile(
        "DeepSeek-V4-Flash.md", {"a2": "w8a8", "a3": "w8a8"},
        revision=_MODEL_REV, lines="147-180,227-261", mtp=1),
    "deepseek-v4-pro-0813": _profile(
        "DeepSeek-V4-Pro.md", {"a2": "w4a8", "a3": "w4a8"},
        revision=_MODEL_REV, lines="26-29,562-568"),
    "glm-5.1": _profile("GLM5.md", {"a2": "w8a8", "a3": "w8a8"},
                       lines="22-27,171,185,261", mtp=3,
                       env={"VLLM_ASCEND_ENABLE_MLAPO": "1"}),
    "glm-5.2": _profile("GLM5.2.md", {"a2": "w8a8", "a3": "w8a8"},
                       lines="17-21,165-179", mtp=3),
    "glm-5.3": _profile("GLM5.3.md", {"a2": "w8a8", "a3": "w8a8"},
                       revision=_MODEL_REV, lines="27-29,175-207", mtp=3,
                       env={"VLLM_ASCEND_ENABLE_MLAPO": "1"}),
    "glm-5.3-flash": _profile(
        "GLM5.3-Flash.md", {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
        revision=_MODEL_REV, lines="19-21,158-220", mtp=3),
    "qwen3.5-397b-a17b": _profile(
        "Qwen3.5-397B-A17B.md", {"a2": "w8a8", "a3": "w8a8", "a5": "mxfp8"},
        lines="25-29,176-217,220-280", mtp=3),
    "qwen3.5-35b-a3b": _profile(
        "Qwen3.6-35B-A3B.md", {"a2": "w8a8", "a3": "w8a8"},
        lines="163-217,250", mtp=1),
    "qwen3-next-80b-a3b": _profile("Qwen3-Next.md", {}, lines="19-23"),
    "qwen3-235b-a22b": _profile("Qwen3-235B-A22B.md", {"a2": "w8a8", "a3": "w8a8"},
                              lines="25-33"),
    "qwen3-30b-a3b": _profile("Qwen3-30B-A3B.md", {"a2": "w8a8", "a3": "w8a8"},
                             lines="19-26"),
    "kimi-k2.5": _profile("Kimi-K2.5.md", {"a2": "w4a8", "a3": "w4a8"},
                         lines="21,144-165", env={"VLLM_ASCEND_ENABLE_MLAPO": "1"}),
    "kimi-k2.6": _profile("Kimi-K2.6.md", {"a2": "w4a8", "a3": "w4a8"},
                         lines="21"),
    "kimi-k3": _profile("Kimi-K3.md", {"a3": "w4a8"},
                       revision=_MODEL_REV, lines="29,131-193,391-403"),
}


def simulation_config(canonical, sku, arch):
    """Resolve only documented quant alternatives, with explicit coverage limits."""
    profile = VA_RECOMMENDED_PROFILES.get(canonical, {})
    quant = profile.get("quant", {}).get(arch)
    return {
        "model": canonical, "sku": sku, "hardware_family": arch, "quant": quant,
        "graph_capture_mode": "full", "sp": True,
        "mtp_proposal_tokens": int(profile.get("mtp_proposal_tokens", 0)),
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
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
