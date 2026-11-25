"""Pure declarative checkpoint repository & HF weight-key prefix metadata tables.

[Role] Hold static, data-only checkpoint repository IDs, alias normalization
  for CLI/launch-script checkpoint lookup, and HF/ModelSlim weight-key prefix
  rewrite tables for ``vllm-ascend`` so executable runtime/quant/adapter modules
  remain 100% ``AG014 = 0`` (zero per-model-name executable routing).
[Invariants] Strictly declarative metadata + pure string lookup helpers. Never
  performs cross-generation hardware fallback (``a5 -> a3 -> a2``), never routes
  operator lowering or collective policy by model name, and never imports from
  ``mindie_sim.export`` or ``mindie_sim.compose`` (AG010).
[Timeouts & Locks] Pure in-memory dictionary lookup; no I/O or locks.
[Side-Effect Policy] Read-only over immutable module tables; zero mutation.
"""
from __future__ import annotations

import re

from mindie_sim.schema.model import ModelSpec

_DIFFUSION_TOKENS = ("dit", "unet", "diffusion", "hunyuan_video", "wan2", "flux", "sdxl", "sd3")
_VALID_ASCEND_HW = frozenset({"a2", "a3", "a5"})

_SUFFIX_STRIP_RE = re.compile(
    r"(?:[-_](?:w\d+a\d+(?:c\d+)?(?:s|sc)?|mxfp\d*|mxfp|fp8|bf16|mtp|floatmtp"
    r"|quarot|laos|pdmix|nopdmix|xllm|310p?|vllm|mindie|exp|terminus|speciale"
    r"|0626|0731|0813|1017|0528|0324|0905|2507|it|instruct|pro))+$"
)

#: Pure checkpoint-lookup alias table (used strictly for repo-ID resolution).
MODEL_CHECKPOINT_ALIASES: dict[str, str] = {
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

#: Curated default checkpoint repository per ``(canonical_model, hw_family)``.
#: Strict A2/A3 INT8/INT4/C8 vs A5 MXFP8/MXFP4/FP8 ISA separation; exact match only.
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

_KNOWN_CHECKPOINT_MODELS: frozenset[str] = frozenset(
    model_key for model_key, _hw in DEFAULT_CHECKPOINTS_BY_MODEL_HW
)

#: Verbatim upstream ``modelslim_config.py`` HF weight-key prefix rewrite tables.
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

#: Verbatim upstream ``modelslim_config.py`` HF weight-key substring rewrite tables.
QUANT_MODEL_SUBSTR_MAPPINGS: dict[str, tuple[tuple[str, str], ...]] = {
    "qwen3_5": ((".mlp.shared_expert.", ".mlp.shared_experts."),),
    "qwen3_5_moe": ((".mlp.shared_expert.", ".mlp.shared_experts."),),
    "qwen4_exp": ((".mlp.shared_expert.", ".mlp.shared_experts."),),
    "qwen4_exp_text": ((".mlp.shared_expert.", ".mlp.shared_experts."),),
}

#: Universal HF VLM / MoE wrapper key aliases applied structurally without model_type dispatch.
UNIVERSAL_CHECKPOINT_PREFIX_REWRITES: tuple[tuple[str, str], ...] = (
    ("model.language_model.", "model."),
    ("language_model.model.", "model."),
)
UNIVERSAL_CHECKPOINT_SUBSTR_REWRITES: tuple[tuple[str, str], ...] = (
    (".mlp.shared_expert.", ".mlp.shared_experts."),
)


def normalize_checkpoint_model_key(model_or_name: ModelSpec | str) -> str:
    """Normalize a model name, alias, or HF repo slug for pure checkpoint-repo lookup."""
    raw = model_or_name.name if isinstance(model_or_name, ModelSpec) else str(model_or_name or "")
    text = raw.strip()
    if not text:
        return ""
    tail = text.rsplit("/", 1)[-1] if "/" in text else text
    for cand in (text.lower(), tail.lower()):
        if cand in _KNOWN_CHECKPOINT_MODELS:
            return cand
        if cand in MODEL_CHECKPOINT_ALIASES:
            return MODEL_CHECKPOINT_ALIASES[cand]
    stripped = _SUFFIX_STRIP_RE.sub("", tail.lower())
    if stripped in _KNOWN_CHECKPOINT_MODELS:
        return stripped
    if stripped in MODEL_CHECKPOINT_ALIASES:
        return MODEL_CHECKPOINT_ALIASES[stripped]
    return tail.lower()


def lookup_default_checkpoint_repo(canonical_model: str, hw_family: str) -> str | None:
    """Return exact curated checkpoint repo for ``(canonical_model, hw_family)`` or ``None``.

    Enforces strict hardware-generation isolation: zero cross-generation fallback
    (``a5 -> a3 -> a2`` is forbidden).
    """
    canon = normalize_checkpoint_model_key(canonical_model)
    if not canon or any(tok in canon for tok in _DIFFUSION_TOKENS):
        return None
    hw = str(hw_family or "").strip().lower()
    if hw not in _VALID_ASCEND_HW:
        return None
    return DEFAULT_CHECKPOINTS_BY_MODEL_HW.get((canon, hw))


def expand_checkpoint_tensor_paths(tensor_path: str, model_type: str = "") -> tuple[str, ...]:
    """Expand an HF/ModelSlim weight key across VLM/MoE prefix and shared-expert spellings."""
    base = str(tensor_path or "").strip()
    if not base:
        return ()
    mtype = str(model_type or "").strip().lower()
    prefix_pairs = list(UNIVERSAL_CHECKPOINT_PREFIX_REWRITES)
    if mtype and mtype in QUANT_MODEL_PREFIX_MAPPINGS:
        prefix_pairs.extend(QUANT_MODEL_PREFIX_MAPPINGS[mtype])
    substr_pairs = list(UNIVERSAL_CHECKPOINT_SUBSTR_REWRITES)
    if mtype and mtype in QUANT_MODEL_SUBSTR_MAPPINGS:
        substr_pairs.extend(QUANT_MODEL_SUBSTR_MAPPINGS[mtype])

    candidates: list[str] = [base]
    for src_prefix, dst_prefix in dict.fromkeys(prefix_pairs):
        if base.startswith(src_prefix):
            candidates.append(dst_prefix + base[len(src_prefix):])
        if base.startswith(dst_prefix):
            candidates.append(src_prefix + base[len(dst_prefix):])

    expanded: list[str] = []
    for cand in candidates:
        expanded.append(cand)
        for src_sub, dst_sub in dict.fromkeys(substr_pairs):
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


__all__ = [
    "DEFAULT_CHECKPOINTS_BY_MODEL_HW",
    "MODEL_CHECKPOINT_ALIASES",
    "QUANT_MODEL_PREFIX_MAPPINGS",
    "QUANT_MODEL_SUBSTR_MAPPINGS",
    "UNIVERSAL_CHECKPOINT_PREFIX_REWRITES",
    "UNIVERSAL_CHECKPOINT_SUBSTR_REWRITES",
    "expand_checkpoint_tensor_paths",
    "lookup_default_checkpoint_repo",
    "normalize_checkpoint_model_key",
]
