"""vLLM-Ascend speculative decoding, ACLGraph buckets, and RecoverSSM policies.

[Role] Own all ``vllm-ascend`` speculative decoding rules (DSpark, MTP,
  DeepSeek-MTP, Qwen3.5-MTP, EAGLE/EAGLE-3, DFlash/DFlash2, n-gram), Ascend FIA
  ``(K + 1) <= 16`` decode ceilings, ``n * (K + 1)`` ACLGraph capture-bucket
  generation preserving ``FULL_AND_PIECEWISE``, PD matching-K vs P1/DK rules,
  and ``RecoverSSM`` post-accept recurrent state commit policies.
[Invariants] Zero imports from ``mindie_sim.export`` or ``mindie_sim.compose``
  (AG010). ``FULL_AND_PIECEWISE`` is never collapsed to ``piecewise``.
  ``PostAcceptCommitSpec`` always advances ``published_version > base_version``
  and counts all native recurrent owners (e.g., 69 KDA layers on Kimi-K3.1).
  Shipping ``merged + DSpark`` enables KDA ``RecoverSSM`` consistently via
  merged PR #21549.
[Timeouts & Locks] Pure in-memory rules and dataclass construction; no I/O or
  locks.
[Side-Effect Policy] Returns fresh immutable records and plain dicts; never
  mutates caller inputs or global state.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any

from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.stack import (
    CaptureBucketRule,
    CaptureFallback,
    CaptureMode,
    PostAcceptCommitSpec,
    StackTrack,
)

from .profiles import (
    A2_A3_INT_ONLY_FAMILIES,
    normalize_canonical_model,
    normalize_hw_family,
)

#: Ascend ``npu_fused_infer_attention_score`` maximum tokens per decode round.
ASCEND_FIA_MAX_DECODE_TOKENS = 16
MAX_SPECULATIVE_TOKENS = ASCEND_FIA_MAX_DECODE_TOKENS - 1
EP_WEIGHT_FILTER_MIN_EP = 8
A2_MAX_INTRA_NODE_TP = 8
A3_DUAL_DIE_NODE_RANKS = 16
A2_GDN_STRIDED_RECURRENT_MIN_VERIFY_TOKENS = 9

DEFAULT_MAX_NUM_SEQS = 64
DEFAULT_CAPTURE_BATCH_TIERS: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64)
DEFAULT_PIECEWISE_TOKEN_TIERS: tuple[int, ...] = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
)
DEFAULT_PROMPT_LOOKUP_MAX = 4
DEFAULT_MTP_K = 3
DEFAULT_DSPARK_K = 5
DSPARK_MIN_K = 3
DSPARK_LONG_K = 7
EAGLE3_DEFAULT_K = 3
EAGLE_DEFAULT_K = 2
NGRAM_DEFAULT_K = 5
DFLASH_DEFAULT_K = 7
DFLASH2_DEFAULT_K = 8
DFLASH_MAX_K = 15
DEEPSEEK_V3_DEFAULT_K = 2
GLM5_MTP_MAX_K = 5
QWEN24T_DEFAULT_K = 1

DEFAULT_RECOVER_SSM_BATCHED_LAUNCHES = 1
DEFAULT_ACCEPTANCE_PREFIX = 1

SUPPORTED_SPECULATIVE_METHODS = frozenset({
    "ngram", "medusa", "eagle", "eagle3", "mtp",
    "deepseek_mtp", "qwen3_5_mtp", "dflash", "dspark", "draft_model",
})
NONCAUSAL_BLOCK_DRAFT_METHODS = frozenset({"dspark", "dflash"})
SERIAL_DRAFT_METHODS = frozenset({
    "mtp", "deepseek_mtp", "qwen3_5_mtp", "eagle", "eagle3", "draft_model", "medusa",
})
ZERO_NEURAL_DRAFT_METHODS = frozenset({"ngram"})
FULL_GRAPH_CUDAGRAPH_MODES = frozenset({"FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"})

_CUDAGRAPH_MODE_MAP: dict[str, CaptureMode] = {
    "NONE": "none",
    "EAGER": "none",
    "PIECEWISE": "piecewise",
    "FULL": "full",
    "FULL_DECODE_ONLY": "full_decode_only",
    "FULL_AND_PIECEWISE": "full_and_piecewise",
}

HYBRID_RECURRENT_MODELS = frozenset({
    "qwen3-next-80b-a3b",
    "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b", "qwen3.5-moe",
    "qwen3.6-27b", "qwen3.6-35b-a3b",
    "qwen3.8-27b", "qwen3.8-2.4t-a95b", "qwen3.8-flash-next",
    "kimi-k3", "kimi-k3.1",
    "glm-5.3-flash",
    "bailing-v3",
})
MULTIMODAL_MODELS = frozenset({
    "kimi-k3", "kimi-k3.1", "kimi-k2.5", "kimi-k2.6",
    "deepseek-v4.1-flash",
    "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b",
    "qwen3.6-27b", "qwen3.6-35b-a3b",
    "qwen3.8-27b", "qwen3.8-flash-next",
    "qwen3-vl-8b", "qwen3-vl-30b-a3b", "qwen3-vl-32b", "qwen3-vl-235b-a22b",
    "minimax-m3",
    "gemma-4-31b", "gemma-4-26b-a4b",
})
LAZY_LOAD_STRATEGY_MODELS = frozenset({
    "qwen3.8-2.4t-a95b",
    "deepseek-v4.1-flash",
    "qwen3.6-35b-a3b",
})
PREFETCH_LOAD_STRATEGY_MODELS = frozenset({
    "deepseek-v4-flash-0731", "deepseek-v4-pro-0813", "deepseek-v4-10t",
    "glm-5.3", "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5",
    "kimi-k3", "kimi-k3.1", "kimi-k2.6", "kimi-k2.5",
    "minimax-m3", "minimax-m3.1-flash",
    "step-5-preview",
    "qwen3.5-397b-a17b", "qwen3.8-flash-next",
})

_SHIPPING_KDA_RECOVERSSM_MODELS = frozenset({"kimi-k3", "kimi-k3.1"})
_OPT_IN_KDA_RECOVERSSM_MODELS = frozenset({"glm-5.3-flash"})
_SPEC_MODEL_ALIASES: dict[str, str] = {
    "meta-llama/meta-llama-3.1-8b-instruct": "llama-3.1-8b",
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b",
    "llama-3.1-8b-instruct": "llama-3.1-8b",
    "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b",
    "llama3": "llama-3",
    "llama-3": "llama-3",
    "qwen/qwen3-8b": "qwen3-8b",
    "qwen3-8b": "qwen3-8b",
    "qwen/qwen3-14b": "qwen3-14b",
    "qwen3-14b": "qwen3-14b",
    "qwen/qwen3-32b": "qwen3-32b",
    "qwen/qwen3-30b-a3b": "qwen3-30b-a3b",
}


@dataclass(frozen=True, slots=True)
class SpeculativeCompanion:
    """One documented speculative decoding companion or native draft head."""

    canonical_model: str
    method: str
    draft_model: str
    is_builtin: bool
    num_speculative_tokens: int
    min_speculative_tokens: int
    max_speculative_tokens: int
    draft_tp_policy: str = "auto"
    draft_kv_dtype_on_fp8_target: str = ""
    enforce_eager_draft: bool | None = None
    requires_no_prefix_caching: bool = False
    prompt_lookup_max: int = 0
    draft_sample_method: str = ""
    hw_draft_overrides: tuple[tuple[str, str], ...] = ()
    pd_matching_spec_tokens: bool = False
    is_primary: bool = True
    doc_source: str = ""
    notes: str = ""

    def draft_model_for_hw(self, hw_family: str) -> str:
        hw = normalize_hw_family(hw_family)
        for key, repo in self.hw_draft_overrides:
            if key == hw:
                return repo
        return self.draft_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_model": self.canonical_model,
            "method": self.method,
            "draft_model": self.draft_model,
            "is_builtin": self.is_builtin,
            "num_speculative_tokens": self.num_speculative_tokens,
            "min_speculative_tokens": self.min_speculative_tokens,
            "max_speculative_tokens": self.max_speculative_tokens,
            "draft_tp_policy": self.draft_tp_policy,
            "draft_kv_dtype_on_fp8_target": self.draft_kv_dtype_on_fp8_target,
            "enforce_eager_draft": self.enforce_eager_draft,
            "requires_no_prefix_caching": self.requires_no_prefix_caching,
            "prompt_lookup_max": self.prompt_lookup_max,
            "draft_sample_method": self.draft_sample_method,
            "hw_draft_overrides": dict(self.hw_draft_overrides),
            "pd_matching_spec_tokens": self.pd_matching_spec_tokens,
            "is_primary": self.is_primary,
            "doc_source": self.doc_source,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ResolvedSpeculativeCompanion:
    """Resolved speculative decoding config tailored to target TP, KV dtype, and hardware."""

    canonical_model: str
    hw_family: str
    companion: SpeculativeCompanion
    method: str
    draft_model: str
    num_speculative_tokens: int
    draft_tensor_parallel_size: int | None
    draft_kv_cache_dtype: str
    speculative_config: dict[str, Any]
    alternatives: tuple[SpeculativeCompanion, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_model": self.canonical_model,
            "hw_family": self.hw_family,
            "method": self.method,
            "draft_model": self.draft_model,
            "is_builtin": self.companion.is_builtin,
            "num_speculative_tokens": self.num_speculative_tokens,
            "draft_tensor_parallel_size": self.draft_tensor_parallel_size,
            "draft_kv_cache_dtype": self.draft_kv_cache_dtype,
            "speculative_config": dict(self.speculative_config),
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "warnings": list(self.warnings),
            "doc_source": self.companion.doc_source,
        }


@dataclass(frozen=True, slots=True)
class RecoverSsmPolicy:
    """Resolved vLLM-Ascend RecoverSSM recurrent state verification/commit policy."""

    enabled: bool
    kind: str
    readiness: str
    source_ref: str
    owner_count: int
    batched_kernel_launches: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "kind": self.kind,
            "readiness": self.readiness,
            "source_ref": self.source_ref,
            "owner_count": self.owner_count,
            "batched_kernel_launches": self.batched_kernel_launches,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class VaServingRules:
    """Validated vLLM-Ascend serving configuration, CLI flags, and ISA/PD guidance."""

    canonical_model: str
    hw_family: str
    effective_kv_cache_dtype: str
    is_hybrid_recurrent: bool
    is_multimodal: bool
    speculative_enabled: bool
    resolved_speculative: ResolvedSpeculativeCompanion | None
    speculative_config: dict[str, Any]
    prefill_speculative_config: dict[str, Any]
    decode_speculative_config: dict[str, Any]
    compilation_config: dict[str, Any]
    additional_config: dict[str, Any]
    safetensors_load_strategy: str
    enable_ep_weight_filter: bool
    mm_processor_cache_gb: int | None
    cli_flags: tuple[str, ...]
    cli_command_snippet: str
    collective_guidance: str
    hw_isa_guidance: str
    rules_applied: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_model": self.canonical_model,
            "hw_family": self.hw_family,
            "effective_kv_cache_dtype": self.effective_kv_cache_dtype,
            "is_hybrid_recurrent": self.is_hybrid_recurrent,
            "is_multimodal": self.is_multimodal,
            "speculative_enabled": self.speculative_enabled,
            "resolved_speculative": (
                self.resolved_speculative.to_dict() if self.resolved_speculative else None
            ),
            "speculative_config": dict(self.speculative_config),
            "prefill_speculative_config": dict(self.prefill_speculative_config),
            "decode_speculative_config": dict(self.decode_speculative_config),
            "compilation_config": dict(self.compilation_config),
            "additional_config": dict(self.additional_config),
            "safetensors_load_strategy": self.safetensors_load_strategy,
            "enable_ep_weight_filter": self.enable_ep_weight_filter,
            "mm_processor_cache_gb": self.mm_processor_cache_gb,
            "cli_flags": list(self.cli_flags),
            "cli_command_snippet": self.cli_command_snippet,
            "collective_guidance": self.collective_guidance,
            "hw_isa_guidance": self.hw_isa_guidance,
            "rules_applied": list(self.rules_applied),
            "warnings": list(self.warnings),
        }


def _c(
    model: str, method: str, draft_model: str, k: int, *,
    is_builtin: bool = False, k_min: int = 1, k_max: int = MAX_SPECULATIVE_TOKENS,
    draft_tp_policy: str = "auto", draft_kv_on_fp8: str = "",
    enforce_eager: bool | None = None, no_prefix_cache: bool = False,
    prompt_lookup_max: int = 0, sample_method: str = "",
    hw_overrides: tuple[tuple[str, str], ...] = (),
    pd_match: bool = False, primary: bool = True, doc: str = "", notes: str = "",
) -> SpeculativeCompanion:
    return SpeculativeCompanion(
        canonical_model=model, method=method, draft_model=draft_model,
        is_builtin=is_builtin, num_speculative_tokens=k,
        min_speculative_tokens=k_min, max_speculative_tokens=k_max,
        draft_tp_policy=draft_tp_policy, draft_kv_dtype_on_fp8_target=draft_kv_on_fp8,
        enforce_eager_draft=enforce_eager, requires_no_prefix_caching=no_prefix_cache,
        prompt_lookup_max=prompt_lookup_max, draft_sample_method=sample_method,
        hw_draft_overrides=hw_overrides, pd_matching_spec_tokens=pd_match,
        is_primary=primary, doc_source=doc, notes=notes,
    )


_DOC_SD = "user_guide/feature_guide/speculative_decoding.md"
_BASE_SPECULATIVE_COMPANIONS: dict[str, tuple[SpeculativeCompanion, ...]] = {
    "kimi-k3.1": (
        _c("kimi-k3.1", "dspark", "Eco-Tech/Kimi-K3-DSpark-MLA-NPU", DEFAULT_DSPARK_K,
           k_min=DSPARK_MIN_K, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target",
           draft_kv_on_fp8="bfloat16", sample_method="greedy", pd_match=True,
           doc="tutorials/models/Kimi-K3.1-Pro.md",
           notes="DSpark external draft; draft_tensor_parallel_size must equal target TP"),
        _c("kimi-k3.1", "dspark", "Inferact/Kimi-K3-DSpark-Block5", DEFAULT_DSPARK_K,
           k_min=DSPARK_MIN_K, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target",
           draft_kv_on_fp8="bfloat16", sample_method="greedy", pd_match=True, primary=False,
           doc="tutorials/models/Kimi-K3.md",
           notes="950DT DSpark Block-5 alternative draft checkpoint"),
    ),
    "kimi-k3": (
        _c("kimi-k3", "dspark", "Eco-Tech/Kimi-K3-DSpark-MLA-NPU", DEFAULT_DSPARK_K,
           k_min=DSPARK_MIN_K, k_max=DSPARK_LONG_K, draft_tp_policy="match_target",
           draft_kv_on_fp8="bfloat16", sample_method="greedy", pd_match=True,
           hw_overrides=(("a3", "Eco-Tech/Kimi-K3-DSpark-MLA-NPU"), ("a5", "Eco-Tech/Kimi-K3-DSpark-MLA-NPU")),
           doc="tutorials/models/Kimi-K3.md",
           notes="DSpark MLA draft with merged RecoverSSM #21549"),
        _c("kimi-k3", "dspark", "Inferact/Kimi-K3-DSpark-Block5", DEFAULT_DSPARK_K,
           k_min=DSPARK_MIN_K, k_max=DSPARK_LONG_K, draft_tp_policy="match_target",
           draft_kv_on_fp8="bfloat16", sample_method="greedy", pd_match=True, primary=False,
           doc="tutorials/models/Kimi-K3.md"),
    ),
    "kimi-k2.6": (
        _c("kimi-k2.6", "eagle3", "lightseekorg/kimi-k2.6-eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one",
           doc="tutorials/models/Kimi-K2.6.md"),
        _c("kimi-k2.6", "dflash", "z-lab/Kimi-K2.5-DFlash", DFLASH_MAX_K,
           k_min=1, k_max=DFLASH_MAX_K, no_prefix_cache=True, primary=False,
           doc="tutorials/models/Kimi-K2.6.md"),
    ),
    "kimi-k2.5": (
        _c("kimi-k2.5", "eagle3", "lightseekorg/kimi-k2.6-eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one",
           doc="tutorials/models/Kimi-K2.6.md"),
        _c("kimi-k2.5", "dflash", "z-lab/Kimi-K2.5-DFlash", DFLASH_MAX_K,
           k_min=1, k_max=DFLASH_MAX_K, no_prefix_cache=True, primary=False,
           doc="tutorials/models/Kimi-K2.6.md"),
    ),
    "minimax-m3": (
        _c("minimax-m3", "eagle3", "Inferact/MiniMax-M3-EAGLE3-GQA", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one",
           draft_kv_on_fp8="bfloat16", doc="tutorials/models/MiniMax-M3.md"),
    ),
    "minimax-m3.1-flash": (
        _c("minimax-m3.1-flash", "dspark", "", DSPARK_LONG_K, is_builtin=True,
           k_min=DSPARK_MIN_K, k_max=DSPARK_LONG_K, draft_kv_on_fp8="bfloat16",
           pd_match=True, doc="tutorials/models/MiniMax-M3.1-Flash.md",
           notes="Built-in DSpark; keep draft kv_cache_dtype=bfloat16 when target KV is fp8"),
        _c("minimax-m3.1-flash", "eagle3", "Inferact/MiniMax-M3-EAGLE3-GQA", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one", primary=False,
           doc="tutorials/models/MiniMax-M3.md"),
    ),
    "qwen3-8b": (
        _c("qwen3-8b", "eagle3", "yuhuili/EAGLE3-Qwen3-8B", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one", doc="tutorials/models/Qwen3-Dense.md"),
        _c("qwen3-8b", "dflash", "z-lab/Qwen3-8B-DFlash-b16", DFLASH_DEFAULT_K,
           k_min=1, k_max=DFLASH_MAX_K, no_prefix_cache=True, primary=False, doc=_DOC_SD),
        _c("qwen3-8b", "dspark", "deepseek-ai/dspark_qwen3_8b_block7", DSPARK_LONG_K,
           k_min=1, k_max=DSPARK_LONG_K, enforce_eager=True, no_prefix_cache=True,
           pd_match=True, primary=False, doc=_DOC_SD),
    ),
    "qwen3-14b": (
        _c("qwen3-14b", "eagle3", "AngelSlim/Qwen3-14B_eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one", doc="tutorials/models/Qwen3-Dense.md"),
        _c("qwen3-14b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "qwen3-32b": (
        _c("qwen3-32b", "eagle3", "AngelSlim/Qwen3-32B_eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one", doc="tutorials/models/Qwen3-Dense.md"),
        _c("qwen3-32b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "qwen3-30b-a3b": (
        _c("qwen3-30b-a3b", "eagle3", "AngelSlim/Qwen3-a3B_eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one", doc="tutorials/models/Qwen3-30B-A3B.md"),
        _c("qwen3-30b-a3b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "qwen3-235b-a22b": (
        _c("qwen3-235b-a22b", "eagle3", "AngelSlim/Qwen3-235B-A22B_eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one", doc="tutorials/models/Qwen3-30B-A3B.md"),
    ),
    "qwen3.8-27b": (
        _c("qwen3.8-27b", "qwen3_5_mtp", "", DEFAULT_MTP_K, is_builtin=True,
           k_min=1, k_max=DEFAULT_DSPARK_K, pd_match=True, doc="tutorials/models/Qwen3.8-27B.md"),
        _c("qwen3.8-27b", "dflash", "z-lab/Qwen3.8-27B-DFlash2", DFLASH2_DEFAULT_K,
           k_min=1, k_max=DFLASH2_DEFAULT_K, no_prefix_cache=True, pd_match=True,
           primary=False, doc=_DOC_SD, notes="DFlash2 candidate-lattice parallel drafter"),
        _c("qwen3.8-27b", "dflash", "niannianxu/Qwen3.8-27B-LiLiCorr-block8-epoch0", DFLASH_DEFAULT_K,
           k_min=1, k_max=DFLASH_DEFAULT_K, enforce_eager=True, no_prefix_cache=True,
           pd_match=True, primary=False, doc=_DOC_SD,
           notes="LiLiCorr correlator draft; requires enforce_eager=true on Ascend"),
    ),
    "qwen3.8-2.4t-a95b": (
        _c("qwen3.8-2.4t-a95b", "qwen3_5_mtp", "", QWEN24T_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DEFAULT_MTP_K, pd_match=True, doc="tutorials/models/Qwen3.8-2.4T-A95B.md"),
    ),
    "llama-3.1-8b": (
        _c("llama-3.1-8b", "eagle3", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one", doc=_DOC_SD),
        _c("llama-3.1-8b", "eagle", "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", EAGLE_DEFAULT_K,
           k_min=1, k_max=DEFAULT_MTP_K, draft_tp_policy="one", primary=False, doc=_DOC_SD),
        _c("llama-3.1-8b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "llama-3.1-70b": (
        _c("llama-3.1-70b", "eagle3", "yuhuili/EAGLE3-LLaMA3.1-Instruct-70B", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one", doc=_DOC_SD),
        _c("llama-3.1-70b", "eagle", "yuhuili/EAGLE-LLaMA3.1-Instruct-70B", EAGLE_DEFAULT_K,
           k_min=1, k_max=DEFAULT_MTP_K, draft_tp_policy="one", primary=False, doc=_DOC_SD),
        _c("llama-3.1-70b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "llama-3.3-70b": (
        _c("llama-3.3-70b", "eagle3", "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one", doc=_DOC_SD),
        _c("llama-3.3-70b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "llama-3": (
        _c("llama-3", "eagle3", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one", doc=_DOC_SD),
        _c("llama-3", "eagle", "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", EAGLE_DEFAULT_K,
           k_min=1, k_max=DEFAULT_MTP_K, draft_tp_policy="one", primary=False, doc=_DOC_SD),
        _c("llama-3", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "step-5-preview": (
        _c("step-5-preview", "mtp", "", DEFAULT_MTP_K, is_builtin=True,
           k_min=1, k_max=DEFAULT_MTP_K, doc="tutorials/models/Step-5-Preview.md"),
    ),
}


def _build_speculative_companions() -> dict[str, tuple[SpeculativeCompanion, ...]]:
    reg: dict[str, tuple[SpeculativeCompanion, ...]] = dict(_BASE_SPECULATIVE_COMPANIONS)
    for model_key in (
        "qwen3.8-flash-next", "qwen3.6-35b-a3b", "qwen3.6-27b", "qwen3.5-397b-a17b",
        "qwen3.5-122b-a10b", "qwen3.5-35b-a3b", "qwen3.5-27b", "qwen3.5-moe",
        "qwen3-next-80b-a3b",
    ):
        reg[model_key] = (
            _c(model_key, "qwen3_5_mtp", "", DEFAULT_MTP_K, is_builtin=True,
               k_min=1, k_max=DEFAULT_DSPARK_K, pd_match=True,
               doc="tutorials/models/Qwen3.8-Flash-Next.md"),
        )
    for model_key in ("deepseek-v4-flash-0731", "deepseek-v4-pro-0813", "deepseek-v4.1-flash", "deepseek-v4-10t"):
        reg[model_key] = (
            _c(model_key, "dspark", "", DEFAULT_DSPARK_K, is_builtin=True,
               k_min=DEFAULT_DSPARK_K, k_max=DSPARK_LONG_K, pd_match=True,
               doc="tutorials/models/DeepSeek-V4-Pro.md",
               notes="Built-in DSpark checkpoint; P and D must share num_speculative_tokens"),
            _c(model_key, "mtp", "", DEFAULT_MTP_K, is_builtin=True,
               k_min=1, k_max=DEFAULT_MTP_K, primary=False,
               doc="tutorials/models/DeepSeek-V4-Pro.md"),
        )
    for model_key in ("deepseek-v3", "deepseek-v3.1", "deepseek-v3.2", "deepseek-r1"):
        reg[model_key] = (
            _c(model_key, "deepseek_mtp", "", DEEPSEEK_V3_DEFAULT_K, is_builtin=True,
               k_min=1, k_max=DEFAULT_MTP_K, doc=_DOC_SD,
               notes="Single-layer exposed DeepSeek MTP; P=1, D=K under PD disaggregation"),
        )
    for model_key in ("glm-5.3", "glm-5.2", "glm-5.1", "glm-5"):
        reg[model_key] = (
            _c(model_key, "deepseek_mtp", "", DEFAULT_MTP_K, is_builtin=True,
               k_min=DEFAULT_MTP_K, k_max=GLM5_MTP_MAX_K, enforce_eager=True,
               doc="tutorials/models/GLM5.2.md"),
            _c(model_key, "dspark", "", DEFAULT_DSPARK_K, is_builtin=False,
               k_min=DSPARK_MIN_K, k_max=DSPARK_LONG_K, draft_kv_on_fp8="bfloat16",
               pd_match=True, primary=False, doc=_DOC_SD,
               notes="GLM DSpark supports Glm5DSparkForCausalLM or Qwen3DSparkModel"),
        )
    reg["glm-5.3-flash"] = (
        _c("glm-5.3-flash", "deepseek_mtp", "", GLM5_MTP_MAX_K, is_builtin=True,
           k_min=DEFAULT_MTP_K, k_max=GLM5_MTP_MAX_K, pd_match=True,
           doc="tutorials/models/GLM5.3-Flash.md",
           notes="Hybrid KDA target; P and D require matching num_speculative_tokens (opt-in --use-replayssm)"),
        _c("glm-5.3-flash", "dspark", "", DEFAULT_DSPARK_K, is_builtin=False,
           k_min=DSPARK_MIN_K, k_max=DSPARK_LONG_K, draft_kv_on_fp8="bfloat16",
           pd_match=True, primary=False, doc=_DOC_SD),
    )
    for model_key in ("kimi-k2", "kimi-k2-thinking"):
        reg[model_key] = (
            _c(model_key, "eagle3", "lightseekorg/kimi-k2.6-eagle3", EAGLE3_DEFAULT_K,
               k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="one",
               doc="tutorials/models/Kimi-K2.6.md"),
        )
    return reg


SPECULATIVE_COMPANIONS: dict[str, tuple[SpeculativeCompanion, ...]] = _build_speculative_companions()


def _resolve_spec_model_key(model_id: str) -> str:
    raw = str(model_id or "").strip().lower()
    if raw in _SPEC_MODEL_ALIASES:
        return _SPEC_MODEL_ALIASES[raw]
    canon = normalize_canonical_model(model_id)
    if canon in SPECULATIVE_COMPANIONS:
        return canon
    if canon in _SPEC_MODEL_ALIASES:
        return _SPEC_MODEL_ALIASES[canon]
    if "llama-3.3" in raw or "llama3.3" in raw:
        return "llama-3.3-70b"
    if "llama-3" in raw or "llama3" in raw:
        return "llama-3.1-70b" if "70b" in raw else ("llama-3.1-8b" if "8b" in raw else "llama-3")
    return canon


def is_hybrid_recurrent_model(model_or_id: ModelSpec | str) -> bool:
    """Return True when the model has KDA/GDN/Mamba recurrent layers."""
    if isinstance(model_or_id, ModelSpec):
        if any(
            str(plan.attention).lower() in {"kda", "gdn"}
            or str(plan.architecture).lower() in {"kda", "gdn"}
            or plan.recurrent_state_elements > 0
            for plan in model_or_id.layer_plans
        ):
            return True
        model_id = model_or_id.name
    else:
        model_id = str(model_or_id or "")
    key = _resolve_spec_model_key(model_id)
    return key in HYBRID_RECURRENT_MODELS or any(
        tok in key for tok in ("qwen3-next", "qwen3.5", "qwen3.6", "qwen3.8", "kimi-k3", "glm-5.3-flash")
    )


def is_multimodal_model(model_or_id: ModelSpec | str) -> bool:
    """Return True when the model carries a multimodal/vision tower."""
    if isinstance(model_or_id, ModelSpec):
        if model_or_id.vision is not None:
            return True
        model_id = model_or_id.name
    else:
        model_id = str(model_or_id or "")
    key = _resolve_spec_model_key(model_id)
    return key in MULTIMODAL_MODELS or "-vl-" in key


def draft_invocation_count_for_method(method: str, proposal_count: int) -> int:
    """Distinguish physical draft forward count from speculative proposal count K."""
    k = max(0, int(proposal_count))
    if k <= 0:
        return 0
    norm = str(method or "").strip().lower()
    if not norm or norm in ZERO_NEURAL_DRAFT_METHODS:
        return 0
    if norm in NONCAUSAL_BLOCK_DRAFT_METHODS:
        return 1
    return k


def clamp_fia_speculative_tokens(num_speculative_tokens: int) -> tuple[int, str | None]:
    """Clamp speculative token count to Ascend FIA ``(K + 1) <= 16`` ceiling."""
    requested = int(num_speculative_tokens)
    if requested < 0:
        return 0, f"fia_spec_tokens_negative_clamped:{requested}->0"
    if requested + 1 > ASCEND_FIA_MAX_DECODE_TOKENS:
        return (
            MAX_SPECULATIVE_TOKENS,
            f"fia_16_token_limit_clamped:{requested}->{MAX_SPECULATIVE_TOKENS}:"
            "npu_fused_infer_attention_score requires (num_speculative_tokens + 1) <= 16",
        )
    return requested, None


def normalize_capture_mode(raw_mode: str | None, *, enforce_eager: bool = False) -> CaptureMode:
    """Map vLLM-Ascend cudagraph mode string to canonical L0 ``CaptureMode`` without loss."""
    if enforce_eager:
        return "none"
    text = str(raw_mode or "FULL_DECODE_ONLY").strip().upper()
    return _CUDAGRAPH_MODE_MAP.get(text, "full_decode_only")


def compute_aclgraph_capture_sizes(
    batch_sizes: Sequence[int] | None = None,
    *,
    num_speculative_tokens: int = 0,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    cudagraph_mode: str = "FULL_DECODE_ONLY",
) -> tuple[int, ...]:
    """Generate ACLGraph capture sizes respecting ``n * (K + 1)`` under full-graph modes."""
    k, _ = clamp_fia_speculative_tokens(max(0, int(num_speculative_tokens)))
    stride = k + 1
    limit = max(1, int(max_num_seqs))
    if batch_sizes is not None:
        raw_bs = sorted({max(1, int(b)) for b in batch_sizes if int(b) > 0})
    else:
        raw_bs = [b for b in DEFAULT_CAPTURE_BATCH_TIERS if b <= limit]
        if limit not in raw_bs:
            raw_bs.append(limit)
    mode_upper = str(cudagraph_mode or "FULL_DECODE_ONLY").strip().upper()
    if mode_upper in FULL_GRAPH_CUDAGRAPH_MODES and k > 0:
        return tuple(sorted({b * stride for b in raw_bs}))
    return tuple(sorted(set(raw_bs)))


def validate_aclgraph_capture_sizes(
    capture_sizes: Sequence[int],
    *,
    num_speculative_tokens: int = 0,
    cudagraph_mode: str = "FULL_DECODE_ONLY",
) -> tuple[str, ...]:
    """Validate that capture sizes are exact multiples of ``(K + 1)`` under full graph."""
    k = max(0, int(num_speculative_tokens))
    mode_upper = str(cudagraph_mode or "FULL_DECODE_ONLY").strip().upper()
    if k <= 0 or mode_upper not in FULL_GRAPH_CUDAGRAPH_MODES:
        return ()
    stride = k + 1
    bad = [int(sz) for sz in capture_sizes if int(sz) % stride != 0]
    if not bad:
        return ()
    return (
        f"aclgraph_capture_sizes_not_multiples_of_{stride}:{bad}:"
        f"{mode_upper} with num_speculative_tokens={k} requires sizes = n * {stride}",
    )


def build_capture_bucket_rule(
    *,
    cudagraph_mode: str = "FULL_DECODE_ONLY",
    num_speculative_tokens: int = 0,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    batch_sizes: Sequence[int] | None = None,
    explicit_capture_sizes: Sequence[int] | None = None,
    enforce_eager: bool = False,
    enforce_eager_draft: bool = False,
    allow_mixed: bool = False,
    allow_ragged: bool = False,
) -> CaptureBucketRule:
    """Build non-lossy L0 ``CaptureBucketRule`` preserving ``FULL_AND_PIECEWISE``."""
    mode = normalize_capture_mode(cudagraph_mode, enforce_eager=enforce_eager)
    eff_k, _ = clamp_fia_speculative_tokens(max(0, int(num_speculative_tokens)))
    stride = eff_k + 1
    eager_scopes = ("draft",) if enforce_eager_draft else ()
    if mode == "none":
        return CaptureBucketRule(
            mode="none",
            units="verify_slots" if eff_k > 0 else "tokens",
            verify_stride=stride,
            max_slots=max(1, int(max_num_seqs)) * stride,
            allow_mixed=allow_mixed,
            allow_ragged=allow_ragged,
            fallback_mode="eager",
            enforce_eager_scopes=eager_scopes,
        )

    if explicit_capture_sizes is not None:
        full_keys = tuple(sorted({max(1, int(s)) for s in explicit_capture_sizes if int(s) > 0}))
    elif mode in {"full", "full_decode_only", "full_and_piecewise"}:
        full_keys = compute_aclgraph_capture_sizes(
            batch_sizes,
            num_speculative_tokens=eff_k,
            max_num_seqs=max_num_seqs,
            cudagraph_mode=cudagraph_mode,
        )
    else:
        full_keys = ()

    max_slots = max(full_keys) if full_keys else max(1, int(max_num_seqs)) * stride
    piecewise_keys: tuple[int, ...] = ()
    if mode in {"piecewise", "full_and_piecewise"}:
        pw_cap = max(max_slots, max(1, int(max_num_seqs)) * stride)
        pw_vals = [t for t in DEFAULT_PIECEWISE_TOKEN_TIERS if t <= pw_cap]
        if pw_cap not in pw_vals:
            pw_vals.append(pw_cap)
        for fk in full_keys:
            if fk not in pw_vals:
                pw_vals.append(fk)
        piecewise_keys = tuple(sorted(set(pw_vals)))

    fallback: CaptureFallback = "piecewise" if mode == "full_and_piecewise" else "eager"
    return CaptureBucketRule(
        mode=mode,
        units="verify_slots" if eff_k > 0 else "tokens",
        full_keys=full_keys,
        piecewise_keys=piecewise_keys,
        verify_stride=stride,
        max_slots=max_slots,
        allow_mixed=allow_mixed,
        allow_ragged=allow_ragged,
        fallback_mode=fallback,
        enforce_eager_scopes=eager_scopes,
    )


def resolve_recover_ssm_policy(
    model: ModelSpec,
    *,
    track: StackTrack,
    parallel: ParallelSpec,
    spec_method: str,
    proposal_count: int,
    opt_in_replayssm: bool = False,
) -> RecoverSsmPolicy:
    """Resolve upstream RecoverSSM policy (merged #21549 on shipping Kimi-K3/K3.1 DSpark)."""
    k = max(0, int(proposal_count))
    kda_owners = sum(
        1 for plan in model.layer_plans
        if str(plan.attention).lower() == "kda" or str(plan.architecture).lower() == "kda"
    )
    gdn_owners = sum(
        1 for plan in model.layer_plans
        if str(plan.attention).lower() == "gdn" or str(plan.architecture).lower() == "gdn"
    )
    if k <= 0 or (kda_owners == 0 and gdn_owners == 0):
        return RecoverSsmPolicy(
            enabled=False,
            kind="none",
            readiness="shipping",
            source_ref="",
            owner_count=0,
            batched_kernel_launches=0,
            reason="no_speculative_recurrent_state_recovery_required",
        )

    canon = normalize_canonical_model(model)
    method = str(spec_method or "").strip().lower()
    if kda_owners > 0:
        if canon in _SHIPPING_KDA_RECOVERSSM_MODELS:
            if parallel.pcp > 1:
                if track in {"proposed", "target"}:
                    return RecoverSsmPolicy(
                        enabled=True,
                        kind="kda",
                        readiness="proposed_pr",
                        source_ref="vllm-project/vllm-ascend#22146",
                        owner_count=kda_owners,
                        batched_kernel_launches=DEFAULT_RECOVER_SSM_BATCHED_LAUNCHES,
                        reason=f"pcp_kda_recoverssm_enabled_on_{track}:method={method or 'dspark'}",
                    )
                return RecoverSsmPolicy(
                    enabled=False,
                    kind="kda",
                    readiness="shipping",
                    source_ref="vllm-project/vllm-ascend#22146",
                    owner_count=kda_owners,
                    batched_kernel_launches=0,
                    reason="pcp_kda_recoverssm_requires_proposed_pr_22146",
                )
            if track == "eager":
                return RecoverSsmPolicy(
                    enabled=False,
                    kind="kda",
                    readiness="shipping",
                    source_ref="vllm-project/vllm-ascend#21549",
                    owner_count=kda_owners,
                    batched_kernel_launches=0,
                    reason="eager_track_disables_recoverssm_fusion",
                )
            return RecoverSsmPolicy(
                enabled=True,
                kind="kda",
                readiness="shipping",
                source_ref="vllm-project/vllm-ascend#21549",
                owner_count=kda_owners,
                batched_kernel_launches=DEFAULT_RECOVER_SSM_BATCHED_LAUNCHES,
                reason=f"merged_kda_recoverssm_with_{method or 'dspark'}:pr_21549",
            )
        if canon in _OPT_IN_KDA_RECOVERSSM_MODELS:
            if opt_in_replayssm and parallel.pp == 1 and parallel.cp == 1:
                return RecoverSsmPolicy(
                    enabled=True,
                    kind="kda",
                    readiness="shipping",
                    source_ref="vllm_ascend/models/glm5next/recoverssm_config.py",
                    owner_count=kda_owners,
                    batched_kernel_launches=DEFAULT_RECOVER_SSM_BATCHED_LAUNCHES,
                    reason="glm5next_opt_in_use_replayssm_pp1_dcp1",
                )
            reason = (
                "glm5next_replayssm_requires_pp1_dcp1"
                if opt_in_replayssm
                else "glm5next_replayssm_opt_in_flag_not_set"
            )
            return RecoverSsmPolicy(
                enabled=False,
                kind="kda",
                readiness="shipping",
                source_ref="vllm_ascend/models/glm5next/recoverssm_config.py",
                owner_count=kda_owners,
                batched_kernel_launches=0,
                reason=reason,
            )

    if gdn_owners > 0:
        if track in {"proposed", "target"}:
            return RecoverSsmPolicy(
                enabled=True,
                kind="gdn",
                readiness="proposed_pr",
                source_ref="vllm-project/vllm-ascend#22045",
                owner_count=gdn_owners,
                batched_kernel_launches=DEFAULT_RECOVER_SSM_BATCHED_LAUNCHES,
                reason=f"gdn_recoverssm_enabled_on_{track}:pr_22045",
            )
        return RecoverSsmPolicy(
            enabled=False,
            kind="gdn",
            readiness="shipping",
            source_ref="vllm-project/vllm-ascend#22045",
            owner_count=gdn_owners,
            batched_kernel_launches=0,
            reason="gdn_recoverssm_runtime_wiring_requires_pr_22045",
        )

    return RecoverSsmPolicy(
        enabled=False,
        kind="none",
        readiness="shipping",
        source_ref="",
        owner_count=0,
        batched_kernel_launches=0,
        reason="recoverssm_not_configured_for_model",
    )


def build_post_accept_commits(
    policy: RecoverSsmPolicy,
    *,
    recurrent_abi_keys: Sequence[str] = (),
    request_id: str = "req_0",
    slot_id: int = 0,
    accepted_prefix: int = DEFAULT_ACCEPTANCE_PREFIX,
    base_version: int = 0,
    crosses_checkpoint_boundary: bool = False,
) -> tuple[PostAcceptCommitSpec, ...]:
    """Emit batched causal ``PostAcceptCommitSpec`` when RecoverSSM is active."""
    if not policy.enabled or policy.owner_count <= 0:
        return ()
    anchor_key = (
        str(recurrent_abi_keys[0])
        if recurrent_abi_keys
        else f"target.L0.{policy.kind}_recurrent_state"
    )
    next_version = int(base_version) + 1
    return (
        PostAcceptCommitSpec(
            recurrent_abi_key=anchor_key,
            request_id=request_id,
            slot_id=slot_id,
            accepted_prefix=max(0, int(accepted_prefix)),
            base_version=int(base_version),
            published_version=next_version,
            verify_record_ready_event=f"ev.verify_record_ready.{request_id}.v{base_version}",
            acceptance_ready_event=f"ev.acceptance_ready.{request_id}.v{base_version}",
            base_version_ready_event=f"ev.recurrent_state_ready.{request_id}.v{base_version}",
            publish_event=f"ev.recurrent_state_published.{request_id}.v{next_version}",
            dependent_consumer_events=(
                f"ev.token_commit.{request_id}.v{next_version}",
                f"ev.next_step_recurrent_read.{request_id}.v{next_version}",
            ),
            crosses_checkpoint_boundary=bool(crosses_checkpoint_boundary),
            owner_count=policy.owner_count,
            batched_kernel_launches=max(1, policy.batched_kernel_launches),
        ),
    )


def list_speculative_companions(model_id: str) -> tuple[SpeculativeCompanion, ...]:
    """List all documented speculative decoding options for ``model_id``."""
    key = _resolve_spec_model_key(model_id)
    if key in SPECULATIVE_COMPANIONS:
        return SPECULATIVE_COMPANIONS[key]
    return (
        _c(key or "generic", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           doc=_DOC_SD,
           notes="Generic n-gram fallback when no neural draft checkpoint is registered"),
    )


def resolve_speculative_companion(
    model_id: str,
    *,
    target_tp: int = 1,
    target_kv_dtype: str = "bf16",
    hw_family: str = "a5",
    method: str | None = None,
    num_speculative_tokens: int | None = None,
    draft_model: str | None = None,
) -> ResolvedSpeculativeCompanion:
    """Resolve the primary or method-selected speculative companion with Ascend rules applied."""
    key = _resolve_spec_model_key(model_id)
    hw = normalize_hw_family(hw_family)
    tp = max(1, int(target_tp))
    companions = list_speculative_companions(key)
    warnings: list[str] = []

    selected = companions[0]
    if method:
        req_m = str(method).strip().lower()
        matched = next((c for c in companions if c.method == req_m), None)
        if matched is not None:
            selected = matched
        else:
            warnings.append(f"spec_method_not_registered_for_{key}:{req_m}:synthesizing_companion")
            selected = _c(
                key, req_m, draft_model or "",
                num_speculative_tokens if num_speculative_tokens is not None else DEFAULT_MTP_K,
                is_builtin=req_m in {"mtp", "deepseek_mtp", "qwen3_5_mtp", "ngram"},
                draft_tp_policy="one" if req_m in {"eagle", "eagle3"} else "auto",
                pd_match=is_hybrid_recurrent_model(key) or req_m == "dspark",
            )

    raw_k = selected.num_speculative_tokens if num_speculative_tokens is None else int(num_speculative_tokens)
    eff_k, clamp_warn = clamp_fia_speculative_tokens(raw_k)
    if clamp_warn:
        warnings.append(clamp_warn)

    resolved_draft = str(draft_model) if draft_model else selected.draft_model_for_hw(hw)
    if selected.draft_tp_policy == "one" or selected.method in {"eagle", "eagle3"}:
        draft_tp: int | None = 1
    elif selected.draft_tp_policy == "match_target":
        draft_tp = tp
    else:
        draft_tp = None

    kv_lower = str(target_kv_dtype or "bf16").strip().lower()
    needs_bf16_draft_kv = bool(
        selected.draft_kv_dtype_on_fp8_target
        and (
            kv_lower in {"fp8", "fp8_e4m3", "mxfp8"}
            or (hw in A2_A3_INT_ONLY_FAMILIES and kv_lower in {"int8", "c8"} and selected.method == "dspark")
        )
    )
    draft_kv = selected.draft_kv_dtype_on_fp8_target if needs_bf16_draft_kv else ""

    spec_cfg: dict[str, Any] = {
        "method": selected.method,
        "num_speculative_tokens": eff_k,
    }
    if resolved_draft:
        spec_cfg["model"] = resolved_draft
    if draft_tp is not None:
        spec_cfg["draft_tensor_parallel_size"] = draft_tp
    if draft_kv:
        spec_cfg["kv_cache_dtype"] = draft_kv
    if selected.draft_sample_method:
        spec_cfg["draft_sample_method"] = selected.draft_sample_method
    if selected.enforce_eager_draft is not None:
        spec_cfg["enforce_eager"] = selected.enforce_eager_draft
    if selected.prompt_lookup_max > 0:
        spec_cfg["prompt_lookup_max"] = selected.prompt_lookup_max

    if hw == "a2" and is_hybrid_recurrent_model(key) and (eff_k + 1) >= A2_GDN_STRIDED_RECURRENT_MIN_VERIFY_TOKENS:
        warnings.append(
            f"a2_gdn_strided_recurrent_workaround:verify_tokens={eff_k + 1}:"
            "A2 BF16 GDN >= 9 verify tokens uses strided Triton recurrent update"
        )

    alts = tuple(c for c in companions if c != selected)
    return ResolvedSpeculativeCompanion(
        canonical_model=key, hw_family=hw, companion=selected,
        method=selected.method, draft_model=resolved_draft,
        num_speculative_tokens=eff_k, draft_tensor_parallel_size=draft_tp,
        draft_kv_cache_dtype=draft_kv, speculative_config=spec_cfg,
        alternatives=alts, warnings=tuple(warnings),
    )


def _collective_and_isa_guidance(hw_family: str, *, tp: int, ep: int, pp: int) -> tuple[str, str]:
    hw = normalize_hw_family(hw_family)
    if hw == "a2":
        tp_note = (
            f"TP={tp} crosses single-node 8-card HCCS domain onto RoCE; keep TP <= {A2_MAX_INTRA_NODE_TP}"
            if tp > A2_MAX_INTRA_NODE_TP else
            f"TP={tp} fits inside single-node 8-card HCCS AIV full-mesh; cross-node traffic (EP={ep}, PP={pp}) rides RoCE"
        )
        isa = (
            "Atlas A2 (910B): INT8 (W8A8_DYNAMIC) / INT4 routed experts (W4A8_DYNAMIC) + "
            "C8 INT8 KV cache only; enable FRACTAL_NZ weight layout for integer weight ABIs."
        )
        return tp_note, isa
    if hw == "a3":
        node_note = (
            f"Atlas A3 dual-Die (910C, {A3_DUAL_DIE_NODE_RANKS} dies/node): align TP={tp} pairs "
            f"across intra-package SIO links, use HCCS_SW AIV expansion, and select eligible MC2 for EP={ep}."
        )
        isa = (
            "Atlas A3 (910C dual-Die): INT8 (W8A8_DYNAMIC) / INT4 routed experts (W4A8_DYNAMIC) + "
            "C8 INT8 KV cache only; enable FRACTAL_NZ weight layout for integer weight ABIs."
        )
        return node_note, isa
    if hw == "a5":
        coll = (
            f"Atlas A5 (950DT/950PR, DAV_3510): hardware CCU + UB-Direct fabric across TP={tp}, EP={ep}, PP={pp}."
        )
        isa = (
            "Atlas A5 (950DT/950PR): native MXFP8/MXFP4 Tensor Cores (group=32, E8M0) and FP8 KV cache; "
            "keep DSpark/EAGLE3 draft kv_cache_dtype=bfloat16 when target KV is fp8."
        )
        return coll, isa
    return (
        f"Hardware ({hw_family}): TP={tp}, EP={ep}, PP={pp}.",
        "Use declared device compute precision and KV cache formats.",
    )


def build_pd_spec_configs(
    resolved: ResolvedSpeculativeCompanion, *, is_hybrid: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Build Prefill and Decode speculative configs enforcing matching-K for hybrid/DSpark."""
    base_d = dict(resolved.speculative_config)
    base_p = dict(resolved.speculative_config)
    must_match = bool(is_hybrid or resolved.companion.pd_matching_spec_tokens or resolved.method == "dspark")
    if must_match:
        base_p["enforce_eager"] = True
        rule = (
            f"pd_spec_rule_matching_k:hybrid_or_dspark requires num_speculative_tokens="
            f"{resolved.num_speculative_tokens} on both Prefill and Decode nodes"
        )
    else:
        base_p["num_speculative_tokens"] = 1
        base_p["enforce_eager"] = True
        rule = (
            f"pd_spec_rule_asymmetric_p1_dk:pure_attention sets num_speculative_tokens=1 on Prefill "
            f"and {resolved.num_speculative_tokens} on Decode"
        )
    return base_p, base_d, rule


def build_va_serving_rules(
    model_id: str,
    *,
    hw_family: str = "a5",
    tp: int = 1,
    dp: int = 1,
    ep: int = 1,
    pp: int = 1,
    target_kv_dtype: str = "bf16",
    enable_speculative: bool = True,
    spec_method: str | None = None,
    num_speculative_tokens: int | None = None,
    draft_model: str | None = None,
    cudagraph_mode: str = "FULL_DECODE_ONLY",
    batch_sizes: Sequence[int] | None = None,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    pd_disaggregated: bool = False,
    pd_role: str = "colocated",
    text_only_serving: bool = True,
    is_hybrid_recurrent: bool | None = None,
    is_multimodal: bool | None = None,
    safetensors_load_strategy: str | None = None,
    enable_weight_nz_layout: bool | None = None,
    enable_cpu_binding: bool = True,
    enable_replayssm: bool = False,
) -> VaServingRules:
    """Validate and emit official vLLM-Ascend speculative, ACLGraph, and loading rules."""
    del dp
    key = _resolve_spec_model_key(model_id)
    hw = normalize_hw_family(hw_family)
    hybrid = is_hybrid_recurrent_model(key) if is_hybrid_recurrent is None else bool(is_hybrid_recurrent)
    multimodal = is_multimodal_model(key) if is_multimodal is None else bool(is_multimodal)
    warnings: list[str] = []
    rules: list[str] = []

    kv_lower = str(target_kv_dtype or "bf16").strip().lower()
    if hw in A2_A3_INT_ONLY_FAMILIES and kv_lower in {"fp8", "fp8_e4m3", "mxfp8"}:
        warnings.append(
            f"a2_a3_fp8_kv_unsupported_using_int8_c8:{hw}:requested={target_kv_dtype}->int8"
        )
        eff_kv = "int8"
    elif kv_lower == "c8":
        eff_kv = "int8"
    else:
        eff_kv = kv_lower or "bf16"

    resolved: ResolvedSpeculativeCompanion | None = None
    spec_cfg: dict[str, Any] = {}
    p_spec_cfg: dict[str, Any] = {}
    d_spec_cfg: dict[str, Any] = {}
    eff_k = 0
    if enable_speculative:
        resolved = resolve_speculative_companion(
            key, target_tp=tp, target_kv_dtype=eff_kv, hw_family=hw,
            method=spec_method, num_speculative_tokens=num_speculative_tokens,
            draft_model=draft_model,
        )
        warnings.extend(resolved.warnings)
        eff_k = resolved.num_speculative_tokens
        rules.append(f"fia_16_token_ceiling_checked:K={eff_k}")
        if resolved.draft_tensor_parallel_size is not None:
            rules.append(f"draft_tp_rule:{resolved.method}:draft_tp={resolved.draft_tensor_parallel_size}")
        if pd_disaggregated:
            p_spec_cfg, d_spec_cfg, pd_rule = build_pd_spec_configs(resolved, is_hybrid=hybrid)
            rules.append(pd_rule)
            role_low = str(pd_role or "decode").strip().lower().removeprefix("pd_")
            spec_cfg = dict(p_spec_cfg if role_low == "prefill" else d_spec_cfg)
            eff_k = int(spec_cfg.get("num_speculative_tokens", eff_k))
        else:
            spec_cfg = dict(resolved.speculative_config)

    if enable_replayssm:
        if pp != 1:
            warnings.append(f"replayssm_requires_pp_1:got_pp={pp}")
        if not enable_speculative:
            warnings.append("replayssm_requires_speculative_decoding")
        rules.append("kda_recoverssm_enabled")

    capture_sizes = compute_aclgraph_capture_sizes(
        batch_sizes, num_speculative_tokens=eff_k,
        max_num_seqs=max_num_seqs, cudagraph_mode=cudagraph_mode,
    )
    comp_cfg: dict[str, Any] = {
        "cudagraph_mode": str(cudagraph_mode or "FULL_DECODE_ONLY").upper(),
        "cudagraph_capture_sizes": list(capture_sizes),
    }
    if eff_k > 0 and comp_cfg["cudagraph_mode"] in FULL_GRAPH_CUDAGRAPH_MODES:
        rules.append(f"aclgraph_capture_multiples_of_{eff_k + 1}")

    if hw in A2_A3_INT_ONLY_FAMILIES:
        nz_enabled = True if enable_weight_nz_layout is None else bool(enable_weight_nz_layout)
    else:
        nz_enabled = False
        if enable_weight_nz_layout is True:
            warnings.append(f"weight_nz_layout_not_applicable_on_{hw}")
    add_cfg: dict[str, Any] = {}
    if enable_cpu_binding:
        add_cfg["enable_cpu_binding"] = True
        rules.append("enable_cpu_binding")
    if nz_enabled:
        add_cfg["enable_weight_nz_layout"] = True
        rules.append("enable_weight_nz_layout")

    if safetensors_load_strategy:
        load_strat = str(safetensors_load_strategy).strip().lower()
    elif key in LAZY_LOAD_STRATEGY_MODELS:
        load_strat = "lazy"
    elif key in PREFETCH_LOAD_STRATEGY_MODELS:
        load_strat = "prefetch"
    else:
        load_strat = "prefetch"
    ep_filter = int(ep) >= EP_WEIGHT_FILTER_MIN_EP
    mm_cache = 0 if (multimodal and text_only_serving) else None

    flags: list[str] = [f"--safetensors-load-strategy {load_strat}"]
    if ep_filter:
        flags.append("--enable-ep-weight-filter")
        rules.append(f"enable_ep_weight_filter:ep={ep}")
    if mm_cache is not None:
        flags.append(f"--mm-processor-cache-gb {mm_cache}")
        rules.append("mm_processor_cache_gb_0_text_only")
    if eff_kv in {"int8", "fp8", "fp8_e4m3"}:
        cli_kv = "fp8" if eff_kv == "fp8_e4m3" else eff_kv
        flags.append(f"--kv-cache-dtype {cli_kv}")
    if resolved is not None and resolved.companion.requires_no_prefix_caching:
        flags.append("--no-enable-prefix-caching")
        rules.append("no_enable_prefix_caching_for_block_drafter")
    if enable_replayssm:
        flags.append("--use-replayssm")
    flags.append(f"--compilation-config '{json.dumps(comp_cfg, separators=(',', ':'))}'")
    if spec_cfg:
        flags.append(f"--speculative-config '{json.dumps(spec_cfg, separators=(',', ':'))}'")
    if add_cfg:
        flags.append(f"--additional-config '{json.dumps(add_cfg, separators=(',', ':'))}'")

    coll_guidance, isa_guidance = _collective_and_isa_guidance(hw, tp=tp, ep=ep, pp=pp)
    return VaServingRules(
        canonical_model=key, hw_family=hw, effective_kv_cache_dtype=eff_kv,
        is_hybrid_recurrent=hybrid, is_multimodal=multimodal,
        speculative_enabled=bool(spec_cfg), resolved_speculative=resolved,
        speculative_config=spec_cfg, prefill_speculative_config=p_spec_cfg,
        decode_speculative_config=d_spec_cfg, compilation_config=comp_cfg,
        additional_config=add_cfg, safetensors_load_strategy=load_strat,
        enable_ep_weight_filter=ep_filter, mm_processor_cache_gb=mm_cache,
        cli_flags=tuple(flags), cli_command_snippet=" \\\n  ".join(flags),
        collective_guidance=coll_guidance, hw_isa_guidance=isa_guidance,
        rules_applied=tuple(rules), warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = [
    "A2_GDN_STRIDED_RECURRENT_MIN_VERIFY_TOKENS",
    "A2_MAX_INTRA_NODE_TP",
    "A3_DUAL_DIE_NODE_RANKS",
    "ASCEND_FIA_MAX_DECODE_TOKENS",
    "DEFAULT_CAPTURE_BATCH_TIERS",
    "DEFAULT_MAX_NUM_SEQS",
    "DEFAULT_PIECEWISE_TOKEN_TIERS",
    "EP_WEIGHT_FILTER_MIN_EP",
    "FULL_GRAPH_CUDAGRAPH_MODES",
    "HYBRID_RECURRENT_MODELS",
    "MAX_SPECULATIVE_TOKENS",
    "MULTIMODAL_MODELS",
    "NONCAUSAL_BLOCK_DRAFT_METHODS",
    "SERIAL_DRAFT_METHODS",
    "SPECULATIVE_COMPANIONS",
    "SUPPORTED_SPECULATIVE_METHODS",
    "ZERO_NEURAL_DRAFT_METHODS",
    "RecoverSsmPolicy",
    "ResolvedSpeculativeCompanion",
    "SpeculativeCompanion",
    "VaServingRules",
    "build_capture_bucket_rule",
    "build_pd_spec_configs",
    "build_post_accept_commits",
    "build_va_serving_rules",
    "clamp_fia_speculative_tokens",
    "compute_aclgraph_capture_sizes",
    "draft_invocation_count_for_method",
    "is_hybrid_recurrent_model",
    "is_multimodal_model",
    "list_speculative_companions",
    "normalize_capture_mode",
    "resolve_recover_ssm_policy",
    "resolve_speculative_companion",
    "validate_aclgraph_capture_sizes",
]
