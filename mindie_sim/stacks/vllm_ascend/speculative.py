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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from mindie_sim.catalog.draft_spec import (
    MTP_SPECULATIVE_METHOD_ALIASES,
    NGRAM_CPU_PROVENANCE,
    NO_DRAFT_PROVENANCE,
    REAL_CHECKPOINT_PROVENANCE,
    SUPPORTED_SPECULATIVE_METHODS,
    DraftSpec,
    SpeculativeExecutionContract,
    canonical_speculative_method,
    is_mtp_speculative_method,
    is_supported_speculative_method,
)
from mindie_sim.schema.common import ModelingError
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
WEIGHT_NZ_MODE_DISABLED = 0
WEIGHT_NZ_MODE_QUANT = 1
WEIGHT_NZ_MODE_ALL = 2
VALID_WEIGHT_NZ_MODES = frozenset({
    WEIGHT_NZ_MODE_DISABLED,
    WEIGHT_NZ_MODE_QUANT,
    WEIGHT_NZ_MODE_ALL,
})

ASCEND_V2_REJECTED_SPECULATIVE_METHODS = frozenset({"medusa"})
NONCAUSAL_BLOCK_DRAFT_METHODS = frozenset({"dspark", "dflash"})
SERIAL_DRAFT_METHODS = MTP_SPECULATIVE_METHOD_ALIASES | frozenset({
    "eagle", "eagle3", "draft_model",
})
ZERO_NEURAL_DRAFT_METHODS = frozenset({"ngram", "ngram_gpu"})
FULL_GRAPH_CUDAGRAPH_MODES = frozenset({"FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"})
MRV2_SEPARATE_ROUTINE_MODES = frozenset({"FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"})
KDA_DSPARK_ALLOWED_PD_CONNECTORS = frozenset({"MooncakeConnectorV2", "MooncakePullConnector"})
KIMI_RECOVERSSM_SOURCE_REF = "vllm_ascend/models/kimi_k3/recoverssm_config.py"
RECOVER_SSM_ELIGIBLE_FAMILIES = frozenset({"kda", "gdn"})
ALL_RECURRENT_ATTN_FAMILIES = frozenset({"kda", "gdn", "mamba", "mamba2"})

_CUDAGRAPH_MODE_MAP: dict[str, CaptureMode] = {
    "NONE": "none",
    "EAGER": "none",
    "PIECEWISE": "piecewise",
    "FULL": "full",
    "FULL_DECODE_ONLY": "full_decode_only",
    "FULL_AND_PIECEWISE": "full_and_piecewise",
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
    draft_architecture: str = ""
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
    pd_match: bool = False, primary: bool = True, draft_arch: str = "", doc: str = "", notes: str = "",
) -> SpeculativeCompanion:
    return SpeculativeCompanion(
        canonical_model=model, method=method, draft_model=draft_model,
        is_builtin=is_builtin, num_speculative_tokens=k,
        min_speculative_tokens=k_min, max_speculative_tokens=k_max,
        draft_tp_policy=draft_tp_policy, draft_kv_dtype_on_fp8_target=draft_kv_on_fp8,
        enforce_eager_draft=enforce_eager, requires_no_prefix_caching=no_prefix_cache,
        prompt_lookup_max=prompt_lookup_max, draft_sample_method=sample_method,
        hw_draft_overrides=hw_overrides, pd_matching_spec_tokens=pd_match,
        is_primary=primary, draft_architecture=draft_arch, doc_source=doc, notes=notes,
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
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target",
           doc="tutorials/models/Kimi-K2.6.md"),
        _c("kimi-k2.6", "dflash", "z-lab/Kimi-K2.5-DFlash", DFLASH_MAX_K,
           k_min=1, k_max=DFLASH_MAX_K, no_prefix_cache=True, primary=False,
           doc="tutorials/models/Kimi-K2.6.md"),
    ),
    "kimi-k2.5": (
        _c("kimi-k2.5", "eagle3", "lightseekorg/kimi-k2.6-eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target",
           doc="tutorials/models/Kimi-K2.6.md"),
        _c("kimi-k2.5", "dflash", "z-lab/Kimi-K2.5-DFlash", DFLASH_MAX_K,
           k_min=1, k_max=DFLASH_MAX_K, no_prefix_cache=True, primary=False,
           doc="tutorials/models/Kimi-K2.6.md"),
    ),
    "minimax-m3": (
        _c("minimax-m3", "eagle3", "Inferact/MiniMax-M3-EAGLE3-GQA", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target",
           draft_kv_on_fp8="bfloat16", doc="tutorials/models/MiniMax-M3.md"),
    ),
    "minimax-m3.1-flash": (
        _c("minimax-m3.1-flash", "dspark", "", DSPARK_LONG_K, is_builtin=True,
           k_min=DSPARK_MIN_K, k_max=DSPARK_LONG_K, draft_kv_on_fp8="bfloat16",
           pd_match=True, doc="tutorials/models/MiniMax-M3.1-Flash.md",
           notes="Built-in DSpark; keep draft kv_cache_dtype=bfloat16 when target KV is fp8"),
        _c("minimax-m3.1-flash", "eagle3", "Inferact/MiniMax-M3-EAGLE3-GQA", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target", primary=False,
           doc="tutorials/models/MiniMax-M3.md"),
    ),
    "qwen3-8b": (
        _c("qwen3-8b", "eagle3", "yuhuili/EAGLE3-Qwen3-8B", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target", doc="tutorials/models/Qwen3-Dense.md"),
        _c("qwen3-8b", "dflash", "z-lab/Qwen3-8B-DFlash-b16", DFLASH_DEFAULT_K,
           k_min=1, k_max=DFLASH_MAX_K, no_prefix_cache=True, primary=False, doc=_DOC_SD),
        _c("qwen3-8b", "dspark", "deepseek-ai/dspark_qwen3_8b_block7", DSPARK_LONG_K,
           k_min=1, k_max=DSPARK_LONG_K, enforce_eager=True, no_prefix_cache=True,
           pd_match=True, primary=False, doc=_DOC_SD),
    ),
    "qwen3-14b": (
        _c("qwen3-14b", "eagle3", "AngelSlim/Qwen3-14B_eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target", doc="tutorials/models/Qwen3-Dense.md"),
        _c("qwen3-14b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "qwen3-32b": (
        _c("qwen3-32b", "eagle3", "AngelSlim/Qwen3-32B_eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target", doc="tutorials/models/Qwen3-Dense.md"),
        _c("qwen3-32b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "qwen3-30b-a3b": (
        _c("qwen3-30b-a3b", "eagle3", "AngelSlim/Qwen3-a3B_eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target", doc="tutorials/models/Qwen3-30B-A3B.md"),
        _c("qwen3-30b-a3b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "qwen3-235b-a22b": (
        _c("qwen3-235b-a22b", "eagle3", "AngelSlim/Qwen3-235B-A22B_eagle3", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target", doc="tutorials/models/Qwen3-30B-A3B.md"),
    ),
    "qwen3.8-27b": (
        _c("qwen3.8-27b", "qwen3_5_mtp", "", DEFAULT_MTP_K, is_builtin=True,
           k_min=1, k_max=DEFAULT_DSPARK_K, pd_match=True, doc="tutorials/models/Qwen3.8-27B.md"),
        _c("qwen3.8-27b", "dflash", "z-lab/Qwen3.8-27B-DFlash2", DFLASH2_DEFAULT_K,
           k_min=1, k_max=DFLASH2_DEFAULT_K, no_prefix_cache=True, pd_match=True,
           primary=False, draft_arch="DFlash2DraftModel", doc=_DOC_SD,
           notes="DFlash2 candidate-lattice parallel drafter"),
        _c("qwen3.8-27b", "dflash", "niannianxu/Qwen3.8-27B-LiLiCorr-block8-epoch0", DFLASH_DEFAULT_K,
           k_min=1, k_max=DFLASH_DEFAULT_K, enforce_eager=True, no_prefix_cache=True,
           pd_match=True, primary=False, draft_arch="LiLiCorrDraftModel", doc=_DOC_SD,
           notes="LiLiCorr correlator draft; requires enforce_eager=true on Ascend"),
    ),
    "qwen3.8-2.4t-a95b": (
        _c("qwen3.8-2.4t-a95b", "qwen3_5_mtp", "", QWEN24T_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DEFAULT_MTP_K, pd_match=True, doc="tutorials/models/Qwen3.8-2.4T-A95B.md"),
    ),
    "llama-3.1-8b": (
        _c("llama-3.1-8b", "eagle3", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target", doc=_DOC_SD),
        _c("llama-3.1-8b", "eagle", "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", EAGLE_DEFAULT_K,
           k_min=1, k_max=DEFAULT_MTP_K, draft_tp_policy="match_target", primary=False, doc=_DOC_SD),
        _c("llama-3.1-8b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "llama-3.1-70b": (
        _c("llama-3.1-70b", "eagle3", "yuhuili/EAGLE3-LLaMA3.1-Instruct-70B", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target", doc=_DOC_SD),
        _c("llama-3.1-70b", "eagle", "yuhuili/EAGLE-LLaMA3.1-Instruct-70B", EAGLE_DEFAULT_K,
           k_min=1, k_max=DEFAULT_MTP_K, draft_tp_policy="match_target", primary=False, doc=_DOC_SD),
        _c("llama-3.1-70b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "llama-3.3-70b": (
        _c("llama-3.3-70b", "eagle3", "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target", doc=_DOC_SD),
        _c("llama-3.3-70b", "ngram", "", NGRAM_DEFAULT_K, is_builtin=True,
           k_min=1, k_max=DFLASH_DEFAULT_K, prompt_lookup_max=DEFAULT_PROMPT_LOOKUP_MAX,
           primary=False, doc=_DOC_SD),
    ),
    "llama-3": (
        _c("llama-3", "eagle3", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", EAGLE3_DEFAULT_K,
           k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target", doc=_DOC_SD),
        _c("llama-3", "eagle", "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", EAGLE_DEFAULT_K,
           k_min=1, k_max=DEFAULT_MTP_K, draft_tp_policy="match_target", primary=False, doc=_DOC_SD),
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
               k_min=1, k_max=DEFAULT_DSPARK_K, draft_tp_policy="match_target",
               doc="tutorials/models/Kimi-K2.6.md"),
        )
    return reg


SPECULATIVE_COMPANIONS: dict[str, tuple[SpeculativeCompanion, ...]] = _build_speculative_companions()


_COMPACT_COMPANION_KEY_MAP: dict[str, str] = {
    "".join(ch for ch in key.lower() if ch.isalnum()): key
    for key in SPECULATIVE_COMPANIONS
}


def _resolve_spec_model_key(model_id: str) -> str:
    norm = normalize_canonical_model(model_id)
    if norm in SPECULATIVE_COMPANIONS:
        return norm
    hyphenated = norm.replace("_", "-")
    if hyphenated in SPECULATIVE_COMPANIONS:
        return hyphenated
    compact = "".join(ch for ch in norm.lower() if ch.isalnum())
    if compact in _COMPACT_COMPANION_KEY_MAP:
        return _COMPACT_COMPANION_KEY_MAP[compact]
    return norm


def _load_catalog_model_structurally(model_or_id: ModelSpec | str) -> ModelSpec | None:
    """Resolve a catalog model name into its structural ModelSpec without name heuristics."""
    if isinstance(model_or_id, ModelSpec):
        return model_or_id
    raw = str(model_or_id or "").strip()
    if not raw:
        return None
    from mindie_sim.model_catalog import load_model

    for candidate in dict.fromkeys((raw, _resolve_spec_model_key(raw), normalize_canonical_model(raw))):
        if not candidate:
            continue
        try:
            return load_model(candidate)
        except TimeoutError:
            raise
        except (KeyError, FileNotFoundError):
            continue
        except ModelingError as exc:
            if str(exc).startswith("model_catalog_unknown:"):
                continue
            raise
    return None


def is_hybrid_recurrent_model(model_or_id: ModelSpec | str) -> bool:
    """Return True structurally when the model declares KDA/GDN/Mamba recurrent layers."""
    model = _load_catalog_model_structurally(model_or_id)
    if model is None:
        return False
    return any(
        str(plan.attention).lower() in ALL_RECURRENT_ATTN_FAMILIES
        or int(plan.recurrent_state_elements) > 0
        for plan in model.layer_plans
    )


def is_multimodal_model(model_or_id: ModelSpec | str) -> bool:
    """Return True structurally when the model carries a multimodal/vision tower."""
    model = _load_catalog_model_structurally(model_or_id)
    if model is None:
        return False
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    return model.vision is not None or bool(facts.get("vision_present"))


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


def validate_fia_speculative_tokens(num_speculative_tokens: int) -> int:
    """Delegate strict Ascend FIA K <= 15 check to canonical L1 draft_spec validator."""
    from mindie_sim.catalog.draft_spec import validate_fia_speculative_tokens as _validate
    return _validate(num_speculative_tokens)


def validate_dcp_for_speculative(
    *,
    spec_method: str,
    draft_spec: DraftSpec | None,
    parallel: ParallelSpec,
    num_speculative_tokens: int,
    draft_tp: int | None = None,
    spec_draft_tp: int | None = None,
    dynamic_spec_widths: bool = False,
    use_v2_model_runner: bool = True,
    optimization_mode: str = "simulation",
    allow_synthetic_gqa: bool = False,
) -> None:
    """Enforce verbatim VA/platform.py:2253-2321,2143 DCP/PCP validation order."""
    norm = str(spec_method or "none").strip().lower()
    if norm in ASCEND_V2_REJECTED_SPECULATIVE_METHODS:
        raise ModelingError(f"unsupported_speculative_method_on_ascend_v2:{norm}")
    if norm != "none" and not is_supported_speculative_method(norm):
        raise ModelingError(f"unsupported_speculative_method:{norm}")
    canonical_method = canonical_speculative_method(norm)
    archetype = (
        "none" if canonical_method == "none"
        else "ngram" if canonical_method in ZERO_NEURAL_DRAFT_METHODS
        else "dspark" if canonical_method in NONCAUSAL_BLOCK_DRAFT_METHODS
        else canonical_method
    )
    provenance = (
        NO_DRAFT_PROVENANCE if archetype == "none"
        else NGRAM_CPU_PROVENANCE if archetype == "ngram"
        else (
            draft_spec.draft_provenance
            if draft_spec is not None and draft_spec.is_synthetic
            else REAL_CHECKPOINT_PROVENANCE
        )
    )
    eff_k = validate_fia_speculative_tokens(num_speculative_tokens)
    contract = SpeculativeExecutionContract(
        archetype=archetype,  # type: ignore[arg-type]
        spec_method=canonical_method,  # type: ignore[arg-type]
        num_speculative_tokens=eff_k,
        requires_draft_spec=archetype not in {"none", "ngram"},
        draft_tp=int(draft_tp or spec_draft_tp or parallel.tp or 1),
        draft_dcp=1,
        neural_draft_forwards=draft_invocation_count_for_method(canonical_method, eff_k),
        requires_draft_prepare=archetype not in {"none", "ngram"},
        optimization_mode=optimization_mode,  # type: ignore[arg-type]
        allow_synthetic_gqa=bool(allow_synthetic_gqa),
        draft_provenance=provenance,  # type: ignore[arg-type]
    )
    contract.validate_dcp_spec_gate(
        draft_spec,
        parallel,
        dynamic_spec_widths=dynamic_spec_widths,
        use_v2_model_runner=use_v2_model_runner,
        draft_tp_override=draft_tp,
        spec_draft_tp=spec_draft_tp,
    )


def _require_int_spec_k(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelingError(f"ascend_fia_speculative_tokens_invalid:{value!r}")
    return int(value)


def clamp_fia_speculative_tokens(num_speculative_tokens: int) -> tuple[int, str | None]:
    """Clamp speculative token count to Ascend FIA ``(K + 1) <= 16`` ceiling."""
    requested = _require_int_spec_k(num_speculative_tokens)
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
    if raw_mode is None or not str(raw_mode).strip():
        return "full_decode_only"
    text = str(raw_mode).strip().upper()
    if text not in _CUDAGRAPH_MODE_MAP:
        raise ModelingError(f"unsupported_cudagraph_mode:{raw_mode!r}")
    return _CUDAGRAPH_MODE_MAP[text]


def compute_aclgraph_capture_sizes(
    batch_sizes: Sequence[int] | None = None,
    *,
    num_speculative_tokens: int = 0,
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    cudagraph_mode: str = "FULL_DECODE_ONLY",
) -> tuple[int, ...]:
    """Generate ACLGraph capture sizes respecting ``n * (K + 1)`` under full-graph modes."""
    k = validate_fia_speculative_tokens(_require_int_spec_k(num_speculative_tokens))
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


def _round_up_to_stride(size: int, stride: int) -> int:
    return ((size + stride - 1) // stride) * stride


def validate_aclgraph_capture_sizes(
    capture_sizes: Sequence[int],
    *,
    num_speculative_tokens: int = 0,
    cudagraph_mode: str = "FULL_DECODE_ONLY",
    use_v2_model_runner: bool = True,
    max_cudagraph_capture_size: int | None = None,
) -> tuple[str, ...]:
    """Validate ACLGraph capture sizes under MRV2 rounding or strict ``n * (K + 1)``."""
    k = validate_fia_speculative_tokens(_require_int_spec_k(num_speculative_tokens))
    mode_upper = str(cudagraph_mode or "FULL_DECODE_ONLY").strip().upper()
    if k <= 0 or mode_upper not in FULL_GRAPH_CUDAGRAPH_MODES:
        return ()
    raw = [int(sz) for sz in capture_sizes]
    if not raw:
        return ()
    stride = k + 1
    if use_v2_model_runner and mode_upper in MRV2_SEPARATE_ROUTINE_MODES:
        max_cap = int(max_cudagraph_capture_size) if max_cudagraph_capture_size is not None else max(raw)
        captured = [_round_up_to_stride(sz, stride) for sz in raw if _round_up_to_stride(sz, stride) <= max_cap]
        if captured:
            return ()
        bad = list(raw)
    else:
        bad = [sz for sz in raw if sz % stride != 0]
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
    max_num_batched_tokens: int | None = None,
    batch_sizes: Sequence[int] | None = None,
    explicit_capture_sizes: Sequence[int] | None = None,
    enforce_eager: bool = False,
    enforce_eager_draft: bool = False,
    allow_mixed: bool = False,
    allow_ragged: bool = False,
    use_v2_model_runner: bool = True,
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

    full_keys: tuple[int, ...] = ()
    piecewise_keys: tuple[int, ...] = ()
    if explicit_capture_sizes is not None:
        explicit_sorted = tuple(sorted({max(1, int(s)) for s in explicit_capture_sizes if int(s) > 0}))
        if mode == "piecewise":
            piecewise_keys = explicit_sorted
        elif mode in {"full", "full_decode_only", "full_and_piecewise"}:
            if eff_k > 0 and use_v2_model_runner and mode in {"full_decode_only", "full_and_piecewise"} and explicit_sorted:
                max_cap = max(explicit_sorted)
                rounded = tuple(sorted({
                    _round_up_to_stride(sz, stride)
                    for sz in explicit_sorted
                    if _round_up_to_stride(sz, stride) <= max_cap
                }))
                full_keys = rounded or explicit_sorted
            else:
                full_keys = explicit_sorted
            if mode == "full_and_piecewise":
                piecewise_keys = explicit_sorted
    else:
        if mode in {"full", "full_decode_only", "full_and_piecewise"}:
            full_keys = compute_aclgraph_capture_sizes(
                batch_sizes,
                num_speculative_tokens=eff_k,
                max_num_seqs=max_num_seqs,
                cudagraph_mode=cudagraph_mode,
            )
        if mode in {"piecewise", "full_and_piecewise"}:
            max_decode_slots = max(full_keys) if full_keys else max(1, int(max_num_seqs)) * stride
            if max_num_batched_tokens is not None:
                pw_cap = max(1, int(max_num_batched_tokens))
            else:
                pw_cap = max(max_decode_slots, DEFAULT_PIECEWISE_TOKEN_TIERS[-1])
            pw_vals = [t for t in DEFAULT_PIECEWISE_TOKEN_TIERS if t <= pw_cap]
            if pw_cap not in pw_vals:
                pw_vals.append(pw_cap)
            for fk in full_keys:
                if fk <= pw_cap and fk not in pw_vals:
                    pw_vals.append(fk)
            piecewise_keys = tuple(sorted(set(pw_vals)))

    if full_keys:
        max_slots = max(full_keys)
    elif piecewise_keys:
        max_slots = max(piecewise_keys)
    else:
        max_slots = max(1, int(max_num_seqs)) * stride

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


@dataclass(frozen=True, slots=True)
class RecoverSsmEligibility:
    """Typed structural RecoverSSM eligibility and family-specific validation result."""

    enabled: bool
    explicitly_requested: bool
    policy_kind: str
    kda_owner_layers: tuple[int, ...] = ()
    gdn_owner_layers: tuple[int, ...] = ()
    readiness: str = "shipping"
    source_ref: str = ""
    rejection_reason: str = ""


def _infer_recover_ssm_policy_kind(
    model: ModelSpec,
    *,
    kda_owners: tuple[int, ...],
    gdn_owners: tuple[int, ...],
    other_rec_owners: tuple[int, ...],
    spec_method: str,
) -> str:
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    meta = dict(getattr(model, "metadata", ()) or ())
    declared = str(
        getattr(model, "recover_ssm_policy_kind", "")
        or meta.get("recover_ssm_policy_kind", "")
        or facts.get("recover_ssm_policy_kind", "")
        or ""
    ).strip().lower()
    if declared in {"kda_opt_in", "kda_dspark_mrv2", "gdn_v2", "none"}:
        if declared in {"kda_opt_in", "kda_dspark_mrv2"} and not kda_owners:
            raise ModelingError(f"recover_ssm_policy_kind_mismatch_with_layers:{declared}:no_kda_owners")
        if declared == "gdn_v2" and not gdn_owners:
            raise ModelingError(f"recover_ssm_policy_kind_mismatch_with_layers:{declared}:no_gdn_owners")
        return declared
    if kda_owners:
        aux = model.auxiliary
        attached = (aux.dspark or aux.mtp) if aux is not None else None
        has_dspark = bool(
            attached is not None
            and (
                attached.kind in {"dspark", "mtp_dspark", "dflash"}
                or bool(getattr(attached, "noncausal_query_block", False))
                or str(getattr(attached, "execution_mode", "") or "") == "block_dspark"
            )
        )
        return "kda_dspark_mrv2" if has_dspark else "kda_opt_in"
    if gdn_owners:
        return "gdn_v2"
    if other_rec_owners:
        return "unsupported_recurrent"
    return "none"


def _expected_mamba_cache_mode(enable_prefix_caching: bool) -> str:
    return "align" if enable_prefix_caching else "none"


def _has_active_binding(active_bindings: Sequence[Any], mechanism_id: str) -> bool:
    for item in active_bindings:
        mid = str(getattr(item, "mechanism_id", item) or "").strip()
        if mid == mechanism_id:
            return True
    return False


def _validate_kda_opt_in_recover_ssm(
    *,
    parallel: ParallelSpec,
    enable_prefix_caching: bool,
    mamba_cache_mode: str,
    kv_transfer_config: Mapping[str, Any] | None,
) -> str:
    expected = _expected_mamba_cache_mode(enable_prefix_caching)
    if mamba_cache_mode != expected:
        return f"mamba_cache_mode_mismatch:{mamba_cache_mode}!=expected_{expected}"
    if int(parallel.pp) != 1:
        return f"kda_opt_in_requires_pp1:pp={parallel.pp}"
    eff_dcp = int(getattr(parallel, "dcp", 0) or getattr(parallel, "cp", 1) or 1)
    if eff_dcp != 1:
        return f"kda_opt_in_requires_dcp1:dcp={eff_dcp}"
    if int(parallel.pcp) > 1:
        kv_role = str((kv_transfer_config or {}).get("kv_role", "") or "").strip()
        if kv_transfer_config is None or kv_role != "kv_producer":
            return f"kda_opt_in_pcp_requires_kv_producer:pcp={parallel.pcp}"
    return ""


def _validate_kda_dspark_recover_ssm(
    *,
    method: str,
    enable_prefix_caching: bool,
    mamba_cache_mode: str,
    kv_transfer_config: Mapping[str, Any] | None,
    model_runner_v2: bool,
    adaptive_verification: bool,
    conv_state_dim_first: bool,
) -> tuple[str, str, str]:
    if method != "dspark":
        return f"kda_dspark_requires_dspark_method:{method or 'unset'}", "shipping", KIMI_RECOVERSSM_SOURCE_REF
    if not model_runner_v2:
        return "kda_dspark_requires_model_runner_v2", "shipping", KIMI_RECOVERSSM_SOURCE_REF
    if adaptive_verification:
        return "kda_dspark_forbids_adaptive_verification", "shipping", KIMI_RECOVERSSM_SOURCE_REF
    expected = _expected_mamba_cache_mode(enable_prefix_caching)
    if mamba_cache_mode != expected:
        return f"mamba_cache_mode_mismatch:{mamba_cache_mode}!=expected_{expected}", "shipping", KIMI_RECOVERSSM_SOURCE_REF
    if kv_transfer_config is not None:
        conn = str(kv_transfer_config.get("kv_connector", "") or "").strip()
        if conn not in KDA_DSPARK_ALLOWED_PD_CONNECTORS:
            return f"kda_dspark_unsupported_kv_connector:{conn}", "shipping", KIMI_RECOVERSSM_SOURCE_REF
        if conv_state_dim_first:
            return "kda_dspark_requires_sd_conv_state_layout", "shipping", KIMI_RECOVERSSM_SOURCE_REF
    return "", "shipping", KIMI_RECOVERSSM_SOURCE_REF


def _validate_gdn_recover_ssm(
    *,
    track: StackTrack,
    parallel: ParallelSpec,
    enable_prefix_caching: bool,
    mamba_cache_mode: str,
    active_bindings: Sequence[Any],
) -> tuple[str, str, str]:
    binding_ok = track in {"proposed", "target"} or _has_active_binding(active_bindings, "recoverssm_gdn")
    if not binding_ok:
        return "gdn_recoverssm_runtime_wiring_requires_pr_22045", "shipping", "vllm-project/vllm-ascend#22045"
    expected = _expected_mamba_cache_mode(enable_prefix_caching)
    if mamba_cache_mode != expected:
        return f"mamba_cache_mode_mismatch:{mamba_cache_mode}!=expected_{expected}", "proposed_pr", "vllm-project/vllm-ascend#22045"
    eff_dcp = int(getattr(parallel, "dcp", 0) or getattr(parallel, "cp", 1) or 1)
    if int(parallel.pp) != 1 or eff_dcp != 1 or int(parallel.pcp) != 1:
        return f"gdn_recoverssm_requires_pp1_dcp1_pcp1:pp={parallel.pp}:dcp={eff_dcp}:pcp={parallel.pcp}", "proposed_pr", "vllm-project/vllm-ascend#22045"
    return "", "proposed_pr", "vllm-project/vllm-ascend#22045"


def validate_recover_ssm_eligibility(
    model: ModelSpec,
    *,
    recover_ssm_arg: bool | None,
    track: StackTrack,
    parallel: ParallelSpec,
    spec_method: str,
    proposal_count: int,
    enable_prefix_caching: bool = False,
    mamba_cache_mode: str | None = None,
    kv_transfer_config: Mapping[str, Any] | None = None,
    model_runner_v2: bool = True,
    adaptive_verification: bool = False,
    conv_state_dim_first: bool = False,
    stochastic_rounding: bool = False,
    raw_eager: bool = False,
    active_bindings: Sequence[Any] = (),
) -> RecoverSsmEligibility:
    """Validate RecoverSSM eligibility per family with zero model-name heuristics."""
    k = max(0, int(proposal_count))
    kda_owners = tuple(p.index for p in model.layer_plans if str(p.attention).lower() == "kda")
    gdn_owners = tuple(p.index for p in model.layer_plans if str(p.attention).lower() == "gdn")
    other_rec = tuple(
        p.index for p in model.layer_plans
        if int(p.recurrent_state_elements) > 0 and str(p.attention).lower() not in RECOVER_SSM_ELIGIBLE_FAMILIES
    )
    if kda_owners and gdn_owners:
        raise ModelingError("recover_ssm_mixed_kda_gdn_unsupported")
    method = str(spec_method or "").strip().lower()
    policy_kind = _infer_recover_ssm_policy_kind(
        model, kda_owners=kda_owners, gdn_owners=gdn_owners,
        other_rec_owners=other_rec, spec_method=method,
    )
    explicit = recover_ssm_arg is True
    if recover_ssm_arg is False:
        return RecoverSsmEligibility(False, False, policy_kind, kda_owners, gdn_owners)
    eff_cache_mode = (
        _expected_mamba_cache_mode(enable_prefix_caching)
        if mamba_cache_mode is None else str(mamba_cache_mode).strip().lower()
    )
    if explicit:
        if raw_eager or track == "eager":
            raise ModelingError("recover_ssm_incompatible_with_raw_eager")
        if stochastic_rounding:
            raise ModelingError("recover_ssm_incompatible_with_stochastic_rounding")
        if policy_kind in {"none", "unsupported_recurrent"}:
            raise ModelingError("recover_ssm_unsupported_recurrent_family")
        if k <= 0:
            raise ModelingError("recover_ssm_requires_positive_spec_k")
    elif k <= 0 or raw_eager or track == "eager" or stochastic_rounding or policy_kind in {"none", "unsupported_recurrent"}:
        return RecoverSsmEligibility(
            False, False, policy_kind, kda_owners, gdn_owners,
            rejection_reason="no_speculative_recurrent_state_recovery_required",
        )
    if policy_kind == "kda_opt_in":
        if method in NONCAUSAL_BLOCK_DRAFT_METHODS:
            if explicit:
                raise ModelingError(f"draft_structure_incompatible_for_speculative_method:{method}:mtp")
            return RecoverSsmEligibility(
                False, False, policy_kind, kda_owners, gdn_owners,
                source_ref="vllm_ascend/models/glm5next/recoverssm_config.py",
                rejection_reason=f"draft_structure_incompatible_for_speculative_method:{method}:mtp",
            )
        if not explicit:
            return RecoverSsmEligibility(
                False, False, policy_kind, kda_owners, gdn_owners,
                source_ref="vllm_ascend/models/glm5next/recoverssm_config.py",
                rejection_reason="kda_opt_in_replayssm_flag_not_set",
            )
        reason = _validate_kda_opt_in_recover_ssm(
            parallel=parallel, enable_prefix_caching=enable_prefix_caching,
            mamba_cache_mode=eff_cache_mode, kv_transfer_config=kv_transfer_config,
        )
        if reason:
            raise ModelingError(f"recover_ssm_topology_ineligible:{reason}")
        return RecoverSsmEligibility(
            True, True, policy_kind, kda_owners, gdn_owners,
            readiness="shipping", source_ref="vllm_ascend/models/glm5next/recoverssm_config.py",
        )
    if policy_kind == "kda_dspark_mrv2":
        if not explicit:
            return RecoverSsmEligibility(
                False, False, policy_kind, kda_owners, gdn_owners,
                readiness="shipping", source_ref=KIMI_RECOVERSSM_SOURCE_REF,
                rejection_reason="kda_dspark_replayssm_flag_not_set",
            )
        reason, readiness, ref = _validate_kda_dspark_recover_ssm(
            method=method,
            enable_prefix_caching=enable_prefix_caching, mamba_cache_mode=eff_cache_mode,
            kv_transfer_config=kv_transfer_config, model_runner_v2=model_runner_v2,
            adaptive_verification=adaptive_verification,
            conv_state_dim_first=conv_state_dim_first,
        )
        if reason:
            raise ModelingError(f"recover_ssm_topology_ineligible:{reason}")
        return RecoverSsmEligibility(True, True, policy_kind, kda_owners, gdn_owners, readiness=readiness, source_ref=ref)
    reason, readiness, ref = _validate_gdn_recover_ssm(
        track=track, parallel=parallel, enable_prefix_caching=enable_prefix_caching,
        mamba_cache_mode=eff_cache_mode, active_bindings=active_bindings,
    )
    if reason:
        if explicit:
            raise ModelingError(f"recover_ssm_topology_ineligible:{reason}")
        return RecoverSsmEligibility(
            False, False, policy_kind, kda_owners, gdn_owners,
            readiness=readiness, source_ref=ref, rejection_reason=reason,
        )
    return RecoverSsmEligibility(True, explicit, policy_kind, kda_owners, gdn_owners, readiness=readiness, source_ref=ref)


def resolve_recover_ssm_policy(
    model: ModelSpec,
    *,
    track: StackTrack,
    parallel: ParallelSpec,
    spec_method: str,
    proposal_count: int,
    opt_in_replayssm: bool = False,
    enable_prefix_caching: bool = False,
    mamba_cache_mode: str | None = None,
    kv_transfer_config: Mapping[str, Any] | None = None,
    active_bindings: Sequence[Any] = (),
    raw_eager: bool = False,
    stochastic_rounding: bool = False,
) -> RecoverSsmPolicy:
    """Resolve upstream RecoverSSM policy through the single structural eligibility validator."""
    kda_owners = tuple(p.index for p in model.layer_plans if str(p.attention).lower() == "kda")
    gdn_owners = tuple(p.index for p in model.layer_plans if str(p.attention).lower() == "gdn")
    if kda_owners and gdn_owners:
        raise ModelingError("recover_ssm_mixed_kda_gdn_unsupported")
    fam_kind = "kda" if kda_owners else ("gdn" if gdn_owners else "none")
    owners = kda_owners if kda_owners else gdn_owners
    method = str(spec_method or "").strip().lower()
    no_spec_active = max(0, int(proposal_count)) <= 0 and method in {"", "none"}
    if (not opt_in_replayssm and (max(0, int(proposal_count)) <= 0 or not owners)) or (
        no_spec_active and not (raw_eager or track == "eager" or stochastic_rounding)
    ):
        return RecoverSsmPolicy(
            enabled=False, kind="none", readiness="shipping", source_ref="",
            owner_count=0, batched_kernel_launches=0,
            reason="no_speculative_recurrent_state_recovery_required",
        )
    recover_arg: bool | None = True if opt_in_replayssm else None
    elig = validate_recover_ssm_eligibility(
        model, recover_ssm_arg=recover_arg, track=track, parallel=parallel,
        spec_method=method, proposal_count=proposal_count,
        enable_prefix_caching=enable_prefix_caching,
        mamba_cache_mode=mamba_cache_mode,
        kv_transfer_config=kv_transfer_config,
        raw_eager=raw_eager, stochastic_rounding=stochastic_rounding,
        active_bindings=active_bindings,
    )
    if not elig.enabled:
        reason = elig.rejection_reason or "recoverssm_not_configured_for_model"
        if track == "eager" and reason == "no_speculative_recurrent_state_recovery_required":
            reason = "eager_track_disables_recoverssm_fusion"
        return RecoverSsmPolicy(
            enabled=False, kind=fam_kind, readiness=elig.readiness,
            source_ref=elig.source_ref, owner_count=0,
            batched_kernel_launches=0, reason=reason,
        )
    if elig.policy_kind == "kda_dspark_mrv2":
        ok_reason = (
            f"kda_dspark_recoverssm_enabled:pcp={parallel.pcp}:pp={parallel.pp}"
            if int(parallel.pcp) > 1 or int(parallel.pp) > 1
            else "kda_dspark_recoverssm_enabled:use_replayssm"
        )
    elif elig.policy_kind == "kda_opt_in":
        ok_reason = "kda_opt_in_use_replayssm_pp1_dcp1"
    else:
        ok_reason = f"gdn_recoverssm_enabled_on_{track}:pr_22045"
    return RecoverSsmPolicy(
        enabled=True, kind=fam_kind, readiness=elig.readiness,
        source_ref=elig.source_ref, owner_count=len(owners),
        batched_kernel_launches=DEFAULT_RECOVER_SSM_BATCHED_LAUNCHES,
        reason=ok_reason,
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


def _model_has_builtin_draft(model_id: str, family: str) -> bool:
    """Return True strictly when the target model structurally carries a real built-in draft."""
    spec = _load_catalog_model_structurally(model_id)
    if spec is None:
        return False
    aux = spec.auxiliary
    if aux is not None and bool(getattr(aux, "standalone_draft", False)):
        return False
    if family == "dspark":
        if aux is not None and aux.dspark is not None and not bool(getattr(aux.dspark, "is_synthetic", False)):
            return True
        if (
            aux is not None
            and aux.mtp is not None
            and not bool(getattr(aux.mtp, "is_synthetic", False))
            and str(getattr(aux.mtp, "kind", "") or "") in {"dspark", "mtp_dspark", "dflash"}
        ):
            return True
        return False
    if family == "mtp":
        if aux is not None and aux.mtp is not None and not bool(getattr(aux.mtp, "is_synthetic", False)):
            return True
        return int(dict(spec.catalog_facts).get("num_nextn_predict_layers", 0) or 0) > 0
    return False


def list_speculative_companions(model_id: str) -> tuple[SpeculativeCompanion, ...]:
    """List all documented speculative decoding options for ``model_id`` (no fabricated fallback)."""
    key = _resolve_spec_model_key(model_id)
    if key in SPECULATIVE_COMPANIONS:
        return SPECULATIVE_COMPANIONS[key]
    return ()


_DRAFT_CONFIG_MAX_BYTES = 262144
_LILICORR_ARCH = "LiLiCorrDraftModel"


def _read_local_draft_architectures(explicit_draft: str) -> tuple[str, ...]:
    if not explicit_draft:
        return ()
    path = Path(explicit_draft)
    cfg_path = path / "config.json" if path.is_dir() else (path if path.is_file() and path.name == "config.json" else None)
    if cfg_path is None or not cfg_path.is_file():
        return ()
    try:
        if cfg_path.stat().st_size > _DRAFT_CONFIG_MAX_BYTES:
            return ()
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except (OSError, ValueError):
        return ()
    if not isinstance(data, Mapping):
        return ()
    raw_archs = data.get("architectures")
    if isinstance(raw_archs, Sequence) and not isinstance(raw_archs, (str, bytes)):
        return tuple(str(a).strip() for a in raw_archs if str(a).strip())
    return ()


def _match_companion_by_draft(
    candidates: Sequence[SpeculativeCompanion],
    explicit_draft: str,
    hw: str,
    draft_archs: Sequence[str] = (),
) -> SpeculativeCompanion | None:
    arch_set = {str(a).strip() for a in draft_archs if str(a).strip()}
    if arch_set:
        for cand in candidates:
            if cand.draft_architecture and cand.draft_architecture in arch_set:
                return cand
    draft_low = explicit_draft.strip().lower()
    if not draft_low:
        return None
    for cand in candidates:
        for repo in (cand.draft_model, cand.draft_model_for_hw(hw)):
            repo_low = repo.strip().lower()
            if repo_low and repo_low == draft_low:
                return cand
    return None


def resolve_speculative_companion(
    model_id: str,
    *,
    target_tp: int = 1,
    target_kv_dtype: str = "bf16",
    hw_family: str = "a5",
    method: str | None = None,
    num_speculative_tokens: int | None = None,
    draft_model: str | None = None,
    draft_architecture: str | None = None,
) -> ResolvedSpeculativeCompanion:
    """Resolve the primary or method-selected speculative companion with Ascend rules applied."""
    key = _resolve_spec_model_key(model_id)
    hw = normalize_hw_family(hw_family)
    tp = max(1, int(target_tp))
    companions = list_speculative_companions(key)
    warnings: list[str] = []
    explicit_draft = str(draft_model or "").strip()
    draft_archs: list[str] = list(_read_local_draft_architectures(explicit_draft))
    if draft_architecture and str(draft_architecture).strip():
        draft_archs.insert(0, str(draft_architecture).strip())
    matched_companion_exact = False

    if method:
        req_m = str(method).strip().lower()
        if req_m in ASCEND_V2_REJECTED_SPECULATIVE_METHODS:
            raise ModelingError(f"unsupported_speculative_method_on_ascend_v2:{req_m}")
        if not is_supported_speculative_method(req_m):
            raise ModelingError(f"unsupported_speculative_method:{req_m}")
        method_matches = [c for c in companions if c.method == req_m]
        if req_m in ZERO_NEURAL_DRAFT_METHODS:
            matched_zero = next((c for c in companions if c.method in ZERO_NEURAL_DRAFT_METHODS), None)
            selected = _c(
                key or str(model_id), req_m, "",
                num_speculative_tokens if num_speculative_tokens is not None else (
                    matched_zero.num_speculative_tokens if matched_zero is not None else NGRAM_DEFAULT_K
                ),
                is_builtin=True, k_min=1, k_max=MAX_SPECULATIVE_TOKENS,
                prompt_lookup_max=(
                    matched_zero.prompt_lookup_max if matched_zero is not None else DEFAULT_PROMPT_LOOKUP_MAX
                ),
                doc=_DOC_SD,
            )
        else:
            draft_hit = (
                _match_companion_by_draft(method_matches, explicit_draft, hw, draft_archs)
                if ((explicit_draft or draft_archs) and method_matches)
                else None
            )
            if draft_hit is not None:
                selected = draft_hit
                matched_companion_exact = True
            elif explicit_draft:
                is_lilicorr = _LILICORR_ARCH in draft_archs
                base_match = method_matches[0] if method_matches else None
                default_k = (
                    DFLASH_DEFAULT_K if is_lilicorr
                    else (base_match.num_speculative_tokens if base_match is not None else DEFAULT_MTP_K)
                )
                selected = _c(
                    key or str(model_id), req_m, explicit_draft,
                    num_speculative_tokens if num_speculative_tokens is not None else default_k,
                    is_builtin=False,
                    k_min=1,
                    k_max=DFLASH_DEFAULT_K if is_lilicorr else MAX_SPECULATIVE_TOKENS,
                    draft_tp_policy=(
                        base_match.draft_tp_policy
                        if base_match is not None
                        else ("match_target" if req_m in {"eagle", "eagle3", "draft_model"} else "auto")
                    ),
                    draft_kv_on_fp8=(base_match.draft_kv_dtype_on_fp8_target if base_match is not None else ""),
                    enforce_eager=True if is_lilicorr else None,
                    no_prefix_cache=(req_m in NONCAUSAL_BLOCK_DRAFT_METHODS),
                    pd_match=is_hybrid_recurrent_model(key) or req_m in NONCAUSAL_BLOCK_DRAFT_METHODS,
                    draft_arch=_LILICORR_ARCH if is_lilicorr else (draft_archs[0] if draft_archs else ""),
                )
                matched_companion_exact = is_lilicorr
            elif method_matches:
                matched = method_matches[0]
                if not matched.is_builtin and not matched.draft_model_for_hw(hw):
                    raise ModelingError(f"draft_structure_required_for_speculative_method:{req_m}")
                selected = matched
                matched_companion_exact = True
            elif is_mtp_speculative_method(req_m) and _model_has_builtin_draft(key or str(model_id), "mtp"):
                selected = _c(
                    key or str(model_id), req_m, "",
                    num_speculative_tokens if num_speculative_tokens is not None else DEFAULT_MTP_K,
                    is_builtin=True,
                    pd_match=is_hybrid_recurrent_model(key),
                )
            elif req_m in NONCAUSAL_BLOCK_DRAFT_METHODS and _model_has_builtin_draft(key or str(model_id), "dspark"):
                selected = _c(
                    key or str(model_id), req_m, "",
                    num_speculative_tokens if num_speculative_tokens is not None else DEFAULT_DSPARK_K,
                    is_builtin=True,
                    pd_match=True,
                )
            else:
                raise ModelingError(f"draft_structure_required_for_speculative_method:{req_m}")
    elif companions:
        draft_hit = _match_companion_by_draft(companions, explicit_draft, hw, draft_archs) if (explicit_draft or draft_archs) else None
        selected = draft_hit if draft_hit is not None else companions[0]
        matched_companion_exact = (not explicit_draft) or (draft_hit is not None)
    elif _model_has_builtin_draft(key or str(model_id), "dspark"):
        selected = _c(
            key or str(model_id), "dspark", "",
            num_speculative_tokens if num_speculative_tokens is not None else DEFAULT_DSPARK_K,
            is_builtin=True, pd_match=True,
        )
    elif _model_has_builtin_draft(key or str(model_id), "mtp"):
        selected = _c(
            key or str(model_id), "mtp", "",
            num_speculative_tokens if num_speculative_tokens is not None else DEFAULT_MTP_K,
            is_builtin=True, pd_match=is_hybrid_recurrent_model(key),
        )
    else:
        raise ModelingError(f"draft_structure_required_for_speculative_companion:{key or model_id}")

    if num_speculative_tokens is not None:
        eff_k = validate_fia_speculative_tokens(_require_int_spec_k(num_speculative_tokens))
        if (
            matched_companion_exact
            and selected.method not in ZERO_NEURAL_DRAFT_METHODS
            and eff_k > selected.max_speculative_tokens
        ):
            raise ModelingError(
                f"speculative_tokens_exceed_companion_limit:{selected.draft_model or selected.method}:"
                f"k={eff_k}>max={selected.max_speculative_tokens}"
            )
    else:
        eff_k, clamp_warn = clamp_fia_speculative_tokens(_require_int_spec_k(selected.num_speculative_tokens))
        if clamp_warn:
            warnings.append(clamp_warn)

    resolved_draft = str(draft_model) if draft_model else selected.draft_model_for_hw(hw)
    if selected.draft_tp_policy == "one":
        draft_tp: int | None = 1
    elif selected.draft_tp_policy == "match_target" or selected.method in {"eagle", "eagle3"}:
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


def _resolve_va_weight_nz_mode(
    hw: str,
    *,
    weight_nz_mode: int | None,
    enable_weight_nz_layout: bool | None,
    warnings: list[str],
) -> int | None:
    if weight_nz_mode is not None:
        if isinstance(weight_nz_mode, bool) or not isinstance(weight_nz_mode, int) or weight_nz_mode not in VALID_WEIGHT_NZ_MODES:
            raise ModelingError(f"weight_nz_mode_invalid:{weight_nz_mode!r}")
        return int(weight_nz_mode)
    if enable_weight_nz_layout is not None:
        requested = WEIGHT_NZ_MODE_QUANT if enable_weight_nz_layout else WEIGHT_NZ_MODE_DISABLED
        if hw in A2_A3_INT_ONLY_FAMILIES:
            return requested
        if requested != WEIGHT_NZ_MODE_DISABLED:
            warnings.append(f"weight_nz_mode_not_applicable_on_{hw}")
        return None
    return WEIGHT_NZ_MODE_QUANT if hw in A2_A3_INT_ONLY_FAMILIES else None


def build_va_serving_rules(
    model_id: str,
    *,
    hw_family: str = "a5",
    tp: int = 1,
    dp: int = 1,
    ep: int = 1,
    pp: int = 1,
    dcp: int = 1,
    pcp: int = 1,
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
    weight_nz_mode: int | None = None,
    enable_weight_nz_layout: bool | None = None,
    enable_cpu_binding: bool = True,
    enable_replayssm: bool = False,
) -> VaServingRules:
    """Validate and emit official vLLM-Ascend speculative, ACLGraph, and loading rules."""
    key = _resolve_spec_model_key(model_id)
    hw = normalize_hw_family(hw_family)
    loaded_model = _load_catalog_model_structurally(key)
    hybrid = is_hybrid_recurrent_model(loaded_model or key) if is_hybrid_recurrent is None else bool(is_hybrid_recurrent)
    multimodal = is_multimodal_model(loaded_model or key) if is_multimodal is None else bool(is_multimodal)
    warnings: list[str] = []
    rules: list[str] = []

    kv_lower = str(target_kv_dtype or "bf16").strip().lower()
    if hw in A2_A3_INT_ONLY_FAMILIES and kv_lower in {"fp8", "fp8_e4m3", "mxfp8"}:
        raise ModelingError(f"kv_cache_dtype_unsupported_on_hardware:{hw}:{target_kv_dtype}")
    eff_kv = "int8" if kv_lower == "c8" else (kv_lower or "bf16")

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
            if role_low not in {"prefill", "decode"}:
                raise ModelingError(f"unsupported_pd_role:{pd_role!r}")
            spec_cfg = dict(p_spec_cfg if role_low == "prefill" else d_spec_cfg)
            eff_k = int(spec_cfg.get("num_speculative_tokens", eff_k))
        else:
            spec_cfg = dict(resolved.speculative_config)

    if enable_replayssm:
        if not enable_speculative or eff_k <= 0:
            raise ModelingError("replayssm_requires_speculative_decoding")
        if loaded_model is not None:
            eff_spec_method = resolved.method if resolved is not None else str(spec_method or "")
            no_prefix = bool(resolved is not None and resolved.companion.requires_no_prefix_caching)
            kv_xfer_for_check: dict[str, str] | None = None
            if pd_disaggregated:
                role_low = str(pd_role or "decode").strip().lower().removeprefix("pd_")
                kv_xfer_for_check = {
                    "kv_connector": "MooncakeConnectorV2",
                    "kv_role": "kv_producer" if role_low == "prefill" else "kv_consumer",
                }
            validate_recover_ssm_eligibility(
                loaded_model,
                recover_ssm_arg=True,
                track="shipping",
                parallel=ParallelSpec(
                    dp=max(1, int(dp)), tp=max(1, int(tp)), pp=max(1, int(pp)),
                    ep=max(1, int(ep)), cp=max(1, int(dcp)), pcp=max(1, int(pcp)),
                ),
                spec_method=eff_spec_method,
                proposal_count=eff_k,
                enable_prefix_caching=not no_prefix,
                kv_transfer_config=kv_xfer_for_check,
            )
        elif pp != 1:
            raise ModelingError(f"replayssm_requires_pp_1:got_pp={pp}")
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

    resolved_nz_mode = _resolve_va_weight_nz_mode(
        hw, weight_nz_mode=weight_nz_mode,
        enable_weight_nz_layout=enable_weight_nz_layout, warnings=warnings,
    )
    add_cfg: dict[str, Any] = {}
    if enable_cpu_binding:
        add_cfg["enable_cpu_binding"] = True
        rules.append("enable_cpu_binding")
    if resolved_nz_mode is not None:
        add_cfg["weight_nz_mode"] = resolved_nz_mode
        if resolved_nz_mode != WEIGHT_NZ_MODE_DISABLED:
            rules.append(f"weight_nz_mode={resolved_nz_mode}")

    if safetensors_load_strategy:
        load_strat = str(safetensors_load_strategy).strip().lower()
    else:
        facts = dict(getattr(loaded_model, "catalog_facts", ()) or ()) if loaded_model is not None else {}
        load_strat = str(facts.get("safetensors_load_strategy") or "prefetch").strip().lower()
    ep_filter = int(ep) >= EP_WEIGHT_FILTER_MIN_EP
    mm_cache = 0 if (multimodal and text_only_serving) else None

    flags: list[str] = [f"--safetensors-load-strategy {load_strat}"]
    if ep_filter:
        flags.append("--enable-ep-weight-filter")
        rules.append(f"enable_ep_weight_filter:ep={ep}")
    if mm_cache is not None:
        flags.append(f"--mm-processor-cache-gb {mm_cache}")
        rules.append("mm_processor_cache_gb_0_text_only")
    if eff_kv in {"int8", "fp8", "fp8_e4m3", "mxfp8"}:
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
    "MAX_SPECULATIVE_TOKENS",
    "NONCAUSAL_BLOCK_DRAFT_METHODS",
    "SERIAL_DRAFT_METHODS",
    "SPECULATIVE_COMPANIONS",
    "SUPPORTED_SPECULATIVE_METHODS",
    "ZERO_NEURAL_DRAFT_METHODS",
    "RecoverSsmEligibility",
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
    "validate_dcp_for_speculative",
    "validate_fia_speculative_tokens",
    "validate_recover_ssm_eligibility",
]
