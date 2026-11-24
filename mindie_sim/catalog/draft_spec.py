"""Typed structural draft, MTP, DSpark and Eagle3 auxiliary specifications.

[Role]
  Declare geometry, context-projection mode, execution schedule and cache
  sharding contracts for optional auxiliary proposal and verification heads.

[Invariants]
  * Context independence, projection layout and DCP sharding are explicit
    structural fields; execution and residency never inspect ``source.kind``.
  * Execution schedules come from ``ALLOWED_EXECUTION_MODES`` and context
    projection modes come from ``ALLOWED_CONTEXT_PROJECTION_MODES``.

[Timeouts & Locks]
  Pure immutable dataclass construction and validation; no I/O or locks.

[Side-Effect Policy]
  No filesystem, network or global state mutation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import SYNTHETIC_EAGLE3_GQA_PROVENANCE

DEFAULT_DRAFT_PP_STAGES: tuple[int, ...] = (1, 2, 4, 8, 16)
MAX_SPECULATIVE_TOKENS = 15
GEMM_FLOPS_PER_MAC = 2
GQA_KV_BANK_MULTIPLIER = 2
MLA_KV_BANK_MULTIPLIER = 1
REAL_CHECKPOINT_PROVENANCE = "real_checkpoint"
NGRAM_CPU_PROVENANCE = "ngram_cpu_match"
NO_DRAFT_PROVENANCE = "none"

SpeculativeArchetype = Literal["none", "mtp", "dspark", "eagle", "eagle3", "ngram"]
SPECULATIVE_ARCHETYPES: frozenset[str] = frozenset({
    "none", "mtp", "dspark", "eagle", "eagle3", "ngram",
})
SpeculativeMethod = Literal["none", "mtp", "dspark", "eagle", "eagle3", "ngram", "ngram_gpu"]
SPECULATIVE_METHODS: frozenset[str] = frozenset({
    "none", "mtp", "dspark", "eagle", "eagle3", "ngram", "ngram_gpu",
})
SpeculativeDraftProvenance = Literal[
    "real_checkpoint", "ngram_cpu_match", "synthetic_eagle3_gqa", "none",
]
SPECULATIVE_DRAFT_PROVENANCES: frozenset[str] = frozenset({
    REAL_CHECKPOINT_PROVENANCE,
    NGRAM_CPU_PROVENANCE,
    SYNTHETIC_EAGLE3_GQA_PROVENANCE,
    NO_DRAFT_PROVENANCE,
})
_METHOD_TO_ARCHETYPE: dict[str, SpeculativeArchetype] = {
    "none": "none",
    "mtp": "mtp",
    "dspark": "dspark",
    "eagle": "eagle",
    "eagle3": "eagle3",
    "ngram": "ngram",
    "ngram_gpu": "ngram",
}
ALLOWED_EXECUTION_MODES: frozenset[str] = frozenset({
    "stack",
    "sequential_mtp",
    "cyclic_nextn",
    "block_dspark",
    "eagle3_serial",
})
ALLOWED_CONTEXT_PROJECTION_MODES: frozenset[str] = frozenset({
    "none",
    "fused_kv_bank",
    "per_layer_qkv",
    "mla_prolog",
    "shared_compressed_kv",
})


@dataclass(frozen=True)
class DraftSpec:
    """Optional MTP/DSpark/Eagle3 proposal, verification and Markov/confidence state."""

    kind: str
    layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_width: int
    kv_width: int
    attention_kind: str = "gqa"
    mla_variant: str = "latent_mla"
    kv_cache_elements: int = 0
    input_mode: str = "concat"
    weight_category: str = ""
    qk_norm: bool = False
    q_output_gate: bool = False
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    o_lora_rank: int = 0
    o_groups: int = 0
    num_experts: int = 0
    top_k: int = 0
    expert_intermediate_size: int = 0
    num_shared_experts: int = 0
    target_layer_ids: tuple[int, ...] = ()
    draft_compress_ratios: tuple[int, ...] = ()
    sliding_window: int = 0
    index_share_for_mtp_iteration: bool = False
    block_size: int = 1
    markov_rank: int = 0
    confidence_enabled: bool = False
    confidence_executes: bool = False
    confidence_input_width: int = 0
    target_shared_embedding: bool = False
    target_shared_lm_head: bool = False
    base_logits: bool = False
    markov_replicated: bool = True
    noncausal_query_block: bool = False
    pp_capability: tuple[int, ...] = DEFAULT_DRAFT_PP_STAGES
    vocab_size: int = 0
    mhc_mult: int = 0
    mhc_sinkhorn_iters: int = 0
    mhc_lowrank: int = 0
    mhc_final_collapse: str = ""
    activation_dtype: str = "bf16"
    input_projection_category: str = "dspark_input"
    layer_plans: tuple[Any, ...] = ()
    execution_mode: str = "stack"
    independent_context: bool = False
    context_projection_mode: str = "none"
    dcp_sharded: bool = True
    input_application: str = "per_proposal"
    input_count: int = 1
    embedding_owned: bool = False
    embedding_count: int = 0
    output_head_owned: bool = False
    output_head_count: int = 0
    max_position_embeddings: int = 0
    is_synthetic: bool = False
    draft_provenance: str = REAL_CHECKPOINT_PROVENANCE
    source: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.is_synthetic, bool):
            raise ModelingError("catalog_draft_is_synthetic_invalid")
        if not isinstance(self.draft_provenance, str) or not self.draft_provenance.strip():
            raise ModelingError("catalog_draft_provenance_required")
        prov = self.draft_provenance.strip()
        object.__setattr__(self, "draft_provenance", prov)
        if self.is_synthetic:
            if prov != SYNTHETIC_EAGLE3_GQA_PROVENANCE:
                raise ModelingError(f"catalog_draft_synthetic_provenance_invalid:{prov!r}")
            if self.kind != "eagle3":
                raise ModelingError(f"catalog_draft_synthetic_requires_eagle3:{self.kind!r}")
            if self.attention_kind != "gqa":
                raise ModelingError(f"catalog_draft_synthetic_requires_gqa:{self.attention_kind!r}")
            if self.execution_mode != "eagle3_serial":
                raise ModelingError(f"catalog_draft_synthetic_requires_eagle3_serial:{self.execution_mode!r}")
        elif prov == SYNTHETIC_EAGLE3_GQA_PROVENANCE:
            raise ModelingError("catalog_draft_real_cannot_claim_synthetic_provenance")
        for key in ("target_layer_ids", "draft_compress_ratios", "pp_capability"):
            raw = getattr(self, key)
            if isinstance(raw, (list, tuple)) and not isinstance(raw, tuple):
                object.__setattr__(self, key, tuple(int(v) for v in raw))
        if isinstance(self.layer_plans, list):
            object.__setattr__(self, "layer_plans", tuple(self.layer_plans))
        if isinstance(self.notes, list):
            object.__setattr__(self, "notes", tuple(str(v) for v in self.notes))
        if isinstance(self.source, Mapping):
            object.__setattr__(
                self,
                "source",
                tuple(sorted((str(k), str(v)) for k, v in self.source.items())),
            )
        elif isinstance(self.source, list):
            object.__setattr__(
                self, "source", tuple((str(k), str(v)) for k, v in self.source)
            )
        if not self.is_synthetic and dict(self.source).get("kind") == "explicit_synthetic_gqa_mtp":
            raise ModelingError("catalog_draft_synthetic_source_requires_synthetic_flag")
        inferred_external = (
            not self.independent_context
            and self.kind == "dspark"
            and self.noncausal_query_block
            and self.context_projection_mode == "none"
        )
        if inferred_external:
            object.__setattr__(self, "independent_context", True)
            if self.attention_kind == "gqa" and self.dcp_sharded:
                object.__setattr__(self, "dcp_sharded", False)
        if self.independent_context and self.context_projection_mode == "none":
            if self.attention_kind == "mla":
                mode = "mla_prolog"
            elif self.weight_category:
                mode = "per_layer_qkv"
            else:
                mode = "fused_kv_bank"
            object.__setattr__(self, "context_projection_mode", mode)
        if self.attention_kind == "gqa" and self.kv_cache_elements == 0:
            if self.num_kv_heads > 0 and self.head_dim > 0 and self.independent_context:
                object.__setattr__(
                    self, "kv_cache_elements", 2 * int(self.num_kv_heads) * int(self.head_dim)
                )

    @property
    def input_width(self) -> int:
        if self.kind in {"dspark", "mtp_dspark", "standalone_draft"} and self.target_layer_ids:
            return int(len(self.target_layer_ids) * self.hidden_size)
        return int(2 * self.hidden_size)

    @property
    def proposal_tokens(self) -> int:
        return int(self.block_size if self.kind in {"dspark", "mtp_dspark", "standalone_draft"} else 1)

    @property
    def verify_tokens(self) -> int:
        return int(self.block_size if self.kind in {"dspark", "mtp_dspark", "standalone_draft"} else 1)

    def validate(self) -> None:
        for name, value in (
            ("layers", self.layers),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_heads", self.num_heads),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
        ):
            if int(value) <= 0:
                raise ModelingError(f"catalog_draft_field_invalid:{name}")
        if self.q_width <= 0 or self.kv_width <= 0:
            raise ModelingError("catalog_draft_projection_width_invalid")
        if self.num_experts and self.top_k <= 0:
            raise ModelingError("catalog_draft_topk_missing")
        if self.kind in {"dspark", "mtp_dspark", "standalone_draft"} and self.block_size <= 0:
            raise ModelingError("catalog_dspark_block_missing")
        if self.kind in {"dspark", "mtp_dspark", "standalone_draft"} and self.markov_rank <= 0:
            raise ModelingError("catalog_dspark_markov_missing")
        if self.execution_mode not in ALLOWED_EXECUTION_MODES:
            raise ModelingError(f"catalog_draft_execution_mode_invalid:{self.execution_mode}")
        if self.execution_mode == "cyclic_nextn" and not self.layer_plans:
            raise ModelingError("catalog_draft_cyclic_nextn_requires_layer_plans")
        if self.context_projection_mode not in ALLOWED_CONTEXT_PROJECTION_MODES:
            raise ModelingError(
                f"catalog_draft_context_projection_mode_invalid:{self.context_projection_mode}"
            )
        if not isinstance(self.independent_context, bool):
            raise ModelingError("catalog_draft_independent_context_invalid")
        if self.context_projection_mode == "shared_compressed_kv" and self.independent_context:
            raise ModelingError("catalog_draft_shared_compressed_kv_requires_target_context")
        if not isinstance(self.dcp_sharded, bool):
            raise ModelingError("catalog_draft_dcp_sharded_invalid")
        if isinstance(self.max_position_embeddings, bool) or int(self.max_position_embeddings) < 0:
            raise ModelingError("catalog_draft_max_position_embeddings_invalid")
        if self.confidence_enabled:
            expected = self.hidden_size + self.markov_rank
            if self.confidence_input_width not in {0, expected}:
                raise ModelingError(
                    f"catalog_dspark_confidence_width_invalid:{self.confidence_input_width}!={expected}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "attention_kind": self.attention_kind,
            "mla_variant": self.mla_variant,
            "kv_cache_elements": self.kv_cache_elements,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "q_width": self.q_width,
            "kv_width": self.kv_width,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "o_lora_rank": self.o_lora_rank,
            "o_groups": self.o_groups,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_intermediate_size": self.expert_intermediate_size,
            "num_shared_experts": self.num_shared_experts,
            "target_layer_ids": list(self.target_layer_ids),
            "draft_compress_ratios": list(self.draft_compress_ratios),
            "sliding_window": self.sliding_window,
            "index_share_for_mtp_iteration": self.index_share_for_mtp_iteration,
            "block_size": self.block_size,
            "markov_rank": self.markov_rank,
            "confidence_enabled": self.confidence_enabled,
            "confidence_executes": self.confidence_executes,
            "confidence_input_width": self.confidence_input_width,
            "target_shared_embedding": self.target_shared_embedding,
            "target_shared_lm_head": self.target_shared_lm_head,
            "base_logits": self.base_logits,
            "markov_replicated": self.markov_replicated,
            "noncausal_query_block": self.noncausal_query_block,
            "pp_capability": list(self.pp_capability),
            "input_width": self.input_width,
            "proposal_tokens": self.proposal_tokens,
            "vocab_size": self.vocab_size,
            "mhc_mult": self.mhc_mult,
            "mhc_sinkhorn_iters": self.mhc_sinkhorn_iters,
            "mhc_lowrank": self.mhc_lowrank,
            **({"mhc_final_collapse": self.mhc_final_collapse} if self.mhc_final_collapse else {}),
            "activation_dtype": self.activation_dtype,
            "input_mode": self.input_mode,
            "weight_category": self.weight_category,
            "qk_norm": self.qk_norm,
            "q_output_gate": self.q_output_gate,
            "input_projection_category": self.input_projection_category,
            "layer_plans": [
                plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
                for plan in self.layer_plans
            ],
            "execution_mode": self.execution_mode,
            "independent_context": self.independent_context,
            "context_projection_mode": self.context_projection_mode,
            "dcp_sharded": self.dcp_sharded,
            "input_application": self.input_application,
            "input_count": self.input_count,
            "embedding_owned": self.embedding_owned,
            "embedding_count": self.embedding_count,
            "output_head_owned": self.output_head_owned,
            "output_head_count": self.output_head_count,
            "max_position_embeddings": self.max_position_embeddings,
            "is_synthetic": self.is_synthetic,
            "draft_provenance": self.draft_provenance,
            "source": dict(self.source),
            "notes": list(self.notes),
        }


def _check_nonneg_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _check_pos_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


@dataclass(frozen=True, slots=True)
class DSparkPreparePlan:
    """Prefill context-initialization contract for DSpark (zero routed MoE FFN)."""

    prompt_tokens: int
    draft_tp: int
    draft_layers: int
    zero_ffn_context_init: bool = True
    skip_write_when_slot_mapping_none: bool = True
    rd2h_tail_valid_bytes: int = 0
    exact_gemm_flops: int = 0
    target_layer_ids: tuple[int, ...] = ()
    block_size: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_tokens", _check_nonneg_int(self.prompt_tokens, "dspark_prepare_prompt_invalid"))
        object.__setattr__(self, "draft_tp", _check_pos_int(self.draft_tp, "dspark_prepare_tp_invalid"))
        object.__setattr__(self, "draft_layers", _check_pos_int(self.draft_layers, "dspark_prepare_layers_invalid"))
        object.__setattr__(
            self, "rd2h_tail_valid_bytes", _check_nonneg_int(self.rd2h_tail_valid_bytes, "dspark_prepare_rd2h_invalid")
        )
        object.__setattr__(
            self, "exact_gemm_flops", _check_nonneg_int(self.exact_gemm_flops, "dspark_prepare_flops_invalid")
        )
        object.__setattr__(self, "block_size", _check_pos_int(self.block_size, "dspark_prepare_block_size_invalid"))
        layer_ids = tuple(_check_nonneg_int(v, "dspark_prepare_target_layer_invalid") for v in self.target_layer_ids)
        if len(set(layer_ids)) != len(layer_ids):
            raise ModelingError("dspark_prepare_duplicate_target_layer_ids")
        object.__setattr__(self, "target_layer_ids", layer_ids)
        if not isinstance(self.zero_ffn_context_init, bool) or not isinstance(self.skip_write_when_slot_mapping_none, bool):
            raise ModelingError("dspark_prepare_flags_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "draft_tp": self.draft_tp,
            "draft_layers": self.draft_layers,
            "zero_ffn_context_init": self.zero_ffn_context_init,
            "skip_write_when_slot_mapping_none": self.skip_write_when_slot_mapping_none,
            "rd2h_tail_valid_bytes": self.rd2h_tail_valid_bytes,
            "exact_gemm_flops": self.exact_gemm_flops,
            "target_layer_ids": list(self.target_layer_ids),
            "block_size": self.block_size,
        }


@dataclass(frozen=True, slots=True)
class DraftStepSchedule:
    """Resolved per-step proposal schedule across MTP, DSpark, EAGLE/EAGLE3, and NGram."""

    archetype: SpeculativeArchetype
    serial: bool
    num_steps: int
    body_q_len_per_step: tuple[int, ...] = ()
    physical_draft_sp_rows: tuple[int, ...] = ()
    selected_logit_rows_per_req: tuple[int, ...] = ()
    step_replay_scopes: tuple[str, ...] = ()
    markov_chain_steps: int = 0

    def __post_init__(self) -> None:
        if self.archetype not in SPECULATIVE_ARCHETYPES:
            raise ModelingError(f"draft_step_schedule_archetype_invalid:{self.archetype!r}")
        if not isinstance(self.serial, bool):
            raise ModelingError("draft_step_schedule_serial_invalid")
        steps = _check_nonneg_int(self.num_steps, "draft_step_schedule_num_steps_invalid")
        object.__setattr__(self, "num_steps", steps)
        body_q = tuple(_check_nonneg_int(v, "draft_step_schedule_body_q_invalid") for v in self.body_q_len_per_step)
        sp_rows = tuple(_check_nonneg_int(v, "draft_step_schedule_sp_rows_invalid") for v in self.physical_draft_sp_rows)
        logits = tuple(_check_nonneg_int(v, "draft_step_schedule_logits_invalid") for v in self.selected_logit_rows_per_req)
        scopes = tuple(str(s) for s in self.step_replay_scopes)
        object.__setattr__(self, "body_q_len_per_step", body_q)
        object.__setattr__(self, "physical_draft_sp_rows", sp_rows)
        object.__setattr__(self, "selected_logit_rows_per_req", logits)
        object.__setattr__(self, "step_replay_scopes", scopes)
        markov = _check_nonneg_int(self.markov_chain_steps, "draft_step_schedule_markov_invalid")
        object.__setattr__(self, "markov_chain_steps", markov)
        if self.archetype in {"none", "ngram"}:
            if steps != 0 or body_q or sp_rows or logits or markov != 0:
                raise ModelingError(f"draft_step_schedule_zero_neural_violated:{self.archetype}")
        elif body_q and len(body_q) != steps:
            raise ModelingError(f"draft_step_schedule_length_mismatch:body_q={len(body_q)}!=steps={steps}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype": self.archetype,
            "serial": self.serial,
            "num_steps": self.num_steps,
            "body_q_len_per_step": list(self.body_q_len_per_step),
            "physical_draft_sp_rows": list(self.physical_draft_sp_rows),
            "selected_logit_rows_per_req": list(self.selected_logit_rows_per_req),
            "step_replay_scopes": list(self.step_replay_scopes),
            "markov_chain_steps": self.markov_chain_steps,
        }


@dataclass(frozen=True, slots=True)
class SpeculativeExecutionContract:
    """Single-owner speculative execution contract encoding real-vs-synthetic draft policy."""

    archetype: SpeculativeArchetype
    spec_method: SpeculativeMethod
    num_speculative_tokens: int
    requires_draft_spec: bool
    draft_kv_dtype: str = "bf16"
    draft_tp: int = 1
    draft_pp: int = 1
    draft_dcp: int = 1
    draft_ep_enabled: bool = False
    draft_eplb_enabled: bool = False
    neural_draft_forwards: int = 0
    requires_draft_prepare: bool = False
    sample_from_anchor: bool = False
    hc_collapse_before_sp_gather: bool = False
    connector_kind: str = "none"
    layout_kind: str = "standard"
    optimization_mode: Literal["simulation", "modeling"] = "simulation"
    allow_synthetic_gqa: bool = False
    draft_provenance: SpeculativeDraftProvenance = REAL_CHECKPOINT_PROVENANCE
    acceptance_rate: float | None = None

    def __post_init__(self) -> None:
        if self.spec_method not in SPECULATIVE_METHODS:
            raise ModelingError(f"speculative_contract_method_invalid:{self.spec_method!r}")
        if self.archetype not in SPECULATIVE_ARCHETYPES:
            raise ModelingError(f"speculative_contract_archetype_invalid:{self.archetype!r}")
        if _METHOD_TO_ARCHETYPE[self.spec_method] != self.archetype:
            raise ModelingError(
                f"speculative_contract_method_archetype_mismatch:{self.spec_method}!={self.archetype}"
            )
        spec_k = _check_nonneg_int(self.num_speculative_tokens, "speculative_contract_k_invalid")
        if spec_k > MAX_SPECULATIVE_TOKENS:
            raise ModelingError(f"speculative_contract_k_exceeds_max:{spec_k}>{MAX_SPECULATIVE_TOKENS}")
        object.__setattr__(self, "num_speculative_tokens", spec_k)
        object.__setattr__(self, "draft_tp", _check_pos_int(self.draft_tp, "speculative_contract_draft_tp_invalid"))
        object.__setattr__(self, "draft_pp", _check_pos_int(self.draft_pp, "speculative_contract_draft_pp_invalid"))
        object.__setattr__(self, "draft_dcp", _check_pos_int(self.draft_dcp, "speculative_contract_draft_dcp_invalid"))
        object.__setattr__(
            self, "neural_draft_forwards", _check_nonneg_int(self.neural_draft_forwards, "speculative_contract_forwards_invalid")
        )
        if self.optimization_mode not in {"simulation", "modeling"}:
            raise ModelingError(f"speculative_contract_mode_invalid:{self.optimization_mode!r}")
        if not isinstance(self.allow_synthetic_gqa, bool) or not isinstance(self.requires_draft_spec, bool):
            raise ModelingError("speculative_contract_bool_flags_invalid")
        if self.draft_provenance not in SPECULATIVE_DRAFT_PROVENANCES:
            raise ModelingError(f"speculative_contract_provenance_invalid:{self.draft_provenance!r}")
        self._validate_mode_and_archetype(spec_k)
        if self.acceptance_rate is not None:
            if isinstance(self.acceptance_rate, bool):
                raise ModelingError("speculative_contract_acceptance_rate_invalid")
            rate = float(self.acceptance_rate)
            if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
                raise ModelingError(f"speculative_contract_acceptance_rate_range:{rate!r}")
            object.__setattr__(self, "acceptance_rate", rate)

    def _validate_mode_and_archetype(self, spec_k: int) -> None:
        if self.allow_synthetic_gqa:
            if self.optimization_mode != "modeling":
                raise ModelingError("speculative_contract_simulation_forbids_synthetic_gqa")
            if self.spec_method != "eagle3":
                raise ModelingError(f"speculative_contract_synthetic_gqa_requires_eagle3:{self.spec_method!r}")
            if spec_k < 1:
                raise ModelingError("speculative_contract_synthetic_gqa_requires_positive_k")
            if self.draft_provenance != SYNTHETIC_EAGLE3_GQA_PROVENANCE:
                raise ModelingError(f"speculative_contract_synthetic_provenance_required:{self.draft_provenance!r}")
        elif self.draft_provenance == SYNTHETIC_EAGLE3_GQA_PROVENANCE:
            raise ModelingError("speculative_contract_synthetic_provenance_without_opt_in")
        if self.archetype == "none":
            if spec_k != 0 or self.requires_draft_spec or self.neural_draft_forwards != 0:
                raise ModelingError("speculative_contract_none_must_be_inactive")
            if self.draft_provenance != NO_DRAFT_PROVENANCE:
                raise ModelingError(f"speculative_contract_none_provenance_invalid:{self.draft_provenance!r}")
        elif self.archetype == "ngram":
            if spec_k < 1 or self.requires_draft_spec or self.neural_draft_forwards != 0 or self.requires_draft_prepare:
                raise ModelingError("speculative_contract_ngram_must_have_zero_neural_draft")
            if self.draft_provenance != NGRAM_CPU_PROVENANCE:
                raise ModelingError(f"speculative_contract_ngram_provenance_invalid:{self.draft_provenance!r}")
        else:
            if spec_k < 1 or not self.requires_draft_spec:
                raise ModelingError(f"speculative_contract_neural_requires_draft_and_k:{self.archetype}")

    def require_draft_for_execution(self, draft: DraftSpec | None) -> DraftSpec | None:
        """Fail fast when a real draft is missing or a synthetic draft enters simulation."""
        if self.archetype in {"none", "ngram"}:
            if draft is not None:
                raise ModelingError(f"speculative_contract_forbids_neural_draft:{self.archetype}")
            return None
        if draft is None:
            raise ModelingError(
                f"speculative_contract_missing_real_draft_spec:method={self.spec_method}:mode={self.optimization_mode}"
            )
        if not isinstance(draft, DraftSpec):
            raise ModelingError("speculative_contract_draft_type_invalid")
        if draft.is_synthetic:
            if self.optimization_mode != "modeling" or not self.allow_synthetic_gqa:
                raise ModelingError("speculative_contract_rejects_synthetic_draft_without_modeling_opt_in")
            if draft.draft_provenance != SYNTHETIC_EAGLE3_GQA_PROVENANCE:
                raise ModelingError("speculative_contract_synthetic_draft_provenance_mismatch")
            if draft.kind != "eagle3" or draft.attention_kind != "gqa" or draft.execution_mode != "eagle3_serial":
                raise ModelingError(
                    f"speculative_contract_synthetic_draft_must_be_eagle3_gqa:"
                    f"kind={draft.kind}:attn={draft.attention_kind}:mode={draft.execution_mode}"
                )
        elif self.draft_provenance == SYNTHETIC_EAGLE3_GQA_PROVENANCE:
            raise ModelingError("speculative_contract_real_draft_conflicts_with_synthetic_provenance")
        draft.validate()
        return draft

    def validate_dcp_spec_gate(self, draft_spec: DraftSpec | None, parallel: Any) -> None:
        """Validate draft availability and DCP/PCP compatibility before lowering."""
        checked = self.require_draft_for_execution(draft_spec)
        dcp = int(getattr(parallel, "dcp", 1) or 1)
        if dcp < 1:
            raise ModelingError(f"speculative_contract_parallel_dcp_invalid:{dcp}")
        if checked is not None and self.draft_dcp > dcp:
            raise ModelingError(f"speculative_contract_draft_dcp_exceeds_target:{self.draft_dcp}>{dcp}")

    def build_step_schedule(self, batch_views: Sequence[Any]) -> DraftStepSchedule:
        """Build the deterministic per-step draft schedule for the active decode batch."""
        if self.archetype in {"none", "ngram"}:
            return DraftStepSchedule(archetype=self.archetype, serial=False, num_steps=0)
        active_reqs = [v for v in batch_views if not getattr(v, "padding", False)]
        if not active_reqs:
            return DraftStepSchedule(
                archetype=self.archetype, serial=(self.archetype != "dspark"), num_steps=0,
            )
        verify_rows = 0
        req_budgets: list[int] = []
        for view in active_reqs:
            q_len = int(getattr(view, "q_len", 1) or 1)
            prop = int(getattr(view, "draft_proposal_tokens", 0) or 0)
            if prop > 0:
                verify_rows += max(q_len, 1 + prop)
                req_budgets.append(prop)
            elif getattr(view, "draft_verify", False):
                verify_rows += max(q_len, 1 + self.num_speculative_tokens)
                req_budgets.append(self.num_speculative_tokens)
            else:
                req_budgets.append(0)
        steps = max(req_budgets, default=0)
        if steps <= 0:
            return DraftStepSchedule(
                archetype=self.archetype, serial=(self.archetype != "dspark"), num_steps=0,
            )
        if self.archetype == "dspark":
            sp_rows = self.draft_tp * ((verify_rows + self.draft_tp - 1) // self.draft_tp)
            return DraftStepSchedule(
                archetype="dspark",
                serial=False,
                num_steps=1,
                body_q_len_per_step=(verify_rows,),
                physical_draft_sp_rows=(sp_rows,),
                selected_logit_rows_per_req=(self.num_speculative_tokens,),
                step_replay_scopes=("draft_block",),
                markov_chain_steps=self.num_speculative_tokens,
            )
        body_q = (verify_rows,) + tuple(sum(1 for b in req_budgets if b > step_idx) for step_idx in range(1, steps))
        sp_rows_tuple = tuple(self.draft_tp * ((q + self.draft_tp - 1) // self.draft_tp) for q in body_q)
        return DraftStepSchedule(
            archetype=self.archetype,
            serial=True,
            num_steps=steps,
            body_q_len_per_step=body_q,
            physical_draft_sp_rows=sp_rows_tuple,
            selected_logit_rows_per_req=(1,) * steps,
            step_replay_scopes=tuple(f"draft_step_{idx}" for idx in range(steps)),
            markov_chain_steps=0,
        )

    def build_prepare_plan(
        self, prompt_len: int, *, draft_spec: DraftSpec | None = None,
    ) -> DSparkPreparePlan:
        """Build the DSpark prefill context-initialization plan (zero routed FFN)."""
        checked = self.require_draft_for_execution(draft_spec)
        if self.archetype != "dspark" or checked is None:
            raise ModelingError(f"speculative_contract_prepare_plan_requires_dspark:{self.archetype}")
        prompt = _check_nonneg_int(prompt_len, "dspark_prepare_prompt_len_invalid")
        input_flops = GEMM_FLOPS_PER_MAC * prompt * checked.input_width * checked.hidden_size
        ctx_mode = checked.context_projection_mode
        if not ctx_mode or ctx_mode == "none":
            ctx_mode = dict(checked.source).get("context_projection_mode", "none")
        if ctx_mode == "per_layer_qkv":
            proj_width = checked.q_width + GQA_KV_BANK_MULTIPLIER * checked.kv_width
        elif checked.attention_kind == "gqa":
            proj_width = GQA_KV_BANK_MULTIPLIER * checked.kv_width
        else:
            mla_qkv = checked.q_lora_rank + checked.kv_lora_rank + checked.qk_rope_head_dim
            proj_width = mla_qkv if mla_qkv > 0 else (MLA_KV_BANK_MULTIPLIER * checked.kv_width)
        kv_flops = GEMM_FLOPS_PER_MAC * prompt * checked.hidden_size * proj_width * checked.layers
        return DSparkPreparePlan(
            prompt_tokens=prompt,
            draft_tp=self.draft_tp,
            draft_layers=checked.layers,
            zero_ffn_context_init=True,
            skip_write_when_slot_mapping_none=True,
            rd2h_tail_valid_bytes=0,
            exact_gemm_flops=int(input_flops + kv_flops),
            target_layer_ids=checked.target_layer_ids,
            block_size=checked.block_size,
        )


def parse_draft_max_position_embeddings(
    config: Mapping[str, Any], *, is_standalone_draft: bool = False,
) -> int:
    """Extract an explicit draft context-length cap from HF/companion config mappings."""
    def _pos_int(value: Any) -> int:
        if isinstance(value, bool) or value is None:
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0

    for key in (
        "dspark_max_model_len",
        "dspark_max_position_embeddings",
        "draft_max_model_len",
        "draft_max_position_embeddings",
        "mtp_max_model_len",
        "mtp_max_position_embeddings",
    ):
        found = _pos_int(config.get(key))
        if found > 0:
            return found
    for nested_key in (
        "dspark_config",
        "dflash_config",
        "draft_config",
        "speculative_config",
        "mtp_config",
    ):
        nested = config.get(nested_key)
        if isinstance(nested, Mapping):
            for key in ("draft_max_model_len", "max_model_len", "max_position_embeddings"):
                found = _pos_int(nested.get(key))
                if found > 0:
                    return found
    if is_standalone_draft:
        for key in ("max_model_len", "max_position_embeddings"):
            found = _pos_int(config.get(key))
            if found > 0:
                return found
    return 0


def effective_draft_context_cap(draft: Any, runtime: Any) -> int:
    """Return the smallest positive declared context-length cap for a drafter."""
    candidates: list[int] = []
    spec_cap = int(getattr(draft, "max_position_embeddings", 0) or 0)
    if spec_cap > 0:
        candidates.append(spec_cap)
    if isinstance(runtime, Mapping):
        meta = dict(runtime)
    else:
        meta = dict(getattr(runtime, "metadata", ()) or ())
    rt_cap = int(meta.get("draft_max_model_len") or 0)
    if rt_cap > 0:
        candidates.append(rt_cap)
    spec_cfg = meta.get("speculative_config")
    if isinstance(spec_cfg, Mapping):
        for key in ("draft_max_model_len", "max_model_len", "max_position_embeddings"):
            cfg_cap = int(spec_cfg.get(key) or 0)
            if cfg_cap > 0:
                candidates.append(cfg_cap)
    return min(candidates) if candidates else 0


__all__ = [
    "ALLOWED_CONTEXT_PROJECTION_MODES",
    "ALLOWED_EXECUTION_MODES",
    "DEFAULT_DRAFT_PP_STAGES",
    "MAX_SPECULATIVE_TOKENS",
    "NGRAM_CPU_PROVENANCE",
    "NO_DRAFT_PROVENANCE",
    "REAL_CHECKPOINT_PROVENANCE",
    "SPECULATIVE_ARCHETYPES",
    "SPECULATIVE_DRAFT_PROVENANCES",
    "SPECULATIVE_METHODS",
    "DSparkPreparePlan",
    "DraftSpec",
    "DraftStepSchedule",
    "SpeculativeArchetype",
    "SpeculativeDraftProvenance",
    "SpeculativeExecutionContract",
    "SpeculativeMethod",
    "effective_draft_context_cap",
    "parse_draft_max_position_embeddings",
]
