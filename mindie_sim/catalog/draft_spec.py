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

from dataclasses import dataclass
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError

DEFAULT_DRAFT_PP_STAGES: tuple[int, ...] = (1, 2, 4, 8, 16)
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
    source: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
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
            "source": dict(self.source),
            "notes": list(self.notes),
        }


__all__ = [
    "ALLOWED_CONTEXT_PROJECTION_MODES",
    "ALLOWED_EXECUTION_MODES",
    "DEFAULT_DRAFT_PP_STAGES",
    "DraftSpec",
]
