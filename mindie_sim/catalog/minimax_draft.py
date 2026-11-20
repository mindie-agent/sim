"""MiniMax-M3 / M3.1-Flash optional dense GQA DSpark and Eagle3 auxiliary specs.

[Role]
  Construct the optional seven-module dense GQA DSpark block drafter
  (``MiniMaxFlashDSparkForCausalLM``, PR #20167) or single-layer NextN/Eagle3
  auxiliary specification from MiniMax-M3 HuggingFace config mappings while
  keeping ``model.layers`` strictly equal to the body layer count.

[Invariants]
  * Every MiniMax-M3 draft layer is a dense non-sparse GQA block
    (``ffn="dense"``, ``indexer=False``) using ``dense_intermediate_size`` and
    preserving the checkpoint's ``swigluoai`` clamping scalars and ``qk_norm``.
  * Parallel multi-module DSpark (``num_mtp_modules > 0``) executes as a single
    non-causal query block pass and shares the target embedding and LM head.
  * Optional ``DraftSpec`` context-preparation flags are forwarded dynamically
    via ``DraftSpec.__dataclass_fields__`` without ``inspect.signature``.

[Timeouts & Locks]
  Pure in-memory dataclass construction bounded by ``num_mtp_modules``; no I/O,
  network, subprocess, or locks.

[Side-Effect Policy]
  Side-effect free; returns new immutable ``AuxiliarySpec`` / ``DraftSpec``
  objects without mutating input mappings.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.catalog.config import _int, _require
from mindie_sim.catalog.hf_config import declared_partial_rotary, rotary_dimensions
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, merge_source
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan

KV_CACHE_PLANES = 2
UNOWNED_LAYER_ID = -1
DENSE_CANDIDATE_BLOCK = 1
SINGLE_PROPOSAL_BLOCK = 1
DEFAULT_SWIGLU_OAI_ALPHA = 1.702
DEFAULT_SWIGLU_OAI_BETA = 1.0
MINIMAX_DSPARK_PP_CAPABILITY = (1, 2, 4, 8, 16)

MINIMAX_DSPARK_SOURCE: dict[str, str] = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/models/minimax_m3/minimax_m3_dspark.py",
    "source_sha256": "124448ab8840d1c07636b65d89b344e0a280fff1ef3d3782e58c407313d18b32",
}


def _draft_compat_kwargs(**candidates: Any) -> dict[str, Any]:
    declared = DraftSpec.__dataclass_fields__
    return {key: value for key, value in candidates.items() if key in declared}


def build_minimax_auxiliary(
    c: Mapping[str, Any], *, body_layers: int,
) -> AuxiliarySpec | None:
    """Build the optional MiniMax-M3 DSpark / Eagle3 ``AuxiliarySpec``."""
    dspark_modules = _int(c, "num_mtp_modules")
    nextn_layers = _int(c, "num_nextn_predict_layers")
    modules = dspark_modules if dspark_modules > 0 else nextn_layers
    if modules <= 0:
        return None

    hidden = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads"))
    kv_heads = int(_require(c, "num_key_value_heads"))
    head_dim = int(_require(c, "head_dim"))
    vocab = int(_require(c, "vocab_size"))
    dense_i = _int(c, "dense_intermediate_size") or _int(c, "intermediate_size")
    if dense_i <= 0:
        raise ModelingError("model_config_minimax_draft_intermediate_missing")

    qk_norm = bool(c.get("use_qk_norm", True))
    gate = bool(c.get("attention_output_gate", False))
    rope_dim, nope_dim = rotary_dimensions(head_dim, declared_partial_rotary(c))
    activation = str(c.get("hidden_act", "silu"))
    if activation == "swigluoai":
        oai_limit = float(c.get("swiglu_limit") or 0.0)
        if oai_limit <= 0.0:
            raise ModelingError("model_config_minimax_swigluoai_limit_missing")
        oai_alpha = float(c.get("swiglu_alpha") or DEFAULT_SWIGLU_OAI_ALPHA)
        oai_beta = float(c.get("swiglu_beta") or DEFAULT_SWIGLU_OAI_BETA)
    else:
        oai_limit = oai_alpha = oai_beta = 0.0
    shared_limit = float(c.get("swiglu_limits_shared") or oai_limit)

    is_parallel_dspark = dspark_modules > 0
    raw_targets = c.get("dspark_target_layer_ids")
    targets = (
        tuple(int(v) for v in raw_targets)
        if isinstance(raw_targets, (list, tuple)) and raw_targets
        else (max(0, int(body_layers) - 1),)
    )
    kv_elements = KV_CACHE_PLANES * kv_heads * head_dim
    draft_plans = tuple(
        LayerPlan(
            index=index,
            native_layer_id=int(body_layers) + index,
            attention="gqa",
            architecture="gqa",
            ffn="dense",
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=dense_i,
            qk_norm=qk_norm,
            q_output_gate=gate,
            qk_rope_head_dim=rope_dim,
            qk_nope_head_dim=nope_dim,
            ffn_activation=activation,
            swiglu_limit=oai_limit,
            swiglu_shared_limit=shared_limit,
            swiglu_alpha=oai_alpha,
            swiglu_beta=oai_beta,
            noncausal_attention=is_parallel_dspark,
            indexer=False,
            indexer_type="none",
            owns_index_k=False,
            index_k_owner_layer_id=UNOWNED_LAYER_ID,
            owns_topk=False,
            topk_owner_layer_id=UNOWNED_LAYER_ID,
            candidate_block_size=DENSE_CANDIDATE_BLOCK,
            kv_cache_elements=kv_elements,
            kv_owner_layer_id=index,
            owns_long_kv=True,
            scope="draft",
        )
        for index in range(modules)
    )
    block_tokens = modules if is_parallel_dspark else SINGLE_PROPOSAL_BLOCK
    source_kind = "minimax_m3_dspark" if is_parallel_dspark else "minimax_m3_eagle3"
    spec = DraftSpec(
        kind="mtp",
        layers=modules,
        hidden_size=hidden,
        intermediate_size=dense_i,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        q_width=heads * head_dim,
        kv_width=kv_heads * head_dim,
        attention_kind="gqa",
        kv_cache_elements=kv_elements,
        input_mode="projected_concat",
        weight_category="mtp",
        qk_norm=qk_norm,
        q_output_gate=gate,
        target_layer_ids=targets,
        block_size=block_tokens,
        target_shared_embedding=True,
        target_shared_lm_head=True,
        base_logits=True,
        noncausal_query_block=is_parallel_dspark,
        pp_capability=MINIMAX_DSPARK_PP_CAPABILITY,
        vocab_size=vocab,
        layer_plans=draft_plans,
        source=merge_source(
            MINIMAX_DSPARK_SOURCE,
            {
                "kind": source_kind,
                "independent_context": "true",
                "context_projection_mode": "per_layer_qkv",
                "dcp_sharded": "false",
            },
        ),
        notes=("MiniMax-M3 dense GQA DSpark modules; PR #20167",),
        **_draft_compat_kwargs(
            independent_context=True,
            context_projection_mode="per_layer_qkv",
            dcp_sharded=False,
        ),
    )
    return AuxiliarySpec(
        mtp=spec,
        dspark=spec if is_parallel_dspark else None,
        notes=("minimax_m3_optional_dspark",),
    )


__all__ = ["MINIMAX_DSPARK_SOURCE", "build_minimax_auxiliary"]
