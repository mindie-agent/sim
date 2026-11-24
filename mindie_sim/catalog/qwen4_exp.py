"""Unplugged W5a structural sockets for PLE, low-rank gated HC, and QSA terms.

[Role]
  W5a Commit #1 scorched-earth subtraction base: purged dead L1->L2/L3 upward
  imports (``build_ple_uva_npu_dequant``, ``build_ple_host_staging``),
  hardcoded ``arch_family="A5"`` / ``readiness="shipping"`` / ``64``-owner
  bindings, silent ``bf16``/``int8`` fallbacks, and AG014 model-name gates.
  Pure structural weight-term builders remain importable while broken PLE
  lowering and model-gated pool breakdown raise loud ``LEGACY_PURGED`` sockets.
[Invariants]
  * Zero imports from ``mindie_sim.prediction.*`` (AG010 clean).
  * Zero AG014 model-name comparisons.
  * Zero hardcoded A5/shipping/64-owner bindings or silent dtype defaults.
[Timeouts & Locks]
  Pure arithmetic; no I/O or locks.
[Side-Effect Policy]
  Side-effect free.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.catalog.types import DraftSpec, EngramSpec
from mindie_sim.catalog.weight_terms import WeightTerm, _t
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.parallel import ParallelSpec

DEFAULT_PLE_CONV_KERNEL = 4


def build_ple_spec(c: Mapping[str, Any]) -> EngramSpec:
    del c
    raise NotImplementedError(
        "LEGACY_PURGED: wire structural PLE EngramSpec with explicit storage_kind "
        "(zero AG014 model-name tag or hardcoded BF16_BYTES)"
    )


def is_qwen4_exp_ple(engram: Any) -> bool:
    del engram
    raise NotImplementedError(
        "LEGACY_PURGED: replace AG014 qwen4_exp_ple tag check with structural "
        "engram.storage_kind / PLE predicate"
    )


def ple_lookup_format(model: Any, runtime: Any = None) -> str:
    del model, runtime
    raise NotImplementedError(
        "LEGACY_PURGED: require explicit declared PLE table format from checkpoint/runtime "
        "contract (zero silent bf16/int8 fallback)"
    )


def lower_ple(
    *, prefix: str, plan: Any, stage: int, deps: str, model: ModelSpec,
    parallel: ParallelSpec, quant: Any, mode: str, padded_tokens: int,
    views: Any, runtime: Any,
) -> tuple[list[Any], str]:
    del prefix, plan, stage, deps, model, parallel, quant, mode, padded_tokens, views, runtime
    raise NotImplementedError(
        "LEGACY_PURGED: resolve PLE KernelBinding & owner schedules in L3 governed operators "
        "from die.compute_arch.aiv_cores + memory_hierarchy + ProbedAbiRegistry "
        "(zero hardcoded A5/shipping/64-owner bindings or BF16_BYTES)"
    )


def ple_table_terms(model: ModelSpec) -> list[WeightTerm]:
    engram = model.engram
    if engram is None:
        return []
    return [
        WeightTerm(f"ple_table_{table.table_id}", table.rows, table.width, "engram_table", "out_tp", 1)
        for table in engram.tables
    ]


def ple_gate_terms_by_layer(model: ModelSpec) -> dict[int, list[WeightTerm]]:
    engram = model.engram
    if engram is None:
        return {}
    hc = max((p.hc_mult for p in model.layer_plans), default=1)
    h = model.hidden_size
    hc_h = hc * h
    embed = engram.columns_per_table * engram.width
    conv_k = int(dict(engram.source).get("ple_conv_kernel_size", DEFAULT_PLE_CONV_KERNEL))
    by_layer: dict[int, list[WeightTerm]] = {}
    for table in engram.tables:
        by_layer.setdefault(table.layer_id, []).extend([
            _t("ple_kv_proj", hc_h + h, embed, "ple_proj", count=1),
            _t("ple_norm_key", hc_h, 1, "norm", count=1),
            _t("ple_norm_query", hc_h, 1, "norm", count=1),
            _t("ple_norm_conv", hc_h, 1, "norm", count=1),
            _t("ple_conv1d", hc_h, conv_k, "conv_weight_bf16", count=1),
        ])
    return by_layer


def lowrank_hc_layer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    hc_dim = plan.hc_mult * model.hidden_size
    terms: list[WeightTerm] = []
    for part in ("attn", "ffn"):
        terms.extend([
            _t(f"hc_{part}_norm", hc_dim, 1, "norm"),
            _t(f"hc_{part}_down_inject", plan.hc_lowrank + plan.hc_mult, hc_dim, "linear_param_bf16"),
            _t(f"hc_{part}_up", hc_dim, plan.hc_lowrank, "linear_param_bf16"),
        ])
    return terms


def lowrank_hc_head_terms(model: ModelSpec) -> list[WeightTerm]:
    hc = max((p.hc_mult for p in model.layer_plans), default=0)
    lr = max((p.hc_lowrank for p in model.layer_plans), default=0)
    hc_dim = hc * model.hidden_size
    return [
        _t("hc_head_norm", hc_dim, 1, "norm"),
        _t("hc_head_down", lr, hc_dim, "linear_param_bf16"),
        _t("hc_head_up", hc_dim, lr, "linear_param_bf16"),
    ]


def lowrank_hc_draft_terms(draft: DraftSpec) -> list[WeightTerm]:
    if draft.mhc_mult <= 0 or draft.mhc_lowrank <= 0:
        return []
    hc_dim = draft.mhc_mult * draft.hidden_size
    terms: list[WeightTerm] = []
    for part in ("attn", "ffn"):
        terms.extend([
            _t(f"draft_hc_{part}_norm", hc_dim, 1, "norm"),
            _t(f"draft_hc_{part}_down_inject", draft.mhc_lowrank + draft.mhc_mult, hc_dim, "linear_param_bf16"),
            _t(f"draft_hc_{part}_up", hc_dim, draft.mhc_lowrank, "linear_param_bf16"),
        ])
    return terms


def lowrank_hc_draft_head_terms(draft: DraftSpec) -> list[WeightTerm]:
    if draft.mhc_mult <= 0 or draft.mhc_lowrank <= 0:
        return []
    hc_dim = draft.mhc_mult * draft.hidden_size
    return [
        _t("draft_hc_head_norm", hc_dim, 1, "norm"),
        _t("draft_hc_head_down", draft.mhc_lowrank, hc_dim, "linear_param_bf16"),
        _t("draft_hc_head_up", hc_dim, draft.mhc_lowrank, "linear_param_bf16"),
    ]


def qsa_indexer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    if not getattr(plan, "indexer", False) or plan.index_heads <= 0 or plan.index_dim <= 0:
        return []
    return [
        _t("index_qk_proj", (plan.index_heads + 1) * plan.index_dim, model.hidden_size, "indexer"),
        _t("indexer_q_norm", plan.index_dim, 1, "norm"),
        _t("indexer_k_norm", plan.index_dim, 1, "norm"),
    ]


def ple_conv_state_bytes(
    model: Any, runtime: Any = None, *, query_tokens: int | None = None,
) -> float:
    del model, runtime, query_tokens
    raise NotImplementedError(
        "LEGACY_PURGED: compute PLE conv state structurally using model.activation_dtype "
        "and SpeculativeExecutionContract (zero hardcoded BF16_BYTES or AG014 tag)"
    )


def qwen4_exp_pool_breakdown(
    model: Any, parallel: ParallelSpec, sequence_tokens: int, *,
    runtime: Any = None, query_tokens: int | None = None,
) -> dict[str, Any] | None:
    del model, parallel, sequence_tokens, runtime, query_tokens
    raise NotImplementedError(
        "LEGACY_PURGED: delegate uniformly to structural HybridPoolReceipt in "
        "catalog/cache_pools.py with explicit CacheStateABI dtypes "
        "(zero AG014 model_type gate or qsa[0] single-layer assumption)"
    )


__all__ = [
    "build_ple_spec", "is_qwen4_exp_ple", "lower_ple", "ple_lookup_format",
    "ple_table_terms", "ple_gate_terms_by_layer", "lowrank_hc_layer_terms",
    "lowrank_hc_head_terms", "lowrank_hc_draft_terms", "lowrank_hc_draft_head_terms",
    "qsa_indexer_terms", "ple_conv_state_bytes", "qwen4_exp_pool_breakdown",
]
