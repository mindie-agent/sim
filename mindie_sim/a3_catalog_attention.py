"""Attention-lowering dispatcher for catalog models.

[Role]
  Select the actual GQA/MLA/DSA/KDA/GDN implementation.  No generic MLA
  fallback exists for a different architecture.

[Invariants]
  * Missing dimensions/unsupported kinds raise A3InputError.
  * Dispatched functions preserve all family invariants.

[Timeouts & Locks]
  Pure dispatch; no I/O.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .a3_catalog_attention_common import AttentionResult, _requests
from .a3_catalog_attention_gqa import _gqa_attention
from .a3_catalog_attention_mla import _mla_attention
from .a3_catalog_linear_attention import _linear_attention
from .a3_catalog_types import QuantSpec
from .a3_model_types import LayerPlan, ModelSpec
from .a3_types import A3InputError, ParallelSpec


def lower_attention(
    *, prefix: str, plan: LayerPlan, views: Sequence[Any], stage: int, deps: str,
    model: ModelSpec, parallel: ParallelSpec, quant: QuantSpec, mode: str,
    padded_tokens: int, cache_ids: Mapping[int, str] | None = None,
    topk_ids: Mapping[int, str] | None = None, stage_of: Any = None,
    index_cache_ids: Mapping[int, str] | None = None,
    candidate_ids: Mapping[int, str] | None = None,
) -> AttentionResult:
    cache_ids = dict(cache_ids or {})
    topk_ids = dict(topk_ids or {})
    index_cache_ids = dict(index_cache_ids or {})
    candidate_ids = dict(candidate_ids or {})
    if plan.attention == "gqa":
        return _gqa_attention(
            prefix=prefix, plan=plan, views=views, stage=stage, deps=deps,
            model=model, parallel=parallel, quant=quant, mode=mode,
            padded_tokens=padded_tokens,
        )
    if plan.attention == "mla":
        return _mla_attention(
            prefix=prefix, plan=plan, views=views, stage=stage, deps=deps,
            model=model, parallel=parallel, quant=quant, mode=mode,
            padded_tokens=padded_tokens, cache_ids=cache_ids,
            topk_ids=topk_ids, index_cache_ids=index_cache_ids,
            candidate_ids=candidate_ids, stage_of=stage_of,
        )
    if plan.attention in {"kda", "gdn"}:
        return _linear_attention(
            prefix=prefix, plan=plan, views=views, stage=stage, deps=deps,
            model=model, parallel=parallel, quant=quant, mode=mode,
            padded_tokens=padded_tokens,
        )
    raise A3InputError(f"catalog_attention_kind_unsupported:{plan.attention}")


__all__ = ["AttentionResult", "lower_attention"]
