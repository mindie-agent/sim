"""Shared GDN chunk-kernel head-ratio and dimension domain predicates.

[Role]
  Centralize structural admission checks for Ascend Arch35 ``fla-npu`` GDN
  prefill chunk kernels: ``Chunk64`` (``Rung 1: merged``) and ``Chunk32``
  (``PR #22147``, ``Rung 2: unmerged_pr``, ``Hv/Hk in {1,2,3,4}`` with ``K=V=128``).

[Invariants]
  * Domain checks inspect ``LayerPlan`` / local TP head counts and precision only;
    zero model-name strings or runtime observations participate.
  * ``can_run_gdn_chunk64`` matches ``chunk_gated_delta_rule_fwd_prepare_tiling.cpp``
    (``Hv % Hk == 0`` and ``Hv // Hk in {1, 2, 3, 4}``).
  * ``can_run_gdn_chunk32`` matches ``fla-npu`` ``26.10.0+feature`` silicon admission
    (``Hv % Hk == 0``, ``Hv // Hk in {1, 2, 3, 4}``, and ``dk == dv == 128``;
    ``Hv // Hk > 4`` rejects with ``161002`` and falls back to Triton ``ChunkGdn``).

[Timeouts & Locks]
  O(layers) pure integer predicates; no I/O or locks.

[Side-Effect Policy]
  Pure functions and frozen constants only.
"""
from __future__ import annotations

import math
from typing import Any

GDN_CHUNK64_TOKENS = 64
GDN_CHUNK32_TOKENS = 32
GDN_HEAD_DIM_K = 128
GDN_CHUNK32_HEAD_DIM_V = 128
GDN_CHUNK64_VALUE_DIMS = frozenset({128, 256})
GDN_CHUNK64_GQA_RATIOS = frozenset({1, 2, 3, 4})

FLA_GDN_CHUNK_SOURCE = {
    "repository": "flash-linear-attention-npu-a5",
    "revision": "26.10.0+feature",
    "relative_path": (
        "fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/"
        "fla_npu_transformer_impl/dynamic/chunk_gated_delta_rule_fwd.py"
    ),
    "source_sha256": "524967882b658fd57b7d5516deb518314fe231a2877885cf38d47d874db4dbb5",
}

VLLM_ASCEND_GDN_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/ops/gdn.py",
    "source_sha256": "b170b30626cf46bc4ec1e47080f9487c1e5443e51add42bb8c67bfd20a79fac4",
}


def can_run_gdn_chunk64(
    local_hk: int,
    local_hv: int,
    k_dim: int,
    v_dim: int,
    *,
    activation_dtype: str = "bf16",
    state_dtype: str = "fp32",
    composite_v128_only: bool = False,
) -> bool:
    """True when local head geometry satisfies the Arch35 ``Chunk64`` tiling domain."""
    if local_hk <= 0 or local_hv <= 0 or local_hv % local_hk != 0:
        return False
    if (local_hv // local_hk) not in GDN_CHUNK64_GQA_RATIOS:
        return False
    if int(k_dim) != GDN_HEAD_DIM_K:
        return False
    if composite_v128_only:
        if int(v_dim) != GDN_CHUNK32_HEAD_DIM_V:
            return False
    elif int(v_dim) not in GDN_CHUNK64_VALUE_DIMS:
        return False
    return activation_dtype == "bf16" and state_dtype == "fp32"


def can_run_gdn_chunk32(
    local_hk: int,
    local_hv: int,
    k_dim: int,
    v_dim: int,
    *,
    activation_dtype: str = "bf16",
    state_dtype: str = "fp32",
) -> bool:
    """True when local head geometry satisfies the Arch35 ``Chunk32`` (PR #22147) domain."""
    if local_hk <= 0 or local_hv <= 0 or local_hv % local_hk != 0:
        return False
    if (local_hv // local_hk) not in GDN_CHUNK64_GQA_RATIOS:
        return False
    if int(k_dim) != GDN_HEAD_DIM_K or int(v_dim) != GDN_CHUNK32_HEAD_DIM_V:
        return False
    return activation_dtype == "bf16" and state_dtype == "fp32"


def local_gdn_geometry(plan: Any, tp: int) -> tuple[int, int, int, int]:
    """Return ``(local_hk, local_hv, dk, dv)`` for one GDN ``LayerPlan`` at ``tp``."""
    degree = max(1, int(tp))
    raw_hk = int(getattr(plan, "linear_key_heads", 0) or getattr(plan, "linear_heads", 0))
    raw_hv = int(getattr(plan, "linear_value_heads", 0) or getattr(plan, "linear_heads", 0))
    dk = int(getattr(plan, "linear_key_head_dim", 0) or getattr(plan, "linear_head_dim", 0))
    dv = int(getattr(plan, "linear_head_dim", 0))
    return math.ceil(raw_hk / degree), math.ceil(raw_hv / degree), dk, dv


def model_supports_gdn_chunk64(model: Any, tp: int) -> bool:
    """True when every GDN layer of ``model`` fits the native ``Chunk64`` composite."""
    gdn_plans = [p for p in getattr(model, "layer_plans", ()) if p.attention == "gdn"]
    if not gdn_plans:
        return False
    act = str(getattr(model, "activation_dtype", "bf16"))
    for plan in gdn_plans:
        if str(getattr(plan, "conv_state_dtype", act)) != act:
            return False
        hk, hv, dk, dv = local_gdn_geometry(plan, tp)
        state_dt = str(getattr(plan, "recurrent_state_dtype", "fp32") or "fp32")
        if not can_run_gdn_chunk64(
            hk, hv, dk, dv,
            activation_dtype=act, state_dtype=state_dt, composite_v128_only=True,
        ):
            return False
    return True


def model_supports_gdn_chunk32(model: Any, tp: int) -> bool:
    """True when every GDN layer of ``model`` fits the native ``Chunk32`` composite."""
    gdn_plans = [p for p in getattr(model, "layer_plans", ()) if p.attention == "gdn"]
    if not gdn_plans:
        return False
    act = str(getattr(model, "activation_dtype", "bf16"))
    for plan in gdn_plans:
        if str(getattr(plan, "conv_state_dtype", act)) != act:
            return False
        hk, hv, dk, dv = local_gdn_geometry(plan, tp)
        state_dt = str(getattr(plan, "recurrent_state_dtype", "fp32") or "fp32")
        if not can_run_gdn_chunk32(
            hk, hv, dk, dv, activation_dtype=act, state_dtype=state_dt,
        ):
            return False
    return True


__all__ = [
    "FLA_GDN_CHUNK_SOURCE",
    "GDN_CHUNK32_HEAD_DIM_V",
    "GDN_CHUNK32_TOKENS",
    "GDN_CHUNK64_GQA_RATIOS",
    "GDN_CHUNK64_TOKENS",
    "GDN_CHUNK64_VALUE_DIMS",
    "GDN_HEAD_DIM_K",
    "VLLM_ASCEND_GDN_SOURCE",
    "can_run_gdn_chunk32",
    "can_run_gdn_chunk64",
    "local_gdn_geometry",
    "model_supports_gdn_chunk32",
    "model_supports_gdn_chunk64",
]
