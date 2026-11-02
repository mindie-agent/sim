"""MindIE-SD (mindiesd) custom diffusion & DiT operator lowerings.

[Role]
Lower multimodal diffusion / DiT operators to ``MindIE-SD`` custom AscendC/TIK
and Catlass kernels (``rainfusion``, ``ada_bsa``, ``eagle_bsa``, ``eagle_qbsa``,
``adalayernorm``, ``norm_rope_concat``, ``laurel``, ``quant_four_over_six_a5``,
``eagle_ffn``) and model ``DiTCache`` / ``TeaCache`` / ``TaylorSeer`` effective
denoising step reduction.

[Invariants]
- Sparse attention lowerings (``rainfusion``, ``ada_bsa``, ``eagle_bsa``,
  ``eagle_qbsa``) scale active Cube FLOPs and KV HBM read traffic by
  ``(1.0 - sparsity_ratio)`` while preserving explicit block-mask estimation or
  gather traffic.
- ``quant_four_over_six_a5`` enforces Atlas A5 (``Ascend950``) SoC when
  ``soc_family`` is declared and requires group size 32 alignment.
- ``compute_mindie_sd_effective_denoising_steps`` never invents negative or
  zero computed steps (``1 <= computed_steps <= infer_steps``).

[Timeouts & Locks]
Pure bounded arithmetic; no subprocesses, sleeps, or locks.

[Side-Effect Policy]
Returns immutable ``OpSpec`` and summary dicts; no filesystem or device access.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.operators import OpSpec

MINDIE_SD_VENDOR = "mindie_sd"

_ALL_SOCS = ("Ascend910B", "Ascend910_93", "Ascend950")
_ATTN_FAMILIES = ("rainfusion", "ada_bsa", "eagle_bsa", "eagle_qbsa", "indexed_attention", "gqa_attention")

MINDIE_SD_CUSTOM_OPS: dict[str, dict[str, Any]] = {
    "rainfusion": {"native_kind": "RainFusionBlockSparseAttention", "bound": "cube", "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_rainfusion")},
    "ada_bsa": {"native_kind": "AdaBlockSparseAttention", "bound": "cube", "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_ada_bsa")},
    "eagle_bsa": {"native_kind": "EagleBlockSparseAttention", "bound": "cube", "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_eagle_bsa")},
    "eagle_qbsa": {"native_kind": "EagleQuantBlockSparseAttention", "bound": "cube", "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_eagle_qbsa")},
    "adalayernorm": {"native_kind": "AdaLayerNormFused", "bound": "memory", "supported_socs": _ALL_SOCS, "families": ("adalayernorm", "mindie_sd_adalayernorm", "rmsnorm", "add_rmsnorm")},
    "norm_rope_concat": {"native_kind": "RmsNormRopeConcatFused", "bound": "memory", "supported_socs": _ALL_SOCS, "families": ("norm_rope_concat", "mindie_sd_norm_rope_concat", "rmsnorm", "elementwise")},
    "laurel": {"native_kind": "LaurelLowRankResidual", "bound": "cube", "supported_socs": _ALL_SOCS, "families": ("laurel", "mindie_sd_laurel", "gemm", "elementwise")},
    "quant_four_over_six_a5": {"native_kind": "QuantFourOverSixA5", "bound": "vector", "supported_socs": ("Ascend950",), "families": ("quant_four_over_six_a5", "mindie_sd_quant_four_over_six_a5", "quantize", "elementwise")},
    "eagle_ffn": {"native_kind": "CatlassMatmulAddFfn", "bound": "cube", "supported_socs": _ALL_SOCS, "families": ("eagle_ffn", "mindie_sd_eagle_ffn", "gemm", "ffn")},
}

_FAMILY_TO_DEFAULT_OP: dict[str, str] = {
    "indexed_attention": "rainfusion",
    "gqa_attention": "rainfusion",
    "rmsnorm": "adalayernorm",
    "add_rmsnorm": "adalayernorm",
    "quantize": "quant_four_over_six_a5",
    "gemm": "eagle_ffn",
    "ffn": "eagle_ffn",
    "elementwise": "norm_rope_concat",
}


def recipes() -> dict[str, Any]:
    """MindIE-SD custom operators use semantic resource envelopes."""
    return {}


def _normalize_op_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    clean = str(raw).strip().lower().removeprefix("mindie_sd_")
    return clean if clean in MINDIE_SD_CUSTOM_OPS else str(raw).strip()


def _resolve_op_name(op: OpSpec, explicit_op_name: str | None) -> str:
    norm = _normalize_op_name(explicit_op_name)
    if norm is not None:
        if norm not in MINDIE_SD_CUSTOM_OPS:
            raise ModelingError(f"mindie_sd_custom_op_unknown:{explicit_op_name}")
        return norm
    for candidate in (op.recipe, op.family):
        norm_cand = _normalize_op_name(candidate)
        if norm_cand in MINDIE_SD_CUSTOM_OPS:
            return str(norm_cand)
    for candidate in (op.recipe, op.family):
        if candidate in _FAMILY_TO_DEFAULT_OP:
            return _FAMILY_TO_DEFAULT_OP[candidate]
    raise ModelingError(f"mindie_sd_custom_op_cannot_resolve:{op.family}:{op.recipe}")


def lower_op(
    op: OpSpec,
    *,
    op_name: str | None = None,
    soc_family: str | None = None,
    sparsity_ratio: float = 0.5,
    low_rank: int = 64,
    block_size: int = 64,
) -> OpSpec:
    """Lower an OpSpec into a MindIE-SD custom operator."""
    resolved = _resolve_op_name(op, op_name)
    spec = MINDIE_SD_CUSTOM_OPS[resolved]
    if op.family not in spec["families"] and op.recipe not in spec["families"]:
        raise ModelingError(f"mindie_sd_custom_op_family_mismatch:{resolved}:{op.family}")
    if soc_family is not None and soc_family not in spec["supported_socs"]:
        raise ModelingError(f"mindie_sd_custom_op_soc_unsupported:{resolved}:{soc_family}")
    if not (0.0 <= float(sparsity_ratio) < 1.0):
        raise ModelingError(f"mindie_sd_invalid_sparsity_ratio:{sparsity_ratio}")
    if block_size <= 0 or low_rank <= 0:
        raise ModelingError("mindie_sd_invalid_block_or_rank")

    meta = dict(op.metadata)
    rows = int(op.local_shape[0]) if op.local_shape else 1
    width = int(op.local_shape[-1]) if len(op.local_shape) >= 2 else rows
    elem_bytes = int(dtype_bytes(op.dtype or "bf16"))

    flops = float(op.flops)
    vec_ops = float(op.vector_ops)
    read_bytes = float(op.mte2_bytes)
    write_bytes = float(op.mte3_bytes)
    returns = tuple(op.returns)
    active_ratio = 1.0 - float(sparsity_ratio)

    if resolved in {"rainfusion", "ada_bsa", "eagle_bsa", "eagle_qbsa"}:
        q_factor = 0.5 if resolved == "eagle_qbsa" else 1.0
        base_flops = flops if flops > 0 else float(2 * rows * rows * width)
        flops = base_flops * active_ratio
        block_count = max(1, align_up(rows, block_size) // block_size)
        index_bytes = float(block_count * block_count * 4)
        estimate_ops = float(block_count * block_count * 8) if resolved == "ada_bsa" else float(block_count * 4)
        base_read = read_bytes if read_bytes > 0 else float(3 * rows * width * elem_bytes)
        read_bytes = base_read * active_ratio * q_factor + index_bytes
        vec_ops += estimate_ops
        meta["sparsity_ratio"] = float(sparsity_ratio)
        meta["block_size"] = int(block_size)
    elif resolved in {"adalayernorm", "norm_rope_concat"}:
        scale_shift_bytes = float(width * elem_bytes * (6 if resolved == "adalayernorm" else 2))
        read_bytes = max(read_bytes, float(rows * width * elem_bytes) + scale_shift_bytes)
        write_bytes = max(write_bytes, float(rows * width * elem_bytes))
        vec_ops = max(vec_ops, float(rows * width * (6 if resolved == "norm_rope_concat" else 5)))
    elif resolved == "laurel":
        flops = float(4 * rows * width * low_rank)
        weight_bytes = float(2 * width * low_rank * elem_bytes)
        read_bytes = float(rows * width * elem_bytes) + weight_bytes
        write_bytes = float(rows * width * elem_bytes)
        vec_ops = max(vec_ops, float(rows * width))
        meta["laurel_rank"] = int(low_rank)
    elif resolved == "quant_four_over_six_a5":
        aligned_width = align_up(width, 32)
        read_bytes = max(read_bytes, float(rows * aligned_width * elem_bytes))
        # MXFP4 0.5B payload + 1B E8M0 scale per 32 elements
        write_bytes = float((rows * aligned_width) // 2 + (rows * aligned_width) // 32)
        vec_ops = max(vec_ops, float(rows * aligned_width * 4))
        if "mxfp4_scale" not in returns:
            returns = (*returns, "mxfp4_scale")
    elif resolved == "eagle_ffn":
        flops = flops if flops > 0 else float(2 * rows * width * (width * 4))
        vec_ops = max(vec_ops, float(rows * width))

    meta.update({
        "mindie_sd_custom_op": f"mindie_sd_{resolved}",
        "native_vendor": MINDIE_SD_VENDOR,
        "native_kind": spec["native_kind"],
        "bound": spec["bound"],
        "native_implementation": {
            "framework": "mindie-sd",
            "vendor": MINDIE_SD_VENDOR,
            "native_kinds": (spec["native_kind"],),
            "op_name": f"mindie_sd_{resolved}",
            "selection": "explicit_caller_implementation",
            "soc_family": soc_family,
        },
    })
    return replace(
        op,
        recipe=f"mindie_sd_{resolved}",
        flops=flops,
        vector_ops=vec_ops,
        mte2_bytes=read_bytes,
        mte3_bytes=write_bytes,
        hbm_bytes=read_bytes + write_bytes,
        returns=returns,
        metadata=tuple(meta.items()),
    )


def lower_mindie_sd_custom_op(
    op_spec: OpSpec,
    hardware: Mapping[str, Any] | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Lower a MindIE-SD operator and return an analytical Roofline cost summary."""
    lowered = lower_op(op_spec, **options)
    meta = dict(lowered.metadata)
    hw = dict(hardware or {})
    cube_tflops = float(hw.get("cube_tflops", 352.0))
    vec_tflops = float(hw.get("vector_tflops", 44.0))
    hbm_gbps = float(hw.get("hbm_bandwidth_gbps", 1600.0))
    cube_us = lowered.flops / (cube_tflops * 1e6) if cube_tflops > 0 else 0.0
    vec_us = lowered.vector_ops / (vec_tflops * 1e6) if vec_tflops > 0 else 0.0
    mem_us = lowered.hbm_bytes / (hbm_gbps * 1e3) if hbm_gbps > 0 else 0.0
    return {
        "op_name": meta["mindie_sd_custom_op"],
        "native_kind": meta["native_kind"],
        "family": lowered.family,
        "bound": meta["bound"],
        "cube_flops": lowered.flops,
        "vector_flops": lowered.vector_ops,
        "memory_bytes": lowered.hbm_bytes,
        "estimated_us": max(cube_us, vec_us, mem_us, 1e-3),
        "sparsity_ratio": meta.get("sparsity_ratio", 0.0),
        "lowered_op": lowered,
    }


def compute_mindie_sd_effective_denoising_steps(
    infer_steps: int,
    warmup_steps: int = 2,
    cache_interval: int = 2,
    cache_hit_ratio: float | None = None,
) -> dict[str, float | int]:
    """Model DiTCache / TeaCache / TaylorSeer denoising step skipping."""
    total = int(infer_steps)
    warmup = max(0, min(int(warmup_steps), total))
    interval = max(1, int(cache_interval))
    if total <= 0:
        raise ModelingError(f"mindie_sd_invalid_infer_steps:{infer_steps}")
    remaining = max(0, total - warmup)
    if cache_hit_ratio is not None:
        hit = max(0.0, min(1.0, float(cache_hit_ratio)))
        cached = int(round(remaining * hit))
        computed = max(1, total - cached)
        cached = total - computed
    else:
        computed_after_warmup = math.ceil(remaining / interval) if remaining > 0 else 0
        computed = max(1, warmup + computed_after_warmup)
        cached = max(0, total - computed)
    ratio = float(computed) / float(total)
    return {
        "infer_steps": total,
        "warmup_steps": warmup,
        "cache_interval": interval,
        "computed_steps": computed,
        "cached_steps": cached,
        "cache_hit_ratio": float(cached) / float(total),
        "effective_step_ratio": ratio,
        "speedup_factor": float(total) / float(computed),
    }


__all__ = [
    "MINDIE_SD_CUSTOM_OPS",
    "MINDIE_SD_VENDOR",
    "compute_mindie_sd_effective_denoising_steps",
    "lower_mindie_sd_custom_op",
    "lower_op",
    "recipes",
]
