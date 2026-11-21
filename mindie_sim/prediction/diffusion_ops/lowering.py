"""Diffusion operator lowering and denoising summaries.

[Role] Lower declared diffusion operators and summarize denoising work.
[Invariants] Explicit operator names and rate declarations determine lowering and summaries.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from .attention import _norm_rope_contract, _sparse_contract
from .block_ops import (
    _adalayernorm_contract,
    _eagle_ffn_contract,
    _laurel_contract,
    _quant_four_over_six_contract,
)
from .contracts import _input_dtype, _shape, _tokens_width
from .registry import MINDIE_SD_CUSTOM_OPS, MINDIE_SD_VENDOR, _resolve_op_name


def lower_op(
    op: OpSpec,
    *,
    op_name: str | None = None,
    soc_family: str | None = None,
    sparsity_ratio: float | None = None,
    low_rank: int | None = None,
    block_size: int | None = None,
    kv_dtype: str | None = None,
) -> OpSpec:
    """Lower one exact semantic occurrence into a MindIE-SD custom operator."""
    resolved = _resolve_op_name(op, op_name)
    spec = MINDIE_SD_CUSTOM_OPS[resolved]
    if op.family not in spec["families"] and op.recipe not in spec["families"]:
        raise ModelingError(f"mindie_sd_custom_op_family_mismatch:{resolved}:{op.family}")
    if soc_family is not None and soc_family not in spec["supported_socs"]:
        raise ModelingError(f"mindie_sd_custom_op_soc_unsupported:{resolved}:{soc_family}")

    meta = dict(op.metadata)
    dtype = _input_dtype(op, meta)
    if dtype not in spec["input_dtypes"]:
        raise ModelingError(
            f"mindie_sd_custom_op_input_dtype_unsupported:{resolved}:{dtype}")

    flops = float(op.flops)
    vec_ops = float(op.vector_ops)
    read_bytes = float(op.mte2_bytes)
    write_bytes = float(op.mte3_bytes)
    returns = tuple(op.returns)
    shape = _shape(op)
    rows, width = _tokens_width(op)
    elem_bytes = dtype_bytes(dtype)
    contract = None
    matrix_flops = op.matrix_flops
    options = {"sparsity_ratio": sparsity_ratio, "block_size": block_size,
               "low_rank": low_rank, "kv_dtype": kv_dtype}

    if spec.get("sparse"):
        contract = _sparse_contract(op, meta, dtype, returns, resolved, options)
        if contract.get("attributes", {}).get("kv_dtype") is None and spec.get("quantized_cache"):
            raise ModelingError("mindie_sd_eagle_qbsa_requires_declared_cache_dtype")
    elif resolved == "norm_rope_concat":
        contract = _norm_rope_contract(op, meta, dtype, returns)
    elif resolved == "adalayernorm":
        contract = _adalayernorm_contract(op, meta, dtype, returns)
    elif resolved == "laurel":
        if low_rank is not None:
            meta["low_rank"] = int(low_rank)
        contract = _laurel_contract(op, meta, dtype, returns)
    elif resolved == "quant_four_over_six_a5":
        contract = _quant_four_over_six_contract(op, meta, dtype, returns)
    elif resolved == "eagle_ffn":
        contract = _eagle_ffn_contract(op, meta, dtype, returns)
    if contract is not None:
        read_bytes = float(contract["mte2_bytes"])
        write_bytes = float(contract["mte3_bytes"])
        vec_ops = float(contract["vector_ops"])
        flops = float(contract["logical_flops"])
        matrix_flops = tuple(contract["matrix_flops"])
        meta["mindie_sd_contract"] = contract
        meta.update(contract["attributes"])
        if resolved == "norm_rope_concat":
            extra = (
                "query_output", "key_output", "value_output",
                "norm_query_mean", "norm_query_rstd", "norm_key_mean",
                "norm_key_rstd", "norm_added_query_mean",
                "norm_added_query_rstd", "norm_added_key_mean",
                "norm_added_key_rstd",
            )
        elif resolved == "eagle_ffn":
            extra = ("y",)
        elif spec.get("sparse"):
            extra = ("attention_out",)
            if "softmax_lse" in {item["name"] for item in contract["writes"]}:
                extra = (*extra, "softmax_lse")
        elif resolved == "adalayernorm":
            extra = ("y",) + tuple(item["name"] for item in contract["writes"]
                                    if item["name"] in {"mean", "rstd"})
        elif resolved == "laurel":
            extra = ("y",)
            meta["laurel_rank"] = int(contract["attributes"]["low_rank"])
        elif resolved == "quant_four_over_six_a5":
            extra = ("mxfp4_payload", "mxfp4_scale")
        else:
            extra = ()
        returns = tuple(dict.fromkeys((*returns, *extra)))
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
    if contract is not None:
        public = next((item for item in contract["writes"]
                       if item.get("materialized", True)
                       and item.get("kind") in {"value", "legacy"}
                       and item["name"] not in {"mean", "rstd"}),
                      next((item for item in contract["writes"]
                            if item.get("materialized", True)), None))
        output_shape = tuple(public["shape"]) if public else tuple(op.local_shape)
    else:
        output_shape = tuple(op.local_shape)
    return replace(
        op,
        recipe=f"mindie_sd_{resolved}",
        flops=flops,
        vector_ops=vec_ops,
        mte2_bytes=read_bytes,
        mte3_bytes=write_bytes,
        hbm_bytes=read_bytes + write_bytes,
        returns=returns,
        matrix_flops=matrix_flops,
        local_shape=output_shape,
        padded_shape=output_shape,
        launch_shape=output_shape,
        metadata=tuple(meta.items()),
    )


def lower_mindie_sd_custom_op(
    op_spec: OpSpec,
    hardware: Mapping[str, Any] | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Lower one operator and return its work plus a disclosed Roofline view.

    No hardware rate is defaulted.  Without declared ``cube_tflops``,
    ``vector_tflops`` and ``hbm_bandwidth_gbps`` the result carries work and a
    basis string instead of an invented duration.
    """
    lowered = lower_op(op_spec, **options)
    meta = dict(lowered.metadata)
    result: dict[str, Any] = {
        "op_name": meta["mindie_sd_custom_op"],
        "native_kind": meta["native_kind"],
        "family": lowered.family,
        "bound": meta["bound"],
        "cube_flops": lowered.flops,
        "vector_flops": lowered.vector_ops,
        "memory_bytes": lowered.hbm_bytes,
        "sparsity_ratio": float(meta.get("sparsity_ratio", 0.0)),
        **({"laurel_rank": meta["laurel_rank"]} if "laurel_rank" in meta else {}),
        "hardware": dict(hardware or {}),
        "estimate_basis": "requires_declared_hardware_rates",
        "estimated_us": None,
    }
    hw = dict(hardware or {})
    required = ("cube_tflops", "vector_tflops", "hbm_bandwidth_gbps")
    if hw:
        if any(key not in hw for key in required):
            result["estimate_basis"] = "declared_hardware_rates_incomplete"
            result["lowered_op"] = lowered
            return result
        cube_tflops = float(hw["cube_tflops"])
        vec_tflops = float(hw["vector_tflops"])
        hbm_gbps = float(hw["hbm_bandwidth_gbps"])
        if min(cube_tflops, vec_tflops, hbm_gbps) <= 0:
            raise ModelingError("mindie_sd_hardware_rates_must_be_positive")
        cube_us = lowered.flops / (cube_tflops * 1e6)
        vec_us = lowered.vector_ops / (vec_tflops * 1e6)
        mem_us = lowered.hbm_bytes / (hbm_gbps * 1e3)
        result["estimated_us"] = max(cube_us, vec_us, mem_us)
        result["estimate_basis"] = "declared_hardware_roofline"
    result["lowered_op"] = lowered
    return result


def compute_mindie_sd_effective_denoising_steps(
    infer_steps: int,
    warmup_steps: int = 2,
    cache_interval: int = 2,
    cache_hit_ratio: float | None = None,
) -> dict[str, float | int]:
    """Model DiTCache/TeaCache/TaylorSeer denoising step skipping."""
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
