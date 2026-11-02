"""xLLM custom_xllm_math operator lowerings for Ascend A2/A3/A5.

[Role]
Lower semantic attention, MoE, GDN, OneRec beam search, and A5 KV-cache write
operators to xLLM's ``vendors/custom_xllm_math`` custom operator suite
(``smla_va``, ``sparse_flash_attention_lse``, ``moe_grouped_matmul_swiglu_quant``,
``mega_gdn_decode``, ``onerec_final_beam_select``, ``reshape_and_cache_a5``).

[Invariants]
- Every lowered operator preserves logical FLOPs and records explicit
  ``custom_xllm_math`` vendor provenance in ``metadata["native_implementation"]``.
- ``sparse_flash_attention_lse`` appends ``softmax_lse`` (FP32 per head-row) to
  ``returns`` and ``mte3_bytes`` when ``return_lse=True``.
- ``moe_grouped_matmul_swiglu_quant`` fuses gate/up GEMM, SwiGLU, and dynamic
  INT8/MXFP8 output quantization without fabricating expert routing counts.
- ``reshape_and_cache_a5`` enforces Ascend 950 (A5) SoC compatibility when
  ``soc_family`` is declared and aligns page writes to ``kv_page_size``.

[Timeouts & Locks]
Pure bounded operator arithmetic; no subprocesses, sleeps, or locks.

[Side-Effect Policy]
Returns immutable ``OpSpec`` records; no filesystem or device access.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.operators import OpSpec

XLLM_VENDOR = "custom_xllm_math"

XLLM_CUSTOM_OPS: dict[str, dict[str, Any]] = {
    "smla_va": {
        "native_kind": "SmlaVa",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/smla_va",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("smla_va", "mla_attention", "paged_mla_attention"),
    },
    "sparse_flash_attention_lse": {
        "native_kind": "SparseFlashAttentionLse",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/sparse_flash_attention_lse",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("sparse_flash_attention_lse", "indexed_attention", "mla_attention"),
    },
    "moe_grouped_matmul_swiglu_quant": {
        "native_kind": "MoeGroupedMatmulSwigluQuant",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/moe_grouped_matmul_swiglu_quant",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("moe_grouped_matmul_swiglu_quant", "grouped_gemm"),
    },
    "mega_gdn_decode": {
        "native_kind": "MegaGdnDecode",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/mega_gdn_decode",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("mega_gdn_decode", "gdn_decode"),
    },
    "onerec_final_beam_select": {
        "native_kind": "OnerecFinalBeamSelect",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/onerec_final_beam_select",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("onerec_final_beam_select", "beam_select", "elementwise"),
    },
    "reshape_and_cache_a5": {
        "native_kind": "ReshapeAndCacheA5",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/reshape_and_cache_a5",
        "supported_socs": ("Ascend950",),
        "families": ("reshape_and_cache_a5", "kv_cache_write"),
    },
}

_FAMILY_TO_DEFAULT_OP: dict[str, str] = {
    "smla_va": "smla_va",
    "mla_attention": "smla_va",
    "paged_mla_attention": "smla_va",
    "sparse_flash_attention_lse": "sparse_flash_attention_lse",
    "indexed_attention": "sparse_flash_attention_lse",
    "moe_grouped_matmul_swiglu_quant": "moe_grouped_matmul_swiglu_quant",
    "grouped_gemm": "moe_grouped_matmul_swiglu_quant",
    "mega_gdn_decode": "mega_gdn_decode",
    "gdn_decode": "mega_gdn_decode",
    "onerec_final_beam_select": "onerec_final_beam_select",
    "beam_select": "onerec_final_beam_select",
    "elementwise": "onerec_final_beam_select",
    "reshape_and_cache_a5": "reshape_and_cache_a5",
    "kv_cache_write": "reshape_and_cache_a5",
}


def recipes() -> dict[str, Any]:
    """xLLM custom_xllm_math operators use semantic resource envelopes."""
    return {}


def _resolve_op_name(op: OpSpec, explicit_op_name: str | None) -> str:
    if explicit_op_name is not None:
        clean = str(explicit_op_name).strip()
        if clean not in XLLM_CUSTOM_OPS:
            raise ModelingError(f"xllm_custom_op_unknown:{clean}")
        return clean
    if op.recipe in XLLM_CUSTOM_OPS:
        return op.recipe
    if op.family in _FAMILY_TO_DEFAULT_OP:
        return _FAMILY_TO_DEFAULT_OP[op.family]
    raise ModelingError(f"xllm_custom_op_cannot_resolve:{op.family}:{op.recipe}")


def lower_op(
    op: OpSpec,
    *,
    op_name: str | None = None,
    soc_family: str | None = None,
    return_lse: bool = True,
    beam_width: int | None = None,
    kv_page_size: int = 128,
) -> OpSpec:
    """Lower one OpSpec into an explicit xLLM custom_xllm_math operator."""
    resolved = _resolve_op_name(op, op_name)
    spec = XLLM_CUSTOM_OPS[resolved]
    if op.family not in spec["families"] and op.recipe not in spec["families"]:
        raise ModelingError(f"xllm_custom_op_family_mismatch:{resolved}:{op.family}")
    if soc_family is not None and soc_family not in spec["supported_socs"]:
        raise ModelingError(f"xllm_custom_op_soc_unsupported:{resolved}:{soc_family}")
    if kv_page_size <= 0:
        raise ModelingError(f"xllm_custom_op_invalid_page_size:{kv_page_size}")

    meta = dict(op.metadata)
    rows = int(op.local_shape[0]) if op.local_shape else 1
    width = int(op.local_shape[-1]) if len(op.local_shape) >= 2 else rows
    elem_bytes = int(dtype_bytes(op.dtype or "bf16"))

    read_bytes = float(op.mte2_bytes)
    write_bytes = float(op.mte3_bytes)
    vec_ops = float(op.vector_ops)
    returns = tuple(op.returns)

    if resolved == "smla_va":
        # Vector-absorption Sparse MLA avoids materializing uncompressed KV intermediates.
        vec_ops = max(vec_ops, float(rows * width))
        if "attn_out" not in returns:
            returns = (*returns, "attn_out")
    elif resolved == "sparse_flash_attention_lse":
        lse_bytes = float(rows * 4) if return_lse else 0.0
        write_bytes += lse_bytes
        vec_ops += float(rows * 2) if return_lse else 0.0
        if return_lse and "softmax_lse" not in returns:
            returns = (*returns, "softmax_lse")
        meta["xllm_lse_bytes"] = int(lse_bytes)
    elif resolved == "moe_grouped_matmul_swiglu_quant":
        # Fuses gate+up projection with SwiGLU and dynamic int8/mxfp8 quantization;
        # output width is halved by SwiGLU and quantized to 1B + FP32 per-row scale.
        out_cols = max(1, width // 2)
        quant_out_bytes = float(rows * out_cols + rows * 4)
        write_bytes = min(write_bytes, quant_out_bytes) if write_bytes > 0 else quant_out_bytes
        vec_ops += float(rows * out_cols * 4)
        if "dynamic_scale" not in returns:
            returns = (*returns, "dynamic_scale")
        meta["xllm_swiglu_quant_fused"] = True
    elif resolved == "mega_gdn_decode":
        # Fuses GDN recurrent state read/update with gate/beta vector ops in one kernel.
        state_bytes = float(rows * width * elem_bytes)
        read_bytes = max(read_bytes, state_bytes)
        write_bytes = max(write_bytes, float(rows * width * elem_bytes))
        vec_ops += float(rows * width * 3)
        if "recurrent_state" not in returns:
            returns = (*returns, "recurrent_state")
    elif resolved == "onerec_final_beam_select":
        k_beam = int(beam_width if beam_width is not None else meta.get("beam_width", 32))
        if k_beam <= 0:
            raise ModelingError(f"xllm_onerec_invalid_beam_width:{k_beam}")
        # Writes selected token IDs (int64) and accumulated log-probabilities (fp32).
        beam_write = float(rows * k_beam * (8 + 4))
        write_bytes = max(write_bytes, beam_write)
        vec_ops = max(vec_ops, float(rows * max(width, k_beam) * 2))
        for item in ("beam_ids", "beam_scores"):
            if item not in returns:
                returns = (*returns, item)
        meta["beam_width"] = k_beam
    elif resolved == "reshape_and_cache_a5":
        aligned_rows = align_up(rows, kv_page_size)
        page_write = float(aligned_rows * width * elem_bytes)
        write_bytes = max(write_bytes, page_write)
        meta["kv_page_size"] = int(kv_page_size)
        meta["aligned_page_rows"] = int(aligned_rows)
        if "xtensor_kv_page" not in returns:
            returns = (*returns, "xtensor_kv_page")

    meta.update({
        "xllm_custom_op": resolved,
        "native_vendor": XLLM_VENDOR,
        "native_kind": spec["native_kind"],
        "native_source": {
            "repository": "xllm",
            "vendor": XLLM_VENDOR,
            "path": spec["source_path"],
            "scope": "independent_source_semantics; custom_xllm_math_lowering",
        },
        "native_implementation": {
            "framework": "xllm",
            "vendor": XLLM_VENDOR,
            "native_kinds": (spec["native_kind"],),
            "op_name": resolved,
            "selection": "explicit_caller_implementation",
            "soc_family": soc_family,
        },
    })

    return replace(
        op,
        recipe=resolved,
        vector_ops=vec_ops,
        mte2_bytes=read_bytes,
        mte3_bytes=write_bytes,
        hbm_bytes=read_bytes + write_bytes,
        returns=returns,
        metadata=tuple(meta.items()),
    )


__all__ = ["XLLM_CUSTOM_OPS", "XLLM_VENDOR", "lower_op", "recipes"]
