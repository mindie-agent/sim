"""Catalog quantization-plan adapters for the nine config families.

[Role]
  Translate native quantization configs into explicit weight/activation/expert
  formats, block shapes, scale dtypes and unquantized exclusions.

[Invariants]
  * Source checkpoint format and runtime residency are separate.
  * FP8 block scales and FP4 group32 scales retain their native dtypes.
  * No global activation dtype is inferred from a quantization method.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping

from .a3_catalog_types import QuantSpec, merge_source


def _quant_deepseek(c: Mapping[str, Any], *, v41: bool) -> QuantSpec:
    q = c.get("quantization_config")
    q = q if isinstance(q, Mapping) else {}
    block = q.get("weight_block_size", [32, 32] if v41 else [128, 128])
    block = (int(block[0]), int(block[1]))
    scale_fmt = str(q.get("scale_fmt", "ue8m0")).lower()
    scale_bytes = 1.0 if "e8m0" in scale_fmt else 4.0
    return QuantSpec(
        name="fp8_dynamic+mxfp4_experts" if v41 else "fp8_dynamic+mxfp4_experts",
        weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        expert_weight_format="mxfp4",
        weight_block=block, expert_block=32,
        scale_format="fp8_e8m0" if scale_bytes == 1 else "fp32",
        scale_bytes=scale_bytes, kv_dtype="bf16",
        index_dtype="int8", index_scale_dtype="fp16",
        exclusions=("router", "lm_head", "embedding", "norm", "mhc",
                    "compressor", "vision", "projector", "engram_gate"),
        source=merge_source({"quant_method": q.get("quant_method", "fp8"),
                             "weight_block_size": str(block),
                             "scale_fmt": scale_fmt,
                             "expert_dtype": c.get("expert_dtype", "fp4")}),
        notes=(
            "ordinary linears use FP8 weights with the declared block scales; "
            "routed experts use FP4 group32 with E8M0 scales",
            "activation_scheme=dynamic is a kernel path, not a global model dtype",
        ),
    )


def _quant_glm(c: Mapping[str, Any]) -> QuantSpec:
    q = c.get("quantization_config")
    q = q if isinstance(q, Mapping) else {}
    block = q.get("weight_block_size", [128, 128])
    block = (int(block[0]), int(block[1]))
    excludes = tuple(str(v) for v in (q.get("modules_to_not_convert") or ()))
    return QuantSpec(
        name="fp8_dynamic",
        weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        expert_weight_format="fp8_e4m3",
        weight_block=block, expert_block=32,
        scale_format="fp32", scale_bytes=4.0,
        kv_dtype="bf16", index_dtype="bf16", index_scale_dtype="none",
        exclusions=("router", "lm_head", "embedding", "norm", "mhc",
                    "compressor", "vision", "projector")
        + (tuple("module:" + name for name in excludes) if excludes else ()),
        source=merge_source({"quant_method": q.get("quant_method", "fp8"),
                             "weight_block_size": str(block),
                             "module_exclusions": str(len(excludes))}),
        notes=("FP8 128x128 blocks use FP32 scales in this model",),
    )


def _quant_kimi(c: Mapping[str, Any]) -> QuantSpec:
    q = c.get("quantization_config")
    q = q if isinstance(q, Mapping) else {}
    return QuantSpec(
        name="mxfp4_routed_only",
        weight_format="bf16", activation_format="bf16",
        expert_weight_format="mxfp4",
        weight_block=(0, 0), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0,
        kv_dtype="bf16", index_dtype="bf16", index_scale_dtype="none",
        exclusions=("attn", "mla", "kda", "shared_expert", "dense_mlp",
                    "router", "lm_head", "embedding", "norm", "mhc",
                    "vision", "projector", "conv_weight_f32"),
        source=merge_source({"quant_method": q.get("quant_method", "compressed-tensors"),
                             "format": q.get("format", "mxfp4-pack-quantized")}),
        notes=("MXFP4 group32 applies to routed expert matrices only; "
               "attention/shared/dense/LM-head/vision stay BF16",),
    )


def _quant_bf16(c: Mapping[str, Any], *, name: str) -> QuantSpec:
    return QuantSpec(
        name=name, weight_format="bf16", activation_format="bf16",
        expert_weight_format="bf16", kv_dtype="bf16",
        index_dtype="bf16", index_scale_dtype="none",
        exclusions=(), source=merge_source({"quantization": "none"}),
        notes=("no weight quantization declared; compute and activation are BF16",),
    )
