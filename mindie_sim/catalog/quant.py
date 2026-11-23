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
from dataclasses import replace

from mindie_sim.catalog.types import QuantSpec, merge_source

FP32_SCALE_BYTES = 4.0


def projection_quant(quant: QuantSpec, plan) -> QuantSpec:
    """Resolve per-layer native projection defaults before explicit roles.

    A global quantization preset describes the default representation. Native
    source modules may opt out; user parent and child category declarations
    still win. This same record is used by execution and resident weights.
    """
    defaults = {}
    if plan.attention == "mla":
        # Historically the index weighting projection stayed native while
        # index Q/K may be quantized. Give that matrix its own override role.
        defaults["indexer.weights"] = {
            "weight_format": quant.native_format,
            "activation_format": quant.native_format,
        }
    if plan.projection_dtype:
        fields = {"weight_format": plan.projection_dtype,
                  "activation_format": plan.projection_dtype}
        defaults[plan.attention] = fields
        if plan.attention == "mla":
            defaults["indexer_fp8"] = fields
    if plan.index_kpool > 1 and plan.attention != "gqa":
        # The retained analytic fallback casts hidden once and keeps its own
        # FP32 K/gate copies.  ``indexer.weights`` is deliberately *not* in
        # this set: the checkpoint stores BF16 head weights and both the native
        # CANN callers and the generic projection consume them directly, so a
        # synthesised ``_prepared_fp32`` copy would be a fabricated parameter.
        for category in ("indexer.pool_projection", "indexer.pool_gate"):
            defaults[category] = {"weight_format": "fp32", "activation_format": "fp32"}
    explicit = {key: dict(value) for key, value in quant.category_overrides}
    # Apply explicit ancestors after *all* source defaults: an explicit
    # indexer parent must override the native indexer.weights child default.
    merged = {}
    for category in defaults.keys() | explicit.keys():
        values = dict(defaults.get(category, {}))
        names = category.split(".")
        for i in range(len(names)):
            values.update(explicit.get(".".join(names[:i + 1]), {}))
        merged[category] = values
    return replace(quant, category_overrides=merged) if merged else quant


#: Routed MXFP4 experts keep the 32-element group regardless of the plain
#: linear weight block (128 for V4, 32 for V4.1).
MX_EXPERT_BLOCK = 32
MX_EXPERT_SCALE_PACK = 2
LONG_CONTEXT_KV_QUANT_THRESHOLD = 32768
_STANDARD_MXFP4_EXCLUSIONS: tuple[str, ...] = (
    "mtp", "norm", "router", "lm_head", "embedding", "mhc",
    "compressor", "vision", "projector", "conv_weight_f32", "conv_weight_bf16",
)
_MXFP4_ROUTED_ACTIVATION = {
    "activation_format": "fp8_e4m3",
    "activation_group_size": MX_EXPERT_BLOCK,
    "activation_scale_format": "fp8_e8m0",
    "activation_scale_pack": MX_EXPERT_SCALE_PACK,
}
_MXFP4_LINEAR_OVERRIDE = {
    "weight_format": "mxfp4",
    "weight_block": (1, MX_EXPERT_BLOCK),
    "expert_block": MX_EXPERT_BLOCK,
    "scale_format": "fp8_e8m0",
    "scale_bytes": 1.0,
    **_MXFP4_ROUTED_ACTIVATION,
}


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
        weight_block=block, expert_block=MX_EXPERT_BLOCK,
        scale_format="fp8_e8m0" if scale_bytes == 1 else "fp32",
        scale_bytes=scale_bytes, kv_dtype="bf16",
        activation_group_size=block[1],
        activation_scale_format="fp8_e8m0" if scale_bytes == 1 else "fp32",
        activation_scale_pack=2 if scale_bytes == 1 else 1,
        index_dtype="int8", index_scale_dtype="fp16",
        # V4.1 executes the Engram gate as an MXFP8 (32, 32) projection; the
        # earlier V4 checkpoint keeps it in the BF16 exclusion set.
        exclusions=(("router", "lm_head", "embedding", "norm", "mhc",
                     "compressor", "vision", "projector")
                    + (() if v41 else ("engram_gate",))),
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
    if not isinstance(q, Mapping) or not q:
        # The A5 Flash release is served as the dequantised BF16 checkpoint:
        # the serve log reports no quantization signature and the 642.65 GB
        # shard set is exactly two bytes per logical parameter.
        return _quant_bf16(c, name="bf16_glm")
    block = q.get("weight_block_size", [128, 128])
    block = (int(block[0]), int(block[1]))
    excludes = tuple(str(v) for v in (q.get("modules_to_not_convert") or ()))
    return QuantSpec(
        name="fp8_dynamic",
        weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        expert_weight_format="fp8_e4m3",
        weight_block=block, expert_block=32,
        scale_format="fp32", scale_bytes=4.0,
        activation_group_size=block[1], activation_scale_format="fp32",
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
        weight_block=(0, 0), expert_block=MX_EXPERT_BLOCK,
        scale_format="fp8_e8m0", scale_bytes=1.0,
        kv_dtype="bf16", index_dtype="bf16", index_scale_dtype="none",
        category_overrides={
            "routed_expert": _MXFP4_ROUTED_ACTIVATION,
            "dspark_routed_expert": _MXFP4_ROUTED_ACTIVATION,
            "latent_moe": _MXFP4_LINEAR_OVERRIDE,
            "attn_res_proj": _MXFP4_LINEAR_OVERRIDE,
        },
        exclusions=("attn", "mla", "kda", "shared_expert", "dense_mlp",
                    "router", "lm_head", "embedding", "norm", "mhc",
                    "vision", "projector", "conv_weight_f32"),
        source=merge_source({"quant_method": q.get("quant_method", "compressed-tensors"),
                             "format": q.get("format", "mxfp4-pack-quantized")}),
        notes=("MXFP4 group32 applies to routed expert, latent MoE (down/up) and "
               "AttnRes projection matrices with FP8 E4M3 activations and E8M0 "
               "microscales; attention/shared/dense/router/LM-head/vision stay BF16",),
    )


def _quant_qwen_mxfp8() -> QuantSpec:
    """Declared W8A8-MXFP8 recipe the A5 950DT Qwen3.8 serving row used.

    The checkpoint publishes ``W8A8_MXFP8`` (FP8 E4M3 weights and dynamic
    activations, E8M0 block-32 scales).  Only the full-attention q/k/v/o
    projections and the dense MLP gate/up/down matrices are quantized.  The
    checkpoint's safetensors headers show the rest is BF16/FP32: norm,
    embedding, LM head, vision, the depthwise conv weights, the 48 GDN
    ``linear_attn`` projections (``in_proj_qkv/qkvz``, ``in_proj_z/ba``,
    ``out_proj``) and the MTP draft layer (``mtp.layers.0.*``).  This is the
    *declared* runtime representation, not a kernel claim: the FP8 GEMM is
    priced through the measured MXFP8/fp8 dense row when the artifact
    publishes one.
    """
    return QuantSpec(
        name="W8A8_MXFP8",
        weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        expert_weight_format="fp8_e4m3",
        weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0,
        activation_group_size=32, activation_scale_format="fp8_e8m0", activation_scale_pack=2,
        kv_dtype="bf16", index_dtype="bf16", index_scale_dtype="none",
        # ``gdn``: every GDN linear_attn projection is stored BF16 in this
        # checkpoint; ``mtp``: the draft/MTP layer is BF16 too.  Leaving them
        # unexcluded priced 1 872 GDN projection calls at p4096 on the FP8 row.
        exclusions=("gdn", "mtp", "norm", "lm_head", "embedding", "vision",
                    "projector", "conv_weight_f32", "conv_weight_bf16"),
        source=merge_source({
            "quant_method": "mxfp8",
            "weight_block_size": "1x32",
            "activation_scheme": "dynamic",
            "scale_fmt": "ue8m0",
            "checkpoint": "Eco-Tech/Qwen3.8-27B-w8a8-mxfp8",
        }),
        notes=(
            "measured A5 recipe: W8A8-MXFP8 (FP8 E4M3 / E8M0 block-32)",
            "only full-attention q/k/v/o and dense MLP gate/up/down are quantized",
            "GDN linear_attn projections + conv, MTP layer, norms, embedding, "
            "LM head and vision stay BF16/FP32",
        ),
    )



def _quant_w8a8_dynamic() -> QuantSpec:
    """Declared W8A8-DYNAMIC int8 recipe of the Qwen3.5-35B-A3B A5 checkpoint.

    The checkpoint's ``quant_model_description.json`` marks every body layer's
    attention projections and every routed expert gate/up/down as
    ``W8A8_DYNAMIC`` (int8 weights, per-channel float32 ``weight_scale``,
    dynamic per-token int8 activation quant).  Norms, embedding, ``lm_head``,
    the GDN depthwise conv, vision and the whole MTP draft layer are ``FLOAT``
    (bf16/fp32), so they must not be priced on an int8 GEMM row.

    This is the declared runtime representation, not a kernel claim.  The int8
    cube row is reached only when the op's *physical* identity is its compute
    dtype (see :func:`mindie_sim.catalog.ops.compute_dtype_for`), so the
    artifact's ``npu_quant_matmul_int8_w8a8_dynamic`` rows are asked for a W8A8
    case and never for a bf16/mxfp8 one.
    """
    return QuantSpec(
        name="W8A8_DYNAMIC",
        weight_format="int8", activation_format="int8",
        expert_weight_format="int8",
        weight_block=(1, 1), expert_block=1,
        scale_format="fp32", scale_bytes=FP32_SCALE_BYTES,
        # QuantBatchMatmulV3 normal-caller execution ABI: one BF16 weight
        # scale per output channel and one FLOAT dynamic scale per activation
        # row.  The checkpoint storage remains per-channel FP32 (above); this
        # runtime separation is not a general "all W8A8 is 2-byte" rule.
        runtime_weight_scale_format="bf16", runtime_weight_scale_bytes=2.0,
        runtime_activation_scale_format="fp32",
        runtime_int8_weight_metadata="scale_only",
        # The five INT8xINT8 QuantBatchMatmulV3 families return BF16; this is
        # declared per source role, not a blanket quantized-projection rule.
        category_overrides={
            "gdn": {"output_format": "bf16"},
            "dense_mlp": {"output_format": "bf16"},
            "attn.qkv": {"output_format": "bf16"},
            "attn.o": {"output_format": "bf16"},
        },
        kv_dtype="bf16", index_dtype="bf16", index_scale_dtype="none",
        exclusions=("mtp", "norm", "router", "lm_head", "embedding", "mhc",
                    "compressor", "vision", "projector",
                    "conv_weight_f32", "conv_weight_bf16"),
        source=merge_source({
            "quant_method": "w8a8_dynamic",
            "activation_scheme": "dynamic",
            "weight_scale": "per_channel_fp32",
            "checkpoint": "Qwen3.5-35B-A3B-w8a8-mtp",
        }),
        notes=(
            "measured A5 recipe: W8A8 int8 dynamic (per-channel fp32 weight scale, "
            "per-token int8 activations)",
            "body attention + routed expert projections are int8; MTP, norms, "
            "embedding, LM head, conv and vision stay BF16/FP32",
        ),
    )

def _quant_w4a8_dynamic() -> QuantSpec:
    """Declared W4A8-DYNAMIC expert + W8A8-DYNAMIC linear recipe of GLM-5.3-W4A8.

    The measured A3 checkpoint's ``quant_model_description.json`` carries three
    classes: ``W4A8_DYNAMIC`` (233 472 entries) on the routed expert
    gate/up/down matrices, ``W8A8_DYNAMIC`` (1 726 entries) on the attention,
    shared-expert and first-dense matrices, and ``FLOAT`` (644 entries) on the
    norms, router gates, embedding and LM head.  This is the declared runtime
    representation of the *served* checkpoint, not a kernel claim: the int4
    expert stream is priced through the measured W4A8 grouped/MC2 row when one
    exists, and the int8 projections through the W8A8 row.

    The per-channel FP32 weight scale is one value per output row, which is why
    ``expert_block=1`` and the int4 byte model charges ``rows`` scales rather
    than a 2-D block grid.
    """
    return QuantSpec(
        name="W4A8_DYNAMIC",
        weight_format="int8", activation_format="int8",
        expert_weight_format="int4",
        weight_block=(1, 1), expert_block=1,
        scale_format="fp32", scale_bytes=FP32_SCALE_BYTES,
        kv_dtype="bf16", index_dtype="bf16", index_scale_dtype="none",
        exclusions=("mtp", "norm", "router", "lm_head", "embedding", "mhc",
                    "compressor", "vision", "projector",
                    "conv_weight_f32", "conv_weight_bf16"),
        source=merge_source({
            "quant_method": "w4a8_dynamic",
            "activation_scheme": "dynamic",
            "expert_weight_scale": "per_channel_fp32",
            "linear_weight_scale": "per_channel_fp32",
            "checkpoint": "GLM-5.3-W4A8-INT",
        }),
        notes=(
            "measured A3 recipe: W4A8_DYNAMIC routed experts (packed int4, "
            "per-channel fp32 scale) + W8A8_DYNAMIC attention/shared/dense",
            "MTP, norms, router, embedding and LM head stay BF16/FP32",
        ),
    )


def _quant_fp16() -> QuantSpec:
    return QuantSpec(
        name="fp16", weight_format="fp16", activation_format="fp16",
        expert_weight_format="fp16", kv_dtype="fp16", index_dtype="fp16",
        index_scale_dtype="none", native_format="fp16",
        exclusions=(), source=merge_source({"quantization": "none"}),
        notes=("no weight quantization declared; compute and activation are FP16",),
    )


def _quant_mxfp4_all(
    *, exclusions: tuple[str, ...] = _STANDARD_MXFP4_EXCLUSIONS,
) -> QuantSpec:
    return QuantSpec(
        name="MXFP4", weight_format="mxfp4", activation_format="fp8_e4m3",
        expert_weight_format="mxfp4",
        weight_block=(1, MX_EXPERT_BLOCK),
        expert_block=MX_EXPERT_BLOCK,
        scale_format="fp8_e8m0", scale_bytes=1.0,
        activation_group_size=MX_EXPERT_BLOCK,
        activation_scale_format="fp8_e8m0",
        activation_scale_pack=MX_EXPERT_SCALE_PACK,
        kv_dtype="bf16", index_dtype="bf16", index_scale_dtype="none",
        exclusions=exclusions,
        source=merge_source({
            "quant_method": "mxfp4",
            "weight_block_size": "1x32",
            "scale_fmt": "ue8m0",
        }),
        notes=(
            "MXFP4 group32 weights with FP8 E4M3 activations and E8M0 microscales",
            "MTP, norms, router, embedding, LM head, mHC, vision and conv stay BF16/FP32",
        ),
    )


def _quant_w4a8_mxfp() -> QuantSpec:
    """MXFP8 linears + MXFP4 routed experts (matching Eco-Tech W4A8_MXFP checkpoints)."""
    return QuantSpec(
        name="W8A8_MXFP8+W4A8_MXFP",
        weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        expert_weight_format="mxfp4",
        weight_block=(1, MX_EXPERT_BLOCK),
        expert_block=MX_EXPERT_BLOCK,
        scale_format="fp8_e8m0", scale_bytes=1.0,
        activation_group_size=MX_EXPERT_BLOCK,
        activation_scale_format="fp8_e8m0",
        activation_scale_pack=MX_EXPERT_SCALE_PACK,
        kv_dtype="bf16", index_dtype="bf16", index_scale_dtype="none",
        category_overrides={
            "routed_expert": _MXFP4_ROUTED_ACTIVATION,
            "dspark_routed_expert": _MXFP4_ROUTED_ACTIVATION,
            "latent_moe": _MXFP4_LINEAR_OVERRIDE,
            "attn_res_proj": _MXFP4_LINEAR_OVERRIDE,
        },
        exclusions=_STANDARD_MXFP4_EXCLUSIONS,
        source=merge_source({
            "quant_method": "w4a8_mxfp",
            "weight_block_size": "1x32",
            "scale_fmt": "ue8m0",
        }),
        notes=(
            "MXFP8 group32 linears + MXFP4 group32 routed experts with FP8 E4M3 activations",
            "norm, router, embedding, LM head, MTP, mHC, vision and conv stay BF16/FP32",
        ),
    )


def _quant_mxfp8_mxfp4() -> QuantSpec:
    """QuaRot / msmodelslim Ascend A5 production mixed MXFP8 linear + MXFP4 expert recipe."""
    return QuantSpec(
        name="MXFP8_MXFP4",
        weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        expert_weight_format="mxfp4",
        weight_block=(1, MX_EXPERT_BLOCK),
        expert_block=MX_EXPERT_BLOCK,
        scale_format="fp8_e8m0", scale_bytes=1.0,
        activation_group_size=MX_EXPERT_BLOCK,
        activation_scale_format="fp8_e8m0",
        activation_scale_pack=MX_EXPERT_SCALE_PACK,
        kv_dtype="fp8_e4m3", index_dtype="fp8_e4m3", index_scale_dtype="fp8_e8m0",
        category_overrides={
            "routed_expert": _MXFP4_ROUTED_ACTIVATION,
            "dspark_routed_expert": _MXFP4_ROUTED_ACTIVATION,
            "latent_moe": _MXFP4_LINEAR_OVERRIDE,
            "attn_res_proj": _MXFP4_LINEAR_OVERRIDE,
        },
        exclusions=_STANDARD_MXFP4_EXCLUSIONS,
        source=merge_source({
            "quant_method": "quarot_msmodelslim_mxfp8_mxfp4",
            "weight_block_size": "1x32",
            "scale_fmt": "ue8m0",
            "kv_cache_dtype": "fp8_e4m3",
        }),
        notes=(
            "Ascend A5 production quantization (QuaRot/msmodelslim): MXFP8 group32 on "
            "MLA/GQA/KDA/GDN/shared/dense linears + MXFP4 group32 on routed experts, "
            "latent MoE down/up, and AttnRes projections",
            "norm, router, embedding, LM head, MTP, mHC, vision and conv stay BF16/FP32",
        ),
    )


def resolve_auto_va_quant(
    model_spec: Any,
    hardware_sku: str,
    *,
    context_tokens: int = 0,
) -> tuple[str, str, str]:
    """Resolve chip-matched ``vllm-ascend`` production ``(weight_quant, kv_dtype, indexer_dtype)``."""
    from mindie_sim.catalog.npu_checkpoints import normalize_hw_family

    del context_tokens  # Production target KV quantization stays active across prompt lengths.
    hw = normalize_hw_family(str(hardware_sku or ""))
    plans = tuple(getattr(model_spec, "layer_plans", ()) or ())
    is_moe = (
        int(getattr(model_spec, "num_experts", 0) or 0) > 1
        or any(str(getattr(lp, "ffn", "")).lower() == "moe" for lp in plans)
    )
    has_kpool = any(int(getattr(lp, "index_kpool", 1) or 1) > 1 for lp in plans)
    if hw == "a5":
        return ("mxfp8_mxfp4" if is_moe else "mxfp8", "fp8_e4m3", "fp8_e4m3")
    if hw in {"a2", "a3"}:
        # K-pool index caches reject INT8 storage; keep BF16 indexer KV for K-pool on A2/A3.
        idx_dtype = "bf16" if has_kpool else "int8"
        return ("w4a8" if is_moe else "w8a8", "int8", idx_dtype)
    native = str(getattr(model_spec, "activation_dtype", None) or "bf16")
    return (native, native, native)


#: Explicit quantization presets a case may declare on a catalog model.
QUANTIZATION_PRESETS: dict[str, Any] = {
    # Short names are explicit representation declarations, not dispatch claims.
    # The FP8 short name intentionally selects the retained MXFP8 checkpoint
    # policy, including its GDN/MTP exclusions; it is not every FP8 format.
    "fp8": _quant_qwen_mxfp8,
    "w8a8_mxfp8": _quant_qwen_mxfp8,
    "w8a8-mxfp8": _quant_qwen_mxfp8,
    "mxfp8": _quant_qwen_mxfp8,
    "fp8_mxfp8": _quant_qwen_mxfp8,
    "mxfp8_mxfp4": _quant_mxfp8_mxfp4,
    "mxfp8+mxfp4": _quant_mxfp8_mxfp4,
    "quarot_mxfp8_mxfp4": _quant_mxfp8_mxfp4,
    "w4a8_mxfp": _quant_w4a8_mxfp,
    "w4a8-mxfp": _quant_w4a8_mxfp,
    "w4a8_mxfp4": _quant_w4a8_mxfp,
    "w4a8-mxfp4": _quant_w4a8_mxfp,
    "mxfp4_w4a8": _quant_w4a8_mxfp,
    "mxfp4": _quant_mxfp4_all,
    "mxfp4_all": lambda: _quant_mxfp4_all(exclusions=()),
    "mxfp4_routed_only": lambda: _quant_kimi({}),
    "w8a8_dynamic": _quant_w8a8_dynamic,
    "w8a8": _quant_w8a8_dynamic,
    "w8a8_int8": _quant_w8a8_dynamic,
    "int8": _quant_w8a8_dynamic,
    "w4a8_dynamic": _quant_w4a8_dynamic,
    "w4a8": _quant_w4a8_dynamic,
    "w4a8-int8": _quant_w4a8_dynamic,
    "bf16": lambda: _quant_bf16({}, name="bf16"),
    "fp16": _quant_fp16,
}


def quant_preset(name: str, *, model=None) -> QuantSpec:
    """Resolve a case-declared quantization preset; unknown names are refused."""
    key = str(name or "").strip().lower()
    builder = QUANTIZATION_PRESETS.get(key)
    if builder is None:
        raise ValueError(f"quantization_preset_unknown:{name}")
    quant = builder()
    if (builder is _quant_qwen_mxfp8 and model is not None
            and dict(model.metadata).get("source_model_type") == "glm5_next_text"):
        native = {"weight_format": "bf16", "activation_format": "bf16"}
        quant = replace(
            quant,
            exclusions=("norm", "lm_head", "embedding", "vision", "projector",
                        "router", "conv_weight_f32", "conv_weight_bf16"),
            category_overrides={
                "kda": native, "mla_absorb": native, "indexer": native,
                "indexer_fp8": native, "dspark_input": native,
                "mla": {"weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3"},
            },
            source=merge_source({
                "quant_method": "mxfp8",
                "checkpoint": "Eco-Tech/GLM-5.3-Flash-w8a8-mxfp8",
                "weight_block_size": "1x32", "scale_fmt": "ue8m0",
            }),
            notes=("MLA projections and body/MTP experts use MXFP8; "
                   "KDA, absorbed MLA, indexer and MTP input stay FLOAT",))
    return quant


def _quant_bf16(c: Mapping[str, Any], *, name: str) -> QuantSpec:
    return QuantSpec(
        name=name, weight_format="bf16", activation_format="bf16",
        expert_weight_format="bf16", kv_dtype="bf16",
        index_dtype="bf16", index_scale_dtype="none",
        exclusions=(), source=merge_source({"quantization": "none"}),
        notes=("no weight quantization declared; compute and activation are BF16",),
    )
