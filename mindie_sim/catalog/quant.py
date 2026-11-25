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
from mindie_sim.schema.common import ModelingError

BF16_SCALE_BYTES = 2.0
FP32_SCALE_BYTES = 4.0


_LAYER_QUANT_OVERRIDE_FIELDS = (
    "weight_format", "activation_format", "expert_weight_format",
    "kv_dtype", "index_dtype", "index_scale_dtype", "scale_format",
    "scale_bytes", "weight_block", "expert_block",
    "activation_group_size", "activation_scale_format",
    "activation_scale_pack", "output_format", "native_format",
)


def merge_layer_quant_override(base: QuantSpec, raw: Any) -> QuantSpec:
    """Apply one per-layer quant override (``QuantSpec`` or mapping) onto ``base``."""
    if isinstance(raw, QuantSpec):
        return raw
    if not isinstance(raw, Mapping):
        return base
    updates: dict[str, Any] = {}
    for field in _LAYER_QUANT_OVERRIDE_FIELDS:
        if field in raw and raw[field] is not None:
            updates[field] = raw[field]
    if "exclusions" in raw and raw["exclusions"] is not None:
        updates["exclusions"] = tuple(dict.fromkeys((*base.exclusions, *raw["exclusions"])))
    if "category_overrides" in raw and isinstance(raw["category_overrides"], Mapping):
        merged = {k: dict(v) for k, v in base.category_overrides}
        for cat, fields in raw["category_overrides"].items():
            cur = dict(merged.get(str(cat), {}))
            cur.update(dict(fields))
            merged[str(cat)] = cur
        updates["category_overrides"] = merged
    return replace(base, **updates) if updates else base


def lookup_layer_quant_override(overrides: Mapping[str, Any] | None, plan: Any) -> Any | None:
    """Resolve one per-layer override entry from ``layers`` / ``native_layers``."""
    if not isinstance(overrides, Mapping) or plan is None:
        return None
    raw_idx = getattr(plan, "index", None)
    if raw_idx is None:
        raw_idx = getattr(plan, "layer", None)
    layers_map = overrides.get("layers") if isinstance(overrides.get("layers"), Mapping) else overrides
    if raw_idx is not None:
        idx = int(raw_idx)
        for key in (str(idx), idx):
            if key in layers_map and layers_map[key] is not None:
                return layers_map[key]
    native_id = int(getattr(plan, "native_layer_id", -1))
    native_map = overrides.get("native_layers")
    if isinstance(native_map, Mapping) and native_id >= 0:
        for key in (str(native_id), native_id):
            if key in native_map and native_map[key] is not None:
                return native_map[key]
    return None


def apply_layer_quant_override(quant: QuantSpec, plan: Any) -> QuantSpec:
    """Merge any matching ``quant.layer_quant_overrides`` entry for ``plan``."""
    if quant is None or plan is None or not getattr(quant, "layer_quant_overrides", ()):
        return quant
    ov_dict = quant.layer_quant_overrides_dict()
    matched = lookup_layer_quant_override(ov_dict, plan)
    return merge_layer_quant_override(quant, matched) if matched is not None else quant


def projection_quant(quant: QuantSpec, plan) -> QuantSpec:
    """Resolve per-layer native projection defaults before explicit roles.

    A global quantization preset describes the default representation. Native
    source modules may opt out; user parent and child category declarations
    still win. This same record is used by execution and resident weights.
    """
    if quant is None:
        return quant
    if plan is not None and getattr(quant, "layer_quant_overrides", ()):
        quant = apply_layer_quant_override(quant, plan)
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
    "mtp", "norm", "router", "lm_head", "embedding", "mhc", "attn_res_proj",
    "compressor", "vision", "projector", "conv_weight_f32", "conv_weight_bf16",
)
_STANDARD_MXFP8_EXCLUSIONS: tuple[str, ...] = ("gdn", *_STANDARD_MXFP4_EXCLUSIONS)
_MXFP4_ROUTED_ACTIVATION = {
    "activation_format": "fp8_e4m3",
    "activation_group_size": MX_EXPERT_BLOCK,
    "activation_scale_format": "fp8_e8m0",
    "activation_scale_pack": MX_EXPERT_SCALE_PACK,
}
_MXFP8_LINEAR_OVERRIDE = {
    "weight_format": "fp8_e4m3",
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
        name="fp8_dynamic+mxfp4_experts",
        weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        expert_weight_format="mxfp4",
        weight_block=block, expert_block=MX_EXPERT_BLOCK,
        scale_format="fp8_e8m0" if scale_bytes == 1 else "fp32",
        scale_bytes=scale_bytes, kv_dtype="bf16",
        activation_group_size=block[1],
        activation_scale_format="fp8_e8m0" if scale_bytes == 1 else "fp32",
        activation_scale_pack=2 if scale_bytes == 1 else 1,
        index_dtype="int8", index_scale_dtype="fp16",
        category_overrides=(
            {} if v41 else {
                "routed_expert": dict(_MXFP4_ROUTED_ACTIVATION),
                "dspark_routed_expert": dict(_MXFP4_ROUTED_ACTIVATION),
            }
        ),
        # V4.1 executes the Engram gate as an MXFP8 (32, 32) projection; the
        # earlier V4 checkpoint keeps it in the BF16 exclusion set.
        exclusions=(("router", "lm_head", "embedding", "norm", "mhc", "attn_res_proj",
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
        weight_block=block, expert_block=MX_EXPERT_BLOCK,
        scale_format="fp32", scale_bytes=FP32_SCALE_BYTES,
        activation_group_size=block[1], activation_scale_format="fp32",
        kv_dtype="bf16", index_dtype="bf16", index_scale_dtype="none",
        exclusions=("router", "lm_head", "embedding", "norm", "mhc", "attn_res_proj",
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
        },
        exclusions=("attn", "mla", "kda", "shared_expert", "dense_mlp",
                    "latent_moe", "attn_res_proj",
                    "router", "lm_head", "embedding", "norm", "mhc",
                    "vision", "projector", "conv_weight_f32"),
        source=merge_source({"quant_method": q.get("quant_method", "compressed-tensors"),
                             "format": q.get("format", "mxfp4-pack-quantized")}),
        notes=("MXFP4 group32 applies only to routed expert matrices with FP8 E4M3 "
               "activations and E8M0 microscales; attention/shared/dense/latent_moe/"
               "attn_res_proj/router/LM-head/vision stay BF16",),
    )


def _quant_mxfp8() -> QuantSpec:
    """Model-agnostic W8A8-MXFP8 preset (FP8 E4M3 weights/activations, E8M0 group-32 scales)."""
    return QuantSpec(
        name="W8A8_MXFP8",
        weight_format="fp8_e4m3",
        activation_format="fp8_e4m3",
        expert_weight_format="fp8_e4m3",
        weight_block=(1, MX_EXPERT_BLOCK),
        expert_block=MX_EXPERT_BLOCK,
        scale_format="fp8_e8m0",
        scale_bytes=1.0,
        activation_group_size=MX_EXPERT_BLOCK,
        activation_scale_format="fp8_e8m0",
        activation_scale_pack=MX_EXPERT_SCALE_PACK,
        kv_dtype="bf16",
        index_dtype="bf16",
        index_scale_dtype="none",
        exclusions=_STANDARD_MXFP8_EXCLUSIONS,
        source=merge_source({
            "quant_method": "mxfp8",
            "weight_block_size": "1x32",
            "activation_scheme": "dynamic",
            "scale_fmt": "ue8m0",
        }),
        notes=(
            "Model-agnostic W8A8-MXFP8 preset (FP8 E4M3 weights/activations, E8M0 group-32 scales)",
            "Standard non-linear exclusions (norm, router, embedding, LM head, MTP, mHC, "
            "attn_res_proj, compressor, vision, projector, conv) stay native float",
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
        exclusions=("mtp", "norm", "router", "lm_head", "embedding", "mhc", "attn_res_proj",
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
        runtime_weight_scale_format="bf16",
        runtime_weight_scale_bytes=BF16_SCALE_BYTES,
        runtime_activation_scale_format="fp32",
        runtime_int8_weight_metadata="scale_only",
        category_overrides={
            "routed_expert": {
                "runtime_weight_scale_format": "fp32",
                "runtime_weight_scale_bytes": FP32_SCALE_BYTES,
            },
        },
        kv_dtype="bf16", index_dtype="bf16", index_scale_dtype="none",
        exclusions=("mtp", "norm", "router", "lm_head", "embedding", "mhc", "attn_res_proj",
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


def _quant_w4a4_mxfp4(
    *, exclusions: tuple[str, ...] = _STANDARD_MXFP4_EXCLUSIONS,
) -> QuantSpec:
    return QuantSpec(
        name="W4A4_MXFP4", weight_format="mxfp4", activation_format="fp4_e2m1",
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
            "quant_method": "w4a4_mxfp4",
            "weight_block_size": "1x32",
            "scale_fmt": "ue8m0",
        }),
        notes=(
            "MXFP4 group32 weights and FP4 E2M1 activations with E8M0 microscales",
            "MTP, norms, router, embedding, LM head, mHC, attn_res_proj, vision and conv stay BF16/FP32",
        ),
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
            "latent_moe": _MXFP8_LINEAR_OVERRIDE,
        },
        exclusions=_STANDARD_MXFP4_EXCLUSIONS,
        source=merge_source({
            "quant_method": "w4a8_mxfp",
            "weight_block_size": "1x32",
            "scale_fmt": "ue8m0",
        }),
        notes=(
            "MXFP8 group32 linears (including latent MoE) + MXFP4 group32 routed experts "
            "with FP8 E4M3 activations",
            "norm, router, embedding, LM head, MTP, mHC, attn_res_proj, vision and conv stay BF16/FP32",
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
            "latent_moe": _MXFP8_LINEAR_OVERRIDE,
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
            "MLA/GQA/KDA/GDN/shared/dense/latent_moe linears + MXFP4 group32 on routed experts",
            "norm, router, embedding, LM head, MTP, mHC, attn_res_proj, vision and conv stay BF16/FP32",
        ),
    )


def resolve_auto_va_quant(
    model_spec: Any,
    hardware_sku: str,
    *,
    context_tokens: int = 0,
) -> tuple[str, str, str]:
    """Resolve chip-matched structural ``(weight_quant, kv_dtype, indexer_dtype)``.

    Window-compressed latent-MQA models (e.g. CSA2 compressed-KV) never coerce
    target KV to INT8 on A2/A3; they retain their declared native float KV
    format. K-pool indexers also retain native float on A2/A3. Missing native
    dtypes fail closed rather than silently coercing to BF16.
    """
    from mindie_sim.catalog.npu_checkpoints import normalize_hw_family

    del context_tokens
    hw = normalize_hw_family(str(hardware_sku or ""))
    plans = tuple(getattr(model_spec, "layer_plans", ()) or ())
    raw_native = str(
        getattr(model_spec, "activation_dtype", None)
        or getattr(model_spec, "weight_dtype", None)
        or ""
    ).strip().lower()
    if not raw_native:
        raise ModelingError("auto_quant_native_dtype_missing")
    is_moe = (
        int(getattr(model_spec, "num_experts", 0) or 0) > 1
        or any(str(getattr(lp, "ffn", "")).lower() == "moe" for lp in plans)
    )
    has_kpool = any(int(getattr(lp, "index_kpool", 1) or 1) > 1 for lp in plans)
    is_window_compressed = any(
        str(getattr(lp, "cache_kind", "")).lower() == "window_compressed"
        or str(getattr(lp, "mla_variant", "")).lower() == "latent_mqa"
        for lp in plans
    )
    if hw == "a5":
        return ("mxfp8_mxfp4" if is_moe else "mxfp8", "fp8_e4m3", "fp8_e4m3")
    if hw in {"a2", "a3"}:
        # Window-compressed latent MQA (CSA2) does NOT support INT8 KV on A2/A3.
        kv_dtype = raw_native if is_window_compressed else "int8"
        idx_dtype = raw_native if (has_kpool or is_window_compressed) else "int8"
        return ("w4a8" if is_moe else "w8a8", kv_dtype, idx_dtype)
    return (raw_native, raw_native, raw_native)


#: Explicit quantization presets a case may declare on a catalog model.
QUANTIZATION_PRESETS: dict[str, Any] = {
    "fp8": _quant_mxfp8,
    "w8a8_mxfp8": _quant_mxfp8,
    "w8a8-mxfp8": _quant_mxfp8,
    "mxfp8": _quant_mxfp8,
    "fp8_mxfp8": _quant_mxfp8,
    "mxfp8_mxfp4": _quant_mxfp8_mxfp4,
    "mxfp8+mxfp4": _quant_mxfp8_mxfp4,
    "quarot_mxfp8_mxfp4": _quant_mxfp8_mxfp4,
    "w4a8_mxfp": _quant_w4a8_mxfp,
    "w4a8-mxfp": _quant_w4a8_mxfp,
    "w4a8_mxfp4": _quant_w4a8_mxfp,
    "w4a8-mxfp4": _quant_w4a8_mxfp,
    "mxfp4_w4a8": _quant_w4a8_mxfp,
    "mxfp4": _quant_mxfp4_all,
    "w4a4_mxfp4": _quant_w4a4_mxfp4,
    "w4a4-mxfp4": _quant_w4a4_mxfp4,
    "mxfp4_all": lambda: _quant_mxfp4_all(exclusions=()),
    "fp8_dynamic+mxfp4_experts": lambda: _quant_deepseek({}, v41=True),
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
    del model
    key = str(name or "").strip().lower()
    builder = QUANTIZATION_PRESETS.get(key)
    if builder is None:
        raise ValueError(f"quantization_preset_unknown:{name}")
    return builder()


def _quant_bf16(c: Mapping[str, Any], *, name: str) -> QuantSpec:
    return QuantSpec(
        name=name, weight_format="bf16", activation_format="bf16",
        expert_weight_format="bf16", kv_dtype="bf16",
        index_dtype="bf16", index_scale_dtype="none",
        exclusions=(), source=merge_source({"quantization": "none"}),
        notes=("no weight quantization declared; compute and activation are BF16",),
    )
