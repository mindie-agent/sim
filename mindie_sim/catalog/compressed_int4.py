"""Compressed INT4 checkpoint representation.

[Role] Resolve groupwise integer packing and standard role exclusions from a
  ``compressed-tensors`` ``pack-quantized`` checkpoint descriptor.
[Invariants] Weight packing never implies an activation quantizer: only an
  explicit integer token-activation group makes the compute identity W4A8;
  otherwise the packed weights stream through native BF16/FP16 compute.
[Timeouts & Locks] Bounded config metadata only.
[Side-Effect Policy] Pure immutable records; no weights or device access.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError, dtype_bytes

INT4_BITS = 4
INT8_BITS = 8
NATIVE_ROLES = ("norm", "router", "embedding", "lm_head")
ROLE_CATEGORIES = {
    "attention": ("attn", "mla", "mla_absorb", "indexer", "indexer_fp8"),
    "shared_expert": ("shared_expert",),
    "routed_expert": ("routed_expert",),
}
EXACT_EXCLUSIONS = {
    r"re:.*mlp\.(gate|up|gate_up|down)_proj.*": ("dense_mlp",),
    "re:vision_tower.*": ("vision", "vision_pos", "image_token"),
    "re:mm_projector.*": ("projector",),
}
FORMATS = {"torch.bfloat16": "bf16", "bfloat16": "bf16",
           "torch.float16": "fp16", "float16": "fp16",
           "torch.float32": "fp32", "float32": "fp32"}


def resolve_compressed_int4(raw, base, *, has_mla: bool = False, has_kda: bool = False):
    """Return a W4A16/W4A8 plan, or ``None`` when the descriptor is not exact."""
    from .hf_config import resolve_indexed_layer_exclusions, resolve_role_exclusions
    if raw.get("format") != "pack-quantized":
        return None
    groups = tuple((raw.get("config_groups") or {}).values())
    if not groups:
        return None
    first = groups[0]
    first_weights = first.get("weights") or {}
    for group in groups:
        weights = group.get("weights") or {}
        if (group.get("targets") != ["Linear"]
                or group.get("format", raw["format"]) != "pack-quantized"
                or weights.get("num_bits") != INT4_BITS
                or weights.get("type") != "int"
                or weights.get("strategy") != "group"
                or int(weights.get("group_size") or 0) <= 0
                or weights.get("dynamic", False)
                or not weights.get("symmetric", True)
                or group.get("output_activations") is not None
                or weights != first_weights
                or group.get("input_activations") != first.get("input_activations")):
            return None
    weights = first_weights
    activation = first.get("input_activations")
    if activation is not None and not (
            activation.get("num_bits") == INT8_BITS
            and activation.get("type") == "int"
            and activation.get("symmetric", True)
            and activation.get("dynamic") is True
            and activation.get("strategy") == "token"):
        return None
    layer_overrides, remaining_ignores = resolve_indexed_layer_exclusions(
        raw.get("ignore") or (),
        native=base.native_format,
        has_mla=has_mla,
        has_kda=has_kda,
    )
    categories = list(NATIVE_ROLES)
    for pattern in remaining_ignores:
        if pattern in EXACT_EXCLUSIONS:
            categories.extend(EXACT_EXCLUSIONS[pattern])
            continue
        roles, unresolved = resolve_role_exclusions((pattern,))
        if unresolved:
            return None
        for role in roles:
            categories.extend(ROLE_CATEGORIES.get(role, (role,)))
    if raw.get("kv_cache_scheme") is not None:
        return None
    scale = weights.get("scale_dtype") or base.native_format
    scale = FORMATS.get(scale, scale)
    try:
        scale_bytes = dtype_bytes(scale)
    except ModelingError:
        return None
    native = {"weight_format": base.native_format,
              "activation_format": base.native_format}
    return replace(
        base,
        name="compressed_tensors_w4a8" if activation else "compressed_tensors_w4a16",
        weight_format="int4", expert_weight_format="int4",
        activation_format="int8" if activation else base.native_format,
        weight_block=(1, int(weights["group_size"])),
        expert_block=int(weights["group_size"]),
        scale_format=scale, scale_bytes=scale_bytes,
        activation_scale_format="fp32" if activation else "",
        exclusions=tuple(dict.fromkeys(categories)),
        category_overrides={role: native for role in dict.fromkeys(categories)},
        layer_quant_overrides=layer_overrides,
    )
