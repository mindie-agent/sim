"""Declared storage, native stream and per-role matrix value formats for HF configs.

[Role]
  Normalize explicit storage/representation spellings, bind native precision
  streams without overriding explicit quantization plans, and project resolved
  ``ModelSpec`` layer schedules into per-layer/per-role value format records.

[Invariants]
  * Never invents a storage format; only maps known spelling aliases and
    returns empty string on unknown formats so callers retain explicit gaps.
  * Quantized plans keep weight/expert/scale fields untouched when binding
    the native activation stream.
  * Per-layer role formats use the exact ``projection_quant -> for_category ->
    matrix_output_format`` chain shared with execution lowering.

[Timeouts & Locks]
  Pure bounded config/layer traversal; no I/O, locks or device context.

[Side-Effect Policy]
  Returns new mappings and immutable ``QuantSpec`` copies; never mutates inputs.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .config import flatten_config

_STORAGE_FORMATS = frozenset({
    "bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2", "int8",
    "mxfp4", "fp4", "int4", "w4a8",
})
_HIGH_PRECISION_FORMATS = frozenset({"bf16", "fp16", "fp32"})
_HIGH_PRECISION_ALIASES = frozenset({
    "bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "",
})
_MATRIX_VALUE_ROLES = (
    "attn.q", "attn.kv", "attn.o", "mla", "mla_absorb", "dense_mlp",
    "routed_expert", "shared_expert", "router", "indexer", "mtp",
)
_ATTENTION_MATRIX_ROLES = {
    "gqa": ("attn.qkv", "attn.o"),
    "mha": ("attn.qkv", "attn.o"),
    "mqa": ("attn.qkv", "attn.o"),
    "projection_only": ("attn.qkv", "attn.o"),
    "gdn": ("gdn.qkvz", "gdn.o", "gdn.beta_alpha"),
    "kda": ("kda.q", "kda.k", "kda.v", "kda.b", "kda.f_a", "kda.f_b",
            "kda.g", "kda.g_a", "kda.g_b", "kda.o"),
}


def native_dtype(c: Mapping[str, Any]) -> str:
    """Resolve declared native floating-point dtype spelling from flattened config."""
    name = str(c.get("activation_dtype", c.get("torch_dtype", c.get("dtype", "bf16"))))
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(name, name)


def normalize_declared_format(value: Any, *, expert: bool = False) -> str:
    """Canonicalise one explicitly declared representation, or return empty.

    This helper never invents a format.  It only maps spelling aliases that
    the source configs already use and rejects everything else so the caller
    can retain an explicit coverage gap.
    """
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_")
    text = {
        "bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
        "float8_e4m3": "fp8_e4m3", "float8_e4m3fn": "fp8_e4m3",
        "float8_e5m2": "fp8_e5m2", "fp8": "fp8_e4m3",
    }.get(text, text)
    if expert:
        text = {"fp4": "mxfp4", "fp8": "fp8_e4m3"}.get(text, text)
    return text if text in _STORAGE_FORMATS else ""


def declared_weight_dtype(config: Mapping[str, Any]) -> str:
    c, _, _ = flatten_config(config)
    return normalize_declared_format(c.get("weight_dtype"))


def resolve_native_formats(quant: Any, config: Mapping[str, Any]) -> Any:
    """Bind declared native stream and independent storage formats.

    The activation stream may inherit a declared FP16/FP32/FP8 native format
    only for the unquantized default representation.  Explicit KV, weight,
    expert, index and scale declarations stay independent: equality with the
    old native spelling is never used as proof of inheritance.  A quantized
    plan keeps every weight/expert/scale field; only its native stream label
    and an explicitly declared KV format are updated.
    """
    c, _, _ = flatten_config(config)
    native = normalize_declared_format(native_dtype(c))
    if quant is None:
        return quant
    raw_quant = c.get("quantization_config")
    quantized = isinstance(raw_quant, Mapping) and bool(raw_quant)
    explicit_weight = normalize_declared_format(c.get("weight_dtype"))
    explicit_expert = normalize_declared_format(c.get("expert_dtype"), expert=True)
    explicit_kv = normalize_declared_format(c.get("kv_dtype"))
    kv_from_quant = (isinstance(raw_quant, Mapping)
                     and raw_quant.get("kv_cache_scheme") is not None)
    changes: dict[str, Any] = {}
    if native and quant.native_format != native:
        changes["native_format"] = native
    if explicit_kv and not kv_from_quant and explicit_kv != quant.kv_dtype:
        changes["kv_dtype"] = explicit_kv
    if not quantized:
        if explicit_weight in _HIGH_PRECISION_FORMATS:
            changes["weight_format"] = explicit_weight
            changes["expert_weight_format"] = explicit_weight
        if explicit_expert in _HIGH_PRECISION_FORMATS:
            changes["expert_weight_format"] = explicit_expert
        if native:
            if "weight_format" not in changes and not explicit_weight:
                changes["weight_format"] = native
            if "activation_format" not in changes:
                changes["activation_format"] = native
            if ("expert_weight_format" not in changes
                    and not explicit_weight and not explicit_expert):
                changes["expert_weight_format"] = native
            if "kv_dtype" not in changes and not explicit_kv:
                changes["kv_dtype"] = native
    return replace(quant, **changes) if changes else quant


def _role_value_formats(quant: Any, role: str, *, native: str,
                        compute_override: str = "",
                        output_override: str = "") -> dict[str, Any]:
    """Resolve one matrix role through its effective category QuantSpec."""
    from .ops import compute_dtype_for, matrix_output_format

    resolved = quant.for_category(role)
    compute, activation = compute_dtype_for(quant, role)
    weight_storage = resolved.source_format_for(role)
    weight_runtime = resolved.format_for(role)
    output = matrix_output_format(
        quant, role, activation_dtype=activation, prequant=native)
    if compute_override:
        compute = compute_override
    if output_override:
        output = output_override
    return {
        "weight_storage": weight_storage,
        "weight_runtime": weight_runtime,
        "activation": activation,
        "compute": compute,
        "output": output,
        "weight_scale": ("" if weight_runtime in _HIGH_PRECISION_ALIASES
                         else resolved.execution_weight_scale_format()),
        "activation_scale": ("" if activation in _HIGH_PRECISION_ALIASES
                             else resolved.execution_activation_scale_format()),
    }


def _plan_matrix_roles(plan: Any) -> set[str]:
    """Actual matrix categories reachable from one parsed LayerPlan."""
    roles: set[str] = set()
    attention = str(getattr(plan, "attention", "") or "")
    if attention == "mla":
        roles.add("mla")
        if int(getattr(plan, "q_lora_rank", 0) or 0):
            roles.update(("mla.q_a", "mla.q_b"))
        else:
            roles.add("mla.q_proj")
        roles.add("mla.kv_a")
        roles.add("mla_absorb")
        if int(getattr(plan, "o_groups", 0) or 0):
            roles.update(("mla.o_a", "mla.o_b"))
        else:
            roles.add("mla.o")
        if bool(getattr(plan, "mla_output_gate", False)):
            roles.add("mla.g")
    else:
        roles.update(_ATTENTION_MATRIX_ROLES.get(attention, ()))
    ffn = str(getattr(plan, "ffn", "") or "")
    if ffn == "moe":
        roles.update(("router", "routed_expert"))
        if int(getattr(plan, "num_shared_experts", 0) or 0):
            roles.add("shared_expert")
        if int(getattr(plan, "expert_latent_size", 0) or 0):
            roles.add("latent_moe")
    elif ffn == "dense":
        roles.add("dense_mlp")
    if (bool(getattr(plan, "indexer", False))
            or bool(getattr(plan, "owns_topk", False))
            or bool(getattr(plan, "index_query_per_layer", False))):
        roles.update(("indexer.weights", "indexer_fp8"))
        if int(getattr(plan, "index_kpool", 0) or 0) > 1:
            roles.update(("indexer.pool_projection", "indexer.pool_gate"))
    if str(getattr(plan, "scope", "") or "") == "mtp":
        roles.add("mtp")
    return roles


def model_value_formats(model: Any) -> dict[str, Any]:
    """Project one resolved ModelSpec into per-layer/role value formats.

    Every layer uses the same pure chain as execution:
    ``projection_quant(model.quant, plan)`` -> ``for_category(category)`` ->
    the shared ``matrix_output_format`` rule.  A role summary is shared only
    when every applicable layer resolves the same complete record; otherwise
    the caller receives the per-layer binding.
    """
    from .quant import projection_quant
    from .types import QuantSpec

    native = str(getattr(model, "activation_dtype", "") or "bf16")
    quant = getattr(model, "quant", None)
    compute_override = ""
    if quant is None:
        weight = (normalize_declared_format(getattr(model, "weight_dtype", ""))
                  or native)
        kv = (normalize_declared_format(getattr(model, "kv_dtype", ""))
              or native)
        activation = str(getattr(model, "activation_dtype", "") or native)
        compute_override = str(getattr(model, "compute_dtype", "") or activation)
        quant = QuantSpec(
            name="declared", weight_format=weight, activation_format=activation,
            expert_weight_format=weight, kv_dtype=kv, index_dtype=native,
            native_format=activation,
        )
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    metadata = dict(getattr(model, "metadata", ()) or ())
    roles = {
        role: _role_value_formats(quant, role, native=native,
                                  compute_override=compute_override)
        for role in _MATRIX_VALUE_ROLES
    }
    layer_records: dict[str, dict[int, dict[str, Any]]] = {}
    for plan in plans:
        layer_quant = projection_quant(quant, plan)
        layer_index = int(getattr(plan, "index", 0))
        for role in _plan_matrix_roles(plan):
            record = _role_value_formats(
                layer_quant, role, native=native,
                compute_override=compute_override)
            if role == "router":
                router_dtype = str(getattr(plan, "router_dtype", "") or "")
                router_output = str(getattr(plan, "router_output_dtype", "") or "")
                if router_dtype:
                    record.update(weight_storage=router_dtype,
                                  weight_runtime=router_dtype,
                                  activation=router_dtype, compute=router_dtype,
                                  weight_scale="", activation_scale="")
                if router_output:
                    record["output"] = router_output
                elif router_dtype:
                    record["output"] = router_dtype
            layer_records.setdefault(role, {})[layer_index] = record
    for role, records in layer_records.items():
        if not records:
            continue
        variants = {tuple(sorted(record.items())) for record in records.values()}
        if len(variants) == 1:
            summary = dict(next(iter(records.values())))
            summary["layers"] = tuple(sorted(records))
            roles[role] = summary
        else:
            roles[role] = {
                "heterogeneous": True,
                "by_layer": {layer: dict(records[layer])
                             for layer in sorted(records)},
            }
    rope = metadata.get("rope_parameters")
    if not isinstance(rope, Mapping):
        unconsumed = metadata.get("unconsumed_config_fields")
        rope = (unconsumed.get("rope_parameters")
                if isinstance(unconsumed, Mapping) else None)
    rope = dict(rope) if isinstance(rope, Mapping) else {}
    for key in ("rope_theta", "partial_rotary_factor", "mrope_section",
                "mrope_interleaved", "mla_use_nope"):
        if key not in rope and key in metadata:
            rope[key] = metadata[key]
    partial = rope.get("partial_rotary_factor", 1.0)
    try:
        partial = float(partial)
    except (TypeError, ValueError):
        partial = 1.0
    mla = getattr(model, "mla", None)
    model_head_dim = int(getattr(model, "head_dim", 0) or 0)
    rotary: dict[str, set[int]] = {}
    for plan in plans:
        attention = str(getattr(plan, "attention", "") or "gqa")
        head_dim = int(getattr(plan, "head_dim", 0) or model_head_dim)
        rope_dim = int(getattr(plan, "qk_rope_head_dim", 0) or 0)
        nope_dim = int(getattr(plan, "qk_nope_head_dim", 0) or 0)
        if attention == "mla":
            dim = rope_dim if (rope_dim > 0 or nope_dim > 0) else (
                int(mla.qk_rope_head_dim) if mla is not None else 0
            )
        elif rope_dim > 0 or nope_dim > 0:
            dim = rope_dim
        elif attention in {"gdn", "kda", "mamba"}:
            dim = 0
        elif 0.0 < partial < 1.0:
            dim = int(round(head_dim * partial))
        else:
            dim = head_dim
        rotary.setdefault(attention, set()).add(dim)
    return {
        "activation_stream": native,
        "compute": str(getattr(model, "compute_dtype", "") or native),
        "output_stream": native,
        "kv_cache": str(quant.kv_dtype),
        "index": str(quant.index_dtype),
        "index_scale": str(quant.index_scale_dtype),
        "weight_scale": quant.execution_weight_scale_format(),
        "activation_scale": quant.execution_activation_scale_format(),
        "matrix_roles": roles,
        "heads": {
            "num_attention_heads": int(getattr(model, "num_heads", 0) or 0),
            "num_key_value_heads": int(getattr(model, "num_kv_heads", 0) or 0),
            "head_dim": int(getattr(model, "head_dim", 0) or 0),
            "mla": (mla.to_dict() if mla is not None else None),
        },
        "rotary": {
            "rope_parameters": rope,
            "partial_rotary_factor": partial,
            "rotary_dim_by_attention": {
                key: tuple(sorted(values)) for key, values in sorted(rotary.items())
            },
            "max_position_embeddings": int(metadata.get("max_position_embeddings", 0) or 0),
        },
    }


__all__ = [
    "declared_weight_dtype",
    "model_value_formats",
    "native_dtype",
    "normalize_declared_format",
    "resolve_native_formats",
]
