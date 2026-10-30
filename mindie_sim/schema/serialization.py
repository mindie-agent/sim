"""Lossless structural input construction.

[Role] Reconstruct normalized models and apply explicit catalog modifications.
[Invariants] Catalogs are defaults, never immutable execution identities.
[Timeouts & Locks] Metadata-only construction, no I/O.
[Side-Effect Policy] Creates new records.
"""
from dataclasses import fields, replace

from .common import ModelingError
from .model import LayerPlan, MlaSpec, ModelSpec
from mindie_sim.catalog.types import (
    AuxiliarySpec, DraftSpec, EngramSpec, EngramTableSpec, QuantSpec, VisionSpec,
)


def immutable_sequences(value):
    if isinstance(value, (list, tuple)):
        return tuple(immutable_sequences(item) for item in value)
    if isinstance(value, dict):
        return {key: immutable_sequences(item) for key, item in value.items()}
    return value


def engram(raw):
    data = dict(raw)
    data["tables"] = tuple(EngramTableSpec(**value) for value in data.get("tables", ()))
    return EngramSpec(**data)


def auxiliary(raw):
    data = dict(raw)
    for name in ("mtp", "dspark"):
        if data.get(name) is not None:
            data[name] = DraftSpec(**data[name])
    return AuxiliarySpec(**data)


def explicit_model(raw):
    data = immutable_sequences(raw)
    data["layer_plans"] = tuple(LayerPlan(**value) for value in data["layer_plans"])
    for key in ("metadata", "catalog_facts"):
        if isinstance(data.get(key), dict):
            data[key] = tuple(sorted(data[key].items()))
    for name, factory in (("quant", QuantSpec), ("mla", MlaSpec),
                          ("vision", VisionSpec), ("engram", engram),
                          ("auxiliary", auxiliary)):
        if isinstance(data.get(name), dict):
            data[name] = (factory(data[name]) if name in {"auxiliary", "engram"}
                          else factory(**data[name]))
    data["layers"] = len(data["layer_plans"])
    return ModelSpec(**data)


def override_catalog(model, raw):
    names = {field.name for field in fields(ModelSpec)}
    scalar = {key: value for key, value in raw.items() if key in names and key not in {
        "layer_plans", "quantization", "metadata", "mla", "auxiliary",
    }}
    unknown = raw.keys() - names - {"catalog_key", "model_catalog_key", "preset", "layer_overrides"}
    if unknown:
        raise ModelingError("catalog_override_requires_layer_plans:" + ",".join(sorted(unknown)))
    if "mla" in raw:
        raise ModelingError("catalog_mla_override_requires_layer_overrides")
    if "quant" in scalar:
        scalar["quant"] = QuantSpec(**immutable_sequences(scalar["quant"]))
    elif any(key in scalar for key in ("weight_dtype", "kv_dtype")):
        quant_changes = {}
        if "kv_dtype" in scalar:
            quant_changes["kv_dtype"] = scalar["kv_dtype"]
        if "weight_dtype" in scalar:
            quant_changes.update(weight_format=scalar["weight_dtype"], expert_weight_format=scalar["weight_dtype"])
        scalar["quant"] = replace(model.quant, **quant_changes)
    plans = model.layer_plans
    if raw.get("layer_plans") is not None:
        plans = tuple(LayerPlan(**value) for value in raw["layer_plans"])
        scalar["layers"] = len(plans)
    elif "layers" in scalar:
        count = int(scalar["layers"])
        if count > len(plans):
            raise ModelingError("extended_catalog_requires_explicit_layer_plans")
        plans = plans[:count]
    layer_names = {field.name for field in fields(LayerPlan)}
    changes = {key: value for key, value in scalar.items() if key in layer_names}
    overrides = raw.get("layer_overrides", {})
    plans = tuple(replace(plan, **{
        **{key: value for key, value in changes.items()
           if key not in {"index", "native_layer_id"} and (
               getattr(plan, key) or key in {"num_shared_experts", "ffn", "attention"})},
        **overrides.get(str(plan.index), {}),
    }) for plan in plans)
    if "auxiliary" in raw:
        scalar["auxiliary"] = auxiliary(immutable_sequences(raw["auxiliary"])) if raw["auxiliary"] else None
    return replace(model, **scalar, layer_plans=plans)


def apply_model_overrides(model, raw):
    """Use the same explicit representation/structure boundary for HF and catalog."""
    from .common import _metadata
    declared = raw.get("quantization")
    if declared is not None and str(declared).strip():
        # A case may declare the measured runtime representation of a
        # catalog model (the W8A8-MXFP8 checkpoint served on A5) without
        # forking the architecture.  The structure, dimensions and layer
        # plan stay the catalog's; only the weight/activation plan moves.
        from mindie_sim.catalog.quant import quant_preset, _quant_bf16

        try:
            quant = (_quant_bf16({}, name="bf16") if str(declared).lower() in {"none", "bf16", "bf16_qwen"}
                     else quant_preset(str(declared)))
        except ValueError as exc:
            raise ModelingError(str(exc)) from exc
        # A weight preset does not erase the catalog model's declared
        # quantized indexer.  A model whose catalog plan declares INT8
        # would otherwise be reset to BF16 by a generic weight preset,
        # silently doubling the index-cache row width.  Preserve only a
        # non-BF16 indexer the preset did not itself specify.
        if (
            model.quant is not None
            and model.quant.index_dtype not in {"", "bf16"}
            and quant.index_dtype == "bf16"
        ):
            quant = replace(
                quant,
                index_dtype=model.quant.index_dtype,
                index_scale_dtype=model.quant.index_scale_dtype,
            )
        model = replace(
            model, quantization=str(declared), quant=quant,
            weight_dtype=quant.weight_format,
            # Matrix constructors derive their own precision. A
            # quantized projection does not quantize attention math.
        )
    extra_metadata = _metadata(raw)
    if extra_metadata:
        model = replace(
            model,
            metadata=tuple(sorted(
                {**dict(model.metadata), **dict(extra_metadata)}.items()
            )),
        )
    result = override_catalog(model, raw)
    if "quant" in raw:
        result = replace(result, quantization=result.quant.name,
                         weight_dtype=result.quant.weight_format)
    return result
