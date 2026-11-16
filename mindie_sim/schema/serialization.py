"""Lossless structural input construction.

[Role] Reconstruct normalized models and apply explicit catalog modifications.
[Invariants] Catalogs are defaults, never immutable execution identities.
[Timeouts & Locks] Metadata-only construction, no I/O.
[Side-Effect Policy] Creates new records.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, replace

from .common import ModelingError, _int
from .model import LayerPlan, MlaSpec, ModelSpec
from mindie_sim.catalog.types import (
    AuxiliarySpec, DraftSpec, EngramSpec, EngramTableSpec, QuantSpec, VisionSpec,
)


_SCALARS = (int, str, bool, float, type(None))


def immutable_sequences(value):
    if type(value) in _SCALARS:
        return value
    if isinstance(value, (list, tuple)):
        return tuple(immutable_sequences(item) for item in value)
    if isinstance(value, dict):
        return {key: immutable_sequences(item) for key, item in value.items()}
    return value


def _declared_pairs(value):
    """Keep declared metadata values and order, including JSON list shapes.

    Metadata is documentation/provenance, not model geometry.  Converting an
    arbitrary nested list to a tuple or reordering mapping keys here changes
    the exported representation (and breaks an exact ``to_dict``/``from_mapping``
    round trip) without making the geometry any more immutable.
    """
    if value is None:
        return ()
    is_mapping = isinstance(value, Mapping)
    items = value.items() if is_mapping else value
    return tuple((key, deepcopy(item)) for key, item in items)


def _resolve_layer_count(plans, declared):
    """Preserve a declared layer count and reject plan/count disagreement.

    An empty plan set is an unexpanded specification; it is legal only with a
    positive declared count, which the existing normalizer later expands.  A
    non-empty plan set may derive the count only when none was declared.
    """
    if declared is None:
        if not plans:
            raise ModelingError("model.layers_required")
        return len(plans)
    declared = _int(declared, "model.layers_invalid", minimum=1)
    if plans and len(plans) != declared:
        raise ModelingError(
            "layer_plan_count_differs_from_layers:"
            f"{len(plans)}!={declared}")
    return declared


def engram(raw):
    data = dict(raw)
    data["tables"] = tuple(EngramTableSpec(**value) for value in data.get("tables", ()))
    return EngramSpec(**data)


def auxiliary(raw):
    data = dict(raw)
    derived = ("input_width", "proposal_tokens")
    for name in ("mtp", "dspark"):
        if data.get(name) is not None:
            draft = {key: value for key, value in dict(data[name]).items()
                     if key not in derived}
            plans = tuple(
                LayerPlan(**(dict(plan) if isinstance(plan, Mapping) else plan))
                for plan in draft.get("layer_plans", ()))
            draft["layer_plans"] = plans
            data[name] = DraftSpec(**draft)
    return AuxiliarySpec(**data)


def explicit_model(raw):
    # Capture declared metadata before normalizing geometry sequences.  The
    # nested values remain exactly as exported (including list-valued source
    # contracts), so a model read from a real case round-trips without
    # changing its reported provenance.
    metadata = _declared_pairs(raw.get("metadata"))
    catalog_facts = _declared_pairs(raw.get("catalog_facts"))
    # Metadata may be a large source-config snapshot; normalizing it to tuples
    # only to replace it below would copy the whole provenance tree twice.
    data = immutable_sequences({
        key: value for key, value in raw.items()
        if key not in {"metadata", "catalog_facts", "layer_plans"}
    })
    raw_plans = raw["layer_plans"]
    data["layer_plans"] = tuple(
        LayerPlan(**(value if type(value) is dict and all(type(v) in _SCALARS for v in value.values())
                     else immutable_sequences(value)))
        for value in raw_plans
    )
    data["layers"] = _resolve_layer_count(data["layer_plans"], raw.get("layers"))
    data["metadata"] = metadata
    data["catalog_facts"] = catalog_facts
    for name, factory in (("quant", QuantSpec), ("mla", MlaSpec),
                          ("vision", VisionSpec), ("engram", engram),
                          ("auxiliary", auxiliary)):
        if isinstance(data.get(name), dict):
            if name in {"auxiliary", "engram"}:
                data[name] = factory(data[name])
            elif name == "mla":
                data[name] = factory.from_mapping(data[name])
            else:
                data[name] = factory(**data[name])
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
        scalar["layers"] = _resolve_layer_count(plans, raw.get("layers"))
    elif "layers" in scalar:
        count = _int(scalar["layers"], "model.layers_invalid", minimum=1)
        if count > len(plans):
            raise ModelingError("extended_catalog_requires_explicit_layer_plans")
        plans = plans[:count]
        scalar["layers"] = count
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
