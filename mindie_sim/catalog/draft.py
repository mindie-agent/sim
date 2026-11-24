"""Select declared draft geometry.

[Role] Separate optional draft structure from target model structure.
[Invariants] Selection never depends on a model name or measured duration.
[Timeouts & Locks] Pure computation.
[Side-Effect Policy] No I/O.
"""
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.catalog.types import DraftSpec
from dataclasses import replace

ZERO_EXPERTS = 0
FLOAT_WEIGHT_FORMATS = frozenset({"bf16", "fp16", "fp32", "bfloat16", "float16", "float32"})


def _draft_scoped_overrides(quant, draft) -> dict[str, dict[str, object]]:
    """Extract only overrides explicitly scoped to this draft, never target roles."""
    prefix = f"{draft.weight_category}."
    kept: dict[str, dict[str, object]] = {}
    for key, raw in quant.category_overrides:
        fields = dict(raw)
        if key.startswith(prefix):
            kept[key] = fields
        elif (
            key == "dspark_routed_expert"
            and draft.kind in {"dspark", "mtp_dspark", "standalone_draft"}
            and str(fields.get("weight_format", "")).strip().lower() not in {"", *FLOAT_WEIGHT_FORMATS}
        ):
            kept[key] = fields
    return kept


def _draft_declared_native_dtype(model, draft, quant) -> str:
    raw = str(
        getattr(draft, "activation_dtype", "")
        or getattr(quant, "native_format", "")
        or getattr(model, "activation_dtype", "")
        or ""
    ).strip().lower()
    if raw not in FLOAT_WEIGHT_FORMATS:
        raise ModelingError(f"draft_native_dtype_missing_or_invalid:{raw!r}")
    return raw


def _draft_declared_kv_dtype(draft, *, fallback: str) -> str:
    src = dict(getattr(draft, "source", ()) or ())
    declared = str(src.get("kv_dtype") or getattr(draft, "kv_dtype", "") or "").strip().lower()
    return declared if declared else fallback


def draft_quant(model, draft):
    """Use the draft's declared representation in both work and residency.

    An explicitly all-FLOAT draft clears inherited target expert formats AND
    target role overrides so draft graphs and weights never inherit target
    quantization onto native float MTP/DSpark blocks. Preserves declared native
    float activation formats (``fp16`` / ``bf16`` / ``fp32``) and declared
    draft ``kv_dtype`` without silent BF16 coercion.
    """
    quant = model.quant
    if quant is None:
        raise ModelingError("draft_quant_requires_model_quant")
    from .paired_sfa import independent_draft
    is_independent = independent_draft(draft)
    if is_independent and not draft.weight_category:
        native = _draft_declared_native_dtype(model, draft, quant)
        kv_dtype = _draft_declared_kv_dtype(draft, fallback=native)
        return replace(
            quant,
            weight_format=native,
            expert_weight_format=native,
            activation_format=native,
            kv_dtype=kv_dtype,
            native_format=native,
            activation_group_size=0,
            activation_scale_format="",
            activation_scale_pack=1,
            scale_format="",
            scale_bytes=0.0,
            output_format="",
            category_overrides=(),
        )
    if not draft.weight_category:
        return quant
    fmt = str(quant.format_for(draft.weight_category)).strip().lower()
    if fmt in FLOAT_WEIGHT_FORMATS:
        draft_overrides = _draft_scoped_overrides(quant, draft)
        has_quantized_draft_override = any(
            str(fields.get("weight_format", "")).strip().lower() not in {"", *FLOAT_WEIGHT_FORMATS}
            for fields in draft_overrides.values()
        )
        if not has_quantized_draft_override:
            native_kv = _draft_declared_kv_dtype(
                draft,
                fallback=(
                    _draft_declared_native_dtype(model, draft, quant)
                    if is_independent and getattr(draft, "context_projection_mode", "") == "fused_kv_bank"
                    else str(quant.kv_dtype)
                ),
            )
            return replace(
                quant,
                weight_format=fmt,
                expert_weight_format=fmt,
                activation_format=fmt,
                kv_dtype=native_kv,
                activation_group_size=0,
                activation_scale_format="",
                activation_scale_pack=1,
                scale_format="",
                scale_bytes=0.0,
                output_format="",
                category_overrides=draft_overrides,
            )
    expert_fmt = quant.expert_weight_format if draft.num_experts > ZERO_EXPERTS else fmt
    act_fmt = fmt if fmt in FLOAT_WEIGHT_FORMATS else quant.activation_format
    kv_dt = _draft_declared_kv_dtype(
        draft,
        fallback=(
            _draft_declared_native_dtype(model, draft, quant)
            if is_independent and getattr(draft, "context_projection_mode", "") == "fused_kv_bank"
            else str(quant.kv_dtype)
        ),
    )
    return replace(
        quant,
        weight_format=fmt,
        expert_weight_format=expert_fmt,
        activation_format=act_fmt,
        kv_dtype=kv_dt,
    )


def optional_draft_spec(model: ModelSpec, kind: str) -> DraftSpec:
    aux = model.auxiliary
    if aux is None:
        raise ModelingError("catalog_draft_missing")
    dspark_mode = kind in {"dspark", "verify", "draft_verify"}
    draft = aux.dspark if dspark_mode else aux.mtp
    if draft is None:
        if dspark_mode and aux.mtp is not None and aux.mtp.kind in {"mtp_dspark", "dspark"}:
            draft = aux.mtp
        elif not dspark_mode and aux.dspark is not None:
            raise ModelingError("catalog_draft_mode_unavailable:mtp")
        elif dspark_mode:
            raise ModelingError("catalog_draft_mode_unavailable:dspark")
    if draft is None:
        raise ModelingError("catalog_draft_mode_missing")
    draft.validate()
    return draft


def declare_native_auxiliary_plans(model: ModelSpec) -> ModelSpec:
    """Attach explicit native auxiliary plans once, at model-load time.

    Every supported adapter declares its auxiliary blocks as ordinary
    ``LayerPlan`` objects here.  Geometry, weight residency and execution then
    consume the declared plans; nothing re-derives a donor per proposal and no
    last-layer synthesis is permitted downstream.
    """
    aux = model.auxiliary
    if aux is None:
        return model
    from .draft_geometry import native_draft_layer_plans
    changes = {}
    for name in ("mtp", "dspark"):
        spec = getattr(aux, name)
        if spec is None or tuple(spec.layer_plans):
            continue
        changes[name] = replace(
            spec, layer_plans=native_draft_layer_plans(model, spec))
    if not changes:
        return model
    return replace(model, auxiliary=replace(aux, **changes))
