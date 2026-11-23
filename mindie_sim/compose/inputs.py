"""Hardware, model, quantization and speculation input resolvers for ``compose``.

[Role]
  Resolve hardware SKU and topology aliases, read bounded local model configs,
  apply quantization and structural overrides, and validate/declare auxiliary
  draft structures before runtime synthesis in ``compose_case``.

[Invariants]
  * Hardware SKUs resolve strictly against registered Ascend, NVIDIA, or custom
    profiles; unknown identifiers raise ``ModelingError("unknown_hardware_sku:...")``.
  * Local JSON metadata reads are capped at 50 MiB.
  * Synthetic GQA draft synthesis requires explicit ``allow_synthetic_gqa=True``.

[Timeouts & Locks]
  Bounded local JSON config reads (< 50 MiB); no network, devices, or locks.

[Side-Effect Policy]
  Pure in-memory transformations; never writes to disk or mutates caller mappings.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import difflib
import json
from pathlib import Path
from typing import Any

from mindie_sim.backends.ascend import (
    ASCEND_TOPOLOGY_PRESETS,
    profiles as ascend_profiles,
)
from mindie_sim.backends.nvidia import profiles as nvidia_profiles
from mindie_sim.catalog.specs import build_model_spec
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, merge_source
from mindie_sim.model_catalog import (
    ARCHITECTURE_PRESET_ALIASES,
    ARCHITECTURE_PRESET_CATALOG,
    CATALOG,
    CONFIG_DIR,
    MODEL_ALIASES,
    customize_model_spec,
    load_model,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.serialization import apply_model_overrides

MAX_METADATA_MIB = 50
BYTES_PER_KIB = 1024
BYTES_PER_GIB = BYTES_PER_KIB * BYTES_PER_KIB * BYTES_PER_KIB
MX_EXPERT_BLOCK_GROUP = 32
MX_SCALE_PACK = 2
NEXTN_DEFAULT_HEAD_DIM = 192
from mindie_sim.backends.ascend import (
    A5_DT_144_GIB,
    A5_DT_425_AIC_CORES,
    A5_DT_486_AIC_CORES,
    A5_DT_547_AIC_CORES,
    A5_DT_96_GIB,
    DERIVED_ASCEND_SKUS,
)

PP_CAP_DEFAULT = (1, 2, 4, 8, 16)

HARDWARE_ALIASES: dict[str, str] = {
    "a2": "Ascend910B4", "910b": "Ascend910B4", "910b4": "Ascend910B4",
    "910b1": "Ascend910B1", "910b2": "Ascend910B2", "910b3": "Ascend910B3",
    "a3": "Ascend910_9392", "910c": "Ascend910_9392", "910_9392": "Ascend910_9392",
    "910_9382": "Ascend910_9382",
    "a5": "Ascend950DT_9582", "950": "Ascend950DT_9582",
    "a5-pod": "Ascend950DT_9581", "atlas-950-superpod": "Ascend950DT_9581",
    "950pr_958b": "Ascend950PR_958b", "ascend950pr_958b": "Ascend950PR_958b",
    "h20": "NVIDIA_H20", "h800": "NVIDIA_H800", "h200": "NVIDIA_H200",
    "b300": "NVIDIA_B300",
    "800i-a2": "Ascend910B4", "a2-8p": "Ascend910B4",
    "800i-a3": "Ascend910_9392",
    "800i-a3-8p": "Ascend910_9392", "a3-8p": "Ascend910_9392",
    "800i-a3-16p": "Ascend910_9392", "a3-16p": "Ascend910_9392",
    "a3-pod": "Ascend910_9392",
    "950dt-96g": "Ascend950DT_9582", "950dt-96g-pod": "Ascend950DT_9582",
    "a5-96g-pod": "Ascend950DT_9582",
    "950dt-144g": "Ascend950DT_9581", "950dt-144g-pod": "Ascend950DT_9581",
    "a5-144g-pod": "Ascend950DT_9581",
    "a5-pod-dt": "Ascend950DT_9581", "a5-pod-pr": "Ascend950PR_9579",
    "a5-server-dt": "Ascend950DT_9581", "a5-server-pr": "Ascend950PR_9579",
    "a5-server-pr-pcie-16p": "Ascend950PR_9579",
}
HARDWARE_ALIASES.update(dict.fromkeys(
    ("950pr", "ascend950pr", "950pr_a5", "ascend950pr_a5"),
    HARDWARE_ALIASES["a5-server-pr"],
))
HARDWARE_ALIASES.update(dict.fromkeys(
    ("950dt", "950dt_a5", "ascend950dt_a5", "910_9599", "ascend910_9599"),
    HARDWARE_ALIASES["a5"],
))
for _canon_dt, _short_dt in (
    ("Ascend950DT_425_96", "425-96"),
    ("Ascend950DT_486_96", "486-96"),
    ("Ascend950DT_547_96", "547-96"),
    ("Ascend950DT_547_144", "547-144"),
):
    _underscored = _short_dt.replace("-", "_")
    HARDWARE_ALIASES.update(dict.fromkeys(
        (
            _short_dt,
            _underscored,
            f"950dt_{_underscored}",
            f"950dt-{_short_dt}",
            f"ascend950dt_{_underscored}",
            f"ascend950dt-{_short_dt}",
        ),
        _canon_dt,
    ))

HARDWARE_TOPOLOGY_ALIASES: dict[str, tuple[str, str]] = {
    "800i-a2": ("Ascend910B4", "a2_standard"),
    "a2-8p": ("Ascend910B4", "a2_standard"),
    "800i-a3": ("Ascend910_9392", "a3_16p"),
    "800i-a3-8p": ("Ascend910_9392", "a3_8p"),
    "a3-8p": ("Ascend910_9392", "a3_8p"),
    "800i-a3-16p": ("Ascend910_9392", "a3_16p"),
    "a3-16p": ("Ascend910_9392", "a3_16p"),
    "a3-pod": ("Ascend910_9392", "a3_superpod"),
    "a5-pod": ("Ascend950DT_9581", "a5_pod"),
    "atlas-950-superpod": ("Ascend950DT_9581", "a5_pod"),
    "950dt-96g-pod": ("Ascend950DT_9582", "a5_pod"),
    "a5-96g-pod": ("Ascend950DT_9582", "a5_pod"),
    "950dt-144g-pod": ("Ascend950DT_9581", "a5_pod"),
    "a5-144g-pod": ("Ascend950DT_9581", "a5_pod"),
    "a5-pod-dt": ("Ascend950DT_9581", "a5_pod"),
    "a5-pod-pr": ("Ascend950PR_9579", "a5_pod"),
    "a5-server-dt": ("Ascend950DT_9581", "a5_server"),
    "a5-server-pr": ("Ascend950PR_9579", "a5_server"),
    "a5-server-pr-pcie-16p": ("Ascend950PR_9579", "a5_pcie_16p"),
}

_QUANT_ALIASES: dict[str, str] = {
    "w4a8_mxfp": "mxfp4",
    "w4a8-mxfp": "mxfp4",
    "w4a8_mxfp4": "mxfp4",
    "w4a8-mxfp4": "mxfp4",
    "mxfp4_w4a8": "mxfp4",
    "w8a8_mxfp": "mxfp8",
    "w8a8-mxfp": "mxfp8",
    "w8a8_mxfp8": "mxfp8",
    "w8a8-mxfp8": "mxfp8",
    "c8": "int8",
}

_SUGGESTION_LIMIT = 5
_SUGGESTION_CUTOFF = 0.35


def _arch_key(sku: str) -> str:
    raw = str(sku or "").strip()
    if not raw:
        return "other"
    profiles = _ensure_registered_profiles()
    entry = profiles.get(raw)
    if entry is None and raw in DERIVED_ASCEND_SKUS:
        entry = profiles.get(DERIVED_ASCEND_SKUS[raw]["base"])
    if isinstance(entry, Mapping):
        arch = str(entry.get("architecture") or "").strip().lower()
        if arch in {"a2", "a3", "a5"}:
            return arch
    low = raw.lower()
    if low in {"a5", "arch35", "dav_3510"} or low.startswith(("ascend950", "ascend910_95")):
        return "a5"
    if low in {"a3", "arch22"} or low.startswith("ascend910_93"):
        return "a3"
    if low in {"a2", "arch20"} or low.startswith("ascend910b"):
        return "a2"
    return "other"


def _ensure_registered_profiles() -> dict[str, Any]:
    return ascend_profiles()


def _sku_hbm_gib(sku: str, reg: Mapping[str, Any]) -> int | None:
    entry = reg.get(sku)
    if not isinstance(entry, Mapping):
        return None
    mem = entry.get("memory_bytes")
    if mem is None and isinstance(entry.get("device"), Mapping):
        mem = entry["device"].get("memory_bytes")
    return int(round(float(mem) / BYTES_PER_GIB)) if mem else None


def _format_sku_suggestions(raw: str, reg: Mapping[str, Any]) -> str:
    pool = sorted(set(reg) | set(HARDWARE_ALIASES))
    matches = difflib.get_close_matches(
        str(raw).strip(), pool, n=_SUGGESTION_LIMIT, cutoff=_SUGGESTION_CUTOFF,
    )
    if not matches:
        matches = ["Ascend950DT_9582", "Ascend950DT_9581", "Ascend910_9392", "Ascend910B4", "950dt-96g-pod"]
    parts: list[str] = []
    for item in matches:
        canon = HARDWARE_ALIASES.get(item.lower(), item)
        gib = _sku_hbm_gib(canon, reg)
        preset = HARDWARE_TOPOLOGY_ALIASES.get(item.lower(), ("", ""))[1]
        detail = ", ".join(p for p in (f"{gib}GiB" if gib else "", f"preset={preset}" if preset else "") if p)
        parts.append(f"{item} ({detail})" if detail else item)
    presets = ", ".join(sorted(ASCEND_TOPOLOGY_PRESETS))
    return f"closest valid: {', '.join(parts)}; presets: {presets}"


def _format_model_suggestions(raw: str) -> str:
    pool = sorted(set(CATALOG) | set(MODEL_ALIASES) | set(ARCHITECTURE_PRESET_CATALOG))
    matches = difflib.get_close_matches(
        str(raw).strip().lower(), pool, n=_SUGGESTION_LIMIT, cutoff=_SUGGESTION_CUTOFF,
    ) or sorted(CATALOG)[:_SUGGESTION_LIMIT]
    return f"closest valid: {', '.join(matches)}; or pass a path to HuggingFace config.json or model directory"


def _format_quant_suggestions(raw: str) -> str:
    from mindie_sim.catalog.quant import QUANTIZATION_PRESETS
    pool = sorted(set(QUANTIZATION_PRESETS) | set(_QUANT_ALIASES))
    matches = difflib.get_close_matches(
        str(raw).strip().lower(), pool, n=_SUGGESTION_LIMIT, cutoff=_SUGGESTION_CUTOFF,
    ) or ["bf16", "fp16", "w8a8", "w4a8", "mxfp8", "mxfp4", "fp8"]
    return f"closest valid: {', '.join(matches)}"


def resolve_hardware_sku(hardware: str) -> str:
    """Resolve a hardware alias or exact SKU name against registered profiles."""
    reg = {**_ensure_registered_profiles(), **DERIVED_ASCEND_SKUS, **nvidia_profiles()}
    if not isinstance(hardware, str) or not hardware.strip():
        raise ModelingError(f"unknown_hardware_sku:{hardware} — {_format_sku_suggestions(str(hardware), reg)}")
    raw = hardware.strip()
    lower = raw.lower()
    if lower in HARDWARE_ALIASES:
        sku = HARDWARE_ALIASES[lower]
        if sku in reg:
            return sku
    if raw in reg:
        return raw
    for registered_sku in reg:
        if registered_sku.lower() == lower:
            return registered_sku
    from mindie_sim.custom_hardware import resolve_custom_hardware_sku
    custom = resolve_custom_hardware_sku(raw)
    if custom is not None:
        return str(custom["sku"])
    raise ModelingError(f"unknown_hardware_sku:{hardware} — {_format_sku_suggestions(raw, reg)}")


def _hardware_backend(sku: str) -> str:
    if sku in nvidia_profiles():
        return "nvidia"
    if sku in _ensure_registered_profiles() or sku in DERIVED_ASCEND_SKUS:
        return "ascend"
    from mindie_sim.custom_hardware import resolve_custom_hardware_sku
    if resolve_custom_hardware_sku(sku) is not None:
        return "custom"
    raise ModelingError(f"unknown_hardware_sku:{sku}")


def _resolve_hardware(hardware: str | Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the canonical SKU, vendor adapter and any alias topology preset."""
    if isinstance(hardware, Mapping):
        raw = dict(hardware)
        requested = raw.get("sku", "a3")
        preset = str(raw.get("topology_preset")) if raw.get("topology_preset") else ""
        lower_req = str(requested).strip().lower()
        if lower_req in HARDWARE_TOPOLOGY_ALIASES:
            sku, alias_preset = HARDWARE_TOPOLOGY_ALIASES[lower_req]
            return sku, _hardware_backend(sku), (preset or alias_preset)
        try:
            sku = resolve_hardware_sku(str(requested))
            return sku, _hardware_backend(sku), preset
        except ModelingError:
            if any(k in raw for k in ("archetype", "hbm_gb", "hbm_bw_gbps", "cube_fp16_tflops")):
                return str(requested).strip() or "CUSTOM_INLINE_SKU", "custom", preset
            raise
    lower = str(hardware).strip().lower()
    if lower in HARDWARE_TOPOLOGY_ALIASES:
        sku, preset = HARDWARE_TOPOLOGY_ALIASES[lower]
        return sku, _hardware_backend(sku), preset
    sku = resolve_hardware_sku(str(hardware))
    return sku, _hardware_backend(sku), ""


def _read_config_json(path: Path) -> dict[str, Any]:
    target = path / "config.json" if path.is_dir() else path
    if not target.is_file():
        raise ModelingError(f"model_config_not_found:{path}")
    if target.stat().st_size > MAX_METADATA_MIB * BYTES_PER_KIB * BYTES_PER_KIB:
        raise ValueError("metadata_input_exceeds_50_MiB")
    with target.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ModelingError(f"model_config_invalid_mapping:{path}")
    return dict(data)


def _attach_checkpoint_quant(
    spec: ModelSpec,
    quant_spec: Any,
    *,
    repo_id: str = "",
) -> ModelSpec:
    if quant_spec is None and not repo_id:
        return spec
    meta = dict(spec.metadata)
    if repo_id:
        meta["checkpoint_repo_id"] = repo_id
    if quant_spec is not None:
        meta["checkpoint_quant_declared"] = True
        return replace(
            spec,
            quant=quant_spec,
            quantization=str(quant_spec.name),
            kv_dtype=str(quant_spec.kv_dtype),
            metadata=tuple(sorted(meta.items())),
        )
    return replace(spec, metadata=tuple(sorted(meta.items())))


def _read_companion_quant_spec(path: Path, raw_config: Mapping[str, Any]) -> Any:
    base_dir = path if path.is_dir() else path.parent
    desc_path = base_dir / "quant_model_description.json"
    if not desc_path.is_file():
        return None
    if desc_path.stat().st_size > MAX_METADATA_MIB * BYTES_PER_KIB * BYTES_PER_KIB:
        raise ValueError("metadata_input_exceeds_50_MiB")
    with desc_path.open(encoding="utf-8") as stream:
        desc_data = json.load(stream)
    if not isinstance(desc_data, Mapping):
        return None
    from mindie_sim.catalog.npu_checkpoints import parse_quant_model_description
    return parse_quant_model_description(desc_data, base_config=raw_config)


def _resolve_checkpoint_model_spec(key: str) -> ModelSpec | None:
    from mindie_sim.catalog.npu_checkpoints import (
        lookup_eco_tech_repo,
        resolve_checkpoint_metadata,
    )
    is_repo_or_url = (
        lookup_eco_tech_repo(key) is not None
        or "/" in key
        or key.lower().startswith(("http://", "https://"))
    )
    if not is_repo_or_url:
        return None
    meta = resolve_checkpoint_metadata(key, allow_network=False)
    canon = MODEL_ALIASES.get(
        meta.canonical_model,
        ARCHITECTURE_PRESET_ALIASES.get(meta.canonical_model, meta.canonical_model),
    )
    if canon in CATALOG or canon in ARCHITECTURE_PRESET_CATALOG:
        base_spec = load_model(canon)
        return _attach_checkpoint_quant(base_spec, meta.quant_spec, repo_id=meta.repo_id)
    if isinstance(meta.config, Mapping) and meta.config:
        name = str(meta.config.get("name") or meta.canonical_model or meta.repo_id)
        base_spec = build_model_spec(name, meta.config)
        return _attach_checkpoint_quant(base_spec, meta.quant_spec, repo_id=meta.repo_id)
    return None


def _resolve_model_spec(model: str | Path | Mapping[str, Any]) -> ModelSpec:
    if isinstance(model, Mapping):
        raw = dict(model)
        if "layers" in raw and "hidden_size" in raw and "model_type" not in raw:
            return ModelSpec.from_mapping(raw)
        name = str(raw.get("name") or raw.get("model_type") or "inline-model")
        return build_model_spec(name, raw)
    if isinstance(model, Path):
        resolved_path = model.expanduser()
        raw = _read_config_json(resolved_path)
        fallback_name = (
            resolved_path.name if resolved_path.is_dir()
            else (resolved_path.parent.name if resolved_path.stem == "config" and resolved_path.parent.name else resolved_path.stem)
        )
        name = str(raw.get("name") or fallback_name)
        if "layers" in raw and "hidden_size" in raw and "model_type" not in raw:
            spec = ModelSpec.from_mapping(raw)
        else:
            spec = build_model_spec(name, raw)
        qspec = _read_companion_quant_spec(resolved_path, raw)
        return _attach_checkpoint_quant(spec, qspec)
    if not isinstance(model, str) or not model.strip():
        raise ModelingError(f"invalid_model_identifier:{model}")
    key = model.strip()
    canonical = MODEL_ALIASES.get(key, ARCHITECTURE_PRESET_ALIASES.get(key, key))
    if canonical in CATALOG or canonical in ARCHITECTURE_PRESET_CATALOG:
        return load_model(canonical)
    candidate = Path(key).expanduser()
    if candidate.exists():
        return _resolve_model_spec(candidate)
    cfg_candidate = CONFIG_DIR / (key if key.endswith(".json") else f"{key}.json")
    if cfg_candidate.is_file():
        return _resolve_model_spec(cfg_candidate)
    checkpoint_spec = _resolve_checkpoint_model_spec(key)
    if checkpoint_spec is not None:
        return checkpoint_spec
    try:
        return load_model(key)
    except (KeyError, ValueError, ModelingError) as exc:
        raise ModelingError(f"unknown_model:{key} — {_format_model_suggestions(key)}") from exc


def _apply_quant_override(
    model_spec: ModelSpec,
    quant: str | Mapping[str, Any] | None,
    machine: Mapping[str, Any],
) -> ModelSpec:
    if quant is None:
        peaks = machine.get("device", {}).get("compute_peaks", {})
        has_fp8 = any(k in peaks for k in ("fp8", "fp8_e4m3", "mxfp4"))
        if not has_fp8 and model_spec.quant is not None:
            wf = str(model_spec.quant.weight_format)
            ewf = str(model_spec.quant.expert_weight_format)
            if "fp8" in wf or "fp8" in ewf or "mxfp4" in ewf:
                return apply_model_overrides(model_spec, {"quantization": "w8a8"})
        return model_spec
    if isinstance(quant, str):
        raw_key = quant.strip().lower()
        qkey = _QUANT_ALIASES.get(raw_key, raw_key)
        native_q = model_spec.quant
        native_mxfp4_excluded = (
            native_q is not None
            and str(native_q.expert_weight_format) == "mxfp4"
            and bool(native_q.exclusions)
        )
        if qkey in {"mxfp4", "mxfp4_routed_only"} and native_mxfp4_excluded:
            return model_spec
        try:
            return apply_model_overrides(model_spec, {"quantization": qkey})
        except (ValueError, ModelingError) as exc:
            raise ModelingError(
                f"invalid_quant_override:{quant} — {_format_quant_suggestions(raw_key)}"
            ) from exc
    if isinstance(quant, Mapping):
        qdict = dict(quant)
        return apply_model_overrides(
            model_spec,
            {"quant": qdict, "kv_dtype": str(qdict.get("kv_dtype", model_spec.kv_dtype))},
        )
    raise ModelingError(f"invalid_quant_override:{quant} — {_format_quant_suggestions(str(quant))}")


def _resolve_max_position_embeddings(
    model: str | Path | Mapping[str, Any], model_spec: ModelSpec,
) -> int:
    for src in (
        getattr(model_spec, "max_position_embeddings", 0),
        dict(model_spec.catalog_facts).get("max_position_embeddings", 0),
        dict(model_spec.metadata).get("max_position_embeddings", 0),
    ):
        if src:
            return int(src)
    raw: Mapping[str, Any] | None = model if isinstance(model, Mapping) else None
    if isinstance(model, Path):
        raw = _read_config_json(model)
    elif isinstance(model, str):
        key = model.strip()
        canonical = MODEL_ALIASES.get(key, ARCHITECTURE_PRESET_ALIASES.get(key, key))
        entries = {**CATALOG, **ARCHITECTURE_PRESET_CATALOG}
        cfg_path = (CONFIG_DIR / entries[canonical]) if canonical in entries else Path(key)
        if cfg_path.exists():
            raw = _read_config_json(cfg_path)
    if isinstance(raw, Mapping):
        tc = raw.get("text_config") if isinstance(raw.get("text_config"), Mapping) else {}
        val = raw.get("max_position_embeddings") or tc.get("max_position_embeddings")
        if val:
            return int(val)
    return 0


def _apply_structural_overrides(
    model_spec: ModelSpec, *, layers: int | None, n_routed_experts: int | None,
    num_experts_per_tok: int | None, promote_dense_to_moe: bool = False,
) -> ModelSpec:
    if (layers is None and n_routed_experts is None and num_experts_per_tok is None
            and not promote_dense_to_moe):
        return model_spec
    return customize_model_spec(
        model_spec, layers=layers, n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        promote_dense_to_moe=promote_dense_to_moe,
    )


def prepare_speculation(model: ModelSpec, proposals: int, *,
                        allow_synthetic_gqa: bool = False) -> ModelSpec:
    """Declare a native draft only when the model already owns one."""
    if proposals <= 0:
        return model
    aux = model.auxiliary
    if aux is not None and aux.standalone_draft:
        raise ModelingError("standalone_draft_requires_target_model")
    if aux is not None and (aux.mtp is not None or aux.dspark is not None):
        return model
    if model.attention == "mla":
        return model
    if not allow_synthetic_gqa or model.attention != "gqa":
        raise ModelingError("draft_structure_required_for_mtp_proposal_tokens")
    from mindie_sim.catalog.draft import declare_native_auxiliary_plans
    draft = DraftSpec(
        kind="mtp", layers=1, hidden_size=model.hidden_size,
        intermediate_size=model.intermediate_size, num_heads=model.num_heads,
        num_kv_heads=model.num_kv_heads, head_dim=model.head_dim,
        q_width=model.num_heads * model.head_dim,
        kv_width=2 * model.num_kv_heads * model.head_dim,
        target_layer_ids=(model.layers - 1,), activation_dtype=model.activation_dtype,
        source=(("kind", "explicit_synthetic_gqa_mtp"),),
    )
    return declare_native_auxiliary_plans(
        replace(model, auxiliary=replace(aux or AuxiliarySpec(), mtp=draft))
    )


def ensure_native_mla_draft(model_spec: ModelSpec, proposals: int) -> ModelSpec:
    """Synthesize standard NextN MLA MTP when an MLA MoE model requests proposals."""
    aux = model_spec.auxiliary
    needs_draft = aux is None or (aux.mtp is None and aux.dspark is None)
    if proposals <= 0 or not needs_draft:
        return model_spec
    from mindie_sim.catalog.draft import declare_native_auxiliary_plans

    facts = dict(model_spec.catalog_facts)
    nextn = int(facts.get("num_nextn_predict_layers", 0) or 0)
    moe_ref = next((p for p in model_spec.layer_plans if p.ffn == "moe"), None)
    if (nextn <= 0 and moe_ref is not None and moe_ref.attention == "mla"
            and moe_ref.q_lora_rank > 0):
        nextn = 1
    if nextn > 0 and moe_ref is not None and moe_ref.attention == "mla":
        qk_dim = moe_ref.qk_nope_head_dim + moe_ref.qk_rope_head_dim
        draft = DraftSpec(
            kind="mtp", layers=nextn, hidden_size=model_spec.hidden_size,
            intermediate_size=moe_ref.moe_intermediate_size,
            num_heads=moe_ref.num_heads, num_kv_heads=1,
            head_dim=qk_dim or NEXTN_DEFAULT_HEAD_DIM,
            q_width=moe_ref.num_heads * qk_dim, kv_width=qk_dim,
            attention_kind="mla", kv_cache_elements=moe_ref.kv_cache_elements,
            weight_category="mtp", q_lora_rank=moe_ref.q_lora_rank,
            kv_lora_rank=moe_ref.kv_lora_rank,
            qk_nope_head_dim=moe_ref.qk_nope_head_dim,
            qk_rope_head_dim=moe_ref.qk_rope_head_dim,
            v_head_dim=moe_ref.v_head_dim, num_experts=moe_ref.num_experts,
            top_k=moe_ref.top_k,
            expert_intermediate_size=moe_ref.moe_intermediate_size,
            num_shared_experts=moe_ref.num_shared_experts,
            target_shared_embedding=True, target_shared_lm_head=True,
            base_logits=True, pp_capability=PP_CAP_DEFAULT,
            vocab_size=model_spec.vocab_size,
            source=merge_source({"kind": "standard_deepseek_nextn_mtp"}),
        )
        return declare_native_auxiliary_plans(
            replace(model_spec, auxiliary=AuxiliarySpec(mtp=draft)))
    return model_spec


def _tier4_router_spec(model_spec: ModelSpec, sku: str, *, raw_eager: bool) -> ModelSpec:
    """A5 production executes FP32 checkpoint routers directly in Tier-4 form."""
    if raw_eager or _arch_key(sku) != "a5":
        return model_spec

    def rewrite(plans):
        return tuple(
            replace(plan, router_dtype="", router_output_dtype="fp32",
                    router_precast_weight_copy=False)
            if plan.router_dtype == "fp32" else plan
            for plan in plans)

    aux = model_spec.auxiliary
    changes = {}
    if aux is not None:
        for name in ("mtp", "dspark"):
            spec = getattr(aux, name)
            if spec is not None and tuple(getattr(spec, "layer_plans", ())):
                changes[name] = replace(spec, layer_plans=rewrite(spec.layer_plans))
    if changes:
        aux = replace(aux, **changes)
    return replace(model_spec, layer_plans=rewrite(model_spec.layer_plans), auxiliary=aux)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
