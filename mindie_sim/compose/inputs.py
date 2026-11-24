"""Hardware, model, quantization and speculation input resolvers for ``compose``.

[Role]
  Resolve hardware SKU and topology aliases, read bounded local model configs,
  apply quantization and structural overrides, and validate/declare auxiliary
  draft structures before runtime synthesis in ``compose_case``.

[Invariants]
  * Hardware SKUs resolve strictly against registered Ascend, NVIDIA, or custom
    profiles; unknown identifiers raise ``ModelingError("unknown_hardware_sku:...")``.
  * Local JSON metadata reads are capped at 50 MiB.
  * Direct ``prepare_speculation`` synthesis of a 1-layer EAGLE3/GQA draft
    requires explicit ``allow_synthetic_gqa=True`` and stamps mandatory
    ``is_synthetic=True`` + ``draft_provenance="synthetic_eagle3_gqa"``;
    ``compose_case`` forbids synthetic draft synthesis in simulation mode and
    never silently enables it in modeling mode.

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
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, QuantSpec, merge_source
from mindie_sim.model_catalog import (
    ARCHITECTURE_PRESET_ALIASES,
    ARCHITECTURE_PRESET_CATALOG,
    CATALOG,
    CONFIG_DIR,
    MODEL_ALIASES,
    customize_model_spec,
    load_model,
)
from mindie_sim.schema.checkpoint import CheckpointProvenance, CheckpointQuantMap
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.serialization import apply_model_overrides
from mindie_sim.schema.stack import WorkUnitKind, WorkloadDomain
from mindie_sim.schema.workload import SYNTHETIC_EAGLE3_GQA_PROVENANCE

MAX_METADATA_MIB = 50
BYTES_PER_KIB = 1024
BYTES_PER_GIB = BYTES_PER_KIB * BYTES_PER_KIB * BYTES_PER_KIB
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
    "910_9362": "Ascend910_9362", "910_9382": "Ascend910_9382",
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
    "w8a8_mxfp": "mxfp8",
    "w8a8-mxfp": "mxfp8",
    "w8a8_mxfp8": "mxfp8",
    "w8a8-mxfp8": "mxfp8",
    "c8": "int8",
}

_SUGGESTION_LIMIT = 5
_SUGGESTION_CUTOFF = 0.35
_DIFFUSION_DOMAIN_TOKENS = (
    "dit", "diffusion", "hunyuan_video", "hunyuan-video",
    "wan", "flux", "sd3", "cogvideo", "sora",
)
_CANONICAL_PROJECTION_PRESETS = frozenset({
    "mxfp8", "mxfp8_mxfp4", "mxfp4_routed_only", "mxfp4_all", "mxfp4",
    "w4a8_mxfp", "w4a4_mxfp4", "fp8", "w8a8", "w8a8_dynamic",
    "w8a8_static", "w8a8_mix", "w4a8", "w4a8_dynamic", "w4a16",
    "bf16", "fp16", "fp32", "none",
})
_MXFP4_FMT_TOKENS = frozenset({"mxfp4", "mx_fp4", "mxfp4_e2m1", "fp4", "fp4_e2m1"})
_FP8_FMT_TOKENS = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8", "mx_fp8", "mxfp8_e4m3"})
_MX_SCALE_FMT_TOKENS = frozenset({"fp8_e8m0", "ue8m0", "e8m0"})
_INT4_FMT_TOKENS = frozenset({"int4", "int4_weight", "w4a8", "w4a16"})
_AUTO_VA_QUANT_TOKENS = frozenset({"auto_va", "auto"})
_MX_FP8_FORMATS = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8"})
_MX_FP4_FORMATS = frozenset({"mxfp4", "fp4", "fp4_e2m1"})
_MX_PROFILE_KEYS = frozenset({"mxfp8", "mxfp4", "mxfp8_mxfp4", "w4a8_mxfp", "w4a4_mxfp4"})
_INT8_PROFILE_KEYS = frozenset({"w8a8", "w8a8_dynamic", "w8a8_int8", "int8"})
_INT4_PROFILE_KEYS = frozenset({"w4a8", "w4a8_dynamic", "w4a8-int8"})


def _qspec_weight_formats(qspec: QuantSpec) -> set[str]:
    fmts = {str(qspec.weight_format).lower(), str(qspec.expert_weight_format).lower()}
    for _, override in getattr(qspec, "category_overrides", ()) or ():
        for key, value in override:
            if key in {"weight_format", "expert_weight_format"}:
                fmts.add(str(value).lower())
    return fmts


def curated_quant_compatible_with_profile(
    ckpt_qspec: QuantSpec | None,
    profile_quant: str | None,
    *,
    model_spec: ModelSpec | None = None,
    mlapo: bool = False,
) -> bool:
    """Return True when a curated checkpoint matches the profile's precision family."""
    if ckpt_qspec is None or not profile_quant:
        return False
    pq = str(profile_quant).strip().lower()
    if pq == "mxfp4_routed_only":
        return False
    if mlapo and model_spec is not None and any(
        p.attention == "mla" and (p.indexer or p.owns_topk or p.index_query_per_layer)
        for p in model_spec.layer_plans
    ):
        proj_act = str(ckpt_qspec.for_category("mla.q_b").activation_format).strip().lower()
        idx_act = str(ckpt_qspec.for_category("indexer.wq").activation_format).strip().lower()
        native_act = str(model_spec.activation_dtype or "bf16").strip().lower()
        if proj_act != native_act and idx_act != proj_act:
            return False
    fmts = _qspec_weight_formats(ckpt_qspec)
    has_int = bool(fmts & {"int8", "int4"})
    has_fp8 = bool(fmts & _MX_FP8_FORMATS)
    has_fp4 = bool(fmts & _MX_FP4_FORMATS)
    if pq in _MX_PROFILE_KEYS:
        if has_int or not (has_fp8 or has_fp4):
            return False
        if pq == "mxfp8":
            return has_fp8 and not has_fp4
        return has_fp4
    if pq in _INT8_PROFILE_KEYS:
        return "int8" in fmts and "int4" not in fmts and not (has_fp8 or has_fp4)
    if pq in _INT4_PROFILE_KEYS:
        return "int4" in fmts and not (has_fp8 or has_fp4)
    return False
_CHECKPOINT_NATIVE_QUANT_TOKENS = frozenset({"checkpoint", "native"})
_UNQUANTIZED_QUANT_TOKENS = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "none"})
DEFAULT_CONTEXT_TOKENS_FOR_QUANT = 4096


def canonical_quant_preset_key(qspec: QuantSpec | Mapping[str, Any] | None) -> str | None:
    """Derive canonical stack projection preset key from a ``QuantSpec`` or mapping.

    Returns ``None`` for native unquantized float specifications so downstream
    resolvers classify them as ``unquantized_native`` without passing arbitrary
    user/test ``QuantSpec.name`` labels as checkpoint quantization tags.
    """
    if qspec is None:
        return None
    if isinstance(qspec, Mapping):
        raw_name = str(qspec.get("name") or "").strip().lower()
        wf = str(qspec.get("weight_format") or "").strip().lower().replace("-", "_")
        ewf = str(qspec.get("expert_weight_format") or wf).strip().lower().replace("-", "_")
        af = str(qspec.get("activation_format") or "").strip().lower().replace("-", "_")
        sf = str(qspec.get("scale_format") or "").strip().lower().replace("-", "_")
    else:
        raw_name = str(qspec.name or "").strip().lower()
        wf = str(qspec.weight_format or "").strip().lower().replace("-", "_")
        ewf = str(qspec.expert_weight_format or wf).strip().lower().replace("-", "_")
        af = str(qspec.activation_format or "").strip().lower().replace("-", "_")
        sf = str(qspec.scale_format or "").strip().lower().replace("-", "_")
    norm_name = _QUANT_ALIASES.get(raw_name, raw_name)
    unquant_fmts = _UNQUANTIZED_QUANT_TOKENS | {""}
    if wf in unquant_fmts and ewf in unquant_fmts:
        if norm_name in {"w8a8", "w8a8_dynamic", "w8a8_static", "w8a8_mix", "w4a8", "w4a8_dynamic", "w4a16", "fp8", "mxfp8", "mxfp4", "mxfp8_mxfp4", "mxfp4_routed_only", "mxfp4_all"}:
            return norm_name
        return None
    if norm_name in _CANONICAL_PROJECTION_PRESETS and norm_name not in _UNQUANTIZED_QUANT_TOKENS:
        return norm_name
    if ewf in _MXFP4_FMT_TOKENS or wf in _MXFP4_FMT_TOKENS:
        if wf in _MXFP4_FMT_TOKENS:
            return "mxfp4_all" if af in _FP8_FMT_TOKENS else "w4a4_mxfp4"
        if wf in _FP8_FMT_TOKENS:
            return "mxfp8_mxfp4"
        return "mxfp4_routed_only"
    if wf in _FP8_FMT_TOKENS or ewf in _FP8_FMT_TOKENS:
        return "mxfp8" if sf in _MX_SCALE_FMT_TOKENS else "fp8"
    if ewf in _INT4_FMT_TOKENS or wf in _INT4_FMT_TOKENS:
        return "w4a8" if af == "int8" else "w4a16"
    if wf in {"int8", "w8a8"} or ewf in {"int8", "w8a8"} or af == "int8":
        return "w8a8"
    return None


def synthetic_checkpoint_quant_map(
    model_spec: ModelSpec,
    *,
    canon: str,
    native_dtype: str = "bf16",
) -> CheckpointQuantMap:
    """Build an explicit-preset or unquantized ``CheckpointQuantMap`` from ``model_spec.quant``."""
    preset = canonical_quant_preset_key(model_spec.quant)
    if preset is None:
        return CheckpointQuantMap(
            provenance=CheckpointProvenance(
                f"native/{canon or model_spec.name}", "unquantized_native", origin="unquantized_native",
            ),
            native_float_dtype=native_dtype,
        )
    return CheckpointQuantMap(
        provenance=CheckpointProvenance(
            f"explicit-preset/{preset}", "explicit_synthetic_preset", origin="explicit_synthetic_preset",
        ),
        native_float_dtype=native_dtype,
        model_quant_type=preset,
    )


def infer_workload_domain(model_spec: ModelSpec) -> WorkloadDomain:
    """Classify a ``ModelSpec`` structurally into ``llm`` / ``vlm`` / ``diffusion``."""
    meta = dict(getattr(model_spec, "metadata", ()) or ())
    facts = dict(getattr(model_spec, "catalog_facts", ()) or ())
    explicit = str(meta.get("domain") or facts.get("workload_domain") or "").strip().lower()
    if explicit in {"llm", "vlm", "diffusion", "custom"}:
        return explicit  # type: ignore[return-value]
    for plan in model_spec.layer_plans:
        arch = str(getattr(plan, "architecture", "") or "").lower()
        if any(tok in arch for tok in _DIFFUSION_DOMAIN_TOKENS):
            return "diffusion"
    model_type = str(facts.get("model_type") or facts.get("family") or "").strip().lower()
    name_low = str(model_spec.name or "").strip().lower()
    if any(tok in model_type or tok in name_low for tok in _DIFFUSION_DOMAIN_TOKENS):
        return "diffusion"
    if model_spec.vision is not None or bool(facts.get("vision_present")):
        return "vlm"
    return "llm"


def default_work_unit_for_domain(domain: WorkloadDomain) -> WorkUnitKind:
    """Return canonical default work-unit kind for one ``WorkloadDomain``."""
    if domain == "diffusion":
        return "denoise_timestep"
    if domain == "custom":
        return "custom_step"
    return "prefill_decode_step"


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
        if getattr(quant_spec, "layer_quant_overrides", ()) and "layer_quant_overrides" not in meta:
            meta["layer_quant_overrides"] = quant_spec.layer_quant_overrides_dict()
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
    from mindie_sim.catalog.checkpoint_reader import (
        quant_spec_from_checkpoint_map,
        read_modelslim_description_file,
    )

    qmap = read_modelslim_description_file(
        desc_path,
        checkpoint_id=f"local/{base_dir.name or 'checkpoint'}",
        base_config=raw_config,
    )
    return quant_spec_from_checkpoint_map(qmap, base_config=raw_config)


def _canonical_model_id(model_arg: Any, model_spec: ModelSpec) -> str:
    if isinstance(model_arg, str):
        key = model_arg.strip()
        canon = MODEL_ALIASES.get(key, ARCHITECTURE_PRESET_ALIASES.get(key, key))
        if canon in CATALOG or canon in ARCHITECTURE_PRESET_CATALOG:
            return canon
    name_key = str(model_spec.name or "").strip().lower()
    return MODEL_ALIASES.get(name_key, ARCHITECTURE_PRESET_ALIASES.get(name_key, name_key))


def _local_checkpoint_quant_map(
    model_arg: Any,
    model_spec: ModelSpec,
) -> CheckpointQuantMap | None:
    if not isinstance(model_arg, (Path, str)):
        return None
    candidate = Path(model_arg).expanduser() if isinstance(model_arg, str) else model_arg.expanduser()
    if not candidate.exists():
        return None
    from mindie_sim.catalog.checkpoint_reader import (
        read_hf_quantization_config,
        read_modelslim_description_file,
    )

    base_dir = candidate if candidate.is_dir() else candidate.parent
    cfg_path = base_dir / "config.json" if candidate.is_dir() else candidate
    raw_cfg: Mapping[str, Any] = _read_config_json(cfg_path) if cfg_path.is_file() else {}
    ckpt_id = f"local/{base_dir.name or model_spec.name}"
    desc_path = base_dir / "quant_model_description.json"
    if desc_path.is_file():
        return read_modelslim_description_file(desc_path, checkpoint_id=ckpt_id, base_config=raw_cfg)
    if raw_cfg:
        return read_hf_quantization_config(raw_cfg, checkpoint_id=ckpt_id)
    return None


def _explicit_checkpoint_quant_map(
    quant: str | Mapping[str, Any] | QuantSpec,
    *,
    canon: str,
    model_name: str,
    native_dtype: str,
    lookup_curated: Any,
) -> tuple[CheckpointQuantMap, QuantSpec | None, str, str | None, str | None] | None:
    if isinstance(quant, str):
        raw_q_str = quant.strip().lower()
        if raw_q_str in _UNQUANTIZED_QUANT_TOKENS:
            dtype_val = (
                "fp16" if raw_q_str in {"fp16", "float16"}
                else ("fp32" if raw_q_str == "fp32" else "bf16")
            )
            return (
                CheckpointQuantMap(
                    provenance=CheckpointProvenance(
                        f"native/{canon or model_name}", "unquantized_native", origin="unquantized_native",
                    ),
                    native_float_dtype=dtype_val,
                ),
                None, "", None, None,
            )
        if raw_q_str and raw_q_str not in (_AUTO_VA_QUANT_TOKENS | _CHECKPOINT_NATIVE_QUANT_TOKENS):
            norm_key = _QUANT_ALIASES.get(raw_q_str, raw_q_str)
            return (
                CheckpointQuantMap(
                    provenance=CheckpointProvenance(
                        f"explicit-preset/{norm_key}", "explicit_synthetic_preset", origin="explicit_synthetic_preset",
                    ),
                    native_float_dtype=native_dtype,
                    model_quant_type=norm_key,
                ),
                None, "", None, None,
            )
        return None
    if isinstance(quant, QuantSpec):
        ckpt_ref = dict(getattr(quant, "source", ()) or ()).get("checkpoint")
        curated = lookup_curated(str(ckpt_ref)) if ckpt_ref else None
        if curated is not None:
            return curated, quant, "", None, None
        preset = canonical_quant_preset_key(quant)
        if preset is None:
            return (
                CheckpointQuantMap(
                    provenance=CheckpointProvenance(
                        f"native/{canon or model_name}", "unquantized_native", origin="unquantized_native",
                    ),
                    native_float_dtype=native_dtype,
                ),
                quant, "", None, None,
            )
        return (
            CheckpointQuantMap(
                provenance=CheckpointProvenance(
                    f"explicit-preset/{preset}", "explicit_synthetic_preset", origin="explicit_synthetic_preset",
                ),
                native_float_dtype=native_dtype,
                model_quant_type=preset,
            ),
            quant, "", None, None,
        )
    if isinstance(quant, Mapping):
        preset = canonical_quant_preset_key(quant)
        if preset is None:
            return (
                CheckpointQuantMap(
                    provenance=CheckpointProvenance(
                        f"native/{canon or model_name}", "unquantized_native", origin="unquantized_native",
                    ),
                    native_float_dtype=native_dtype,
                ),
                None, "", None, None,
            )
        return (
            CheckpointQuantMap(
                provenance=CheckpointProvenance(
                    f"explicit-preset/{preset}", "explicit_synthetic_preset", origin="explicit_synthetic_preset",
                ),
                native_float_dtype=native_dtype,
                model_quant_type=preset,
            ),
            None, "", None, None,
        )
    return None


def resolve_checkpoint_quant_map(
    model_arg: Any,
    model_spec: ModelSpec,
    *,
    sku: str,
    hardware_obj: Any,
    adapter: Any,
    quant: str | Mapping[str, Any] | QuantSpec | None,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS_FOR_QUANT,
) -> tuple[CheckpointQuantMap, QuantSpec | None, str, str | None, str | None]:
    """Resolve on-disk ``CheckpointQuantMap`` once without bare-string re-expansion.

    Returns ``(checkpoint_map, resolved_quant_spec, target_quant_projection,
    suggested_kv_dtype, suggested_idx_dtype)``.
    """
    from mindie_sim.catalog.checkpoint_reader import (
        lookup_curated_checkpoint,
        quant_spec_from_checkpoint_map,
    )
    from mindie_sim.catalog.quant import resolve_auto_va_quant

    native_dtype = str(model_spec.weight_dtype or model_spec.activation_dtype or "bf16").strip().lower()
    if native_dtype not in {"bf16", "fp16", "fp32"}:
        native_dtype = "bf16"
    canon = _canonical_model_id(model_arg, model_spec)
    spec_meta = dict(getattr(model_spec, "metadata", ()) or ())
    raw_q_str = quant.strip().lower() if isinstance(quant, str) else ""

    if quant is not None:
        explicit = _explicit_checkpoint_quant_map(
            quant, canon=canon, model_name=model_spec.name,
            native_dtype=native_dtype, lookup_curated=lookup_curated_checkpoint,
        )
        if explicit is not None:
            return explicit

    local_map = _local_checkpoint_quant_map(model_arg, model_spec)
    if local_map is not None:
        if raw_q_str in _AUTO_VA_QUANT_TOKENS and local_map.source_format == "unquantized_native":
            proj, auto_kv, auto_idx = resolve_auto_va_quant(model_spec, sku, context_tokens=int(context_tokens))
            return local_map, None, proj, auto_kv, auto_idx
        qspec = quant_spec_from_checkpoint_map(local_map) if local_map.source_format != "unquantized_native" else None
        return local_map, qspec, "", None, None

    for repo_candidate in (spec_meta.get("checkpoint_repo_id"), model_arg if isinstance(model_arg, str) else None):
        if repo_candidate:
            curated = lookup_curated_checkpoint(str(repo_candidate))
            if curated is not None:
                qspec = quant_spec_from_checkpoint_map(curated) if curated.source_format != "unquantized_native" else None
                return curated, qspec, "", None, None

    if raw_q_str in _AUTO_VA_QUANT_TOKENS:
        default_repo = (
            adapter.default_checkpoint_for(canon, hardware_obj)
            if adapter is not None and hardware_obj is not None else None
        )
        curated = lookup_curated_checkpoint(default_repo) if default_repo else None
        proj, auto_kv, auto_idx = resolve_auto_va_quant(model_spec, sku, context_tokens=int(context_tokens))
        if curated is not None and curated.source_format != "unquantized_native":
            return curated, quant_spec_from_checkpoint_map(curated), "", auto_kv, auto_idx
        unquant = CheckpointQuantMap(
            provenance=CheckpointProvenance(
                f"unquantized/{canon or model_spec.name}", "unquantized_native", origin="unquantized_native",
            ),
            native_float_dtype=native_dtype,
        )
        return unquant, None, proj, auto_kv, auto_idx

    if quant is None and adapter is not None and hardware_obj is not None:
        default_repo = adapter.default_checkpoint_for(canon, hardware_obj)
        curated = lookup_curated_checkpoint(default_repo) if default_repo else None
        if curated is not None and curated.source_format != "unquantized_native":
            return curated, quant_spec_from_checkpoint_map(curated), "", None, None

    return (
        synthetic_checkpoint_quant_map(model_spec, canon=canon, native_dtype=native_dtype),
        None, "", None, None,
    )


def _resolve_checkpoint_model_spec(key: str, *, hw_family: str = "a5") -> ModelSpec | None:
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
    meta = resolve_checkpoint_metadata(key, hw_family=hw_family, allow_network=False)
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


def _resolve_model_spec(
    model: str | Path | Mapping[str, Any], *, hw_family: str = "a5",
) -> ModelSpec:
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
        return _resolve_model_spec(candidate, hw_family=hw_family)
    cfg_candidate = CONFIG_DIR / (key if key.endswith(".json") else f"{key}.json")
    if cfg_candidate.is_file():
        return _resolve_model_spec(cfg_candidate, hw_family=hw_family)
    checkpoint_spec = _resolve_checkpoint_model_spec(key, hw_family=hw_family)
    if checkpoint_spec is not None:
        return checkpoint_spec
    try:
        return load_model(key)
    except (KeyError, ValueError, ModelingError) as exc:
        raise ModelingError(f"unknown_model:{key} — {_format_model_suggestions(key)}") from exc


def _validate_explicit_quant_for_hardware(
    spec: ModelSpec, machine: Mapping[str, Any], raw_quant: Any,
) -> None:
    if spec.quant is None:
        return
    sku = str(machine.get("sku") or "")
    arch = _arch_key(sku)
    peaks = machine.get("device", {}).get("compute_peaks", {})
    has_fp8 = any(k in peaks for k in ("fp8", "fp8_e4m3", "mxfp4"))
    if arch not in {"a2", "a3"} and has_fp8:
        return
    fmts = (
        str(spec.quant.weight_format or "").lower(),
        str(spec.quant.activation_format or "").lower(),
        str(spec.quant.expert_weight_format or "").lower(),
    )
    if any(tok in fmt for fmt in fmts for tok in ("fp8", "mxfp4", "fp4")):
        raise ModelingError(
            f"quant_unsupported_on_hardware:{sku or arch}:{raw_quant}"
        )


def _apply_quant_override(
    model_spec: ModelSpec,
    quant: str | Mapping[str, Any] | QuantSpec | None,
    machine: Mapping[str, Any],
) -> ModelSpec:
    if isinstance(quant, QuantSpec):
        base_quant = getattr(model_spec, "quant", None)
        if (
            base_quant is not None
            and str(quant.index_dtype or "bf16") == "bf16"
            and str(quant.index_scale_dtype or "none") in {"", "none"}
            and str(base_quant.index_dtype or "bf16") not in {"", "bf16", "fp16"}
        ):
            quant = replace(
                quant,
                index_dtype=str(base_quant.index_dtype),
                index_scale_dtype=str(base_quant.index_scale_dtype),
            )
        meta = dict(model_spec.metadata)
        if getattr(quant, "layer_quant_overrides", ()) and "layer_quant_overrides" not in meta:
            meta["layer_quant_overrides"] = quant.layer_quant_overrides_dict()
        updated = replace(
            model_spec,
            quant=quant,
            quantization=str(quant.name),
            kv_dtype=str(quant.kv_dtype),
            metadata=tuple(sorted(meta.items())),
        )
        _validate_explicit_quant_for_hardware(updated, machine, quant.name)
        return updated
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
        if raw_key in _CHECKPOINT_NATIVE_QUANT_TOKENS:
            return model_spec
        qkey = _QUANT_ALIASES.get(raw_key, raw_key)
        try:
            updated = apply_model_overrides(model_spec, {"quantization": qkey})
        except (ValueError, ModelingError) as exc:
            raise ModelingError(
                f"invalid_quant_override:{quant} — {_format_quant_suggestions(raw_key)}"
            ) from exc
        _validate_explicit_quant_for_hardware(updated, machine, quant)
        return updated
    if isinstance(quant, Mapping):
        qdict = dict(quant)
        updated = apply_model_overrides(
            model_spec,
            {"quant": qdict, "kv_dtype": str(qdict.get("kv_dtype", model_spec.kv_dtype))},
        )
        _validate_explicit_quant_for_hardware(updated, machine, qdict.get("name", qdict))
        return updated
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
    """Declare a native draft or explicit 1-layer EAGLE/GQA draft when allowed."""
    aux = model.auxiliary
    if aux is not None:
        attached = tuple(d for d in (aux.mtp, aux.dspark) if d is not None)
        if any(bool(getattr(d, "is_synthetic", False)) for d in attached) and not allow_synthetic_gqa:
            raise ModelingError("synthetic_draft_requires_explicit_allow_synthetic_gqa")
    if proposals <= 0:
        return model
    if aux is not None and aux.standalone_draft:
        raise ModelingError("standalone_draft_requires_target_model")
    if aux is not None and (aux.mtp is not None or aux.dspark is not None):
        return model
    if model.attention == "mla":
        return model
    if not allow_synthetic_gqa or model.attention != "gqa":
        raise ModelingError("draft_structure_required_for_mtp_proposal_tokens")
    from mindie_sim.catalog.draft import declare_native_auxiliary_plans
    gqa_plans = [p for p in model.layer_plans if p.attention == "gqa"]
    full_gqa = [p for p in gqa_plans if p.window <= 0]
    ref = full_gqa[-1] if full_gqa else (gqa_plans[-1] if gqa_plans else None)
    num_heads = int(ref.num_heads) if ref is not None else int(model.num_heads)
    num_kv_heads = int(ref.num_kv_heads) if ref is not None else int(model.num_kv_heads)
    head_dim = int(ref.head_dim) if ref is not None else int(model.head_dim)
    qk_norm = bool(ref.qk_norm) if ref is not None else False
    q_gate = bool(ref.q_output_gate) if ref is not None else False
    kv_elems = (
        int(ref.kv_cache_elements)
        if ref is not None and int(ref.kv_cache_elements) > 0
        else 2 * num_kv_heads * head_dim
    )
    inter = int(
        model.intermediate_size
        or (ref.moe_intermediate_size if ref is not None else 0)
        or model.hidden_size
    )
    draft = DraftSpec(
        kind="eagle3", layers=1, hidden_size=model.hidden_size,
        intermediate_size=inter, num_heads=num_heads,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        q_width=(2 if q_gate else 1) * num_heads * head_dim,
        kv_width=num_kv_heads * head_dim,
        attention_kind="gqa",
        kv_cache_elements=kv_elems,
        weight_category="mtp",
        qk_norm=qk_norm,
        q_output_gate=q_gate,
        target_layer_ids=(model.layers - 1,),
        target_shared_embedding=True,
        target_shared_lm_head=True,
        base_logits=True,
        vocab_size=model.vocab_size,
        activation_dtype=model.activation_dtype,
        execution_mode="eagle3_serial",
        is_synthetic=True,
        draft_provenance=SYNTHETIC_EAGLE3_GQA_PROVENANCE,
        source=(("kind", "explicit_synthetic_gqa_mtp"),),
    )
    return declare_native_auxiliary_plans(
        replace(model, auxiliary=replace(aux or AuxiliarySpec(), mtp=draft))
    )


def ensure_native_mla_draft(model_spec: ModelSpec, proposals: int) -> ModelSpec:
    """Attach NextN MLA MTP only when the checkpoint declares ``num_nextn_predict_layers > 0``."""
    aux = model_spec.auxiliary
    facts = dict(model_spec.catalog_facts)
    if aux is not None and aux.mtp is not None:
        src_kind = dict(getattr(aux.mtp, "source", ()) or ()).get("kind")
        if (
            src_kind == "standard_deepseek_nextn_mtp"
            and int(facts.get("num_nextn_predict_layers", 0) or 0) <= 0
        ):
            raise ModelingError("fabricated_mla_nextn_draft_contradicts_checkpoint_zero_nextn")
    needs_draft = aux is None or (aux.mtp is None and aux.dspark is None)
    if proposals <= 0 or not needs_draft:
        return model_spec
    from mindie_sim.catalog.draft import declare_native_auxiliary_plans

    nextn = int(facts.get("num_nextn_predict_layers", 0) or 0)
    moe_ref = next((p for p in model_spec.layer_plans if p.ffn == "moe"), None)
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
    if bool(dict(getattr(model_spec, "catalog_facts", ()) or ()).get("need_fp32_gate", False)):
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
