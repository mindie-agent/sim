"""Layer 2 pluggable RuntimeStackProfile registry and JSON manifest loader.

[Role]
  Map audited inference serving frameworks (built-in ``vllm-ascend`` and
  ``raw_eager``, plus user-supplied JSON manifests) onto universal Layer-1
  ``OptimizationMechanism`` IDs across four explicit states:
  ``raw_eager`` -> ``merged`` -> ``unmerged_pr`` -> ``expected_ideal``.

[Invariants]
  * Built-in stacks are limited to audited ``vllm-ascend`` and framework-neutral
    ``raw_eager``; third-party stacks are never shipped as placeholder profiles.
  * ``raw_eager`` activates zero stack fusions and is never model-gated.
  * ``merged`` activates only mechanisms merged on the stack's main branch.
  * ``unmerged_pr`` activates ``merged`` plus open PRs registered on that stack,
    preserving multi-PR variants without dictionary-key overwrite.
  * ``expected_ideal`` is stack-neutral and activates every structurally
    eligible ``OptimizationMechanism`` for ``(model, hardware, parallel)``.

[Timeouts & Locks]
  Bounded JSON manifest reads (< 50 MiB); pure in-memory resolution, no locks.

[Side-Effect Policy]
  No filesystem writes or network calls.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from mindie_sim.feature_mechanisms import MECHANISM_CATALOG, MechanismContext
from mindie_sim.schema.common import ModelingError
from mindie_sim.va_profiles import VA_RECOMMENDED_PROFILES

STACK_MANIFEST_SCHEMA = "mindie-sim.stack-manifest/1"
MAX_MANIFEST_BYTES = 50 * 1024 * 1024

TRACK_ALIASES: dict[str, str] = {
    "raw_eager": "raw_eager",
    "eager": "raw_eager",
    "merged": "merged",
    "current_merged": "merged",
    "unmerged_pr": "unmerged_pr",
    "current_unmerged_pr": "unmerged_pr",
    "expected_ideal": "expected_ideal",
    "ideal": "expected_ideal",
}

_MECHANISM_LOWERING_SELECTORS: dict[str, tuple[str, ...]] = {
    "add_rmsnorm_quant_fusion": (
        "add_rmsnorm", "add_rmsnorm_quant", "rmsnorm_quant"),
    "mla_prolog_fusion": ("mla_prolog",),
    "attn_res_prefix_fusion": ("attention_residual",),
    "recurrent_decode_native_kernels": ("kda", "gdn_decode"),
    "gdn_chunk64_prefill": ("gdn_prefill",),
}
_MECHANISM_LOWERING_PASSES: dict[str, tuple[str, ...]] = {
    "add_rmsnorm_quant_fusion": ("residual_norm",),
    "mhc_boundary_fusion_overlap": ("mhc_boundary_pr", "mhc_boundary_ideal"),
    "gdn_projection_multistream_overlap": ("gdn_projection_overlap",),
    "kda_in_kernel_l2norm": ("kda_in_kernel_l2norm",),
    "qkv_norm_rope_fusion": ("qkv_norm_rope",),
    "indexer_online_ub_topk": ("indexer_online_topk",),
    "activation_quant_fusion": ("activation_quant",),
    "gmm_swiglu_quant_fusion": ("gmm_swiglu_quant",),
    "moe_tail_residual_fusion": ("moe_tail_residual",),
}


def excluded_lowering_bindings(
    exclude_optimizations: Sequence[str] = (),
) -> tuple[frozenset[str], frozenset[str]]:
    """Resolve excluded mechanism IDs into installed selectors and passes."""
    excluded = frozenset(
        str(value).strip() for value in exclude_optimizations if str(value).strip()
    )
    for mechanism_id in excluded:
        if mechanism_id not in MECHANISM_CATALOG:
            raise ModelingError(f"unknown_optimization_mechanism:{mechanism_id}")
    selectors = frozenset(
        selector for mechanism_id in excluded
        for selector in _MECHANISM_LOWERING_SELECTORS.get(mechanism_id, ())
    )
    passes = frozenset(
        name for mechanism_id in excluded
        for name in _MECHANISM_LOWERING_PASSES.get(mechanism_id, ())
    )
    return selectors, passes


@dataclass(frozen=True)
class StackMechanismBinding:
    """Binds one universal mechanism ID to a stack's merge state and PR ref."""

    mechanism_id: str
    state: str  # "merged" | "unmerged_pr"
    ref: str | None = None
    models: tuple[str, ...] = ()
    archs: tuple[str, ...] = ()
    phases: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.mechanism_id not in MECHANISM_CATALOG:
            raise ModelingError(f"unknown_optimization_mechanism:{self.mechanism_id}")
        if self.state not in {"merged", "unmerged_pr"}:
            raise ModelingError(f"invalid_stack_binding_state:{self.state}")

    def matches(self, model_key: str, arch: str, phase: str | None = None) -> bool:
        if self.models and model_key not in self.models:
            return False
        if self.archs and arch not in self.archs:
            return False
        if phase and self.phases and phase.lower() not in self.phases:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "state": self.state,
            "ref": self.ref,
            "models": list(self.models),
            "archs": list(self.archs),
            "phases": list(self.phases),
            "options": dict(self.options),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RuntimeStackProfile:
    """Complete feature-track declaration for one serving runtime stack."""

    name: str
    repository: str
    description: str = ""
    model_defaults: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    bindings: tuple[StackMechanismBinding, ...] = ()
    supported_models: tuple[str, ...] = ()
    unmerged_pr_models: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STACK_MANIFEST_SCHEMA,
            "name": self.name,
            "repository": self.repository,
            "description": self.description,
            "supported_models": list(self.supported_models),
            "unmerged_pr_models": list(self.unmerged_pr_models),
            "bindings": [b.to_dict() for b in self.bindings],
        }


RUNTIME_STACKS: dict[str, RuntimeStackProfile] = {}


def normalize_feature_track(track: str) -> str:
    key = str(track or "").strip().lower()
    if key not in TRACK_ALIASES:
        raise ModelingError(f"unknown_feature_track:{track}")
    return TRACK_ALIASES[key]


def register_runtime_stack(profile: RuntimeStackProfile) -> RuntimeStackProfile:
    if not isinstance(profile, RuntimeStackProfile) or not profile.name.strip():
        raise ModelingError("invalid_runtime_stack_profile")
    RUNTIME_STACKS[profile.name.strip().lower()] = profile
    return profile


def load_runtime_stack_manifest(source: str | Path | Mapping[str, Any]) -> RuntimeStackProfile:
    """Load a ``RuntimeStackProfile`` from a JSON file or plain Python dict."""
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise ModelingError(f"stack_manifest_not_found:{path}")
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ModelingError(f"stack_manifest_exceeds_50_mib:{path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(source, Mapping):
        raw = dict(source)
    else:
        raise ModelingError("stack_manifest_requires_path_or_mapping")
    if not isinstance(raw, Mapping):
        raise ModelingError("stack_manifest_invalid_mapping")
    schema = raw.get("schema", STACK_MANIFEST_SCHEMA)
    if schema != STACK_MANIFEST_SCHEMA:
        raise ModelingError(f"stack_manifest_schema_unsupported:{schema}")
    bindings = tuple(
        StackMechanismBinding(
            mechanism_id=str(item["mechanism_id"]),
            state=str(item["state"]),
            ref=item.get("ref"),
            models=tuple(str(m) for m in (item.get("models") or ())),
            archs=tuple(str(a).lower() for a in (item.get("archs") or ())),
            phases=tuple(str(p).lower() for p in (item.get("phases") or ())),
            options=dict(item.get("options") or {}),
            notes=str(item.get("notes") or ""),
        )
        for item in (raw.get("bindings") or ())
    )
    return RuntimeStackProfile(
        name=str(raw["name"]).strip(),
        repository=str(raw.get("repository") or ""),
        description=str(raw.get("description") or ""),
        model_defaults=dict(raw.get("model_defaults") or {}),
        bindings=bindings,
        supported_models=tuple(str(m).strip() for m in (raw.get("supported_models") or ())),
        unmerged_pr_models=tuple(str(m).strip() for m in (raw.get("unmerged_pr_models") or ())),
    )


def resolve_runtime_stack(
    stack: str | Path | Mapping[str, Any] | RuntimeStackProfile,
) -> RuntimeStackProfile:
    if isinstance(stack, RuntimeStackProfile):
        return stack
    if isinstance(stack, (Mapping, Path)):
        return load_runtime_stack_manifest(stack)
    key = str(stack or "").strip()
    lower = key.lower()
    if lower in RUNTIME_STACKS:
        return RUNTIME_STACKS[lower]
    candidate = Path(key).expanduser()
    if candidate.is_file():
        return load_runtime_stack_manifest(candidate)
    raise ModelingError(f"unknown_runtime_stack:{stack}")


def _collapse_binding_group(
    mid: str, state: str, group: Sequence[StackMechanismBinding],
) -> dict[str, Any]:
    merged_options: dict[str, Any] = {}
    refs: list[str] = []
    notes: list[str] = []
    for item in group:
        merged_options.update(dict(item.options))
        if item.ref and item.ref not in refs:
            refs.append(item.ref)
        if item.notes and item.notes not in notes:
            notes.append(item.notes)
    primary_ref = refs[0] if len(refs) == 1 else (", ".join(refs) if refs else None)
    entry: dict[str, Any] = {
        "id": mid,
        "state": state,
        "ref": primary_ref,
        "options": merged_options,
        "notes": "; ".join(notes),
    }
    if len(refs) > 1:
        entry["refs"] = refs
    return entry


def resolve_stack_track(
    stack_profile: RuntimeStackProfile,
    *,
    track: str,
    model_key: str,
    ctx: MechanismContext,
    optimizations: Sequence[str] = (),
    exclude_optimizations: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve active mechanisms, PR provenance, and unimplemented gaps."""
    canonical_track = normalize_feature_track(track)
    if stack_profile.supported_models or stack_profile.unmerged_pr_models:
        if canonical_track == "merged" and model_key not in stack_profile.supported_models:
            raise ModelingError(
                f"model_not_supported_by_runtime_stack:{stack_profile.name}:{model_key}:merged")
        if (canonical_track == "unmerged_pr"
                and model_key not in stack_profile.supported_models
                and model_key not in stack_profile.unmerged_pr_models):
            raise ModelingError(
                f"model_not_supported_by_runtime_stack:{stack_profile.name}:{model_key}:unmerged_pr")

    include_ids = tuple(str(x).strip() for x in optimizations if str(x).strip())
    exclude_ids = frozenset(str(x).strip() for x in exclude_optimizations if str(x).strip())
    for mid in (*include_ids, *exclude_ids):
        if mid not in MECHANISM_CATALOG:
            raise ModelingError(f"unknown_optimization_mechanism:{mid}")
    overlap_conflict = set(include_ids) & exclude_ids
    if overlap_conflict:
        raise ModelingError(
            "optimization_include_exclude_conflict:" + ",".join(sorted(overlap_conflict)))

    model_cfg = dict(stack_profile.model_defaults.get(model_key) or {})
    ctx_phase = str((ctx.options or {}).get("phase") or "").strip().lower() or None
    eligible_ids = [
        mech.id for mech in MECHANISM_CATALOG.values()
        if mech.eligible(ctx) and (not ctx_phase or not mech.phases or ctx_phase in mech.phases)
    ]
    eligible_set = set(eligible_ids)
    merged_list: list[StackMechanismBinding] = []
    pr_list: list[StackMechanismBinding] = []
    merged_by_id: dict[str, list[StackMechanismBinding]] = defaultdict(list)
    pr_by_id: dict[str, list[StackMechanismBinding]] = defaultdict(list)
    for binding in stack_profile.bindings:
        if not binding.matches(model_key, ctx.arch, phase=ctx_phase):
            continue
        if binding.mechanism_id not in eligible_set:
            continue
        if binding.state == "merged":
            if binding.mechanism_id == "sequence_parallel" and model_cfg.get("sp") is False:
                continue
            merged_list.append(binding)
            merged_by_id[binding.mechanism_id].append(binding)
        else:
            pr_list.append(binding)
            pr_by_id[binding.mechanism_id].append(binding)

    active: dict[str, dict[str, Any]] = {}
    if canonical_track in {"merged", "unmerged_pr", "expected_ideal"}:
        for mid, group in merged_by_id.items():
            active[mid] = _collapse_binding_group(mid, "merged", group)
    if canonical_track in {"unmerged_pr", "expected_ideal"}:
        for mid, group in pr_by_id.items():
            prev_opts = dict(active[mid]["options"]) if mid in active else {}
            collapsed = _collapse_binding_group(mid, "unmerged_pr", group)
            collapsed["options"] = {**prev_opts, **collapsed["options"]}
            active[mid] = collapsed
    if canonical_track == "expected_ideal":
        for mid in eligible_ids:
            prev = active.get(mid)
            opts = {"unbounded_rows": True} if mid == "mhc_boundary_fusion_overlap" else {}
            if prev is None:
                active[mid] = {
                    "id": mid, "state": "expected_ideal", "ref": None,
                    "options": opts, "notes": "architectural_ideal",
                }
            elif opts:
                active[mid] = {**prev, "options": {**prev["options"], **opts}}

    for mid in include_ids:
        group = pr_by_id.get(mid) or merged_by_id.get(mid)
        if group:
            collapsed = _collapse_binding_group(mid, group[0].state, group)
            collapsed["notes"] = "caller_enabled"
            active[mid] = collapsed
        else:
            active[mid] = {
                "id": mid, "state": "explicit_override", "ref": None,
                "options": {}, "notes": "caller_enabled",
            }
    for mid in exclude_ids:
        active.pop(mid, None)

    unimplemented_gaps = [
        mid for mid in eligible_ids
        if mid not in merged_by_id and mid not in pr_by_id
    ]
    return {
        "stack": stack_profile.name,
        "repository": stack_profile.repository,
        "feature_track": canonical_track,
        "eligible_mechanisms": eligible_ids,
        "active_mechanisms": [active[key] for key in MECHANISM_CATALOG if key in active],
        "merged_mechanisms": [b.to_dict() for b in merged_list],
        "unmerged_pr_mechanisms": [b.to_dict() for b in pr_list],
        "unimplemented_gaps": unimplemented_gaps,
        "explicit_included": list(include_ids),
        "explicit_excluded": sorted(exclude_ids),
    }


def apply_resolved_mechanisms(
    runtime_dict: dict[str, Any],
    resolved_track: Mapping[str, Any],
    base_ctx: MechanismContext,
) -> None:
    """Apply every active mechanism in deterministic catalog order."""
    active = {entry["id"]: entry for entry in resolved_track["active_mechanisms"]}
    for mid, mech in MECHANISM_CATALOG.items():
        entry = active.get(mid)
        if entry is None:
            continue
        ctx = MechanismContext(
            model=base_ctx.model,
            sku=base_ctx.sku,
            arch=base_ctx.arch,
            parallel=base_ctx.parallel,
            caller_explicit=base_ctx.caller_explicit,
            options={**dict(base_ctx.options or {}), **dict(entry.get("options") or {})},
        )
        if mech.eligible(ctx):
            mech.apply(runtime_dict, ctx)


# --- Built-in Stack Profiles ------------------------------------------------

import importlib

try:
    importlib.import_module("mindie_sim.feature_mechanisms_cann93")
except ModuleNotFoundError as _cann93_mech_exc:
    if _cann93_mech_exc.name != "mindie_sim.feature_mechanisms_cann93":
        raise

try:
    _va_ext = importlib.import_module("mindie_sim.va_stack_bindings")
except ModuleNotFoundError as _va_bind_exc:
    if _va_bind_exc.name != "mindie_sim.va_stack_bindings":
        raise
    _va_ext = None

_VA_REPO = "vllm-project/vllm-ascend"
_VA_DOCUMENTED_MODELS = tuple(sorted(VA_RECOMMENDED_PROFILES))
_VA_MODEL_DEFAULTS = {
    k: {
        "quant": dict(v.get("quant") or {}),
        "sp": v.get("sp", True),
        "mlapo": "VLLM_ASCEND_ENABLE_MLAPO" in (v.get("env") or {}),
        "kv_block_size": v.get("kv_block_size"),
        "mtp_proposal_tokens": v.get("mtp_proposal_tokens", 0),
        "graph_capture_mode": v.get("graph_capture_mode", "full"),
    }
    for k, v in VA_RECOMMENDED_PROFILES.items()
}

_FALLBACK_VA_BINDINGS: tuple[StackMechanismBinding, ...] = (
    StackMechanismBinding("aclgraph_decode_capture", "merged", f"{_VA_REPO}@main",
                          models=_VA_DOCUMENTED_MODELS),
    StackMechanismBinding("sequence_parallel", "merged", f"{_VA_REPO}@main",
                          models=_VA_DOCUMENTED_MODELS),
    StackMechanismBinding("add_rmsnorm_quant_fusion", "merged", f"{_VA_REPO}@main"),
    StackMechanismBinding("mla_prolog_fusion", "merged", f"{_VA_REPO}@main"),
    StackMechanismBinding("attn_res_prefix_fusion", "merged", f"{_VA_REPO}@main"),
    StackMechanismBinding("recurrent_decode_native_kernels", "merged", f"{_VA_REPO}@main"),
    StackMechanismBinding(
        "mhc_boundary_fusion_overlap", "unmerged_pr", f"{_VA_REPO}#22142",
        models=("deepseek-v4.1-flash", "deepseek-v4-flash-0731"),
        notes="Fuse DeepSeek V4.1 HcPost+HcPreMix+RMSNorm and overlap HcPreCoefficients (<=72 rows)",
    ),
    StackMechanismBinding(
        "mhc_boundary_fusion_overlap", "unmerged_pr", f"{_VA_REPO}#20824",
        models=("qwen3.8-flash-next",),
        notes="Overlap Qwen4Exp low-rank HC injection on auxiliary stream with Attention/MoE",
    ),
    StackMechanismBinding(
        "gdn_chunk64_prefill", "unmerged_pr", f"{_VA_REPO}#22147",
        archs=("a5",),
        notes="Arch35 FLA Chunk64 GDN prefill composite (+ PR #21693 Qwen3.8-Flash-Next PCP)",
    ),
    StackMechanismBinding(
        "gdn_strided_gate_zero_copy", "unmerged_pr", f"{_VA_REPO}#21466",
        notes="Direct strided BA inputs in GDN gating + head-first QKV storage reuse (#22010, #21467, #21820)",
    ),
    StackMechanismBinding(
        "gdn_projection_multistream_overlap", "unmerged_pr", f"{_VA_REPO}#21037",
        notes="Overlap Qwen GDN BA projection on auxiliary NPU stream during ACLGraph decode",
    ),
    StackMechanismBinding(
        "kda_in_kernel_l2norm", "unmerged_pr", f"{_VA_REPO}#22048",
        models=("glm-5.3-flash", "kimi-k3", "kimi-k3.1"),
        notes="In-kernel Q/K L2 normalization for chunk KDA (+ #21510/#21017 TPxPCP KDA sharding)",
    ),
    StackMechanismBinding(
        "fused_recover_ssm", "unmerged_pr", f"{_VA_REPO}#22045",
        models=("qwen3.8-27b", "qwen3.6-35b-a3b", "qwen3.8-flash-next", "glm-5.3-flash"),
        notes="Triton RecoverSSM compact speculative state records + colocated PCP (#22045, #22146)",
    ),
    StackMechanismBinding(
        "qkv_norm_rope_fusion", "unmerged_pr", f"{_VA_REPO}#22086",
        models=("qwen3.8-flash-next", "qwen3.8-27b"),
        notes="Fuse QSA/GQA QKV RMSNorm + MRoPE head preparation (#22086, #22101)",
    ),
    StackMechanismBinding(
        "unit_offset_rmsnorm_fusion", "unmerged_pr", f"{_VA_REPO}#21449",
        models=("qwen3.8-27b", "gemma-4-31b", "gemma-4-26b-a4b"),
        notes="Pre-fold Gemma/Qwen (1+w) RMSNorm unit-offset weights enabling fused AddRmsNorm",
    ),
    StackMechanismBinding(
        "activation_quant_fusion", "unmerged_pr", f"{_VA_REPO}#21543",
        notes="FX SwiGLU/SiTU graph fusion + dynamic quant fusion (#21543)",
    ),
    StackMechanismBinding(
        "moe_tail_residual_fusion", "unmerged_pr", f"{_VA_REPO}#21999",
        notes="Plain tensor add / fused residual combine for MoE shared+routed tail (#21999)",
    ),
    StackMechanismBinding(
        "shared_expert_multistream_overlap", "unmerged_pr", f"{_VA_REPO}#22128",
        models=("deepseek-v4.1-flash", "deepseek-v4-10t", "glm-5.3-flash",
                "qwen3.8-flash-next", "kimi-k3.1"),
        notes="Multi-stream / MegaMoE shared-expert overlap (#22128, #21095, #21624)",
    ),
)

_VA_ACTIVE_BINDINGS = (
    tuple(_va_ext.VLLM_ASCEND_CANN93_BINDINGS)
    if _va_ext is not None and hasattr(_va_ext, "VLLM_ASCEND_CANN93_BINDINGS")
    else _FALLBACK_VA_BINDINGS
)

register_runtime_stack(RuntimeStackProfile(
    name="vllm-ascend",
    repository=str(getattr(_va_ext, "VLLM_ASCEND_REPOSITORY", _VA_REPO)),
    description="vLLM-Ascend main baseline and audited open performance PRs",
    model_defaults=_VA_MODEL_DEFAULTS,
    bindings=_VA_ACTIVE_BINDINGS,
    supported_models=tuple(getattr(_va_ext, "VLLM_ASCEND_SUPPORTED_MODELS", ())),
    unmerged_pr_models=tuple(getattr(_va_ext, "VLLM_ASCEND_UNMERGED_PR_MODELS", ())),
))

register_runtime_stack(RuntimeStackProfile(
    name="raw_eager",
    repository="eager-baseline",
    description="Unfused eager PyTorch/ATen execution baseline with zero optional fusions",
    bindings=(),
))
