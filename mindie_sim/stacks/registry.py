"""Unified runtime-agnostic ``RuntimeStackAdapter`` registry and manifest loader.

[Role] Own dynamic plugin registration (``register_stack``, ``unregister_stack``,
  ``get_stack``, ``list_stacks``), lazy ``importlib`` discovery of installed
  adapters (including optional ``vllm-ascend`` and built-in ``ideal-target``,
  ``raw-eager``, ``diffusion-native``), domain-aware stack routing
  (``resolve_stack_for_scenario``), and custom JSON manifest loading
  (``load_manifest_stack`` / ``ManifestStackAdapter``).
[Invariants] Zero hard static dependencies on optional plugins (such as
  ``vllm-ascend``). Always resolves ``workload_domain`` and ``work_unit``
  before choosing a default stack. Diffusion/DiT NEVER falls back to an
  LLM-only stack; forcing an incompatible stack raises ``ModelingError``
  (``stack_domain_mismatch:...``). Never imports ``export/`` or ``compose/``,
  and maintains zero SCC cycles inside ``mindie_sim/stacks/``.
[Timeouts & Locks] Bounded manifest reads (< 50 MiB); re-entrant in-process
  registration lock for thread safety; zero network or subprocess I/O.
[Side-Effect Policy] Read-only over filesystem manifests; mutates only the
  in-memory registry table on explicit ``register_stack`` / ``unregister_stack``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import re
import threading
from typing import Any, Literal

from mindie_sim.catalog.checkpoint_reader import resolve_checkpoint_tensor_tag
from mindie_sim.schema.checkpoint import CheckpointQuantMap, TensorQuantRule
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import (
    KERNEL_READINESS_VALUES,
    KernelReadiness,
    TensorPrecisionABI,
)
from mindie_sim.schema.parallel import ParallelSpec, local_kv_heads
from mindie_sim.schema.scenario import (
    DOMAIN_WORK_UNITS,
    resolve_default_stack_for_domain,
)
from mindie_sim.schema.stack import (
    CAPTURE_MODES,
    FRAMEWORK_REWRITE_KINDS,
    REWRITE_TRACKS,
    STACK_ADAPTER_PROTOCOL_VERSION,
    STACK_TRACKS,
    WORK_UNIT_KINDS,
    WORKLOAD_DOMAINS,
    CachePlacementRule,
    CachePlaneABI,
    CacheStateABI,
    CaptureBucketRule,
    CaptureMode,
    FrameworkRewriteKind,
    IterationDecision,
    IterationSchedulerState,
    LaunchConfigCodec,
    ResolvedStackOptions,
    RewriteTrack,
    RuntimeQuantPlan,
    RuntimeStackAdapter,
    SchedulerDone,
    SchedulingContract,
    StackCapabilitySpec,
    StackLoweringPlan,
    StackTrack,
    StarsSubmitPolicy,
    WaitForEvent,
    WorkUnitKind,
    WorkloadDomain,
    resolve_checkpoint_cache_abis,
    resolve_checkpoint_tensor_abi,
)
from mindie_sim.schema.workload import IterationPlan, RequestView

STACK_MANIFEST_SCHEMA = "mindie-sim.stack-manifest/1"
STACK_ENTRY_POINT_GROUP = "mindie_sim.stacks"
MAX_MANIFEST_BYTES = 50 * 1024 * 1024
DEFAULT_KV_BLOCK_SIZE = 128
DEFAULT_LATENT_PATCHES = 4096
DEFAULT_CFG_BRANCHES = 2
DEFAULT_TEXT_ENCODE_TOKENS = 256
DEFAULT_FULL_CAPTURE_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_PIECEWISE_BUCKETS: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
EXPERT_FULL_ENUM_LIMIT = 16
MX_BLOCK_ELEMENTS = 32
FP8_BLOCK_ELEMENTS = 128
INT4_GROUP_ELEMENTS = 128
FP32_SCALE_BYTES = 4
FP16_SCALE_BYTES = 2
E8M0_SCALE_BYTES = 1
KV_CACHE_PLANES = 2
DIFFUSION_PHASES = frozenset({"denoise_step", "vae_decode", "text_encode"})
_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")

DEFAULT_DOMAIN_WORK_UNITS: Mapping[WorkloadDomain, WorkUnitKind] = {
    "llm": "prefill_decode_step",
    "vlm": "prefill_decode_step",
    "diffusion": "denoise_timestep",
    "custom": "custom_step",
}
_DOMAIN_DEFAULT_PHASES: Mapping[WorkloadDomain, tuple[str, ...]] = {
    "llm": ("prefill", "decode", "mixed"),
    "vlm": ("prefill", "decode", "mixed"),
    "diffusion": ("denoise_step", "vae_decode", "text_encode"),
    "custom": ("prefill", "decode", "mixed", "denoise_step"),
}
_DOMAIN_PREFERRED_STACKS: Mapping[WorkloadDomain, tuple[str, ...]] = {
    "llm": ("vllm-ascend",),
    "vlm": ("vllm-ascend",),
    "diffusion": ("diffusion-native",),
    "custom": (),
}
_BASELINE_FALLBACK_ORDER: tuple[str, ...] = ("ideal-target", "raw-eager")
_BASELINE_STACK_IDS = frozenset(_BASELINE_FALLBACK_ORDER)

# (stack_id, module_path, class_name, optional)
_BUILTIN_PLUGIN_SPECS: tuple[tuple[str, str, str, bool], ...] = (
    ("vllm-ascend", "mindie_sim.stacks.vllm_ascend.adapter", "VllmAscendStackAdapter", True),
    ("ideal-target", "mindie_sim.stacks.ideal_target.adapter", "IdealTargetStackAdapter", False),
    ("raw-eager", "mindie_sim.stacks.raw_eager.adapter", "RawEagerStackAdapter", False),
    ("diffusion-native", "mindie_sim.stacks.diffusion_native.adapter", "DiffusionNativeStackAdapter", False),
)

_ASCEND_EXACT_TOKENS = frozenset({
    "ascend", "a2", "a3", "a5", "arch20", "arch22", "arch35", "dav_3510",
})
_ASCEND_HW_PREFIXES = (
    "ascend", "dav_", "davinci", "atlas", "cann", "910b", "910c", "910_", "950",
)
_NVIDIA_EXACT_TOKENS = frozenset({
    "nvidia", "cuda", "hopper", "blackwell", "ampere", "ada",
})
_NVIDIA_HW_PREFIXES = (
    "nvidia", "cuda", "h100", "h200", "h800", "b100", "b200", "gb200", "a100", "sm_",
)
_STANDARD_LAYER_ROLES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_DIT_LAYER_ROLES: tuple[str, ...] = (
    "attn.qkv_proj",
    "attn.o_proj",
    "ffn.fc1",
    "ffn.fc2",
    "adaln.modulation",
)
_GQA_ATTN_ROLES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
_MLA_Q_LORA_ATTN_ROLES: tuple[str, ...] = (
    "self_attn.q_a_proj",
    "self_attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_b_proj",
    "self_attn.o_proj",
)
_MLA_DIRECT_ATTN_ROLES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.kv_a_proj_with_mqa",
    "self_attn.kv_b_proj",
    "self_attn.o_proj",
)
_DENSE_MLP_ROLES: tuple[str, ...] = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_SHARED_EXPERT_ROLES: tuple[str, ...] = (
    "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down_proj",
)
_EXPERT_PROJECTION_SUFFIXES: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")


def normalize_hardware_backend(hardware_family: str | None) -> str | None:
    """Normalize a hardware family / SKU string to a canonical backend label."""
    if hardware_family is None:
        return None
    raw = str(hardware_family).strip().lower()
    if not raw:
        return None
    if raw in {"ascend", "nvidia", "generic"}:
        return raw
    if raw == "custom":
        return "generic"
    tokens = {tok for part in raw.replace("-", "_").split() for tok in part.split("_") if tok}
    if raw in _ASCEND_EXACT_TOKENS or (tokens & _ASCEND_EXACT_TOKENS) or raw.startswith(_ASCEND_HW_PREFIXES):
        return "ascend"
    if raw in _NVIDIA_EXACT_TOKENS or (tokens & _NVIDIA_EXACT_TOKENS) or raw.startswith(_NVIDIA_HW_PREFIXES):
        return "nvidia"
    return raw


def _canonical_alias(stack_id: str) -> str:
    return stack_id.strip().lower().replace("_", "-")


def _validate_adapter(adapter: Any) -> StackCapabilitySpec:
    if not isinstance(adapter, RuntimeStackAdapter):
        raise ModelingError(
            f"stack_adapter_protocol_invalid:{type(adapter).__name__}:"
            "must implement RuntimeStackAdapter"
        )
    cap = adapter.capabilities()
    if not isinstance(cap, StackCapabilitySpec):
        raise ModelingError(
            f"stack_adapter_capabilities_invalid:{type(adapter).__name__}"
        )
    return cap


_manifest_abi_for_tag = resolve_checkpoint_tensor_abi


def _resolve_tag_and_rule(
    checkpoint: CheckpointQuantMap,
    tensor_path: str,
    *,
    scope: str = "target",
    source_native_layer_id: int | None = None,
    expert_id: int | None = None,
) -> tuple[str, TensorQuantRule | None]:
    raw = str(tensor_path or "").strip()
    mod_path = raw[:-len(".weight")] if raw.endswith(".weight") else raw
    exc_or_ign = checkpoint.resolve_exception_or_ignored(
        mod_path, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
    )
    if exc_or_ign is not None:
        return exc_or_ign, None
    for rule in checkpoint.rules:
        if rule.matches(
            mod_path, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
        ):
            return rule.raw_tag, rule
    if mod_path != raw:
        exc_raw = checkpoint.resolve_exception_or_ignored(
            raw, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
        )
        if exc_raw is not None:
            return exc_raw, None
        for rule in checkpoint.rules:
            if rule.matches(
                raw, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
            ):
                return rule.raw_tag, rule
    if checkpoint.source_format == "unquantized_native":
        return "FLOAT", None
    if checkpoint.source_format == "hf_quantization_config":
        parts = set(mod_path.lower().split("."))
        if "experts" in parts:
            for rule in checkpoint.rules:
                if rule.pattern == "*.experts.*":
                    return rule.raw_tag, rule
        for rule in checkpoint.rules:
            if rule.pattern == "*":
                return rule.raw_tag, rule
    tag = resolve_checkpoint_tensor_tag(
        checkpoint, tensor_path, scope=scope, source_native_layer_id=source_native_layer_id, expert_id=expert_id,  # type: ignore[arg-type]
    )
    return (tag if tag is not None else (checkpoint.model_quant_type or "FLOAT")), None


def _expert_boundary_ids(checkpoint: CheckpointQuantMap, num_experts: int) -> tuple[int, ...]:
    count = max(1, int(num_experts))
    if count <= EXPERT_FULL_ENUM_LIMIT:
        return tuple(range(count))
    ids: set[int] = {0, 1, count - 1}
    for exc in checkpoint.layer_exceptions:
        for val in exc.expert_indices:
            if 0 <= val < count:
                ids.add(val)
        for m in _EXPERT_ID_RE.findall(exc.pattern):
            val = int(m)
            if 0 <= val < count:
                ids.add(val)
    for entry in checkpoint.ignored_modules:
        for m in _EXPERT_ID_RE.findall(entry):
            val = int(m)
            if 0 <= val < count:
                ids.add(val)
    for rule in checkpoint.rules:
        for m in _EXPERT_ID_RE.findall(rule.pattern):
            val = int(m)
            if 0 <= val < count:
                ids.add(val)
        if rule.presence is not None and rule.presence.expert_indices:
            present = [v for v in rule.presence.expert_indices if 0 <= v < count]
            if present:
                ids.add(min(present))
                ids.add(max(present))
    return tuple(sorted(ids))


def _layer_prefix_for(checkpoint: CheckpointQuantMap, *, is_diffusion: bool) -> str:
    if is_diffusion:
        return "transformer_blocks"
    for candidate in ("model.language_model.layers", "language_model.model.layers", "model.layers"):
        marker = candidate + "."
        if any(r.pattern.startswith(marker) for r in checkpoint.rules) or any(
            e.pattern.startswith(marker) for e in checkpoint.layer_exceptions
        ):
            return candidate
    return "model.layers"


def _layer_roles_for(
    plan: LayerPlan, checkpoint: CheckpointQuantMap, *, is_diffusion: bool,
) -> list[tuple[str, int | None]]:
    if is_diffusion:
        return [(role, None) for role in _DIT_LAYER_ROLES]
    attn_roles = (
        (_MLA_Q_LORA_ATTN_ROLES if int(plan.q_lora_rank or 0) > 0 else _MLA_DIRECT_ATTN_ROLES)
        if str(plan.attention or "").lower() == "mla"
        else _GQA_ATTN_ROLES
    )
    roles: list[tuple[str, int | None]] = [(role, None) for role in attn_roles]
    if str(plan.ffn or "").lower() == "moe" and int(plan.num_experts or 0) > 0:
        for eid in _expert_boundary_ids(checkpoint, int(plan.num_experts)):
            for suffix in _EXPERT_PROJECTION_SUFFIXES:
                roles.append((f"mlp.experts.{eid}.{suffix}", eid))
        if int(plan.num_shared_experts or 0) > 0:
            roles.extend((role, None) for role in _SHARED_EXPERT_ROLES)
    else:
        roles.extend((role, None) for role in _DENSE_MLP_ROLES)
    return roles


def _synthesize_layer_plans(model: ModelSpec) -> tuple[LayerPlan, ...]:
    if model.layer_plans:
        return model.layer_plans
    mla = model.mla
    return tuple(
        LayerPlan(
            index=idx,
            native_layer_id=idx,
            attention=model.attention,
            ffn=model.ffn,
            num_heads=model.num_heads,
            num_kv_heads=model.num_kv_heads or model.num_heads,
            head_dim=model.head_dim,
            q_lora_rank=mla.q_lora_rank if mla is not None else 0,
            kv_lora_rank=mla.kv_lora_rank if mla is not None else 0,
            qk_nope_head_dim=mla.qk_nope_head_dim if mla is not None else 0,
            qk_rope_head_dim=mla.qk_rope_head_dim if mla is not None else 0,
            v_head_dim=mla.v_head_dim if mla is not None else 0,
            num_experts=model.num_experts,
            top_k=model.top_k,
            num_shared_experts=model.num_shared_experts,
        )
        for idx in range(model.layers)
    )


def _stars_replay_mode(
    capture_rule: CaptureBucketRule | None, fallback_mode: CaptureMode,
) -> Literal["eager", "captured_segment", "full_graph"]:
    mode = capture_rule.mode if capture_rule is not None else fallback_mode
    has_full = bool(capture_rule.full_keys) if capture_rule is not None else (mode in {"full", "full_decode_only", "full_and_piecewise"})
    has_pw = bool(capture_rule.piecewise_keys) if capture_rule is not None else (mode in {"piecewise", "full_and_piecewise"})
    if mode == "none":
        return "eager"
    if mode in {"full", "full_decode_only"}:
        return "full_graph" if has_full else "eager"
    if mode in {"piecewise", "full_and_piecewise"}:
        if has_pw:
            return "captured_segment"
        if mode == "full_and_piecewise" and has_full:
            return "full_graph"
        return "eager"
    return "eager"


def _positive_option_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _parse_manifest_capture_mode(raw: Any) -> tuple[CaptureMode, str]:
    text = str(raw if raw is not None else "none").strip().lower()
    if text == "eager":
        return "none", "eager"
    if text not in CAPTURE_MODES:
        raise ModelingError(f"stack_manifest_capture_mode_invalid:{raw!r}")
    return text, ("eager" if text == "none" else text)  # type: ignore[return-value]


def _parse_manifest_recipe_bindings(
    raw: Any,
) -> tuple[tuple[str, str, tuple[tuple[str, Any], ...]], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ModelingError("stack_manifest_recipe_bindings_invalid")
    out: list[tuple[str, str, tuple[tuple[str, Any], ...]]] = []
    for entry in raw:
        if isinstance(entry, Mapping):
            selector = str(entry.get("selector", "")).strip()
            recipe_id = str(entry.get("recipe_id", "")).strip()
            opts = entry.get("options", ())
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) in {2, 3}:
            selector = str(entry[0]).strip()
            recipe_id = str(entry[1]).strip()
            opts = entry[2] if len(entry) == 3 else ()
        else:
            raise ModelingError(f"stack_manifest_recipe_binding_entry_invalid:{entry!r}")
        if not selector or not recipe_id:
            raise ModelingError("stack_manifest_recipe_binding_empty_fields")
        if isinstance(opts, Mapping):
            frozen_opts = tuple(sorted((str(k), v) for k, v in opts.items()))
        else:
            frozen_opts = tuple((str(k), v) for k, v in opts)
        out.append((selector, recipe_id, frozen_opts))
    return tuple(out)


def _validate_manifest_payload(data: Any, source_label: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ModelingError(f"stack_manifest_root_not_object:{source_label}")
    schema = data.get("schema")
    if schema != STACK_MANIFEST_SCHEMA:
        raise ModelingError(
            f"stack_manifest_schema_invalid:{source_label}:expected={STACK_MANIFEST_SCHEMA}:got={schema!r}"
        )
    stack_id = str(data.get("stack_id") or data.get("name") or "").strip()
    if not stack_id:
        raise ModelingError(f"stack_manifest_stack_id_required:{source_label}")
    manifest_rev = str(data.get("manifest_revision", "")).strip()
    if not manifest_rev:
        raise ModelingError(f"stack_manifest_revision_required:{source_label}")
    raw_domains = data.get("supported_domains")
    if not isinstance(raw_domains, Sequence) or isinstance(raw_domains, (str, bytes)) or not raw_domains:
        raise ModelingError(f"stack_manifest_domains_required:{source_label}")
    domains: list[WorkloadDomain] = []
    for dom in raw_domains:
        d_str = str(dom).strip().lower()
        if d_str not in WORKLOAD_DOMAINS:
            raise ModelingError(f"stack_manifest_domain_invalid:{source_label}:{dom!r}")
        domains.append(d_str)  # type: ignore[arg-type]
    raw_units = data.get("supported_work_units")
    if raw_units is None:
        inferred: list[WorkUnitKind] = []
        for dom in domains:
            for wu in sorted(DOMAIN_WORK_UNITS[dom]):
                if wu not in inferred:
                    inferred.append(wu)
        work_units = tuple(inferred)
    else:
        if not isinstance(raw_units, Sequence) or isinstance(raw_units, (str, bytes)) or not raw_units:
            raise ModelingError(f"stack_manifest_work_units_invalid:{source_label}")
        parsed_units: list[WorkUnitKind] = []
        for wu in raw_units:
            w_str = str(wu).strip().lower()
            if w_str not in WORK_UNIT_KINDS:
                raise ModelingError(f"stack_manifest_work_unit_invalid:{source_label}:{wu!r}")
            parsed_units.append(w_str)  # type: ignore[arg-type]
        work_units = tuple(parsed_units)
    raw_hw = data.get("supported_hardware_backends", ("ascend", "nvidia", "generic"))
    if not isinstance(raw_hw, Sequence) or isinstance(raw_hw, (str, bytes)) or not raw_hw:
        raise ModelingError(f"stack_manifest_hardware_backends_invalid:{source_label}")
    raw_phases = data.get("supported_phases")
    if raw_phases is None:
        phases_list: list[str] = []
        for dom in domains:
            for ph in _DOMAIN_DEFAULT_PHASES[dom]:
                if ph not in phases_list:
                    phases_list.append(ph)
        phases = tuple(phases_list)
    else:
        if not isinstance(raw_phases, Sequence) or isinstance(raw_phases, (str, bytes)) or not raw_phases:
            raise ModelingError(f"stack_manifest_phases_invalid:{source_label}")
        phases = tuple(str(p).strip() for p in raw_phases)
    raw_ckpts = data.get(
        "supported_checkpoint_formats",
        ("modelslim_description", "hf_quantization_config", "unquantized_native", "explicit_synthetic_preset"),
    )
    raw_tracks = data.get("supported_tracks", ("shipping",))
    if not isinstance(raw_tracks, Sequence) or isinstance(raw_tracks, (str, bytes)) or not raw_tracks:
        raise ModelingError(f"stack_manifest_tracks_invalid:{source_label}")
    tracks: list[StackTrack] = []
    for tr in raw_tracks:
        t_str = str(tr).strip().lower()
        if t_str not in STACK_TRACKS:
            raise ModelingError(f"stack_manifest_track_invalid:{source_label}:{tr!r}")
        tracks.append(t_str)  # type: ignore[arg-type]
    raw_rw_tracks = data.get("allowed_rewrite_tracks", ("config",))
    rw_tracks: list[RewriteTrack] = []
    for item in raw_rw_tracks:
        r_str = str(item).strip().lower()
        if r_str not in REWRITE_TRACKS:
            raise ModelingError(f"stack_manifest_rewrite_track_invalid:{source_label}:{item!r}")
        rw_tracks.append(r_str)  # type: ignore[arg-type]
    raw_rw_classes = data.get("allowed_rewrite_classes", ("shipping",))
    rw_classes: set[KernelReadiness] = set()
    for item in raw_rw_classes:
        c_str = str(item).strip().lower()
        if c_str not in KERNEL_READINESS_VALUES:
            raise ModelingError(f"stack_manifest_rewrite_class_invalid:{source_label}:{item!r}")
        rw_classes.add(c_str)  # type: ignore[arg-type]
    if "kernel_new" in rw_tracks:
        rw_classes.add("synthesized_target")
    if "kernel_tune" in rw_tracks:
        rw_classes.add("tiler_extension")
    raw_fw = data.get("allowed_framework_rewrites", ())
    fw_rewrites: list[FrameworkRewriteKind] = []
    for item in raw_fw:
        f_str = str(item).strip().lower()
        if f_str not in FRAMEWORK_REWRITE_KINDS:
            raise ModelingError(f"stack_manifest_framework_rewrite_invalid:{source_label}:{item!r}")
        fw_rewrites.append(f_str)  # type: ignore[arg-type]
    cap_mode, cap_label = _parse_manifest_capture_mode(data.get("capture_mode", "none"))
    default_ckpts_raw = data.get("default_checkpoints", {})
    default_ckpts = (
        {str(k): str(v) for k, v in default_ckpts_raw.items()}
        if isinstance(default_ckpts_raw, Mapping)
        else {}
    )
    return {
        "schema": schema,
        "protocol_version": str(data.get("protocol_version", STACK_ADAPTER_PROTOCOL_VERSION)),
        "stack_id": stack_id,
        "manifest_revision": manifest_rev,
        "description": str(data.get("description", "")),
        "supported_domains": tuple(domains),
        "supported_work_units": work_units,
        "supported_hardware_backends": tuple(str(h).strip().lower() for h in raw_hw),
        "supported_phases": phases,
        "supported_checkpoint_formats": tuple(str(c).strip() for c in raw_ckpts),
        "supported_tracks": tuple(tracks),
        "allowed_rewrite_tracks": tuple(rw_tracks),
        "allowed_rewrite_classes": frozenset(rw_classes) or frozenset({"shipping"}),
        "allowed_framework_rewrites": tuple(fw_rewrites),
        "supported_recipe_ids": frozenset(str(r).strip() for r in data.get("supported_recipe_ids", ())),
        "lowering_passes": tuple(str(p).strip() for p in data.get("lowering_passes", ())),
        "recipe_bindings": _parse_manifest_recipe_bindings(data.get("recipe_bindings")),
        "capture_mode": cap_mode,
        "capture_mode_label": cap_label,
        "has_launch_codec": False,
        "default_checkpoints": default_ckpts,
        "source_label": source_label,
    }


class ManifestStackAdapter:
    """Declarative ``RuntimeStackAdapter`` instantiated from a JSON manifest file."""

    def __init__(self, manifest: Mapping[str, Any], *, source_label: str = "manifest.json") -> None:
        validated = _validate_manifest_payload(manifest, source_label)
        self._spec = validated
        self._capabilities = StackCapabilitySpec(
            stack_id=validated["stack_id"],
            manifest_revision=validated["manifest_revision"],
            supported_domains=validated["supported_domains"],
            supported_work_units=validated["supported_work_units"],
            supported_hardware_backends=validated["supported_hardware_backends"],
            supported_phases=validated["supported_phases"],
            supported_checkpoint_formats=validated["supported_checkpoint_formats"],
            supported_tracks=validated["supported_tracks"],
            supported_recipe_ids=validated["supported_recipe_ids"],
            has_launch_codec=False,
            protocol_version=validated["protocol_version"],
        )

    def capabilities(self) -> StackCapabilitySpec:
        return self._capabilities

    def default_checkpoint_for(
        self, canonical_model: str, hardware: Hardware
    ) -> str | None:
        del hardware
        return self._spec["default_checkpoints"].get(canonical_model)

    def resolve_options(
        self,
        model: ModelSpec,
        hardware: Hardware,
        parallel: ParallelSpec,
        *,
        role: str,
        track: StackTrack,
        requested_options: Mapping[str, Any],
        field_origins: Mapping[str, str],
    ) -> ResolvedStackOptions:
        del model, hardware, parallel, field_origins
        if track not in self._capabilities.supported_tracks:
            raise ModelingError(
                f"stack_track_unsupported:stack={self._capabilities.stack_id}:track={track}"
            )
        req = dict(requested_options)
        cap_mode: CaptureMode = "none" if track == "eager" else self._spec["capture_mode"]
        full_keys = () if cap_mode in {"none", "piecewise"} else DEFAULT_FULL_CAPTURE_BUCKETS
        pw_keys = () if cap_mode in {"none", "full", "full_decode_only"} else DEFAULT_PIECEWISE_BUCKETS
        capture_rule = CaptureBucketRule(
            mode=cap_mode,
            full_keys=full_keys,
            piecewise_keys=pw_keys,
            fallback_mode="piecewise" if cap_mode == "full_and_piecewise" else "eager",
        )
        raw_kv_block = req.get("kv_block_size")
        kv_block_size = (
            _positive_option_int(raw_kv_block, "manifest_kv_block_size_invalid")
            if raw_kv_block is not None
            else DEFAULT_KV_BLOCK_SIZE
        )
        dispositions = tuple((k, "accepted", "manifest_option_applied") for k in sorted(req))
        effective: dict[str, Any] = {
            "capture_mode": "eager" if cap_mode == "none" else self._spec["capture_mode_label"],
            "kv_block_size": kv_block_size,
            "role": role,
            "manifest_source": self._spec["source_label"],
        }
        if any(k in req for k in ("latent_patches", "cfg_branches", "text_encode_tokens")):
            patches = _positive_option_int(req.get("latent_patches", DEFAULT_LATENT_PATCHES), "diffusion_latent_patches_invalid")
            cfg = _positive_option_int(req.get("cfg_branches", DEFAULT_CFG_BRANCHES), "diffusion_cfg_branches_invalid")
            text_toks = _positive_option_int(req.get("text_encode_tokens", DEFAULT_TEXT_ENCODE_TOKENS), "diffusion_text_tokens_invalid")
            effective["latent_patches"] = patches
            effective["cfg_branches"] = cfg
            effective["text_encode_tokens"] = text_toks
            effective["tokens_per_denoise_step"] = patches * cfg
        return ResolvedStackOptions(
            stack_id=self._capabilities.stack_id,
            track=track,
            requested_mechanisms=tuple(sorted(req.items())),
            effective_mechanisms=tuple(sorted(effective.items())),
            dispositions=dispositions,
            capture_rule=capture_rule,
            graph_capture_mode=cap_mode,
            graph_capture_keys=full_keys,
            piecewise_capture_keys=pw_keys,
        )

    def resolve_quant_instantiation(
        self,
        model: ModelSpec,
        checkpoint: CheckpointQuantMap,
        hardware: Hardware,
        parallel: ParallelSpec,
        options: ResolvedStackOptions,
        *,
        role: str,
        phase: str,
    ) -> RuntimeQuantPlan:
        del hardware
        if phase not in self._capabilities.supported_phases:
            raise ModelingError(
                f"stack_phase_unsupported:stack={self._capabilities.stack_id}:phase={phase}"
            )
        native_dtype = (checkpoint.native_float_dtype or model.activation_dtype or "bf16").strip().lower()
        is_diffusion = phase in DIFFUSION_PHASES
        prefix = _layer_prefix_for(checkpoint, is_diffusion=is_diffusion)
        port_abis: list[tuple[tuple[str, int, str], TensorPrecisionABI]] = []
        dispositions: list[tuple[str, str, str]] = []
        layers = _synthesize_layer_plans(model)
        for plan in layers:
            native_id = plan.native_layer_id if plan.native_layer_id >= 0 else plan.index
            for logical_role, expert_id in _layer_roles_for(plan, checkpoint, is_diffusion=is_diffusion):
                tensor_path = f"{prefix}.{native_id}.{logical_role}"
                tag, matched = _resolve_tag_and_rule(
                    checkpoint,
                    tensor_path,
                    scope="target",
                    source_native_layer_id=native_id,
                    expert_id=expert_id,
                )
                abi = _manifest_abi_for_tag(tag, native_dtype, matched)
                port_abis.append((("target", plan.index, logical_role), abi))
                dispositions.append((tensor_path, tag, "manifest_checkpoint_precision_honored"))
        for rule in checkpoint.rules:
            _manifest_abi_for_tag(rule.raw_tag, native_dtype, rule)
        for exc in checkpoint.layer_exceptions:
            _manifest_abi_for_tag(exc.raw_tag, native_dtype)
        if is_diffusion:
            kv_abis: tuple[CacheStateABI, ...] = ()
            idx_abis: tuple[CacheStateABI, ...] = ()
            rec_abis: tuple[CacheStateABI, ...] = ()
        else:
            eff = dict(options.effective_mechanisms)
            kv_block = int(eff.get("kv_block_size") or DEFAULT_KV_BLOCK_SIZE)
            kv_abis, idx_abis, rec_abis = resolve_checkpoint_cache_abis(
                layers, checkpoint, parallel, native_dtype,
                role=role, phase=phase, kv_block_size=kv_block,
            )
        return RuntimeQuantPlan(
            checkpoint_digest=checkpoint.digest(),
            stack_id=self._capabilities.stack_id,
            port_abis=tuple(port_abis),
            packed_groups=(),
            kv_cache_abi=kv_abis,
            index_cache_abi=idx_abis,
            recurrent_state_abi=rec_abis,
            prepared_weights=(),
            dispositions=tuple(dispositions),
            provenance=checkpoint.provenance.origin,
        )

    def select_lowering_plan(
        self,
        model: ModelSpec,
        quant_plan: RuntimeQuantPlan,
        hardware: Hardware,
        parallel: ParallelSpec,
        options: ResolvedStackOptions,
    ) -> StackLoweringPlan:
        del model, quant_plan, hardware, parallel
        if options.track == "eager":
            allowed_tracks: tuple[RewriteTrack, ...] = ()
            allowed_classes: frozenset[KernelReadiness] = frozenset({"shipping"})
            allowed_fw: tuple[FrameworkRewriteKind, ...] = ()
            passes: tuple[str, ...] = ()
            bindings: tuple[tuple[str, str, tuple[tuple[str, Any], ...]], ...] = ()
        else:
            allowed_tracks = self._spec["allowed_rewrite_tracks"]
            allowed_classes = self._spec["allowed_rewrite_classes"]
            allowed_fw = self._spec["allowed_framework_rewrites"]
            passes = self._spec["lowering_passes"]
            bindings = self._spec["recipe_bindings"]
        scheduling = SchedulingContract(
            capture_bucket_rule=options.capture_rule,
            stars_submit_policy=(
                StarsSubmitPolicy(
                    die_id=0,
                    stream_id="main",
                    owner_kind="aic",
                    replay_mode=_stars_replay_mode(options.capture_rule, options.graph_capture_mode),
                ),
            ),
            chunked_prefill_enabled=not set(self._capabilities.supported_domains).issubset({"diffusion"}),
        )
        return StackLoweringPlan(
            stack_id=self._capabilities.stack_id,
            track=options.track,
            recipe_bindings=bindings,
            lowering_passes=passes,
            allowed_rewrite_classes=allowed_classes,
            allowed_rewrite_tracks=allowed_tracks,
            allowed_framework_rewrites=allowed_fw,
            scheduling=scheduling,
            binding_origins=(("stack", "stack_profile"),),
        )

    def plan_iteration(
        self,
        model: ModelSpec,
        state: IterationSchedulerState,
        options: ResolvedStackOptions,
        *,
        phase: str,
    ) -> IterationDecision:
        del model
        if phase not in self._capabilities.supported_phases:
            raise ModelingError(
                f"stack_phase_unsupported:stack={self._capabilities.stack_id}:phase={phase}"
            )
        if not state.ready_request_ids:
            if state.in_flight_completions:
                next_ms = min(ts for _, ts in state.in_flight_completions)
                return WaitForEvent(reason="completion", until_ms=next_ms)
            return SchedulerDone(completed_requests=len(state.request_progress))
        eff = dict(options.effective_mechanisms)
        priorities = dict(state.request_priorities)
        progress = {rid: (ctx, rem) for rid, ctx, rem in state.request_progress}
        views: list[RequestView] = []
        batch_budget = state.max_num_batched_tokens
        used_tokens = 0
        if phase in DIFFUSION_PHASES:
            if phase == "denoise_step":
                q_len = int(eff.get("tokens_per_denoise_step", DEFAULT_LATENT_PATCHES * DEFAULT_CFG_BRANCHES))
                exec_mode = "denoise_timestep"
            elif phase == "text_encode":
                q_len = int(eff.get("text_encode_tokens", DEFAULT_TEXT_ENCODE_TOKENS))
                exec_mode = "text_encode"
            else:
                q_len = int(eff.get("latent_patches", DEFAULT_LATENT_PATCHES))
                exec_mode = "vae_decode"
            for rid in state.ready_request_ids:
                if len(views) >= state.max_num_seqs:
                    break
                if used_tokens + q_len > batch_budget:
                    continue
                views.append(
                    RequestView(
                        request_id=rid,
                        q_len=q_len,
                        context=0,
                        samples=False,
                        phase=phase,
                        semantic_phase=phase,
                        execution_mode=exec_mode,
                        priority=priorities.get(rid, 0),
                        draft_verify=False,
                        draft_prepare=False,
                    )
                )
                used_tokens += q_len
        else:
            input_budget = (
                min(batch_budget, state.max_num_input_tokens)
                if state.max_num_input_tokens is not None
                else batch_budget
            )
            used_input_tokens = 0
            for rid in state.ready_request_ids:
                if len(views) >= state.max_num_seqs or used_tokens >= batch_budget:
                    break
                ctx, rem = progress.get(rid, (0, 0 if phase == "decode" else 1))
                is_decode_req = phase == "decode" or (phase == "mixed" and rem <= 0)
                if is_decode_req:
                    q_len = 1
                    if used_tokens + q_len > batch_budget:
                        continue
                    req_phase = "decode"
                    samples = True
                else:
                    raw_q = max(1, rem)
                    avail = min(batch_budget - used_tokens, input_budget - used_input_tokens)
                    if avail <= 0:
                        continue
                    if state.chunked_prefill:
                        q_len = min(raw_q, avail)
                    elif raw_q <= avail:
                        q_len = raw_q
                    else:
                        continue
                    req_phase = "prefill"
                    samples = q_len >= raw_q
                    used_input_tokens += q_len
                views.append(
                    RequestView(
                        request_id=rid,
                        q_len=q_len,
                        context=ctx,
                        samples=samples,
                        phase=req_phase,
                        priority=priorities.get(rid, 0),
                        draft_verify=False,
                        draft_prepare=False,
                    )
                )
                used_tokens += q_len
        if not views:
            if state.in_flight_completions:
                next_ms = min(ts for _, ts in state.in_flight_completions)
                return WaitForEvent(reason="completion", until_ms=next_ms)
            raise ModelingError(
                f"iteration_step_exceeds_token_budget:stack={self._capabilities.stack_id}:phase={phase}:budget={batch_budget}"
            )
        return IterationPlan(requests=tuple(views), reason="scheduled", all_request_ids=state.ready_request_ids)

    @property
    def launch_codec(self) -> LaunchConfigCodec | None:
        return None


def load_manifest_stack(path: Path | str | Mapping[str, Any]) -> ManifestStackAdapter:
    """Load and validate a custom stack plugin from a JSON manifest file or mapping."""
    if isinstance(path, Mapping):
        validated = _validate_manifest_payload(dict(path), "inline_manifest")
        return ManifestStackAdapter(validated)
    manifest_path = Path(path)
    try:
        if not manifest_path.is_file():
            raise ModelingError(f"stack_manifest_not_found:{manifest_path.name}")
        size = manifest_path.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise ModelingError(f"stack_manifest_too_large:{manifest_path.name}:{size}")
        raw_text = manifest_path.read_text(encoding="utf-8")
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"stack_manifest_read_failed:{manifest_path.name}:{exc}") from exc
    try:
        parsed = json.loads(raw_text)
    except ValueError as exc:
        raise ModelingError(f"stack_manifest_json_invalid:{manifest_path.name}:{exc}") from exc
    return ManifestStackAdapter(parsed, source_label=manifest_path.name)


def _instantiate_plugin_from_module(module_path: str, class_name: str, *, optional: bool) -> RuntimeStackAdapter | None:
    package_root = module_path.rsplit(".", 1)[0]
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        missing = str(exc.name or "")
        if optional and (missing == module_path or missing == package_root or package_root.startswith(missing + ".")):
            return None
        raise
    factory = getattr(mod, class_name, None) or getattr(mod, "create_adapter", None)
    if factory is None or not callable(factory):
        raise ModelingError(f"stack_plugin_factory_missing:{module_path}:{class_name}")
    instance = factory()
    _validate_adapter(instance)
    return instance


class RuntimeStackRegistry:
    """Thread-safe, runtime-agnostic registry of ``RuntimeStackAdapter`` plugins."""

    def __init__(self, *, auto_discover: bool = True) -> None:
        self._lock = threading.RLock()
        self._adapters: dict[str, RuntimeStackAdapter] = {}
        self._aliases: dict[str, str] = {}
        self._auto_discover = bool(auto_discover)
        self._discovered = False

    def _ensure_discovered(self) -> None:
        if self._discovered or not self._auto_discover:
            return
        with self._lock:
            if self._discovered or not self._auto_discover:
                return
            self._discovered = True
            for _expected_id, module_path, class_name, optional in _BUILTIN_PLUGIN_SPECS:
                adapter = _instantiate_plugin_from_module(module_path, class_name, optional=optional)
                if adapter is not None:
                    self._register_locked(adapter, replace=False)
            self._discover_entry_points_locked()

    def _discover_entry_points_locked(self) -> None:
        eps = importlib_metadata.entry_points(group=STACK_ENTRY_POINT_GROUP)
        for ep in eps:
            loaded = ep.load()
            adapter = loaded() if callable(loaded) else loaded
            _validate_adapter(adapter)
            cap = adapter.capabilities()
            if cap.stack_id not in self._adapters:
                self._register_locked(adapter, replace=False)

    def _register_locked(self, adapter: RuntimeStackAdapter, *, replace: bool) -> RuntimeStackAdapter:
        cap = _validate_adapter(adapter)
        stack_id = cap.stack_id
        alias = _canonical_alias(stack_id)
        existing_id = self._aliases.get(alias)
        if existing_id is not None and not replace:
            raise ModelingError(f"stack_already_registered:{stack_id}")
        if existing_id is not None and existing_id != stack_id:
            self._adapters.pop(existing_id, None)
        self._adapters[stack_id] = adapter
        self._aliases[alias] = stack_id
        return adapter

    def register(self, adapter: RuntimeStackAdapter, *, replace: bool = False) -> RuntimeStackAdapter:
        self._ensure_discovered()
        with self._lock:
            return self._register_locked(adapter, replace=replace)

    def unregister(self, stack_id: str) -> RuntimeStackAdapter | None:
        self._ensure_discovered()
        key = str(stack_id or "").strip()
        if not key:
            return None
        with self._lock:
            target = key if key in self._adapters else self._aliases.get(_canonical_alias(key))
            if target is None:
                return None
            removed = self._adapters.pop(target, None)
            self._aliases.pop(_canonical_alias(target), None)
            return removed

    def get(self, stack_id: str) -> RuntimeStackAdapter:
        self._ensure_discovered()
        key = str(stack_id or "").strip()
        if not key:
            raise ModelingError("requested_stack_invalid")
        with self._lock:
            target = key if key in self._adapters else self._aliases.get(_canonical_alias(key))
            if target is None:
                raise ModelingError(
                    f"stack_not_registered:{key}:available={sorted(self._adapters)}"
                )
            return self._adapters[target]

    def list_stacks(self, *, domain: WorkloadDomain | None = None) -> tuple[str, ...]:
        self._ensure_discovered()
        if domain is not None and domain not in WORKLOAD_DOMAINS:
            raise ModelingError(f"workload_domain_invalid:{domain!r}")
        with self._lock:
            if domain is None:
                return tuple(sorted(self._adapters))
            return tuple(
                sid
                for sid in sorted(self._adapters)
                if self._adapters[sid].capabilities().supports_domain(domain)
            )

    def _ordered_candidates_for_domain(self, domain: WorkloadDomain) -> tuple[StackCapabilitySpec, ...]:
        with self._lock:
            ordered_ids: list[str] = []
            seen: set[str] = set()
            for pref in _DOMAIN_PREFERRED_STACKS.get(domain, ()):
                actual = pref if pref in self._adapters else self._aliases.get(_canonical_alias(pref))
                if actual is not None and actual not in seen:
                    ordered_ids.append(actual)
                    seen.add(actual)
            for sid in sorted(self._adapters):
                if sid not in seen and sid not in _BASELINE_STACK_IDS:
                    ordered_ids.append(sid)
                    seen.add(sid)
            for base_id in _BASELINE_FALLBACK_ORDER:
                actual = base_id if base_id in self._adapters else self._aliases.get(_canonical_alias(base_id))
                if actual is not None and actual not in seen:
                    ordered_ids.append(actual)
                    seen.add(actual)
            return tuple(self._adapters[sid].capabilities() for sid in ordered_ids)

    def resolve_for_scenario(
        self,
        requested_stack: str | Path | Mapping[str, Any] | None = None,
        workload_domain: WorkloadDomain = "llm",
        work_unit: WorkUnitKind | None = None,
        hardware_family: str | None = None,
        *,
        required_phases: Sequence[str] = (),
        checkpoint_format: str | None = None,
        track: StackTrack | None = None,
        required_recipes: Sequence[str] = (),
    ) -> RuntimeStackAdapter:
        """Resolve ``workload_domain`` + ``work_unit`` first, then validate and return adapter."""
        if workload_domain not in WORKLOAD_DOMAINS:
            raise ModelingError(f"workload_domain_invalid:{workload_domain!r}")
        effective_unit: WorkUnitKind = (
            DEFAULT_DOMAIN_WORK_UNITS[workload_domain] if work_unit is None else work_unit
        )
        if effective_unit not in WORK_UNIT_KINDS:
            raise ModelingError(f"work_unit_kind_invalid:{effective_unit!r}")
        if effective_unit not in DOMAIN_WORK_UNITS[workload_domain]:
            raise ModelingError(
                f"workload_domain_work_unit_mismatch:domain={workload_domain}:"
                f"work_unit={effective_unit}:allowed={sorted(DOMAIN_WORK_UNITS[workload_domain])}"
            )
        hw_backend = normalize_hardware_backend(hardware_family)
        if isinstance(requested_stack, Mapping) or (requested_stack is not None and str(requested_stack).strip()):
            return self._resolve_explicit(
                requested_stack,
                workload_domain=workload_domain,
                work_unit=effective_unit,
                hardware_backend=hw_backend,
                required_phases=required_phases,
                checkpoint_format=checkpoint_format,
                track=track,
                required_recipes=required_recipes,
            )
        self._ensure_discovered()
        candidates = self._ordered_candidates_for_domain(workload_domain)
        chosen_id = resolve_default_stack_for_domain(
            workload_domain,
            work_unit=effective_unit,
            available_stacks=candidates,
            requested_stack=None,
            required_phases=required_phases,
            hardware_backend=hw_backend,
            checkpoint_format=checkpoint_format,
            track=track,
            required_recipes=required_recipes,
        )
        return self.get(chosen_id)

    def _resolve_explicit(
        self,
        requested_stack: str | Path | Mapping[str, Any],
        *,
        workload_domain: WorkloadDomain,
        work_unit: WorkUnitKind,
        hardware_backend: str | None,
        required_phases: Sequence[str],
        checkpoint_format: str | None,
        track: StackTrack | None,
        required_recipes: Sequence[str],
    ) -> RuntimeStackAdapter:
        if isinstance(requested_stack, Mapping):
            adapter: RuntimeStackAdapter = load_manifest_stack(requested_stack)
        else:
            is_manifest_path = isinstance(requested_stack, Path)
            if not is_manifest_path:
                req_str = str(requested_stack).strip()
                is_manifest_path = req_str.endswith(".json") or (
                    ("/" in req_str or "\\" in req_str) and Path(req_str).suffix.lower() == ".json"
                )
            if is_manifest_path:
                adapter = load_manifest_stack(requested_stack)
            else:
                adapter = self.get(str(requested_stack))
        cap = adapter.capabilities()
        resolve_default_stack_for_domain(
            workload_domain,
            work_unit=work_unit,
            available_stacks=(cap,),
            requested_stack=cap.stack_id,
            required_phases=required_phases,
            hardware_backend=hardware_backend,
            checkpoint_format=checkpoint_format,
            track=track,
            required_recipes=required_recipes,
        )
        return adapter


_REGISTRY_LOCK = threading.RLock()
_DEFAULT_REGISTRY: RuntimeStackRegistry | None = None


def default_registry() -> RuntimeStackRegistry:
    """Return the process-wide default ``RuntimeStackRegistry`` (lazily discovered)."""
    global _DEFAULT_REGISTRY
    with _REGISTRY_LOCK:
        if _DEFAULT_REGISTRY is None:
            _DEFAULT_REGISTRY = RuntimeStackRegistry(auto_discover=True)
        return _DEFAULT_REGISTRY


def reset_default_registry() -> RuntimeStackRegistry:
    """Reset the process-wide default registry to a freshly discovered instance."""
    global _DEFAULT_REGISTRY
    with _REGISTRY_LOCK:
        _DEFAULT_REGISTRY = RuntimeStackRegistry(auto_discover=True)
        return _DEFAULT_REGISTRY


def register_stack(
    adapter: RuntimeStackAdapter,
    *,
    replace: bool = False,
    registry: RuntimeStackRegistry | None = None,
) -> RuntimeStackAdapter:
    """Register a ``RuntimeStackAdapter`` dynamically with zero Core edits."""
    target = default_registry() if registry is None else registry
    return target.register(adapter, replace=replace)


def unregister_stack(
    stack_id: str,
    *,
    registry: RuntimeStackRegistry | None = None,
) -> RuntimeStackAdapter | None:
    """Remove a registered stack adapter by ID."""
    target = default_registry() if registry is None else registry
    return target.unregister(stack_id)


def get_stack(
    stack_id: str,
    *,
    registry: RuntimeStackRegistry | None = None,
) -> RuntimeStackAdapter:
    """Retrieve a registered ``RuntimeStackAdapter`` by ID."""
    target = default_registry() if registry is None else registry
    return target.get(stack_id)


def list_stacks(
    *,
    domain: WorkloadDomain | None = None,
    registry: RuntimeStackRegistry | None = None,
) -> tuple[str, ...]:
    """List registered stack IDs, optionally filtered by ``WorkloadDomain``."""
    target = default_registry() if registry is None else registry
    return target.list_stacks(domain=domain)


def resolve_stack_for_scenario(
    requested_stack: str | Path | Mapping[str, Any] | None = None,
    workload_domain: WorkloadDomain = "llm",
    work_unit: WorkUnitKind | None = None,
    hardware_family: str | None = None,
    *,
    required_phases: Sequence[str] = (),
    checkpoint_format: str | None = None,
    track: StackTrack | None = None,
    required_recipes: Sequence[str] = (),
    registry: RuntimeStackRegistry | None = None,
) -> RuntimeStackAdapter:
    """Resolve and capability-validate the stack adapter for a scenario."""
    target = default_registry() if registry is None else registry
    return target.resolve_for_scenario(
        requested_stack=requested_stack,
        workload_domain=workload_domain,
        work_unit=work_unit,
        hardware_family=hardware_family,
        required_phases=required_phases,
        checkpoint_format=checkpoint_format,
        track=track,
        required_recipes=required_recipes,
    )


__all__ = [
    "DEFAULT_DOMAIN_WORK_UNITS",
    "MAX_MANIFEST_BYTES",
    "STACK_ENTRY_POINT_GROUP",
    "STACK_MANIFEST_SCHEMA",
    "ManifestStackAdapter",
    "RuntimeStackRegistry",
    "default_registry",
    "get_stack",
    "list_stacks",
    "load_manifest_stack",
    "normalize_hardware_backend",
    "register_stack",
    "reset_default_registry",
    "resolve_stack_for_scenario",
    "unregister_stack",
]
