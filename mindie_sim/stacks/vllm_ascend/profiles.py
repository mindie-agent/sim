"""vLLM-Ascend runtime profiles, manifest loader, ISA guards, and MC2/MSO policy.

[Role] Own the packaged ``vllm-ascend`` plugin manifest (``manifest.json``),
  strict A2/A3 (INT8/INT4/C8) vs A5 (MXFP8/MXFP4/FP8) ISA guards,
  workload-domain classification, and MC2-vs-MSO mutual exclusion.
[Invariants] Strictly LLM/VLM scope (Diffusion/DiT models classify as
  ``"diffusion"`` and are rejected at the plugin boundary). Zero imports from
  ``mindie_sim.export`` or ``mindie_sim.compose`` (AG010). Zero per-model-name
  executable branches (AG014). A2/A3 never admit MXFP/FP8 weights or FP8 KV
  cache.
[Timeouts & Locks] Single bounded read (< 50 MiB) of packaged ``manifest.json``
  cached via ``lru_cache``; no network I/O or locks.
[Side-Effect Policy] Read-only over packaged manifest; returns fresh dicts and
  immutable tuples without mutating caller inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.stack import ManifestBinding, StackTrack, WorkloadDomain

VLLM_ASCEND_STACK_ID = "vllm-ascend"
VLLM_ASCEND_REPO = "vllm-project/vllm-ascend"
MANIFEST_SCHEMA = "mindie-sim.stack-manifest/1"
MANIFEST_PATH = Path(__file__).resolve().with_name("manifest.json")
MAX_MANIFEST_BYTES = 50 * 1024 * 1024

FUSED_MC2_DISABLED = 0
FUSED_MC2_DISPATCH_FFN_COMBINE = 1
FUSED_MC2_MEGAMOE_ROUTED = 2
FUSED_MC2_MEGAMOE_SHARED = 3

NO_MTP_PROPOSAL_TOKENS = 0
SINGLE_MTP_PROPOSAL_TOKEN = 1
MTP_STANDARD_PROPOSAL_TOKENS = 3
MTP_DSPARK_PROPOSAL_TOKENS = 5
PAIRED_SFA_DOC_BLOCK_SIZE = 544

_A5_FAMILY_TOKENS = ("a5", "950", "9582", "9581", "9579", "958b", "9599", "3510", "arch35")
_A3_FAMILY_TOKENS = ("a3", "910c", "910_93", "9392", "9391", "9382", "9362")
_A2_FAMILY_TOKENS = ("a2", "910b", "310p", "300i")
_NVIDIA_TOKENS = ("nvidia", "cuda", "h100", "h200", "b200", "gb200", "a100", "hopper", "blackwell")

A2_A3_INT_ONLY_FAMILIES = frozenset({"a2", "a3"})
A2_A3_ALLOWED_KV_DTYPES = frozenset({"bf16", "fp16", "int8", "c8", "auto", ""})
A5_ONLY_QUANT_TOKENS = ("mxfp", "fp8", "fp4")

_DIFFUSION_ARCH_TOKENS = ("dit", "unet", "diffusion", "hunyuan_video", "wan2", "flux", "sdxl", "sd3")

# Purged model-name lookup tables: Commit #2 must move pure checkpoint URL
# metadata to stacks/vllm_ascend/checkpoint_registry.py and remove model-keyed
# runtime profile defaults in favor of structural ModelSpec + Hardware rules.
DEFAULT_CHECKPOINTS_BY_MODEL_HW: dict[tuple[str, str], str] = {}
VA_RECOMMENDED_PROFILES: dict[str, dict[str, Any]] = {}


def normalize_hw_family(hw: Hardware | str) -> str:
    """Resolve hardware object or SKU/family string to ``a2`` / ``a3`` / ``a5`` / ``other``."""
    if isinstance(hw, Hardware):
        probes = (
            hw.arch_family,
            hw.sku,
            hw.name,
            hw.dies[0].arch_family if hw.dies else "",
            hw.dies[0].sku if hw.dies else "",
            hw.dies[0].compute_arch.preset if hw.dies else "",
        )
        raw = " ".join(str(p or "") for p in probes).lower()
    else:
        raw = str(hw or "").strip().lower()
    if any(tok in raw for tok in _A5_FAMILY_TOKENS):
        return "a5"
    if any(tok in raw for tok in _A3_FAMILY_TOKENS):
        return "a3"
    if any(tok in raw for tok in _A2_FAMILY_TOKENS):
        return "a2"
    return "other"


def hardware_backend_kind(hardware: Hardware) -> str:
    """Classify a ``Hardware`` object into ``ascend`` / ``nvidia`` / ``generic``."""
    if normalize_hw_family(hardware) in {"a2", "a3", "a5"}:
        return "ascend"
    probes = " ".join(
        str(x or "") for x in (
            hardware.arch_family,
            hardware.sku,
            hardware.name,
            hardware.dies[0].arch_family if hardware.dies else "",
            hardware.dies[0].compute_arch.preset if hardware.dies else "",
        )
    ).lower()
    if "ascend" in probes or "npu" in probes or "dav_" in probes:
        return "ascend"
    if any(tok in probes for tok in _NVIDIA_TOKENS):
        return "nvidia"
    return "generic"


def normalize_canonical_model(model_or_name: ModelSpec | str) -> str:
    """Unplugged socket: model-name routing purged per AG014."""
    raise NotImplementedError(
        "LEGACY_PURGED: wire pure checkpoint-URL normalization in "
        "stacks/vllm_ascend/checkpoint_registry.py and replace executable "
        "model-name routing with structural ModelSpec predicates"
    )


def resolve_model_type(model: ModelSpec) -> str:
    """Unplugged socket: model-name fallback mapping purged per AG014."""
    raise NotImplementedError(
        "LEGACY_PURGED: eliminate model-name resolve_model_type heuristics; "
        "dispatch purely on structural LayerPlan and ModelSpec fields"
    )


def infer_model_workload_domain(model: ModelSpec) -> WorkloadDomain:
    """Classify ``ModelSpec`` structurally into ``llm`` / ``vlm`` / ``diffusion``."""
    meta = dict(getattr(model, "metadata", ()) or ())
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    explicit_domain = str(meta.get("domain") or facts.get("workload_domain") or "").strip().lower()
    if explicit_domain in {"llm", "vlm", "diffusion", "custom"}:
        return explicit_domain  # type: ignore[return-value]
    for plan in model.layer_plans:
        arch = str(getattr(plan, "architecture", "") or "").lower()
        if any(tok in arch for tok in _DIFFUSION_ARCH_TOKENS):
            return "diffusion"
    if model.vision is not None or bool(facts.get("vision_present")):
        return "vlm"
    return "llm"


def default_checkpoint_repo(canonical_model: str, hw_family: str) -> str | None:
    """Unplugged socket: checkpoint URL table + cross-gen fallback purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: move curated checkpoint lookup into "
        "stacks/vllm_ascend/checkpoint_registry.py without cross-generation fallback"
    )


def validate_vllm_ascend_quant_isa(
    quant_preset_or_tag: str,
    hw_family: str,
    *,
    kv_cache_dtype: str = "bf16",
    strict: bool = False,
) -> tuple[str, ...]:
    """Enforce strict A2/A3 INT-only vs A5 MXFP/FP8 ISA separation."""
    hw = normalize_hw_family(hw_family)
    warnings: list[str] = []
    if hw in A2_A3_INT_ONLY_FAMILIES:
        q_lower = str(quant_preset_or_tag or "").strip().lower()
        kv_lower = str(kv_cache_dtype or "bf16").strip().lower()
        if any(tok in q_lower for tok in A5_ONLY_QUANT_TOKENS):
            msg = (
                f"hw_isa_incompatible:{hw}:weight_quant={quant_preset_or_tag}:"
                "A2/A3 lacks hardware FP8/MXFP; use W8A8_DYNAMIC or W4A8_DYNAMIC"
            )
            if strict:
                raise ModelingError(msg)
            warnings.append(msg)
        if kv_lower and kv_lower not in A2_A3_ALLOWED_KV_DTYPES:
            msg = (
                f"hw_isa_incompatible:{hw}:kv_cache_dtype={kv_cache_dtype}:"
                "A2/A3 supports BF16 or C8 (INT8) KV cache only"
            )
            if strict:
                raise ModelingError(msg)
            warnings.append(msg)
    return tuple(warnings)


@lru_cache(maxsize=1)
def load_vllm_ascend_manifest() -> dict[str, Any]:
    """Load and strictly validate packaged ``vllm-ascend`` plugin ``manifest.json``."""
    if not MANIFEST_PATH.is_file():
        raise ModelingError(f"vllm_ascend_manifest_missing:{MANIFEST_PATH}")
    if MANIFEST_PATH.stat().st_size > MAX_MANIFEST_BYTES:
        raise ModelingError(f"vllm_ascend_manifest_exceeds_50_mib:{MANIFEST_PATH}")
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema") != MANIFEST_SCHEMA:
        raise ModelingError("vllm_ascend_manifest_schema_invalid")
    for item in raw.get("bindings") or ():
        mech_id = str(item.get("mechanism_id") or "")
        if item.get("models"):
            raise ModelingError(f"vllm_ascend_manifest_ag014_models_forbidden:{mech_id}")
        if not str(item.get("lowering_pass_id") or "").strip():
            raise ModelingError(f"vllm_ascend_manifest_missing_lowering_pass_id:{mech_id}")
        prov_raw = item.get("provenance") or {}
        if not str(prov_raw.get("source_sha256") or "").strip():
            raise ModelingError(f"vllm_ascend_manifest_missing_source_sha256:{mech_id}")
    return dict(raw)


@lru_cache(maxsize=1)
def load_manifest_bindings() -> tuple[ManifestBinding, ...]:
    """Return immutable ``ManifestBinding`` records from ``manifest.json``."""
    raw = load_vllm_ascend_manifest()
    bindings: list[ManifestBinding] = []
    for item in raw.get("bindings") or ():
        prov_raw = item.get("provenance") or {}
        source_sha = str(prov_raw.get("source_sha256") or "").strip()
        lowering_pass_id = str(item.get("lowering_pass_id") or "").strip()
        mech_id = str(item["mechanism_id"])
        if item.get("models"):
            raise ModelingError(f"vllm_ascend_manifest_ag014_models_forbidden:{mech_id}")
        if not lowering_pass_id:
            raise ModelingError(f"vllm_ascend_manifest_missing_lowering_pass_id:{mech_id}")
        if not source_sha:
            raise ModelingError(f"vllm_ascend_manifest_missing_source_sha256:{mech_id}")
        prov = SourceRef(
            repository=str(prov_raw.get("repository") or VLLM_ASCEND_REPO),
            revision=str(prov_raw.get("revision") or raw.get("manifest_revision") or ""),
            relative_path=str(prov_raw.get("relative_path") or "manifest.json"),
            source_sha256=source_sha,
        )
        bindings.append(
            ManifestBinding(
                mechanism_id=mech_id,
                state=str(item["state"]),  # type: ignore[arg-type]
                ref=str(item.get("ref") or ""),
                lowering_pass_id=lowering_pass_id,
                provenance=prov,
                models=(),
                archs=tuple(str(a).lower() for a in (item.get("archs") or ())),
                phase_scope=tuple(str(p).lower() for p in (item.get("phases") or item.get("phase_scope") or ())),
                options=tuple(sorted((str(k), v) for k, v in (item.get("options") or {}).items())),
                capability_predicates=tuple(str(p) for p in (item.get("capability_predicates") or ())),
                notes=str(item.get("notes") or ""),
            )
        )
    return tuple(bindings)


def resolve_manifest_bindings(
    track: StackTrack,
    canonical_model: str,
    arch: str,
    *,
    phase: str | None = None,
) -> tuple[ManifestBinding, ...]:
    """Unplugged socket: model-name manifest matching purged per AG014."""
    raise NotImplementedError(
        "LEGACY_PURGED: wire structural capability_predicates + canonical StackTrack "
        "+ phase_scope filtering in resolve_manifest_bindings (zero canonical_model)"
    )


def shipping_collective_policy(
    arch: str, *, canonical: str = "", role: str | None = None,
) -> dict[str, Any] | None:
    """Unplugged socket: model-name collective policy sets purged per AG014."""
    raise NotImplementedError(
        "LEGACY_PURGED: derive shipping collective policy strictly from Hardware arch, "
        "deployment role, and structural ModelSpec predicates (zero model-name sets)"
    )


def resolve_fused_mc2_and_mso(
    *,
    arch: str,
    model: ModelSpec,
    parallel: ParallelSpec,
    requested_fused_mc2: int | None,
    requested_mso: bool,
) -> tuple[int, bool, tuple[tuple[str, str, str], ...]]:
    """Resolve upstream ``enable_fused_mc2`` level vs MSO mutex and ISA legality."""
    hw = normalize_hw_family(arch)
    dispositions: list[tuple[str, str, str]] = []
    has_moe = bool(model.num_experts > 0 or any(p.ffn == "moe" for p in model.layer_plans))
    has_shared = bool(model.num_shared_experts > 0 or any(p.num_shared_experts > 0 for p in model.layer_plans))
    eff_mc2 = FUSED_MC2_DISABLED
    if requested_fused_mc2 is not None:
        lvl = int(requested_fused_mc2)
        if not has_moe and lvl > FUSED_MC2_DISABLED:
            dispositions.append(("enable_fused_mc2", "rejected", "dense_model_has_no_moe_mc2"))
        elif parallel.ep <= 1 and lvl > FUSED_MC2_DISABLED:
            dispositions.append(("enable_fused_mc2", "downgraded", "mc2_requires_ep_gt_1"))
        elif lvl == FUSED_MC2_DISPATCH_FFN_COMBINE and hw != "a3":
            dispositions.append(("enable_fused_mc2", "rejected", f"level_1_dispatch_ffn_combine_a3_only:hw={hw}"))
        elif lvl == FUSED_MC2_MEGAMOE_ROUTED and hw not in {"a3", "a5"}:
            dispositions.append(("enable_fused_mc2", "rejected", f"level_2_megamoe_requires_a3_or_a5:hw={hw}"))
        elif lvl == FUSED_MC2_MEGAMOE_SHARED and hw != "a5":
            dispositions.append(("enable_fused_mc2", "downgraded", f"level_3_megamoe_shared_a5_only:hw={hw}"))
            if hw == "a3" and has_moe and parallel.ep > 1:
                eff_mc2 = FUSED_MC2_MEGAMOE_ROUTED
        elif lvl == FUSED_MC2_MEGAMOE_SHARED and not has_shared:
            eff_mc2 = FUSED_MC2_MEGAMOE_ROUTED
            dispositions.append(("enable_fused_mc2", "downgraded", "level_3_downgraded_no_shared_experts"))
        elif FUSED_MC2_DISABLED <= lvl <= FUSED_MC2_MEGAMOE_SHARED:
            eff_mc2 = lvl
            dispositions.append(("enable_fused_mc2", "accepted", f"arch={hw}:level={lvl}"))
        else:
            dispositions.append(("enable_fused_mc2", "rejected", f"unsupported_mc2_level:{lvl}"))

    eff_mso = bool(requested_mso)
    if eff_mso and not has_shared:
        eff_mso = False
        dispositions.append(("multistream_overlap_shared_expert", "rejected", "model_has_no_shared_experts"))
    elif eff_mso and eff_mc2 > FUSED_MC2_DISABLED:
        eff_mso = False
        dispositions.append((
            "multistream_overlap_shared_expert",
            "downgraded",
            f"mutex_with_enable_fused_mc2={eff_mc2}",
        ))
    elif eff_mso:
        dispositions.append(("multistream_overlap_shared_expert", "accepted", "aux_stream_shared_expert_overlap"))
    return eff_mc2, eff_mso, tuple(dispositions)


def simulation_config(
    canonical: str, sku: str, arch: str, *, role: str | None = None,
) -> dict[str, Any]:
    """Unplugged socket: model-keyed VA_RECOMMENDED_PROFILES lookup purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: retire model-keyed VA_RECOMMENDED_PROFILES/simulation_config "
        "and arbitrary role-topology fallback; resolve defaults purely from structural "
        "ModelSpec + Hardware capabilities"
    )


__all__ = [
    "A2_A3_ALLOWED_KV_DTYPES",
    "A2_A3_INT_ONLY_FAMILIES",
    "A5_ONLY_QUANT_TOKENS",
    "DEFAULT_CHECKPOINTS_BY_MODEL_HW",
    "FUSED_MC2_DISABLED",
    "FUSED_MC2_DISPATCH_FFN_COMBINE",
    "FUSED_MC2_MEGAMOE_ROUTED",
    "FUSED_MC2_MEGAMOE_SHARED",
    "MANIFEST_PATH",
    "MANIFEST_SCHEMA",
    "MTP_DSPARK_PROPOSAL_TOKENS",
    "MTP_STANDARD_PROPOSAL_TOKENS",
    "NO_MTP_PROPOSAL_TOKENS",
    "PAIRED_SFA_DOC_BLOCK_SIZE",
    "SINGLE_MTP_PROPOSAL_TOKEN",
    "VA_RECOMMENDED_PROFILES",
    "VLLM_ASCEND_REPO",
    "VLLM_ASCEND_STACK_ID",
    "ManifestBinding",
    "default_checkpoint_repo",
    "hardware_backend_kind",
    "infer_model_workload_domain",
    "load_manifest_bindings",
    "load_vllm_ascend_manifest",
    "normalize_canonical_model",
    "normalize_hw_family",
    "resolve_fused_mc2_and_mso",
    "resolve_manifest_bindings",
    "resolve_model_type",
    "shipping_collective_policy",
    "simulation_config",
    "validate_vllm_ascend_quant_isa",
]
