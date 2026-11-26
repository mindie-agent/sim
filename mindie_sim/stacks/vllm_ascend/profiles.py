"""vLLM-Ascend runtime profiles, manifest loader, ISA guards, and MC2/MSO policy.

[Role] Own the packaged ``vllm-ascend`` plugin manifest (``manifest.json``),
  10-axis structural capability extraction, strict A2/A3 (INT8/INT4/C8) vs A5
  (MXFP8/MXFP4/FP8) ISA guards, workload-domain classification, structural
  collective policy, and MC2-vs-MSO mutual exclusion.
[Invariants] Strictly LLM/VLM scope (Diffusion/DiT models classify as
  ``"diffusion"`` and are rejected at the plugin boundary). Zero imports from
  ``mindie_sim.export`` or ``mindie_sim.compose`` (AG010). Zero per-model-name
  executable branches (AG014). A2/A3 never admit MXFP/FP8 weights or FP8 KV
  cache. Zero cross-generation checkpoint fallback.
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
from mindie_sim.schema.stack import STACK_TRACKS, ManifestBinding, StackTrack, WorkloadDomain

from .checkpoint_registry import (
    DEFAULT_CHECKPOINTS_BY_MODEL_HW,
    lookup_default_checkpoint_repo,
    normalize_checkpoint_model_key,
)

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
COMPRESSED_KV_BLOCK_SIZE = 32
QSA_GROUP_KPOOL_RATIO = 4

_A5_FAMILY_TOKENS = ("a5", "950", "9582", "9581", "9579", "958b", "9599", "3510", "arch35")
_A3_FAMILY_TOKENS = ("a3", "910c", "910_93", "9392", "9391", "9382", "9362")
_A2_FAMILY_TOKENS = ("a2", "910b", "310p", "300i")

A2_A3_INT_ONLY_FAMILIES = frozenset({"a2", "a3"})
A2_A3_ALLOWED_KV_DTYPES = frozenset({"bf16", "fp16", "int8", "c8", "auto", ""})
A5_ONLY_QUANT_TOKENS = ("mxfp", "fp8", "fp4")

_DIFFUSION_ARCH_TOKENS = ("dit", "unet", "diffusion", "hunyuan_video", "wan2", "flux", "sdxl", "sd3")
_VALID_ROLES = frozenset({"colocated", "pd_disaggregated", "pd_prefill", "pd_decode"})
_MXFP_FORMATS = frozenset({"mxfp8", "mx_fp8", "mxfp4", "mx_fp4", "fp4_e2m1", "fp4"})
_MXFP_SCALE_FORMATS = frozenset({"e8m0", "fp8_e8m0"})
_FP4_W4A8_EXPERT_FORMATS = frozenset({"mxfp4", "mx_fp4", "fp4", "fp4_e2m1", "int4", "w4a8"})
_W4A8_EXPERT_FORMATS = frozenset({"int4", "w4a8"})
_FP8_KV_MODES = frozenset({"fp8", "fp8_e4m3", "mxfp8"})


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
    """Classify a ``Hardware`` object into ``ascend`` or ``generic``."""
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
    return "generic"


def normalize_canonical_model(model_or_name: ModelSpec | str) -> str:
    """Delegate pure checkpoint-URL slug normalization to ``checkpoint_registry``."""
    return normalize_checkpoint_model_key(model_or_name)


def resolve_model_type(model: ModelSpec) -> str:
    """Extract declared HF ``model_type`` metadata without model-name heuristics."""
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    meta = dict(getattr(model, "metadata", ()) or ())
    return str(
        facts.get("model_type") or meta.get("source_model_type") or meta.get("model_type") or ""
    ).strip().lower()


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
    """Return exact curated checkpoint repository for ``(canonical_model, hw_family)``."""
    hw = normalize_hw_family(hw_family)
    if hw not in {"a2", "a3", "a5"}:
        return None
    return lookup_default_checkpoint_repo(canonical_model, hw)


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


def _quant_capabilities(model: ModelSpec, quant_plan: Any) -> dict[str, Any]:
    """Extract weight/activation/KV quantization capability flags structurally."""
    qspec = getattr(model, "quant", None)
    act_dtype = str(getattr(model, "activation_dtype", "bf16") or "bf16").strip().lower()
    w_fmts: set[str] = set()
    e_fmts: set[str] = set()
    has_quant_act = False
    has_mxfp_scale = False
    if qspec is not None:
        wf = str(getattr(qspec, "weight_format", "") or "").strip().lower()
        ef = str(getattr(qspec, "expert_weight_format", "") or "").strip().lower()
        af = str(getattr(qspec, "activation_format", "") or "").strip().lower()
        sf = str(getattr(qspec, "scale_format", "") or "").strip().lower()
        asf = str(getattr(qspec, "activation_scale_format", "") or "").strip().lower()
        if wf:
            w_fmts.add(wf)
        if ef:
            e_fmts.add(ef)
            w_fmts.add(ef)
        if sf in _MXFP_SCALE_FORMATS or asf in _MXFP_SCALE_FORMATS:
            has_mxfp_scale = True
        if af and af != act_dtype:
            has_quant_act = True
        for _, pairs in getattr(qspec, "category_overrides", ()) or ():
            for k, v in pairs:
                v_low = str(v or "").strip().lower()
                if k == "activation_format" and v_low and v_low != act_dtype:
                    has_quant_act = True
                elif k in {"scale_format", "activation_scale_format"} and v_low in _MXFP_SCALE_FORMATS:
                    has_mxfp_scale = True
                elif k in {"weight_format", "expert_weight_format"} and v_low:
                    w_fmts.add(v_low)
                    if k == "expert_weight_format":
                        e_fmts.add(v_low)
    has_fp8_kv = str(getattr(qspec, "kv_dtype", "") or "").strip().lower() in _FP8_KV_MODES
    has_mxfp_weights = bool(w_fmts & _MXFP_FORMATS) or has_mxfp_scale
    if quant_plan is not None:
        plan_mxfp = False
        for (_sc, _lid, role_name), abi in getattr(quant_plan, "port_abis", ()) or ():
            lm = str(getattr(abi, "linear_method", "") or "").lower()
            wd = str(getattr(abi, "weight_dtype", "") or "").lower()
            wsd = str(getattr(abi, "weight_scale_dtype", "") or "").lower()
            id_ = str(getattr(abi, "input_dtype", "") or "").lower()
            if lm in {"mx_fp8", "mx_fp4"} or wd in _MXFP_FORMATS or wsd in _MXFP_SCALE_FORMATS:
                plan_mxfp = True
            if "experts" in str(role_name) and (lm in {"mx_fp4", "w4a8"} or wd in {"fp4_e2m1", "int4"}):
                e_fmts.add(wd or lm)
            if id_ and id_ not in {"bf16", "fp16", "fp32", act_dtype}:
                has_quant_act = True
        has_mxfp_weights = plan_mxfp
        for k_abi in getattr(quant_plan, "kv_cache_abi", ()) or ():
            if any(str(pl.dtype).lower() in _FP8_KV_MODES for pl in getattr(k_abi, "planes", ())):
                has_fp8_kv = True
    return {
        "has_quant_activation": has_quant_act,
        "has_mxfp_weights": has_mxfp_weights,
        "has_fp4_or_w4a8_experts": bool(e_fmts & _FP4_W4A8_EXPERT_FORMATS),
        "has_w4a8_experts": bool(e_fmts & _W4A8_EXPERT_FORMATS),
        "has_fp8_kv": has_fp8_kv,
    }


def extract_model_capabilities(
    model: ModelSpec | None,
    *,
    arch: str,
    parallel: ParallelSpec | None = None,
    quant_plan: Any = None,
    speculative_k: int = 0,
    sp_enabled: bool = False,
) -> dict[str, Any]:
    """Derive 10-axis structural capability predicates solely from ``ModelSpec`` & topology."""
    hw = normalize_hw_family(arch)
    tp = int(parallel.tp) if parallel is not None else 1
    ep = int(parallel.ep) if parallel is not None else 1
    pp = int(parallel.pp) if parallel is not None else 1
    pcp = int(parallel.pcp) if parallel is not None else 1
    dcp = int(parallel.cp) if parallel is not None else 1
    base: dict[str, Any] = {
        "arch": hw,
        "workload_domain": "llm",
        "tp": tp,
        "ep": ep,
        "pp": pp,
        "pcp": pcp,
        "dcp": dcp,
        "sp_enabled": bool(sp_enabled),
        "speculative_k": int(speculative_k),
        "has_quant_activation": False,
        "has_mla": False,
        "qk_rope_head_dim": 0,
        "mla_use_nope": False,
        "mla_qkv_a_gate_fused": False,
        "has_attn_res": False,
        "has_mhc_sinkhorn": False,
        "has_low_rank_hc": False,
        "has_unit_offset_rmsnorm": False,
        "has_fp32_residual_norm": False,
        "has_recurrent": False,
        "has_kda": False,
        "has_gdn": False,
        "gdn_hv_hk_ratio": 0,
        "has_moe": False,
        "has_shared_expert": False,
        "has_mxfp_weights": False,
        "has_fp4_or_w4a8_experts": False,
        "has_w4a8_experts": False,
        "ffn_activation": "silu",
        "has_gqa": False,
        "has_mla_indexer": False,
        "has_kpool_indexer": False,
        "has_csa2_compressed_indexer": False,
        "has_qsa_indexed_gbsa": False,
        "has_msa_gbsa": False,
        "has_region_gbsa_indexer": False,
        "has_sfa_packed_fp8_kv": False,
        "supports_torch_compile": True,
        "has_dspark_draft": False,
        "has_dspark_mla_draft": False,
    }
    if model is None:
        return base

    plans = tuple(model.layer_plans or ())
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    q_caps = _quant_capabilities(model, quant_plan)
    mla_plans = [p for p in plans if str(p.attention).lower() == "mla"]
    gdn_plans = [p for p in plans if str(p.attention).lower() == "gdn"]
    kda_plans = [p for p in plans if str(p.attention).lower() == "kda"]
    gqa_plans = [p for p in plans if str(p.attention).lower() not in {"mla", "kda", "gdn"}]
    gdn_ratio = 0
    if gdn_plans:
        p0 = gdn_plans[0]
        hk = int(p0.linear_key_heads or p0.linear_heads or 0)
        hv = int(p0.linear_value_heads or p0.linear_heads or 0)
        gdn_ratio = (hv // hk) if (hk > 0 and hv % hk == 0) else 0

    moe_acts = [str(p.ffn_activation or "silu").lower() for p in plans if p.ffn == "moe"]
    all_acts = [str(p.ffn_activation or "silu").lower() for p in plans]
    ffn_act = moe_acts[0] if moe_acts else (all_acts[0] if all_acts else "silu")
    aux = getattr(model, "auxiliary", None)
    dspark = getattr(aux, "dspark", None) if aux is not None else None
    has_mla_idx = any(p.indexer for p in mla_plans)
    norm_offset = float(facts.get("norm_weight_offset", 0.0) or 0.0)
    base.update({
        "workload_domain": infer_model_workload_domain(model),
        "has_quant_activation": q_caps["has_quant_activation"],
        "has_mla": bool(mla_plans),
        "qk_rope_head_dim": max((int(p.qk_rope_head_dim) for p in mla_plans), default=0),
        "mla_use_nope": bool(mla_plans) and all(bool(p.mla_use_nope) for p in mla_plans),
        "mla_qkv_a_gate_fused": any(bool(p.mla_qkv_a_gate_fused) for p in mla_plans),
        "has_attn_res": any(int(p.residual_block_size) > 0 for p in plans),
        "has_mhc_sinkhorn": any(int(p.hc_mult) > 1 and int(p.hc_sinkhorn_iters) > 0 for p in plans),
        "has_low_rank_hc": any(int(p.hc_mult) > 1 and int(p.hc_lowrank) > 0 for p in plans),
        "has_unit_offset_rmsnorm": any(bool(p.norm_unit_offset) for p in plans) or norm_offset != 0.0,
        "has_fp32_residual_norm": bool(facts.get("fp32_residual_connection", False)),
        "has_recurrent": bool(kda_plans or gdn_plans),
        "has_kda": bool(kda_plans),
        "has_gdn": bool(gdn_plans),
        "gdn_hv_hk_ratio": gdn_ratio,
        "has_moe": bool(model.num_experts > 0 or any(p.ffn == "moe" for p in plans)),
        "has_shared_expert": bool(
            model.num_shared_experts > 0 or any(int(p.num_shared_experts) > 0 for p in plans)
        ),
        "has_mxfp_weights": q_caps["has_mxfp_weights"],
        "has_fp4_or_w4a8_experts": q_caps["has_fp4_or_w4a8_experts"],
        "has_w4a8_experts": q_caps["has_w4a8_experts"],
        "ffn_activation": ffn_act,
        "has_gqa": bool(gqa_plans),
        "has_mla_indexer": has_mla_idx,
        "has_kpool_indexer": any(p.indexer and int(p.index_kpool) > 1 for p in mla_plans),
        "has_csa2_compressed_indexer": any(
            p.mla_variant == "latent_mqa" and p.cache_kind == "window_compressed" and int(p.compress_ratio) > 0
            for p in mla_plans
        ),
        "has_qsa_indexed_gbsa": any(
            p.indexer and int(p.index_kpool) == QSA_GROUP_KPOOL_RATIO and bool(p.q_output_gate)
            for p in gqa_plans
        ),
        "has_msa_gbsa": any(p.indexer and int(p.index_kpool) == 1 for p in gqa_plans),
        "has_region_gbsa_indexer": any(p.indexer and int(p.index_kpool) > QSA_GROUP_KPOOL_RATIO for p in gqa_plans),
        "has_sfa_packed_fp8_kv": bool(has_mla_idx and q_caps["has_fp8_kv"]),
        "supports_torch_compile": bool(facts.get("supports_torch_compile", True)),
        "has_dspark_draft": dspark is not None,
        "has_dspark_mla_draft": dspark is not None and str(getattr(dspark, "attention_kind", "")).lower() == "mla",
    })
    return base


def _load_structural_model_if_str(model_or_name: ModelSpec | str | None) -> ModelSpec | None:
    if isinstance(model_or_name, ModelSpec):
        return model_or_name
    if isinstance(model_or_name, str) and model_or_name.strip():
        from mindie_sim.model_catalog import load_model
        key = normalize_checkpoint_model_key(model_or_name)
        for cand in dict.fromkeys((model_or_name.strip(), key)):
            try:
                return load_model(cand)
            except (KeyError, FileNotFoundError, ModelingError):
                continue
    return None


def resolve_manifest_bindings(
    track: StackTrack,
    model_or_name: ModelSpec | str | None,
    arch: str,
    *,
    phase: str | None = None,
    capabilities: Mapping[str, Any] | None = None,
    parallel: ParallelSpec | None = None,
    quant_plan: Any = None,
    speculative_k: int = 0,
    sp_enabled: bool = False,
) -> tuple[ManifestBinding, ...]:
    """Select active manifest bindings strictly via canonical ``StackTrack`` & structural predicates."""
    if track not in STACK_TRACKS:
        raise ModelingError(f"stack_track_invalid:{track!r}:expected={sorted(STACK_TRACKS)}")
    if track == "eager":
        return ()
    allowed_states = frozenset({"merged"}) if track == "shipping" else frozenset({"merged", "unmerged_pr"})
    hw = normalize_hw_family(arch)
    if capabilities is None:
        spec = _load_structural_model_if_str(model_or_name)
        caps: Mapping[str, Any] = extract_model_capabilities(
            spec, arch=hw, parallel=parallel, quant_plan=quant_plan,
            speculative_k=speculative_k, sp_enabled=sp_enabled,
        )
    else:
        caps = capabilities
    phase_norm = str(phase).strip().lower() if phase else None
    matched: list[ManifestBinding] = []
    seen_keys: set[tuple[str, str]] = set()
    for binding in load_manifest_bindings():
        if binding.state not in allowed_states:
            continue
        if not binding.matches("", hw, phase=phase_norm, capabilities=caps):
            continue
        key = (binding.mechanism_id, binding.state)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        matched.append(binding)
    return tuple(matched)


def shipping_collective_policy(
    arch: str,
    *,
    canonical: str = "",
    role: str | None = None,
    model: ModelSpec | None = None,
) -> dict[str, Any] | None:
    """Resolve arch-disjoint shipping collective policy strictly from hardware arch and role."""
    del canonical, model
    arch_norm = normalize_hw_family(arch)
    role_norm = str(role or "").strip().lower() or "colocated"
    if role_norm not in _VALID_ROLES:
        raise ModelingError(f"shipping_collective_policy_invalid_role:{role!r}")
    if arch_norm == "a5":
        env_mode = "CCU_SCHED" if role_norm == "pd_decode" else "CCU_MS"
        return {
            "arch": "a5",
            "role": role_norm,
            "process_env_mode": env_mode,
            "executor": "ccu_ms",
            "fabric": "ub_direct",
            "oneshot_active": True,
        }
    if arch_norm == "a3":
        return {
            "arch": "a3",
            "role": role_norm,
            "process_env_mode": "AIV",
            "executor": "aiv",
            "fabric": "hccs_sw",
            "oneshot_active": True,
        }
    if arch_norm == "a2":
        return {
            "arch": "a2",
            "role": role_norm,
            "process_env_mode": "AIV",
            "executor": "aiv",
            "fabric": "hccs_8p_roce",
            "oneshot_active": True,
        }
    return None


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
    has_gated_shared = any(
        p.ffn == "moe" and bool(getattr(p, "shared_expert_gate", False))
        for p in model.layer_plans
    )
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
        elif lvl == FUSED_MC2_MEGAMOE_SHARED and has_gated_shared:
            eff_mc2 = FUSED_MC2_MEGAMOE_ROUTED
            dispositions.append(("enable_fused_mc2", "downgraded", "level_3_downgraded_gated_shared_expert"))
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
    model_or_key: ModelSpec | str,
    sku: str,
    arch: str,
    *,
    role: str | None = None,
    model: ModelSpec | None = None,
) -> dict[str, Any]:
    """Derive vLLM-Ascend runtime defaults purely from structural ``ModelSpec``, hardware, and role."""
    if not str(sku or "").strip():
        raise ModelingError("simulation_config_sku_required")
    hw = normalize_hw_family(arch or sku)
    if hw not in {"a2", "a3", "a5"}:
        raise ModelingError(f"simulation_config_unsupported_arch:{arch!r}:{sku!r}")
    role_norm = str(role or "colocated").strip().lower()
    if role_norm not in _VALID_ROLES:
        raise ModelingError(f"simulation_config_invalid_role:{role!r}")

    spec = model if model is not None else _load_structural_model_if_str(model_or_key)
    caps = extract_model_capabilities(spec, arch=hw)
    coll = shipping_collective_policy(hw, role=role_norm, model=spec)
    env: dict[str, str] = {"VLLM_USE_V1": "1"}
    if coll and coll.get("process_env_mode"):
        env["HCCL_OP_EXPANSION_MODE"] = str(coll["process_env_mode"])

    has_latent_mla_rope = bool(
        spec is not None and any(
            p.attention == "mla"
            and p.mla_variant == "latent_mla"
            and p.cache_kind == "dense"
            and int(p.qk_rope_head_dim or 0) > 0
            and not bool(p.mla_use_nope)
            for p in spec.layer_plans
        )
    )
    mlapo = bool(has_latent_mla_rope and role_norm != "pd_prefill")
    if mlapo:
        env["VLLM_ASCEND_ENABLE_MLAPO"] = "1"

    has_latent_moe = bool(
        spec is not None and any(int(p.expert_latent_size or 0) > 0 for p in spec.layer_plans)
    )
    tutorial_flags: dict[str, Any] = {}
    if hw == "a5" and caps["has_moe"] and (caps["ffn_activation"] == "situ" or has_latent_moe):
        tutorial_flags["enable_fused_mc2"] = FUSED_MC2_DISABLED
    if hw == "a5" and caps["has_kpool_indexer"]:
        tutorial_flags["kv_cache_dtype"] = "fp8"
        tutorial_flags["indexer_kv_dtype"] = "fp8"
    elif hw in A2_A3_INT_ONLY_FAMILIES and caps["has_kpool_indexer"]:
        tutorial_flags["kv_cache_dtype"] = "int8"
    mso = bool(
        caps["has_moe"]
        and caps["has_shared_expert"]
        and int(tutorial_flags.get("enable_fused_mc2") or 0) <= FUSED_MC2_DISABLED
    )
    engram_offload = bool(spec is not None and getattr(spec, "engram", None) is not None)
    spec_meta = dict(getattr(spec, "metadata", ()) or ()) if spec is not None else {}
    if spec_meta.get("cache_allocator") == "paired_sfa":
        kv_block_size: int | None = PAIRED_SFA_DOC_BLOCK_SIZE
    elif hw == "a5" and caps["has_csa2_compressed_indexer"] and not engram_offload:
        kv_block_size = COMPRESSED_KV_BLOCK_SIZE
    else:
        kv_block_size = None
    default_quant: str | None = None
    if hw in A2_A3_INT_ONLY_FAMILIES and spec is not None and spec.quant is not None:
        q_fmts = {
            str(spec.quant.weight_format or "").strip().lower(),
            str(spec.quant.expert_weight_format or "").strip().lower(),
            str(spec.quant.name or "").strip().lower(),
        }
        if (
            caps["has_mxfp_weights"]
            or any(any(tok in fmt for tok in A5_ONLY_QUANT_TOKENS) for fmt in q_fmts)
        ):
            default_quant = "w4a8" if (caps["has_moe"] and caps["has_fp4_or_w4a8_experts"]) else "w8a8"
    q_basis = (
        "hardware_isa_int_quant_fallback" if default_quant is not None
        else "preserve_caller_no_documented_hardware_quantization"
    )
    return {
        "model": str(spec.name) if spec is not None else str(model_or_key or ""),
        "sku": str(sku).strip(),
        "hardware_arch": hw,
        "role": role_norm,
        "quant": default_quant,
        "quantization_basis": q_basis,
        "coverage": "supported_simulation_subset",
        "graph_capture_mode": "full",
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "sp": True,
        "mlapo": mlapo,
        "mtp_proposal_tokens": NO_MTP_PROPOSAL_TOKENS,
        "multistream_overlap_shared_expert": mso,
        "engram_cpu_offload": engram_offload,
        "kv_block_size": kv_block_size,
        "env": env,
        "tutorial_flags": tutorial_flags,
        "shipping_collective_policy": coll,
        "source": "structural_vllm_ascend_defaults",
    }


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
    "VLLM_ASCEND_REPO",
    "VLLM_ASCEND_STACK_ID",
    "ManifestBinding",
    "default_checkpoint_repo",
    "extract_model_capabilities",
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
