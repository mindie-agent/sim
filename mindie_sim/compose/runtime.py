"""Runtime profile, lowering selection, and compilation spine helpers for ``compose``.

[Role]
  Resolve vLLM-Ascend documented runtime subsets, sequence-parallel eligibility,
  graph capture buckets, orthogonal compilation flags (``graph_capture_mode`` vs
  ``enable_torch_compile`` / ``enable_npugraph_ex``), and binding-origin stamps.

[Invariants]
  * ``do_not_compile = True`` (``supports_torch_compile == False``) keeps
    ``graph_capture_mode="full"`` intact while setting ``enable_torch_compile=False``.
  * ``graph_capture_mode != "full"`` disables ``enable_npugraph_ex``,
    ``enable_static_kernel``, and ``enable_super_kernel``.
  * Bindings installed by ``compose_case`` / ``apply_va_recommended`` stamp
    ``recipe_binding_origins`` as ``"stack_profile"`` so caller pins stay distinct.
  * Zero process-wide LRU contract caches, zero ``is_a5``-only AttnRes gates,
    zero ``except ModelingError -> fallback_note`` swallows, zero missing-SKU
    defaults.

[Timeouts & Locks]
  Pure in-memory dictionary and metadata resolution; no I/O or locks.

[Side-Effect Policy]
  Returns new dictionaries or mutates only caller-supplied builder dicts in place.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindie_sim.prediction.feedforward import resolve_moe_route_policy
from mindie_sim.prediction.offload.policy import resolve_sku_host_dma_rates
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.scenario import ScenarioContract
from mindie_sim.schema.stack import RuntimeQuantPlan, StackLoweringPlan

MX_GROUP_SIZE = 32
MX_SCALE_PACK = 2
KDA_DECODE_MAX_QUERY_TOKENS = 8
GDN_A5_MAX_QUERY_TOKENS = 8
CAPTURE_BASE_KEYS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PIECEWISE_CAPTURE_BASE_KEYS = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
)
MIN_DEFAULT_MAX_SEQS = 16
TRACKED_KERNEL_SECTIONS = ("families", "recipes", "lowerings", "programs")
FULL_DECODE_CUDAGRAPH_MODE = "FULL_DECODE_ONLY"
FULL_CUDAGRAPH_MODES = frozenset({"FULL", FULL_DECODE_CUDAGRAPH_MODE})
_INT_WEIGHT_FORMATS = frozenset({"int8", "int4"})
_FP4_ACT_FORMATS = frozenset({"fp4", "fp4_e2m1", "mxfp4"})
_NORM_QUANT_FAMILY_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gate_up_projection", ("dense_mlp", "shared_expert")),
    ("qkv_projection", ("attn.qkv", "attn")),
    ("latent_projection", ("mla", "mla.q_a", "mla.kv_a")),
    ("index_projection", ("indexer", "indexer.wq")),
    ("router", ("router",)),
)
_WEIGHT_NZ_ARCHES = frozenset({"a2", "a3"})


def register_composed_contracts(
    *,
    quant_plan: RuntimeQuantPlan,
    lowering_plan: StackLoweringPlan,
    scenario_contract: ScenarioContract | None = None,
) -> None:
    """Unplugged socket: 64-entry process-wide mutable LRU caches purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: attach frozen ScenarioContract, RuntimeQuantPlan, and "
        "StackLoweringPlan directly on immutable CaseSpec/RuntimeSpec fields "
        "instead of process-wide 64-entry LRU caches"
    )


def resolve_case_contracts(
    case_or_dict: Any,
) -> tuple[ScenarioContract | None, RuntimeQuantPlan | None, StackLoweringPlan | None]:
    """Unplugged socket: 64-entry process-wide mutable LRU caches purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: read ScenarioContract, RuntimeQuantPlan, and "
        "StackLoweringPlan directly from CaseSpec/RuntimeSpec fields instead of "
        "process-wide LRU caches"
    )


def summarize_single_spine_metadata(
    *,
    stack_id: str,
    stack_track: str,
    workload_domain: str,
    work_unit_kind: str,
    deployment_mode: str,
    deployment_mode_origin: str,
    checkpoint_map: Any,
    quant_plan: RuntimeQuantPlan,
    lowering_plan: StackLoweringPlan,
    scenario_contract: ScenarioContract | None = None,
) -> dict[str, Any]:
    """Build a compact JSON-serializable summary of the single-spine contracts."""
    return {
        "stack_id": stack_id,
        "stack_track": stack_track,
        "workload_domain": workload_domain,
        "work_unit_kind": work_unit_kind,
        "deployment_mode": deployment_mode,
        "deployment_mode_origin": deployment_mode_origin,
        "checkpoint_id": checkpoint_map.checkpoint_id,
        "checkpoint_digest": checkpoint_map.digest(),
        "checkpoint_format": checkpoint_map.source_format,
        "checkpoint_origin": checkpoint_map.provenance.origin,
        "quant_plan_digest": quant_plan.plan_digest,
        "quant_provenance": quant_plan.provenance,
        "kv_cache_abi_count": len(quant_plan.kv_cache_abi),
        "index_cache_abi_count": len(quant_plan.index_cache_abi),
        "recurrent_state_abi_count": len(quant_plan.recurrent_state_abi),
        "prepared_weights": [item.allocation_id for item in quant_plan.prepared_weights],
        "packed_groups_count": len(quant_plan.packed_groups),
        "lowering_plan_digest": lowering_plan.plan_digest,
        "allowed_rewrite_tracks": list(lowering_plan.allowed_rewrite_tracks),
        "recipe_bindings_count": len(lowering_plan.recipe_bindings),
        "scenario_contract_digest": scenario_contract.digest() if scenario_contract is not None else "",
    }


def uses_integer_weights(model_spec: ModelSpec) -> bool:
    """Return True when resident target weights use integer quantization (W8/W4)."""
    quant = getattr(model_spec, "quant", None)
    if quant is None:
        return False
    return (
        str(getattr(quant, "weight_format", "") or "").strip().lower() in _INT_WEIGHT_FORMATS
        or str(getattr(quant, "expert_weight_format", "") or "").strip().lower() in _INT_WEIGHT_FORMATS
    )


def apply_arch_weight_nz_policy(
    additional: dict[str, Any],
    *,
    arch: str,
    model_spec: ModelSpec,
) -> None:
    """Enable FRACTAL_NZ weight layout only on A2/A3 integer weights; never on A5 MXFP."""
    if arch in _WEIGHT_NZ_ARCHES and uses_integer_weights(model_spec):
        additional.setdefault("enable_weight_nz_layout", True)
    elif arch == "a5":
        additional.pop("enable_weight_nz_layout", None)


def sync_moe_route_on_runtime(
    runtime_dict: dict[str, Any],
    *,
    arch: str,
    model_spec: ModelSpec,
    tp: int,
    strict_level3: bool = False,
) -> dict[str, Any]:
    """Enforce upstream ``enable_fused_mc2`` vs MSO mutex and level-3 legality on runtime."""
    meta = dict(runtime_dict.get("metadata") or {})
    additional = dict(meta.get("additional_config") or {})
    prev_route = meta.get("moe_route") if isinstance(meta.get("moe_route"), Mapping) else {}
    requested_fused = additional.get("enable_fused_mc2")
    if requested_fused is None and prev_route.get("requested_enable_fused_mc2") is not None:
        requested_fused = prev_route.get("requested_enable_fused_mc2")
    requested_mso = bool(
        runtime_dict.get("multistream_overlap_shared_expert", False)
        or meta.get("multistream_overlap_shared_expert", False)
        or prev_route.get("requested_multistream_overlap_shared_expert", False)
    )
    route = resolve_moe_route_policy(
        arch=arch,
        model=model_spec,
        requested_fused_mc2=requested_fused,
        requested_mso=requested_mso,
        tp=int(tp),
        enable_shared_expert_dp=bool(meta.get("enable_shared_expert_dp", False)),
        strict_level3=strict_level3,
    )
    if (prev_route.get("requested_enable_fused_mc2") is not None
            and requested_fused == prev_route.get("effective_enable_fused_mc2")
            and prev_route.get("requested_enable_fused_mc2") != requested_fused):
        route["requested_enable_fused_mc2"] = prev_route["requested_enable_fused_mc2"]
        if not route["shared_fusion_disabled_reason"] and prev_route.get("shared_fusion_disabled_reason"):
            route["shared_fusion_disabled_reason"] = prev_route["shared_fusion_disabled_reason"]
    eff_fused = route["effective_enable_fused_mc2"]
    if eff_fused is not None:
        additional["enable_fused_mc2"] = int(eff_fused)
    if additional:
        meta["additional_config"] = additional
    if route["effective_multistream_overlap_shared_expert"]:
        runtime_dict["multistream_overlap_shared_expert"] = True
        meta["multistream_overlap_shared_expert"] = True
    else:
        runtime_dict.pop("multistream_overlap_shared_expert", None)
        meta.pop("multistream_overlap_shared_expert", None)
    meta["moe_route"] = route
    runtime_dict["metadata"] = meta
    return route


def capture_keys(max_num_seqs: int, mtp_proposal_tokens: int = 0) -> list[int]:
    """Capture buckets up to ``max_num_seqs * (1 + mtp)`` including strides."""
    sequences = max(1, int(max_num_seqs))
    stride = max(1, 1 + int(mtp_proposal_tokens))
    keys = set(CAPTURE_BASE_KEYS)
    keys.update(stride * step for step in range(1, sequences + 1))
    keys.add(sequences * stride)
    return sorted(keys)


def va_recommended_config(
    model: str, hardware: str = "a3", *, role: str | None = None,
) -> dict[str, Any]:
    """Unplugged socket: model-name profile lookup purged per AG014."""
    raise NotImplementedError(
        "LEGACY_PURGED: retire model-name va_recommended_config lookup; derive "
        "runtime defaults strictly from structural ModelSpec, Hardware, and role"
    )


def _supports_sequence_parallel(
    model_spec: ModelSpec, profile: Mapping[str, Any], *,
    tp: int, ep: int, pp: int, pcp: int, notes: list[dict[str, Any]],
) -> bool:
    """Structural sequence-parallel applicability without silent ModelingError swallowing."""
    if profile.get("sp") is False:
        return False
    if not (pcp >= 1 and pp >= 1 and tp > 1 and ep >= tp and ep % tp == 0):
        return False
    raise NotImplementedError(
        "LEGACY_PURGED: evaluate sequence_parallel eligibility structurally without "
        "swallowing ModelingError into fallback_note"
    )


def filter_fp4_norm_quant_domain(
    kernels: Mapping[str, Any], quant: Any,
) -> dict[str, Any]:
    """Exclude FP4-activation consumers from ``norm_quant`` fusion domains."""
    out = dict(kernels)
    if quant is None:
        return out
    domains = dict(out.get("fusion_domains") or {})
    nq = domains.get("norm_quant")
    if not isinstance(nq, Mapping):
        return out
    blocked: set[str] = set()
    for family, categories in _NORM_QUANT_FAMILY_CATEGORIES:
        if any(
            str(quant.for_category(cat).activation_format).strip().lower() in _FP4_ACT_FORMATS
            for cat in categories
        ):
            blocked.add(family)
    if not blocked:
        return out
    consumers = [
        item for item in (nq.get("consumers") or ())
        if not (isinstance(item, Mapping) and str(item.get("family", "")) in blocked)
    ]
    domains["norm_quant"] = {**nq, "consumers": consumers}
    out["fusion_domains"] = domains
    return out


def _execution_policy(case: Mapping[str, Any]) -> dict[str, Any]:
    """Resolved execution policy stored by ``compose_case`` on the declaration."""
    runtime = case.get("runtime") if isinstance(case, Mapping) else None
    if not isinstance(runtime, Mapping):
        return {}
    policy = dict(runtime.get("metadata") or {}).get("execution_policy")
    return dict(policy) if isinstance(policy, Mapping) else {}


def _resolve_va_lowerings(
    model_spec: ModelSpec, parallel: Mapping[str, int], sku: str,
    lowering_passes: tuple[str, ...], *, notes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Unplugged socket: is_a5-only AttnRes, blind MLA Prolog, ModelingError swallow, and CANN compressor hardcode purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: wire cross-arch A3/A5 AttnResFwd, canonical M1 5-branch MLA Prolog "
        "without swallowing ModelingError into fallback_note, and M12 structural KV compressor selection"
    )


def snapshot_kernel_keys(kernels: Mapping[str, Any] | None) -> dict[str, set[str]]:
    raw = kernels or {}
    return {
        section: set((raw.get(section) or {}).keys())
        for section in TRACKED_KERNEL_SECTIONS
    }


def stamp_stack_profile_origins(
    runtime_dict: dict[str, Any],
    *,
    prior_keys: Mapping[str, set[str]] | None = None,
) -> None:
    """Stamp ``stack_profile`` origin on kernel bindings added by compose/profile."""
    from mindie_sim.prediction.implementation_selection import record_binding_origin

    kernels = runtime_dict.get("kernels")
    if not isinstance(kernels, Mapping):
        return
    meta = dict(runtime_dict.get("metadata") or {})
    baseline = prior_keys or {}
    changed = False
    for section in TRACKED_KERNEL_SECTIONS:
        sec_map = kernels.get(section)
        if not isinstance(sec_map, Mapping):
            continue
        existing = baseline.get(section, set())
        for name in sorted(set(sec_map.keys()) - existing):
            meta = record_binding_origin(meta, section, str(name), "stack_profile")
            changed = True
    if changed:
        runtime_dict["metadata"] = meta


def finalize_compilation_spine(
    runtime_dict: dict[str, Any],
    *,
    raw_eager: bool,
) -> None:
    """Enforce orthogonal ``graph_capture_mode`` vs ``torch.compile``/``npugraph_ex``."""
    capture_mode = "none" if raw_eager else str(runtime_dict.get("graph_capture_mode") or "none")
    runtime_dict["graph_capture_mode"] = capture_mode
    runtime_dict["graph_mode"] = capture_mode
    raw_comp = runtime_dict.get("compilation_config")
    if capture_mode == "full":
        comp = dict(raw_comp) if isinstance(raw_comp, Mapping) else {}
        if str(comp.get("cudagraph_mode") or "").upper() not in FULL_CUDAGRAPH_MODES:
            comp["cudagraph_mode"] = FULL_DECODE_CUDAGRAPH_MODE
        runtime_dict["compilation_config"] = comp
        captured_mode = str(comp["cudagraph_mode"]).upper()
    else:
        if isinstance(raw_comp, Mapping):
            comp = dict(raw_comp)
            if str(comp.get("cudagraph_mode") or "").upper() in FULL_CUDAGRAPH_MODES:
                comp.pop("cudagraph_mode", None)
            if comp:
                runtime_dict["compilation_config"] = comp
            else:
                runtime_dict.pop("compilation_config", None)
        captured_mode = "PIECEWISE" if capture_mode == "piecewise" else "NONE"
    compile_on = False if raw_eager else bool(runtime_dict.get("enable_torch_compile", True))
    raw_npugraph = runtime_dict.get("enable_npugraph_ex")
    npugraph_on = bool(
        compile_on
        and capture_mode == "full"
        and (True if raw_npugraph is None else bool(raw_npugraph))
    )
    static_on = bool(npugraph_on and runtime_dict.get("enable_static_kernel", False))
    super_on = bool(npugraph_on and runtime_dict.get("enable_super_kernel", False))
    runtime_dict["enable_torch_compile"] = compile_on
    runtime_dict["enable_npugraph_ex"] = npugraph_on
    runtime_dict["enable_static_kernel"] = static_on
    runtime_dict["enable_super_kernel"] = super_on
    meta = dict(runtime_dict.get("metadata") or {})
    policy = dict(meta.get("execution_policy") or {})
    policy.update({
        "graph_capture_mode": capture_mode,
        "captured_execution_mode": captured_mode,
        "enable_torch_compile": compile_on,
        "enable_npugraph_ex": npugraph_on,
        "enable_static_kernel": static_on,
        "enable_super_kernel": super_on,
    })
    meta["execution_policy"] = policy
    runtime_dict["metadata"] = meta


def bind_sku_host_dma_rates(
    machine: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind SKU-calibrated L0/profile PCIe H2D/D2H & host DDR rates onto ``machine``."""
    host_rates = resolve_sku_host_dma_rates(machine)
    missing = [
        name
        for name, key in (("h2d", "h2d_bps"), ("host_memory", "host_memory_bps"))
        if not host_rates.get(key)
    ]
    if missing:
        raise ModelingError(
            "missing_required_hardware_host_dma_rate:" + ",".join(missing)
        )
    updated = dict(machine)
    dev = dict(updated.get("device") or {})
    rates = dict(dev.get("rates") or {})
    rates.setdefault("h2d", float(host_rates["h2d_bps"]))
    if host_rates.get("d2h_bps"):
        rates.setdefault("d2h", float(host_rates["d2h_bps"]))
    rates.setdefault("host_memory", float(host_rates["host_memory_bps"]))
    if host_rates.get("host_vector_ops_per_s"):
        rates.setdefault("host_vector", float(host_rates["host_vector_ops_per_s"]))
    dev["rates"] = rates
    updated["device"] = dev
    if isinstance(updated.get("devices"), list) and updated["devices"]:
        patched_devices = []
        for entry in updated["devices"]:
            if isinstance(entry, Mapping):
                ed = dict(entry)
                ed_rates = dict(ed.get("rates") or {})
                for k, v in rates.items():
                    ed_rates.setdefault(k, v)
                ed["rates"] = ed_rates
                patched_devices.append(ed)
            else:
                patched_devices.append(entry)
        updated["devices"] = patched_devices
    return updated, host_rates


def needs_activation_quantization(model_spec: Any) -> bool:
    """Return True when top-level or any category override quantizes activations."""
    quant = getattr(model_spec, "quant", None)
    if quant is None:
        return False
    act_dtype = str(getattr(model_spec, "activation_dtype", "bf16"))
    if str(quant.activation_format) != act_dtype:
        return True
    for _, override in getattr(quant, "category_overrides", ()) or ():
        for key, value in override:
            if key == "activation_format" and str(value) != act_dtype:
                return True
    return False


def apply_va_recommended(case: Mapping[str, Any]) -> dict[str, Any]:
    """Unplugged socket: missing-SKU default, model-name routing, legacy tracks, and role-blind mutation purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: require explicit SKU, deployment role, canonical StackTrack, "
        "and structural ModelSpec predicates in apply_va_recommended (zero Ascend910_9392 "
        "fallback, zero ModelingError swallow)"
    )
