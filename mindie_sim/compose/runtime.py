"""Runtime profile, lowering selection, and compilation spine helpers for ``compose``.

[Role]
  Resolve vLLM-Ascend documented runtime subsets, sequence-parallel eligibility,
  graph capture buckets, orthogonal compilation flags (``graph_capture_mode`` vs
  ``enable_torch_compile`` / ``enable_npugraph_ex``), stateless single-spine
  contract rehydration, and binding-origin stamps.

[Invariants]
  * ``do_not_compile = True`` (``supports_torch_compile == False``) keeps
    ``graph_capture_mode="full"`` intact while setting ``enable_torch_compile=False``.
  * ``graph_capture_mode != "full"`` disables ``enable_npugraph_ex``,
    ``enable_static_kernel``, and ``enable_super_kernel``.
  * Bindings installed by ``compose_case`` / ``apply_va_recommended`` stamp
    ``recipe_binding_origins`` as ``"stack_profile"`` so caller pins stay distinct.
  * Zero process-wide LRU contract caches, zero ``is_a5``-only AttnRes gates,
    zero ``except ModelingError -> fallback_note`` swallows, zero missing-SKU
    defaults, zero AG014 model-name profile lookups.

[Timeouts & Locks]
  Pure in-memory dictionary and metadata resolution; no I/O or locks.

[Side-Effect Policy]
  Returns new dictionaries or mutates only caller-supplied builder dicts in place.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.prediction.feedforward import resolve_moe_route_policy
from mindie_sim.prediction.offload.policy import resolve_sku_host_dma_rates
from mindie_sim.prediction.operators.mla_prolog import resolve_mla_preprocess_mode
from mindie_sim.schema.checkpoint import (
    CacheQuantRule,
    CheckpointProvenance,
    CheckpointQuantMap,
    LayerQuantException,
    TensorPresenceDomain,
    TensorQuantRule,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.scenario import ScenarioContract
from mindie_sim.schema.stack import STACK_TRACKS, RuntimeQuantPlan, StackLoweringPlan

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
_VALID_ROLES = frozenset({"colocated", "pd_disaggregated", "pd_prefill", "pd_decode"})
_VALID_SP_SCOPES = frozenset({"body", "draft"})


def _arch_key(sku_or_arch: str) -> str:
    from mindie_sim.stacks.vllm_ascend.profiles import normalize_hw_family

    hw = normalize_hw_family(sku_or_arch)
    if hw not in {"a2", "a3", "a5"}:
        raise ModelingError(f"unsupported_hardware_arch:{sku_or_arch!r}")
    return hw


def _checkpoint_map_from_dict(raw: Mapping[str, Any]) -> CheckpointQuantMap:
    prov_raw = raw.get("provenance") or {}
    prov = CheckpointProvenance(
        checkpoint_id=str(prov_raw.get("checkpoint_id") or ""),
        source_format=prov_raw["source_format"],
        origin=prov_raw.get("origin", "verified_checkpoint_descriptor"),
        descriptor_path=str(prov_raw.get("descriptor_path") or ""),
        source_sha256=str(prov_raw.get("source_sha256") or ""),
        source_revision=str(prov_raw.get("source_revision") or ""),
    )
    rules: list[TensorQuantRule] = []
    for r in raw.get("rules") or ():
        pres_raw = r.get("presence")
        pres = (
            TensorPresenceDomain(
                scope=pres_raw["scope"],
                layer_indices=tuple(pres_raw.get("layer_indices") or ()),
                expert_indices=tuple(pres_raw.get("expert_indices") or ()),
            )
            if isinstance(pres_raw, Mapping) else None
        )
        rules.append(TensorQuantRule(
            pattern=str(r["pattern"]),
            raw_tag=str(r["raw_tag"]),
            presence=pres,
            group_size=tuple(r.get("group_size") or ()),
            scale_dtype=str(r.get("scale_dtype") or ""),
            offset_dtype=str(r.get("offset_dtype") or ""),
        ))
    exceptions = tuple(
        LayerQuantException(
            pattern=str(exc["pattern"]),
            scope=exc["scope"],
            layer_indices=tuple(exc.get("layer_indices") or ()),
            raw_tag=str(exc["raw_tag"]),
            expert_indices=tuple(exc.get("expert_indices") or ()),
        )
        for exc in (raw.get("layer_exceptions") or ())
    )
    cr = raw.get("cache_rule") or {}
    cache_rule = CacheQuantRule(
        kv_tag=str(cr.get("kv_tag") or ""),
        index_tag=str(cr.get("index_tag") or ""),
        kv_scale_dtype=str(cr.get("kv_scale_dtype") or ""),
        index_scale_dtype=str(cr.get("index_scale_dtype") or ""),
        kv_offset_dtype=str(cr.get("kv_offset_dtype") or ""),
        layer_scoped_kv=tuple((int(k), str(v)) for k, v in (cr.get("layer_scoped_kv") or ())),
        layer_scoped_index=tuple((int(k), str(v)) for k, v in (cr.get("layer_scoped_index") or ())),
        provenance=cr.get("provenance", "native"),
    )
    hf_pairs = tuple((str(k), v) for k, v in (raw.get("hf_quant_config") or ()))
    return CheckpointQuantMap(
        provenance=prov,
        native_float_dtype=str(raw.get("native_float_dtype") or ""),
        model_quant_type=str(raw.get("model_quant_type") or ""),
        rules=tuple(rules),
        layer_exceptions=exceptions,
        auxiliary_buffers=tuple(str(x) for x in (raw.get("auxiliary_buffers") or ())),
        hf_quant_config=hf_pairs,
        ignored_modules=tuple(str(x) for x in (raw.get("ignored_modules") or ())),
        cache_rule=cache_rule,
    )


def _encode_checkpoint_for_metadata(checkpoint_map: CheckpointQuantMap) -> dict[str, Any]:
    from mindie_sim.catalog.checkpoint_reader import lookup_curated_checkpoint

    cid = str(checkpoint_map.checkpoint_id or "")
    digest = checkpoint_map.digest()
    if cid:
        curated = lookup_curated_checkpoint(cid)
        if curated is not None and curated.digest() == digest:
            return {"ref": cid, "digest": digest}
    return {"payload": checkpoint_map.to_dict(), "digest": digest}


def _rehydrate_checkpoint_from_metadata(encoded: Mapping[str, Any]) -> CheckpointQuantMap:
    from mindie_sim.catalog.checkpoint_reader import lookup_curated_checkpoint

    ref = str(encoded.get("ref") or "")
    if ref:
        curated = lookup_curated_checkpoint(ref)
        if curated is not None:
            return curated
    payload = encoded.get("payload")
    if not isinstance(payload, Mapping):
        raise ModelingError("single_spine_checkpoint_payload_missing")
    return _checkpoint_map_from_dict(payload)


def register_composed_contracts(
    *,
    quant_plan: RuntimeQuantPlan,
    lowering_plan: StackLoweringPlan,
    scenario_contract: ScenarioContract | None = None,
) -> None:
    """Stateless contract validation hook (zero process-wide mutable LRU caches)."""
    if not isinstance(quant_plan, RuntimeQuantPlan):
        raise ModelingError("register_composed_contracts_invalid_quant_plan")
    if not isinstance(lowering_plan, StackLoweringPlan):
        raise ModelingError("register_composed_contracts_invalid_lowering_plan")
    if scenario_contract is not None and not isinstance(scenario_contract, ScenarioContract):
        raise ModelingError("register_composed_contracts_invalid_scenario_contract")


def resolve_case_contracts(
    case_or_dict: Any,
) -> tuple[ScenarioContract | None, RuntimeQuantPlan | None, StackLoweringPlan | None]:
    """Deterministically rehydrate frozen single-spine contracts directly from ``case_or_dict``."""
    from mindie_sim.compose import _build_scenario_contract
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.stacks.registry import resolve_stack_for_scenario

    if isinstance(case_or_dict, Mapping):
        runtime_raw = case_or_dict.get("runtime")
        meta = dict(runtime_raw.get("metadata") or {}) if isinstance(runtime_raw, Mapping) else {}
        single_spine = meta.get("single_spine")
        if not isinstance(single_spine, Mapping):
            return None, None, None
        parsed = parse_case(case_or_dict)
    else:
        runtime_obj = getattr(case_or_dict, "runtime", None)
        meta = dict(getattr(runtime_obj, "metadata", None) or {})
        single_spine = meta.get("single_spine")
        if not isinstance(single_spine, Mapping):
            return None, None, None
        parsed = case_or_dict

    inputs = single_spine.get("contract_inputs")
    if not isinstance(inputs, Mapping):
        return None, None, None

    stack_id = str(single_spine.get("stack_id") or "")
    adapter_track = str(single_spine.get("stack_track") or "shipping")
    workload_domain = str(single_spine.get("workload_domain") or "llm")
    work_unit_kind = str(single_spine.get("work_unit_kind") or "token")
    eff_role = str(inputs.get("eff_role") or single_spine.get("deployment_mode") or "colocated")
    eff_phase = str(inputs.get("eff_phase") or "decode")
    req_opts = dict(inputs.get("requested_options") or {})
    field_origins = dict(inputs.get("field_origins") or {})
    ckpt_encoded = inputs.get("checkpoint")
    if not isinstance(ckpt_encoded, Mapping):
        raise ModelingError("single_spine_checkpoint_inputs_missing")
    checkpoint_map = _rehydrate_checkpoint_from_metadata(ckpt_encoded)

    hw_selector = str(parsed.machine.architecture or parsed.machine.sku or "ascend")
    adapter = resolve_stack_for_scenario(
        requested_stack=stack_id or None,
        workload_domain=workload_domain,
        work_unit=work_unit_kind,
        hardware_family=hw_selector,
    )
    resolved_opts = adapter.resolve_options(
        parsed.model, parsed.machine, parsed.runtime.parallel,
        role=eff_role, track=adapter_track,  # type: ignore[arg-type]
        requested_options=req_opts, field_origins=field_origins,
    )
    quant_plan = adapter.resolve_quant_instantiation(
        parsed.model, checkpoint_map, parsed.machine, parsed.runtime.parallel,
        resolved_opts, role=eff_role, phase=eff_phase,
    )
    lowering_plan = adapter.select_lowering_plan(
        parsed.model, quant_plan, parsed.machine, parsed.runtime.parallel, resolved_opts,
    )
    scenario_contract = _build_scenario_contract(
        model_spec=parsed.model,
        checkpoint_map=checkpoint_map,
        hw_obj=parsed.machine,
        stack_id=stack_id,
        adapter_track=adapter_track,  # type: ignore[arg-type]
        workload_domain=workload_domain,
        work_unit_kind=work_unit_kind,
        resolved_opts=resolved_opts,
        quant_plan=quant_plan,
        lowering_plan=lowering_plan,
        eff_role=eff_role,
        norm_role=inputs.get("norm_role"),
        prompt_tokens=inputs.get("prompt_tokens"),
        ttft_slo_ms=inputs.get("ttft_slo_ms"),
        tpot_slo_ms=inputs.get("tpot_slo_ms"),
        resolved_max_seqs=int(inputs.get("resolved_max_seqs") or parsed.runtime.max_num_seqs),
        concurrency=inputs.get("concurrency"),
        canonical_track=str(inputs.get("canonical_track") or adapter_track),
        mtp_tokens=int(inputs.get("mtp_tokens") or 0),
        require_scenario=bool(inputs.get("require_scenario", False)),
    )
    return scenario_contract, quant_plan, lowering_plan


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
    contract_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact JSON-serializable summary & stateless rehydration payload."""
    summary: dict[str, Any] = {
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
    if contract_inputs is not None:
        encoded = dict(contract_inputs)
        encoded["checkpoint"] = _encode_checkpoint_for_metadata(checkpoint_map)
        summary["contract_inputs"] = encoded
    return summary


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
    model: ModelSpec | str, hardware: str = "a3", *, role: str | None = None,
) -> dict[str, Any]:
    """Derive vLLM-Ascend runtime defaults purely structurally from ``ModelSpec``, hardware, and role."""
    from mindie_sim.compose.inputs import resolve_hardware_sku
    from mindie_sim.model_catalog import load_model
    from mindie_sim.stacks.vllm_ascend.profiles import simulation_config

    hw_raw = str(hardware or "").strip()
    if not hw_raw:
        raise ModelingError("va_recommended_config_hardware_required")
    sku = resolve_hardware_sku(hw_raw)
    arch = _arch_key(sku)
    role_norm = str(role or "colocated").strip().lower()
    if role_norm not in _VALID_ROLES:
        raise ModelingError(f"va_recommended_config_invalid_role:{role!r}")
    spec = model if isinstance(model, ModelSpec) else load_model(str(model))
    return simulation_config(spec, sku, arch, role=role_norm, model=spec)


def _supports_sequence_parallel(
    model_spec: ModelSpec, profile: Mapping[str, Any], *,
    tp: int, ep: int, pp: int, pcp: int, notes: list[dict[str, Any]],
) -> bool:
    """Structural sequence-parallel applicability without catching/swallowing ModelingError."""
    del notes
    if profile.get("sp") is False:
        return False
    if not (pcp >= 1 and pp >= 1 and tp > 1 and ep >= tp and ep % tp == 0):
        return False
    plans = tuple(getattr(model_spec, "layer_plans", ()))
    if not plans:
        return False
    return all(getattr(layer, "scope", "body") in _VALID_SP_SCOPES for layer in plans)


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
    """Select ABI-grounded vLLM-Ascend lowerings (cross-arch AttnRes, M1 MLA Prolog, M12 compressor)."""
    from mindie_sim.catalog.paired_sfa import execution
    from mindie_sim.schema.parallel import ParallelSpec

    del notes
    arch = _arch_key(sku)
    arch_upper = arch.upper()
    is_a5 = arch == "a5"
    registry = default_abi_registry()
    p_cfg = ParallelSpec(
        dp=int(parallel.get("dp", 1)), tp=int(parallel.get("tp", 1)),
        pp=int(parallel.get("pp", 1)), ep=int(parallel.get("ep", 1)),
        cp=int(parallel.get("cp", parallel.get("dcp", 1))), pcp=int(parallel.get("pcp", 1)),
    )
    lowerings: dict[str, Any] = {}
    kda_layers = [p for p in model_spec.layer_plans if p.attention == "kda"]
    if (kda_layers and p_cfg.pcp >= 1
            and (p_cfg.cp == 1 or execution(model_spec).get("recurrent_context_sharded") is False)):
        kda_binding = ("vllm_ascend_recurrent_kda_arch35" if is_a5
                       else "vllm_ascend_recurrent_kda_arch22")
        kda_options = {}
        if is_a5:
            from mindie_sim.prediction.operators.kda.kda_source import resolve as resolve_kda_source
            bounds = {getattr(p, "kda_gate_lower_bound", None) for p in kda_layers}
            if len(bounds) == 1:
                kda_options = resolve_kda_source(
                    kda_layers[0], arch="A5", phase="decode", kernels={},
                    activation_dtype=model_spec.activation_dtype) or {}
        lowerings["kda"] = {
            "cases": [{
                "domain": {"phases": ["decode"],
                           "max_query_tokens": {"max": KDA_DECODE_MAX_QUERY_TOKENS}},
                "binding": ({"implementation": kda_binding, "options": kda_options}
                            if kda_options else kda_binding),
            }],
            "otherwise": "analytical",
        }

    # M7/M8: AttnResFwd ships across A2/A3/A5 grounded in ProbedAbiRegistry (never is_a5-only).
    if (
        any(int(getattr(p, "residual_block_size", 0) or 0) > 0 for p in model_spec.layer_plans)
        and registry.is_shipping_available("AttnResFwd", arch_upper)
    ):
        lowerings["attention_residual"] = "vllm_ascend_attn_res_prefix"

    if any(p.attention == "gdn" for p in model_spec.layer_plans) and p_cfg.pcp >= 1:
        opts = {"recurrent_implementation": "arch35_fp32"} if is_a5 else {}
        binding = {"implementation": "vllm_ascend_gdn_decode", "options": opts}
        lowerings["gdn_decode"] = {
            "cases": [
                {"domain": {"phases": ["decode"], "min_query_tokens": {"min": 1},
                            "max_query_tokens": {"max": 1}},
                 "binding": binding},
                {"domain": {"phases": ["decode"], "min_query_tokens": {"min": 2},
                            **({"max_query_tokens": {"max": GDN_A5_MAX_QUERY_TOKENS}} if is_a5 else {})},
                 "binding": binding},
            ],
            "otherwise": "analytical",
        }

    # M1: 5-branch MLA Prolog predicate grounded in ProbedAbiRegistry (zero ModelingError swallow).
    mla_layers = [
        p for p in model_spec.layer_plans
        if p.attention == "mla" and int(p.q_lora_rank or 0) > 0 and p.mla_variant != "latent_mqa"
    ]
    if mla_layers:
        quant = getattr(model_spec, "quant", None)
        act_dtype = str(getattr(model_spec, "activation_dtype", "") or "bf16").lower()
        weight_quant = str(
            getattr(quant, "weight_format", "") if quant is not None else getattr(model_spec, "weight_dtype", act_dtype)
        ).lower() or act_dtype
        is_w4a8 = weight_quant == "int4"
        is_int8_kv = bool(quant is not None and str(getattr(quant, "kv_dtype", "")).lower() == "int8")
        modes = [
            resolve_mla_preprocess_mode(
                arch=arch_upper,
                phase="decode",
                num_tokens=1,
                q_lora_rank=int(plan.q_lora_rank or 0),
                kv_lora_rank=int(plan.kv_lora_rank or 0),
                qk_rope_head_dim=int(plan.qk_rope_head_dim or 0),
                has_rope=(not bool(plan.mla_use_nope)) and int(plan.qk_rope_head_dim or 0) > 0,
                is_sparse_sfa=bool(plan.indexer),
                dcp_size=int(p_cfg.cp or 1),
                pcp_size=int(p_cfg.pcp or 1),
                enable_mlapo=False,
                enable_mla_prolog=True,
                qkv_a_gate_fused=bool(plan.mla_qkv_a_gate_fused or plan.mla_output_gate),
                is_w4a8=is_w4a8,
                is_int8_kv=is_int8_kv,
                has_kv_scale=is_int8_kv,
                act_dtype=act_dtype,
                weight_quant=weight_quant,
                input_size=int(model_spec.hidden_size),
                num_kv_heads=1,
                num_q_heads=int(plan.num_heads or model_spec.num_heads),
                qk_nope_head_dim=int(plan.qk_nope_head_dim or 0),
                registry=registry,
            )
            for plan in mla_layers
        ]
        if modes and all(m in {"prolog_v3", "prolog_v3_learned_rope_tail"} for m in modes):
            prolog_options: dict[str, Any] = {}
            if arch in {"a3", "a5"} and is_int8_kv:
                from mindie_sim.prediction.operators.mla_prolog.mla_prolog_packed import LAYOUT
                prolog_options["cache_layout"] = LAYOUT
            lowerings["mla_prolog"] = (
                {"implementation": "vllm_ascend_mla_prolog_v3", "options": prolog_options}
                if prolog_options else "vllm_ascend_mla_prolog_v3"
            )

    normalization = dict(model_spec.catalog_facts).get("normalization", {})
    if (model_spec.activation_dtype == "bf16"
            and not normalization.get("norm_weight_offset", 0.0)):
        lowerings["add_rmsnorm"] = "vllm_ascend_native_norm"
        if "activation_quantization" in lowering_passes and model_spec.quant is not None:
            q = model_spec.quant
            is_mx_fp8 = (
                q.activation_format == "fp8_e4m3"
                and q.activation_group_size == MX_GROUP_SIZE
                and q.activation_scale_format == "fp8_e8m0"
                and q.activation_scale_pack == MX_SCALE_PACK
            )
            if q.activation_format == "int8":
                lowerings["add_rmsnorm_quant"] = "vllm_ascend_native_norm"
            elif is_mx_fp8:
                lowerings["add_rmsnorm_quant"] = "vllm_ascend_native_norm"
                lowerings["rmsnorm_quant"] = "vllm_ascend_native_norm"

    # M12: only bind fused CANN compressor if sealed ABI ships aclnnCompressor (C2 uses Triton).
    if (
        any(int(getattr(plan, "compress_ratio", 0) or 0) > 0 for plan in model_spec.layer_plans)
        and registry.is_shipping_available("aclnnCompressor", arch_upper)
    ):
        lowerings.setdefault("compressor", "vllm_ascend_cann_compressor")

    if any(int(getattr(plan, "index_kpool", 1) or 1) > 1 for plan in model_spec.layer_plans):
        lowerings.setdefault("pool_key_indexer", "vllm_ascend_cann_keypool_caller")
    return lowerings


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


def _detect_case_role(out: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    single_spine = meta.get("single_spine")
    for cand in (
        meta.get("deployment_role"),
        meta.get("role"),
        single_spine.get("deployment_mode") if isinstance(single_spine, Mapping) else None,
        (out.get("runtime") or {}).get("deployment_role") if isinstance(out.get("runtime"), Mapping) else None,
    ):
        text = str(cand or "").strip().lower()
        if text in _VALID_ROLES:
            return text
    return "colocated"


def apply_va_recommended(case: Mapping[str, Any]) -> dict[str, Any]:
    """Apply supported vLLM-Ascend execution subset while preserving SKU, role, and quantization."""
    from mindie_sim.prediction.structure import parse_case

    out = deepcopy(dict(case))
    parsed = parse_case(out)
    machine_raw = out.get("machine") if isinstance(out.get("machine"), Mapping) else {}
    sku = str(machine_raw.get("sku") or parsed.machine.sku or "").strip()
    if not sku:
        raise ModelingError("apply_va_recommended_sku_required")
    arch = _arch_key(sku)
    rt = dict(out.get("runtime", {}))
    meta = dict(rt.get("metadata", {}))
    role_norm = _detect_case_role(out, meta)

    va_profile = va_recommended_config(parsed.model, sku, role=role_norm)
    va_profile["quantization_action"] = "existing_case_preserved"
    prior = snapshot_kernel_keys(rt.get("kernels"))
    par = dict(rt.get("parallel", {
        "dp": parsed.runtime.parallel.dp, "tp": parsed.runtime.parallel.tp,
        "pp": parsed.runtime.parallel.pp, "ep": parsed.runtime.parallel.ep,
        "cp": parsed.runtime.parallel.cp, "dcp": parsed.runtime.parallel.dcp,
        "pcp": parsed.runtime.parallel.pcp,
    }))
    if rt.get("graph_capture_mode", "none") == "none":
        rt["graph_mode"] = "full"
        rt["graph_capture_mode"] = "full"
        rt.setdefault("compilation_config", {"cudagraph_mode": FULL_DECODE_CUDAGRAPH_MODE})
        max_seqs = int(rt.get("max_num_seqs", MIN_DEFAULT_MAX_SEQS))
        rt["graph_capture_keys"] = capture_keys(
            max_seqs, int(rt.get("mtp_proposal_tokens", 0) or 0))
    meta["va_recommended"] = True
    meta["va_profile"] = deepcopy(va_profile)
    meta["va_env"] = dict(va_profile["env"])
    if va_profile.get("shipping_collective_policy"):
        meta["shipping_collective_policy"] = deepcopy(va_profile["shipping_collective_policy"])
    meta.setdefault("moe_dispatch_quantization", "routed_expert")
    if (rt.get("multistream_overlap_shared_expert") is not False
            and va_profile.get("multistream_overlap_shared_expert")):
        rt["multistream_overlap_shared_expert"] = True
        meta["multistream_overlap_shared_expert"] = True
    wants_engram = bool(
        va_profile.get("engram_cpu_offload")
        or getattr(parsed.model, "engram", None) is not None
    )
    if rt.get("engram_cpu_offload") is not False and wants_engram:
        rt["engram_cpu_offload"] = True
        out["machine"], _ = bind_sku_host_dma_rates(out.get("machine") or {})
    prof_flags = dict(va_profile.get("tutorial_flags") or {})
    if prof_flags.get("enable_shared_expert_dp") and "enable_shared_expert_dp" not in meta:
        meta["enable_shared_expert_dp"] = True
    additional = dict(meta.get("additional_config") or {})
    track_name = str(
        (meta.get("feature_track_report") or {}).get("feature_track")
        or meta.get("feature_track")
        or "shipping"
    ).strip().lower()
    if track_name not in STACK_TRACKS:
        raise ModelingError(f"apply_va_recommended_invalid_stack_track:{track_name!r}")
    allow_shipping_flags = track_name in {"shipping", "proposed"}
    for flag_key in ("enable_flashcomm1", "enable_fused_mc2", "enable_prefill_mc2"):
        if flag_key not in prof_flags or flag_key in additional:
            continue
        if flag_key == "enable_fused_mc2" and not allow_shipping_flags and int(prof_flags[flag_key]) == 0:
            continue
        additional[flag_key] = prof_flags[flag_key]
    apply_arch_weight_nz_policy(additional, arch=arch, model_spec=parsed.model)
    if additional:
        meta["additional_config"] = additional
    rt["metadata"] = meta
    sync_moe_route_on_runtime(
        rt, arch=arch, model_spec=parsed.model, tp=int(par.get("tp", 1)),
    )
    meta = dict(rt.get("metadata") or {})
    notes: list[dict[str, Any]] = []
    kernels = dict(rt.get("kernels", {}))
    tp, ep = int(par.get("tp", 1)), int(par.get("ep", 1))
    pp, pcp = int(par.get("pp", 1)), int(par.get("pcp", 1))
    if (
        "sequence_parallel" not in kernels
        and _supports_sequence_parallel(parsed.model, va_profile, tp=tp, ep=ep, pp=pp, pcp=pcp, notes=notes)
    ):
        from mindie_sim.prediction.sharding.sequence_parallel import DECLARATION

        kernels["sequence_parallel"] = dict(DECLARATION, padding="right_to_tp")
        meta["sp"] = True
    passes = tuple(rt.get("lowering_passes", ()))
    if not passes and needs_activation_quantization(parsed.model):
        passes = ("activation_quantization", "norm_quant")
        rt["lowering_passes"] = list(passes)
    lowerings = _resolve_va_lowerings(parsed.model, par, sku, passes, notes=notes)
    if "add_rmsnorm" in lowerings and "residual_norm" not in passes:
        passes = (*passes, "residual_norm")
        rt["lowering_passes"] = list(passes)
    if lowerings:
        existing_low = dict(kernels.get("lowerings", {}))
        kernels["lowerings"] = {**lowerings, **existing_low}
    if kernels:
        rt["kernels"] = kernels
    from mindie_sim.compose.operator_defaults import DECLARATION_PHASE
    from mindie_sim.prediction.implementation_selection import (
        merge_selection_summaries, operator_selection_summary,
    )
    meta["operator_selection_summary"] = merge_selection_summaries(
        meta.get("operator_selection_summary"),
        operator_selection_summary(arch_family=arch, phase=DECLARATION_PHASE, notes=notes),
    )
    rt["metadata"] = meta
    stamp_stack_profile_origins(rt, prior_keys=prior)
    out["runtime"] = rt
    parse_case(out)
    return out
