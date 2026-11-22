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

[Timeouts & Locks]
  Pure in-memory dictionary and metadata resolution; no I/O or locks.

[Side-Effect Policy]
  Returns new dictionaries or mutates only caller-supplied builder dicts in place.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from mindie_sim.compose.inputs import _arch_key, _resolve_hardware
from mindie_sim.model_catalog import ARCHITECTURE_PRESET_ALIASES, MODEL_ALIASES
from mindie_sim.prediction.offload.policy import resolve_sku_host_dma_rates
from mindie_sim.prediction.structure import parse_case
from mindie_sim.runtime_profiles import simulation_config
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec

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
    """Return a source-scoped simulation subset, not a complete vLLM launch file."""
    key = str(model).strip()
    canonical = MODEL_ALIASES.get(key, ARCHITECTURE_PRESET_ALIASES.get(key, key))
    sku, _, _ = _resolve_hardware(hardware)
    return simulation_config(canonical, sku, _arch_key(sku), role=role)


def _supports_sequence_parallel(
    model_spec: ModelSpec, profile: Mapping[str, Any], *,
    tp: int, ep: int, pp: int, pcp: int, notes: list[dict[str, Any]],
) -> bool:
    """Structural sequence-parallel applicability, never a model-name whitelist."""
    if profile.get("sp") is False:
        return False
    if not (pcp >= 1 and pp >= 1 and tp > 1 and ep >= tp and ep % tp == 0):
        return False
    from mindie_sim.prediction.sharding.sequence_parallel import _validate_model
    try:
        _validate_model(model_spec, padding="right_to_tp")
    except ModelingError as exc:
        from mindie_sim.prediction.implementation_selection import fallback_note
        notes.append(fallback_note("sequence_parallel", str(exc)))
        return False
    return True


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
    from mindie_sim.catalog.paired_sfa import execution
    from mindie_sim.schema.parallel import ParallelSpec

    is_a5 = _arch_key(sku) == "a5"
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
                           "max_query_tokens": {"min": 1, "max": KDA_DECODE_MAX_QUERY_TOKENS}},
                "binding": {"implementation": kda_binding, "options": kda_options},
            }],
            "otherwise": "analytical",
        }
    if any(p.residual_block_size > 0 for p in model_spec.layer_plans) and p_cfg.pp >= 1:
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
    mla_layers = [p for p in model_spec.layer_plans
                  if p.attention == "mla" and p.q_lora_rank > 0
                  and p.mla_variant != "latent_mqa"]
    if mla_layers:
        from mindie_sim.prediction.operators.mla_prolog.family_mla_prolog import validate as validate_mla_prolog
        prolog_options: dict[str, Any] = {}
        quant = model_spec.quant
        if is_a5 and quant is not None and quant.kv_dtype == "int8":
            from mindie_sim.prediction.operators.mla_prolog.mla_prolog_packed import LAYOUT
            prolog_options["cache_layout"] = LAYOUT
        try:
            for plan in mla_layers:
                validate_mla_prolog(plan=plan, parallel=p_cfg, model=model_spec,
                                    quant=quant, lowering_passes=lowering_passes,
                                    **prolog_options)
            lowerings["mla_prolog"] = (
                {"implementation": "vllm_ascend_mla_prolog_v3", "options": prolog_options}
                if prolog_options else "vllm_ascend_mla_prolog_v3")
        except ModelingError as exc:
            from mindie_sim.prediction.implementation_selection import fallback_note
            notes.append(fallback_note("mla_prolog", str(exc)))
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
    if any(int(getattr(plan, "compress_ratio", 0) or 0) > 0 for plan in model_spec.layer_plans):
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
    """Bind SKU-calibrated L0/profile PCIe H2D/D2H & host DDR rates onto ``machine``.

    Fails closed when a hardware profile declares neither explicit host DMA rates
    nor calibrated L0 ``P20_host_dma_sync`` / ``transfer_channels``.
    """
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


def apply_va_recommended(case: Mapping[str, Any]) -> dict[str, Any]:
    """Apply supported vLLM-Ascend execution subset while preserving quantization."""
    out = deepcopy(dict(case))
    parsed = parse_case(out)
    sku = str(out.get("machine", {}).get("sku", "Ascend910_9392"))
    va_profile = va_recommended_config(str(parsed.model.name), sku)
    va_profile["quantization_action"] = "existing_case_preserved"
    rt = dict(out.get("runtime", {}))
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
        max_seqs = int(rt.get("max_num_seqs", MIN_DEFAULT_MAX_SEQS))
        rt["graph_capture_keys"] = capture_keys(
            max_seqs, int(rt.get("mtp_proposal_tokens", 0) or 0))
    meta = dict(rt.get("metadata", {}))
    meta["va_recommended"] = True
    meta["va_profile"] = deepcopy(va_profile)
    meta["va_env"] = dict(va_profile["env"])
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
        or "merged"
    ).strip().lower()
    allow_merged_workarounds = track_name == "merged"
    for flag_key in ("enable_flashcomm1", "enable_fused_mc2", "enable_prefill_mc2"):
        if flag_key not in prof_flags or flag_key in additional:
            continue
        if flag_key == "enable_fused_mc2" and not allow_merged_workarounds and int(prof_flags[flag_key]) == 0:
            continue
        additional[flag_key] = prof_flags[flag_key]
    if additional:
        meta["additional_config"] = additional
    notes: list[dict[str, Any]] = []
    kernels = dict(rt.get("kernels", {}))
    tp, ep = int(par.get("tp", 1)), int(par.get("ep", 1))
    pp, pcp = int(par.get("pp", 1)), int(par.get("pcp", 1))
    if (va_profile.get("sp", True) and "sequence_parallel" not in kernels
            and pcp >= 1 and pp >= 1 and tp > 1 and ep >= tp and (ep % tp == 0)):
        from mindie_sim.prediction.sharding.sequence_parallel import DECLARATION, _validate_model
        try:
            _validate_model(parsed.model, padding="right_to_tp")
            kernels["sequence_parallel"] = dict(DECLARATION, padding="right_to_tp")
            meta["sp"] = True
        except ModelingError as exc:
            from mindie_sim.prediction.implementation_selection import fallback_note
            notes.append(fallback_note("sequence_parallel", str(exc)))
    passes = tuple(rt.get("lowering_passes", ()))
    if not passes and str(parsed.model.quant.activation_format) != str(parsed.model.activation_dtype):
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
        operator_selection_summary(arch_family=_arch_key(sku),
                                   phase=DECLARATION_PHASE, notes=notes))
    rt["metadata"] = meta
    stamp_stack_profile_origins(rt, prior_keys=prior)
    out["runtime"] = rt
    parse_case(out)
    return out
