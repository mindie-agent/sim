"""Deterministic Case synthesizer for catalog, preset, and custom models.

[Role]
  Resolve hardware SKU aliases and synthesize complete, closed-world simulation
  ``Case`` dictionaries from catalog models, local config paths, or inline
  mappings alongside explicit parallelism, workload, runtime, and SLO inputs.

[Invariants]
  * Hardware SKUs resolve against registered public Ascend and NVIDIA profiles;
    topology aliases only add an explicit multi-tier preset. Unknown SKUs raise
    ``ModelingError("unknown_hardware_sku:...")``.
  * Parallelism dimensions and structural overrides are validated explicitly;
    invalid dimensions raise ``ModelingError`` or ``ValueError`` without silent
    fallback.
  * Every synthesized dictionary is verified via ``parse_case`` before return.

[Timeouts & Locks]
  Bounded local JSON config reads (< 50 MiB); no network, NPU, subprocess, or
  file locks.

[Side-Effect Policy]
  Pure in-memory case synthesis; no filesystem writes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from mindie_sim.backends.ascend import (
    ASCEND_TOPOLOGY_PRESETS,
    materialize as materialize_ascend,
    profiles as ascend_profiles,
)
from mindie_sim.backends.nvidia import (
    materialize as materialize_nvidia,
    profiles as nvidia_profiles,
)
from mindie_sim.catalog.specs import build_model_spec
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, QuantSpec
from mindie_sim.model_catalog import (
    ARCHITECTURE_PRESET_ALIASES,
    ARCHITECTURE_PRESET_CATALOG,
    CATALOG,
    CONFIG_DIR,
    MODEL_ALIASES,
    customize_model_spec,
    load_model,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.serialization import apply_model_overrides
from mindie_sim.va_profiles import VA_RECOMMENDED_PROFILES, simulation_config


HARDWARE_ALIASES: dict[str, str] = {
    "a2": "Ascend910B4", "910b": "Ascend910B4", "910b4": "Ascend910B4",
    "910b1": "Ascend910B1", "910b2": "Ascend910B2", "910b3": "Ascend910B3",
    "a3": "Ascend910_9392", "910c": "Ascend910_9392", "910_9392": "Ascend910_9392",
    "910_9382": "Ascend910_9382",
    "a5": "Ascend910_9599", "950": "Ascend910_9599", "910_9599": "Ascend910_9599",
    "h20": "NVIDIA_H20", "h800": "NVIDIA_H800", "h200": "NVIDIA_H200",
    "b300": "NVIDIA_B300",
    "800i-a2": "Ascend910B4", "800i-a3": "Ascend910_9392",
    "800i-a3-8p": "Ascend910_9392", "800i-a3-16p": "Ascend910_9392",
    "a3-pod": "Ascend910_9392",
    "a5-pod-dt": "Ascend950DT_9581", "a5-pod-pr": "Ascend950PR_9579",
    "a5-server-dt": "Ascend950DT_9581", "a5-server-pr": "Ascend950PR_9579",
    "a5-server-pr-pcie-16p": "Ascend950PR_9579",
}

HARDWARE_TOPOLOGY_ALIASES: dict[str, tuple[str, str]] = {
    "800i-a2": ("Ascend910B4", "a2_standard"),
    "800i-a3": ("Ascend910_9392", "a3_16p"),
    "800i-a3-8p": ("Ascend910_9392", "a3_8p"),
    "800i-a3-16p": ("Ascend910_9392", "a3_16p"),
    "a3-pod": ("Ascend910_9392", "a3_superpod"),
    "a5-pod-dt": ("Ascend950DT_9581", "a5_pod"),
    "a5-pod-pr": ("Ascend950PR_9579", "a5_pod"),
    "a5-server-dt": ("Ascend950DT_9581", "a5_server"),
    "a5-server-pr": ("Ascend950PR_9579", "a5_server"),
    "a5-server-pr-pcie-16p": ("Ascend950PR_9579", "a5_pcie_16p"),
}

_EXTRA_QUANT_SPECS: dict[str, QuantSpec] = {
    "fp16": QuantSpec("fp16", "fp16", "fp16", "fp16", "fp16", "fp16", "fp16"),
    "mxfp4": QuantSpec(
        name="MXFP4", weight_format="mxfp4", activation_format="fp8_e4m3",
        expert_weight_format="mxfp4", weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0, kv_dtype="bf16",
    ),
}

def _arch_key(sku: str) -> str:
    if sku.startswith(("Ascend950", "Ascend910_95")):
        return "a5"
    return "a3" if sku.startswith("Ascend910_93") else ("a2" if sku.startswith("Ascend910B") else "other")


def va_recommended_config(model: str, hardware: str = "a3") -> dict[str, Any]:
    """Return a source-scoped simulation subset, not a complete vLLM launch file."""
    key = str(model).strip()
    canonical = MODEL_ALIASES.get(key, ARCHITECTURE_PRESET_ALIASES.get(key, key))
    sku, _, _ = _resolve_hardware(hardware)
    return simulation_config(canonical, sku, _arch_key(sku))


def _resolve_va_lowerings(
    model_spec: ModelSpec, parallel: Mapping[str, int], sku: str, lowering_passes: tuple[str, ...],
) -> dict[str, Any]:
    from mindie_sim.schema.parallel import ParallelSpec
    is_a5 = _arch_key(sku) == "a5"
    p_cfg = ParallelSpec(
        dp=int(parallel.get("dp", 1)), tp=int(parallel.get("tp", 1)),
        pp=int(parallel.get("pp", 1)), ep=int(parallel.get("ep", 1)),
        cp=int(parallel.get("cp", parallel.get("dcp", 1))), pcp=int(parallel.get("pcp", 1)),
    )
    lowerings: dict[str, Any] = {}
    if any(p.attention == "kda" for p in model_spec.layer_plans) and p_cfg.pcp >= 1:
        # This source kernel owns complete decode batches with q1-q8. Prefill,
        # mixed batches and longer queries retain explicit semantic work.
        lowerings["kda"] = {
            "cases": [{
                "domain": {"phases": ["decode"], "max_query_tokens": {"min": 1, "max": 8}},
                "binding": ("vllm_ascend_recurrent_kda_arch35" if is_a5
                            else "vllm_ascend_recurrent_kda_arch22"),
            }],
            "otherwise": "analytical",
        }
    if any(p.residual_block_size > 0 for p in model_spec.layer_plans) and p_cfg.pp >= 1:
        lowerings["attention_residual"] = "vllm_ascend_attn_res_prefix"
    if any(p.attention == "gdn" for p in model_spec.layer_plans) and p_cfg.pcp >= 1:
        opts = {"recurrent_implementation": "arch35_fp32"} if is_a5 else {}
        binding = {"implementation": "vllm_ascend_gdn_decode", "options": opts}
        # The source caller has separate ordinary and speculative paths. A
        # heterogeneous q1/q>1 batch must retain the semantic implementation.
        lowerings["gdn_decode"] = {
            "cases": [
                {"domain": {"phases": ["decode"], "min_query_tokens": {"min": 1},
                            "max_query_tokens": {"max": 1}},
                 "binding": binding},
                {"domain": {"phases": ["decode"], "min_query_tokens": {"min": 2},
                            **({"max_query_tokens": {"max": 8}} if is_a5 else {})},
                 "binding": binding},
            ],
            "otherwise": "analytical",
        }
    mla_layers = [p for p in model_spec.layer_plans if p.attention == "mla" and p.q_lora_rank > 0]
    if mla_layers:
        from mindie_sim.prediction.family_mla_prolog import validate as validate_mla_prolog
        try:
            for plan in mla_layers:
                validate_mla_prolog(plan=plan, parallel=p_cfg, model=model_spec,
                                    quant=model_spec.quant, lowering_passes=lowering_passes)
            lowerings["mla_prolog"] = "vllm_ascend_mla_prolog_v3"
        except ModelingError:
            pass
    normalization = dict(model_spec.catalog_facts).get("normalization", {})
    if (model_spec.activation_dtype == "bf16"
            and not normalization.get("norm_weight_offset", 0.0)
            and not any(p.qk_norm for p in model_spec.layer_plans)):
        lowerings["add_rmsnorm"] = "vllm_ascend_native_norm"
        if "activation_quantization" in lowering_passes and model_spec.quant.activation_format == "int8":
            lowerings["add_rmsnorm_quant"] = "vllm_ascend_native_norm"
    return lowerings


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _ensure_registered_profiles() -> dict[str, Any]:
    reg = ascend_profiles()
    if "Ascend910_9599" not in reg and "Ascend950DT_9582" in reg:
        reg["Ascend910_9599"] = deepcopy(reg["Ascend950DT_9582"])
    return reg


def resolve_hardware_sku(hardware: str) -> str:
    """Resolve a hardware alias or exact SKU name against registered profiles."""
    if not isinstance(hardware, str) or not hardware.strip():
        raise ModelingError(f"unknown_hardware_sku:{hardware}")
    reg = {**_ensure_registered_profiles(), **nvidia_profiles()}
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
    raise ModelingError(f"unknown_hardware_sku:{hardware}")


def _hardware_backend(sku: str) -> str:
    if sku in nvidia_profiles():
        return "nvidia"
    if sku in _ensure_registered_profiles():
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
    if target.stat().st_size > 50 * 1024**2:
        raise ValueError("metadata_input_exceeds_50_MiB")
    with target.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ModelingError(f"model_config_invalid_mapping:{path}")
    return dict(data)


def _resolve_model_spec(model: str | Path | Mapping[str, Any]) -> ModelSpec:
    if isinstance(model, Mapping):
        raw = dict(model)
        if "layers" in raw and "hidden_size" in raw and "model_type" not in raw:
            return ModelSpec.from_mapping(raw)
        name = str(raw.get("name") or raw.get("model_type") or "inline-model")
        return build_model_spec(name, raw)

    if isinstance(model, Path):
        raw = _read_config_json(model)
        name = str(raw.get("name") or (model.name if model.is_dir() else model.stem))
        if "layers" in raw and "hidden_size" in raw and "model_type" not in raw:
            return ModelSpec.from_mapping(raw)
        return build_model_spec(name, raw)

    if not isinstance(model, str) or not model.strip():
        raise ModelingError(f"invalid_model_identifier:{model}")

    key = model.strip()
    canonical = MODEL_ALIASES.get(key, ARCHITECTURE_PRESET_ALIASES.get(key, key))
    if canonical in CATALOG or canonical in ARCHITECTURE_PRESET_CATALOG:
        return load_model(canonical)

    candidate = Path(key)
    if candidate.exists():
        return _resolve_model_spec(candidate)

    cfg_candidate = CONFIG_DIR / (key if key.endswith(".json") else f"{key}.json")
    if cfg_candidate.is_file():
        return _resolve_model_spec(cfg_candidate)

    return load_model(key)


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
        qkey = quant.strip().lower()
        if qkey in _EXTRA_QUANT_SPECS:
            qspec = _EXTRA_QUANT_SPECS[qkey]
            return apply_model_overrides(
                model_spec, {"quant": qspec.to_dict(), "kv_dtype": qspec.kv_dtype},
            )
        return apply_model_overrides(model_spec, {"quantization": qkey})
    if isinstance(quant, Mapping):
        qdict = dict(quant)
        return apply_model_overrides(
            model_spec,
            {"quant": qdict, "kv_dtype": str(qdict.get("kv_dtype", model_spec.kv_dtype))},
        )
    raise ModelingError(f"invalid_quant_override:{quant}")


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
    model_spec: ModelSpec, *, layers: int | None, n_routed_experts: int | None, num_experts_per_tok: int | None,
) -> ModelSpec:
    if layers is None and n_routed_experts is None and num_experts_per_tok is None:
        return model_spec
    return customize_model_spec(
        model_spec, layers=layers, n_routed_experts=n_routed_experts, num_experts_per_tok=num_experts_per_tok,
    )


def compose_case(
    model: str | Path | Mapping[str, Any],
    hardware: str | Mapping[str, Any] = "a3",
    tp: int = 1,
    dp: int = 1,
    ep: int = 1,
    pp: int = 1,
    pcp: int = 1,
    dcp: int = 1,
    sp: bool = False,
    num_devices: int | None = None,
    ranks_per_node: int | None = None,
    quant: str | Mapping[str, Any] | None = None,
    layers: int | None = None,
    n_routed_experts: int | None = None,
    num_experts_per_tok: int | None = None,
    prompt_tokens: int = 4096,
    output_tokens: int = 128,
    concurrency: int = 1,
    prefix_tokens: int = 0,
    arrival_gap_ms: float = 0.0,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    block_size: int = 128,
    enable_chunked_prefill: bool = True,
    async_scheduling: bool = False,
    graph_capture_mode: str = "none",
    mtp_proposal_tokens: int = 0,
    mtp_acceptance_length: float = 1.0,
    ttft_slo_ms: float = 5000.0,
    tpot_slo_ms: float = 50.0,
    inter_node_bw_gbps: float | None = None,
    intra_node_bw_gbps: float | None = None,
    intra_package_bw_gbps: float | None = None,
    inter_node_bw_util: float | None = None,
    intra_node_bw_util: float | None = None,
    intra_package_bw_util: float | None = None,
    inter_node_latency_us: float | None = None,
    intra_node_latency_us: float | None = None,
    intra_package_latency_us: float | None = None,
    topology_preset: str | None = None,
    superpod_size: int | None = None,
    superpod_bw_gbps: float | None = None,
    inter_pod_bw_gbps: float | None = None,
    intra_bw_4p_gbps: float | None = None,
    intra_bw_8p_gbps: float | None = None,
    va_recommended: bool = False,
    pp_layer_partition: Sequence[int] | None = None,
    enable_shared_expert_dp: bool = False,
) -> dict[str, Any]:
    """Synthesize and validate a complete simulation Case dictionary."""
    for name, val in (("tp", tp), ("dp", dp), ("ep", ep), ("pp", pp), ("pcp", pcp), ("dcp", dcp)):
        if int(val) <= 0:
            raise ModelingError(f"invalid_parallel_dimension:{name}={val}")
    tp, dp, ep, pp, pcp, dcp = int(tp), int(dp), int(ep), int(pp), int(pcp), int(dcp)
    if not (ep == 1 or (dp * tp * pcp) % ep == 0 or ep == dp * tp):
        raise ModelingError(f"invalid_ep_partition:ep={ep},dp={dp},tp={tp}")
    if int(prompt_tokens) <= 0 or int(output_tokens) <= 0 or int(concurrency) <= 0:
        raise ModelingError("invalid_workload_dimensions")
    if int(prefix_tokens) < 0 or int(prefix_tokens) > int(prompt_tokens):
        raise ModelingError("invalid_prefix_tokens")
    if graph_capture_mode not in {"none", "full", "piecewise"}:
        raise ModelingError(f"invalid_graph_capture_mode:{graph_capture_mode}")
    if int(mtp_proposal_tokens) < 0:
        raise ModelingError(f"invalid_mtp_proposal_tokens:{mtp_proposal_tokens}")

    required_devices = dp * tp * pp * pcp
    hw_raw = dict(hardware) if isinstance(hardware, Mapping) else {}
    sku, backend, alias_preset = _resolve_hardware(hardware)
    preset = str(topology_preset or alias_preset or hw_raw.get("topology_preset") or "")
    if preset and backend != "ascend":
        raise ModelingError(f"hardware_topology_preset_unsupported:{backend}:{preset}")
    if preset and preset not in ASCEND_TOPOLOGY_PRESETS:
        raise ModelingError(f"ascend_topology_preset_unknown:{preset}")
    preset_defaults = ASCEND_TOPOLOGY_PRESETS.get(preset, {})
    resolved_devices = int(
        num_devices if num_devices is not None else hw_raw.get("num_devices", required_devices)
    )
    if resolved_devices < required_devices:
        raise ModelingError(f"insufficient_num_devices:{resolved_devices}:required:{required_devices}")
    default_rpn = min(resolved_devices, 16 if sku.startswith("Ascend910_93") else 8)
    if preset_defaults:
        # A preset owns its topology geometry even when fewer ranks are placed;
        # placement then stays on the first declared node.
        default_rpn = int(preset_defaults.get("ranks_per_node", default_rpn))
    resolved_rpn = int(
        ranks_per_node if ranks_per_node is not None else hw_raw.get("ranks_per_node", default_rpn)
    )
    network: dict[str, Any] = {}
    for kind, value in (
        ("inter_node", inter_node_bw_gbps),
        ("intra_node", intra_node_bw_gbps),
        ("intra_package", intra_package_bw_gbps),
    ):
        if value is not None:
            # compose_case declares GB/s; the adapter's convenience alias uses
            # bytes/s to retain that unit independently of its legacy Gb/s path.
            network[f"{kind}_bw_bytes_per_s"] = float(value) * 1e9
    for name, value in (
        ("inter_node_bw_util", inter_node_bw_util),
        ("intra_node_bw_util", intra_node_bw_util),
        ("intra_package_bw_util", intra_package_bw_util),
        ("inter_node_latency_us", inter_node_latency_us),
        ("intra_node_latency_us", intra_node_latency_us),
        ("intra_package_latency_us", intra_package_latency_us),
        ("superpod_size", superpod_size),
        ("superpod_bw_gbps", superpod_bw_gbps),
        ("inter_pod_bw_gbps", inter_pod_bw_gbps),
        ("intra_bw_4p_gbps", intra_bw_4p_gbps),
        ("intra_bw_8p_gbps", intra_bw_8p_gbps),
    ):
        if value is not None:
            network[name] = value
    machine_input = {**hw_raw, **network, "sku": sku,
                     "num_devices": resolved_devices, "ranks_per_node": resolved_rpn}
    if preset:
        machine_input["topology_preset"] = preset
    if "device" in machine_input and "links" in machine_input:
        from mindie_sim.schema.hardware import Link, apply_network_overrides
        parsed_links = tuple(
            Link.from_mapping(l, default_members=range(resolved_devices))
            for l in machine_input["links"]
        )
        parsed_links = apply_network_overrides(parsed_links, network)
        machine = {**machine_input, "links": [l.to_dict() for l in parsed_links]}
    elif backend == "custom":
        from mindie_sim.custom_hardware import (
            build_custom_hardware_machine,
            resolve_custom_hardware_sku,
        )
        custom_spec = resolve_custom_hardware_sku(sku) or {}
        custom_overrides = {
            k: v for k, v in (
                ("ranks_per_node", ranks_per_node),
                ("inter_node_bw_gbps", inter_node_bw_gbps),
                ("intra_node_bw_gbps", intra_node_bw_gbps),
                ("intra_package_bw_gbps", intra_package_bw_gbps),
                ("inter_node_bw_util", inter_node_bw_util),
                ("intra_node_bw_util", intra_node_bw_util),
                ("intra_package_bw_util", intra_package_bw_util),
                ("inter_node_latency_us", inter_node_latency_us),
                ("intra_node_latency_us", intra_node_latency_us),
                ("intra_package_latency_us", intra_package_latency_us),
                ("superpod_size", superpod_size),
                ("superpod_bw_gbps", superpod_bw_gbps),
                ("inter_pod_bw_gbps", inter_pod_bw_gbps),
                ("intra_bw_4p_gbps", intra_bw_4p_gbps),
                ("intra_bw_8p_gbps", intra_bw_8p_gbps),
            ) if v is not None
        }
        merged_spec = {**custom_spec, **hw_raw, "sku": sku}
        merged_spec.pop("num_devices", None)
        machine = build_custom_hardware_machine(
            merged_spec, num_devices=resolved_devices, **custom_overrides,
        )
    elif backend == "nvidia":
        machine = materialize_nvidia(machine_input)
    else:
        machine = materialize_ascend(machine_input)
    if preset:
        machine = {**machine, "topology_preset": preset}

    model_spec = _resolve_model_spec(model)
    model_limit = _resolve_max_position_embeddings(model, model_spec)
    va_profile: dict[str, Any] | None = None
    if va_recommended and backend == "ascend":
        model_id = str(model) if isinstance(model, str) else str(model_spec.name)
        va_profile = va_recommended_config(model_id, sku)
        va_profile["quantization_action"] = (
            "explicit_quant_preserved" if quant is not None else
            "documented_alternative_applied" if va_profile["quant"] is not None else
            "no_profile_override")
        if quant is None:
            quant = va_profile["quant"]
        if graph_capture_mode == "none":
            graph_capture_mode = str(va_profile.get("graph_capture_mode", "full"))
    model_spec = _apply_quant_override(model_spec, quant, machine)
    model_spec = _apply_structural_overrides(
        model_spec,
        layers=layers,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    if (va_profile is not None and not sp and va_profile.get("sp", True)
            and pcp >= 1 and pp >= 1 and tp > 1 and ep >= tp and (ep % tp == 0)):
        from mindie_sim.prediction.sequence_parallel import _validate_model
        try:
            _validate_model(model_spec, padding="right_to_tp")
            sp = True
        except ModelingError:
            pass

    if int(mtp_proposal_tokens) > 0:
        has_mtp = getattr(model_spec.auxiliary, "mtp", None) is not None
        has_dspark = getattr(model_spec.auxiliary, "dspark", None) is not None
        if (not (has_mtp or has_dspark) and model_spec.layer_plans
                and model_spec.layer_plans[-1].attention in {"mla", "gqa"}):
            last = model_spec.layer_plans[-1]
            synth_mtp = DraftSpec(
                kind="mtp", layers=1, hidden_size=model_spec.hidden_size,
                intermediate_size=last.intermediate_size or last.moe_intermediate_size,
                num_heads=last.num_heads, num_kv_heads=last.num_kv_heads,
                head_dim=last.head_dim,
                q_width=last.num_heads * last.head_dim,
                kv_width=last.num_kv_heads * last.head_dim,
                attention_kind=last.attention,
                q_lora_rank=last.q_lora_rank, kv_lora_rank=last.kv_lora_rank,
                qk_nope_head_dim=last.qk_nope_head_dim,
                qk_rope_head_dim=last.qk_rope_head_dim, v_head_dim=last.v_head_dim,
                o_lora_rank=last.o_lora_rank, o_groups=last.o_groups,
                num_experts=last.num_experts, top_k=last.top_k,
                expert_intermediate_size=last.moe_intermediate_size,
                num_shared_experts=last.num_shared_experts,
                target_layer_ids=(last.index,), block_size=1,
                target_shared_embedding=True, target_shared_lm_head=True,
                mla_variant=last.mla_variant,
                kv_cache_elements=last.kv_cache_elements,
                input_mode="dual_normalized", base_logits=True,
                pp_capability=(1, 2, 4, 8, 16),
                vocab_size=model_spec.vocab_size,
            )
            model_spec = replace(model_spec, auxiliary=AuxiliarySpec(mtp=synth_mtp))
            has_mtp = True
        if not (has_mtp or has_dspark):
            raise ModelingError("draft_structure_required_for_mtp_proposal_tokens")
        if not (1.0 <= float(mtp_acceptance_length) <= float(1 + int(mtp_proposal_tokens))):
            raise ModelingError("invalid_mtp_acceptance_length")

    required_len = int(prompt_tokens) + int(output_tokens) + max(0, int(mtp_proposal_tokens))
    if max_model_len is not None:
        if int(max_model_len) < required_len:
            raise ModelingError(f"max_model_len_smaller_than_workload:{max_model_len}<{required_len}")
        resolved_max_model_len = int(max_model_len)
    else:
        if model_limit > 0 and required_len > model_limit:
            raise ModelingError(
                f"workload_exceeds_model_max_position_embeddings:{required_len}>{model_limit}"
            )
        resolved_max_model_len = max(32768, required_len)

    model_dict = _to_jsonable(model_spec)
    model_dict["n_routed_experts"] = int(model_spec.num_experts)
    model_dict["num_experts_per_tok"] = int(model_spec.top_k)

    resolved_max_seqs = int(max_num_seqs) if max_num_seqs is not None else max(int(concurrency), 16)
    resolved_max_batched = (
        int(max_num_batched_tokens)
        if max_num_batched_tokens is not None
        else max(int(prompt_tokens) * int(concurrency), 8192)
    )
    requests = [
        {
            "id": f"req-{idx}",
            "arrival_ms": float(idx * float(arrival_gap_ms)),
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": int(output_tokens),
            "prefix_tokens": int(prefix_tokens),
        }
        for idx in range(int(concurrency))
    ]
    runtime_meta: dict[str, Any] = {
        "sp": bool(sp),
        "moe_dispatch_quantization": "routed_expert",
    }
    if int(mtp_proposal_tokens) > 0:
        runtime_meta["draft_mode"] = "mtp" if getattr(model_spec.auxiliary, "mtp", None) is not None else "dspark"
        runtime_meta["draft_proposal_tokens"] = int(mtp_proposal_tokens)
    if enable_shared_expert_dp:
        runtime_meta["enable_shared_expert_dp"] = True

    runtime_dict: dict[str, Any] = {
        "parallel": {"dp": dp, "tp": tp, "pp": pp, "ep": ep, "cp": dcp, "dcp": dcp, "pcp": pcp},
        "max_num_seqs": resolved_max_seqs,
        "max_num_batched_tokens": resolved_max_batched,
        "max_model_len": resolved_max_model_len,
        "kv_block_size": int(block_size),
        "enable_chunked_prefill": bool(enable_chunked_prefill),
        "async_scheduling": bool(async_scheduling),
        "graph_mode": str(graph_capture_mode),
        "graph_capture_mode": str(graph_capture_mode),
        "mtp_acceptance_length": float(mtp_acceptance_length),
        "metadata": runtime_meta,
    }
    if graph_capture_mode == "full":
        max_cap = max(1, resolved_max_seqs * max(1, 1 + int(mtp_proposal_tokens)))
        runtime_dict["graph_capture_keys"] = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, max_cap})
    elif graph_capture_mode == "piecewise":
        runtime_dict["piecewise_capture_keys"] = sorted(
            {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, resolved_max_batched}
        )
    if pp_layer_partition is not None:
        runtime_dict["pp_layer_partition"] = [int(x) for x in pp_layer_partition]
    elif pp > 1 and model_spec.layers % pp != 0:
        hi = (model_spec.layers + pp - 1) // pp
        if hi * (pp - 1) < model_spec.layers:
            runtime_dict["pp_layer_partition"] = [hi] * (pp - 1) + [model_spec.layers - hi * (pp - 1)]
        else:
            runtime_dict["pp_layer_partition"] = [
                model_spec.layers // pp + (1 if i < model_spec.layers % pp else 0) for i in range(pp)]
    if sp:
        from mindie_sim.prediction.sequence_parallel import DECLARATION, _validate_model
        if pcp < 1 or pp < 1 or tp <= 1 or ep < tp or (ep % tp != 0):
            raise ModelingError(
                f"sequence_parallel_invalid_topology:tp={tp},ep={ep},pp={pp},pcp={pcp}"
            )
        _validate_model(model_spec, padding="right_to_tp")
        runtime_dict["kernels"] = {
            "sequence_parallel": dict(DECLARATION, padding="right_to_tp")
        }
    if va_profile is not None:
        runtime_meta["va_recommended"] = True
        runtime_meta["va_profile"] = deepcopy(va_profile)
        runtime_meta["va_env"] = dict(va_profile["env"])
        runtime_meta["va_recommended_mtp_proposal_tokens"] = int(va_profile["mtp_proposal_tokens"])
        passes: tuple[str, ...] = ()
        if str(model_spec.quant.activation_format) != str(model_spec.activation_dtype):
            passes = ("activation_quantization", "norm_quant")
            runtime_dict["lowering_passes"] = list(passes)
        lowerings = _resolve_va_lowerings(
            model_spec, runtime_dict["parallel"], sku, passes,
        )
        if lowerings:
            kernels_dict = dict(runtime_dict.get("kernels", {}))
            existing_low = dict(kernels_dict.get("lowerings", {}))
            kernels_dict["lowerings"] = {**lowerings, **existing_low}
            runtime_dict["kernels"] = kernels_dict

    case_dict: dict[str, Any] = {
        "model": model_dict,
        "machine": _to_jsonable(machine),
        "runtime": runtime_dict,
        "workload": {"requests": requests},
        "slo": {"ttft_ms": float(ttft_slo_ms), "tpot_ms": float(tpot_slo_ms)},
    }
    parse_case(case_dict)
    return case_dict


def apply_va_recommended(case: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the supported execution subset while preserving supplied model quantization."""
    out = deepcopy(dict(case))
    parsed = parse_case(out)
    sku = str(out.get("machine", {}).get("sku", "Ascend910_9392"))
    va_profile = va_recommended_config(str(parsed.model.name), sku)
    va_profile["quantization_action"] = "existing_case_preserved"
    rt = dict(out.get("runtime", {}))
    par = dict(rt.get("parallel", {
        "dp": parsed.runtime.parallel.dp, "tp": parsed.runtime.parallel.tp,
        "pp": parsed.runtime.parallel.pp, "ep": parsed.runtime.parallel.ep,
        "cp": parsed.runtime.parallel.cp, "dcp": parsed.runtime.parallel.dcp,
        "pcp": parsed.runtime.parallel.pcp,
    }))
    if rt.get("graph_capture_mode", "none") == "none":
        rt["graph_mode"] = "full"
        rt["graph_capture_mode"] = "full"
        max_seqs = int(rt.get("max_num_seqs", 16))
        rt["graph_capture_keys"] = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, max(1, max_seqs)})
    meta = dict(rt.get("metadata", {}))
    meta["va_recommended"] = True
    meta["va_profile"] = deepcopy(va_profile)
    meta["va_env"] = dict(va_profile["env"])
    meta.setdefault("moe_dispatch_quantization", "routed_expert")
    kernels = dict(rt.get("kernels", {}))
    tp, ep, pp, pcp = int(par.get("tp", 1)), int(par.get("ep", 1)), int(par.get("pp", 1)), int(par.get("pcp", 1))
    if (va_profile.get("sp", True) and "sequence_parallel" not in kernels
            and pcp >= 1 and pp >= 1 and tp > 1 and ep >= tp and (ep % tp == 0)):
        from mindie_sim.prediction.sequence_parallel import DECLARATION, _validate_model
        try:
            _validate_model(parsed.model, padding="right_to_tp")
            kernels["sequence_parallel"] = dict(DECLARATION, padding="right_to_tp")
            meta["sp"] = True
        except ModelingError:
            pass
    passes = tuple(rt.get("lowering_passes", ()))
    if not passes and str(parsed.model.quant.activation_format) != str(parsed.model.activation_dtype):
        passes = ("activation_quantization", "norm_quant")
        rt["lowering_passes"] = list(passes)
    lowerings = _resolve_va_lowerings(parsed.model, par, sku, passes)
    if lowerings:
        existing_low = dict(kernels.get("lowerings", {}))
        kernels["lowerings"] = {**lowerings, **existing_low}
    if kernels:
        rt["kernels"] = kernels
    rt["metadata"] = meta
    out["runtime"] = rt
    parse_case(out)
    return out


__all__ = ["HARDWARE_ALIASES", "HARDWARE_TOPOLOGY_ALIASES", "VA_RECOMMENDED_PROFILES",
           "apply_va_recommended", "compose_case", "resolve_hardware_sku",
           "va_recommended_config"]
