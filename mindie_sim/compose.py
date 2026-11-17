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
  * ``assemble`` submits model and no-layer framework fragments into one
    execution context; pure-model scope is an explicit selection, never a
    fabricated layer or an independent ``simulate_model`` timing splice.

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
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.serialization import apply_model_overrides
from mindie_sim.va_profiles import VA_RECOMMENDED_PROFILES, simulation_config


MX_GROUP_SIZE = 32
MX_SCALE_PACK = 2
NOMINAL_HOST_MEM_BPS = 64_000_000_000.0
NOMINAL_HOST_VEC_OPS = 50_000_000_000.0
NOMINAL_H2D_BPS = 32_000_000_000.0
BYTES_PER_GB = 1_000_000_000.0
NEXTN_DEFAULT_HEAD_DIM = 192
PP_CAP_DEFAULT = (1, 2, 4, 8, 16)

HARDWARE_ALIASES: dict[str, str] = {
    "a2": "Ascend910B4", "910b": "Ascend910B4", "910b4": "Ascend910B4",
    "910b1": "Ascend910B1", "910b2": "Ascend910B2", "910b3": "Ascend910B3",
    "a3": "Ascend910_9392", "910c": "Ascend910_9392", "910_9392": "Ascend910_9392",
    "910_9382": "Ascend910_9382",
    "a5": "Ascend950DT_9582", "950": "Ascend950DT_9582",
    "a5-pod": "Ascend950DT_9581", "atlas-950-superpod": "Ascend950DT_9581",
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
    "a5-pod": ("Ascend950DT_9581", "a5_pod"),
    "atlas-950-superpod": ("Ascend950DT_9581", "a5_pod"),
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

DEFAULT_PARALLEL = 1
DEFAULT_ZERO = 0
DEFAULT_PROMPT_TOKENS = 4096
DEFAULT_OUTPUT_TOKENS = 128
DEFAULT_PREFIX_TOKENS = 0
DEFAULT_CONCURRENCY = 1
DEFAULT_BLOCK_SIZE = 128
DEFAULT_ARRIVAL_GAP_MS = 0.0
DEFAULT_MTP_TOKENS = 0
DEFAULT_MTP_ACCEPT = 1.0
DEFAULT_TTFT_MS = 5000.0
DEFAULT_TPOT_MS = 50.0
DEFAULT_LOCAL_WIDTH = 1
DEFAULT_REDUNDANT_EXPERTS = 0


CAPTURE_BASE_KEYS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CAPTURE_MIN_SEQUENCES = 1
CAPTURE_MIN_STRIDE = 1
CAPTURE_DEFAULT_MTP = 0


def capture_keys(max_num_seqs: int,
                 mtp_proposal_tokens: int = CAPTURE_DEFAULT_MTP) -> list[int]:
    """Capture buckets up to ``max_num_seqs * (1 + mtp)`` including strides."""
    sequences = max(CAPTURE_MIN_SEQUENCES, int(max_num_seqs))
    stride = max(CAPTURE_MIN_STRIDE,
                 CAPTURE_MIN_STRIDE + int(mtp_proposal_tokens))
    keys = set(CAPTURE_BASE_KEYS)
    keys.update(stride * step for step in range(CAPTURE_MIN_SEQUENCES,
                                                sequences + CAPTURE_MIN_SEQUENCES))
    keys.add(sequences * stride)
    return sorted(keys)


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


def _supports_sequence_parallel(model_spec, profile: Mapping[str, Any], *,
                                 tp: int, ep: int, pp: int, pcp: int) -> bool:
    """Structural sequence-parallel applicability, never a model-name whitelist."""
    if profile.get("sp") is False:
        return False
    if not (pcp >= DEFAULT_PARALLEL and pp >= DEFAULT_PARALLEL
            and tp > DEFAULT_PARALLEL and ep >= tp
            and ep % tp == DEFAULT_ZERO):
        return False
    from mindie_sim.prediction.sequence_parallel import _validate_model
    try:
        _validate_model(model_spec, padding="right_to_tp")
    except ModelingError:
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
    if (any(p.attention == "kda" for p in model_spec.layer_plans)
            and p_cfg.pcp >= 1 and p_cfg.cp == 1):
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
    return ascend_profiles()


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
    tp: int = DEFAULT_PARALLEL,
    dp: int = DEFAULT_PARALLEL,
    ep: int = DEFAULT_PARALLEL,
    pp: int = DEFAULT_PARALLEL,
    pcp: int = DEFAULT_PARALLEL,
    dcp: int = DEFAULT_PARALLEL,
    sp: bool | None = None,
    num_devices: int | None = None,
    ranks_per_node: int | None = None,
    quant: str | Mapping[str, Any] | None = None,
    layers: int | None = None,
    n_routed_experts: int | None = None,
    num_experts_per_tok: int | None = None,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    concurrency: int = DEFAULT_CONCURRENCY,
    prefix_tokens: int = DEFAULT_PREFIX_TOKENS,
    arrival_gap_ms: float = DEFAULT_ARRIVAL_GAP_MS,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    enable_chunked_prefill: bool = True,
    async_scheduling: bool = False,
    graph_capture_mode: str | None = None,
    mtp_proposal_tokens: int = DEFAULT_MTP_TOKENS,
    mtp_acceptance_length: float = DEFAULT_MTP_ACCEPT,
    ttft_slo_ms: float = DEFAULT_TTFT_MS,
    tpot_slo_ms: float = DEFAULT_TPOT_MS,
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
    va_recommended: bool | None = None,
    raw_eager: bool = False,
    pp_layer_partition: Sequence[int] | None = None,
    enable_shared_expert_dp: bool = False,
    *,
    multistream_overlap_shared_expert: bool | None = None,
    engram_cpu_offload: bool | None = None,
    enable_dbo: bool = False,
    moe_comm_mode: str | None = None,
    o_proj_tp: int = DEFAULT_LOCAL_WIDTH,
    linear_proj_tp: int = DEFAULT_LOCAL_WIDTH,
    lm_head_tp: int = DEFAULT_LOCAL_WIDTH,
    redundant_experts_per_rank: int = DEFAULT_REDUNDANT_EXPERTS,
    collective_startup_mode: str = "legacy",
    matrix_compute_policy: str = "auto",
    recover_ssm: bool = False,
    prefix_cache_mode: str = "private",
    prefix_read_mode: str = "auto",
    homogeneous_graph_reuse: str = "off",
    prefix_cache_key: str = "",
    hardware_derivation: Mapping[str, Any] | None = None,
    aic_count: int | None = None,
    aiv_count: int | None = None,
    memory_bytes: int | None = None,
    hbm_peak_bytes_per_s: float | None = None,
    hbm_efficiency: Any = None,
    mte2_ceiling_bytes_per_s: float | None = None,
    mte2_efficiency: Any = None,
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
    if graph_capture_mode is not None and graph_capture_mode not in {"none", "full", "piecewise"}:
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

    # ``va_recommended=None`` is the production default on Ascend: documented
    # vLLM-Ascend profiles are a *default*, never an override of an explicit
    # caller choice.  ``raw_eager`` disables every optional fused lowering.
    production = (backend == "ascend" and va_recommended is not False
                  and not raw_eager)
    if raw_eager and va_recommended:
        raise ModelingError("raw_eager_conflicts_with_explicit_fusion")
    model_spec = _resolve_model_spec(model)
    model_limit = _resolve_max_position_embeddings(model, model_spec)
    va_profile: dict[str, Any] | None = None
    if production:
        model_id = str(model) if isinstance(model, str) else str(model_spec.name)
        va_profile = va_recommended_config(model_id, sku)
        va_profile["quantization_action"] = (
            "explicit_quant_preserved" if quant is not None else
            "documented_alternative_applied" if va_profile["quant"] is not None else
            "no_profile_override")
        # ``simulation_config`` no longer fabricates a full capture policy for
        # undocumented models; only a profile that declares one is forwarded.
        va_profile.setdefault("graph_capture_mode", None)
        if quant is None:
            quant = va_profile["quant"]
    model_spec = _apply_quant_override(model_spec, quant, machine)
    model_spec = _apply_structural_overrides(
        model_spec,
        layers=layers,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    # These are caller opt-ins: the profile exposes the documented capability
    # as provenance, but enabling an overlap/offload path changes the pinned
    # physical decomposition and must be an explicit composition choice.
    resolved_mso = bool(multistream_overlap_shared_expert)
    resolved_engram_offload = bool(engram_cpu_offload)
    sp_source = (
        "raw_eager" if raw_eager else
        "explicit" if sp is not None and va_recommended is not False else
        "production" if (production and va_profile is not None and sp is None) else
        "explicit" if sp is not None else "default")
    if raw_eager:
        sp = False
    elif sp is None:
        sp = bool(production and va_profile is not None
                  and _supports_sequence_parallel(model_spec, va_profile,
                                                  tp=tp, ep=ep, pp=pp, pcp=pcp))
    documented_capture = (va_profile or {}).get("graph_capture_mode")
    capture_source = (
        "raw_eager" if raw_eager else
        "explicit" if graph_capture_mode is not None
        else "profile" if (production and documented_capture in
                           {"full", "piecewise", "none"})
        else "none")
    if raw_eager:
        graph_capture_mode = "none"
    elif graph_capture_mode is None:
        documented = (va_profile or {}).get("graph_capture_mode")
        graph_capture_mode = (str(documented)
                              if documented in {"full", "piecewise", "none"} else "none")
    execution_policy = {
        "backend": backend,
        "production": bool(production),
        "va_recommended": va_recommended,
        "raw_eager": bool(raw_eager),
        "automatic_optional_fusions": bool(not raw_eager and va_recommended is not False),
        "sp": bool(sp), "sp_source": sp_source,
        "graph_capture_mode": str(graph_capture_mode),
        "capture_source": capture_source,
        "profile": (str((va_profile or {}).get("model", "")) if va_profile else ""),
        "priority": ("raw_eager > explicit caller > documented profile > "
                     "structural fusion defaults"),
    }

    if int(mtp_proposal_tokens) > DEFAULT_ZERO and model_spec.auxiliary is None:
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
                num_heads=moe_ref.num_heads, num_kv_heads=DEFAULT_PARALLEL,
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
            model_spec = declare_native_auxiliary_plans(
                replace(model_spec, auxiliary=AuxiliarySpec(mtp=draft)))

    if int(mtp_proposal_tokens) > DEFAULT_ZERO:
        # Auxiliary structure is declared natively by the model adapter at load
        # time.  No last-layer synthesis: an undeclared draft structure is an
        # explicit, actionable error instead of a fabricated one-layer module.
        aux = model_spec.auxiliary
        has_mtp = getattr(aux, "mtp", None) is not None
        has_dspark = getattr(aux, "dspark", None) is not None
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
            "prefix_cache_key": str(prefix_cache_key),
        }
        for idx in range(int(concurrency))
    ]
    runtime_meta: dict[str, Any] = {
        "sp": bool(sp),
        "moe_dispatch_quantization": "routed_expert",
        **({"raw_eager": True} if raw_eager else {}),
        "execution_policy": dict(execution_policy),
    }
    if resolved_engram_offload:
        h2d_ch = dict((machine.get("transfer_channels") or {}).get("h2d") or {})
        h2d_bps = (float(h2d_ch.get("effective_uni_bw_gbps") or 0.0)
                   * BYTES_PER_GB)
        dev = dict(machine.get("device") or {})
        rates = dict(dev.get("rates") or {})
        rates.setdefault("host_memory", NOMINAL_HOST_MEM_BPS)
        rates.setdefault("host_vector", NOMINAL_HOST_VEC_OPS)
        rates.setdefault("h2d", h2d_bps if h2d_bps > 0.0 else NOMINAL_H2D_BPS)
        dev["rates"] = rates
        machine = {**machine, "device": dev}
    if resolved_mso:
        runtime_meta["multistream_overlap_shared_expert"] = True
    if moe_comm_mode is not None:
        runtime_meta["moe_comm_mode"] = str(moe_comm_mode)
    if int(mtp_proposal_tokens) > DEFAULT_ZERO:
        runtime_meta["draft_mode"] = "mtp" if getattr(model_spec.auxiliary, "mtp", None) is not None else "dspark"
        runtime_meta["draft_proposal_tokens"] = int(mtp_proposal_tokens)
    if enable_shared_expert_dp:
        runtime_meta["enable_shared_expert_dp"] = True

    runtime_dict: dict[str, Any] = {
        "parallel": {
            "dp": dp, "tp": tp, "pp": pp, "ep": ep, "cp": dcp, "dcp": dcp, "pcp": pcp,
            **({"rank_table": [list(range(g * int(ep), (g + DEFAULT_PARALLEL) * int(ep)))
                               for g in range((int(dp) * int(tp) * int(pcp)) // int(ep))]}
               if int(pcp) > DEFAULT_PARALLEL and int(ep) > int(dp) * int(tp)
               and (int(dp) * int(tp) * int(pcp)) % int(ep) == DEFAULT_ZERO else {}),
            "o_proj_tp": int(o_proj_tp),
            "linear_proj_tp": int(linear_proj_tp),
            "lm_head_tp": int(lm_head_tp),
            "redundant_experts_per_rank": int(redundant_experts_per_rank),
        },
        "collective_startup_mode": str(collective_startup_mode),
        "matrix_compute_policy": str(matrix_compute_policy),
        "recover_ssm": bool(recover_ssm),
        "prefix_cache_mode": str(prefix_cache_mode),
        "prefix_read_mode": str(prefix_read_mode),
        "homogeneous_graph_reuse": str(homogeneous_graph_reuse),
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
    if resolved_mso:
        runtime_dict["multistream_overlap_shared_expert"] = True
    if resolved_engram_offload:
        runtime_dict["engram_cpu_offload"] = True
    if enable_dbo:
        runtime_dict["enable_dbo"] = True
    if graph_capture_mode == "full":
        runtime_dict["graph_capture_keys"] = capture_keys(
            resolved_max_seqs, mtp_proposal_tokens)
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
        # A documented vLLM ``compilation_config`` is forwarded only when it
        # agrees with the resolved graph mode; an explicit caller choice is
        # never silently overridden.
        compilation = va_profile.get("compilation_config")
        if (isinstance(compilation, dict)
                and str(compilation.get("cudagraph_mode") or "").upper()
                in {"FULL", "FULL_DECODE_ONLY"}
                and str(graph_capture_mode) == "full"):
            runtime_dict["compilation_config"] = deepcopy(compilation)
        backend = va_profile.get("all2all_backend")
        if isinstance(backend, str) and backend:
            runtime_dict["all2all_backend"] = backend
        runtime_meta["va_env"] = dict(va_profile["env"])
        runtime_meta["va_recommended_mtp_proposal_tokens"] = int(va_profile["mtp_proposal_tokens"])
        passes: tuple[str, ...] = ()
        if (model_spec.quant is not None
                and str(model_spec.quant.activation_format)
                != str(model_spec.activation_dtype)):
            passes = ("activation_quantization", "norm_quant")
            runtime_dict["lowering_passes"] = list(passes)
        lowerings = _resolve_va_lowerings(
            model_spec, runtime_dict["parallel"], sku, passes,
        )
        if "add_rmsnorm" in lowerings and "residual_norm" not in passes:
            passes = (*passes, "residual_norm")
            runtime_dict["lowering_passes"] = list(passes)
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
    derivation = dict(hardware_derivation or {})
    for key, value in (
        ("aic_count", aic_count), ("aiv_count", aiv_count),
        ("memory_bytes", memory_bytes),
        ("hbm_peak_bytes_per_s", hbm_peak_bytes_per_s),
        ("hbm_efficiency", hbm_efficiency),
        ("mte2_ceiling_bytes_per_s", mte2_ceiling_bytes_per_s),
        ("mte2_efficiency", mte2_efficiency),
    ):
        if value is None:
            continue
        if key in derivation and derivation[key] != value:
            raise ModelingError(f"hardware_derivation_conflict:{key}")
        derivation[key] = value
    if derivation:
        from mindie_sim.hardware_derivation import derived_machine
        parsed = parse_case(case_dict)
        derived = derived_machine(parsed.machine, derivation)
        case_dict["machine"] = _to_jsonable(derived)
        parse_case(case_dict)
    return case_dict


def _assembly_request_specs(case, requests):
    from mindie_sim.schema.model import RequestSpec
    if requests is None:
        return tuple(case.workload.requests)
    result = []
    for index, item in enumerate(requests):
        result.append(item if isinstance(item, RequestSpec)
                      else RequestSpec.from_mapping(item, index))
    return tuple(result)


def _assembly_emit_counts(specs, generated, override):
    """One ordinary model step emits at most one token per active request.

    A speculative step can emit more; this entry has no draft contract and
    therefore never batches several undecoded tokens into one forward/commit.
    """
    if override is not None:
        return {str(key): int(value) for key, value in override.items() if int(value) > 0}
    return {spec.request_id: 1 for spec in specs
            if spec.output_tokens - generated[spec.request_id] > 0}


def _assembly_metric_rows(specs, rows):
    records = {
        spec.request_id: {
            "request_id": spec.request_id,
            "arrival_ms": spec.arrival_ms,
            "output_tokens": spec.output_tokens,
            "generated_tokens": 0,
            "commits": [],
            "host_results": [],
            "deliveries": [],
        }
        for spec in specs
    }
    for row in rows:
        for request, count in row["emitted_tokens"].items():
            record = records[request]
            record["generated_tokens"] += count
            commit = row["commit_ms"].get(request)
            if commit is not None:
                record["commits"].append((commit, count))
        for request, at_ms in row["host_result_ms"].items():
            records[request]["host_results"].append(at_ms)
        for request, at_ms in row["delivery_ms"].items():
            records[request]["deliveries"].append(at_ms)
    result = []
    for spec in specs:
        record = records[spec.request_id]
        commits = sorted(record["commits"])
        host = sorted(record["host_results"])
        delivered = sorted(record["deliveries"])
        first_commit = commits[0][0] if commits else None
        last_commit = commits[-1][0] if commits else None
        generated = record["generated_tokens"]
        intervals = max(0, generated - 1)
        tpot = ((last_commit - first_commit) / intervals
                if intervals and first_commit is not None else None)
        window = (last_commit - spec.arrival_ms
                  if last_commit is not None else None)
        result.append({
            "request_id": spec.request_id,
            "arrival_ms": spec.arrival_ms,
            "generated_tokens": generated,
            "first_commit_ms": first_commit,
            "last_commit_ms": last_commit,
            "ttft_ms": (first_commit - spec.arrival_ms
                        if first_commit is not None else None),
            "tpot_ms": tpot,
            "e2e_ms": window,
            "output_tps": (generated * 1000.0 / window
                           if window and window > 0 else None),
            "first_host_result_ms": host[0] if host else None,
            "last_host_result_ms": host[-1] if host else None,
            "first_delivery_ms": delivered[0] if delivered else None,
            "last_delivery_ms": delivered[-1] if delivered else None,
            "delivery_e2e_ms": (delivered[-1] - spec.arrival_ms
                               if delivered else None),
        })
    return result


def assemble(case, *, scope="service", iterations=None, phase=None, tokens=None,
             context=None, requests=None, emitted_tokens=None,
             include_submission=None, trace_limit=0, resource_fusion=True,
             primitives=None, recipes=None, full_service=False,
             diagnostic_window=False, lower=None):
    """Submit model and framework fragments into one execution context.

    ``scope="model"`` selects the pure forward program and excludes
    sampling/acceptance/token commit. ``scope="service"`` keeps those service
    fragments and attaches the declared serving I/O. ``scope="framework"``
    executes only no-layer host/transfer/delivery work; it has no model layer
    and no fabricated placeholder operator.

    Iterations are serialized by actual completion/delivery events, not by an
    externally injected latency. Every iteration uses the same ``Execution``
    and its resource/service state. For decode, one ordinary model step emits
    one token per active request; ``tokens`` is a per-outer-iteration budget
    expanded into that many real microsteps. There is no batched multi-token
    decode step and no fabricated sleep.

    ``submitted_ms`` is bound only to actual engine host-launch events observed
    in this run (currently the last launch per stage). It is never the
    scheduler event; framework-only work with no launch reports no submit.
    """
    from mindie_sim.prediction.attention import AttentionPolicy
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.frontend import Frontend, SERVICE_FAMILIES
    from mindie_sim.schema.operators import IterationWork
    from mindie_sim.schema.workload import IterationPlan, RequestView, selected_mode

    if scope not in {"model", "service", "framework"}:
        raise ModelingError(f"assembly_scope_unknown:{scope}")
    explicit_window = iterations is not None or tokens is not None
    if phase is None:
        # A full service run starts from the real prefill. A bounded decode
        # window retains the historical decode default.
        phase = ("prefill" if scope == "service" and not explicit_window
                 else "decode")
    if phase not in {"decode", "prefill"}:
        raise ModelingError(f"assembly_phase_unknown:{phase}")
    if iterations is not None and int(iterations) < 1:
        raise ModelingError("assembly_requires_positive_iterations")
    if tokens is not None and int(tokens) < 1:
        raise ModelingError("assembly_requires_positive_tokens")
    if context is not None and int(context) < 0:
        raise ModelingError("assembly_context_requires_nonnegative_tokens")
    if phase == "decode" and tokens is not None and int(tokens) < 1:
        raise ModelingError("assembly_decode_requires_positive_token_budget")
    parsed = parse_case(case)
    specs = _assembly_request_specs(parsed, requests)
    if scope == "service" and not diagnostic_window:
        # Ordinary service assembly is the same production cohort driver used
        # by predict/simulate_service. The former hand loop below is retained
        # only as an explicit diagnostic window.
        from mindie_sim.prediction.api import simulate_service
        scheduler_specs = tuple(specs)
        quota = None
        quotas = None
        if explicit_window and phase == "decode":
            # The quota is an observation/emission boundary for the driver.
            # The request keeps its original output_tokens, KV and closed-loop
            # successor; a short window never retires it as complete.
            quota = int(iterations or 1) * int(tokens or 1)
            quotas = {spec.request_id: quota for spec in specs}
        window_case = replace(parsed, workload=replace(
            parsed.workload, requests=scheduler_specs))
        window = None
        if phase == "decode":
            window = {
                spec.request_id: {
                    "computed": (int(context) if context is not None
                                 else spec.prompt_tokens),
                    "generated": 0,
                }
                for spec in specs
            }
        steps = (None if full_service or not explicit_window
                 or phase == "decode" else int(iterations or 1))
        observed_steps = []
        observed_total = [0]
        def observe(step):
            observed_total[0] += 1
            if len(observed_steps) < 256:
                observed_steps.append(step.to_dict())
        result = simulate_service(
            window_case, primitives=primitives, recipes=recipes,
            trace_limit=trace_limit, max_model_steps=steps, window=window,
            token_quotas=quotas,
            include_submission=(scope != "model" if include_submission is None
                                else bool(include_submission)),
            resource_fusion=resource_fusion, lower=lower,
            observer=observe,
        )
        result["steps"] = observed_steps
        result["steps_total"] = observed_total[0]
        result["assemble_backend"] = "cohort_driver"
        result["assembly_scope"] = scope
        result["assembly"] = result.get("execution", {}).get("assembly")
        original_outputs = {spec.request_id: spec.output_tokens for spec in specs}
        unfinished = [request_id for request_id, output in original_outputs.items()
                      if next((row["generated_tokens"] for row in result["requests"]
                               if row["request_id"] == request_id), 0) < output]
        result["window"] = {
            "iterations": (int(iterations) if iterations is not None else None),
            "tokens": (int(tokens) if tokens is not None else None),
            "phase": phase, "context": context,
            "request_ids": [spec.request_id for spec in specs],
            "original_output_tokens": original_outputs,
            "per_request_token_quota": quota,
            "token_quota_by_request": dict(quotas or {}),
            "unfinished_request_ids": unfinished,
            "max_model_steps": steps,
            "cohort_stop": result.get("execution", {}).get("cohort_stop"),
        }
        if unfinished:
            result["status"] = "incomplete"
            result["execution_status"] = "incomplete"
        return result
    iterations = 1 if iterations is None else int(iterations)
    tokens = 1 if tokens is None else int(tokens)
    if parsed.runtime.parallel.dp != 1:
        raise ModelingError(
            "diagnostic_assembly_window_requires_one_data_replica")
    if not specs:
        raise ModelingError("assembly_requires_at_least_one_request")
    include_submission = (scope != "model" if include_submission is None
                          else bool(include_submission))
    execution = Execution(
        parsed, CostModel(parsed, primitives), recipes=recipes,
        include_submission=include_submission, trace_limit=trace_limit,
        include_serving_io=scope != "model",
    )
    frontend = Frontend(attention_policy=AttentionPolicy(
        **parsed.runtime.attention_policy))
    if scope == "framework" and not execution.serving_io.enabled:
        raise ModelingError("framework_assembly_requires_declared_serving_io")
    execution.serving_io.record_events = True
    generated = {spec.request_id: 0 for spec in specs}
    # Actual KV positions already computed/visible before this assembly window.
    # This is deliberately separate from emitted-token counts: a prefill of
    # prompt_tokens positions computes that KV, while its sampled output token
    # has not yet been written as KV.
    computed_kv = {
        spec.request_id: (int(context) if context is not None
                          else spec.prompt_tokens)
        for spec in specs
    }
    release_ms = max(spec.arrival_ms for spec in specs)
    rows, program_count, first_program = [], 0, None
    service_counts = {family: 0 for family in SERVICE_FAMILIES}
    serving = execution.serving_io
    # A non-draft decode step advances one token. ``tokens`` is the requested
    # token budget per outer iteration and expands into real microsteps, each
    # with its own forward/commit/state dependency; it is never one batched
    # multi-token commit or a sleep for the rest of the budget.
    decode_tokens = int(tokens) if phase == "decode" else 1
    total_steps = (int(iterations) * decode_tokens if phase == "decode"
                   else int(iterations))
    for step_index in range(total_steps):
        views = []
        for spec in specs:
            remaining = spec.output_tokens - generated[spec.request_id]
            if remaining <= 0:
                continue
            if phase == "prefill" and step_index == 0:
                q_len, view_phase = spec.prompt_tokens, "prefill"
                count = 1
            else:
                q_len = 1
                view_phase = "decode"
                count = 1
            if view_phase == "prefill":
                view_context = 0
            elif scope == "framework":
                # Framework-only has no model KV write; its view context tracks
                # declared external request progress, not claimed model KV.
                view_context = (int(context) if context is not None
                                else spec.prompt_tokens + generated[spec.request_id])
            else:
                view_context = computed_kv[spec.request_id]
            views.append(RequestView(
                spec.request_id, q_len, view_context, True, view_phase,
                draft_verify=False, draft_prepare=False,
                draft_advanced_tokens=count, prompt_tokens=spec.prompt_tokens,
            ))
        if not views:
            break
        plan = IterationPlan(
            tuple(views), iteration=step_index,
            output_tokens=tuple((spec.request_id, spec.output_tokens) for spec in specs),
        )
        if scope == "framework":
            work = IterationWork(
                (), plan, parsed.runtime.parallel.pp,
                sum(view.logical_q_len for view in views),
                sum(view.logical_q_len for view in views),
                None, selected_mode(plan), "framework_only", replay_scope="none",
            )
            program = None
        else:
            program = frontend.program(
                plan, parsed.model, parsed.runtime, parsed.machine,
                assembly=("model" if scope == "model" else "service"),
                resource_fusion=resource_fusion,
            )
            work = program.work
            program_count += 1
            if first_program is None:
                first_program = program
            for family in SERVICE_FAMILIES:
                service_counts[family] += sum(
                    op.family == family for op in work.ops)
        override = (emitted_tokens if scope == "framework" else None)
        if scope == "framework" and override is None:
            override = {}
        before_submission = dict(execution.submission_counts)
        emit = (_assembly_emit_counts(specs, generated, override)
                if scope != "model" else None)
        active_requests = {view.request_id for view in views}
        if emit is not None:
            emit = {request: count for request, count in emit.items()
                    if request in generated and request in active_requests}
        step = execution.add(work, 0, release_ms, emitted_tokens=emit)
        events = (step.completion, *step.client_deliveries.values())
        execution.engine.advance(until=tuple(events), first=False)
        prefix = f"{execution.namespace}r0:i{plan.iteration}"
        io = serving.pop_batch_events(prefix) or {}
        # T04 exposes the actual host-submit start events created by this step.
        # Keep only the chosen first/last event and a count; an eager step can
        # create thousands of launches, so the report must stay bounded.
        host_launches = [(replica, stage, event) for replica, stage, event
                         in step.host_submits if event.time is not None]
        host_submit_times = [event.time for _, _, event in host_launches]
        submitted_ms = min(host_submit_times) if host_submit_times else None
        submitted_last_ms = max(host_submit_times) if host_submit_times else None

        def host_event(item):
            replica, stage, event = item
            return {"replica": replica, "stage": stage, "event": event.name,
                    "sequence": event.sequence, "submit_ms": event.time}

        host_first = min(host_launches,
                         key=lambda item: (item[2].time, item[2].sequence),
                         default=None)
        host_last = max(host_launches,
                        key=lambda item: (item[2].time, item[2].sequence),
                        default=None)
        submitted_basis = (
            "actual_step_host_submissions" if host_submit_times else
            ("submission_not_included" if not include_submission
             else "no_engine_launch"))
        # Execution.add owns the per-request split; consume both maps directly.
        # The merged ``commits`` map is retained only as a compatibility view.
        declared_times = {
            request: event.time for request, event in step.declared_commits.items()
            if event.time is not None}
        fallback_times = {
            request: event.time for request, event in step.fallback_commits.items()
            if event.time is not None}
        merged_times = {
            request: event.time for request, event in step.commits.items()
            if event.time is not None}
        commit_times = declared_times
        host_times = ({request: event.time for request, event in step.host_results.items()
                       if event.time is not None} if serving.enabled else {})
        delivery_times = ({request: event.time for request, event in
                           step.client_deliveries.items() if event.time is not None}
                          if serving.enabled else {})
        for view in views:
            if scope == "framework":
                continue
            if view.phase == "prefill":
                computed_kv[view.request_id] = view.prompt_tokens
            else:
                computed_kv[view.request_id] += view.q_len
        row = {
            "iteration": step_index,
            "outer_iteration": (step_index // decode_tokens
                                if phase == "decode" else step_index),
            "decode_step_in_iteration": (step_index % decode_tokens
                                         if phase == "decode" else 0),
            "mode": step.mode,
            "valid_tokens": step.valid_tokens,
            "padded_tokens": step.padded_tokens,
            "graph_bucket": step.graph_bucket,
            "replay_scope": step.replay_scope,
            "release_ms": release_ms,
            "submitted_ms": submitted_ms,
            "submitted_first_ms": submitted_ms,
            "submitted_last_ms": submitted_last_ms,
            "submitted_basis": submitted_basis,
            "submitted_scope": ("actual_step_host_submissions"
                                if host_submit_times else "none"),
            "host_submit_first": host_event(host_first) if host_first else None,
            "host_submit_last": host_event(host_last) if host_last else None,
            "host_submit_event_count": len(host_launches),
            "input_ready_ms": (io.get("input_ready").time
                               if io.get("input_ready") is not None
                               and io["input_ready"].time is not None else None),
            "host_prepare_done_ms": (io.get("host_prepare").time
                                     if io.get("host_prepare") is not None
                                     and io["host_prepare"].time is not None else None),
            "device_done_ms": step.device_complete.time,
            "step_ready_ms": step.completion.time,
            "commit_ms": commit_times,
            "declared_commit_ms": declared_times,
            "fallback_commit_ms": fallback_times,
            "merged_commit_ms": merged_times,
            "host_result_ms": host_times,
            "host_complete_ms": (io.get("host_result").time
                                 if io.get("host_result") is not None
                                 and io["host_result"].time is not None else None),
            "client_propagation_ms": (io.get("client_propagation").time
                                      if io.get("client_propagation") is not None
                                      and io["client_propagation"].time is not None
                                      else None),
            "delivery_ms": delivery_times,
            "emitted_tokens": dict(emit or {}),
            "active_requests": sorted(active_requests),
            "context_tokens": {view.request_id: view.context for view in views},
            "computed_kv_tokens_after": (
                {view.request_id: computed_kv[view.request_id] for view in views}
                if scope != "framework" else None),
            "output_ready_ms": (io.get("output_ready").time
                                if io.get("output_ready") is not None
                                and io["output_ready"].time is not None else None),
            "input_bytes": io.get("input_bytes", 0.0),
            "d2h_bytes": io.get("d2h_bytes", 0.0),
            "declared_commit": bool(step.declared_commits),
            "program_assembly": (program.assembly if program is not None else "framework"),
            "graph_replay_submissions": (
                execution.submission_counts["graph_replays"]
                - before_submission["graph_replays"]),
            "eager_submissions": (
                execution.submission_counts["eager_invocations"]
                - before_submission["eager_invocations"]),
        }
        rows.append(row)
        for request, count in (emit or {}).items():
            if request in generated:
                generated[request] += count
        # The next invocation starts from an actual completion/delivery edge.
        edges = [step.completion.time, *delivery_times.values()]
        release_ms = max(edge for edge in edges if edge is not None)
    execution.engine.advance()
    metrics = _assembly_metric_rows(specs, rows)
    fragments = serving.fragments(
        h2d_hbm=execution.stage_names.get((0, 0), {}).get("hbm"),
        d2h_hbm=execution.stage_names.get(
            (0, parsed.runtime.parallel.pp - 1), {}).get("hbm"),
    )
    has_h2d = serving.transfers.get("h2d") is not None
    has_d2h = serving.transfers.get("d2h") is not None
    occurrence = {
        "host_prepare": sum(row["host_prepare_done_ms"] is not None for row in rows),
        "input_transfer": (len(rows) if has_h2d else 0),
        "output_ready": sum(row["output_ready_ms"] is not None for row in rows),
        "output_transfer": (sum(bool(row["emitted_tokens"]) for row in rows)
                            if has_d2h else 0),
        "host_result": sum(row["host_complete_ms"] is not None for row in rows),
        "client_delivery": sum(row["client_propagation_ms"] is not None
                               for row in rows),
        "client_order": sum(len(row["delivery_ms"]) for row in rows),
    }
    framework = [
        {**fragment.to_dict(), "occurrences": occurrence.get(fragment.name, 0)}
        for fragment in fragments
    ]
    model_fragment = None
    if first_program is not None:
        model_fragment = {
            **first_program.to_dict(include_ops=False),
            "op_count": len(first_program.work.ops),
            "families": sorted({op.family for op in first_program.work.ops}),
            "service_families_present": sorted({
                op.family for op in first_program.work.ops} & SERVICE_FAMILIES),
            "iterations": program_count,
        }
    total_tokens = sum(row["generated_tokens"]
                       for row in metrics if row["generated_tokens"])
    declared_commit_events = sum(len(row["declared_commit_ms"]) for row in rows)
    fallback_commit_events = sum(len(row["fallback_commit_ms"]) for row in rows)
    merged_commit_events = sum(len(row["merged_commit_ms"]) for row in rows)
    return {
        "schema": "mindie-sim.assembly/1",
        "scope": scope,
        "phase": phase,
        "same_context": True,
        "independent_model_invocations": 0,
        "submission_included": include_submission,
        "iterations_requested": int(iterations),
        "decode_tokens_per_iteration": decode_tokens,
        "steps_requested": total_steps,
        "iterations_executed": len(rows),
        "steps_executed": len(rows),
        "model_fragment": model_fragment,
        "framework_fragments": framework,
        "framework_occurrences": occurrence,
        "triggers": {
            "model_invocation": len(rows) if scope != "framework" else 0,
            "batch_prepare": occurrence["host_prepare"],
            "input_transfer": occurrence["input_transfer"],
            "output_ready": occurrence["output_ready"],
            "output_transfer": occurrence["output_transfer"],
            "host_result": occurrence["host_result"],
            "client_delivery_batches": occurrence["client_delivery"],
            "client_order_events": occurrence["client_order"],
            "request_commit_callbacks": declared_commit_events,
            "fallback_completion_callbacks": fallback_commit_events,
            "merged_commit_callbacks": merged_commit_events,
            "sampling": service_counts["sampling"],
            "acceptance": service_counts["acceptance"],
            "token_commit_nodes": service_counts["token_commit"],
            "graph_replay_submissions": sum(
                row["graph_replay_submissions"] for row in rows),
            "eager_submissions": sum(row["eager_submissions"] for row in rows),
        },
        "state_contract": {
            "decode_step": "one_token_per_active_request",
            "release": "previous_step_commit_and_delivery_event",
            "context": "actual_computed_kv_positions_before_each_forward; "
                       "prefill writes prompt_tokens, decode adds its query rows",
            "commit": "declared_token_commit_event_per_step",
        },
        "steps": rows,
        "requests": metrics,
        "metrics": {
            "boundary": "declared commit and delivery events in this context",
            "output_tokens": total_tokens,
            "window_end_ms": max((row["last_commit_ms"] for row in metrics
                                  if row["last_commit_ms"] is not None), default=None),
            "delivery_window_end_ms": max(
                (row["last_delivery_ms"] for row in metrics
                 if row["last_delivery_ms"] is not None), default=None),
            "commit_events": declared_commit_events,
            "fallback_completion_events": fallback_commit_events,
            "merged_commit_events": merged_commit_events,
            "client_propagation_events": sum(
                row["client_propagation_ms"] is not None for row in rows),
            "delivery_events": sum(len(row["delivery_ms"]) for row in rows),
            "model_elapsed_ms": (sum(row["step_ready_ms"] - row["release_ms"]
                                     for row in rows)),
            "host_submit_events": sum(row["host_submit_event_count"]
                                      for row in rows),
        },
        "event_boundaries": {
            "ready": "input value/state producer completion",
            "submit": ("actual engine host-launch event when observed; "
                       "scheduler completion is not a submit boundary"),
            "done": "all declared device work complete",
            "commit": "declared token_commit event, else step completion fallback",
            "delivery": "client-visible completion after host result",
        },
        "execution": execution.summary(),
    }


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
        rt["graph_capture_keys"] = capture_keys(
            max_seqs, int(rt.get("mtp_proposal_tokens", DEFAULT_ZERO) or DEFAULT_ZERO))
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
    if "add_rmsnorm" in lowerings and "residual_norm" not in passes:
        passes = (*passes, "residual_norm")
        rt["lowering_passes"] = list(passes)
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
           "apply_va_recommended", "assemble", "compose_case", "resolve_hardware_sku",
           "va_recommended_config"]
