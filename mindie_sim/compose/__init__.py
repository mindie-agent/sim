"""Deterministic Case synthesizer for catalog, preset, and custom models.

[Role]
  Resolve hardware SKU aliases and synthesize complete, closed-world simulation
  ``Case`` dictionaries from catalog models, local config paths, or inline
  mappings alongside explicit parallelism, workload, runtime, and SLO inputs.

[Invariants]
  * Hardware SKUs resolve against registered public Ascend and NVIDIA profiles;
    topology aliases only add an explicit multi-tier preset.
  * Parallelism dimensions, structural overrides, and orthogonal compilation
    flags (``graph_capture_mode`` vs ``enable_torch_compile``/``enable_npugraph_ex``)
    are validated explicitly without silent fallback.
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
from pathlib import Path
from typing import Any

from mindie_sim.backends.ascend import (
    ASCEND_TOPOLOGY_PRESETS,
    materialize as materialize_ascend,
)
from mindie_sim.backends.nvidia import materialize as materialize_nvidia
from mindie_sim.compose.assembly import (
    _assembly_emit_counts,
    _assembly_metric_rows,
    _assembly_request_specs,
    assemble,
)
from mindie_sim.compose.inputs import (
    DERIVED_ASCEND_SKUS,
    HARDWARE_ALIASES,
    HARDWARE_TOPOLOGY_ALIASES,
    _apply_quant_override,
    _apply_structural_overrides,
    _arch_key,
    _ensure_registered_profiles,
    _hardware_backend,
    _read_config_json,
    _resolve_hardware,
    _resolve_max_position_embeddings,
    _resolve_model_spec,
    _tier4_router_spec,
    _to_jsonable,
    ensure_native_mla_draft,
    prepare_speculation,
    resolve_hardware_sku,
)
from mindie_sim.compose.runtime import (
    PIECEWISE_CAPTURE_BASE_KEYS,
    _execution_policy,
    _resolve_va_lowerings,
    _supports_sequence_parallel,
    apply_va_recommended,
    bind_sku_host_dma_rates,
    capture_keys,
    finalize_compilation_spine,
    snapshot_kernel_keys,
    stamp_stack_profile_origins,
    va_recommended_config,
)
from mindie_sim.prediction.offload.policy import missing_scenario_error
from mindie_sim.prediction.structure import parse_case
from mindie_sim.runtime_profiles import VA_RECOMMENDED_PROFILES
from mindie_sim.schema.common import ModelingError

BYTES_PER_GB = 1_000_000_000.0
A3_DEFAULT_RANKS_PER_NODE = 16
DEFAULT_RANKS_PER_NODE = 8
DEFAULT_MAX_MODEL_LEN = 32768
MIN_DEFAULT_MAX_SEQS = 16
MIN_DEFAULT_MAX_BATCHED_TOKENS = 8192
PD_DECODE_MIN_BATCHED_TOKENS = 768
DEFAULT_PARALLEL = 1
DEFAULT_PREFIX_TOKENS = 0
DEFAULT_CONCURRENCY = 1
DEFAULT_BLOCK_SIZE = 128
DEFAULT_ARRIVAL_GAP_MS = 0.0
DEFAULT_MTP_TOKENS = 0
DEFAULT_MTP_ACCEPT = 1.0
DEFAULT_LOCAL_WIDTH = 1
DEFAULT_REDUNDANT_EXPERTS = 0
DEFAULT_SPECULATIVE_ACCEPT_RATE = 0.70
_PARALLEL_AXES = ("tp", "dp", "ep", "pp", "pcp", "dcp")
_VALID_DEPLOYMENT_MODES = frozenset({"auto", "pd", "pd_disaggregated", "colocated"})
_VALID_DEPLOYMENT_ROLES = frozenset({"colocated", "pd_prefill", "pd_decode"})
_CACHE_DTYPE_ALIASES = {"fp8": "fp8_e4m3", "c8": "int8"}


def _materialize_machine(
    hardware: str | Mapping[str, Any], *, required_devices: int,
    num_devices: int | None, ranks_per_node: int | None,
    topology_preset: str | None, inter_node_bw_gbps: float | None,
    intra_node_bw_gbps: float | None, intra_package_bw_gbps: float | None,
    inter_node_bw_util: float | None, intra_node_bw_util: float | None,
    intra_package_bw_util: float | None, inter_node_latency_us: float | None,
    intra_node_latency_us: float | None, intra_package_latency_us: float | None,
    superpod_size: int | None, superpod_bw_gbps: float | None,
    inter_pod_bw_gbps: float | None, intra_bw_4p_gbps: float | None,
    intra_bw_8p_gbps: float | None,
) -> tuple[str, str, dict[str, Any]]:
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
    base_rpn = A3_DEFAULT_RANKS_PER_NODE if sku.startswith("Ascend910_93") else DEFAULT_RANKS_PER_NODE
    default_rpn = min(resolved_devices, base_rpn)
    if preset_defaults:
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
            network[f"{kind}_bw_bytes_per_s"] = float(value) * BYTES_PER_GB
    for name, value in (
        ("inter_node_bw_util", inter_node_bw_util), ("intra_node_bw_util", intra_node_bw_util),
        ("intra_package_bw_util", intra_package_bw_util), ("inter_node_latency_us", inter_node_latency_us),
        ("intra_node_latency_us", intra_node_latency_us), ("intra_package_latency_us", intra_package_latency_us),
        ("superpod_size", superpod_size), ("superpod_bw_gbps", superpod_bw_gbps),
        ("inter_pod_bw_gbps", inter_pod_bw_gbps), ("intra_bw_4p_gbps", intra_bw_4p_gbps),
        ("intra_bw_8p_gbps", intra_bw_8p_gbps),
    ):
        if value is not None:
            network[name] = value
    machine_input = {
        **hw_raw, **network, "sku": sku,
        "num_devices": resolved_devices, "ranks_per_node": resolved_rpn,
    }
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
            build_custom_hardware_machine, resolve_custom_hardware_sku,
        )
        custom_spec = resolve_custom_hardware_sku(sku) or {}
        custom_overrides = {
            k: v for k, v in (
                ("ranks_per_node", ranks_per_node), ("inter_node_bw_gbps", inter_node_bw_gbps),
                ("intra_node_bw_gbps", intra_node_bw_gbps), ("intra_package_bw_gbps", intra_package_bw_gbps),
                ("inter_node_bw_util", inter_node_bw_util), ("intra_node_bw_util", intra_node_bw_util),
                ("intra_package_bw_util", intra_package_bw_util), ("inter_node_latency_us", inter_node_latency_us),
                ("intra_node_latency_us", intra_node_latency_us), ("intra_package_latency_us", intra_package_latency_us),
                ("superpod_size", superpod_size), ("superpod_bw_gbps", superpod_bw_gbps),
                ("inter_pod_bw_gbps", inter_pod_bw_gbps), ("intra_bw_4p_gbps", intra_bw_4p_gbps),
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
    elif sku in DERIVED_ASCEND_SKUS:
        from mindie_sim.hardware_derivation import derive_hardware_sku
        spec = DERIVED_ASCEND_SKUS[sku]
        base_machine = materialize_ascend({**machine_input, "sku": spec["base"]})
        derived_dict = derive_hardware_sku(
            base_machine,
            sku=sku,
            aic_count=spec["aic_count"],
            memory_bytes=spec["memory_bytes"],
        ).to_dict()
        machine = {
            **base_machine,
            **derived_dict,
            "sku": sku,
            "device": deepcopy(derived_dict["devices"][0]),
        }
    else:
        machine = materialize_ascend(machine_input)
    if preset:
        machine = {**machine, "topology_preset": preset}
    return sku, backend, machine


_KPOOL_ALLOWED_INDEXER_KV_DTYPES = frozenset({
    "auto", "bf16", "fp8", "fp8_e4m3",
})


def _canonicalize_cache_dtype(dtype: str | None) -> str | None:
    if dtype is None:
        return None
    norm = str(dtype).strip().lower()
    return _CACHE_DTYPE_ALIASES.get(norm, norm)


def _apply_cache_dtype_overrides(
    model_spec: Any,
    *,
    kv_cache_dtype: str | None,
    indexer_kv_dtype: str | None,
) -> Any:
    """Apply and validate KV & indexer cache dtype/scale overrides onto ModelSpec."""
    from dataclasses import replace as _replace
    from mindie_sim.catalog.types import QuantSpec
    from mindie_sim.schema.common import dtype_bytes

    has_kpool = any(
        int(getattr(lp, "index_kpool", 1) or 1) > 1
        for lp in getattr(model_spec, "layer_plans", ())
    )
    if kv_cache_dtype is None and indexer_kv_dtype is None:
        if model_spec.quant is None:
            return model_spec
        cur_kv = _canonicalize_cache_dtype(model_spec.quant.kv_dtype) or model_spec.quant.kv_dtype
        cur_idx = _canonicalize_cache_dtype(model_spec.quant.index_dtype) or model_spec.quant.index_dtype
        if has_kpool and cur_idx not in {"bf16", "fp8", "fp8_e4m3"}:
            raise ModelingError(f"kpool_indexer_kv_dtype_unsupported:{cur_idx}")
        cur_scale = model_spec.quant.index_scale_dtype
        if has_kpool and cur_idx in {"fp8", "fp8_e4m3"} and str(cur_scale or "") in {"", "none"}:
            cur_scale = "fp32"
        if (cur_kv != model_spec.quant.kv_dtype
                or cur_idx != model_spec.quant.index_dtype
                or cur_scale != model_spec.quant.index_scale_dtype):
            return _replace(
                model_spec,
                quant=_replace(
                    model_spec.quant, kv_dtype=cur_kv,
                    index_dtype=cur_idx, index_scale_dtype=cur_scale,
                ),
            )
        return model_spec

    kv_raw = _canonicalize_cache_dtype(kv_cache_dtype)
    idx_raw = _canonicalize_cache_dtype(indexer_kv_dtype)
    if has_kpool and idx_raw is not None and idx_raw not in _KPOOL_ALLOWED_INDEXER_KV_DTYPES:
        raise ModelingError(f"kpool_indexer_kv_dtype_unsupported:{idx_raw}")

    quant = model_spec.quant
    act = str(model_spec.activation_dtype or "bf16")
    if quant is None:
        quant = QuantSpec(
            name=act, weight_format=act, activation_format=act,
            expert_weight_format=act, kv_dtype=act, index_dtype=act,
            index_scale_dtype="none", native_format=act,
        )

    resolved_kv = (_canonicalize_cache_dtype(quant.kv_dtype) or act) if kv_raw in {None, "", "auto"} else kv_raw
    if idx_raw == "auto" and has_kpool:
        # Upstream ``glm5next/attention.py:115``: ``auto`` selects BF16 (no scales).
        resolved_idx = "bf16"
    elif idx_raw in {None, "", "auto"}:
        resolved_idx = _canonicalize_cache_dtype(quant.index_dtype) or act
    else:
        resolved_idx = idx_raw
    if has_kpool and str(resolved_idx).strip().lower() not in {"bf16", "fp8", "fp8_e4m3"}:
        raise ModelingError(f"kpool_indexer_kv_dtype_unsupported:{resolved_idx}")
    if kv_raw not in {None, "", "auto"}:
        dtype_bytes(resolved_kv)
    if idx_raw not in {None, "", "auto"}:
        dtype_bytes(resolved_idx)

    resolved_scale = quant.index_scale_dtype
    if has_kpool:
        idx_norm = str(resolved_idx).strip().lower()
        if idx_norm in {"fp8", "fp8_e4m3"}:
            # Upstream ``glm5next/kv_cache.py:117`` & ``indexer_kpool.py:1039``:
            # GLM KPool FP8 cache requires one FP32 per-pool scale.
            resolved_scale = "fp32"
        elif idx_norm == "bf16":
            resolved_scale = "none"

    updated = _replace(
        quant,
        kv_dtype=resolved_kv,
        index_dtype=resolved_idx,
        index_scale_dtype=resolved_scale,
    )
    return _replace(model_spec, quant=updated)


def _apply_cp_and_tutorial_flags(
    runtime_meta: dict[str, Any],
    *,
    model_spec: Any,
    parallel_dict: Mapping[str, Any],
    enable_flashcomm1: bool | None,
    enable_fused_mc2: int | None,
    enable_prefill_mc2: bool | None,
    enable_pcp_o_proj_weight_sharding: bool,
    enable_sfa_dcp_sharded_indexer: bool,
    dcp_q_replicate: bool,
) -> None:
    """Attach tutorial execution flags and non-blocking KV-replication metadata."""
    from mindie_sim.catalog.context_parallel import (
        kv_cache_replication_info,
        legal_vllm_ascend_dcp_values,
        validate_vllm_ascend_dcp_codomain,
    )
    from mindie_sim.schema.parallel import ParallelSpec

    additional = dict(runtime_meta.get("additional_config") or {})
    if enable_flashcomm1 is not None:
        additional["enable_flashcomm1"] = bool(enable_flashcomm1)
    if enable_fused_mc2 is not None:
        additional["enable_fused_mc2"] = int(enable_fused_mc2)
    if enable_prefill_mc2 is not None:
        additional["enable_prefill_mc2"] = bool(enable_prefill_mc2)
    if additional:
        runtime_meta["additional_config"] = additional
    if enable_pcp_o_proj_weight_sharding:
        runtime_meta["enable_pcp_o_proj_weight_sharding"] = True
    if enable_sfa_dcp_sharded_indexer:
        runtime_meta["enable_sfa_dcp_sharded_indexer"] = True
    if dcp_q_replicate:
        runtime_meta["dcp_q_replicate"] = True
    p_spec = ParallelSpec.from_mapping(parallel_dict)
    kv_rep = kv_cache_replication_info(model_spec, p_spec)
    runtime_meta["kv_cache_replicated_tp"] = bool(kv_rep["kv_cache_replicated_tp"])
    runtime_meta["kv_cache_replication_factor"] = int(kv_rep["kv_cache_replication_factor"])
    runtime_meta["canonical_kv_sharding"] = str(kv_rep["canonical_kv_sharding"])
    dcp_ok, dcp_reason = validate_vllm_ascend_dcp_codomain(model_spec, p_spec)
    runtime_meta["vllm_ascend_dcp_codomain_ok"] = bool(dcp_ok)
    if dcp_reason:
        runtime_meta["vllm_ascend_dcp_codomain_reason"] = str(dcp_reason)
    runtime_meta["vllm_ascend_dcp_legal_values"] = list(
        legal_vllm_ascend_dcp_values(model_spec, tp=p_spec.tp, pcp=p_spec.pcp)
    )
    runtime_meta["dcp_reuses_tp_ranks"] = bool(
        p_spec.cp > 1 and p_spec.pcp == 1 and not p_spec.cp_spans_dp
    )


def _resolve_parallel_axes(
    *,
    tp: int | None,
    dp: int | None,
    ep: int | None,
    pp: int | None,
    pcp: int | None,
    dcp: int | None,
    parallel: str | Mapping[str, Any] | None,
    deployment: str | None,
    num_devices: int | None,
    prompt_tokens: int | None,
    output_tokens: int | None,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float | None,
) -> tuple[dict[str, int | None], bool]:
    """Resolve explicit parallel axes or detect Domain-3 auto-optimal search."""
    if deployment is not None and str(deployment).strip().lower() not in _VALID_DEPLOYMENT_MODES:
        raise ModelingError(f"invalid_deployment_mode:{deployment}")
    raw_axes: dict[str, int | None] = {
        "tp": tp, "dp": dp, "ep": ep, "pp": pp, "pcp": pcp, "dcp": dcp,
    }
    parallel_mode: str | None = None
    if isinstance(parallel, Mapping):
        for axis in _PARALLEL_AXES:
            if axis in parallel and raw_axes[axis] is None:
                raw_axes[axis] = int(parallel[axis])
        if "cp" in parallel and raw_axes["dcp"] is None:
            raw_axes["dcp"] = int(parallel["cp"])
    elif parallel is not None:
        parallel_mode = str(parallel).strip().lower()
        if parallel_mode not in {"auto", "fixed"}:
            raise ModelingError(f"invalid_parallel_mode:{parallel}")

    for name, val in raw_axes.items():
        if val is not None and int(val) <= 0:
            raise ModelingError(f"invalid_parallel_dimension:{name}={val}")

    any_pinned = any(v is not None for v in raw_axes.values())
    has_full_serving_budget = (
        num_devices is not None
        and prompt_tokens is not None
        and output_tokens is not None
        and ttft_slo_ms is not None
        and tpot_slo_ms is not None
    )
    auto_mode = (
        parallel_mode == "auto"
        or (parallel_mode is None and not any_pinned and has_full_serving_budget)
    )
    if auto_mode:
        missing = [
            name for name, val in (
                ("num_devices", num_devices),
                ("prompt_tokens", prompt_tokens),
                ("output_tokens", output_tokens),
                ("ttft_slo_ms", ttft_slo_ms),
                ("tpot_slo_ms", tpot_slo_ms),
            ) if val is None
        ]
        if missing:
            raise missing_scenario_error(missing)
        pinned = {k: int(v) for k, v in raw_axes.items() if v is not None}
        return pinned, True

    resolved = {k: int(v if v is not None else DEFAULT_PARALLEL) for k, v in raw_axes.items()}
    ep_v, dp_v, tp_v, pcp_v = resolved["ep"], resolved["dp"], resolved["tp"], resolved["pcp"]
    if not (ep_v == 1 or (dp_v * tp_v * pcp_v) % ep_v == 0 or ep_v == dp_v * tp_v):
        raise ModelingError(f"invalid_ep_partition:ep={ep_v},dp={dp_v},tp={tp_v}")
    return resolved, False


def _resolve_scenario_and_slo(
    *,
    prompt_tokens: int | None,
    output_tokens: int | None,
    concurrency: int | None,
    prefix_tokens: int,
    arrival_gap_ms: float,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float | None,
    require_scenario: bool,
    model_limit: int,
    max_model_len: int | None,
    mtp_tokens: int,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
    prefix_cache_key: str,
    role: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float], int, int, int, bool]:
    """Validate Domain-1 scenario & SLO inputs without guessing prompt/output/SLA."""
    if ttft_slo_ms is not None and float(ttft_slo_ms) <= 0.0:
        raise ModelingError(f"invalid_ttft_slo_ms:{ttft_slo_ms}")
    if tpot_slo_ms is not None and float(tpot_slo_ms) <= 0.0:
        raise ModelingError(f"invalid_tpot_slo_ms:{tpot_slo_ms}")
    if require_scenario:
        missing = [
            name for name, val in (
                ("prompt_tokens", prompt_tokens),
                ("output_tokens", output_tokens),
            ) if val is None
        ]
        if missing:
            raise missing_scenario_error(missing)
    if (prompt_tokens is None) != (output_tokens is None):
        raise missing_scenario_error(
            ("prompt_tokens",) if prompt_tokens is None else ("output_tokens",)
        )
    if prompt_tokens is None and int(prefix_tokens) > 0:
        raise missing_scenario_error(("prompt_tokens", "output_tokens"))

    scenario_declared = prompt_tokens is not None and output_tokens is not None
    eff_conc = int(concurrency) if concurrency is not None else DEFAULT_CONCURRENCY
    if eff_conc <= 0:
        raise ModelingError("invalid_workload_dimensions")
    if scenario_declared:
        eff_prompt, eff_output = int(prompt_tokens), int(output_tokens)
        if eff_prompt <= 0 or eff_output <= 0:
            raise ModelingError("invalid_workload_dimensions")
        if int(prefix_tokens) < 0 or int(prefix_tokens) > eff_prompt:
            raise ModelingError("invalid_prefix_tokens")
    else:
        eff_prompt, eff_output = 1, 1

    required_len = eff_prompt + eff_output + max(0, mtp_tokens)
    if model_limit > 0 and required_len > model_limit:
        raise ModelingError(
            f"workload_exceeds_model_max_position_embeddings:{required_len}>{model_limit}"
        )
    if max_model_len is not None:
        if int(max_model_len) < required_len:
            raise ModelingError(
                f"max_model_len_smaller_than_workload:{max_model_len}<{required_len}"
            )
        resolved_max_model_len = int(max_model_len)
    else:
        resolved_max_model_len = max(DEFAULT_MAX_MODEL_LEN, required_len)

    resolved_max_seqs = (
        int(max_num_seqs) if max_num_seqs is not None else max(eff_conc, MIN_DEFAULT_MAX_SEQS)
    )
    if max_num_batched_tokens is not None:
        resolved_max_batched = int(max_num_batched_tokens)
    elif role == "pd_decode":
        # Decode instances never batch full prompt_tokens * concurrency; size by
        # decode verification rows so 128K-context PD decode does not OOM HBM.
        resolved_max_batched = max(
            resolved_max_seqs * (1 + max(0, mtp_tokens)),
            PD_DECODE_MIN_BATCHED_TOKENS,
        )
    else:
        resolved_max_batched = max(eff_prompt * eff_conc, MIN_DEFAULT_MAX_BATCHED_TOKENS)
    requests = [
        {
            "id": ("__structure_probe__" if not scenario_declared and concurrency is None
                   else f"req-{idx}"),
            "arrival_ms": float(idx * float(arrival_gap_ms)),
            "prompt_tokens": eff_prompt,
            "output_tokens": eff_output,
            "prefix_tokens": int(prefix_tokens) if scenario_declared else 0,
            "prefix_cache_key": str(prefix_cache_key),
        }
        for idx in range(eff_conc)
    ]
    slo_dict: dict[str, float] = {}
    if ttft_slo_ms is not None:
        slo_dict["ttft_ms"] = float(ttft_slo_ms)
    if tpot_slo_ms is not None:
        slo_dict["tpot_ms"] = float(tpot_slo_ms)
    return (
        requests, slo_dict, resolved_max_model_len,
        resolved_max_seqs, resolved_max_batched, scenario_declared,
    )


def _resolve_bool_feature(
    *,
    raw_eager: bool,
    explicit: bool | None,
    excluded: bool = False,
    profile_on: bool = False,
    structural_on: bool = False,
    profile_source: str = "profile",
) -> tuple[bool, str]:
    """Resolve one opt-out Domain-2 boolean feature with explicit provenance."""
    if raw_eager:
        return False, "raw_eager"
    if explicit is not None:
        return bool(explicit), "explicit"
    if excluded:
        return False, "excluded"
    if profile_on:
        return True, profile_source
    if structural_on:
        return True, "structural"
    return False, "default"


def _resolve_domain2_policy(
    *,
    hardware_backend: str,
    production: bool,
    va_recommended: bool | None,
    raw_eager: bool,
    va_profile: Mapping[str, Any] | None,
    model_spec: Any,
    axes: Mapping[str, int],
    sp: bool | None,
    graph_capture_mode: str | None,
    enable_torch_compile: bool | None,
    multistream_overlap_shared_expert: bool | None,
    engram_cpu_offload: bool | None,
    enable_shared_expert_dp: bool | None,
    enable_flashcomm1: bool | None,
    enable_fused_mc2: int | None,
    enable_prefill_mc2: bool | None,
    mtp_tokens: int,
    mtp_source: str,
    exclude_set: frozenset[str],
    notes: list[dict[str, Any]],
    canonical_track: str = "merged",
) -> dict[str, Any]:
    """Resolve Domain-2 model-intrinsic offload & production stack features (opt-out only)."""
    prof = va_profile or {}
    prof_flags = dict(prof.get("tutorial_flags") or {})
    # Preserve explicit stack disable workarounds (e.g. enable_fused_mc2=0) on
    # both merged and unmerged_pr ('Before') tracks; only expected_ideal lifts
    # them when evaluating theoretical ceiling unless caller explicitly pins 0.
    allow_merged_workarounds = canonical_track in {"merged", "unmerged_pr"}
    explicit_sp_flag = (sp is not None) or (enable_flashcomm1 is not None)
    eff_sp = bool(enable_flashcomm1) if (sp is None and enable_flashcomm1 is not None and not raw_eager) else sp

    resolved_mso, mso_src = _resolve_bool_feature(
        raw_eager=raw_eager, explicit=multistream_overlap_shared_expert,
        excluded="shared_expert_multistream_overlap" in exclude_set,
        profile_on=bool(production and axes["ep"] > 1 and prof.get("multistream_overlap_shared_expert", False)),
    )
    resolved_engram, engram_src = _resolve_bool_feature(
        raw_eager=raw_eager, explicit=engram_cpu_offload,
        profile_on=bool(production and prof.get("engram_cpu_offload", False)),
        structural_on=bool(production and getattr(model_spec, "engram", None) is not None),
    )
    resolved_se_dp, se_dp_src = _resolve_bool_feature(
        raw_eager=raw_eager, explicit=enable_shared_expert_dp,
        profile_on=bool(production and prof_flags.get("enable_shared_expert_dp")),
    )
    resolved_flashcomm1 = (
        None if raw_eager else
        bool(enable_flashcomm1) if enable_flashcomm1 is not None else
        None if "sequence_parallel" in exclude_set else
        bool(prof_flags["enable_flashcomm1"]) if (production and "enable_flashcomm1" in prof_flags) else None
    )
    prof_fused_mc2 = prof_flags.get("enable_fused_mc2")
    use_prof_fused_mc2 = (
        production and prof_fused_mc2 is not None
        and (allow_merged_workarounds or int(prof_fused_mc2) != 0)
    )
    resolved_fused_mc2 = (
        None if raw_eager else
        int(enable_fused_mc2) if enable_fused_mc2 is not None else
        None if "cann_megamoe_fusion" in exclude_set else
        int(prof_fused_mc2) if use_prof_fused_mc2 else None
    )
    resolved_prefill_mc2 = (
        None if raw_eager else
        bool(enable_prefill_mc2) if enable_prefill_mc2 is not None else
        bool(prof_flags["enable_prefill_mc2"]) if (production and "enable_prefill_mc2" in prof_flags) else None
    )
    sp_profile_on = (
        bool(_supports_sequence_parallel(
            model_spec, va_profile,
            tp=axes["tp"], ep=axes["ep"], pp=axes["pp"], pcp=axes["pcp"], notes=notes,
        ))
        if (not raw_eager and eff_sp is None and "sequence_parallel" not in exclude_set
            and production and va_profile is not None)
        else False
    )
    resolved_sp, sp_src = _resolve_bool_feature(
        raw_eager=raw_eager, explicit=eff_sp,
        excluded="sequence_parallel" in exclude_set,
        profile_on=sp_profile_on, profile_source="production",
    )
    documented_capture = prof.get("graph_capture_mode")
    if raw_eager:
        resolved_capture, cap_src = "none", "raw_eager"
    elif graph_capture_mode is not None:
        resolved_capture, cap_src = str(graph_capture_mode), "explicit"
    elif "aclgraph_decode_capture" in exclude_set:
        resolved_capture, cap_src = "none", "excluded"
    elif production and documented_capture in {"full", "piecewise", "none"}:
        resolved_capture, cap_src = str(documented_capture), "profile"
    elif production:
        resolved_capture, cap_src = "full", "production_default"
    else:
        resolved_capture, cap_src = "none", "none"

    supports_compile = bool(dict(model_spec.catalog_facts).get("supports_torch_compile", True))
    initial_compile = (
        False if raw_eager else
        bool(enable_torch_compile) if enable_torch_compile is not None else supports_compile
    )
    tutorial_applied = {
        k: v for k, v in (
            ("enable_shared_expert_dp", resolved_se_dp or None),
            ("enable_flashcomm1", resolved_flashcomm1),
            ("enable_fused_mc2", resolved_fused_mc2),
            ("enable_prefill_mc2", resolved_prefill_mc2),
        ) if v is not None
    }
    policy = {
        "backend": hardware_backend, "production": bool(production),
        "va_recommended": va_recommended, "raw_eager": bool(raw_eager),
        "automatic_optional_fusions": bool(not raw_eager and va_recommended is not False),
        "sp": bool(resolved_sp), "sp_source": sp_src,
        "graph_capture_mode": str(resolved_capture), "capture_source": cap_src,
        "enable_torch_compile": initial_compile,
        "engram_cpu_offload": bool(resolved_engram), "engram_cpu_offload_source": engram_src,
        "multistream_overlap_shared_expert": bool(resolved_mso), "mso_source": mso_src,
        "enable_shared_expert_dp": bool(resolved_se_dp), "enable_shared_expert_dp_source": se_dp_src,
        "mlapo": bool(production and "VLLM_ASCEND_ENABLE_MLAPO" in (prof.get("env") or {})),
        "tutorial_flags_applied": tutorial_applied,
        "mtp_proposal_tokens": int(mtp_tokens), "mtp_source": str(mtp_source),
        "recommended_mtp_proposal_tokens": int(prof.get("mtp_proposal_tokens", 0) or 0),
        "profile": str(prof.get("model", "")) if va_profile else "",
        "priority": "raw_eager > explicit caller > exclude_optimizations > documented profile > structural defaults",
    }
    return {
        "mso": resolved_mso, "engram_offload": resolved_engram,
        "enable_shared_expert_dp": resolved_se_dp,
        "enable_flashcomm1": resolved_flashcomm1, "enable_fused_mc2": resolved_fused_mc2,
        "enable_prefill_mc2": resolved_prefill_mc2,
        "sp": resolved_sp, "explicit_sp": explicit_sp_flag,
        "graph_capture_mode": resolved_capture, "enable_torch_compile": initial_compile,
        "execution_policy": policy,
    }


def _apply_va_profile_lowerings(
    runtime_dict: dict[str, Any],
    runtime_meta: dict[str, Any],
    *,
    model_spec: Any,
    va_profile: Mapping[str, Any],
    sku: str,
    excluded_selectors: frozenset[str],
    excluded_passes: frozenset[str],
    notes: list[dict[str, Any]],
) -> None:
    """Attach documented vLLM-Ascend profile compilation config, env, and lowerings."""
    runtime_meta["va_recommended"] = True
    runtime_meta["va_profile"] = deepcopy(dict(va_profile))
    compilation = va_profile.get("compilation_config")
    if (isinstance(compilation, dict)
            and str(compilation.get("cudagraph_mode") or "").upper() in {"FULL", "FULL_DECODE_ONLY"}
            and str(runtime_dict.get("graph_capture_mode")) == "full"):
        runtime_dict["compilation_config"] = deepcopy(compilation)
    all2all = va_profile.get("all2all_backend")
    if isinstance(all2all, str) and all2all:
        runtime_dict["all2all_backend"] = all2all
    runtime_meta["va_env"] = dict(va_profile["env"])
    runtime_meta["va_recommended_mtp_proposal_tokens"] = int(va_profile["mtp_proposal_tokens"])
    passes: tuple[str, ...] = ()
    if (model_spec.quant is not None
            and str(model_spec.quant.activation_format) != str(model_spec.activation_dtype)):
        passes = ("activation_quantization", "norm_quant")
        runtime_dict["lowering_passes"] = list(passes)
    lowerings = _resolve_va_lowerings(
        model_spec, runtime_dict["parallel"], sku, passes, notes=notes,
    )
    lowerings = {k: v for k, v in lowerings.items() if k not in excluded_selectors}
    if excluded_passes:
        passes = tuple(name for name in passes if name not in excluded_passes)
        if passes:
            runtime_dict["lowering_passes"] = list(passes)
        else:
            runtime_dict.pop("lowering_passes", None)
    if "add_rmsnorm" in lowerings and "residual_norm" not in passes:
        passes = (*passes, "residual_norm")
        runtime_dict["lowering_passes"] = list(passes)
    if lowerings:
        kernels_dict = dict(runtime_dict.get("kernels", {}))
        existing_low = dict(kernels_dict.get("lowerings", {}))
        kernels_dict["lowerings"] = {**lowerings, **existing_low}
        runtime_dict["kernels"] = kernels_dict
    runtime_dict["metadata"] = runtime_meta


def _attach_expected_ideal_mla_prolog_fusion(
    lowerings: dict[str, Any],
    *,
    canonical_track: str,
    exclude_optimizations: Sequence[str],
    axes: Mapping[str, int],
    sku: str,
) -> None:
    """Fold ``mla_prolog`` into one ``aclnnMlaPrologV3`` native kernel on ``expected_ideal``."""
    if not (
        canonical_track == "expected_ideal"
        and "mla_prolog" in lowerings
        and "mla_prolog_fusion" not in set(exclude_optimizations)
        and int(axes.get("pcp", 1)) == 1
    ):
        return
    prolog_cfg = lowerings["mla_prolog"]
    prolog_opts = (
        dict(prolog_cfg.get("options") or {})
        if isinstance(prolog_cfg, Mapping) else {}
    )
    prolog_profile = (
        "mla_prolog_packed_int8_tile128_analytic"
        if prolog_opts.get("cache_layout")
        else "mla_prolog_arch35_split_m_n_ae923905"
        if _arch_key(sku) == "a5"
        else "preserve_workspaces"
    )
    existing_nf = lowerings.get("native_fusion")
    nf_opts = (
        dict(existing_nf.get("options") or {})
        if isinstance(existing_nf, Mapping) else {}
    )
    nf_roles = list(nf_opts.get("roles") or ())
    if "mla_prolog" not in nf_roles:
        nf_roles.append("mla_prolog")
    raw_tp = nf_opts.get("traffic_profile")
    tp_map = (
        dict(raw_tp) if isinstance(raw_tp, Mapping)
        else {r: raw_tp for r in nf_roles if raw_tp}
    )
    tp_map.setdefault("mla_prolog", prolog_profile)
    nf_opts["roles"] = nf_roles
    nf_opts["traffic_profile"] = tp_map
    lowerings["native_fusion"] = {
        "implementation": (
            existing_nf.get("implementation", "declared_native_fusion")
            if isinstance(existing_nf, Mapping) else "declared_native_fusion"
        ),
        "options": nf_opts,
    }


def _apply_stack_and_lowerings(
    runtime_dict: dict[str, Any],
    *,
    model_spec: Any,
    model_dict: dict[str, Any],
    machine: Mapping[str, Any],
    sku: str,
    hardware_backend: str,
    va_profile: Mapping[str, Any] | None,
    va_recommended: bool | None,
    raw_eager: bool,
    track_raw_eager: bool,
    canonical_track: str,
    runtime_stack: str | Path | Mapping[str, Any],
    optimizations: Sequence[str],
    exclude_optimizations: Sequence[str],
    caller_explicit: frozenset[str],
    v41_structural: bool,
    axes: Mapping[str, int],
    notes: list[dict[str, Any]],
    _stack_phase: str | None,
) -> None:
    """Apply documented profile lowerings, stack mechanisms, and compilation spine."""
    ascend_backend = hardware_backend == "ascend"
    prior_kernels = snapshot_kernel_keys(runtime_dict.get("kernels"))
    runtime_meta = dict(runtime_dict.get("metadata") or {})
    excluded_selectors: frozenset[str] = frozenset()
    excluded_passes: frozenset[str] = frozenset()
    if ascend_backend and not raw_eager:
        from mindie_sim.stack_tracks import excluded_lowering_bindings
        excluded_selectors, excluded_passes = excluded_lowering_bindings(exclude_optimizations)

    if va_profile is not None:
        _apply_va_profile_lowerings(
            runtime_dict, runtime_meta, model_spec=model_spec, va_profile=va_profile,
            sku=sku, excluded_selectors=excluded_selectors, excluded_passes=excluded_passes,
            notes=notes,
        )

    mechanism_track_active = False
    if ascend_backend and (not raw_eager or track_raw_eager):
        from mindie_sim.feature_mechanisms import MechanismContext
        from mindie_sim.stack_tracks import (
            apply_resolved_mechanisms, resolve_runtime_stack, resolve_stack_track,
        )
        if (track_raw_eager or va_recommended is not False
                or bool(optimizations) or canonical_track != "merged"):
            mechanism_track_active = not raw_eager
            stack_profile = resolve_runtime_stack(runtime_stack)
            stack_model_key = str((va_profile or {}).get("canonical_model") or model_spec.name)
            ctx_opts: dict[str, Any] = {
                "ranks_per_node": int(machine.get("ranks_per_node", 0) or 0),
            }
            additional_cfg = runtime_meta.get("additional_config") or {}
            if isinstance(additional_cfg, Mapping) and additional_cfg.get("enable_fused_mc2") is not None:
                ctx_opts["enable_fused_mc2"] = additional_cfg["enable_fused_mc2"]
            if _stack_phase:
                ctx_opts["phase"] = _stack_phase
            mechanism_ctx = MechanismContext(
                model=model_spec, sku=sku, arch=_arch_key(sku),
                parallel=dict(runtime_dict["parallel"]),
                caller_explicit=caller_explicit, options=ctx_opts,
            )
            resolved_track = resolve_stack_track(
                stack_profile, track=canonical_track, model_key=stack_model_key,
                ctx=mechanism_ctx, optimizations=optimizations,
                exclude_optimizations=exclude_optimizations,
            )
            if not raw_eager or optimizations:
                apply_resolved_mechanisms(runtime_dict, resolved_track, mechanism_ctx)
            runtime_meta = dict(runtime_dict.get("metadata") or {})
            runtime_meta["feature_track_report"] = resolved_track
            runtime_dict["metadata"] = runtime_meta

    if ascend_backend and not raw_eager:
        from mindie_sim.compose.operator_defaults import (
            DECLARATION_PHASE, apply_compose_v41_a5_defaults,
        )
        from mindie_sim.prediction.implementation_selection import operator_selection_summary
        runtime_meta = dict(runtime_dict.get("metadata") or {})
        kernels = dict(runtime_dict.get("kernels") or {})
        lowerings = dict(kernels.get("lowerings") or {})
        notes.extend(apply_compose_v41_a5_defaults(
            model_dict=model_dict, lowerings=lowerings, backend=hardware_backend,
            arch=_arch_key(sku), layer_plans=model_spec.layer_plans,
            activation_dtype=model_spec.activation_dtype, tp=axes["tp"], cp=axes["dcp"],
            runtime_metadata=runtime_meta, structural_v41=v41_structural,
        ))
        if excluded_selectors:
            lowerings = {k: v for k, v in lowerings.items() if k not in excluded_selectors}
        if lowerings:
            kernels["lowerings"] = lowerings
            runtime_dict["kernels"] = kernels
        runtime_meta["operator_selection_summary"] = operator_selection_summary(
            arch_family=_arch_key(sku), phase=DECLARATION_PHASE, notes=notes,
        )
        runtime_dict["metadata"] = runtime_meta

    if ascend_backend and not raw_eager and (va_recommended is not False or mechanism_track_active):
        from mindie_sim.feature_mechanisms import ensure_norm_quant_domain
        kernels = dict(runtime_dict.get("kernels") or {})
        lowerings = dict(kernels.get("lowerings") or {})
        if any(plan.ffn == "moe" for plan in model_spec.layer_plans):
            kernels.setdefault("moe_gating", {"implementation": "vllm_ascend_cann_gating"})
        if any(int(getattr(plan, "compress_ratio", 0) or 0) > 0 for plan in model_spec.layer_plans):
            lowerings.setdefault("compressor", "vllm_ascend_cann_compressor")
        if any(int(getattr(plan, "index_kpool", 1) or 1) > 1 for plan in model_spec.layer_plans):
            lowerings.setdefault("pool_key_indexer", "vllm_ascend_cann_keypool_caller")
            model_dict.setdefault("metadata", {})["native_keypool_caller"] = True
        if "norm_quant" in tuple(runtime_dict.get("lowering_passes") or ()):
            kernels = ensure_norm_quant_domain(kernels)
        _attach_expected_ideal_mla_prolog_fusion(
            lowerings,
            canonical_track=canonical_track,
            exclude_optimizations=exclude_optimizations,
            axes=axes,
            sku=sku,
        )
        if lowerings:
            kernels["lowerings"] = lowerings
        if kernels:
            runtime_dict["kernels"] = kernels

    finalize_compilation_spine(runtime_dict, raw_eager=raw_eager)
    stamp_stack_profile_origins(runtime_dict, prior_keys=prior_kernels)
    runtime_meta = dict(runtime_dict.get("metadata") or {})
    policy = dict(runtime_meta.get("execution_policy") or {})
    policy["multistream_overlap_shared_expert"] = bool(
        runtime_dict.get("multistream_overlap_shared_expert", False)
    )
    policy["engram_cpu_offload"] = bool(runtime_dict.get("engram_cpu_offload", False))
    policy["sp"] = bool(runtime_meta.get("sp", False))
    runtime_meta["execution_policy"] = policy
    runtime_dict["metadata"] = runtime_meta


def _apply_hardware_derivation(
    case_dict: dict[str, Any],
    *,
    hardware_derivation: Mapping[str, Any] | None,
    aic_count: int | None,
    aiv_count: int | None,
    memory_bytes: int | None,
    hbm_peak_bytes_per_s: float | None,
    hbm_efficiency: Any,
    mte2_ceiling_bytes_per_s: float | None,
    mte2_efficiency: Any,
) -> dict[str, Any]:
    derivation = dict(hardware_derivation or {})
    for key, value in (
        ("aic_count", aic_count), ("aiv_count", aiv_count),
        ("memory_bytes", memory_bytes), ("hbm_peak_bytes_per_s", hbm_peak_bytes_per_s),
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


def _build_runtime_dict(
    *,
    axes: Mapping[str, int],
    machine: Mapping[str, Any],
    model_spec: Any,
    policy_out: Mapping[str, Any],
    runtime_meta: dict[str, Any],
    q_proj_tp: int,
    o_proj_tp: int,
    linear_proj_tp: int,
    dcp_linear_tp: int,
    mlp_tp: int,
    shared_expert_tp: int,
    lm_head_tp: int,
    redundant_experts_per_rank: int,
    collective_startup_mode: str | None,
    matrix_compute_policy: str,
    recover_ssm: bool,
    prefix_cache_mode: str,
    prefix_read_mode: str,
    homogeneous_graph_reuse: str,
    resolved_max_seqs: int,
    resolved_max_batched: int,
    resolved_max_model_len: int,
    resolved_block_size: int,
    enable_chunked_prefill: bool,
    async_scheduling: bool,
    mtp_tokens: int,
    mtp_acceptance_length: float,
    enable_npugraph_ex: bool | None,
    enable_static_kernel: bool | None,
    enable_super_kernel: bool | None,
    enable_dbo: bool,
    pp_layer_partition: Sequence[int] | None,
    draft_max_model_len: int | None = None,
    draft_enforce_eager: bool | None = None,
) -> dict[str, Any]:
    del draft_max_model_len, draft_enforce_eager
    tp, dp, ep, pp, pcp, dcp = (
        axes["tp"], axes["dp"], axes["ep"], axes["pp"], axes["pcp"], axes["dcp"],
    )
    graph_capture_mode = str(policy_out["graph_capture_mode"])
    runtime_dict: dict[str, Any] = {
        "parallel": {
            "dp": dp, "tp": tp, "pp": pp, "ep": ep, "cp": dcp, "dcp": dcp, "pcp": pcp,
            **({"rank_table": [
                stage_ranks[offset:offset + ep]
                for stage in range(pp)
                for stage_ranks in ([[
                    int(machine["dies"][idx]["rank"]) if machine.get("dies") else idx
                    for d in range(dp)
                    for c in range(pcp)
                    for t in range(tp)
                    for idx in (((d * pp + stage) * pcp + c) * tp + t,)
                ]][0],)
                for offset in range(0, len(stage_ranks), ep)
            ]}
               if pcp > 1 and ep > dp * tp and (dp * tp * pcp) % ep == 0 else {}),
            "q_proj_tp": int(q_proj_tp), "o_proj_tp": int(o_proj_tp),
            "linear_proj_tp": int(linear_proj_tp), "dcp_linear_tp": int(dcp_linear_tp),
            "mlp_tp": int(mlp_tp), "shared_expert_tp": int(shared_expert_tp),
            "lm_head_tp": int(lm_head_tp),
            "redundant_experts_per_rank": int(redundant_experts_per_rank),
        },
        "collective_startup_mode": (
            str(collective_startup_mode) if collective_startup_mode is not None
            else "topology" if machine.get("collective_protocols") else "legacy"
        ),
        "matrix_compute_policy": str(matrix_compute_policy),
        "recover_ssm": bool(recover_ssm),
        "prefix_cache_mode": str(prefix_cache_mode),
        "prefix_read_mode": str(prefix_read_mode),
        "homogeneous_graph_reuse": str(homogeneous_graph_reuse),
        "max_num_seqs": resolved_max_seqs,
        "max_num_batched_tokens": resolved_max_batched,
        "max_model_len": resolved_max_model_len,
        "kv_block_size": int(resolved_block_size),
        "enable_chunked_prefill": bool(enable_chunked_prefill),
        "async_scheduling": bool(async_scheduling),
        "graph_mode": graph_capture_mode,
        "graph_capture_mode": graph_capture_mode,
        "enable_torch_compile": bool(policy_out["enable_torch_compile"]),
        "mtp_acceptance_length": float(mtp_acceptance_length),
        "metadata": runtime_meta,
    }
    for flag_name, flag_val in (
        ("enable_npugraph_ex", enable_npugraph_ex),
        ("enable_static_kernel", enable_static_kernel),
        ("enable_super_kernel", enable_super_kernel),
    ):
        if flag_val is not None:
            runtime_dict[flag_name] = bool(flag_val)
    if policy_out["mso"]:
        runtime_dict["multistream_overlap_shared_expert"] = True
    if policy_out["engram_offload"]:
        runtime_dict["engram_cpu_offload"] = True
    if enable_dbo:
        runtime_dict["enable_dbo"] = True
    if graph_capture_mode == "full":
        runtime_dict["graph_capture_keys"] = capture_keys(resolved_max_seqs, mtp_tokens)
    elif graph_capture_mode == "piecewise":
        runtime_dict["piecewise_capture_keys"] = sorted(
            {*PIECEWISE_CAPTURE_BASE_KEYS, resolved_max_batched}
        )
    if pp_layer_partition is not None:
        runtime_dict["pp_layer_partition"] = [int(x) for x in pp_layer_partition]
    elif pp > 1 and model_spec.layers % pp != 0:
        hi = (model_spec.layers + pp - 1) // pp
        if hi * (pp - 1) < model_spec.layers:
            runtime_dict["pp_layer_partition"] = [hi] * (pp - 1) + [model_spec.layers - hi * (pp - 1)]
        else:
            runtime_dict["pp_layer_partition"] = [
                model_spec.layers // pp + (1 if i < model_spec.layers % pp else 0)
                for i in range(pp)
            ]
    if policy_out["sp"]:
        from mindie_sim.prediction.sharding.sequence_parallel import DECLARATION, _validate_model
        if pcp < 1 or pp < 1 or tp <= 1 or ep < tp or (ep % tp != 0):
            raise ModelingError(f"sequence_parallel_invalid_topology:tp={tp},ep={ep},pp={pp},pcp={pcp}")
        _validate_model(model_spec, padding="right_to_tp")
        runtime_dict["kernels"] = {"sequence_parallel": dict(DECLARATION, padding="right_to_tp")}
    return runtime_dict


def _resolve_mtp_policy(
    *,
    mtp_proposal_tokens: int | None,
    mtp_acceptance_length: float | None,
    raw_eager: bool,
    production: bool,
    va_profile: Mapping[str, Any] | None,
    model_spec: Any,
    prompt_tokens: int | None,
    output_tokens: int | None,
    ttft_slo_ms: float | None,
    tpot_slo_ms: float | None,
    require_scenario: bool,
    model_limit: int,
    max_model_len: int | None,
) -> tuple[int, float, str]:
    """Resolve Domain-2 MTP proposal width & acceptance length (opt-out with 0)."""
    if mtp_proposal_tokens is not None:
        tokens = int(mtp_proposal_tokens)
        if tokens < 0:
            raise ModelingError(f"invalid_mtp_proposal_tokens:{tokens}")
        accept = (
            float(mtp_acceptance_length)
            if mtp_acceptance_length is not None
            else DEFAULT_MTP_ACCEPT
        )
        return tokens, accept, "explicit"
    if raw_eager:
        return DEFAULT_MTP_TOKENS, DEFAULT_MTP_ACCEPT, "raw_eager"
    scenario_declared = prompt_tokens is not None and output_tokens is not None
    if not (production and va_profile is not None and scenario_declared):
        return DEFAULT_MTP_TOKENS, DEFAULT_MTP_ACCEPT, "structure_probe"
    out_toks = int(output_tokens)
    if out_toks <= 1:
        return DEFAULT_MTP_TOKENS, DEFAULT_MTP_ACCEPT, "prefill_only"
    prof_mtp = int(va_profile.get("mtp_proposal_tokens", 0) or 0)
    if prof_mtp <= 0:
        return DEFAULT_MTP_TOKENS, DEFAULT_MTP_ACCEPT, "profile_none"
    has_slo = ttft_slo_ms is not None or tpot_slo_ms is not None or require_scenario
    if not has_slo and out_toks <= prof_mtp:
        return DEFAULT_MTP_TOKENS, DEFAULT_MTP_ACCEPT, "short_output"
    aux = getattr(model_spec, "auxiliary", None)
    if aux is not None and getattr(aux, "standalone_draft", False):
        return DEFAULT_MTP_TOKENS, DEFAULT_MTP_ACCEPT, "no_structural_draft"
    probe_spec = ensure_native_mla_draft(
        prepare_speculation(model_spec, prof_mtp, allow_synthetic_gqa=False),
        prof_mtp,
    )
    probe_aux = getattr(probe_spec, "auxiliary", None)
    has_draft = bool(
        probe_aux is not None
        and (getattr(probe_aux, "mtp", None) is not None
             or getattr(probe_aux, "dspark", None) is not None)
    )
    if not has_draft:
        return DEFAULT_MTP_TOKENS, DEFAULT_MTP_ACCEPT, "no_structural_draft"
    needed = int(prompt_tokens) + out_toks + prof_mtp
    if (model_limit > 0 and needed > model_limit) or (max_model_len is not None and needed > int(max_model_len)):
        return DEFAULT_MTP_TOKENS, DEFAULT_MTP_ACCEPT, "max_len_bounded"
    from mindie_sim.prediction.frontend.speculation import compute_speculative_accept_step
    accept = (
        float(mtp_acceptance_length)
        if mtp_acceptance_length is not None
        else compute_speculative_accept_step(DEFAULT_SPECULATIVE_ACCEPT_RATE, prof_mtp)
    )
    return prof_mtp, accept, "profile"


def _prepare_composed_model(
    *,
    model: str | Path | Mapping[str, Any],
    sku: str,
    machine: Mapping[str, Any],
    production: bool,
    quant: str | Mapping[str, Any] | None,
    kv_cache_dtype: str | None,
    indexer_kv_dtype: str | None,
    layers: int | None,
    n_routed_experts: int | None,
    num_experts_per_tok: int | None,
    promote_dense_to_moe: bool,
    role: str | None = None,
) -> tuple[Any, int, dict[str, Any] | None, bool]:
    model_spec = _resolve_model_spec(model)
    model_limit = _resolve_max_position_embeddings(model, model_spec)
    explicit_quant = quant is not None
    spec_meta = dict(getattr(model_spec, "metadata", ()) or ())
    checkpoint_quant_declared = bool(spec_meta.get("checkpoint_quant_declared"))
    va_profile: dict[str, Any] | None = None
    if production:
        model_id = (
            str(model_spec.name)
            if spec_meta.get("checkpoint_repo_id") or not isinstance(model, str)
            else str(model)
        )
        va_profile = va_recommended_config(model_id, sku, role=role)
        va_profile["quantization_action"] = (
            "explicit_quant_preserved" if explicit_quant else
            "checkpoint_quant_preserved" if (checkpoint_quant_declared and model_spec.quant is not None) else
            "documented_alternative_applied" if va_profile["quant"] is not None else
            "no_profile_override"
        )
        va_profile.setdefault("graph_capture_mode", None)
        if not explicit_quant and not (checkpoint_quant_declared and model_spec.quant is not None):
            quant = va_profile["quant"]
    model_spec = _apply_quant_override(model_spec, quant, machine)
    resolved_kv_dtype, resolved_idx_dtype = kv_cache_dtype, indexer_kv_dtype
    explicit_kv_in_quant = isinstance(quant, Mapping) and bool(quant.get("kv_dtype"))
    explicit_idx_in_quant = isinstance(quant, Mapping) and bool(quant.get("index_dtype"))
    unquantized_override = (
        explicit_quant and isinstance(quant, str)
        and quant.strip().lower() in {"bf16", "bfloat16", "fp16", "float16", "fp32"}
    )
    if (production and va_profile is not None
            and va_profile.get("quant") is not None and not unquantized_override
            and not (checkpoint_quant_declared and not explicit_quant)):
        prof_flags = dict(va_profile.get("tutorial_flags") or {})
        if resolved_kv_dtype is None and not explicit_kv_in_quant and prof_flags.get("kv_cache_dtype"):
            resolved_kv_dtype = str(prof_flags["kv_cache_dtype"])
        if resolved_idx_dtype is None and not explicit_idx_in_quant and prof_flags.get("indexer_kv_dtype"):
            resolved_idx_dtype = str(prof_flags["indexer_kv_dtype"])
    model_spec = _apply_cache_dtype_overrides(
        model_spec, kv_cache_dtype=resolved_kv_dtype, indexer_kv_dtype=resolved_idx_dtype,
    )
    from mindie_sim.compose.operator_defaults import structural_v41_latent_mqa
    v41_structural = structural_v41_latent_mqa(model_spec.layer_plans)
    model_spec = _apply_structural_overrides(
        model_spec, layers=layers, n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok, promote_dense_to_moe=promote_dense_to_moe,
    )
    return model_spec, model_limit, va_profile, v41_structural


def _compose_auto_deployment(
    model, hardware, axes, *, num_devices, prompt_tokens, output_tokens,
    ttft_slo_ms, tpot_slo_ms, deployment, **options,
):
    from mindie_sim.sweep.pd import solve_optimal_deployment

    solved = solve_optimal_deployment(
        model=model, hardware=hardware, num_devices=int(num_devices),
        prompt_tokens=int(prompt_tokens), output_tokens=int(output_tokens),
        ttft_sla_ms=float(ttft_slo_ms), tpot_sla_ms=float(tpot_slo_ms),
        deployment=str(deployment or "auto"), pinned_parallel=axes, **options,
    )
    winner_case = deepcopy(solved["recommended_case"])
    winner_case.setdefault("runtime", {}).setdefault("metadata", {})["optimal_deployment"] = {
        k: v for k, v in solved.items() if k not in {"recommended_case", "prefill_case", "decode_case"}
    }
    return winner_case


def _compose_explicit_case(
    model, hardware, axes, norm_role, *, graph_capture_mode, raw_eager, feature_track,
    va_recommended, machine_options, model_options, mtp_options, scenario_options,
    policy_options, runtime_options, cp_options, stack_options, moe_comm_mode,
    block_size, allow_synthetic_gqa_draft,
):
    resolved_axes = {k: int(v) for k, v in axes.items() if v is not None}
    if graph_capture_mode is not None and graph_capture_mode not in {"none", "full", "piecewise"}:
        raise ModelingError(f"invalid_graph_capture_mode:{graph_capture_mode}")

    sku, hardware_backend, machine = _materialize_machine(
        hardware,
        required_devices=resolved_axes["dp"] * resolved_axes["tp"] * resolved_axes["pp"] * resolved_axes["pcp"],
        **machine_options,
    )
    ascend_backend = hardware_backend == "ascend"
    from mindie_sim.stack_tracks import normalize_feature_track
    canonical_track = normalize_feature_track(feature_track) if ascend_backend else "merged"
    track_raw_eager = canonical_track == "raw_eager"
    if track_raw_eager:
        raw_eager = True
    if raw_eager and va_recommended:
        raise ModelingError("raw_eager_conflicts_with_explicit_fusion")
    production = ascend_backend and va_recommended is not False and not raw_eager

    model_spec, model_limit, va_profile, v41_structural = _prepare_composed_model(
        model=model, sku=sku, machine=machine, production=production, role=norm_role,
        **model_options,
    )
    mtp_tokens, resolved_mtp_accept, mtp_source = _resolve_mtp_policy(
        raw_eager=raw_eager, production=production, va_profile=va_profile,
        model_spec=model_spec, model_limit=model_limit,
        prompt_tokens=scenario_options["prompt_tokens"],
        output_tokens=scenario_options["output_tokens"],
        ttft_slo_ms=scenario_options["ttft_slo_ms"],
        tpot_slo_ms=scenario_options["tpot_slo_ms"],
        require_scenario=scenario_options["require_scenario"],
        max_model_len=scenario_options["max_model_len"],
        **mtp_options,
    )
    model_spec = prepare_speculation(
        model_spec, mtp_tokens, allow_synthetic_gqa=allow_synthetic_gqa_draft,
    )
    if block_size is not None:
        resolved_block_size = int(block_size)
    else:
        resolved_block_size = int((va_profile or {}).get("kv_block_size") or DEFAULT_BLOCK_SIZE)
        if dict(model_spec.metadata).get("cache_allocator") == "paired_sfa":
            from mindie_sim.catalog.paired_sfa import minimum_block_size
            resolved_block_size = max(
                resolved_block_size,
                minimum_block_size(model_spec, tp=resolved_axes["tp"], proposals=mtp_tokens),
            )

    notes: list[dict[str, Any]] = []
    exclude_set = frozenset(
        str(x).strip() for x in stack_options["exclude_optimizations"] if str(x).strip()
    )
    policy_out = _resolve_domain2_policy(
        hardware_backend=hardware_backend, production=production,
        va_recommended=va_recommended, raw_eager=raw_eager,
        va_profile=va_profile, model_spec=model_spec, axes=resolved_axes,
        graph_capture_mode=graph_capture_mode,
        mtp_tokens=mtp_tokens, mtp_source=mtp_source,
        exclude_set=exclude_set, notes=notes, canonical_track=canonical_track,
        **policy_options,
    )

    model_spec = ensure_native_mla_draft(model_spec, mtp_tokens)
    if mtp_tokens > 0:
        aux = model_spec.auxiliary
        if getattr(aux, "mtp", None) is None and getattr(aux, "dspark", None) is None:
            raise ModelingError("draft_structure_required_for_mtp_proposal_tokens")
        if not (1.0 <= float(resolved_mtp_accept) <= float(1 + mtp_tokens)):
            raise ModelingError("invalid_mtp_acceptance_length")

    (requests, slo_dict, resolved_max_model_len,
     resolved_max_seqs, resolved_max_batched, scenario_declared) = _resolve_scenario_and_slo(
        model_limit=model_limit, mtp_tokens=mtp_tokens, role=norm_role, **scenario_options,
    )

    model_spec = _tier4_router_spec(model_spec, sku, raw_eager=raw_eager)
    if ascend_backend and _arch_key(sku) == "a5":
        from mindie_sim.compose.operator_defaults import resolve_packed_indexed_mla_cache
        model_spec = resolve_packed_indexed_mla_cache(model_spec)
    model_dict = _to_jsonable(model_spec)
    model_dict["n_routed_experts"] = int(model_spec.num_experts)
    model_dict["num_experts_per_tok"] = int(model_spec.top_k)

    return _assemble_composed_case(
        model_spec, model_dict, machine, requests, slo_dict,
        resolved_axes=resolved_axes, sku=sku, hardware_backend=hardware_backend,
        ascend_backend=ascend_backend, norm_role=norm_role, scenario_declared=scenario_declared,
        policy_out=policy_out, raw_eager=raw_eager, moe_comm_mode=moe_comm_mode,
        mtp_tokens=mtp_tokens, resolved_mtp_accept=resolved_mtp_accept,
        resolved_max_seqs=resolved_max_seqs, resolved_max_batched=resolved_max_batched,
        resolved_max_model_len=resolved_max_model_len, resolved_block_size=resolved_block_size,
        graph_capture_mode=graph_capture_mode, canonical_track=canonical_track,
        track_raw_eager=track_raw_eager, va_profile=va_profile, va_recommended=va_recommended,
        v41_structural=v41_structural, notes=notes, policy_options=policy_options,
        runtime_options=runtime_options, cp_options=cp_options, stack_options=stack_options,
    )


def _assemble_composed_case(
    model_spec, model_dict, machine, requests, slo_dict, *,
    resolved_axes, sku, hardware_backend, ascend_backend, norm_role, scenario_declared,
    policy_out, raw_eager, moe_comm_mode, mtp_tokens, resolved_mtp_accept,
    resolved_max_seqs, resolved_max_batched, resolved_max_model_len, resolved_block_size,
    graph_capture_mode, canonical_track, track_raw_eager, va_profile, va_recommended,
    v41_structural, notes, policy_options, runtime_options, cp_options, stack_options,
):
    runtime_meta: dict[str, Any] = {
        "sp": bool(policy_out["sp"]),
        "moe_dispatch_quantization": "routed_expert",
        "scenario_declared": bool(scenario_declared),
        **({"deployment_role": norm_role} if norm_role else {}),
        **({"raw_eager": True} if raw_eager else {}),
        "execution_policy": dict(policy_out["execution_policy"]),
    }
    if ascend_backend:
        runtime_meta.setdefault("arch_family", _arch_key(sku).upper())
    if policy_out["engram_offload"]:
        machine, host_rates = bind_sku_host_dma_rates(machine)
        runtime_meta["execution_policy"]["host_dma_sources"] = {
            "h2d": host_rates["h2d_source"],
            "host_memory": host_rates["host_memory_source"],
        }
    if policy_out["mso"]:
        runtime_meta["multistream_overlap_shared_expert"] = True
    if moe_comm_mode is not None:
        runtime_meta["moe_comm_mode"] = str(moe_comm_mode)
    if mtp_tokens > 0:
        runtime_meta["draft_mode"] = (
            "mtp" if getattr(model_spec.auxiliary, "mtp", None) is not None else "dspark"
        )
        runtime_meta["draft_proposal_tokens"] = mtp_tokens
    if runtime_options.get("draft_max_model_len") is not None:
        raw_dml = runtime_options["draft_max_model_len"]
        if isinstance(raw_dml, bool) or not isinstance(raw_dml, int) or raw_dml < 1:
            raise ModelingError(f"invalid_draft_max_model_len:{raw_dml!r}")
        runtime_meta["draft_max_model_len"] = int(raw_dml)
    if runtime_options.get("draft_enforce_eager") is not None:
        runtime_meta["draft_enforce_eager"] = bool(runtime_options["draft_enforce_eager"])
    if policy_out["enable_shared_expert_dp"]:
        runtime_meta["enable_shared_expert_dp"] = True

    _apply_cp_and_tutorial_flags(
        runtime_meta,
        model_spec=model_spec,
        parallel_dict={
            "dp": resolved_axes["dp"], "tp": resolved_axes["tp"], "pp": resolved_axes["pp"],
            "ep": resolved_axes["ep"], "cp": resolved_axes["dcp"], "dcp": resolved_axes["dcp"],
            "pcp": resolved_axes["pcp"],
            "q_proj_tp": int(runtime_options["q_proj_tp"]), "o_proj_tp": int(runtime_options["o_proj_tp"]),
            "linear_proj_tp": int(runtime_options["linear_proj_tp"]), "dcp_linear_tp": int(runtime_options["dcp_linear_tp"]),
            "mlp_tp": int(runtime_options["mlp_tp"]), "shared_expert_tp": int(runtime_options["shared_expert_tp"]),
            "lm_head_tp": int(runtime_options["lm_head_tp"]),
            "redundant_experts_per_rank": int(runtime_options["redundant_experts_per_rank"]),
        },
        enable_flashcomm1=policy_out["enable_flashcomm1"],
        enable_fused_mc2=policy_out["enable_fused_mc2"],
        enable_prefill_mc2=policy_out["enable_prefill_mc2"],
        **cp_options,
    )

    runtime_dict = _build_runtime_dict(
        axes=resolved_axes, machine=machine, model_spec=model_spec,
        policy_out=policy_out, runtime_meta=runtime_meta,
        resolved_max_seqs=resolved_max_seqs, resolved_max_batched=resolved_max_batched,
        resolved_max_model_len=resolved_max_model_len, resolved_block_size=resolved_block_size,
        mtp_tokens=mtp_tokens, mtp_acceptance_length=resolved_mtp_accept,
        **runtime_options,
    )
    caller_explicit = frozenset(
        name for name, explicit in (
            ("graph_capture_mode", graph_capture_mode is not None),
            ("sp", bool(policy_out["explicit_sp"])),
            ("enable_torch_compile", policy_options["enable_torch_compile"] is not None),
            ("enable_npugraph_ex", runtime_options["enable_npugraph_ex"] is not None),
            ("enable_static_kernel", runtime_options["enable_static_kernel"] is not None),
            ("enable_super_kernel", runtime_options["enable_super_kernel"] is not None),
            ("multistream_overlap_shared_expert", policy_options["multistream_overlap_shared_expert"] is not None),
            ("moe_comm_mode", moe_comm_mode is not None),
            ("enable_fused_mc2", policy_options["enable_fused_mc2"] is not None or (
                canonical_track in {"merged", "unmerged_pr"} and policy_out["enable_fused_mc2"] == 0
            )),
            ("enable_flashcomm1", policy_options["enable_flashcomm1"] is not None),
            ("recover_ssm", runtime_options["recover_ssm"] is True),
        ) if explicit
    )
    _apply_stack_and_lowerings(
        runtime_dict,
        model_spec=model_spec, model_dict=model_dict, machine=machine,
        sku=sku, hardware_backend=hardware_backend, va_profile=va_profile,
        va_recommended=va_recommended, raw_eager=raw_eager,
        track_raw_eager=track_raw_eager, canonical_track=canonical_track,
        caller_explicit=caller_explicit, v41_structural=v41_structural,
        axes=resolved_axes, notes=notes, **stack_options,
    )

    case_dict: dict[str, Any] = {
        "model": model_dict,
        "machine": _to_jsonable(machine),
        "runtime": runtime_dict,
        "workload": {"requests": requests},
        "slo": slo_dict,
    }
    parse_case(case_dict)
    return case_dict


_COMPOSE_OPTION_GROUPS: dict[str, tuple[str, ...]] = {
    "auto": (
        "quant", "layers", "ranks_per_node", "topology_preset", "va_recommended",
        "raw_eager", "feature_track", "runtime_stack", "optimizations",
        "exclude_optimizations", "engram_cpu_offload", "multistream_overlap_shared_expert",
        "enable_shared_expert_dp", "sp", "graph_capture_mode", "mtp_proposal_tokens",
        "mtp_acceptance_length", "kv_cache_dtype", "indexer_kv_dtype",
    ),
    "machine_options": (
        "num_devices", "ranks_per_node", "topology_preset",
        "inter_node_bw_gbps", "intra_node_bw_gbps", "intra_package_bw_gbps",
        "inter_node_bw_util", "intra_node_bw_util", "intra_package_bw_util",
        "inter_node_latency_us", "intra_node_latency_us", "intra_package_latency_us",
        "superpod_size", "superpod_bw_gbps", "inter_pod_bw_gbps",
        "intra_bw_4p_gbps", "intra_bw_8p_gbps",
    ),
    "model_options": (
        "quant", "kv_cache_dtype", "indexer_kv_dtype", "layers",
        "n_routed_experts", "num_experts_per_tok", "promote_dense_to_moe",
    ),
    "mtp_options": ("mtp_proposal_tokens", "mtp_acceptance_length"),
    "scenario_options": (
        "prompt_tokens", "output_tokens", "concurrency", "prefix_tokens",
        "arrival_gap_ms", "ttft_slo_ms", "tpot_slo_ms", "require_scenario",
        "max_model_len", "max_num_seqs", "max_num_batched_tokens", "prefix_cache_key",
    ),
    "policy_options": (
        "sp", "enable_torch_compile", "multistream_overlap_shared_expert",
        "engram_cpu_offload", "enable_shared_expert_dp", "enable_flashcomm1",
        "enable_fused_mc2", "enable_prefill_mc2",
    ),
    "runtime_options": (
        "q_proj_tp", "o_proj_tp", "linear_proj_tp", "dcp_linear_tp",
        "mlp_tp", "shared_expert_tp", "lm_head_tp", "redundant_experts_per_rank",
        "collective_startup_mode", "matrix_compute_policy", "recover_ssm",
        "prefix_cache_mode", "prefix_read_mode", "homogeneous_graph_reuse",
        "enable_chunked_prefill", "async_scheduling", "enable_npugraph_ex",
        "enable_static_kernel", "enable_super_kernel", "enable_dbo",
        "pp_layer_partition", "draft_max_model_len", "draft_enforce_eager",
    ),
    "cp_options": (
        "enable_pcp_o_proj_weight_sharding", "enable_sfa_dcp_sharded_indexer",
        "dcp_q_replicate",
    ),
    "stack_options": (
        "runtime_stack", "optimizations", "exclude_optimizations", "_stack_phase",
    ),
}


def _select_group(scope: Mapping[str, Any], group: str) -> dict[str, Any]:
    return {key: scope[key] for key in _COMPOSE_OPTION_GROUPS[group]}


def compose_case(
    model: str | Path | Mapping[str, Any],
    hardware: str | Mapping[str, Any] = "a3",
    tp: int | None = None,
    dp: int | None = None,
    ep: int | None = None,
    pp: int | None = None,
    pcp: int | None = None,
    dcp: int | None = None,
    sp: bool | None = None,
    num_devices: int | None = None,
    ranks_per_node: int | None = None,
    quant: str | Mapping[str, Any] | None = None,
    layers: int | None = None,
    n_routed_experts: int | None = None,
    num_experts_per_tok: int | None = None,
    promote_dense_to_moe: bool = False,
    allow_synthetic_gqa_draft: bool = False,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    concurrency: int | None = None,
    prefix_tokens: int = DEFAULT_PREFIX_TOKENS,
    arrival_gap_ms: float = DEFAULT_ARRIVAL_GAP_MS,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    block_size: int | None = None,
    enable_chunked_prefill: bool = True,
    async_scheduling: bool = False,
    graph_capture_mode: str | None = None,
    mtp_proposal_tokens: int | None = None,
    mtp_acceptance_length: float | None = None,
    draft_max_model_len: int | None = None,
    draft_enforce_eager: bool | None = None,
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
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
    enable_shared_expert_dp: bool | None = None,
    *,
    enable_torch_compile: bool | None = None,
    enable_npugraph_ex: bool | None = None,
    enable_static_kernel: bool | None = None,
    enable_super_kernel: bool | None = None,
    multistream_overlap_shared_expert: bool | None = None,
    engram_cpu_offload: bool | None = None,
    enable_dbo: bool = False,
    moe_comm_mode: str | None = None,
    q_proj_tp: int = DEFAULT_LOCAL_WIDTH,
    o_proj_tp: int = DEFAULT_LOCAL_WIDTH,
    linear_proj_tp: int = DEFAULT_LOCAL_WIDTH,
    dcp_linear_tp: int = DEFAULT_LOCAL_WIDTH,
    mlp_tp: int = DEFAULT_LOCAL_WIDTH,
    shared_expert_tp: int = DEFAULT_LOCAL_WIDTH,
    lm_head_tp: int = DEFAULT_LOCAL_WIDTH,
    enable_flashcomm1: bool | None = None,
    enable_fused_mc2: int | None = None,
    enable_prefill_mc2: bool | None = None,
    enable_pcp_o_proj_weight_sharding: bool = False,
    enable_sfa_dcp_sharded_indexer: bool = False,
    dcp_q_replicate: bool = False,
    kv_cache_dtype: str | None = None,
    indexer_kv_dtype: str | None = None,
    redundant_experts_per_rank: int = DEFAULT_REDUNDANT_EXPERTS,
    collective_startup_mode: str | None = None,
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
    runtime_stack: str | Path | Mapping[str, Any] = "vllm-ascend",
    feature_track: str = "merged",
    optimizations: Sequence[str] = (),
    exclude_optimizations: Sequence[str] = (),
    parallel: str | Mapping[str, Any] | None = None,
    deployment: str | None = None,
    role: str | None = None,
    require_scenario: bool = False,
    _stack_phase: str | None = None,
) -> dict[str, Any]:
    """Synthesize and validate a complete simulation Case dictionary."""
    args = locals()
    norm_role = str(role).strip().lower() if role is not None else None
    if norm_role is not None and norm_role not in _VALID_DEPLOYMENT_ROLES:
        raise ModelingError(f"invalid_deployment_role:{role}")
    axes, auto_mode = _resolve_parallel_axes(
        tp=tp, dp=dp, ep=ep, pp=pp, pcp=pcp, dcp=dcp,
        parallel=parallel, deployment=deployment, num_devices=num_devices,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
    )
    if auto_mode:
        return _compose_auto_deployment(
            model, hardware, axes, num_devices=num_devices, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms,
            deployment=deployment, **_select_group(args, "auto"),
        )
    grouped = {
        key: _select_group(args, key)
        for key in (
            "machine_options", "model_options", "mtp_options", "scenario_options",
            "policy_options", "runtime_options", "cp_options", "stack_options",
        )
    }
    case_dict = _compose_explicit_case(
        model, hardware, axes, norm_role, graph_capture_mode=graph_capture_mode,
        raw_eager=raw_eager, feature_track=feature_track, va_recommended=va_recommended,
        block_size=block_size, allow_synthetic_gqa_draft=allow_synthetic_gqa_draft,
        moe_comm_mode=moe_comm_mode, **grouped,
    )
    return _apply_hardware_derivation(
        case_dict,
        hardware_derivation=hardware_derivation,
        aic_count=aic_count, aiv_count=aiv_count, memory_bytes=memory_bytes,
        hbm_peak_bytes_per_s=hbm_peak_bytes_per_s, hbm_efficiency=hbm_efficiency,
        mte2_ceiling_bytes_per_s=mte2_ceiling_bytes_per_s, mte2_efficiency=mte2_efficiency,
    )


__all__ = [
    "HARDWARE_ALIASES",
    "HARDWARE_TOPOLOGY_ALIASES",
    "VA_RECOMMENDED_PROFILES",
    "apply_va_recommended",
    "assemble",
    "capture_keys",
    "compose_case",
    "prepare_speculation",
    "resolve_hardware_sku",
    "va_recommended_config",
]
