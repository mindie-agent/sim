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
from mindie_sim.compose_assembly import (
    _assembly_emit_counts,
    _assembly_metric_rows,
    _assembly_request_specs,
    assemble,
)
from mindie_sim.compose_inputs import (
    DERIVED_ASCEND_SKUS,
    HARDWARE_ALIASES,
    HARDWARE_TOPOLOGY_ALIASES,
    _EXTRA_QUANT_SPECS,
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
from mindie_sim.compose_runtime import (
    PIECEWISE_CAPTURE_BASE_KEYS,
    _execution_policy,
    _resolve_va_lowerings,
    _supports_sequence_parallel,
    apply_va_recommended,
    capture_keys,
    finalize_compilation_spine,
    snapshot_kernel_keys,
    stamp_stack_profile_origins,
    va_recommended_config,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.va_profiles import VA_RECOMMENDED_PROFILES

NOMINAL_HOST_MEM_BPS = 64_000_000_000.0
NOMINAL_HOST_VEC_OPS = 50_000_000_000.0
NOMINAL_H2D_BPS = 32_000_000_000.0
BYTES_PER_GB = 1_000_000_000.0
A3_DEFAULT_RANKS_PER_NODE = 16
DEFAULT_RANKS_PER_NODE = 8
DEFAULT_MAX_MODEL_LEN = 32768
MIN_DEFAULT_MAX_SEQS = 16
MIN_DEFAULT_MAX_BATCHED_TOKENS = 8192
DEFAULT_PARALLEL = 1
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
    promote_dense_to_moe: bool = False,
    allow_synthetic_gqa_draft: bool = False,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    concurrency: int = DEFAULT_CONCURRENCY,
    prefix_tokens: int = DEFAULT_PREFIX_TOKENS,
    arrival_gap_ms: float = DEFAULT_ARRIVAL_GAP_MS,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    block_size: int | None = None,
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
    enable_torch_compile: bool | None = None,
    enable_npugraph_ex: bool | None = None,
    enable_static_kernel: bool | None = None,
    enable_super_kernel: bool | None = None,
    multistream_overlap_shared_expert: bool | None = None,
    engram_cpu_offload: bool | None = None,
    enable_dbo: bool = False,
    moe_comm_mode: str | None = None,
    o_proj_tp: int = DEFAULT_LOCAL_WIDTH,
    linear_proj_tp: int = DEFAULT_LOCAL_WIDTH,
    lm_head_tp: int = DEFAULT_LOCAL_WIDTH,
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
    _stack_phase: str | None = None,
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

    sku, hardware_backend, machine = _materialize_machine(
        hardware, required_devices=dp * tp * pp * pcp, num_devices=num_devices,
        ranks_per_node=ranks_per_node, topology_preset=topology_preset,
        inter_node_bw_gbps=inter_node_bw_gbps, intra_node_bw_gbps=intra_node_bw_gbps,
        intra_package_bw_gbps=intra_package_bw_gbps, inter_node_bw_util=inter_node_bw_util,
        intra_node_bw_util=intra_node_bw_util, intra_package_bw_util=intra_package_bw_util,
        inter_node_latency_us=inter_node_latency_us, intra_node_latency_us=intra_node_latency_us,
        intra_package_latency_us=intra_package_latency_us, superpod_size=superpod_size,
        superpod_bw_gbps=superpod_bw_gbps, inter_pod_bw_gbps=inter_pod_bw_gbps,
        intra_bw_4p_gbps=intra_bw_4p_gbps, intra_bw_8p_gbps=intra_bw_8p_gbps,
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

    explicit_graph_capture = graph_capture_mode is not None
    explicit_sp = sp is not None
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
        va_profile.setdefault("graph_capture_mode", None)
        if quant is None:
            quant = va_profile["quant"]
    model_spec = _apply_quant_override(model_spec, quant, machine)
    from mindie_sim.compose_operator_defaults import structural_v41_latent_mqa
    v41_structural = structural_v41_latent_mqa(model_spec.layer_plans)
    model_spec = _apply_structural_overrides(
        model_spec, layers=layers, n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok, promote_dense_to_moe=promote_dense_to_moe,
    )
    model_spec = prepare_speculation(
        model_spec, int(mtp_proposal_tokens), allow_synthetic_gqa=allow_synthetic_gqa_draft,
    )
    if block_size is not None:
        resolved_block_size = int(block_size)
    else:
        resolved_block_size = int((va_profile or {}).get("kv_block_size") or DEFAULT_BLOCK_SIZE)
        if dict(model_spec.metadata).get("cache_allocator") == "paired_sfa":
            from mindie_sim.catalog.paired_sfa import minimum_block_size
            resolved_block_size = max(
                resolved_block_size,
                minimum_block_size(model_spec, tp=tp, proposals=int(mtp_proposal_tokens)))

    notes: list[dict[str, Any]] = []
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
                                                  tp=tp, ep=ep, pp=pp, pcp=pcp, notes=notes))
    documented_capture = (va_profile or {}).get("graph_capture_mode")
    capture_source = (
        "raw_eager" if raw_eager else
        "explicit" if graph_capture_mode is not None else
        "profile" if (production and documented_capture in {"full", "piecewise", "none"}) else
        "none")
    if raw_eager:
        graph_capture_mode = "none"
    elif graph_capture_mode is None:
        graph_capture_mode = (str(documented_capture)
                              if documented_capture in {"full", "piecewise", "none"} else "none")

    supports_compile = bool(dict(model_spec.catalog_facts).get("supports_torch_compile", True))
    initial_compile = (
        False if raw_eager else
        bool(enable_torch_compile) if enable_torch_compile is not None else
        supports_compile
    )
    execution_policy = {
        "backend": hardware_backend, "production": bool(production),
        "va_recommended": va_recommended, "raw_eager": bool(raw_eager),
        "automatic_optional_fusions": bool(not raw_eager and va_recommended is not False),
        "sp": bool(sp), "sp_source": sp_source,
        "graph_capture_mode": str(graph_capture_mode), "capture_source": capture_source,
        "enable_torch_compile": initial_compile,
        "profile": str((va_profile or {}).get("model", "")) if va_profile else "",
        "priority": "raw_eager > explicit caller > documented profile > structural fusion defaults",
    }

    model_spec = ensure_native_mla_draft(model_spec, int(mtp_proposal_tokens))
    if int(mtp_proposal_tokens) > 0:
        aux = model_spec.auxiliary
        if getattr(aux, "mtp", None) is None and getattr(aux, "dspark", None) is None:
            raise ModelingError("draft_structure_required_for_mtp_proposal_tokens")
        if not (1.0 <= float(mtp_acceptance_length) <= float(1 + int(mtp_proposal_tokens))):
            raise ModelingError("invalid_mtp_acceptance_length")

    required_len = int(prompt_tokens) + int(output_tokens) + max(0, int(mtp_proposal_tokens))
    if model_limit > 0 and required_len > model_limit:
        raise ModelingError(
            f"workload_exceeds_model_max_position_embeddings:{required_len}>{model_limit}")
    if max_model_len is not None:
        if int(max_model_len) < required_len:
            raise ModelingError(f"max_model_len_smaller_than_workload:{max_model_len}<{required_len}")
        resolved_max_model_len = int(max_model_len)
    else:
        resolved_max_model_len = max(DEFAULT_MAX_MODEL_LEN, required_len)

    model_spec = _tier4_router_spec(model_spec, sku, raw_eager=raw_eager)
    if ascend_backend and _arch_key(sku) == "a5":
        from mindie_sim.compose_operator_defaults import resolve_packed_indexed_mla_cache
        model_spec = resolve_packed_indexed_mla_cache(model_spec)
    model_dict = _to_jsonable(model_spec)
    model_dict["n_routed_experts"] = int(model_spec.num_experts)
    model_dict["num_experts_per_tok"] = int(model_spec.top_k)

    resolved_max_seqs = (
        int(max_num_seqs) if max_num_seqs is not None else max(int(concurrency), MIN_DEFAULT_MAX_SEQS)
    )
    resolved_max_batched = (
        int(max_num_batched_tokens) if max_num_batched_tokens is not None
        else max(int(prompt_tokens) * int(concurrency), MIN_DEFAULT_MAX_BATCHED_TOKENS)
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
    if ascend_backend:
        runtime_meta.setdefault("arch_family", _arch_key(sku).upper())
    if resolved_engram_offload:
        h2d_ch = dict((machine.get("transfer_channels") or {}).get("h2d") or {})
        h2d_bps = float(h2d_ch.get("effective_uni_bw_gbps") or 0.0) * BYTES_PER_GB
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
    if int(mtp_proposal_tokens) > 0:
        runtime_meta["draft_mode"] = (
            "mtp" if getattr(model_spec.auxiliary, "mtp", None) is not None else "dspark"
        )
        runtime_meta["draft_proposal_tokens"] = int(mtp_proposal_tokens)
    if enable_shared_expert_dp:
        runtime_meta["enable_shared_expert_dp"] = True

    runtime_dict: dict[str, Any] = {
        "parallel": {
            "dp": dp, "tp": tp, "pp": pp, "ep": ep, "cp": dcp, "dcp": dcp, "pcp": pcp,
            **({"rank_table": [
                stage_ranks[offset:offset + int(ep)]
                for stage in range(int(pp))
                for stage_ranks in ([[
                    int(machine["dies"][idx]["rank"]) if machine.get("dies") else idx
                    for d in range(int(dp))
                    for c in range(int(pcp))
                    for t in range(int(tp))
                    for idx in (((d * int(pp) + stage) * int(pcp) + c) * int(tp) + t,)
                ]][0],)
                for offset in range(0, len(stage_ranks), int(ep))
            ]}
               if int(pcp) > 1 and int(ep) > int(dp) * int(tp)
               and (int(dp) * int(tp) * int(pcp)) % int(ep) == 0 else {}),
            "o_proj_tp": int(o_proj_tp),
            "linear_proj_tp": int(linear_proj_tp),
            "lm_head_tp": int(lm_head_tp),
            "redundant_experts_per_rank": int(redundant_experts_per_rank),
        },
        "collective_startup_mode": (
            str(collective_startup_mode) if collective_startup_mode is not None
            else "topology" if machine.get("collective_protocols") else "legacy"),
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
        "graph_mode": str(graph_capture_mode),
        "graph_capture_mode": str(graph_capture_mode),
        "enable_torch_compile": initial_compile,
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
    if resolved_mso:
        runtime_dict["multistream_overlap_shared_expert"] = True
    if resolved_engram_offload:
        runtime_dict["engram_cpu_offload"] = True
    if enable_dbo:
        runtime_dict["enable_dbo"] = True
    if graph_capture_mode == "full":
        runtime_dict["graph_capture_keys"] = capture_keys(resolved_max_seqs, mtp_proposal_tokens)
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
    if sp:
        from mindie_sim.prediction.sequence_parallel import DECLARATION, _validate_model
        if pcp < 1 or pp < 1 or tp <= 1 or ep < tp or (ep % tp != 0):
            raise ModelingError(f"sequence_parallel_invalid_topology:tp={tp},ep={ep},pp={pp},pcp={pcp}")
        _validate_model(model_spec, padding="right_to_tp")
        runtime_dict["kernels"] = {"sequence_parallel": dict(DECLARATION, padding="right_to_tp")}

    prior_kernels = snapshot_kernel_keys(runtime_dict.get("kernels"))
    excluded_selectors: frozenset[str] = frozenset()
    excluded_passes: frozenset[str] = frozenset()
    if ascend_backend and not raw_eager:
        from mindie_sim.stack_tracks import excluded_lowering_bindings
        excluded_selectors, excluded_passes = excluded_lowering_bindings(exclude_optimizations)
    if va_profile is not None:
        runtime_meta["va_recommended"] = True
        runtime_meta["va_profile"] = deepcopy(va_profile)
        compilation = va_profile.get("compilation_config")
        if (isinstance(compilation, dict)
                and str(compilation.get("cudagraph_mode") or "").upper() in {"FULL", "FULL_DECODE_ONLY"}
                and str(graph_capture_mode) == "full"):
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
            caller_explicit = frozenset(
                name for name, explicit in (
                    ("graph_capture_mode", explicit_graph_capture),
                    ("sp", explicit_sp),
                    ("enable_torch_compile", enable_torch_compile is not None),
                    ("enable_npugraph_ex", enable_npugraph_ex is not None),
                    ("enable_static_kernel", enable_static_kernel is not None),
                    ("enable_super_kernel", enable_super_kernel is not None),
                    ("multistream_overlap_shared_expert", multistream_overlap_shared_expert is not None),
                    ("moe_comm_mode", moe_comm_mode is not None),
                    ("recover_ssm", recover_ssm is True),
                ) if explicit)
            ctx_opts: dict[str, Any] = {
                "ranks_per_node": int(machine.get("ranks_per_node", 0) or 0),
            }
            if _stack_phase:
                ctx_opts["phase"] = _stack_phase
            mechanism_ctx = MechanismContext(
                model=model_spec, sku=sku, arch=_arch_key(sku),
                parallel=dict(runtime_dict["parallel"]),
                caller_explicit=caller_explicit, options=ctx_opts)
            resolved_track = resolve_stack_track(
                stack_profile, track=canonical_track, model_key=stack_model_key,
                ctx=mechanism_ctx, optimizations=optimizations,
                exclude_optimizations=exclude_optimizations)
            if not raw_eager or optimizations:
                apply_resolved_mechanisms(runtime_dict, resolved_track, mechanism_ctx)
            runtime_meta = dict(runtime_dict.get("metadata") or {})
            runtime_meta["feature_track_report"] = resolved_track
            runtime_dict["metadata"] = runtime_meta

    if ascend_backend and not raw_eager:
        from mindie_sim.compose_operator_defaults import (
            DECLARATION_PHASE, apply_compose_v41_a5_defaults,
        )
        from mindie_sim.prediction.implementation_selection import operator_selection_summary
        runtime_meta = dict(runtime_dict.get("metadata") or {})
        kernels = dict(runtime_dict.get("kernels") or {})
        lowerings = dict(kernels.get("lowerings") or {})
        notes.extend(apply_compose_v41_a5_defaults(
            model_dict=model_dict, lowerings=lowerings, backend=hardware_backend,
            arch=_arch_key(sku), layer_plans=model_spec.layer_plans,
            activation_dtype=model_spec.activation_dtype, tp=tp, cp=dcp,
            runtime_metadata=runtime_meta, structural_v41=v41_structural,
        ))
        if excluded_selectors:
            lowerings = {k: v for k, v in lowerings.items() if k not in excluded_selectors}
        if lowerings:
            kernels["lowerings"] = lowerings
            runtime_dict["kernels"] = kernels
        runtime_meta["operator_selection_summary"] = operator_selection_summary(
            arch_family=_arch_key(sku), phase=DECLARATION_PHASE, notes=notes)
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
        if lowerings:
            kernels["lowerings"] = lowerings
        if kernels:
            runtime_dict["kernels"] = kernels

    finalize_compilation_spine(runtime_dict, raw_eager=raw_eager)
    stamp_stack_profile_origins(runtime_dict, prior_keys=prior_kernels)

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
