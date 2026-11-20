"""Neutral custom hardware archetypes and input composition.

[Role]
  Expose copyable templates and merge explicit overrides with their defaults.
[Invariants]
  Templates hold structural defaults. Explicit inputs keep their precedence
  and the caller's mappings are not mutated.
[Timeouts & Locks]
  In-memory template construction and copying. No I/O or locks.
[Side-Effect Policy]
  Returns fresh mappings for the builder and public template export.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from mindie_sim.schema.common import ModelingError


_DEFAULT_ARCHETYPE = "ascend_dual_die_ub"
_TEMPLATE_SKU = "CUSTOM_SKU_TEMPLATE"


HARDWARE_ARCHETYPE_TEMPLATES: dict[str, dict[str, Any]] = {
    "ascend_dual_die_ub": {
        "arch_family": "Custom_Ascend_UB", "dies_per_package": 2, "ranks_per_node": 64,
        "superpod_size": 8192, "hbm_gb": 144.0, "hbm_bw_gbps": 4000.0,
        "cube_fp16_tflops": 600.0, "cube_fp8_tflops": 1200.0, "vector_fp32_tflops": 28.0,
        "aic_count": 32, "aiv_count": 64, "l2_mb": 192.0, "ub_kb": 256.0,
        "l1_kb": 1024.0, "l0a_kb": 64.0, "l0b_kb": 64.0, "l0c_kb": 256.0,
        "intra_package_bw_gbps": 800.0, "intra_node_bw_gbps": 800.0,
        "superpod_bw_gbps": 800.0, "inter_node_bw_gbps": 800.0,
        "inter_pod_bw_gbps": 100.0, "intra_node_bw_util": 0.85,
        "inter_node_bw_util": 0.80, "intra_package_latency_us": 0.9,
        "intra_node_latency_us": 1.5, "inter_node_latency_us": 4.5,
        "inter_pod_latency_us": 10.0, "launch_us": 4.0,
    },
    "ascend_dual_die_server": {
        "arch_family": "Custom_Ascend_Server", "dies_per_package": 2, "ranks_per_node": 16,
        "hbm_gb": 64.0, "hbm_bw_gbps": 1600.0, "cube_fp16_tflops": 376.0,
        "cube_fp8_tflops": 752.0, "vector_fp32_tflops": 23.5, "aic_count": 24,
        "aiv_count": 48, "l2_mb": 192.0, "ub_kb": 192.0, "l1_kb": 512.0,
        "l0a_kb": 64.0, "l0b_kb": 64.0, "l0c_kb": 128.0,
        "intra_package_bw_gbps": 392.0, "intra_bw_4p_gbps": 392.0,
        "intra_bw_8p_gbps": 392.0, "intra_node_bw_gbps": 392.0,
        "inter_node_bw_gbps": 50.0, "intra_node_bw_util": 0.80,
        "inter_node_bw_util": 0.80, "intra_package_latency_us": 1.0,
        "intra_node_latency_us": 1.5, "inter_node_latency_us": 5.0, "launch_us": 4.0,
    },
    "ascend_single_die": {
        "arch_family": "Custom_Ascend_SingleDie", "dies_per_package": 1,
        "ranks_per_node": 8, "hbm_gb": 64.0, "hbm_bw_gbps": 1600.0,
        "cube_fp16_tflops": 313.0, "cube_fp8_tflops": 626.0,
        "vector_fp32_tflops": 19.5, "aic_count": 24, "aiv_count": 48,
        "l2_mb": 192.0, "ub_kb": 192.0, "l1_kb": 512.0, "l0a_kb": 64.0,
        "l0b_kb": 64.0, "l0c_kb": 128.0, "intra_node_bw_gbps": 196.0,
        "inter_node_bw_gbps": 25.0, "intra_node_bw_util": 0.70,
        "inter_node_bw_util": 0.80, "intra_node_latency_us": 2.0,
        "inter_node_latency_us": 10.0, "launch_us": 4.5,
    },
    "gpu_nvlink_server": {
        "arch_family": "Custom_GPU", "dies_per_package": 1, "ranks_per_node": 8,
        "hbm_gb": 141.0, "hbm_bw_gbps": 4800.0, "cube_fp16_tflops": 989.0,
        "cube_fp8_tflops": 1978.0, "vector_fp32_tflops": 67.0, "aic_count": 132,
        "aiv_count": 132, "l2_mb": 60.0, "ub_kb": 228.0, "l1_kb": 256.0,
        "l0a_kb": 64.0, "l0b_kb": 64.0, "l0c_kb": 128.0,
        "intra_node_bw_gbps": 450.0, "inter_node_bw_gbps": 50.0,
        "intra_node_bw_util": 0.85, "inter_node_bw_util": 0.80,
        "intra_node_latency_us": 1.5, "inter_node_latency_us": 5.0, "launch_us": 3.5,
    },
    "generic_accelerator": {
        "arch_family": "Custom_Accelerator", "dies_per_package": 1,
        "ranks_per_node": 8, "hbm_gb": 80.0, "hbm_bw_gbps": 2000.0,
        "cube_fp16_tflops": 500.0, "cube_fp8_tflops": 1000.0,
        "vector_fp32_tflops": 32.0, "aic_count": 32, "aiv_count": 64,
        "l2_mb": 64.0, "ub_kb": 192.0, "l1_kb": 512.0, "l0a_kb": 64.0,
        "l0b_kb": 64.0, "l0c_kb": 128.0, "intra_node_bw_gbps": 300.0,
        "inter_node_bw_gbps": 50.0, "intra_node_bw_util": 0.80,
        "inter_node_bw_util": 0.80, "intra_node_latency_us": 2.0,
        "inter_node_latency_us": 5.0, "launch_us": 4.0,
    },
}


_FIELD_DOCS: dict[str, str] = {
    "sku": "Stable custom identifier; arbitrary private text, no vendor name required.",
    "archetype": "Structural template key from HARDWARE_ARCHETYPE_TEMPLATES.",
    "arch_family": "Neutral architecture label used only as provenance.",
    "dies_per_package": "Integer dies (simulated devices) in one physical package.",
    "ranks_per_node": "Integer dies (ranks) sharing one node fabric.",
    "superpod_size": "Maximum dies in one declared superpod fabric.",
    "max_dies": "Alias for superpod_size when declaring the largest fabric group.",
    "hbm_gb": "HBM capacity per die in decimal GB (10^9 bytes).",
    "hbm_bw_gbps": "Aggregate HBM bandwidth per die in decimal GB/s (10^9 bytes/s).",
    "cube_fp16_tflops": "FP16/BF16 matrix throughput in TFLOPS (10^12 FLOP/s).",
    "cube_fp8_tflops": "FP8 matrix throughput in TFLOPS; omitted means twice cube_fp16_tflops.",
    "cube_int4_tflops": "Optional INT4 matrix throughput in TFLOPS; sets the int4 peak.",
    "fp4_tflops": "Alias for cube_int4_tflops for agents that label the format FP4.",
    "vector_fp32_tflops": "Declared FP32 vector throughput in 10^12 elementary ops/s.",
    "aic_count": "Matrix compute units per die (cube scheduling owners).",
    "aiv_count": "Vector compute units per die (vector scheduling owners).",
    "l2_mb": "Shared L2 capacity per die in MiB (2^20 bytes).",
    "ub_kb": "Unified-buffer capacity per vector unit in KiB (2^10 bytes).",
    "l1_kb": "L1 capacity per matrix unit in KiB.",
    "l0a_kb": "L0A operand-buffer capacity per matrix unit in KiB.",
    "l0b_kb": "L0B operand-buffer capacity per matrix unit in KiB.",
    "l0c_kb": "L0C accumulator capacity per matrix unit in KiB.",
    "mte1_bw_gbps": "Optional L1-to-L0 bandwidth in GB/s; omitted scales from HBM.",
    "mte2_bw_gbps": "Optional read bandwidth in GB/s; sets both AIC and AIV supplies.",
    "mte3_bw_gbps": "Optional write bandwidth in GB/s; omitted scales from HBM.",
    "fixpipe_bw_gbps": "Optional accumulator write-out bandwidth in GB/s.",
    "intra_package_bw_gbps": "Peak die-to-die package bandwidth in decimal GB/s.",
    "intra_bw_4p_gbps": "Four-package group bandwidth in decimal GB/s.",
    "intra_bw_8p_gbps": "Eight-package group bandwidth in decimal GB/s.",
    "intra_node_bw_gbps": "Peak node-fabric bandwidth in GB/s before intra_node_bw_util.",
    "superpod_bw_gbps": "Superpod-fabric bandwidth in GB/s before inter_node_bw_util.",
    "inter_node_bw_gbps": "Peak inter-node bandwidth in GB/s before inter_node_bw_util.",
    "inter_pod_bw_gbps": "Peak inter-pod bandwidth in GB/s before inter_node_bw_util.",
    "intra_node_bw_util": "Usable fraction in (0, 1] of node-fabric bandwidth.",
    "inter_node_bw_util": "Usable fraction in (0, 1] of inter-node and pod fabric.",
    "intra_package_latency_us": "Per-hop package-fabric latency in microseconds.",
    "intra_node_latency_us": "Per-hop node-fabric latency in microseconds.",
    "inter_node_latency_us": "Per-hop inter-node fabric latency in microseconds.",
    "inter_pod_latency_us": "Per-hop inter-pod fabric latency in microseconds.",
    "launch_us": "Host submission service per ordinary kernel in microseconds.",
    "capability": (
        "Optional representation/native-compute/cast/path declaration, or a "
        "mapping with an architecture preset. Omitted means the builder derives "
        "only the native modes implied by the synthetic peaks; no data path is "
        "invented."
    ),
}


_AGENT_USAGE_HINT = (
    "Copy this template, change 'sku' and only the key specs you know, remove keys "
    "beginning with '_', then call register_custom_hardware_sku(sku, spec) or pass "
    "the dictionary directly to compose_custom_case(model, spec). An untracked "
    "private file can be loaded by setting MINDIE_SIM_CUSTOM_SKUS to its JSON path."
)


def _plain_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelingError(f"custom_hardware_mapping_required:{context}")
    return {key: item for key, item in value.items()
            if not str(key).startswith("_")}


def _resolve_archetype(
    spec: Mapping[str, Any], archetype: str | None,
    *, prefer_spec_archetype: bool = False,
) -> str:
    declared = spec.get("archetype")
    if archetype is None:
        chosen = declared or _DEFAULT_ARCHETYPE
    elif (prefer_spec_archetype and archetype == _DEFAULT_ARCHETYPE
          and declared in HARDWARE_ARCHETYPE_TEMPLATES):
        chosen = declared
    else:
        chosen = archetype
    if chosen not in HARDWARE_ARCHETYPE_TEMPLATES:
        raise ModelingError(f"unknown_hardware_archetype:{chosen}")
    return str(chosen)


def _merge_inputs(
    spec: Mapping[str, Any], archetype: str | None, overrides: Mapping[str, Any],
    *, prefer_spec_archetype: bool = False,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not isinstance(spec, Mapping):
        raise ModelingError("custom_hardware_spec_must_be_mapping")
    explicit = _plain_mapping(spec, "spec")
    explicit.update(_plain_mapping(overrides, "overrides"))
    resolved = _resolve_archetype(
        spec, archetype, prefer_spec_archetype=prefer_spec_archetype,
    )
    merged = deepcopy(HARDWARE_ARCHETYPE_TEMPLATES[resolved])
    merged.update(explicit)
    merged["archetype"] = resolved
    return resolved, merged, explicit


def export_hardware_sku_template(
    archetype: str = "ascend_dual_die_ub", *, include_schema_docs: bool = True,
) -> dict[str, Any]:
    """Return a fresh, agent-editable template for one structural archetype."""
    if archetype not in HARDWARE_ARCHETYPE_TEMPLATES:
        raise ModelingError(f"unknown_hardware_archetype:{archetype}")
    template: dict[str, Any] = {
        "sku": _TEMPLATE_SKU,
        "archetype": archetype,
        **deepcopy(HARDWARE_ARCHETYPE_TEMPLATES[archetype]),
    }
    if include_schema_docs:
        template["_field_docs"] = dict(_FIELD_DOCS)
        template["_agent_usage_hint"] = _AGENT_USAGE_HINT
    return template
