"""Multi-tier outbound-ratio and A5 EP MC2 analytical models.

[Role]
  Convert declared Ethernet/UB link tiers into an EP all-to-all effective
  bandwidth, and evaluate speculative decoding plus MC2 dispatch/combine
  communication time from structural geometry.

[Invariants]
  * An all-to-all source rank sends ``(ep - 1) / ep`` of its traffic to peers.
    Crossing a declared tier boundary retains only the fraction that leaves
    that domain, so the bottleneck is the minimum ``B / outbound_ratio`` over
    active tiers.
  * Precision sizes are explicit: MXFP8 block scaling is 33/32 bytes per value,
    FP8/INT8 are one byte and BF16/FP16 are two bytes.
  * Timings use topological bytes, declared layer counts and explicit startup
    latency. No serving floor, fitted residual or measurement-presence check is
    applied.

[Timeouts & Locks] Pure bounded arithmetic over the requested EP values.
[Side-Effect Policy] No I/O, device context or mutable global state.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from mindie_sim.schema.common import ModelingError


EP_TIER_PRESETS: dict[str, tuple[dict[str, Any], ...]] = {
    "a5_pcie_16p": (
        {"name": "intra_4p", "max_ranks": 4, "bw_gbps": 288.0},
        {"name": "intra_8p", "max_ranks": 8, "bw_gbps": 128.0},
        {"name": "intra_16p", "max_ranks": 16, "bw_gbps": 116.0},
        {"name": "inter_node", "max_ranks": 8192, "bw_gbps": 50.0},
    ),
    "a5_ub_pod": (
        {"name": "superpod", "max_ranks": 8192, "bw_gbps": 800.0},
    ),
    "a3_superpod": (
        {"name": "intra_node", "max_ranks": 16, "bw_gbps": 392.0},
        {"name": "superpod", "max_ranks": 384, "bw_gbps": 392.0},
        {"name": "inter_pod", "max_ranks": 8192, "bw_gbps": 50.0},
    ),
}

_BYTES_PER_ELEMENT = {
    "mxfp8": 33.0 / 32.0,
    "mx_fp8": 33.0 / 32.0,
    "mxfp8_e4m3": 33.0 / 32.0,
    "fp8": 1.0,
    "int8": 1.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
}


def _dtype_bytes(dtype: Any, label: str) -> float:
    key = str(dtype).strip().lower().replace("-", "_")
    if key not in _BYTES_PER_ELEMENT:
        raise ModelingError(f"ep_mc2_unsupported_{label}_dtype:{dtype}")
    return _BYTES_PER_ELEMENT[key]


def _non_negative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ModelingError(f"{label}_invalid:{value}")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ModelingError(f"{label}_invalid:{value}") from None
    if not math.isfinite(result) or result < 0:
        raise ModelingError(f"{label}_invalid:{value}")
    return result


def _positive(value: Any, label: str) -> float:
    result = _non_negative(value, label)
    if result <= 0:
        raise ModelingError(f"{label}_invalid:{value}")
    return result


def _tiers_for(preset: str, tiers: Sequence[Mapping[str, Any]] | None
               ) -> tuple[dict[str, Any], ...]:
    if tiers is None:
        if preset not in EP_TIER_PRESETS:
            raise ModelingError(f"ep_tier_preset_unknown:{preset}")
        return EP_TIER_PRESETS[preset]
    if isinstance(tiers, (str, bytes)) or not isinstance(tiers, Sequence) or not tiers:
        raise ModelingError("ep_tier_sequence_invalid")
    normalized = []
    previous = 0
    for index, tier in enumerate(tiers):
        if not isinstance(tier, Mapping):
            raise ModelingError(f"ep_tier_record_invalid:{index}")
        unknown = set(tier) - {"name", "max_ranks", "bw_gbps", "latency_us"}
        if unknown:
            raise ModelingError(f"ep_tier_field_unknown:{sorted(unknown)[0]}")
        max_ranks = tier.get("max_ranks")
        if (not isinstance(max_ranks, int) or isinstance(max_ranks, bool)
                or max_ranks <= previous):
            raise ModelingError(f"ep_tier_max_ranks_invalid:{index}")
        bw_gbps = _positive(tier.get("bw_gbps"), f"ep_tier_bw_gbps_invalid:{index}")
        name = str(tier.get("name", f"tier_{index}"))
        normalized.append({"name": name, "max_ranks": max_ranks,
                           "bw_gbps": bw_gbps})
        previous = max_ranks
    return tuple(normalized)


def compute_speculative_accept_step(accept_rate: float, draft_tokens: int) -> float:
    """Expected accepted tokens per verification step.

    ``A5_EP建模.xlsx`` evaluates ``(1 - a^(d+1)) / (1 - a)`` for
    ``a=0.75`` and ``d=3`` to exactly ``2.734375``.
    """
    rate = _non_negative(accept_rate, "speculative_accept_rate")
    if rate > 1.0:
        raise ModelingError("speculative_accept_rate_invalid")
    if (not isinstance(draft_tokens, int) or isinstance(draft_tokens, bool)
            or draft_tokens < 0):
        raise ModelingError("speculative_draft_tokens_invalid")
    if math.isclose(rate, 1.0):
        return float(draft_tokens + 1)
    return (1.0 - rate ** (draft_tokens + 1)) / (1.0 - rate)


def compute_ep_effective_bandwidth(
    ep: int,
    tiers: Sequence[Mapping[str, Any]] | None = None,
    *,
    bw_utilization: float = 1.0,
    preset: str = "a5_pcie_16p",
) -> dict[str, Any]:
    """Return the bottleneck effective all-to-all bandwidth for ``ep`` ranks."""
    if not isinstance(ep, int) or isinstance(ep, bool) or ep < 2:
        raise ModelingError(f"ep_effective_bandwidth_requires_ep_at_least_two:{ep}")
    utilization = _positive(bw_utilization, "ep_bw_utilization_invalid")
    if utilization > 1.0:
        raise ModelingError("ep_bw_utilization_invalid")
    normalized = _tiers_for(str(preset), tiers)
    active: list[dict[str, Any]] = []
    first = normalized[0]
    if first["max_ranks"] > 1:
        outbound = (min(ep, first["max_ranks"]) - 1) / ep
        raw = first["bw_gbps"] / outbound
        active.append({
            "index": 0, "name": first["name"],
            "max_ranks": first["max_ranks"], "bw_gbps": first["bw_gbps"],
            "outbound_ratio": outbound, "raw_effective_bw_gbps": raw,
            "effective_bw_gbps": raw * utilization,
        })
    for index in range(1, len(normalized)):
        previous = normalized[index - 1]
        if ep <= previous["max_ranks"]:
            break
        tier = normalized[index]
        outbound = (ep - previous["max_ranks"]) / ep
        raw = tier["bw_gbps"] / outbound
        active.append({
            "index": index, "name": tier["name"],
            "max_ranks": tier["max_ranks"], "bw_gbps": tier["bw_gbps"],
            "outbound_ratio": outbound, "raw_effective_bw_gbps": raw,
            "effective_bw_gbps": raw * utilization,
        })
    if not active and len(normalized) == 1:
        # A single declared domain still serves EPs above its nominal maximum;
        # the source keeps the same ``(ep - 1) / ep`` outbound fraction.
        outbound = (ep - 1) / ep
        raw = first["bw_gbps"] / outbound
        active.append({
            "index": 0, "name": first["name"],
            "max_ranks": first["max_ranks"], "bw_gbps": first["bw_gbps"],
            "outbound_ratio": outbound, "raw_effective_bw_gbps": raw,
            "effective_bw_gbps": raw * utilization,
        })
    if not active:
        raise ModelingError(f"ep_effective_bandwidth_no_active_tier:{ep}")
    bottleneck = min(active, key=lambda row: row["raw_effective_bw_gbps"])
    for row in active:
        row["bottleneck"] = row is bottleneck
    raw_effective = float(bottleneck["raw_effective_bw_gbps"])
    return {
        "ep": ep,
        "preset": str(preset),
        "bw_utilization": utilization,
        "outbound_ratio": float(bottleneck["outbound_ratio"]),
        "raw_effective_bw_gbps": raw_effective,
        "effective_bw_gbps": raw_effective * utilization,
        "tier_breakdown": active,
        "tiers": active,
    }


def analyze_ep_mc2_communication(
    ep_values: Sequence[int] = (2, 4, 8, 16, 32, 64, 128, 256),
    batch_size: int = 32,
    hidden_size: int = 7168,
    top_k: int = 8,
    num_moe_layers: int = 61,
    dispatch_dtype: str = "mxfp8",
    combine_dtype: str = "bf16",
    eplb_peak_to_avg: float = 1.2,
    accept_rate: float = 0.75,
    draft_tokens: int = 3,
    n_hops: int = 1,
    per_hop_latency_us: float = 1.5,
    switch_base_latency_us: float = 3.0,
    software_launch_latency_us: float = 7.0,
    bw_utilizations: Sequence[float] = (0.55, 0.65, 0.75, 0.85),
    preset: str = "a5_ub_pod",
    tiers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze MC2 dispatch/combine communication over EP and bandwidth tiers."""
    for label, value in (("batch_size", batch_size), ("hidden_size", hidden_size),
                         ("top_k", top_k), ("num_moe_layers", num_moe_layers)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ModelingError(f"ep_mc2_{label}_invalid:{value}")
    if (not isinstance(n_hops, int) or isinstance(n_hops, bool) or n_hops < 0):
        raise ModelingError(f"ep_mc2_n_hops_invalid:{n_hops}")
    per_hop = _non_negative(per_hop_latency_us, "ep_mc2_per_hop_latency_us_invalid")
    switch = _non_negative(switch_base_latency_us, "ep_mc2_switch_base_latency_us_invalid")
    software = _non_negative(software_launch_latency_us,
                             "ep_mc2_software_launch_latency_us_invalid")
    peak_to_avg = _positive(eplb_peak_to_avg, "ep_mc2_eplb_peak_to_avg_invalid")
    dispatch_bytes = _dtype_bytes(dispatch_dtype, "dispatch")
    combine_bytes = _dtype_bytes(combine_dtype, "combine")
    if isinstance(ep_values, (str, bytes)) or not isinstance(ep_values, Sequence):
        raise ModelingError("ep_mc2_ep_values_invalid")
    for ep in ep_values:
        if not isinstance(ep, int) or isinstance(ep, bool) or ep < 2:
            raise ModelingError(f"ep_mc2_ep_value_invalid:{ep}")
    if (isinstance(bw_utilizations, (str, bytes))
            or not isinstance(bw_utilizations, Sequence)):
        raise ModelingError("ep_mc2_bw_utilizations_invalid")
    utilizations = []
    for utilization in bw_utilizations:
        value = _positive(utilization, "ep_mc2_bw_utilization_invalid")
        if value > 1.0:
            raise ModelingError("ep_mc2_bw_utilization_invalid")
        utilizations.append(value)
    if draft_tokens:
        accept_step = compute_speculative_accept_step(accept_rate, draft_tokens)
    else:
        accept_step = 1.0
    startup_us = n_hops * per_hop + switch + software
    dispatch_per_layer = (batch_size * hidden_size * top_k
                          * dispatch_bytes * peak_to_avg)
    combine_per_layer = (batch_size * hidden_size * top_k
                         * combine_bytes * peak_to_avg)
    rows = []
    for ep in ep_values:
        for utilization in utilizations:
            tier = compute_ep_effective_bandwidth(
                ep, tiers, bw_utilization=utilization, preset=preset)
            effective = tier["effective_bw_gbps"]
            dispatch_us = startup_us + (dispatch_per_layer / (effective * 1e9)) * 1e6
            combine_us = startup_us + (combine_per_layer / (effective * 1e9)) * 1e6
            total_comm_ms = (dispatch_us + combine_us) * num_moe_layers / 1000.0
            rows.append({
                "ep": ep,
                "bw_utilization": utilization,
                "outbound_ratio": tier["outbound_ratio"],
                "raw_effective_bw_gbps": tier["raw_effective_bw_gbps"],
                "effective_bw_gbps": effective,
                "dispatch_us": dispatch_us,
                "combine_us": combine_us,
                "total_comm_ms": total_comm_ms,
                "tpot_comm_ms": total_comm_ms / accept_step,
            })
    return {
        "preset": str(preset),
        "accept_step": accept_step,
        "startup_latency_us": startup_us,
        "dispatch_bytes_per_layer": dispatch_per_layer,
        "combine_bytes_per_layer": combine_per_layer,
        "num_moe_layers": num_moe_layers,
        "ep_values": list(ep_values),
        "bw_utilizations": utilizations,
        "rows": rows,
        "results": rows,
        "ep_results": rows,
    }
