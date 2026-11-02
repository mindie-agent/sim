"""Hierarchical expert/KV cache DDR-SSD offload with prefetch masking.

[Role]
  Price host-DDR and SSD tiering for routed MoE experts and KV cache, and
  expose how much of each transfer hides behind attention, shared-expert,
  routed-FFN, and full layer-stack work. The module is an analysis layer over
  the shared causal engine; it never loads weights or touches devices.

[Invariants]
  * Disabled policies reproduce the baseline step exactly: zero transfer,
    zero exposed time, and zero resident-byte movement.
  * Expert bytes are derived from catalog weight terms per stage rank, so TP,
    EP, ETP, PP and quantization keep their existing ownership boundaries.
  * Each transfer tier is charged its own bandwidth, utilization, and latency;
    MTP hit rates discount redundant bytes, never the fixed per-layer latency.
  * Prefetch masks can only hide work, never create negative exposed time, and
    the v1 -> v4 masks are monotone by construction.
  * All ratios and bandwidths are validated before arithmetic is performed.

[Timeouts & Locks]
  Pure in-memory arithmetic and one bounded operator evaluation; no I/O,
  subprocesses, sleeps, device contexts, or locks.

[Side-Effect Policy]
  Inputs are never mutated; every entry point returns a new plain dictionary.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import analyze_network_efficiency
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec

_PREFETCH_VERSIONS = ("v1", "v2", "v3", "v4")
_ATTENTION_FAMILIES = frozenset({
    "attention", "qkv_projection", "output_projection", "rotary", "cache_store",
    "gdn_gating", "short_convolution", "recurrent_vector",
})
_FFN_FAMILIES = frozenset({
    "grouped_gemm", "moe_ffn", "gate_up_projection", "down_projection", "swiglu",
    "situ", "router", "topk", "routing", "alltoall", "alltoallv", "projection",
})
_ATTENTION_RECIPES = frozenset({
    "q_a", "q_b", "kv_a", "kv_b", "qkv", "qkv_projection", "linear_qkvz",
    "linear_output", "q_proj", "kv_proj", "q_absorb", "v_expand", "q_flat",
    "k_flat", "v_flat", "qkv_concatenate", "qkv_split_materialize",
})
_ATTENTION_NORMS = frozenset({"q_norm", "kv_norm", "qk_norm", "q_a_norm", "kv_a_norm"})
_ATTENTION_COMMS = frozenset({"linear_reduce", "catalog_tp_allreduce", "output_reduce"})
_FFN_TOKENS = (
    "expert", "grouped", "moe", "ffn", "router", "route_", "topk", "gate_up",
    "dense_down", "dense_gate_up", "swiglu", "situ", "latent", "shared",
)


@dataclass(frozen=True)
class OffloadPolicy:
    """Validated host-DDR/SSD offload and prefetch policy."""

    expert_offload_enabled: bool = False
    force_expert_offload: bool = False
    expert_ddr_ratio: float = 0.0
    ddr_bw_gbps: float = 200.0
    ddr_bw_util: float = 0.70
    ddr_latency_us: float = 5.0
    ssd_bw_gbps: float = 14.0
    ssd_bw_util: float = 0.70
    ssd_latency_us: float = 80.0
    kvc_ddr_ratio: float = 0.0
    kvc_ssd_ratio: float = 0.0
    kvc_prefetch_version: str = "v3"
    mtp_ffn_hit_rate: float = 0.0
    mtp_kvc_hit_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("expert_offload_enabled", "force_expert_offload"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"offload_policy_boolean_required:{name}")
        for name in ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio",
                     "mtp_ffn_hit_rate", "mtp_kvc_hit_rate"):
            _ratio(name, getattr(self, name))
        for name in ("ddr_bw_util", "ssd_bw_util"):
            _utilization(name, getattr(self, name))
        for name in ("ddr_bw_gbps", "ssd_bw_gbps"):
            _positive(name, getattr(self, name))
        for name in ("ddr_latency_us", "ssd_latency_us"):
            _nonnegative(name, getattr(self, name))
        if self.kvc_prefetch_version not in _PREFETCH_VERSIONS:
            raise ValueError(f"offload_policy_invalid_prefetch_version:{self.kvc_prefetch_version}")
        if self.kvc_ddr_ratio + self.kvc_ssd_ratio > 1.0 + 1e-12:
            raise ValueError("offload_policy_kvc_ratios_exceed_unity")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None = None) -> "OffloadPolicy":
        """Build a policy with strict ratio, bandwidth, and enum validation."""
        if mapping is None:
            return cls()
        if isinstance(mapping, cls):
            return mapping
        if not isinstance(mapping, Mapping):
            raise ValueError("offload_policy_mapping_required")
        unknown = set(mapping) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("offload_policy_unknown_fields:" + ",".join(sorted(unknown)))
        return cls(**dict(mapping))


def _ratio(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"offload_policy_invalid_ratio:{name}:{value}")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"offload_policy_invalid_ratio:{name}:{value}")
    return result


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"offload_policy_must_be_positive:{name}:{value}")
    return float(value)


def _utilization(name: str, value: Any) -> float:
    result = _positive(name, value)
    if result > 1.0:
        raise ValueError(f"offload_policy_invalid_utilization:{name}:{value}")
    return result


def _nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) < 0.0:
        raise ValueError(f"offload_policy_must_be_nonnegative:{name}:{value}")
    return float(value)


def _resolve_policy(policy: OffloadPolicy | Mapping[str, Any] | None) -> OffloadPolicy:
    if policy is None:
        return OffloadPolicy()
    if isinstance(policy, OffloadPolicy):
        return policy
    return OffloadPolicy.from_mapping(policy)


def _validate_geometry(phase: str, tokens: int, context: int, sequences: int, scope: str) -> None:
    if phase not in {"prefill", "decode"} or scope not in {"body", "model"}:
        raise ValueError(f"offload_invalid_phase_or_scope:{phase}:{scope}")
    for name, value, minimum in (("tokens", tokens, 1), ("sequences", sequences, 1), ("context", context, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"offload_invalid_{name}:{value}")


def _text(row: Mapping[str, Any]) -> str:
    return " ".join(str(row.get(key, "")) for key in ("op_id", "family", "recipe")).lower()


def _is_attention_op(row: Mapping[str, Any]) -> bool:
    family = str(row.get("family", "")).lower()
    recipe = str(row.get("recipe", "")).lower()
    if family in _ATTENTION_FAMILIES:
        return True
    if family == "gemm" and recipe in _ATTENTION_RECIPES:
        return True
    if family == "rmsnorm" and recipe in _ATTENTION_NORMS:
        return True
    if family == "allreduce" and recipe in _ATTENTION_COMMS:
        return True
    return "attention" in _text(row) and "output" not in family and "residual" not in recipe


def _is_shared_expert_op(row: Mapping[str, Any]) -> bool:
    text = _text(row)
    if "shared" in text:
        return True
    return (str(row.get("family", "")).lower() in {"gate_up_projection", "down_projection", "swiglu"}
            and str(row.get("recipe", "")).lower() in {"dense_gate_up", "dense_down", "swiglu"})


def _is_dispatch_op(row: Mapping[str, Any]) -> bool:
    family = str(row.get("family", "")).lower()
    return family in {"alltoall", "alltoallv"} and "dispatch" in _text(row)


def _is_ffn_op(row: Mapping[str, Any]) -> bool:
    if _is_attention_op(row):
        return False
    family = str(row.get("family", "")).lower()
    if family in _FFN_FAMILIES:
        return True
    return any(token in _text(row) for token in _FFN_TOKENS)


def _sum_ops(report: Mapping[str, Any], predicate) -> float:
    return float(sum(float(row.get("total_ms", 0.0) or 0.0)
                     for row in report.get("operators", ()) if predicate(row)))


def _baseline_components(parsed: CaseSpec, *, phase: str, tokens: int, context: int,
                         sequences: int, scope: str) -> dict[str, float]:
    from dataclasses import replace as _replace
    clean_meta = tuple((k, v) for k, v in parsed.runtime.metadata if k != "offload")
    clean_mem = {k: v for k, v in (parsed.runtime.memory or {}).items()
                 if k not in {"expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio"}}
    clean_case = _replace(parsed, runtime=_replace(
        parsed.runtime, metadata=clean_meta, memory=clean_mem))
    report = analyze_network_efficiency(
        clean_case, phase=phase, tokens=tokens, context=context, sequences=sequences,
        scope=scope, run_graph=True,
    )
    baseline = float(report["graph_duration_ms"])
    operator_sum = float(report["operator_sum_ms"])
    scale = baseline / operator_sum if operator_sum > 1e-12 else 1.0
    attention = _sum_ops(report, _is_attention_op) * scale
    ffn = _sum_ops(report, _is_ffn_op) * scale
    shared = _sum_ops(report, _is_shared_expert_op) * scale
    dispatch = _sum_ops(report, _is_dispatch_op) * scale
    return {
        "baseline_step_ms": baseline, "attention_ms": attention, "ffn_ms": ffn,
        "shared_expert_ms": shared, "dispatch_ms": dispatch, "operator_sum_ms": operator_sum,
    }


def _moe_geometry(parsed: CaseSpec, memory: Memory, batch_tokens: int) -> dict[str, Any]:
    """Derive routed-expert bytes from the same weight terms that own HBM."""
    parallel = parsed.runtime.parallel
    quant = parsed.model.quant or QuantSpec("bf16")
    stage_routed = [0.0] * max(1, parallel.pp)
    layers: list[dict[str, Any]] = []
    for plan in parsed.model.layer_plans:
        if plan.ffn != "moe" or int(plan.num_experts) <= 0 or int(plan.moe_intermediate_size) <= 0:
            continue
        routed_bytes = float(sum(
            term_bytes(term, quant, parallel, parsed.machine.ranks_per_node)
            for term in layer_weight_terms(plan, parsed.model) if term.category == "routed_expert"))
        local_experts = max(1, (int(plan.num_experts) + max(1, int(parallel.ep)) - 1) // max(1, int(parallel.ep)))
        stage = memory.layer_stage[plan.index] if plan.index < len(memory.layer_stage) else 0
        stage_routed[stage] += routed_bytes
        layers.append({
            "layer": int(plan.index), "stage": int(stage),
            "num_experts": int(plan.num_experts), "top_k": int(plan.top_k),
            "local_experts_per_rank": int(local_experts),
            "one_expert_bytes_per_rank": float(routed_bytes / local_experts),
            "routed_expert_bytes_per_rank": float(routed_bytes),
            "active_experts_per_rank": _active_experts_per_rank(
                int(plan.num_experts), int(plan.top_k), batch_tokens, parallel.ep),
            "ddr_bytes": 0.0,
        })
    return {"layers": layers, "stage_routed_bytes": stage_routed}


def _expected_active_experts(num_experts: int, batch_tokens: int, top_k: int) -> float:
    if num_experts <= 0 or batch_tokens <= 0 or top_k <= 0:
        return 0.0
    k = min(int(num_experts), int(top_k))
    if k >= num_experts:
        return float(num_experts)
    miss_per_token = 1.0 - float(k) / float(num_experts)
    expected = float(num_experts) * (1.0 - (miss_per_token ** float(batch_tokens)))
    return min(float(num_experts), max(0.0, expected))


def _active_experts_per_rank(num_experts: int, top_k: int, batch_tokens: int, ep: int) -> float:
    return _expected_active_experts(num_experts, batch_tokens, top_k) / float(max(1, ep))


def _transfer_ms(byte_count: float, bandwidth_gbps: float, utilization: float,
                 latency_us: float) -> float:
    if byte_count <= 0.0:
        return 0.0
    return float(latency_us) / 1000.0 + float(byte_count) / (float(bandwidth_gbps) * float(utilization) * 1e6)


def prepare_expert_offload(
    case: CaseSpec | Mapping[str, Any],
    policy: OffloadPolicy | Mapping[str, Any] | None = None, *,
    phase: str = "decode", tokens: int = 1, context: int = 4096,
    sequences: int = 16, scope: str = "body",
) -> dict[str, Any]:
    """Price routed-expert DDR offload, auto-sizing it when HBM is exceeded."""
    _validate_geometry(phase, tokens, context, sequences, scope)
    parsed, resolved = parse_case(case), _resolve_policy(policy)
    memory = Memory(parsed)
    parallel = parsed.runtime.parallel
    hbm_capacity = float(min(d.hbm_bytes * parsed.runtime.gpu_memory_utilization
                             for d in parsed.machine.dies))
    batch_tokens = max(1, int(tokens) * int(sequences))
    moe = _moe_geometry(parsed, memory, batch_tokens)
    layers = moe["layers"]
    stage_routed = [float(value) for value in moe["stage_routed_bytes"]]
    num_moe_layers = len(layers)
    kv_live = [value * float(sequences) for value in memory.cache_bytes(context)]
    baseline_weights = [gross + saved for gross, saved in zip(memory.weights, memory.expert_hbm_saved)]
    baseline_workspace = [
        ws - stg for ws, stg in zip(
            memory.workspace, getattr(memory, "expert_staging_workspace", [0.0] * len(memory.workspace)))
    ]
    resident = [
        gross + workspace + memory.policy.reserved + kv
        for gross, workspace, kv in zip(baseline_weights, baseline_workspace, kv_live)
    ]
    resident_before = list(resident)
    deficits = [max(0.0, value - hbm_capacity) for value in resident]
    hbm_deficit = max(deficits, default=0.0)
    pp = max(1, int(parallel.pp))
    stage_staging_base = [0.0] * pp
    for layer in layers:
        st = int(layer["stage"])
        active_bytes = float(layer["active_experts_per_rank"] * layer["one_expert_bytes_per_rank"])
        stage_staging_base[st] = max(stage_staging_base[st], min(float(layer["routed_expert_bytes_per_rank"]), active_bytes))
    ws_moe = _moe_geometry(parsed, memory, max(1, memory._resolve_workspace_tokens()))
    ws_staging_base = [0.0] * pp
    for layer in ws_moe["layers"]:
        st = int(layer["stage"])
        active_bytes = float(layer["active_experts_per_rank"] * layer["one_expert_bytes_per_rank"])
        ws_staging_base[st] = max(ws_staging_base[st], min(float(layer["routed_expert_bytes_per_rank"]), active_bytes))

    auto_size = bool(resolved.expert_offload_enabled and resolved.expert_ddr_ratio == 0.0 and hbm_deficit > 0.0)
    activated = bool(resolved.force_expert_offload or resolved.expert_ddr_ratio > 0.0
                     or (resolved.expert_offload_enabled and hbm_deficit > 0.0))
    effective_ratio = 0.0
    if auto_size:
        required = 0.0
        for deficit, routed, stg, ws_stg in zip(deficits, stage_routed, stage_staging_base, ws_staging_base):
            if deficit <= 0.0:
                continue
            reducible = max(0.0, routed - max(stg, ws_stg))
            if reducible <= 0.0:
                required = math.inf
                break
            required = max(required, deficit / reducible)
        effective_ratio = min(1.0, max(0.0, required))
    elif resolved.expert_ddr_ratio > 0.0:
        effective_ratio = resolved.expert_ddr_ratio
    if not activated:
        effective_ratio = 0.0

    saved_per_stage = [effective_ratio * routed for routed in stage_routed]
    staging_per_stage = [effective_ratio * stg for stg in stage_staging_base]
    net_saved_per_stage = [max(0.0, saved - stg) for saved, stg in zip(saved_per_stage, staging_per_stage)]
    resident_after = [
        max(0.0, value - saved + stg)
        for value, saved, stg in zip(resident_before, saved_per_stage, staging_per_stage)
    ]
    feasible = all(value <= hbm_capacity + 1e-6 for value in resident_after)
    saved_per_rank = max(saved_per_stage, default=0.0)
    routed_per_rank = max(stage_routed, default=0.0)

    baseline = _baseline_components(
        parsed, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope,
    )
    mask_window = min(
        baseline["baseline_step_ms"],
        baseline["attention_ms"] + baseline["shared_expert_ms"] + baseline["dispatch_ms"],
    )
    ddr_bytes_total = 0.0
    for layer in layers:
        per_rank = layer["active_experts_per_rank"]
        layer["ddr_bytes"] = float(
            per_rank * layer["one_expert_bytes_per_rank"] * effective_ratio
            * (1.0 - resolved.mtp_ffn_hit_rate)
        )
        ddr_bytes_total += layer["ddr_bytes"]
    ddr_transfer = 0.0
    if num_moe_layers and activated and effective_ratio > 0.0:
        ddr_transfer = (
            float(num_moe_layers) * resolved.ddr_latency_us / 1000.0
            + ddr_bytes_total / (resolved.ddr_bw_gbps * resolved.ddr_bw_util * 1e6)
        )
    exposed = max(0.0, ddr_transfer - mask_window)

    return {
        "schema": "mindie-sim.expert-offload/1",
        "geometry": {"phase": phase, "tokens": tokens, "context": context,
                     "sequences": sequences, "scope": scope},
        "activated": bool(activated),
        "activation_reason": (
            "forced" if resolved.force_expert_offload else
            "ratio" if resolved.expert_ddr_ratio > 0.0 else
            "hbm_exceeded" if activated else "hbm_sufficient"
        ),
        "effective_expert_ddr_ratio": float(effective_ratio),
        "expert_offload_enabled": bool(resolved.expert_offload_enabled),
        "force_expert_offload": bool(resolved.force_expert_offload),
        "hbm_capacity_bytes": hbm_capacity, "hbm_deficit_bytes": float(hbm_deficit),
        "resident_bytes_per_stage_rank": [float(value) for value in resident_before],
        "resident_after_offload_bytes_per_stage_rank": [float(value) for value in resident_after],
        "routed_expert_weight_bytes_per_stage_rank": [float(value) for value in stage_routed],
        "routed_expert_weight_bytes_per_rank": float(routed_per_rank),
        "hbm_saved_bytes_per_rank": float(saved_per_rank),
        "hbm_saved_bytes_per_stage_rank": [float(value) for value in saved_per_stage],
        "expert_staging_bytes_per_stage_rank": [float(value) for value in staging_per_stage],
        "net_hbm_saved_bytes_per_rank": float(max(net_saved_per_stage, default=0.0)),
        "feasible_after_offload": bool(feasible), "num_moe_layers": int(num_moe_layers),
        "moe_layers": [dict(layer) for layer in layers], "ddr_bytes_total": float(ddr_bytes_total),
        "ddr_bytes_per_layer": float(ddr_bytes_total / max(1, num_moe_layers)),
        "ddr_transfer_ms": float(ddr_transfer), "attention_ms": float(baseline["attention_ms"]),
        "shared_expert_ms": float(baseline["shared_expert_ms"]),
        "dispatch_ms": float(baseline["dispatch_ms"]), "mask_window_ms": float(mask_window),
        "exposed_expert_offload_ms": float(exposed),
        "baseline_step_ms": float(baseline["baseline_step_ms"]),
        "offloaded_step_ms": float(baseline["baseline_step_ms"] + exposed),
        "mtp_ffn_hit_rate": float(resolved.mtp_ffn_hit_rate),
        "ddr_bw_gbps": float(resolved.ddr_bw_gbps), "ddr_bw_util": float(resolved.ddr_bw_util),
        "ddr_latency_us": float(resolved.ddr_latency_us),
        "parallel": {"tp": parallel.tp, "ep": parallel.ep, "pp": parallel.pp},
    }


def analyze_kv_cache_offload(
    case: CaseSpec | Mapping[str, Any],
    policy: OffloadPolicy | Mapping[str, Any] | None = None, *,
    phase: str = "decode", tokens: int = 1, context: int = 4096,
    sequences: int = 16, scope: str = "body",
) -> dict[str, Any]:
    """Price KV cache DDR/SSD tiering with v1-v4 prefetch masking."""
    _validate_geometry(phase, tokens, context, sequences, scope)
    parsed, resolved = parse_case(case), _resolve_policy(policy)
    memory = Memory(parsed)
    kv_per_request = memory.cache_bytes(context, padded=False)
    kv_total = float(sum(kv_per_request)) * float(sequences)
    hbm_ratio = max(0.0, 1.0 - resolved.kvc_ddr_ratio - resolved.kvc_ssd_ratio)
    hbm_bytes = kv_total * hbm_ratio
    ddr_bytes = kv_total * resolved.kvc_ddr_ratio
    ssd_bytes = kv_total * resolved.kvc_ssd_ratio
    ddr_transfer_bytes = ddr_bytes * (1.0 - resolved.mtp_kvc_hit_rate)
    ssd_transfer_bytes = ssd_bytes * (1.0 - resolved.mtp_kvc_hit_rate)
    t_ddr = _transfer_ms(ddr_transfer_bytes, resolved.ddr_bw_gbps,
                         resolved.ddr_bw_util, resolved.ddr_latency_us)
    t_ssd = _transfer_ms(ssd_transfer_bytes, resolved.ssd_bw_gbps,
                         resolved.ssd_bw_util, resolved.ssd_latency_us)

    baseline = _baseline_components(
        parsed, phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope,
    )
    baseline_step = baseline["baseline_step_ms"]
    attention = min(baseline_step, baseline["attention_ms"])
    ffn = min(baseline_step, baseline["ffn_ms"])
    masks = {"v1": 0.0, "v2": attention, "v3": min(baseline_step, attention + 0.5 * ffn), "v4": baseline_step}
    mask_window = masks[resolved.kvc_prefetch_version]
    activated = bool(resolved.kvc_ddr_ratio + resolved.kvc_ssd_ratio > 0.0)
    transfer = max(t_ddr, t_ssd)
    exposed = max(0.0, transfer - mask_window)
    return {
        "schema": "mindie-sim.kv-offload/1",
        "geometry": {"phase": phase, "tokens": tokens, "context": context,
                     "sequences": sequences, "scope": scope},
        "activated": activated, "kvc_ddr_ratio": float(resolved.kvc_ddr_ratio),
        "kvc_ssd_ratio": float(resolved.kvc_ssd_ratio), "kvc_hbm_ratio": float(hbm_ratio),
        "kvc_prefetch_version": resolved.kvc_prefetch_version,
        "mtp_kvc_hit_rate": float(resolved.mtp_kvc_hit_rate), "total_kv_bytes": kv_total,
        "hbm_resident_kv_bytes": float(hbm_bytes), "ddr_kv_bytes": float(ddr_bytes),
        "ssd_kv_bytes": float(ssd_bytes), "ddr_transfer_bytes": float(ddr_transfer_bytes),
        "ssd_transfer_bytes": float(ssd_transfer_bytes), "t_ddr_ms": float(t_ddr),
        "t_ssd_ms": float(t_ssd), "transfer_ms": float(transfer),
        "attention_ms": float(baseline["attention_ms"]), "ffn_ms": float(baseline["ffn_ms"]),
        "layer_step_ms": float(baseline_step),
        "per_layer_step_ms": float(baseline_step / max(1, int(parsed.model.layers))),
        "mask_window_ms": float(mask_window), "exposed_kv_offload_ms": float(exposed),
        "baseline_step_ms": float(baseline_step),
        "offloaded_step_ms": float(baseline_step + exposed),
        "prefetch_masks_ms": {version: float(value) for version, value in masks.items()},
        "ddr_bw_gbps": float(resolved.ddr_bw_gbps), "ddr_bw_util": float(resolved.ddr_bw_util),
        "ddr_latency_us": float(resolved.ddr_latency_us),
        "ssd_bw_gbps": float(resolved.ssd_bw_gbps), "ssd_bw_util": float(resolved.ssd_bw_util),
        "ssd_latency_us": float(resolved.ssd_latency_us),
    }


def resolve_case_offload_policy(case: CaseSpec, memory_policy: Any = None) -> OffloadPolicy | None:
    """Resolve active offload policy from case metadata and memory policy."""
    raw_meta = dict(case.runtime.metadata).get("offload", {})
    if raw_meta is None:
        raw_meta = {}
    if not isinstance(raw_meta, Mapping):
        raise ValueError("memory_offload_metadata_mapping_required")
    merged: dict[str, Any] = dict(raw_meta)
    mem_cfg = dict(case.runtime.memory or {})
    for key in ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio"):
        if mem_cfg.get(key, 0.0):
            merged[key] = mem_cfg[key]
        if memory_policy is not None and getattr(memory_policy, key, 0.0):
            merged[key] = getattr(memory_policy, key)
    if not merged:
        return None
    policy = OffloadPolicy.from_mapping(merged)
    if (policy.expert_ddr_ratio <= 0.0 and policy.kvc_ddr_ratio <= 0.0
            and policy.kvc_ssd_ratio <= 0.0 and not policy.force_expert_offload):
        return None
    return policy


def offload_resources(policy: OffloadPolicy | None, namespace: str, dp: int, pp: int) -> list[Any]:
    """Build host DDR and SSD transfer resources for Execution when offload is active."""
    if policy is None:
        return []
    from mindie_sim.prediction.engine import Resource
    ddr_rate = policy.ddr_bw_gbps * policy.ddr_bw_util * 1e6
    ssd_rate = policy.ssd_bw_gbps * policy.ssd_bw_util * 1e6
    resources = []
    for replica in range(max(1, dp)):
        for stage in range(max(1, pp)):
            resources.append(Resource(f"{namespace}offload_ddr@r{replica}:s{stage}",
                                      ddr_rate, "bytes", "fifo"))
            resources.append(Resource(f"{namespace}offload_ssd@r{replica}:s{stage}",
                                      ssd_rate, "bytes", "fifo"))
    return resources


def _submit_offload_transfer(
    engine: Any, name: str, work: Mapping[str, float], *,
    after: tuple[Any, ...], latency_us: float, owner: str,
) -> Any:
    startup_ms = float(latency_us) / 1000.0
    if startup_ms > 0.0:
        start = engine.submit(f"{name}:startup", after=after, latency=startup_ms, owner=owner)
        return engine.submit(name, work, after=(start,), owner=owner)
    return engine.submit(name, work, after=after, owner=owner)


def is_routed_expert_entry_op(op: Any) -> bool:
    if getattr(op, "layer", -1) < 0:
        return False
    meta = dict(getattr(op, "metadata", ()) or ())
    if getattr(op, "family", "") == "grouped_gemm" or getattr(op, "recipe", "") in {
        "grouped_gate_up", "balanced_grouped_gate_up", "grouped_expert_gemm",
    }:
        return True
    if meta.get("semantic_recipe") in {"grouped_gate_up", "balanced_grouped_gate_up"}:
        return True
    if meta.get("weight_category") == "routed_expert":
        return True
    return any(
        isinstance(r, Mapping) and (
            r.get("family") == "grouped_gemm"
            or r.get("recipe") in {"grouped_gate_up", "balanced_grouped_gate_up"}
        )
        for r in meta.get("fusion_semantic_recipes", ())
    )


def is_kv_or_state_entry_op(op: Any) -> bool:
    if getattr(op, "layer", -1) < 0:
        return False
    return getattr(op, "family", "") in {
        "attention", "attention_online", "mla", "mla_decode",
        "compressed_attention", "dsa", "indexer", "recurrent_scan",
        "recurrent_vector", "gdn_state", "gdn_recurrent", "kda",
        "kda_recurrent", "short_convolution",
    } or getattr(op, "recipe", "") in {"short_convolution", "causal_conv1d_update"}


def _resolve_active_experts_from_op(op: Any) -> float | None:
    if op is None:
        return None
    op_meta = dict(getattr(op, "metadata", ()) or ())
    sem_meta = op_meta.get("semantic_metadata")
    sources = (op_meta, dict(sem_meta) if isinstance(sem_meta, Mapping) else {})
    for meta in sources:
        val = meta.get("expected_active_experts", meta.get("matrix_groups"))
        if val is not None:
            return float(val)
        rows = meta.get("matrix_group_rows")
        if isinstance(rows, (list, tuple)) and rows:
            return float(sum(1 for r in rows if r > 0))
        realization = meta.get("routing_realization")
        if isinstance(realization, Mapping):
            if realization.get("active") is not None:
                return float(realization["active"])
            r_rows = realization.get("local_expert_rows", realization.get("expert_rows"))
            if isinstance(r_rows, (list, tuple)) and r_rows:
                return float(sum(1 for r in r_rows if r > 0))
        gmm = meta.get("grouped_gemm")
        if isinstance(gmm, Mapping):
            g_rows = gmm.get("expert_rows")
            if isinstance(g_rows, (list, tuple)) and g_rows:
                return float(sum(1 for r in g_rows if r > 0))
    return None


def schedule_expert_layer_offload(
    execution: Any, prefix: str, replica: int, stage: int, layer: int,
    valid_tokens: int, input_ready: Any, op: Any = None,
) -> Any | None:
    """Submit per-MoE-layer DDR expert fetch contended on the stage FIFO DDR bus."""
    policy = getattr(execution, "offload_policy", None)
    active_override = _resolve_active_experts_from_op(op)
    if policy is None or policy.expert_ddr_ratio <= 0.0 or (valid_tokens <= 0 and not active_override):
        return None
    memory = getattr(execution, "_offload_memory", None)
    if memory is None:
        memory = Memory(execution.case)
        execution._offload_memory = memory
    moe = _moe_geometry(execution.case, memory, max(1, int(valid_tokens)))
    layer_info = next((item for item in moe["layers"] if item["layer"] == layer), None)
    if layer_info is None:
        return None
    active_experts = float(active_override) if active_override is not None else float(layer_info["active_experts_per_rank"])
    ddr_bytes = float(
        active_experts * layer_info["one_expert_bytes_per_rank"]
        * policy.expert_ddr_ratio * (1.0 - policy.mtp_ffn_hit_rate)
    )
    if ddr_bytes <= 0.0:
        return None
    ddr_res = f"{execution.namespace}offload_ddr@r{replica}:s{stage}"
    hbm_res = execution.stage_names[replica, stage]["hbm"]
    return _submit_offload_transfer(
        execution.engine,
        f"{prefix}:s{stage}.l{layer}:expert_ddr_offload",
        {ddr_res: ddr_bytes, hbm_res: ddr_bytes},
        after=(input_ready,),
        latency_us=policy.ddr_latency_us,
        owner=f"replica:{replica}",
    )


def _layer_kv_bytes(case: CaseSpec, memory: Memory, layer: int, ctx_tokens: int) -> float:
    from mindie_sim.catalog.cache import layer_cache_components, local_kv_heads
    from mindie_sim.catalog.draft import optional_draft_spec
    from mindie_sim.schema.common import dtype_bytes

    model, p = case.model, case.runtime.parallel
    if 0 <= layer < len(model.layer_plans):
        plan = model.layer_plans[layer]
        comps = layer_cache_components(
            model, plan, p, ctx_tokens, capacity_padded=False,
            interleave=case.runtime.kv_block_size)
        total = float(sum(comps.values()))
        if layer == len(model.layer_plans) - 1 and memory.draft_kind and model.auxiliary is not None:
            draft = optional_draft_spec(model, memory.draft_kind)
            row = ((draft.kv_cache_elements or draft.kv_lora_rank + draft.qk_rope_head_dim)
                   if draft.attention_kind == "mla"
                   else 2 * local_kv_heads(draft.num_kv_heads, p.tp) * draft.head_dim)
            total += float(ctx_tokens * row * dtype_bytes(model.kv_dtype) * draft.layers)
        return total
    return 0.0


def _submit_kv_tier_transfers(
    execution: Any, stem: str, replica: int, stage: int, kv_bytes: float,
    anchor: tuple[Any, ...], *, discount_hit: bool,
) -> tuple[Any, ...]:
    if kv_bytes <= 0.0:
        return ()
    policy = execution.offload_policy
    scale = (1.0 - policy.mtp_kvc_hit_rate) if discount_hit else 1.0
    hbm_res = execution.stage_names[replica, stage]["hbm"]
    events = []
    ddr_bytes = kv_bytes * policy.kvc_ddr_ratio * scale
    if ddr_bytes > 0.0:
        ddr_res = f"{execution.namespace}offload_ddr@r{replica}:s{stage}"
        events.append(_submit_offload_transfer(
            execution.engine, f"{stem}_ddr", {ddr_res: ddr_bytes, hbm_res: ddr_bytes},
            after=anchor, latency_us=policy.ddr_latency_us, owner=f"replica:{replica}"))
    ssd_bytes = kv_bytes * policy.kvc_ssd_ratio * scale
    if ssd_bytes > 0.0:
        ssd_res = f"{execution.namespace}offload_ssd@r{replica}:s{stage}"
        events.append(_submit_offload_transfer(
            execution.engine, f"{stem}_ssd", {ssd_res: ssd_bytes, hbm_res: ssd_bytes},
            after=anchor, latency_us=policy.ssd_latency_us, owner=f"replica:{replica}"))
    return tuple(events)


def _request_kv_tokens(view: Any) -> tuple[int, int]:
    q_tokens = max(0, int(getattr(view, "q_len", 0)))
    total_kv = max(q_tokens, int(getattr(view, "kv_len", 0)))
    return max(0, total_kv - q_tokens), max(1, q_tokens)


def schedule_kv_layer_offload(
    execution: Any, prefix: str, replica: int, stage: int, layer: int,
    work: Any, input_ready: Any, consumer_deps: tuple[Any, ...],
    layer_entries: Mapping[int, Any], *, prev_kv_done: Any = None,
) -> tuple[Any, ...]:
    """Submit per-layer KV cache DDR/SSD fetches before the layer's attention consumer."""
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    memory = getattr(execution, "_offload_memory", None)
    if memory is None:
        memory = Memory(execution.case)
        execution._offload_memory = memory
    kv_bytes = 0.0
    for view in work.plan.requests:
        prior_kv, _ = _request_kv_tokens(view)
        if prior_kv > 0:
            kv_bytes += _layer_kv_bytes(execution.case, memory, layer, prior_kv)
    if kv_bytes <= 0.0:
        return ()
    if policy.kvc_prefetch_version == "v1":
        anchor = tuple(consumer_deps) or (input_ready,)
    elif policy.kvc_prefetch_version == "v2":
        anchor = (layer_entries.get(layer, input_ready),)
    elif policy.kvc_prefetch_version == "v3":
        anchor = (layer_entries.get(layer - 1, input_ready),)
    else:
        anchor = (input_ready,)
    if prev_kv_done is not None:
        anchor = tuple(dict.fromkeys((*anchor, prev_kv_done)))
    return _submit_kv_tier_transfers(
        execution, f"{prefix}:s{stage}.l{layer}:kv_offload", replica, stage,
        kv_bytes, anchor, discount_hit=True)


def schedule_kv_layer_writeback(
    execution: Any, prefix: str, replica: int, stage: int, layer: int,
    work: Any, producer_done: Any,
) -> tuple[Any, ...]:
    """Submit per-layer KV/state writebacks to DDR/SSD after the producer completes."""
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    memory = getattr(execution, "_offload_memory", None)
    if memory is None:
        memory = Memory(execution.case)
        execution._offload_memory = memory
    write_bytes = sum(
        _layer_kv_bytes(execution.case, memory, layer, _request_kv_tokens(view)[1])
        for view in work.plan.requests
    )
    return _submit_kv_tier_transfers(
        execution, f"{prefix}:s{stage}.l{layer}:kv_writeback", replica, stage,
        write_bytes, (producer_done,), discount_hit=False)


def schedule_kv_iteration_offload(
    execution: Any, prefix: str, replica: int, work: Any, input_ready: Any,
    physical_ops: tuple[Any, ...] | list[Any], ids: Mapping[str, Any],
    scheduled_layers: set[tuple[int, int]] | None = None,
) -> tuple[Any, ...]:
    """Submit fallback KV cache DDR/SSD tier transfers for layers without scheduled ops."""
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    memory = getattr(execution, "_offload_memory", None)
    if memory is None:
        memory = Memory(execution.case)
        execution._offload_memory = memory
    pp = max(1, int(execution.case.runtime.parallel.pp))
    covered = scheduled_layers or set()
    stage_fetch, stage_write = [0.0] * pp, [0.0] * pp
    if execution.case.model.layer_plans:
        for plan in execution.case.model.layer_plans:
            st = memory.layer_stage[plan.index] if plan.index < len(memory.layer_stage) else 0
            if (st, plan.index) in covered:
                continue
            for view in work.plan.requests:
                prior_kv, q_tokens = _request_kv_tokens(view)
                if prior_kv > 0:
                    stage_fetch[st] += _layer_kv_bytes(execution.case, memory, plan.index, prior_kv)
                stage_write[st] += _layer_kv_bytes(execution.case, memory, plan.index, q_tokens)
    elif not covered:
        for view in work.plan.requests:
            prior_kv, q_tokens = _request_kv_tokens(view)
            if prior_kv > 0:
                for stage, val in enumerate(memory.cache_bytes(prior_kv, padded=False)):
                    stage_fetch[stage] += float(val)
            for stage, val in enumerate(memory.cache_bytes(q_tokens, padded=False)):
                stage_write[stage] += float(val)
    events: list[Any] = []
    for stage in range(pp):
        stage_ops = [op for op in physical_ops if op.stage == stage and op.op_id in ids]
        if policy.kvc_prefetch_version == "v1":
            anchor = tuple(ids[op.op_id] for op in stage_ops) or (input_ready,)
        elif policy.kvc_prefetch_version == "v2":
            attn_ops = {ids[op.op_id] for op in stage_ops if _is_attention_op(
                {"op_id": op.op_id, "family": op.family, "recipe": op.recipe})}
            anchor = tuple(ids[op.op_id] for op in stage_ops if ids[op.op_id] not in attn_ops) or (input_ready,)
        elif policy.kvc_prefetch_version == "v3":
            half = max(1, len(stage_ops) // 2)
            anchor = tuple(ids[op.op_id] for op in stage_ops[:half]) or (input_ready,)
        else:
            anchor = (input_ready,)
        events.extend(_submit_kv_tier_transfers(
            execution, f"{prefix}:s{stage}:kv_offload", replica, stage,
            stage_fetch[stage], anchor, discount_hit=True))
        tail = tuple(ids[op.op_id] for op in stage_ops) or (input_ready,)
        events.extend(_submit_kv_tier_transfers(
            execution, f"{prefix}:s{stage}:kv_writeback", replica, stage,
            stage_write[stage], tail, discount_hit=False))
    return tuple(events)


__all__ = [
    "OffloadPolicy", "prepare_expert_offload", "analyze_kv_cache_offload",
    "resolve_case_offload_policy", "offload_resources", "schedule_expert_layer_offload",
    "schedule_kv_layer_offload", "schedule_kv_layer_writeback", "schedule_kv_iteration_offload",
]
