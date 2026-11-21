"""Routed expert fetch scheduling.

[Role] Derive per-layer expert traffic and schedule each DDR fetch.
[Invariants] Catalog weight terms retain their existing rank ownership.
[Timeouts & Locks] Bounded by model layers and operator metadata. No waits or locks.
[Side-Effect Policy] Submits work to the caller's engine.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.capacity import Memory
from mindie_sim.schema.case import Case as CaseSpec

from .policy import _offload_memory_for, _submit_offload_transfer


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
    memory = _offload_memory_for(execution)
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
