"""Retained-plan sweep rows for an explicitly locked parallel declaration.

[Role] Prepare one baseline batch/prepared plan and emit a row without
  reselecting parallel or batch geometry.
[Invariants] The retained plan identity, batch and layer partition are the
  evaluated ones. Feasibility still requires SLA, configured VRAM batch and the
  real capacity ledger. No cross-candidate metric stitching.
[Timeouts & Locks] Pure in-memory evaluation; cooperative deadline via caller.
[Side-Effect Policy] Returns new dictionaries; never mutates the case.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib

from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    evaluate_fixed_plan, prepare_fixed_plan, solve_sla_sizing,
)
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.sweep_engine import BASELINE_SCENARIO, LOCKED_STAGE, parallel_report

MS_PER_S = 1000.0


def locked_plan_identity(plan: Mapping[str, object]) -> str:
    graphs = plan["graphs"]
    head = (getattr(graphs, "default_selection", None),
            tuple((str(op.op_id), str(op.recipe), tuple(op.local_shape))
                  for op in next(iter(graphs.values()))))
    return hashlib.sha256(repr(head).encode("utf-8")).hexdigest()[:16]


def prepare_locked_plan(parsed: CaseSpec, *, sla_ms: float, context: int,
                        phase: str, scope: str,
                        tokens_per_seq: int | None, output_budget: int,
                        max_sequences_per_dp: int) -> dict[str, object]:
    result = solve_sla_sizing(
        parsed, sla_ms=sla_ms, context=context, phase=phase, scope=scope,
        tokens_per_seq=tokens_per_seq, output_budget=output_budget,
        max_sequences_per_dp=max_sequences_per_dp,
        scenarios=BASELINE_SCENARIO)
    scenario = result["scenarios"][0]
    candidate = scenario.get("feasible_candidate")
    if candidate is None:
        raise ModelingError(
            f"lock_parallel_baseline_infeasible:"
            f"{scenario.get('feasibility_status')}")
    batch, tokens = int(candidate["b"]), int(result["tokens_per_seq"])
    plan = prepare_fixed_plan(parsed, phase=phase, tokens=tokens,
                              context=context, sequences=batch, scope=scope)
    return {
        "parallel": parsed.runtime.parallel,
        "pp_layer_partition": tuple(parsed.runtime.pp_layer_partition),
        "b": batch, "tokens": tokens,
        "emitted": float(scenario["emitted_tokens_per_seq"]), "plan": plan,
        "plan_identity": locked_plan_identity(plan),
        "baseline_status": scenario.get("feasibility_status"),
    }


def validate_locked_arguments(parsed: CaseSpec, *, candidates, fixed_parallel,
                              num_devices, candidate_tps, ep_candidates,
                              pp_candidates, pcp_candidates) -> None:
    base = parsed.runtime.parallel
    for label, value in (("fixed_parallel", fixed_parallel),
                         ("candidates", candidates)):
        if value is not None:
            specs = [value] if label == "fixed_parallel" else list(value)
            if len(specs) != 1 or ParallelSpec.from_mapping(
                    specs[0]) != base:
                raise ModelingError(
                    f"lock_parallel_{label}_conflicts_with_baseline")
    devices = (num_devices if num_devices is not None
               else len(parsed.machine.dies))
    if devices != base.world_size:
        raise ModelingError(
            f"lock_parallel_num_devices_conflicts_with_baseline:{num_devices}")
    for name, value, domain in (("tp", base.tp, candidate_tps),
                                ("ep", base.ep, ep_candidates),
                                ("pp", base.pp, pp_candidates),
                                ("pcp", base.pcp, pcp_candidates)):
        if domain is not None and value not in {int(item) for item in domain}:
            raise ModelingError(
                f"lock_parallel_{name}_candidates_conflict_with_baseline:{value}")


def locked_row(cand_case: CaseSpec, locked: Mapping[str, object], *,
               sla_ms: float, output_budget: int, devices: int,
               ranks_per_node: int) -> dict[str, object]:
    geometry = dict(locked["plan"]["geometry"])
    spec, batch = locked["parallel"], int(locked["b"])
    analysis = evaluate_fixed_plan(
        cand_case, plan=locked["plan"], include_capacity=True,
        output_budget=output_budget,
        emitted_tokens_per_seq=float(locked["emitted"]))
    capacity = analysis["capacity"]
    step = float(analysis["graph_duration_ms"])
    b_vram, sla_ok = int(capacity["b_vram_max_configured"]), step <= sla_ms
    feasible = bool(sla_ok and batch <= b_vram and capacity["is_vram_feasible"])
    tps = ((batch * spec.dp * float(locked["emitted"]) * MS_PER_S / step)
           if feasible and step > 0.0 else 0.0)
    probe = max(1, int(geometry["context"]))
    memory = Memory(cand_case, workspace_tokens=1)
    candidate = {
        "b": batch, "step_ms": step, "feasible": feasible,
        "source": "retained_baseline_plan", "input_boundary": {
            **geometry, "output_budget": output_budget,
            "emitted_tokens_per_seq": locked["emitted"]},
        "plan": {"kind": "prepared_fixed_plan",
                 "identity": locked["plan_identity"],
                 "selection": "baseline_case"},
    }
    return {
        "parallel": {key: int(getattr(spec, key))
                     for key in ("dp", "tp", "ep", "pp", "pcp")},
        "parallel_spec": parallel_report(spec),
        "pp_layer_partition": list(locked["pp_layer_partition"]),
        "status": ("feasible" if feasible
                   else "sla_or_vram_exceeded_at_retained_batch"),
        "feasible": feasible, "plan_mode": "fixed_parallel",
        "evaluation_stage": LOCKED_STAGE, "is_screening_estimate": False,
        "evaluation_boundary":
            "complete_prepared_plan_at_retained_parallel_and_batch",
        "reason": None if feasible else (
            "retained_batch_sla_exceeded" if not sla_ok
            else "retained_batch_vram_exceeded"),
        "rejection_reason": None if feasible else (
            "retained_batch_sla_exceeded" if not sla_ok
            else "retained_batch_vram_exceeded"),
        "num_devices": int(devices), "ranks_per_node": int(ranks_per_node),
        "weight_bytes_per_die": float(max(memory.weights)),
        "kv_bytes_per_token_per_die": float(
            max(memory.cache_bytes(probe, padded=False)) / probe),
        "b_sla": batch if sla_ok else 0,
        "b_sla_lower_bound": batch if sla_ok else 0,
        "b_sla_is_proven_max": False, "b_vram_max": b_vram,
        "b_feasible": batch if feasible else 0,
        "b_retained": batch, "b_evaluated": batch,
        "batch_geometry": {"b": batch, "context": int(geometry["context"]),
                           "tokens": int(geometry["tokens"]),
                           "scope": geometry["scope"],
                           "selection": "retained_baseline_batch"},
        "step_ms_at_b1": None, "step_ms_at_retained_b": step,
        "tps_at_b1": None,
        "latency_candidate": candidate if sla_ok else None,
        "feasible_candidate": candidate if feasible else None,
        "feasibility_status": ("feasible_retained_batch" if feasible
                               else "retained_batch_infeasible"),
        "search": {"method": "retained_baseline_batch_no_reselection",
                   "b_sla_is_proven_max": False,
                   "evaluated_candidates": [batch]},
        "total_feasible_concurrency": (batch if feasible else 0) * spec.dp,
        "step_ms_feasible": step if feasible else None,
        "tps_feasible": tps, "tps_per_die": tps / float(devices),
        "network_mbu": analysis.get("network_mbu"),
        "network_cube_mfu": analysis.get("network_cube_mfu"),
        "network_vec_mfu": analysis.get("network_vec_mfu"),
        "network_link_util": analysis.get("network_link_util"),
        "plan_identity": locked["plan_identity"],
        "metric_provenance":
            "one_prepared_baseline_plan_and_batch_geometry_per_hardware_variant",
    }


__all__ = ["locked_row", "prepare_locked_plan", "validate_locked_arguments"]
