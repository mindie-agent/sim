"""Normative plan identity, diversity ordering and explicit result accounting.

[Role]
  Enumerate legal parallel/topology/fusion/graph/recipe options, preserve
  caller-specified finite domains, and return a deterministic topology-diverse
  evaluation order when the caller supplies an explicit simulation budget.

[Invariants]
  * World size is dp*pp*tp; EP lives inside the dp*tp domain.
  * No architecture-size special case drops candidates.  Any budget is a
    caller-visible count; generated, prefiltered, attempted, feasible,
    successful and unexplored counts use the same denominator.
  * Standalone draft PP capability is explicit (PP=1), not inferred from a
    model name at evaluation time.

[Timeouts & Locks]
  Pure combinatorial enumeration bounded by declared domains.

[Side-Effect Policy]
  No I/O, device, network or global mutation; returns immutable plan records.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Mapping

from .a3_semantics import SemanticWork
from .a3_types import MachineSpec, ModelSpec, ParallelSpec, WorkloadSpec


@dataclass(frozen=True)
class PlanOption:
    tp: int = 1
    ep: int = 1
    dp: int = 1
    pp: int = 1
    fusion: bool = False
    graph_padding: bool = False
    moe_recipe: str = "dense"
    batch_tokens: int = 0
    worker_count: int = 1

    @property
    def devices(self) -> int:
        return self.dp * self.tp * self.pp

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp, "ep": self.ep, "dp": self.dp, "pp": self.pp,
            "fusion": self.fusion, "graph_padding": self.graph_padding,
            "moe_recipe": self.moe_recipe, "batch_tokens": self.batch_tokens,
            "worker_count": self.worker_count,
            "devices": self.devices,
            "ep_domain": self.dp * self.tp,
        }


@dataclass(frozen=True)
class ShouldResult:
    status: str
    hardware_lower_bound_ms: float | None
    feasible_target_ms: float | None
    gap_ms: float | None
    gap_ratio: float | None
    optimal_in_family: bool
    plan_family_size: int
    evaluated_plans: int
    selected_plan: PlanOption | None
    plans: tuple[dict[str, Any], ...]
    work: SemanticWork
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "purpose": "normative_should_cost",
            "hardware_lower_bound_ms": self.hardware_lower_bound_ms,
            "feasible_target_ms": self.feasible_target_ms,
            "gap_ms": self.gap_ms,
            "gap_ratio": self.gap_ratio,
            "optimal_in_family": self.optimal_in_family,
            "plan_family_size": self.plan_family_size,
            "evaluated_plans": self.evaluated_plans,
            "selected_plan": self.selected_plan.to_dict() if self.selected_plan else None,
            "plans": list(self.plans),
            "semantic_work": self.work.to_dict(),
            "reason": self.reason,
            "target_kind": "best_known_feasible_target_in_enumerated_family",
            "not_global_lower_bound": True,
            "predictive_not_used_as_input": True,
            "service_rate_basis": "ideal_peak_service_if_witness_is_run; not a measured A3 rate",
            "timeline_basis": "intrinsic_service_time_origin_zero",
            "arrival_policy": "request arrivals are normalized away in normative U; predictive keeps wall arrivals",
        }



def _plan_parallel(self: PlanOption) -> ParallelSpec:
    return ParallelSpec(dp=self.dp, tp=self.tp, ep=self.ep, pp=self.pp)


# Attach a small helper only for plan validation; it is not part of the public
# data contract and never changes the plan identity.
PlanOption.parallel = property(_plan_parallel)  # type: ignore[attr-defined]


def _candidate_plans(
    model: ModelSpec, machine: MachineSpec, workload: WorkloadSpec,
    constraints: Mapping[str, Any] | None = None,
) -> list[PlanOption]:
    raw = dict(constraints or {})
    devices = len(machine.dies)
    max_layers = model.layers
    if raw.get("tp"):
        tps = [int(tp) for tp in raw["tp"]]
    else:
        tps = [tp for tp in (1, 2, 4, 8, min(devices, model.num_heads))
               if 1 <= tp <= devices and model.num_heads % tp == 0]
    if raw.get("pp"):
        pps = [int(pp) for pp in raw["pp"]]
    elif model.is_standalone_draft:
        pps = [1]
    else:
        pps = [pp for pp in (1, 2, 4, 8, min(devices, max_layers))
               if 1 <= pp <= devices and max_layers % pp == 0]
    if model.is_standalone_draft:
        pps = [pp for pp in pps if pp == 1]
    max_workers = min(
        int(getattr(machine.dies[0], "aic_count", 0) or 1) if machine.dies else 1,
        int(getattr(machine.dies[0], "aiv_count", 0) or 1) if machine.dies else 1,
    )
    if raw.get("worker_count") is not None:
        worker_options = sorted({
            int(value) for value in raw["worker_count"]
        })
        worker_options = [
            value for value in worker_options
            if 1 <= value <= max(1, max_workers)
        ]
        physical_pools = bool(machine.dies and all(
            float(getattr(machine.dies[0], key, 0) or 0) > 0
            for key in ("l1_bytes", "ub_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes")
        ))
        if not physical_pools:
            worker_options = [1]
    else:
        physical_pools = bool(machine.dies and all(
            float(getattr(machine.dies[0], key, 0) or 0) > 0
            for key in ("l1_bytes", "ub_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes")
        ))
        worker_options = sorted({
            value for value in (1, 2, 4, 8)
            if value <= max(1, max_workers)
        }) or [1]
        if not physical_pools:
            worker_options = [1]
    plans: list[PlanOption] = []
    for tp, pp in itertools.product(tps, pps):
        if tp * pp > devices:
            continue
        max_dp = max(1, devices // (tp * pp))
        dp_options = (
            [int(dp) for dp in raw["dp"]] if raw.get("dp") else
            sorted({1, max_dp, min(2, max_dp), min(4, max_dp)})
        )
        for dp in dp_options:
            if dp < 1 or dp * tp * pp > devices:
                continue
            ep_options = [1]
            if model.ffn == "moe":
                domain = dp * tp
                if raw.get("ep"):
                    ep_options = [int(ep) for ep in raw["ep"]]
                else:
                    ep_options = sorted({1, min(2, domain), min(4, domain),
                                         min(8, domain), domain})
                ep_options = [
                    ep for ep in ep_options
                    if 1 <= ep <= domain and domain % ep == 0
                    and model.num_experts % ep == 0
                ]
            for ep in ep_options:
                for fusion in (False, True):
                    for graph in (False, True):
                        recipes = ["dense"] if model.ffn == "dense" else ["alltoall", "mc2"]
                        for recipe in recipes:
                            workers = worker_options if fusion else [1]
                            for worker_count in workers:
                                plans.append(PlanOption(
                                    tp=tp, ep=ep, dp=dp, pp=pp, fusion=fusion,
                                    graph_padding=graph, moe_recipe=recipe,
                                    batch_tokens=max(
                                        1,
                                        workload.requests[0].prompt_tokens
                                        + workload.requests[0].output_tokens,
                                    ),
                                    worker_count=worker_count,
                                ))
    dedup: dict[tuple[Any, ...], PlanOption] = {}
    for plan in plans:
        dedup[(
            plan.tp, plan.ep, plan.dp, plan.pp, plan.fusion,
            plan.graph_padding, plan.moe_recipe, plan.worker_count,
        )] = plan
    result = list(dedup.values())
    if constraints:
        for key in ("tp", "ep", "dp", "pp"):
            if constraints.get(key) is not None:
                allowed = {int(item) for item in constraints[key]}
                result = [plan for plan in result if getattr(plan, key) in allowed]
        if constraints.get("fusion") is not None:
            raw_fusion = constraints["fusion"]
            allowed_fusion = {
                bool(value) for value in (
                    raw_fusion if isinstance(raw_fusion, (list, tuple, set))
                    else (raw_fusion,)
                )
            }
            result = [plan for plan in result if plan.fusion in allowed_fusion]
        if constraints.get("graph_padding") is not None:
            result = [plan for plan in result if plan.graph_padding == bool(constraints["graph_padding"])]
        if constraints.get("moe_recipe") is not None:
            result = [plan for plan in result if plan.moe_recipe == str(constraints["moe_recipe"])]
        if constraints.get("worker_count") is not None:
            allowed_workers = {int(item) for item in constraints["worker_count"]}
            result = [
                plan for plan in result if plan.worker_count in allowed_workers
            ]
    return result




__all__ = ["PlanOption", "ShouldResult", "_candidate_plans"]
