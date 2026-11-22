"""Deterministic candidate identity, static screening and cached sweep evaluation.

[Role] Turn one declared case and a candidate parallel spec into a complete
  sweep row, rejecting structurally invalid or capacity-infeasible candidates
  before any graph lowering, and reusing homogeneous replica results.
[Invariants] A candidate id is a pure function of model/runtime, hardware,
  phase boundary and the complete parallel spec. Static rejection happens
  before ``target_graphs``. A replica-equivalent projection scales only DP
  totals; per-replica step latency, capacity and weight bytes stay identical.
  Cache entries carry no mutable Engine or ledger state.
[Timeouts & Locks]
  Cooperative monotonic deadlines only; no signals, threads or subprocesses.
[Side-Effect Policy]
  Pure evaluation; the caller owns JSONL streaming and checkpoint repair.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import math
import time
from typing import Any

from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    DEFAULT_SLA_SCENARIOS, _resolve_speculative_tokens,
    _vram_max_sequences_per_dp, capacity_window, solve_sla_sizing,
)
from mindie_sim.prediction.evaluation import (
    ACTIVE_DEADLINE_MONO, EvaluationContext, check_active_deadline,
)
from mindie_sim.prediction.sharding.local_tp import (
    local_tp_spans_dp, validate_local_projection_tp,
)
from mindie_sim.prediction.structure import (
    _PARSE_MEMO, _PARSE_MEMO_LIMIT, parse_case,
)
from mindie_sim.catalog.weights import stage_weight_bytes
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import (
    ParallelSpec, local_kv_heads, local_query_heads,
)
from mindie_sim.sweep.resize import (
    check_pos_float, check_pos_int, enumerate_parallel_candidates,
    materialize_machine_for_devices, resolve_ranks_per_node,
)

CANDIDATE_ID_HEX_LEN = 32
CANDIDATE_ID_KEY = "candidate_id"
STATIC_STAGE = "static_validation"
B1_STAGE = "b1_screen"
FULL_STAGE = "full_sla"
LOCKED_STAGE = "locked_plan"
TIMEOUT_STAGE = "timeout"
DEFAULT_MAX_SEQUENCES = 256
LOCKED_PROGRAM_ERROR = "fixed_plan_sweep_not_implemented"
MS_PER_S = 1000.0
PARETO_EPS = 1e-12
SCREEN_BATCH = 1
BASELINE_SCENARIO: tuple[dict[str, Any], ...] = (
    {"id": "baseline", "label": "1x Baseline", "hbm_bw_scale": 1.0,
     "cube_flops_scale": 1.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=repr, ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _runtime_features(runtime: Any) -> tuple:
    return (
        bool(getattr(runtime, "engram_cpu_offload", False)),
        bool(getattr(runtime, "multistream_overlap_shared_expert", False)),
        bool(getattr(runtime, "enable_dbo", False)),
        str(dict(getattr(runtime, "metadata", ()) or {}).get("moe_comm_mode", "")),
        str(getattr(runtime, "graph_capture_mode", "none")),
        str(getattr(runtime, "collective_startup_mode", "legacy")),
        bool(getattr(runtime, "async_scheduling", False)),
    )


def _derivation(parsed: CaseSpec) -> dict[str, Any] | None:
    assumptions = dict(getattr(parsed.machine, "assumptions", ()) or ())
    base = assumptions.get("derivation.base")
    if not base:
        return None
    return {"base": str(base),
            **{str(key): value for key, value in assumptions.items()
               if str(key).startswith("derivation.")}}


def compute_sweep_candidate_id(
    parsed: CaseSpec, cand_p: ParallelSpec, *, cand_devices: int,
    ranks_per_node: int, phase: str, scope: str, context: int,
    q_tokens: int, emit_tokens: float, output_budget: int, sla_ms: float,
    max_sequences_per_dp: int, lock_parallel: bool,
) -> str:
    """Deterministic 32-hex identity before any static or graph evaluation."""
    model = parsed.model
    runtime = parsed.runtime
    model_runtime = {
        "model": model.to_dict(),
        "runtime": runtime.to_dict(),
        "workload": parsed.workload.to_dict(),
    }
    derivation = _derivation(parsed)
    base_sku = (str(derivation["base"]) if derivation
                else str(parsed.machine.sku))
    derived_sku = str(parsed.machine.sku)
    die_digest = _digest(parsed.machine.to_dict())
    hardware_identity = {
        "base_sku": base_sku, "derived_sku": derived_sku,
        "derivation": derivation, "num_devices": int(cand_devices),
        "ranks_per_node": int(ranks_per_node), "die_digest": die_digest,
    }
    phase_boundary = {
        "phase": phase, "scope": scope, "context": int(context),
        "tokens_per_seq": int(q_tokens),
        "emitted_tokens_per_seq": float(emit_tokens),
        "output_budget": int(output_budget), "sla_ms": float(sla_ms),
        "max_sequences_per_dp": int(max_sequences_per_dp),
        "lock_parallel": bool(lock_parallel),
    }
    return _digest({
        "model_runtime_digest": _digest(model_runtime),
        "hardware_identity": hardware_identity,
        "phase_boundary": phase_boundary,
        "parallel_spec": parallel_report(cand_p),
    })[:CANDIDATE_ID_HEX_LEN]


def parallel_report(spec: ParallelSpec) -> dict[str, Any]:
    return {**spec.to_dict(), "dcp": spec.dcp}


def _parallel_short(spec: ParallelSpec) -> dict[str, int]:
    result = {key: int(getattr(spec, key))
              for key in ("dp", "tp", "ep", "pp", "pcp")}
    if spec.dcp > 1:
        result["dcp"] = int(spec.dcp)
    for axis in (
        "q_proj_tp", "o_proj_tp", "linear_proj_tp", "dcp_linear_tp",
        "mlp_tp", "shared_expert_tp", "lm_head_tp",
    ):
        val = int(getattr(spec, axis, 1))
        if val > 1:
            result[axis] = val
    return result


def kv_replication_fields(model: ModelSpec, cand_p: ParallelSpec) -> dict[str, Any]:
    """Transparent non-blocking KV-cache replication metadata for sweep rows."""
    from mindie_sim.catalog.context_parallel import kv_cache_replication_info
    from mindie_sim.public.errors import UnmodeledCapabilityError

    try:
        info = kv_cache_replication_info(model, cand_p)
    except UnmodeledCapabilityError:
        raise
    except (ModelingError, ValueError):
        return {
            "kv_cache_replicated_tp": False,
            "kv_cache_replication_factor": 1,
            "canonical_kv_sharding": "invalid",
        }
    return {
        "kv_cache_replicated_tp": bool(info["kv_cache_replicated_tp"]),
        "kv_cache_replication_factor": int(info["kv_cache_replication_factor"]),
        "canonical_kv_sharding": str(info["canonical_kv_sharding"]),
    }


def compute_pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark exact multi-objective dominance on (tps_per_die, concurrency, -step)."""
    feasible = [row for row in rows
                if row.get("feasible") and _pareto_step(row) is not None]
    for row in rows:
        step = _pareto_step(row)
        if not row.get("feasible") or step is None:
            row["pareto_optimal"] = False
            continue
        dominated = False
        for other in feasible:
            if other is row:
                continue
            other_step = _pareto_step(other)
            ge_all = (
                float(other["tps_per_die"]) >= float(row["tps_per_die"]) - PARETO_EPS
                and int(other["total_feasible_concurrency"])
                >= int(row["total_feasible_concurrency"])
                and other_step <= step + PARETO_EPS
            )
            strict = (
                float(other["tps_per_die"]) > float(row["tps_per_die"]) + PARETO_EPS
                or int(other["total_feasible_concurrency"])
                > int(row["total_feasible_concurrency"])
                or other_step < step - PARETO_EPS
            )
            if ge_all and strict:
                dominated = True
                break
        row["pareto_optimal"] = not dominated
    return [row for row in rows if row.get("pareto_optimal")]


def _pareto_step(row: Mapping[str, Any]) -> float | None:
    value = row.get("step_ms_at_b1")
    if value is None:
        value = row.get("step_ms_at_retained_b")
    return None if value is None else float(value)


class SweepEvaluationCache:
    """Zero-mutation cross-candidate reuse accounting."""

    def __init__(self) -> None:
        self.model_spec_reuses = 0
        self.machine_materialization_hits = 0
        self.machine_materialization_misses = 0
        self.static_topology_rejections = 0
        self.static_capacity_rejections = 0
        self.evaluation_context_hits = 0
        self.evaluation_context_misses = 0
        self.replica_equivalence_hits = 0
        self.replica_equivalence_misses = 0
        self.prepared_program_hits = 0
        self.prepared_program_misses = 0
        self._machines: dict[tuple[str, int], Any] = {}
        self._replica: dict[tuple, dict[str, Any]] = {}

    def machine(self, derivation_key: str, devices: int, factory):
        key = (derivation_key, int(devices))
        if key in self._machines:
            self.machine_materialization_hits += 1
            return self._machines[key]
        self.machine_materialization_misses += 1
        value = factory()
        self._machines[key] = value
        return value

    def replica_result(self, key: tuple) -> dict[str, Any] | None:
        value = self._replica.get(key)
        if value is None:
            self.replica_equivalence_misses += 1
        else:
            self.replica_equivalence_hits += 1
        return value

    def store_replica_result(self, key: tuple, row: dict[str, Any]) -> None:
        self._replica[key] = dict(row)

    def stats(self) -> dict[str, int]:
        return {
            "model_spec_reuses": int(self.model_spec_reuses),
            "machine_materialization_hits": int(self.machine_materialization_hits),
            "machine_materialization_misses": int(self.machine_materialization_misses),
            "static_topology_rejections": int(self.static_topology_rejections),
            "static_capacity_rejections": int(self.static_capacity_rejections),
            "evaluation_context_hits": int(self.evaluation_context_hits),
            "evaluation_context_misses": int(self.evaluation_context_misses),
            "replica_equivalence_hits": int(self.replica_equivalence_hits),
            "replica_equivalence_misses": int(self.replica_equivalence_misses),
            "prepared_program_hits": int(self.prepared_program_hits),
            "prepared_program_misses": int(self.prepared_program_misses),
        }


def _local_head_validation(model: ModelSpec, cand_p: ParallelSpec,
                           mla_replicate: bool) -> None:
    if mla_replicate:
        return
    local_query_heads(model.num_heads, cand_p.tp)
    for plan in model.layer_plans:
        if int(plan.num_heads) > 0:
            local_query_heads(plan.num_heads, cand_p.tp)
        if plan.attention == "gqa" and int(plan.num_kv_heads) > 0:
            local_kv_heads(plan.num_kv_heads, cand_p.tp)
        if plan.attention in {"gdn", "kda"}:
            linear_heads = int(getattr(plan, "linear_value_heads", 0)
                               or getattr(plan, "linear_heads", 0) or 1)
            if linear_heads % cand_p.tp != 0:
                raise ModelingError(
                    f"linear_state_heads_not_divisible_by_tp:"
                    f"{linear_heads}%{cand_p.tp}")


def validate_candidate_statically(
    parsed: CaseSpec, cand_p: ParallelSpec, *,
    cand_devices: int, cand_machine: Any, part: Sequence[int] | None,
    phase: str, context: int, q_tokens: int, emit_tokens: float,
    output_budget: int, max_sequences_per_dp: int,
) -> dict[str, Any]:
    """Cheap topology/capacity rejection before any graph lowering."""
    if cand_p.world_size != cand_devices:
        raise ModelingError(
            f"parallel_world_size_mismatch:{cand_p.world_size}!={cand_devices}")
    model = parsed.model
    mla_replicate = bool(dict(getattr(model, "metadata", ()) or {})
                         .get("mla_replicate", False))
    _local_head_validation(model, cand_p, mla_replicate)
    cand_p.validate(model, layer_partition=part or None)
    cand_machine.validate_placement(cand_p)
    if cand_p.has_local_tp:
        cand_case = replace(parsed, machine=cand_machine,
                            runtime=replace(parsed.runtime, parallel=cand_p,
                                            pp_layer_partition=tuple(part or ())))
        validate_local_projection_tp(cand_case)
    runtime = parsed.runtime
    include_draft = model.auxiliary is not None
    weights = stage_weight_bytes(
        model, cand_p, cand_machine.ranks_per_node,
        include_draft=include_draft,
        include_engram_tables=not bool(getattr(runtime, "engram_cpu_offload", False)),
        layer_partition=part or None,
        enable_shared_expert_dp=bool(dict(runtime.metadata).get(
            "enable_shared_expert_dp", False)),
    )
    usable = min(float(die.memory_bytes) for die in cand_machine.dies) \
        * float(runtime.gpu_memory_utilization)
    if weights and max(weights) > usable:
        return {
            "ok": False, "kind": "capacity",
            "reason": "capacity_exceeded:weights_exceed_usable_hbm",
            "weight_bytes": float(max(weights)),
            "memory": None, "b_vram_max_joint": 0,
        }
    cand_case = replace(parsed, machine=cand_machine,
                        runtime=replace(parsed.runtime, parallel=cand_p,
                                        pp_layer_partition=tuple(part or ())))
    memory = Memory(cand_case, workspace_tokens=max(1, int(q_tokens)))
    if min(capacity for row in memory.capacity for capacity in row) < 0.0:
        return {
            "ok": False, "kind": "capacity",
            "reason": "capacity_exceeded:weights_and_workspace_exceed_usable_hbm",
            "weight_bytes": float(max(memory.weights)),
            "memory": memory, "b_vram_max_joint": 0,
        }
    b1 = capacity_window(
        cand_case, phase=phase, context=context, tokens_per_seq=q_tokens,
        sequences=1, output_budget=output_budget,
        emitted_tokens_per_seq=emit_tokens, memory=memory)
    decode_q, decode_emit = _resolve_speculative_tokens(
        cand_case, output_budget=output_budget, tokens_per_seq=None,
        emitted_tokens_per_seq=None)
    if phase == "prefill":
        lifecycle = capacity_window(
            cand_case, phase="decode", context=context + int(q_tokens),
            tokens_per_seq=decode_q, sequences=1, output_budget=output_budget,
            emitted_tokens_per_seq=decode_emit, memory=memory)
        fits = bool(b1["fits"] and lifecycle["fits"])
    else:
        fits = bool(b1["fits"])
    if not fits:
        return {
            "ok": False, "kind": "capacity",
            "reason": "capacity_exceeded:b1_kv_or_lifecycle_state_exceeds_hbm",
            "weight_bytes": float(max(memory.weights)),
            "memory": memory, "b_vram_max_joint": 0,
        }
    b_sync = _vram_max_sequences_per_dp(
        cand_case, context=context, tokens_per_seq=q_tokens,
        output_budget=output_budget, emitted_tokens_per_seq=emit_tokens,
        async_scheduling=False, phase=phase, memory=memory)
    b_async = _vram_max_sequences_per_dp(
        cand_case, context=context, tokens_per_seq=q_tokens,
        output_budget=output_budget, emitted_tokens_per_seq=emit_tokens,
        async_scheduling=True, phase=phase, memory=memory)
    b_joint = min(b_sync, b_async)
    if phase == "prefill":
        life_sync = _vram_max_sequences_per_dp(
            cand_case, context=context + int(q_tokens),
            tokens_per_seq=decode_q, output_budget=output_budget,
            emitted_tokens_per_seq=decode_emit, async_scheduling=False,
            phase="decode")
        life_async = _vram_max_sequences_per_dp(
            cand_case, context=context + int(q_tokens),
            tokens_per_seq=decode_q, output_budget=output_budget,
            emitted_tokens_per_seq=decode_emit, async_scheduling=True,
            phase="decode")
        b_joint = min(b_joint, life_sync, life_async)
    return {
        "ok": b_joint > 0, "kind": "ok" if b_joint > 0 else "capacity",
        "reason": None if b_joint > 0 else
        "capacity_exceeded:b1_kv_or_lifecycle_state_exceeds_hbm",
        "weight_bytes": float(max(memory.weights)),
        "memory": memory,
        "kv_bytes_per_token": float(max(memory.cache_bytes(
            max(1, int(context)), padded=False))) / max(1, int(context)),
        "b_vram_max_joint": max(0, int(min(b_joint, max_sequences_per_dp))),
    }


def _row_common(*, candidate_id: str, cand_p: ParallelSpec, devices: int,
                ranks_per_node: int, phase: str, scope: str, context: int,
                output_budget: int, emit_tokens: float, sla_ms: float,
                two_stage: bool) -> dict[str, Any]:
    return {
        CANDIDATE_ID_KEY: candidate_id,
        "parallel": _parallel_short(cand_p),
        "parallel_spec": parallel_report(cand_p),
        "num_devices": int(devices), "ranks_per_node": int(ranks_per_node),
        "sweep_meta": {
            "num_devices": int(devices), "sla_ms": float(sla_ms),
            "context": int(context), "phase": phase, "scope": scope,
            "output_budget": int(output_budget),
            "emitted_tokens_per_seq": float(emit_tokens),
            "two_stage": bool(two_stage),
        },
    }


def static_row(common: Mapping[str, Any], *, reason: str, kind: str,
               elapsed_ms: float) -> dict[str, Any]:
    capacity = kind == "capacity"
    return {
        **dict(common), "status": "infeasible", "feasible": False,
        "reason": reason, "rejection_reason": reason,
        "evaluation_stage": STATIC_STAGE, "is_screening_estimate": False,
        "plan_mode": "reselected_parallel",
        "evaluation_boundary": "candidate_rejected_before_graph_evaluation",
        "feasibility_status": ("capacity_infeasible" if capacity
                               else "infeasible_before_evaluation"),
        "weight_bytes_per_die": None, "kv_bytes_per_token_per_die": None,
        "b_sla": 0, "b_vram_max": 0, "b_feasible": 0,
        "b_sla_is_proven_max": False, "total_feasible_concurrency": 0,
        "latency_candidate": None, "feasible_candidate": None,
        "step_ms_at_b1": None, "step_ms_feasible": None,
        "tps_at_b1": 0.0, "tps_feasible": 0.0, "tps_per_die": 0.0,
        "network_mbu": None, "network_cube_mfu": None,
        "network_vec_mfu": None, "network_link_util": None,
        "search": {"method": ("static_capacity_precheck" if capacity
                              else "static_topology_precheck"),
                   "evaluated_candidates": [], "b_sla_is_proven_max": False},
        "elapsed_ms": float(elapsed_ms),
    }


def b1_screen(parsed: CaseSpec, cand_case: CaseSpec, cand_p: ParallelSpec,
              *, common: Mapping[str, Any], ctx: EvaluationContext,
              phase: str, scope: str, context: int, q_tokens: int,
              emit_tokens: float, sla_ms: float, b_vram_max: int,
              max_sequences_per_dp: int, output_budget: int = 0,
              static: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Rank one candidate by its isolated operator sum; never run the graph.

    The isolated operator sum ignores multi-stream overlap, so it cannot
    reject a candidate on SLA before finalist evaluation.  It is a screening
    estimate only; ``full_row`` owns every SLA and capacity verdict.
    """
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.body_defaults import consume_target_declaration
    from mindie_sim.prediction.efficiency import _fast_operator_and_graph_reports

    static = dict(static or {})
    start = time.monotonic()
    check_active_deadline()
    graphs = target_graphs(
        cand_case, phase=phase, tokens=q_tokens, context=context,
        sequences=SCREEN_BATCH, scope=scope)
    effective = consume_target_declaration(cand_case, graphs)
    # The precomputed declaration keeps ``_fast_operator_and_graph_reports``
    # from re-entering ``_lower_and_run_graph``'s kernel-group branch.
    operators, _ = _fast_operator_and_graph_reports(
        effective, phase=phase, tokens=q_tokens, context=context,
        sequences=SCREEN_BATCH, scope=scope, run_graph=False,
        precomputed_graph=(graphs, effective, None), evaluation_context=ctx)
    step = float(operators["operator_sum_ms"])
    tps = (cand_p.dp * float(emit_tokens) * MS_PER_S / step
           if step > 0.0 else 0.0)
    eff_max = max(0, min(int(max_sequences_per_dp), int(b_vram_max)))
    return {
        **dict(common), "status": "screened", "feasible": False,
        "reason": None, "rejection_reason": None,
        "evaluation_stage": B1_STAGE, "is_screening_estimate": True,
        "evaluation_boundary": "b1_isolated_operator_sum",
        "b1_feasible": True, "b1_sla_pass": True,
        "b1_pareto_optimal": False,
        "feasibility_status": "b1_screen_pass_pending_full_sla",
        "plan_mode": "reselected_parallel",
        "weight_bytes_per_die": static.get("weight_bytes"),
        "kv_bytes_per_token_per_die": static.get("kv_bytes_per_token"),
        "b_sla": 0, "b_vram_max": int(b_vram_max), "b_feasible": 0,
        "b_sla_is_proven_max": False, "total_feasible_concurrency": 0,
        "latency_candidate": None, "feasible_candidate": None,
        "step_ms_at_b1": step, "step_ms_feasible": None,
        "tps_at_b1": tps, "tps_screen_b1": tps,
        "tps_feasible": 0.0, "tps_per_die": 0.0,
        "max_concurrency_bound": eff_max * int(cand_p.dp),
        "upper_tps_per_die": (eff_max * tps / float(common["num_devices"])
                              if common["num_devices"] else 0.0),
        "operator_sum_screen_ms": step,
        "graph_step_ms": None,
        "graph_ttft_ms": (step if (phase, scope) == ("prefill", "model")
                          else None),
        "network_mbu": None, "network_cube_mfu": None,
        "network_vec_mfu": None, "network_link_util": None,
        "search": {"method": "b1_isolated_operator_sum",
                   "evaluated_candidates": [SCREEN_BATCH],
                   "b_sla_is_proven_max": False},
        "elapsed_ms": (time.monotonic() - start) * MS_PER_S,
    }


def full_row(parsed: CaseSpec, cand_case: CaseSpec, cand_p: ParallelSpec,
             *, common: Mapping[str, Any], ctx: EvaluationContext,
             phase: str, scope: str, context: int, q_tokens: int,
             emit_tokens: float, sla_ms: float, output_budget: int,
             max_sequences_per_dp: int, b_vram_max: int,
             static: Mapping[str, Any], screening: bool = False) -> dict[str, Any]:
    start = time.monotonic()
    check_active_deadline()
    result = solve_sla_sizing(
        cand_case, sla_ms=sla_ms, context=context, phase=phase, scope=scope,
        tokens_per_seq=q_tokens, output_budget=output_budget,
        max_sequences_per_dp=max_sequences_per_dp,
        scenarios=BASELINE_SCENARIO, evaluation_context=ctx)
    scen = result["scenarios"][0]
    candidate = scen.get("feasible_candidate")
    feasible = candidate is not None
    b_sla = int(scen["b_sla"])
    b_vram = int(scen["b_vram_max"])
    b_feasible = int(candidate["b"]) if candidate is not None else 0
    tps_feasible = (float(candidate["tps_feasible"])
                    if candidate is not None and "tps_feasible" in candidate
                    else float(scen["tps_feasible"]))
    row = {
        **dict(common),
        "status": "feasible" if feasible else "sla_or_vram_exceeded",
        "feasible": feasible,
        "reason": None if feasible else scen.get("feasibility_status"),
        "rejection_reason": None if feasible else scen.get("feasibility_status"),
        "evaluation_stage": FULL_STAGE, "is_screening_estimate": False,
        "plan_mode": "reselected_parallel",
        "evaluation_boundary": "complete_candidate_graph_and_capacity",
        "feasibility_status": scen.get("feasibility_status"),
        "weight_bytes_per_die": static.get("weight_bytes"),
        "kv_bytes_per_token_per_die": static.get("kv_bytes_per_token"),
        "b_sla": b_sla, "b_vram_max": b_vram, "b_feasible": b_feasible,
        "b_sla_is_proven_max": scen.get("search", {}).get(
            "b_sla_is_proven_max", False),
        "total_feasible_concurrency": b_feasible * int(cand_p.dp),
        "latency_candidate": scen.get("latency_candidate"),
        "feasible_candidate": candidate,
        "step_ms_at_b1": float(scen["step_ms_at_b1"]),
        "step_ms_feasible": (float(candidate["step_ms"])
                             if candidate is not None else None),
        "tps_at_b1": float(scen["tps_at_b1"]),
        "tps_feasible": tps_feasible,
        "tps_per_die": tps_feasible / float(common["num_devices"]),
        "network_mbu": (float(candidate["network_mbu"])
                        if candidate is not None else None),
        "network_cube_mfu": (float(candidate["network_cube_mfu"])
                             if candidate is not None else None),
        "network_vec_mfu": (float(candidate["network_vec_mfu"])
                            if candidate is not None else None),
        "network_link_util": (float(candidate["network_link_util"])
                              if candidate is not None else None),
        "search": scen.get("search"),
        "elapsed_ms": (time.monotonic() - start) * MS_PER_S,
    }
    # ``graph_step_ms`` is the B1 model-graph span for this candidate; neither
    # field claims a serving latency.  ``graph_ttft_ms`` is only the B1
    # model-prefill span.
    row["graph_step_ms"] = row.get("step_ms_at_b1")
    row["graph_ttft_ms"] = (row.get("step_ms_at_b1")
                            if (phase, scope) == ("prefill", "model") else None)
    if screening:
        row["b1_pareto_optimal"] = True
    return row


def select_finalists(
    promoted: Sequence[Mapping[str, Any]], finalist_top_k: int,
) -> list[Mapping[str, Any]]:
    """Global top-K screen ranking across device grids and derivations."""
    ranked = [
        row for row in promoted
        if row.get("operator_sum_screen_ms") is not None
    ]
    ranked.sort(key=lambda row: (float(row["operator_sum_screen_ms"]),
                                 str(row.get(CANDIDATE_ID_KEY, ""))))
    return ranked[:int(finalist_top_k)]


def project_replica_row(base: Mapping[str, Any], *,
                        cand_p: ParallelSpec, devices: int,
                        ranks_per_node: int, candidate_id: str) -> dict[str, Any]:
    """Project a per-replica result to one homogeneous DP cohort."""
    base_dp = max(1, int(dict(base["parallel"]).get("dp", 1)))
    row = dict(base)
    row.update({
        CANDIDATE_ID_KEY: candidate_id,
        "parallel": _parallel_short(cand_p),
        "parallel_spec": parallel_report(cand_p),
        "num_devices": int(devices), "ranks_per_node": int(ranks_per_node),
        "evaluation_stage": FULL_STAGE,
        "is_screening_estimate": False,
    })
    base_b = int(base.get("b_feasible", 0))
    base_tps = float(base.get("tps_feasible", 0.0))
    base_tps_b1 = float(base.get("tps_at_b1", 0.0))
    row["total_feasible_concurrency"] = base_b * int(cand_p.dp)
    row["tps_feasible"] = base_tps / float(base_dp) * int(cand_p.dp)
    row["tps_at_b1"] = base_tps_b1 / float(base_dp) * int(cand_p.dp)
    row["tps_per_die"] = row["tps_feasible"] / float(devices)
    return row


def resume_key(row: Mapping[str, Any]) -> str:
    return str(row.get(CANDIDATE_ID_KEY, ""))


def set_deadline(candidate_timeout_s: float | None,
                 outer_deadline: float | None):
    now = time.monotonic()
    limits = []
    if candidate_timeout_s is not None:
        limits.append(now + float(candidate_timeout_s))
    if outer_deadline is not None:
        limits.append(float(outer_deadline))
    return ACTIVE_DEADLINE_MONO.set(min(limits) if limits else None)


def timeout_row(common: Mapping[str, Any], *, elapsed_s: float,
                candidate_timeout_s: float) -> dict[str, Any]:
    reason = (f"candidate_timeout_exceeded:{elapsed_s:.3f}s"
              f">{float(candidate_timeout_s):.3f}s")
    return {
        **dict(common), "status": "timeout", "feasible": False,
        "evaluation_stage": TIMEOUT_STAGE, "is_screening_estimate": False,
        "feasibility_status": "candidate_timeout",
        "reason": reason, "rejection_reason": reason,
        "b_sla": 0, "b_vram_max": 0, "b_feasible": 0,
        "b_sla_is_proven_max": False, "total_feasible_concurrency": 0,
        "step_ms_at_b1": None, "step_ms_feasible": None,
        "tps_at_b1": 0.0, "tps_feasible": 0.0, "tps_per_die": 0.0,
        "search": {"method": "bounded_candidate_timeout",
                   "evaluated_candidates": [],
                   "b_sla_is_proven_max": False},
        "elapsed_ms": float(elapsed_s) * MS_PER_S,
    }


def build_candidate_case(parsed: CaseSpec, machine: Any,
                         cand_p: ParallelSpec, part: Sequence[int]):
    cand_case = replace(
        parsed, machine=machine,
        runtime=replace(parsed.runtime, parallel=cand_p,
                        pp_layer_partition=tuple(part)))
    _PARSE_MEMO[id(cand_case)] = (cand_case, cand_case)
    while len(_PARSE_MEMO) > _PARSE_MEMO_LIMIT:
        _PARSE_MEMO.popitem(last=False)
    return cand_case


def candidate_devices(raw: Mapping[str, Any], num_devices: int | None) -> int:
    if num_devices is not None:
        return int(num_devices)
    return max(1, int(raw.get("dp", 1)) * int(raw.get("tp", 1))
               * int(raw.get("pp", 1)) * int(raw.get("pcp", 1)))


def enumerate_case_candidates(
    parsed: CaseSpec, *, num_devices: int | None, allow_dp: bool,
    allow_ep: bool, allow_pp: bool, allow_pcp: bool,
    allow_dcp: bool = False,
    candidate_tps: Sequence[int] | None, ep_candidates: Sequence[int] | None,
    pp_candidates: Sequence[int] | None, pcp_candidates: Sequence[int] | None,
    dcp_candidates: Sequence[int] | None = None,
    explicit: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if explicit is not None:
        return [dict(item) for item in explicit]
    devices = (int(num_devices) if num_devices is not None
               else len(parsed.machine.dies))
    return enumerate_parallel_candidates(
        parsed, num_devices=devices, allow_dp=allow_dp, allow_ep=allow_ep,
        allow_pp=allow_pp, allow_pcp=allow_pcp, allow_dcp=allow_dcp,
        candidate_tps=candidate_tps, ep_candidates=ep_candidates,
        pp_candidates=pp_candidates, pcp_candidates=pcp_candidates,
        dcp_candidates=dcp_candidates)


def evaluate_candidate_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Pickle-safe worker evaluating one candidate from plain dict payloads."""
    started = time.monotonic()
    parsed = parse_case(task["case_dict"])
    cand_p = ParallelSpec.from_mapping(task["parallel_spec"])
    cand_devices = int(task["num_devices"])
    part = tuple(task["part"] or ())
    common = dict(task["common"])
    probe = dict(task["probe"])
    machine = materialize_machine_for_devices(parsed, parsed, cand_devices)
    cand_case = build_candidate_case(parsed, machine, cand_p, part)
    from mindie_sim.public.errors import UnmodeledCapabilityError

    try:
        static = validate_candidate_statically(
            parsed, cand_p, cand_devices=cand_devices, cand_machine=machine,
            part=part, phase=probe["phase"], context=probe["context"],
            q_tokens=probe["q_tokens"], emit_tokens=probe["emit_tokens"],
            output_budget=probe["output_budget"],
            max_sequences_per_dp=probe["max_sequences_per_dp"])
    except UnmodeledCapabilityError:
        raise
    except (ModelingError, ValueError) as exc:
        return static_row(
            common, reason=str(exc), kind="topology",
            elapsed_ms=(time.monotonic() - started) * MS_PER_S)
    if not static["ok"]:
        return static_row(
            common, reason=str(static["reason"]), kind=str(static["kind"]),
            elapsed_ms=(time.monotonic() - started) * MS_PER_S)
    ctx = EvaluationContext.create(cand_case)
    row = full_row(
        parsed, cand_case, cand_p, common=common, ctx=ctx,
        phase=probe["phase"], scope=probe["scope"], context=probe["context"],
        q_tokens=probe["q_tokens"], emit_tokens=probe["emit_tokens"],
        sla_ms=probe["sla_ms"], output_budget=probe["output_budget"],
        max_sequences_per_dp=probe["max_sequences_per_dp"],
        b_vram_max=int(static["b_vram_max_joint"]), static=static)
    return row


__all__ = [
    "BASELINE_SCENARIO", "B1_STAGE", "CANDIDATE_ID_HEX_LEN",
    "DEFAULT_MAX_SEQUENCES", "FULL_STAGE", "LOCKED_PROGRAM_ERROR",
    "SCREEN_BATCH",
    "SweepEvaluationCache", "b1_screen", "build_candidate_case",
    "candidate_devices", "compute_pareto_frontier",
    "compute_sweep_candidate_id", "enumerate_case_candidates",
    "evaluate_candidate_worker", "full_row", "kv_replication_fields",
    "parallel_report", "project_replica_row", "select_finalists",
    "set_deadline", "static_row", "timeout_row",
    "validate_candidate_statically",
]
