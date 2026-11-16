"""Axiomatic multi-topology parallel sharding and P/D disaggregation Pareto solver.

[Role] Enumerate structurally valid (DP, TP, EP, PP, PCP) sharding topologies,
  evaluate hardware SLA concurrency and throughput across topology grids, and
  solve disaggregated Prefill/Decode instance ratios under KV transfer and
  TTFT/TPOT SLA constraints.
[Invariants] Strictly closed-world, deterministic, axiomatic evaluation:
  physical weight/KV memory bounds come from Memory, step latencies and SLA
  ceilings come from solve_sla_sizing and _lower_and_run_graph,
  and Pareto optimality uses exact multi-objective dominance without heuristic
  weights or subjective scores. Structurally invalid candidates are recorded as
  infeasible with their exact failure reason and never silently dropped.
[Timeouts & Locks] Pure in-memory arithmetic and bounded DAG evaluation; no I/O,
  subprocesses, sleeps, or device contexts.
[Side-Effect Policy] Input cases and mappings are never mutated; returns new
  immutable or freshly constructed dictionaries.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import math
from typing import Any

from mindie_sim.backends.ascend import materialize as ascend_materialize, profiles as ascend_profiles
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    DEFAULT_SLA_SCENARIOS,
    _lower_and_run_graph,
    _resolve_speculative_tokens,
    solve_sla_sizing,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec

_PARETO_EPS = 1e-12
ZERO = 0
ONE = 1
FIRST = 0
DEFAULT_SLA_MS = 30.0
DEFAULT_CONTEXT = 126000
DEFAULT_OUTPUT_BUDGET = 1024
DEFAULT_MAX_SEQUENCES = 256
DEFAULT_PROMPT_TOKENS = 4096
DEFAULT_OUTPUT_TOKENS = 1024
DEFAULT_TTFT_SLA_MS = 3000.0
DEFAULT_KV_TRANSFER_BW_GBPS = 100.0
ONE_FLOAT = 1.0
_BASELINE_SCENARIO: tuple[dict[str, Any], ...] = (
    {"id": "baseline", "label": "1x Baseline", "hbm_bw_scale": 1.0, "cube_flops_scale": 1.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
)
_TRANSFER_OVERRIDE_KEYS = (
    "transfer_channels", "inter_node_bw_gbps", "intra_node_bw_gbps",
    "inter_node_bw_bytes_per_s", "intra_node_bw_bytes_per_s",
    "intra_package_bw_gbps", "intra_package_bw_bytes_per_s",
    "inter_node_bw_util", "intra_node_bw_util", "intra_package_bw_util",
    "inter_node_latency_us", "intra_node_latency_us", "intra_package_latency_us",
    "intra_bw_4p_gbps", "intra_bw_8p_gbps", "superpod_size", "max_dies",
    "superpod_bw_gbps", "inter_pod_bw_gbps", "superpod_latency_us",
    "inter_pod_latency_us", "topology_preset",
)


def _check_pos_int(name: str, val: Any, *, minimum: int = 1) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        raise ValueError(f"{name}_must_be_int_ge_{minimum}:{val}")
    return int(val)


def _check_pos_float(name: str, val: Any) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or float(val) <= 0.0:
        raise ValueError(f"{name}_must_be_positive_float:{val}")
    return float(val)


_PCP_ATTENTION_TYPES = {"gqa", "mla", "kda", "gdn", "projection_only"}


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _attention_types(model: ModelSpec) -> set[str]:
    plans = getattr(model, "layer_plans", ()) or ()
    if plans:
        kinds = {str(plan.attention).strip().lower() for plan in plans if str(plan.attention).strip()}
        if kinds:
            return kinds
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    declared = facts.get("attention_types")
    if isinstance(declared, Mapping):
        kinds = {str(kind).strip().lower() for kind, count in declared.items() if count}
        if kinds:
            return kinds
    return {str(getattr(model, "attention", "gqa") or "gqa").strip().lower()}


def _ep_layout_valid(spec: ParallelSpec) -> bool:
    """EP divisor rules from ParallelSpec.validate, without a model."""
    if spec.ep > spec.ep_domain or spec.ep_domain % spec.ep != 0:
        return False
    if not spec.ep_spans_pcp:
        return not (spec.ep > 1 and spec.ep <= spec.tp and spec.tp % spec.ep != 0)
    local_domain = spec.tp * spec.pcp
    if spec.ep <= local_domain and local_domain % spec.ep != 0:
        return False
    if spec.ep > local_domain and spec.ep % local_domain != 0:
        return False
    return True


def _extract_model_geometry(case_or_model: CaseSpec | ModelSpec | Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(case_or_model, str):
        from mindie_sim.model_catalog import load_model
        return _extract_model_geometry(load_model(case_or_model))
    if isinstance(case_or_model, CaseSpec):
        return _extract_model_geometry(case_or_model.model)
    if isinstance(case_or_model, ModelSpec):
        meta = dict(case_or_model.metadata)
        attention = str(case_or_model.attention).strip().lower()
        return {
            "num_heads": int(case_or_model.num_heads),
            "num_kv_heads": int(case_or_model.num_kv_heads or case_or_model.num_heads),
            "num_experts": int(case_or_model.num_experts),
            "attention": attention,
            "attention_types": _attention_types(case_or_model),
            "layers": int(case_or_model.layers),
            "mla_replicate": bool(meta.get("mla_replicate", False)),
        }
    if isinstance(case_or_model, Mapping):
        raw_m = (
            case_or_model["model"]
            if ("model" in case_or_model and isinstance(case_or_model["model"], (Mapping, ModelSpec)))
            else case_or_model
        )
        if isinstance(raw_m, ModelSpec):
            return _extract_model_geometry(raw_m)
        if not isinstance(raw_m, Mapping):
            raise TypeError(f"unsupported_model_spec_type:{type(raw_m).__name__}")
        raw_meta = raw_m.get("metadata") if isinstance(raw_m.get("metadata"), Mapping) else {}
        mla_rep = bool(raw_m.get("mla_replicate", False) or raw_meta.get("mla_replicate", False))
        try:
            info = _extract_model_geometry(ModelSpec.from_mapping(raw_m))
            info["mla_replicate"] = info["mla_replicate"] or mla_rep
            return info
        except (ModelingError, ValueError, TypeError, KeyError):
            canon = ModelSpec.canonical_fields(raw_m)
            heads = int(canon.get("num_heads", 0) or 0)
            kv_heads = int(canon.get("num_kv_heads", heads) or 0)
            experts = int(canon.get("num_experts", 0) or 0)
            attn = str(canon.get("attention") or ("mla" if ("mla" in canon or "kv_lora_rank" in canon) else "gqa")).strip().lower()
            return {
                "num_heads": heads, "num_kv_heads": kv_heads, "num_experts": experts,
                "attention": attn, "attention_types": {attn},
                "layers": int(canon.get("layers", 1) or 1), "mla_replicate": mla_rep,
            }
    raise TypeError(f"unsupported_case_or_model_type:{type(case_or_model).__name__}")


def enumerate_parallel_candidates(
    case_or_model: CaseSpec | ModelSpec | Mapping[str, Any] | str, *, num_devices: int,
    allow_dp: bool = True, allow_ep: bool = True, allow_pp: bool = False, allow_pcp: bool = False,
    candidate_tps: Sequence[int] | None = None, ep_candidates: Sequence[int] | None = None,
    pp_candidates: Sequence[int] | None = None, pcp_candidates: Sequence[int] | None = None,
) -> list[dict[str, int]]:
    """Deterministically generate valid (dp, tp, ep, pp, pcp) parallel shardings."""
    n_dev = _check_pos_int("num_devices", num_devices, minimum=1)
    geom = _extract_model_geometry(case_or_model)
    num_heads, num_kv_heads, num_experts = geom["num_heads"], geom["num_kv_heads"], geom["num_experts"]
    attention_types = set(geom["attention_types"])
    layers, mla_replicate = geom["layers"], geom["mla_replicate"]

    all_divs = _divisors(n_dev)
    allowed_tps = {_check_pos_int("candidate_tp", t, minimum=1) for t in candidate_tps} if candidate_tps is not None else None
    tps = [tp for tp in all_divs if tp in allowed_tps] if allowed_tps is not None else all_divs
    allowed_eps = {_check_pos_int("ep_candidate", e, minimum=1) for e in ep_candidates} if ep_candidates is not None else None
    allowed_pps = {_check_pos_int("pp_candidate", p, minimum=1) for p in pp_candidates} if pp_candidates is not None else None
    allowed_pcps = {_check_pos_int("pcp_candidate", p, minimum=1) for p in pcp_candidates} if pcp_candidates is not None else None

    candidates: list[dict[str, int]] = []
    for tp in tps:
        if num_heads > 0 and not mla_replicate:
            if num_heads % tp != 0 or (num_kv_heads > 0 and num_kv_heads % tp != 0 and tp % num_kv_heads != 0):
                continue
        rem_tp = n_dev // tp
        pps = (
            [p for p in _divisors(rem_tp) if p in allowed_pps and (layers <= 0 or p <= layers)] if allowed_pps is not None
            else [p for p in _divisors(rem_tp) if p <= max(1, layers)] if allow_pp
            else [1]
        )
        for pp in pps:
            rem_pp = rem_tp // pp
            pcps = (
                [p for p in _divisors(rem_pp) if p in allowed_pcps] if allowed_pcps is not None
                else _divisors(rem_pp) if allow_pcp
                else [1]
            )
            for pcp in pcps:
                if pcp > 1 and not attention_types <= _PCP_ATTENTION_TYPES:
                    continue
                dp = rem_pp // pcp
                if not allow_dp and dp != 1:
                    continue
                dcp = 1
                ep_options = (
                    [d for d in _divisors(dp * tp * pcp)
                     if num_experts >= ZERO and _ep_layout_valid(
                         ParallelSpec(dp=dp, tp=tp, pp=pp, ep=d, cp=dcp, dcp=dcp, pcp=pcp))]
                    if (num_experts > ZERO and allow_ep) else [ONE]
                )
                if allowed_eps is not None:
                    ep_options = [e for e in ep_options if e in allowed_eps]
                for ep in sorted(ep_options):
                    candidates.append({"dp": dp, "tp": tp, "ep": ep, "pp": pp, "pcp": pcp})

    candidates.sort(key=lambda c: (c["tp"], c["dp"], c["ep"], c["pp"], c["pcp"]))
    return candidates


def _resolve_ranks_per_node(parsed: CaseSpec, num_devices: int) -> int:
    """Preserve declared ranks_per_node across multi-node scaling when compatible."""
    orig_rpn = int(parsed.machine.ranks_per_node)
    if orig_rpn >= 1:
        return min(num_devices, orig_rpn)
    from mindie_sim.backends.ascend import ASCEND_TOPOLOGY_PRESETS
    topo_opts = dict(getattr(parsed.machine, "topology_options", ()))
    preset_name = topo_opts.get("topology_preset")
    preset_rpn = int(ASCEND_TOPOLOGY_PRESETS.get(str(preset_name), {}).get("ranks_per_node", 0) or 0)
    if preset_rpn >= 1:
        return min(num_devices, preset_rpn)
    sku_default = 16 if (parsed.machine.arch_family == "A3" or parsed.machine.dies_per_package >= 2) else 8
    return min(num_devices, sku_default)


def _materialize_machine_for_devices(
    parsed: CaseSpec, raw_case: CaseSpec | Mapping[str, Any], num_devices: int,
) -> Hardware:
    """Materialize a Hardware descriptor for num_devices preserving base SKU/rates/node topology."""
    orig_count = len(parsed.machine.dies)
    if orig_count == num_devices:
        return parsed.machine

    base_dev = deepcopy(parsed.machine.dies[0].to_dict())
    for k in ("rank", "node", "package"):
        base_dev.pop(k, None)

    ranks_per_node = _resolve_ranks_per_node(parsed, num_devices)
    sku = parsed.machine.sku
    from mindie_sim.backends.nvidia import materialize as nvidia_materialize, profiles as nvidia_profiles
    is_ascend = bool(sku and sku in ascend_profiles())
    is_nvidia = bool(sku and sku in nvidia_profiles())
    if is_ascend or is_nvidia:
        raw_m: dict[str, Any] = {
            "backend": "ascend" if is_ascend else "nvidia",
            "name": parsed.machine.name, "hardware_key": parsed.machine.hardware_key,
            "sku": sku, "num_devices": num_devices, "ranks_per_node": ranks_per_node,
            "dies_per_package": parsed.machine.dies_per_package, "device": base_dev,
        }
        for k, v in getattr(parsed.machine, "topology_options", ()):
            raw_m[k] = deepcopy(v)
        ch_patch: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in getattr(parsed.machine, "transfer_channels", ())
            if k not in {"d2d_intra_4p", "d2d_intra_8p", "d2d_inter_pod"}
        }
        seen_links: set[str] = set()
        for lnk in parsed.machine.links:
            if lnk.channel in {"d2d_intra_package", "d2d_intra_node", "d2d_inter_node"} and lnk.channel not in seen_links:
                seen_links.add(lnk.channel)
                rec = dict(ch_patch.get(lnk.channel, {}))
                util = float(getattr(lnk, "bw_utilization", 1.0) or 1.0)
                if not math.isclose(util, 1.0):
                    peak_gbps = float(lnk.effective_peak_capacity_bytes_per_ms) / 1e6
                    rec.pop("effective_uni_bw_gbps", None)
                    rec.update(
                        uni_peak_bw_gbps=peak_gbps,
                        bw_utilization=util,
                        latency_us=float(lnk.latency_us),
                        semantics=str(lnk.semantics),
                    )
                else:
                    rec.update(
                        effective_uni_bw_gbps=float(lnk.capacity_bytes_per_ms) / 1e6,
                        latency_us=float(lnk.latency_us),
                        semantics=str(lnk.semantics),
                    )
                for fn in ("transaction_bytes", "alignment_bytes", "protocol_overhead_bytes", "switch_hops", "switch_hop_latency_us", "switch_capacity_bytes_per_ms"):
                    val = getattr(lnk, fn, None)
                    if val is not None:
                        rec[fn] = val
                ch_patch[lnk.channel] = rec
        if ch_patch:
            raw_m["transfer_channels"] = ch_patch
        if isinstance(raw_case, Mapping) and isinstance(raw_case.get("machine"), Mapping):
            src_m = raw_case["machine"]
            for k in _TRANSFER_OVERRIDE_KEYS:
                if k in src_m:
                    if k == "transfer_channels" and isinstance(src_m[k], Mapping):
                        raw_m[k] = {
                            ck: deepcopy(cv) for ck, cv in src_m[k].items()
                            if ck not in {"d2d_intra_4p", "d2d_intra_8p", "d2d_inter_pod"}
                        }
                    else:
                        raw_m[k] = deepcopy(src_m[k])
        mat_fn = ascend_materialize if is_ascend else nvidia_materialize
        return Hardware.from_mapping(mat_fn(raw_m))

    m_dict = deepcopy(parsed.machine.to_dict())
    m_dict["num_devices"], m_dict["ranks_per_node"] = num_devices, ranks_per_node
    m_dict["devices"] = [{**deepcopy(base_dev), "rank": idx} for idx in range(num_devices)]
    new_links = []
    for lnk in m_dict.get("links", ()):
        lnk_copy = deepcopy(lnk)
        members = list(lnk_copy.get("members", ()))
        lnk_copy["members"] = list(range(num_devices)) if len(members) == orig_count else [m for m in members if m < num_devices]
        if len(lnk_copy["members"]) > 1 or num_devices == 1:
            new_links.append(lnk_copy)
    if not new_links and num_devices > 1 and m_dict.get("links"):
        fallback_lnk = deepcopy(m_dict["links"][0])
        fallback_lnk["members"] = list(range(num_devices))
        new_links.append(fallback_lnk)
    m_dict["links"] = new_links
    return Hardware.from_mapping(m_dict)


def compute_pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark pareto_optimal over feasible rows on (tps_per_die, total_feasible_concurrency, -step_ms_at_b1)."""
    feasible_rows = [r for r in rows if r.get("feasible")]
    for row in rows:
        if not row.get("feasible"):
            row["pareto_optimal"] = False
            continue
        dominated = False
        for other in feasible_rows:
            if other is row:
                continue
            ge_all = (
                float(other["tps_per_die"]) >= float(row["tps_per_die"]) - _PARETO_EPS
                and int(other["total_feasible_concurrency"]) >= int(row["total_feasible_concurrency"])
                and float(other["step_ms_at_b1"]) <= float(row["step_ms_at_b1"]) + _PARETO_EPS
            )
            strict_one = (
                float(other["tps_per_die"]) > float(row["tps_per_die"]) + _PARETO_EPS
                or int(other["total_feasible_concurrency"]) > int(row["total_feasible_concurrency"])
                or float(other["step_ms_at_b1"]) < float(row["step_ms_at_b1"]) - _PARETO_EPS
            )
            if ge_all and strict_one:
                dominated = True
                break
        row["pareto_optimal"] = not dominated
    return [r for r in rows if r.get("pareto_optimal")]


def sweep_topologies(
    case: CaseSpec | Mapping[str, Any], *, candidates: Sequence[Mapping[str, Any]] | None = None,
    num_devices: int | None = None, sla_ms: float = DEFAULT_SLA_MS, context: int = DEFAULT_CONTEXT,
    phase: str = "decode", scope: str = "body", tokens_per_seq: int | None = None,
    output_budget: int = DEFAULT_OUTPUT_BUDGET, max_sequences_per_dp: int = DEFAULT_MAX_SEQUENCES,
    allow_dp: bool = True, allow_ep: bool = True, allow_pp: bool = False, allow_pcp: bool = False,
    candidate_tps: Sequence[int] | None = None, ep_candidates: Sequence[int] | None = None,
    pp_candidates: Sequence[int] | None = None, pcp_candidates: Sequence[int] | None = None,
    hardware_derivations: Sequence[Mapping[str, Any]] | None = None,
    lock_parallel: bool = False,
    fixed_parallel: Mapping[str, int] | None = None,
    lock_program: bool = False,
) -> dict[str, Any]:
    """Sweep parallel sharding candidates and identify the axiomatic Pareto frontier."""
    _check_pos_float("sla_ms", sla_ms)
    _check_pos_int("context", context, minimum=0)
    _check_pos_int("output_budget", output_budget, minimum=0)
    _check_pos_int("max_sequences_per_dp", max_sequences_per_dp, minimum=1)
    if phase not in {"prefill", "decode"} or scope not in {"body", "model"}:
        raise ValueError(f"invalid_phase_or_scope:{phase}:{scope}")
    if tokens_per_seq is not None:
        _check_pos_int("tokens_per_seq", tokens_per_seq, minimum=1)

    parsed = parse_case(case)
    if lock_parallel and fixed_parallel is not None:
        candidates = (dict(fixed_parallel),)
    elif lock_parallel and candidates is None:
        base = parsed.runtime.parallel
        candidates = ({"dp": base.dp, "tp": base.tp, "ep": base.ep,
                       "pp": base.pp, "pcp": base.pcp},)
    if hardware_derivations:
        from mindie_sim.hardware_derivation import derived_machine
        combined: list[dict[str, Any]] = []
        for derivation in hardware_derivations:
            declare = dict(derivation)
            sku = str(declare.pop("sku", parsed.machine.sku or "derived"))
            derived = derived_machine(parsed.machine, {**declare, "sku": sku})
            sub = sweep_topologies(
                replace(parsed, machine=derived), candidates=candidates,
                num_devices=num_devices, sla_ms=sla_ms, context=context,
                phase=phase, scope=scope, tokens_per_seq=tokens_per_seq,
                output_budget=output_budget, max_sequences_per_dp=max_sequences_per_dp,
                allow_dp=allow_dp, allow_ep=allow_ep, allow_pp=allow_pp,
                allow_pcp=allow_pcp, candidate_tps=candidate_tps,
                ep_candidates=ep_candidates, pp_candidates=pp_candidates,
                pcp_candidates=pcp_candidates,
            )
            for row in sub["candidates"]:
                row["derived_sku"] = sku
                row["derivation"] = {
                    "sku": derived.sku,
                    "aic_count": derived.dies[FIRST].compute_units if derived.dies else None,
                    "aiv_count": (dict(derived.dies[FIRST].resource_units).get("vector")
                                  if derived.dies else None),
                    "memory_bytes": derived.dies[FIRST].memory_bytes if derived.dies else None,
                    "assumptions": dict(derived.assumptions),
                }
            combined.extend(sub["candidates"])
        base_devices = (num_devices if num_devices is not None
                        else len(parsed.machine.dies))
        return {
            "schema": "mindie-sim.topology-sweep/1",
            "num_devices": base_devices,
            "sla_ms": sla_ms, "context": context, "phase": phase, "scope": scope,
            "search_domain": {
                "num_devices": base_devices, "allow_dp": allow_dp,
                "allow_ep": allow_ep, "allow_pp": allow_pp,
                "allow_pcp": allow_pcp, "candidate_count": len(combined),
                "hardware_derivations": len(hardware_derivations),
            },
            "candidates": combined,
            "pareto_frontier": compute_pareto_frontier(combined),
            "plan_selection": ("fixed_parallel" if lock_parallel
                               else "reselected_parallel_per_candidate"),
            "program_selection": "fixed" if lock_program else "reselected",
            "parallel_signature": str(fixed_parallel or {}),
            "changed_hardware_fields": (),
            "comparison_boundary": "each_row_is_one_complete_candidate_no_cross_candidate_metric_stitching",
            "pareto": {"basis": "exact_multi_objective_dominance_on_single_candidate_metrics",
                       "frontier_is_selected_candidates_not_a_global_optimum_proof": True},
        }
    target_devices = _check_pos_int("num_devices", num_devices, minimum=1) if num_devices is not None else len(parsed.machine.dies)
    cand_list = (
        [dict(c) for c in candidates]
        if candidates is not None
        else enumerate_parallel_candidates(
            parsed, num_devices=target_devices, allow_dp=allow_dp, allow_ep=allow_ep,
            allow_pp=allow_pp, allow_pcp=allow_pcp, candidate_tps=candidate_tps,
            ep_candidates=ep_candidates, pp_candidates=pp_candidates, pcp_candidates=pcp_candidates,
        )
    )

    machines_by_count: dict[int, Hardware] = {}
    rows: list[dict[str, Any]] = []
    seq_probe = max(1, int(context))

    for raw_cand in cand_list:
        cand_dict = {k: int(raw_cand.get(k, 1)) for k in ("dp", "tp", "ep", "pp", "pcp")}
        for opt_k in ("cp", "dcp", "cp_plan"):
            if opt_k in raw_cand:
                cand_dict[opt_k] = raw_cand[opt_k]
        cand_devices = (
            max(1, cand_dict["dp"] * cand_dict["tp"] * cand_dict["pp"] * cand_dict["pcp"])
            if num_devices is None and candidates is not None
            else target_devices
        )
        try:
            cand_p = ParallelSpec.from_mapping(cand_dict)
            if cand_p.world_size != cand_devices:
                raise ModelingError(f"parallel_world_size_mismatch:{cand_p.world_size}!={cand_devices}")
            if cand_devices not in machines_by_count:
                machines_by_count[cand_devices] = _materialize_machine_for_devices(parsed, case, cand_devices)
            part = parsed.runtime.pp_layer_partition if len(parsed.runtime.pp_layer_partition) == cand_p.pp else ()
            cand_case = parse_case(
                replace(parsed, machine=machines_by_count[cand_devices], runtime=replace(parsed.runtime, parallel=cand_p, pp_layer_partition=part))
            )
            mem = Memory(cand_case, workspace_tokens=1)
            weight_bytes_per_die = float(max(mem.weights))
            kv_bytes_per_token_per_die = float(max(mem.cache_bytes(seq_probe, padded=False)) / seq_probe)
            sla_res = solve_sla_sizing(
                cand_case, sla_ms=sla_ms, context=context, phase=phase, scope=scope,
                tokens_per_seq=tokens_per_seq, output_budget=output_budget,
                max_sequences_per_dp=max_sequences_per_dp, scenarios=_BASELINE_SCENARIO,
            )
            scen = sla_res["scenarios"][0]
            candidate = scen.get("feasible_candidate")
            feasible = candidate is not None
            b_sla, b_vram_max = int(scen["b_sla"]), int(scen["b_vram_max"])
            b_feasible = int(candidate["b"]) if candidate is not None else 0
            tps_feasible = float(candidate["tps_feasible"]) if candidate is not None and "tps_feasible" in candidate else float(scen["tps_feasible"])
            row = {
                "parallel": {k: cand_dict[k] for k in ("dp", "tp", "ep", "pp", "pcp")},
                "status": "feasible" if feasible else "sla_or_vram_exceeded",
                "feasible": feasible,
                "plan_mode": "reselected_parallel",
                "evaluation_boundary": "complete_candidate_graph_and_capacity",
                "reason": None if feasible else scen.get("feasibility_status"),
                "num_devices": cand_devices,
                "ranks_per_node": machines_by_count[cand_devices].ranks_per_node,
                "weight_bytes_per_die": weight_bytes_per_die,
                "kv_bytes_per_token_per_die": kv_bytes_per_token_per_die,
                "b_sla": b_sla, "b_vram_max": b_vram_max, "b_feasible": b_feasible,
                "latency_candidate": scen.get("latency_candidate"),
                "feasible_candidate": candidate,
                "feasibility_status": scen.get("feasibility_status"), "search": scen.get("search"),
                "total_feasible_concurrency": b_feasible * cand_p.dp,
                "step_ms_at_b1": float(scen["step_ms_at_b1"]),
                "step_ms_feasible": float(candidate["step_ms"]) if candidate is not None else None,
                "tps_at_b1": float(scen["tps_at_b1"]),
                "tps_feasible": tps_feasible,
                "tps_per_die": tps_feasible / float(cand_devices),
                "network_mbu": float(candidate["network_mbu"]) if candidate is not None else None,
                "network_cube_mfu": float(candidate["network_cube_mfu"]) if candidate is not None else None,
                "network_vec_mfu": float(candidate["network_vec_mfu"]) if candidate is not None else None,
                "network_link_util": float(candidate["network_link_util"]) if candidate is not None else None,
            }
            rows.append(row)
        except (ModelingError, ValueError) as exc:
            rows.append({
                "parallel": {k: cand_dict[k] for k in ("dp", "tp", "ep", "pp", "pcp")},
                "status": "infeasible", "feasible": False, "reason": str(exc),
                "plan_mode": "reselected_parallel", "evaluation_boundary": "candidate_rejected_before_or_during_complete_evaluation",
                "feasibility_status": "infeasible_before_evaluation", "search": None,
                "num_devices": cand_devices, "ranks_per_node": None,
                "weight_bytes_per_die": None, "kv_bytes_per_token_per_die": None,
                "b_sla": 0, "b_vram_max": 0, "b_feasible": 0, "total_feasible_concurrency": 0,
                "latency_candidate": None, "feasible_candidate": None,
                "step_ms_at_b1": None, "step_ms_feasible": None, "tps_at_b1": 0.0,
                "tps_feasible": 0.0, "tps_per_die": 0.0, "network_mbu": None,
                "network_cube_mfu": None, "network_vec_mfu": None, "network_link_util": None,
            })

    pareto_frontier = compute_pareto_frontier(rows)
    return {
        "schema": "mindie-sim.topology-sweep/1",
        "num_devices": target_devices, "sla_ms": sla_ms, "context": context, "phase": phase, "scope": scope,
        "search_domain": {
            "num_devices": target_devices, "allow_dp": allow_dp, "allow_ep": allow_ep,
            "allow_pp": allow_pp, "allow_pcp": allow_pcp, "candidate_count": len(rows),
        },
        "candidates": rows, "pareto_frontier": pareto_frontier,
        "plan_selection": ("fixed_parallel" if lock_parallel
                           else "reselected_parallel_per_candidate"),
        "program_selection": "fixed" if lock_program else "reselected",
        "parallel_signature": str(fixed_parallel or ""),
        "changed_hardware_fields": (),
        "comparison_boundary": "each_row_is_one_complete_candidate_no_cross_candidate_metric_stitching",
        "pareto": {"basis": "exact_multi_objective_dominance_on_single_candidate_metrics",
                   "frontier_is_selected_candidates_not_a_global_optimum_proof": True},
    }


def _pool_kv_hbm_ms(parsed_pool: CaseSpec, mem_pool: Memory, tokens: int) -> float:
    p = parsed_pool.runtime.parallel
    width = p.tp * p.pcp
    best = 0.0
    for st, st_bytes in enumerate(mem_pool.cache_bytes(tokens, padded=False)):
        dies = parsed_pool.machine.dies[st * width:(st + 1) * width] or (parsed_pool.machine.dies[0],)
        hbm_rate_ms = min(float(dict(d.rates)["hbm"]) for d in dies) / 1000.0
        best = max(best, float(st_bytes) / max(hbm_rate_ms, 1e-12))
    return best


def sweep_pd_disaggregation_ratios(
    prefill_case: CaseSpec | Mapping[str, Any],
    decode_case: CaseSpec | Mapping[str, Any], *, total_dies: int,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    decode_context: int | None = None, ttft_sla_ms: float = DEFAULT_TTFT_SLA_MS,
    tpot_sla_ms: float = DEFAULT_SLA_MS,
    kv_transfer_bw_gbps: float = DEFAULT_KV_TRANSFER_BW_GBPS,
) -> dict[str, Any]:
    """Sweep disaggregated Prefill/Decode instance counts and KV handoff bottlenecks."""
    n_total_dies = _check_pos_int("total_dies", total_dies, minimum=1)
    n_prompt = _check_pos_int("prompt_tokens", prompt_tokens, minimum=1)
    n_output = _check_pos_int("output_tokens", output_tokens, minimum=1)
    eff_decode_ctx = n_prompt if decode_context is None else _check_pos_int("decode_context", decode_context, minimum=0)
    _check_pos_float("ttft_sla_ms", ttft_sla_ms)
    _check_pos_float("tpot_sla_ms", tpot_sla_ms)
    _check_pos_float("kv_transfer_bw_gbps", kv_transfer_bw_gbps)

    p_parsed, d_parsed = parse_case(prefill_case), parse_case(decode_case)
    p_par, d_par = p_parsed.runtime.parallel, d_parsed.runtime.parallel
    d_p, d_d, p_dp, d_dp = p_par.world_size, d_par.world_size, int(p_par.dp), int(d_par.dp)

    p_mem = Memory(p_parsed, workspace_tokens=n_prompt)
    p_kv_req = p_mem.cache_bytes(n_prompt + 1, padded=True)
    prefill_hbm_ok = bool(
        n_prompt <= p_parsed.runtime.max_model_len
        and (p_parsed.runtime.enable_chunked_prefill or n_prompt <= p_parsed.runtime.max_num_batched_tokens)
        and all(cap >= 0.0 and cap >= kv_b for row in p_mem.capacity for cap, kv_b in zip(row, p_kv_req))
    )
    p_slots = (
        min(
            [int(p_parsed.runtime.max_num_seqs)]
            + [int(math.floor(max(0.0, float(cap)) / float(kv_b))) for row in p_mem.capacity for cap, kv_b in zip(row, p_kv_req) if kv_b > 0]
        )
        if prefill_hbm_ok else 0
    )

    _, _, p_graph_rep = _lower_and_run_graph(
        p_parsed, phase="prefill", tokens=n_prompt, context=0, sequences=1, scope="body", run_graph=True,
    )
    assert p_graph_rep is not None
    prefill_step_ms = float(p_graph_rep["duration_ms"])
    prefill_sla_ok = bool(prefill_step_ms <= ttft_sla_ms)

    d_mem = Memory(d_parsed)
    decode_tokens_needed = n_output - 1
    if decode_tokens_needed <= ZERO:
        kv_bytes_per_request = 0.0
        kv_transfer_ms_per_request = 0.0
        rps_p1 = ((p_dp * 1000.0) / max(prefill_step_ms, 1e-12)) if (prefill_sla_ok and prefill_hbm_ok) else 0.0
        rps_kv1 = rps_p1
        handoff_tpot_ms = 0.0
        decode_budget_ms = tpot_sla_ms
        b_sla = 0
        b_vram_max = int(d_parsed.runtime.max_num_seqs)
        b_feasible = 0
        step_ms_at_b1 = 0.0
        decode_step_ms = 0.0
        effective_tpot_ms = 0.0
        first_decode_itl_ms = 0.0
        tpot_sla_ok = True
        rps_d1 = rps_p1
        raw_rps_d1 = rps_p1
        decode_tps_1 = rps_d1 * float(n_output)
        infeasible_reason = (
            "prefill_hbm_oom" if not prefill_hbm_ok
            else "prefill_ttft_sla_exceeded" if not prefill_sla_ok
            else None
        )
    else:
        kv_bytes_per_request = float(sum(d_mem.cache_bytes(n_prompt, padded=False)) * d_par.tp * d_par.pcp)
        wire_ms = kv_bytes_per_request / (float(kv_transfer_bw_gbps) * 1e6)
        p_hbm_ms = _pool_kv_hbm_ms(p_parsed, p_mem, n_prompt)
        d_hbm_ms = _pool_kv_hbm_ms(d_parsed, d_mem, n_prompt)
        kv_transfer_ms_per_request = max(wire_ms, p_hbm_ms, d_hbm_ms)
        per_rep_rps = min(
            1000.0 / max(prefill_step_ms, 1e-12),
            (float(p_slots) * 1000.0) / max(prefill_step_ms + kv_transfer_ms_per_request, 1e-12),
        )
        rps_p1 = (p_dp * per_rep_rps) if (prefill_sla_ok and prefill_hbm_ok) else 0.0
        rps_kv1 = 1000.0 / max(kv_transfer_ms_per_request, 1e-9)

        handoff_tpot_ms = kv_transfer_ms_per_request / float(decode_tokens_needed)
        decode_budget_ms = tpot_sla_ms - handoff_tpot_ms
        _, d_emit_est = _resolve_speculative_tokens(
            d_parsed, output_budget=n_output, tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        step_budget_ms = decode_budget_ms * max(1.0, float(d_emit_est))
        d_sla_res = solve_sla_sizing(
            d_parsed, sla_ms=(max(step_budget_ms, _PARETO_EPS)
                              if step_budget_ms > ZERO
                              else tpot_sla_ms * max(ONE_FLOAT, float(d_emit_est))),
            context=eff_decode_ctx, phase="decode", output_budget=n_output, scenarios=[DEFAULT_SLA_SCENARIOS[FIRST]],
        )
        d_scen = d_sla_res["scenarios"][0]
        emit_tokens = float(d_sla_res["emitted_tokens_per_seq"])
        decode_len_ok = bool(max(n_prompt, eff_decode_ctx) + n_output <= d_parsed.runtime.max_model_len)
        b_vram_max = int(d_scen["b_vram_max"]) if decode_len_ok else 0
        b_sla = int(d_scen["b_sla"]) if decode_budget_ms > 0.0 else 0
        b_feasible = int(d_scen["b_feasible"])
        step_ms_at_b1 = float(d_scen["step_ms_at_b1"])
        decode_step_ms = (float(d_scen["step_ms_feasible"])
                          if b_feasible >= 1 and d_scen.get("step_ms_feasible") is not None
                          else step_ms_at_b1)
        effective_tpot_ms = (decode_step_ms / max(emit_tokens, 1e-12)) + handoff_tpot_ms
        first_decode_itl_ms = kv_transfer_ms_per_request + decode_step_ms
        tpot_sla_ok = bool(b_feasible >= 1 and effective_tpot_ms <= tpot_sla_ms)
        rps_d1 = (
            (b_feasible * d_dp * emit_tokens * 1000.0) / (max(decode_step_ms, 1e-12) * float(decode_tokens_needed))
            if tpot_sla_ok else 0.0
        )
        decode_tps_1 = rps_d1 * float(n_output)
        infeasible_reason = (
            "prefill_hbm_oom" if not prefill_hbm_ok
            else "prefill_ttft_sla_exceeded" if not prefill_sla_ok
            else "decode_max_model_len_exceeded" if not decode_len_ok
            else "decode_hbm_oom" if b_vram_max < 1
            else "effective_tpot_sla_exceeded" if not tpot_sla_ok
            else None
        )
        raw_rps_d1 = (max(1, int(d_scen["b_feasible"])) * d_dp * emit_tokens * 1000.0) / (max(decode_step_ms, 1e-12) * float(decode_tokens_needed))

    rows: list[dict[str, Any]] = []
    for n_p in range(ONE, (n_total_dies // d_p) + ONE):
        max_nd = (n_total_dies - n_p * d_p) // d_d
        for n_d in range(ONE, max_nd + ONE):
            used_dies = n_p * d_p + n_d * d_d
            r_p = n_p * rps_p1
            r_kv = r_p if decode_tokens_needed <= 0 else n_p * rps_kv1
            r_d = r_p if decode_tokens_needed <= 0 else n_d * rps_d1
            feasible = bool(infeasible_reason is None and min(r_p, r_kv, r_d) > 0.0)
            r_sys = min(r_p, r_kv, r_d) if feasible else 0.0
            sustainable_tps = r_sys * float(n_output)
            if decode_tokens_needed <= ZERO:
                bottleneck_stage = "prefill"
            else:
                stage_rates = (("prefill", r_p), ("kv_transfer", r_kv), ("decode", r_d if decode_budget_ms > 0.0 else n_d * raw_rps_d1))
                bottleneck_stage = min(stage_rates, key=lambda x: x[1])[0]
            rows.append({
                "n_prefill": n_p, "n_decode": n_d, "pd_ratio": f"{n_p}P:{n_d}D",
                "prefill_dies": n_p * d_p, "decode_dies": n_d * d_d,
                "used_dies": used_dies, "idle_dies": n_total_dies - used_dies,
                "prefill_rps": r_p, "kv_transfer_rps": r_kv, "decode_rps": r_d,
                "sustainable_rps": r_sys, "sustainable_tps": sustainable_tps,
                "sustainable_tps_per_die": sustainable_tps / float(used_dies),
                "decode_step_ms": decode_step_ms, "effective_tpot_ms": effective_tpot_ms,
                "first_decode_itl_ms": first_decode_itl_ms, "bottleneck_stage": bottleneck_stage,
                "feasible": feasible, "infeasible_reason": infeasible_reason, "reason": infeasible_reason,
                "metrics_boundary": "one_pipeline_stage_rate_composition",
            })

    feasible_rows = [r for r in rows if r["feasible"]]
    for row in rows:
        if not row["feasible"]:
            row["pareto_optimal"] = False
            continue
        dominated = False
        for other in feasible_rows:
            if other is row:
                continue
            ge_all = (
                float(other["sustainable_tps"]) >= float(row["sustainable_tps"]) - _PARETO_EPS
                and int(other["used_dies"]) <= int(row["used_dies"])
            )
            strict_one = (
                float(other["sustainable_tps"]) > float(row["sustainable_tps"]) + _PARETO_EPS
                or int(other["used_dies"]) < int(row["used_dies"])
            )
            if ge_all and strict_one:
                dominated = True
                break
        row["pareto_optimal"] = not dominated

    pareto_frontier = [r for r in rows if r.get("pareto_optimal")]
    best_cfg = max(pareto_frontier, key=lambda r: (r["sustainable_tps"], -r["used_dies"]), default=None)
    return {
        "schema": "mindie-sim.pd-sweep/1",
        "total_dies": n_total_dies, "prompt_tokens": n_prompt, "output_tokens": n_output,
        "decode_context": eff_decode_ctx, "ttft_sla_ms": ttft_sla_ms, "tpot_sla_ms": tpot_sla_ms,
        "kv_transfer_bw_gbps": kv_transfer_bw_gbps, "kv_bytes_per_request": kv_bytes_per_request,
        "kv_transfer_ms_per_request": kv_transfer_ms_per_request, "decode_step_ms": decode_step_ms,
        "effective_tpot_ms": effective_tpot_ms, "first_decode_itl_ms": first_decode_itl_ms,
        "prefill_instance": {
            "dies_per_instance": d_p, "parallel": p_par.to_dict(), "step_ms": prefill_step_ms,
            "hbm_ok": prefill_hbm_ok, "ttft_sla_ok": prefill_sla_ok, "rps_per_instance": rps_p1,
        },
        "decode_instance": {
            "dies_per_instance": d_d, "parallel": d_par.to_dict(), "b_sla": b_sla,
            "b_vram_max": b_vram_max, "b_feasible": b_feasible,
            "step_ms_at_b1": step_ms_at_b1, "step_ms_feasible": decode_step_ms,
            "effective_tpot_ms": effective_tpot_ms, "first_decode_itl_ms": first_decode_itl_ms,
            "tpot_sla_ok": tpot_sla_ok, "tps_per_instance": decode_tps_1, "rps_per_instance": rps_d1,
        },
        "candidates": rows, "ratios": rows,
        "pareto_frontier": pareto_frontier, "best_configuration": best_cfg,
        "plan_selection": "fixed_stage_case_per_pipeline",
        "composition": "separate_prefill_and_decode_stage_programs_bound_by_min_stage_rate",
        "comparison_boundary": "stage_rates_are_composed_for_one_pipeline_configuration_not_stitched_per_metric",
    }


__all__ = ["compute_pareto_frontier", "enumerate_parallel_candidates", "sweep_pd_disaggregation_ratios", "sweep_topologies"]
