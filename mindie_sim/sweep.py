"""Axiomatic multi-topology parallel sharding and P/D disaggregation Pareto solver.

[Role] Enumerate structurally valid (DP, TP, EP, PP, PCP) sharding topologies,
  evaluate hardware SLA concurrency and throughput across topology grids, and
  solve disaggregated Prefill/Decode instance ratios under KV transfer and
  TTFT/TPOT SLA constraints.
[Invariants] Strictly closed-world, deterministic, axiomatic evaluation:
  physical weight/KV memory bounds come from Memory, step latencies and SLA
  ceilings come from solve_hardware_sla_throughput and _lower_and_run_graph,
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
    solve_hardware_sla_throughput,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec

_PARETO_EPS = 1e-12
_BASELINE_SCENARIO: tuple[dict[str, Any], ...] = (
    {"id": "baseline", "label": "1x Baseline", "hbm_bw_scale": 1.0, "cube_flops_scale": 1.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
)
_TRANSFER_OVERRIDE_KEYS = (
    "transfer_channels", "inter_node_bw_gbps", "intra_node_bw_gbps",
    "inter_node_bw_bytes_per_s", "intra_node_bw_bytes_per_s",
)


def _check_pos_int(name: str, val: Any, *, minimum: int = 1) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        raise ValueError(f"{name}_must_be_int_ge_{minimum}:{val}")
    return int(val)


def _check_pos_float(name: str, val: Any) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or float(val) <= 0.0:
        raise ValueError(f"{name}_must_be_positive_float:{val}")
    return float(val)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _extract_model_geometry(case_or_model: CaseSpec | ModelSpec | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(case_or_model, CaseSpec):
        return _extract_model_geometry(case_or_model.model)
    if isinstance(case_or_model, ModelSpec):
        meta = dict(case_or_model.metadata)
        return {
            "num_heads": int(case_or_model.num_heads),
            "num_kv_heads": int(case_or_model.num_kv_heads or case_or_model.num_heads),
            "num_experts": int(case_or_model.num_experts),
            "attention": str(case_or_model.attention).strip().lower(),
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
                "attention": attn, "layers": int(canon.get("layers", 1) or 1), "mla_replicate": mla_rep,
            }
    raise TypeError(f"unsupported_case_or_model_type:{type(case_or_model).__name__}")


def enumerate_parallel_candidates(
    case_or_model: CaseSpec | ModelSpec | Mapping[str, Any], *, num_devices: int,
    allow_dp: bool = True, allow_ep: bool = True, allow_pp: bool = False, allow_pcp: bool = False,
    candidate_tps: Sequence[int] | None = None, ep_candidates: Sequence[int] | None = None,
    pp_candidates: Sequence[int] | None = None, pcp_candidates: Sequence[int] | None = None,
) -> list[dict[str, int]]:
    """Deterministically generate valid (dp, tp, ep, pp, pcp) parallel shardings."""
    n_dev = _check_pos_int("num_devices", num_devices, minimum=1)
    geom = _extract_model_geometry(case_or_model)
    num_heads, num_kv_heads, num_experts = geom["num_heads"], geom["num_kv_heads"], geom["num_experts"]
    attention, layers, mla_replicate = geom["attention"], geom["layers"], geom["mla_replicate"]

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
                if pcp > 1 and attention != "gqa":
                    continue
                dp = rem_pp // pcp
                if not allow_dp and dp != 1:
                    continue
                ep_options = (
                    [d for d in _divisors(dp * tp) if num_experts % d == 0 and (d > tp or tp % d == 0)]
                    if (num_experts > 0 and allow_ep and pcp == 1) else [1]
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
    if sku and sku in ascend_profiles():
        raw_m: dict[str, Any] = {
            "backend": "ascend", "name": parsed.machine.name, "hardware_key": parsed.machine.hardware_key,
            "sku": sku, "num_devices": num_devices, "ranks_per_node": ranks_per_node,
            "dies_per_package": parsed.machine.dies_per_package, "device": base_dev,
        }
        ch_patch: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in getattr(parsed.machine, "transfer_channels", ())
        }
        seen_links: set[str] = set()
        for lnk in parsed.machine.links:
            if lnk.channel in {"d2d_intra_package", "d2d_intra_node", "d2d_inter_node"} and lnk.channel not in seen_links:
                seen_links.add(lnk.channel)
                rec = dict(ch_patch.get(lnk.channel, {}))
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
                    raw_m[k] = deepcopy(src_m[k])
        return Hardware.from_mapping(ascend_materialize(raw_m))

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


def sweep_topology_grid(
    case: CaseSpec | Mapping[str, Any], *, candidates: Sequence[Mapping[str, Any]] | None = None,
    num_devices: int | None = None, sla_ms: float = 30.0, context: int = 126000,
    phase: str = "decode", scope: str = "body", tokens_per_seq: int | None = None,
    output_budget: int = 1024, max_sequences_per_dp: int = 256,
    allow_dp: bool = True, allow_ep: bool = True, allow_pp: bool = False, allow_pcp: bool = False,
    candidate_tps: Sequence[int] | None = None, ep_candidates: Sequence[int] | None = None,
    pp_candidates: Sequence[int] | None = None, pcp_candidates: Sequence[int] | None = None,
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
            sla_res = solve_hardware_sla_throughput(
                cand_case, sla_ms=sla_ms, context=context, phase=phase, scope=scope,
                tokens_per_seq=tokens_per_seq, output_budget=output_budget,
                max_sequences_per_dp=max_sequences_per_dp, scenarios=_BASELINE_SCENARIO,
            )
            scen = sla_res["scenarios"][0]
            b_sla, b_vram_max, b_feasible = int(scen["b_sla"]), int(scen["b_vram_max"]), int(scen["b_feasible"])
            feasible = bool(b_feasible >= 1)
            tps_feasible = float(scen["tps_feasible"])
            rows.append({
                "parallel": {k: cand_dict[k] for k in ("dp", "tp", "ep", "pp", "pcp")},
                "status": "feasible" if feasible else "sla_or_vram_exceeded",
                "feasible": feasible,
                "reason": None if feasible else ("vram_oom" if b_vram_max < 1 else "sla_exceeded"),
                "num_devices": cand_devices,
                "ranks_per_node": machines_by_count[cand_devices].ranks_per_node,
                "weight_bytes_per_die": weight_bytes_per_die,
                "kv_bytes_per_token_per_die": kv_bytes_per_token_per_die,
                "b_sla": b_sla, "b_vram_max": b_vram_max, "b_feasible": b_feasible,
                "total_feasible_concurrency": b_feasible * cand_p.dp,
                "step_ms_at_b1": float(scen["step_ms_at_b1"]),
                "step_ms_feasible": float(scen["step_ms_feasible"]),
                "tps_at_b1": float(scen["tps_at_b1"]),
                "tps_feasible": tps_feasible,
                "tps_per_die": tps_feasible / float(cand_devices),
                "network_mbu": float(scen["network_mbu"]),
                "network_cube_mfu": float(scen["network_cube_mfu"]),
                "network_vec_mfu": float(scen["network_vec_mfu"]),
                "network_link_util": float(scen["network_link_util"]),
            })
        except (ModelingError, ValueError) as exc:
            rows.append({
                "parallel": {k: cand_dict[k] for k in ("dp", "tp", "ep", "pp", "pcp")},
                "status": "infeasible", "feasible": False, "reason": str(exc),
                "num_devices": cand_devices, "ranks_per_node": None,
                "weight_bytes_per_die": None, "kv_bytes_per_token_per_die": None,
                "b_sla": 0, "b_vram_max": 0, "b_feasible": 0, "total_feasible_concurrency": 0,
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
    prefill_case: CaseSpec | Mapping[str, Any], decode_case: CaseSpec | Mapping[str, Any], *,
    total_dies: int, prompt_tokens: int = 4096, output_tokens: int = 1024,
    decode_context: int | None = None, ttft_sla_ms: float = 3000.0,
    tpot_sla_ms: float = 30.0, kv_transfer_bw_gbps: float = 100.0,
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
    if decode_tokens_needed <= 0:
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
        d_sla_res = solve_hardware_sla_throughput(
            d_parsed, sla_ms=max(step_budget_ms, 1e-9) if step_budget_ms > 0.0 else tpot_sla_ms * max(1.0, float(d_emit_est)),
            context=eff_decode_ctx, phase="decode", output_budget=n_output, scenarios=[DEFAULT_SLA_SCENARIOS[0]],
        )
        d_scen = d_sla_res["scenarios"][0]
        emit_tokens = float(d_sla_res["emitted_tokens_per_seq"])
        decode_len_ok = bool(max(n_prompt, eff_decode_ctx) + n_output <= d_parsed.runtime.max_model_len)
        b_vram_max = int(d_scen["b_vram_max"]) if decode_len_ok else 0
        b_sla = int(d_scen["b_sla"]) if decode_budget_ms > 0.0 else 0
        b_feasible = min(b_sla, b_vram_max)
        step_ms_at_b1 = float(d_scen["step_ms_at_b1"])
        decode_step_ms = float(d_scen["step_ms_feasible"]) if b_feasible >= 1 else step_ms_at_b1
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
    for n_p in range(1, (n_total_dies // d_p) + 1):
        max_nd = (n_total_dies - n_p * d_p) // d_d
        for n_d in range(1, max_nd + 1):
            used_dies = n_p * d_p + n_d * d_d
            r_p = n_p * rps_p1
            r_kv = r_p if decode_tokens_needed <= 0 else n_p * rps_kv1
            r_d = r_p if decode_tokens_needed <= 0 else n_d * rps_d1
            feasible = bool(infeasible_reason is None and min(r_p, r_kv, r_d) > 0.0)
            r_sys = min(r_p, r_kv, r_d) if feasible else 0.0
            sustainable_tps = r_sys * float(n_output)
            if decode_tokens_needed <= 0:
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
    }


__all__ = ["compute_pareto_frontier", "enumerate_parallel_candidates", "sweep_pd_disaggregation_ratios", "sweep_topology_grid"]
