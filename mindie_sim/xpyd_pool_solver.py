"""xPyD pool sizing with stable request rates and simultaneous fabric cuts.

[Role]
  Compute sharding-independent KV/index/state transfer bytes, enumerate
  ``(N_P, Config_P, N_D, Config_D)`` under a total NPU budget, enforce all
  source/destination HBM, NIC, RoCE and staging cuts with progressive max-min
  allocation, and maximize SLO-qualified completed-request goodput below a
  strict utilization cap.

[Invariants]
  * ``mu_P``/``mu_D`` are completed requests/s; goodput is that rate times
    ``OSL``, never a token/s value compared with requests/s.
  * Decode HBM uses the admitted batch and full ``ISL+OSL`` span.
  * Every candidate passes the per-rank HBM ledger; no feasible row means
    ``best_configuration=None``.
  * The unique transfer ledger is independent of P/D sharding; physical wire
    additionally includes every destination copy and per-message header.
  * A P shard must have a connected declared route to its target D shard, and
    the KV transfer rate is bounded by that path's slowest endpoint/switch cut.
  * ``utilization_cap`` is an upper bound, not an operating point: the solver
    finds the maximum admitted ``lambda`` whose M/D/1 mean TTFT/TPOT and every
    ``rho`` cut satisfy the SLO, instead of testing only ``rho == cap``.
  * Each rate reserves P-HBM, P-NIC, RoCE, D-NIC, D-HBM, KV wire and staging
    together.  Analytic mean admission is labelled separately from any sampled
    tail guarantee.

[Timeouts & Locks] Pure bounded arithmetic over a finite arrival-rate grid.
[Side-Effect Policy] Fresh immutable rows; caller Cases are not mutated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

from mindie_sim.fabric_topology import HierarchicalFabricTree
from mindie_sim.parallel_bridge import bridge_lifetime_plan, kv_rank_fraction
from mindie_sim.parallel_context import bridge_remap_bytes, phase_bridge
from mindie_sim.parallel_cuts import (cut_ceiling, cut_inventory,
                                      merge_demands)
from mindie_sim.parallel_ep_knee import execution_context, validate_slot_vector
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.xpyd_cut_composer import (compose_pool_cut_demands,
                                         routed_rates_within_capacity)
from mindie_sim.xpyd_flow import Flow, progressive_max_min, queue_wait_ms
from mindie_sim.xpyd_draft import draft_geometry_error, raw_shared_geometry_reason
from mindie_sim.xpyd_pool_reports import geometry_rejection, instance_evidence, pool_failure
from mindie_sim.xpyd_instances import (_LanePlan, _default_configs,
                                       p_to_d_path, p_to_d_shared_cuts,
                                       pool_config_facts)
from mindie_sim.xpyd_transfer import (_fraction, _int, _nic_bw_gbps, _positive,
                                      _zero_transfer_handoff,
                                      completed_requests_per_s,
                                      kv_transfer_contention,
                                      logical_transfer_bytes)

DEFAULT_RHO = 0.9; _ZERO_I = 0; _ONE_I = 1; _MAX_INSTANCES = 16; _ZERO_F = 0.0; _ONE_F = 1.0; _TRANSFER_LAT_MS = 0.1

def  solve_xpyd_pools(
    case: CaseSpec | Mapping[str, Any], *, total_npus: int,
    isl: int, osl: int, prefix_hit_rate: float = _ZERO_F,
    ttft_slo_ms: float, tpot_slo_ms: float,
    prefill_configs: Sequence[ParallelSpec | Mapping[str, Any]] | None = None,
    decode_configs: Sequence[ParallelSpec | Mapping[str, Any]] | None = None,
    nic_bw_gbps: float | None = None,
    hccl_buffsize_mb: float | None = None,
    routing_skew: float = _ONE_F,
    eplb_redundant_experts: int = _ZERO_I,
    max_pool_instances: int = _MAX_INSTANCES,
    utilization_cap: float = DEFAULT_RHO,
    roce_bw_gbps: float | None = None,
    roce_oversubscription: float = _ONE_F,
    staging_bytes: float | None = None,
    transfer_latency_ms: float = _TRANSFER_LAT_MS,
    destination_cached_tokens: int = _ZERO_I,
    eplb_redundant_slots_per_rank: Sequence[int] | None = None,
    allow_unqualified: bool = False,
    require_prefix_identity: bool = False,
    prefix_digest: str | None = None,
    prefix_tokens: Sequence[int] | None = None,
    source_prefix_hash: str | None = None,
    destination_prefix_hash: str | None = None,
    source_cache_epoch: int | None = None,
    destination_cache_epoch: int | None = None,
    source_page_ids: Sequence[int] | None = None,
    destination_page_ids: Sequence[int] | None = None,
    hccl_buffsize_bytes: int | None = None,
    comm_arena_qualified: bool = False,
    comm_arena_domains: Mapping[str, float] | None = None,
    strict_memory_qualification: bool = False,
    eplb_rebalance_peak_bytes: float = _ZERO_F,
) -> dict[str, Any]:
    """Enumerate pools and maximize SLO-qualified completed-request goodput."""
    geometry_reason = raw_shared_geometry_reason(case)
    if geometry_reason is not None:
        return geometry_rejection(geometry_reason, total_npus, isl, osl)
    parsed = parse_case(case)
    requested_total = _int("total_npus", total_npus)
    physical = len(parsed.machine.dies)
    oversubscribed = requested_total > physical
    total = min(requested_total, physical)
    prompt, output = _int("isl", isl), _int("osl", osl)
    slot_vector = validate_slot_vector(eplb_redundant_slots_per_rank)
    redundant_experts = _int("eplb_redundant_experts", eplb_redundant_experts,
                             minimum=0)
    hit = _fraction("prefix_hit_rate", prefix_hit_rate)
    ttft = _positive("ttft_slo_ms", ttft_slo_ms)
    tpot = _positive("tpot_slo_ms", tpot_slo_ms)
    cap = _int("max_pool_instances", max_pool_instances)
    rho_cap = _positive("utilization_cap", utilization_cap)
    eplb_peak = _positive("eplb_rebalance_peak_bytes",
                          eplb_rebalance_peak_bytes, strict=False)
    if rho_cap >= 1.0:
        raise ModelingError(f"xpyd_utilization_cap_not_stable:{rho_cap}")
    if nic_bw_gbps is not None:
        nic = _positive("nic_bw_gbps", nic_bw_gbps)
    else:
        nic = _nic_bw_gbps(parsed)
    tree = HierarchicalFabricTree.from_hardware(parsed.machine)
    p_source = (_default_configs(parsed, "prefill", total)
                if prefill_configs is None else prefill_configs)
    d_source = (_default_configs(parsed, "decode", total)
                if decode_configs is None else decode_configs)
    p_specs = [spec if isinstance(spec, ParallelSpec) else ParallelSpec.from_mapping(spec)
               for spec in p_source]
    d_specs = [spec if isinstance(spec, ParallelSpec) else ParallelSpec.from_mapping(spec)
               for spec in d_source]
    eplb_on = bool(redundant_experts > 0
                   or any(value > 0 for value in (slot_vector or ())))
    def _facts(spec: ParallelSpec, phase: str) -> dict[str, Any]:
        try:
            return pool_config_facts(
                parsed, spec, phase=phase, isl=prompt, osl=output,
                prefix_hit_rate=hit, tpot_slo_ms=tpot,
                hccl_buffsize_mb=hccl_buffsize_mb, routing_skew=routing_skew,
                eplb_redundant_experts=redundant_experts,
                enable_eplb=eplb_on, tree=tree, rho=rho_cap,
                eplb_redundant_slots_per_rank=slot_vector,
                allow_unqualified=allow_unqualified,
                staging_bytes=staging_bytes,
                hccl_buffsize_bytes=hccl_buffsize_bytes,
                comm_arena_qualified=comm_arena_qualified,
                comm_arena_domains=comm_arena_domains,
                strict_memory_qualification=strict_memory_qualification)
        except ModelingError as exc:
            return {"parallel": spec, "feasible": False, "reason": str(exc),
                    "world": spec.world_size, "mu_requests_per_s": 0.0,
                    "route_qualified": False}

    p_rows = [_facts(spec, "prefill") for spec in p_specs]
    d_rows = [_facts(spec, "decode") for spec in d_specs]
    p_candidates = [row for row in p_rows if row.get("plan") and row.get("world")]
    d_candidates = [row for row in d_rows if row.get("plan") and row.get("world")]
    requested_cached = _int("destination_cached_tokens",
                            destination_cached_tokens, minimum=0)
    mtp = getattr(parsed.model.auxiliary, "mtp", None)
    kv_case = (replace(parsed, model=replace(
        parsed.model, auxiliary=replace(parsed.model.auxiliary, mtp=None)))
        if mtp is not None and draft_geometry_error(mtp) is not None else parsed)
    try:
        transfer = logical_transfer_bytes(
            kv_case, prompt_tokens=prompt, output_tokens=output,
            destination_cached_tokens=requested_cached,
            prefix_digest=prefix_digest, prefix_tokens=prefix_tokens,
            source_prefix_hash=source_prefix_hash,
            destination_prefix_hash=destination_prefix_hash,
            source_epoch=source_cache_epoch,
            destination_epoch=destination_cache_epoch,
            source_page_ids=source_page_ids,
            destination_page_ids=destination_page_ids)
    except ModelingError as exc:
        return {
            "schema": "mindie-sim.xpyd-pools/3", "total_npus": requested_total,
            "physical_npus": physical, "allocation_npus": total,
            "isl": prompt, "osl": output, "prefix_hit_rate": hit,
            "ttft_slo_ms": ttft, "tpot_slo_ms": tpot,
            "utilization_cap": rho_cap, "nic_bw_gbps": nic,
            "kv_transfer": None, "kv_bytes_per_request": 0.0,
            "kv_cache_identity_verified": False,
            "prefix_identity_verified": False,
            "source_page_ids": list(source_page_ids or ()),
            "destination_page_ids": list(destination_page_ids or ()),
            "source_cache_epoch": source_cache_epoch,
            "destination_cache_epoch": destination_cache_epoch,
            "kv_zero_transfer_claimed": False,
            "require_prefix_identity": bool(require_prefix_identity),
            "prefill_configs": [], "decode_configs": [],
            "rows": [], "feasible_rows": [], "pareto_frontier": [],
            "best_configuration": None, "qualified": False,
            "allow_unqualified": bool(allow_unqualified),
            "reason": str(exc), "status": "no_feasible_configuration",
            "qualification_reason": str(exc),
            "slo_qualification_mode": "analytic_md1_mean",
            "tail_quantile_certified": False,
            "selection": "max_stable_slo_qualified_completed_requests_after_all_memory_and_flow_cuts",
        }
    kv = float(transfer["transfer_total_bytes"])
    zero_transfer_claimed = bool(kv <= 0.0
                                 and transfer["destination_cached_tokens"] > 0)
    identity_verified = bool(transfer["prefix_identity_verified"])
    identity_gate = requested_cached > 0
    identity_ok = bool((not identity_gate) or identity_verified)
    unverified_zero_transfer = bool(zero_transfer_claimed
                                    and not transfer["zero_transfer_qualified"])
    hbm_bw = float(min(dict(die.rates)["hbm"] for die in parsed.machine.dies))
    base_report = {
        "schema": "mindie-sim.xpyd-pools/3", "total_npus": requested_total,
        "physical_npus": physical, "allocation_npus": total,
        "isl": prompt, "osl": output, "prefix_hit_rate": hit,
        "ttft_slo_ms": ttft, "tpot_slo_ms": tpot,
        "utilization_cap": rho_cap, "nic_bw_gbps": nic,
        "kv_transfer": transfer, "kv_bytes_per_request": kv,
        "kv_cache_identity_verified": identity_ok,
        "prefix_identity_verified": identity_verified,
        "source_page_ids": list(source_page_ids or ()),
        "destination_page_ids": list(destination_page_ids or ()),
        "source_cache_epoch": source_cache_epoch,
        "destination_cache_epoch": destination_cache_epoch,
        "kv_zero_transfer_claimed": zero_transfer_claimed,
        "require_prefix_identity": bool(require_prefix_identity),
        "prefill_configs": p_rows, "decode_configs": d_rows,
        "rows": [], "feasible_rows": [], "pareto_frontier": [],
        "best_configuration": None, "qualified": False,
        "allow_unqualified": bool(allow_unqualified),
        "reason": "no_feasible_configuration",
        "status": "no_feasible_configuration",
        "slo_qualification_mode": "analytic_md1_mean",
        "tail_quantile_certified": False,
        "selection": "max_stable_slo_qualified_completed_requests_after_all_memory_and_flow_cuts",
    }
    if oversubscribed:
        base_report["reason"] = "total_npus_exceeds_physical_devices"
        base_report["status"] = "no_feasible_configuration"
        return base_report
    if require_prefix_identity and identity_gate and not identity_verified:
        # A caller-declared full destination cache is not a content-verified
        # prefix; fail closed when identity evidence is required.
        base_report["reason"] = "kv_cache_identity_unverified"
        base_report["qualification_reason"] = "kv_cache_identity_unverified"
        base_report["status"] = "no_feasible_configuration"
        return base_report
    if not p_candidates or not d_candidates:
        base_report["reason"] = ("no_feasible_prefill_configuration"
                                 if not p_candidates
                                 else "no_feasible_decode_configuration")
        return base_report
    rows: list[dict[str, Any]] = []
    bridge_failures: list[dict[str, Any]] = []
    instance_cache: dict[tuple[int, str, int, int], dict[str, Any]] = {}

    def _instance_row(spec: ParallelSpec, phase: str, start: int,
                      world: int, transfer_plan: Any = None,
                      plan_key: Any = None) -> dict[str, Any]:
        key = (id(spec), phase, int(start), int(world),
               plan_key if plan_key is not None else id(transfer_plan))
        cached = instance_cache.get(key)
        if cached is not None:
            return cached
        dies = tuple(parsed.machine.dies[start:start + world])
        if len(dies) != world:
            row = {"parallel": spec, "feasible": False,
                   "reason": "instance_exceeds_physical_devices",
                   "world": world, "mu_requests_per_s": 0.0}
        else:
            sliced = replace(parsed, machine=replace(parsed.machine, dies=dies))
            try:
                row = pool_config_facts(
                    sliced, spec, phase=phase, isl=prompt, osl=output,
                    prefix_hit_rate=hit, tpot_slo_ms=tpot,
                    hccl_buffsize_mb=hccl_buffsize_mb, routing_skew=routing_skew,
                    eplb_redundant_experts=redundant_experts,
                    enable_eplb=eplb_on, tree=tree, rho=rho_cap,
                    eplb_redundant_slots_per_rank=slot_vector,
                    allow_unqualified=allow_unqualified,
                    staging_bytes=staging_bytes, rank_offset=int(start),
                    transfer_plan=transfer_plan,
                    hccl_buffsize_bytes=hccl_buffsize_bytes,
                    comm_arena_qualified=comm_arena_qualified,
                    comm_arena_domains=comm_arena_domains,
                    strict_memory_qualification=strict_memory_qualification)
            except ModelingError as exc:
                row = {"parallel": spec, "feasible": False,
                       "reason": str(exc), "world": world,
                       "mu_requests_per_s": 0.0, "route_qualified": False}
        row["device_ids"] = tuple(range(int(start), int(start) + int(world)))
        instance_cache[key] = row
        return row

    evidence = execution_context(parsed)
    path_cache: dict[tuple[int, ...], tuple[bool, str, float | None, dict]] = {}

    def p_to_d(p_devices: Sequence[int], d_devices: Sequence[int]):
        key = tuple(sorted({int(value) for value in p_devices}
                           | {int(value) for value in d_devices}))
        cached = path_cache.get(key)
        if cached is None:
            cached = (*p_to_d_path(tree, key, (), **evidence),
                      p_to_d_shared_cuts(tree, key))
            path_cache[key] = cached
        return cached

    for p_row in p_candidates:
        wp, plan_p = int(p_row["world"]), p_row["plan"]
        for d_row in d_candidates:
            wd, plan_d = int(d_row["world"]), d_row["plan"]
            # The remap is owned by the *destination* ownership plan: a
            replica_lanes = (str(plan_d.get("pcp_lane_policy",
                                            "park_extra_lanes"))
                             == "replica_lanes" and d_row["parallel"].active_pcp_lanes > 1)
            # ``transfer['page_ids']`` is the authoritative required universe;
            # supplied ownership evidence is reconciled against it, not
            # substituted for it.  Cached destination pages carry no handoff.
            page = max(1, int(transfer["page_size"]))
            cached_pages = tuple(range(
                int(transfer["destination_cached_tokens"]) // page))
            try:
                bridge = phase_bridge(
                    p_row["parallel"], d_row["parallel"],
                    kv_bytes=float(transfer["transferred_kv_bytes"]),
                    index_bytes=float(transfer["transferred_index_bytes"]),
                    state_bytes=float(transfer["fixed_state_transfer_bytes"]),
                    epoch=0, lane_bw_gbps=nic, replicated=replica_lanes,
                    page_ids=tuple(transfer["page_ids"]),
                    source_page_ids=source_page_ids,
                    destination_page_ids=destination_page_ids,
                    source_rank_fraction=kv_rank_fraction(
                        parsed.model, p_row["parallel"]),
                    destination_rank_fraction=kv_rank_fraction(
                        parsed.model, d_row["parallel"]),
                    cached_page_ids=cached_pages, layer_extents=transfer["layer_extents"])
                lifetime = bridge_lifetime_plan(
                    bridge, eplb_rebalance_peak_bytes=eplb_peak, case=parsed, transfer=transfer)
            except ModelingError as exc:
                # Explicit page evidence is checked by logical_transfer_bytes
                # and phase_bridge. A failure here belongs to this pair and
                # cannot discard a feasible sibling, regardless of order.
                bridge_failures.append({"prefill_parallel": p_row["parallel"].to_dict(),
                    "decode_parallel": d_row["parallel"].to_dict(), "feasible": False,
                    "qualified": False, "reason": str(exc)})
                continue
            remap_bridge = bridge.to_dict()
            edge_cache = {}
            life_dict = lifetime.to_dict()
            life_scalars = tuple((k, float(life_dict[k])) for k in ("source_retention_bytes", "destination_retention_bytes", "handoff_staging_bytes", "eplb_rebalance_peak_bytes") if life_dict.get(k) is not None)
            owners_key = tuple(int(r) for r in getattr(lifetime, "page_owner_ranks", ()) or ())
            p_key = ("prefill", float(lifetime.phase_peak_bytes("prefill")), tuple(lifetime.phase_rank_peak_bytes("prefill")), life_scalars, owners_key)
            d_key = ("decode", float(lifetime.phase_peak_bytes("decode")), tuple(lifetime.phase_rank_peak_bytes("decode")), life_scalars, owners_key)
            if replica_lanes:
                remap = bridge_remap_bytes(bridge)
                remap_ms = (float(remap["total_bytes"]) / (nic * 1e9) * 1000.0
                            if nic > 0 else 0.0)
            else:
                remap = {"total_bytes": 0.0, "destination_lanes": 0.0}
                remap_ms = 0.0
            for n_p in range(_ONE_I, min(cap, total // wp) + _ONE_I):
                p_ids = tuple(index * wp for index in range(n_p))
                for n_d in range(_ONE_I, min(cap, (total - n_p * wp) // wd) + _ONE_I):
                    d_ids = tuple(n_p * wp + index * wd for index in range(n_d))
                    p_inst = [_instance_row(p_row["parallel"], "prefill", start, wp, lifetime, p_key) for start in p_ids]
                    d_inst = [_instance_row(d_row["parallel"], "decode", start, wd, lifetime, d_key) for start in d_ids]
                    bad = next((item for item in p_inst + d_inst
                                if not item.get("feasible")
                                or (not allow_unqualified
                                    and not item.get("route_qualified", True))), None)
                    if bad is not None:
                        rows.append(pool_failure(total,
                            n_p, n_d, wp, wd,
                            str(bad.get("reason") or "instance_unqualified"),
                            p_ids, d_ids, p_inst, d_inst))
                        continue
                    # The P->D handoff itself must be connected on the declared
                    # fabric; per-instance membership is not a path proof.
                    path_ok, path_evidence, path_bw, shared_cuts = p_to_d(
                        tuple(rank for item in p_inst
                              for rank in item.get("device_ids", ())),
                        tuple(rank for item in d_inst
                              for rank in item.get("device_ids", ())))
                    if not path_ok and not allow_unqualified:
                        failure = pool_failure(total,
                            n_p, n_d, wp, wd,
                            f"p_to_d_route_unqualified:{path_evidence}",
                            p_ids, d_ids, p_inst, d_inst)
                        failure["route_qualified"] = False
                        rows.append(failure)
                        continue
                    effective_nic = (nic if (path_bw is None or path_bw <= 0.0)
                                     else min(nic, path_bw))
                    shared_cut = (min(shared_cuts.values())
                                  if shared_cuts else None)
                    # Compose every byte class on the physical resource each
                    # *placed* instance actually traverses, weighted by the
                    # capacity-proportional request share it is routed.  The
                    # rank-zero templates p_row/d_row are never substituted for
                    # the assigned physical instances.
                    p_cap = float(sum(float(item["mu_requests_per_s"])
                                      for item in p_inst))
                    d_cap = float(sum(float(item["mu_requests_per_s"])
                                      for item in d_inst))
                    if p_cap <= 0.0 or d_cap <= 0.0:
                        rows.append(pool_failure(total,
                            n_p, n_d, wp, wd, "zero_instance_capacity",
                            p_ids, d_ids, p_inst, d_inst))
                        continue
                    composition = compose_pool_cut_demands(
                        tree, p_inst, d_inst, transfer_bytes=float(kv),
                        remap_bytes=float(remap["total_bytes"]),
                        bridge=bridge, transfer=transfer, case=parsed, edge_cache=edge_cache)
                    if not composition["qualified"]:
                        rows.append(pool_failure(total,
                            n_p, n_d, wp, wd,
                            str(composition.get("reason") or
                                "zero_instance_capacity"),
                            p_ids, d_ids, p_inst, d_inst))
                        continue
                    composed = composition["demands"]
                    transfer_wire = composition["wire_bytes"]
                    transfer_demands = merge_demands(
                        composition["handoff_demands"],
                        composition["remap_demands"])
                    port_demands = {key: float(value)
                                    for key, value in composed.items()
                                    if key.startswith("port:")}
                    port_capacities = {key: float(value) for key, value
                                       in cut_inventory(tree).items()
                                       if key.startswith("port:")}
                    if transfer_wire > 0.0 and not transfer_demands and shared_cuts:
                        # Injected tree without a peer route: keep the declared
                        # shared-cut identity rather than silently dropping the
                        # transfer bytes.  Routing shares sum to one, so one
                        # pool request still moves kv + remap.
                        transfer_demands = {
                            name: transfer_wire
                            for name in shared_cuts}
                        composed = merge_demands(composed, transfer_demands)
                    cut_rps, limiting_cut = cut_ceiling(tree, composed)
                    p_service = max(float(item["step_ms"]) for item in p_inst)
                    d_step = max(float(item["step_ms"]) for item in d_inst)
                    raw = min(p_cap, d_cap)

                    def handoff_at(rate: float) -> dict[str, Any]:
                        if transfer_wire <= 0.0:
                            return _zero_transfer_handoff(float(d_step))
                        return kv_transfer_contention(
                            transfer_wire, n_prefill=n_p, n_decode=n_d,
                            nic_bw_gbps=effective_nic,
                            hbm_bw_bytes_per_s=hbm_bw, p_world=wp, d_world=wd,
                            decode_step_ms=float(d_step), osl=output,
                            requests_per_s=rate, roce_bw_gbps=roce_bw_gbps,
                            roce_oversubscription=roce_oversubscription,
                            staging_bytes=staging_bytes,
                            transfer_latency_ms=transfer_latency_ms,
                            utilization_cap=rho_cap,
                            shared_cut_bytes_per_s=shared_cut,
                            port_demands=port_demands,
                            port_capacities=port_capacities)

                    margin = 1.0 - 1e-9
                    state: dict[str, Any] = {"metrics": None, "reason": None}

                    def evaluate(rate: float) -> dict[str, Any] | None:
                        handoff = handoff_at(rate)
                        jitter = float(handoff["jitter_ratio"])
                        if not math.isfinite(jitter):
                            state["reason"] = "decode_write_fraction_saturated"
                            return None
                        d_eff = d_cap / (1.0 + jitter)
                        wire_eff = float(handoff["wire_requests_per_s"])
                        p_eff = p_cap * max(
                            0.0, 1.0 - min(0.999,
                                           float(handoff["p_source_fraction"])))
                        cap_eff = (wire_eff * margin
                                   if math.isfinite(wire_eff) else d_eff)
                        rho_p = rate / p_eff if p_eff > 0 else float("inf")
                        rho_d = rate / d_eff if d_eff > 0 else float("inf")
                        queue_p = queue_wait_ms(rho_p, p_service)
                        queue_d = queue_wait_ms(rho_d, d_step)
                        queue_kv = float(handoff.get("queue_ms") or 0.0)
                        ttft_ms = (queue_p + p_service
                                   + float(handoff["transfer_ms"])
                                   + float(handoff["handoff_ms"]) + queue_kv
                                   + queue_d + d_step + remap_ms)
                        tpot_ms = (float(handoff["tpot_jitter_ms"] or float("inf"))
                                   + queue_d / max(1, output))
                        metrics = (rate, rho_p, rho_d, queue_p, queue_d,
                                   ttft_ms, tpot_ms, handoff, d_eff, jitter,
                                   queue_kv)
                        if (state["metrics"] is None
                                or rate > state["metrics"][0]):
                            state["metrics"] = metrics
                        if not math.isfinite(queue_kv):
                            state["reason"] = "kv_transfer_wire_saturated"
                            return None
                        if rate > cap_eff + 1e-12:
                            state["reason"] = "kv_wire_capacity_exceeded"
                            return None
                        if rate > rho_cap * raw + 1e-12:
                            state["reason"] = "utilization_cap_exceeded"
                            return None
                        if math.isfinite(cut_rps) and rate > (
                                rho_cap * cut_rps * margin):
                            state["reason"] = "shared_cut_capacity_exceeded"
                            return None
                        if rho_p > rho_cap + 1e-12 or rho_d > rho_cap + 1e-12:
                            state["reason"] = "utilization_cap_exceeded"
                            return None
                        if not handoff["feasible"]:
                            state["reason"] = (handoff.get("reason")
                                               or "flow_infeasible")
                            return None
                        if ttft_ms > ttft:
                            state["reason"] = "ttft_slo_exceeded"
                            return None
                        if tpot_ms > tpot:
                            state["reason"] = "tpot_slo_exceeded"
                            return None
                        state["reason"] = None
                        return metrics

                    # ``rho_cap`` is an upper bound.  TTFT(lambda), TPOT(lambda)
                    # and every rho are monotone in lambda, so the maximum
                    # admitted rate is the feasibility frontier of a 60-step
                    # bisection -- not the single point ``rho == cap``.
                    high = max(0.0, raw)
                    evaluate(high)
                    low = 1e-12
                    feasible_metrics = evaluate(low)
                    if feasible_metrics is not None:
                        for _ in range(60):
                            middle = 0.5 * (low + high)
                            if middle <= low or middle >= high:
                                break
                            trial_metrics = evaluate(middle)
                            if trial_metrics is not None:
                                low = middle
                                feasible_metrics = trial_metrics
                            else:
                                high = middle
                    feasible = feasible_metrics is not None
                    if feasible:
                        # Capacity-proportional routing guarantees each routed
                        # instance rate is below its own capacity whenever the
                        # aggregate rate is below ``raw``; prove it from the
                        # exact placed instance demands rather than assuming it.
                        rates_ok, rates_reason = routed_rates_within_capacity(
                            tree, p_inst, d_inst,
                            admitted_rps=float(feasible_metrics[0]))
                        if not rates_ok:
                            raise ModelingError(
                                rates_reason or
                                "xpyd_routed_instance_capacity_unproven")
                        (admitted, rho_p, rho_d, queue_p, queue_d,
                         ttft_ms, tpot_ms, handoff, d_eff, jitter,
                         queue_kv) = feasible_metrics
                        rho = admitted / raw if raw > 0 else 0.0
                        last_reason = None
                    else:
                        last_reason = state["reason"] or "no_stable_admitted_rate"
                        if state["metrics"] is not None:
                            (_target, rho_p, rho_d, queue_p, queue_d,
                             ttft_ms, tpot_ms, handoff, d_eff, jitter,
                             queue_kv) = state["metrics"]
                            admitted = 0.0
                            rho = 0.0
                        else:
                            rho = admitted = rho_p = rho_d = 0.0
                            queue_p = queue_d = queue_kv = 0.0
                            d_eff = 0.0
                            jitter = float("inf")
                            ttft_ms = float("inf")
                            tpot_ms = float("inf")
                            handoff = {"transfer_ms": float("inf"),
                                       "jitter_ratio": float("inf"),
                                       "wire_requests_per_s": 0.0,
                                       "saturated_resources": []}
                    rows.append({
                        "n_prefill": n_p, "n_decode": n_d,
                        "pd_ratio": f"{n_p}P:{n_d}D",
                        "used_npus": n_p * wp + n_d * wd,
                        "idle_npus": total - n_p * wp - n_d * wd,
                        "prefill_parallel": p_row["parallel"].to_dict(),
                        "decode_parallel": d_row["parallel"].to_dict(),
                        "prefill_world": wp, "decode_world": wd,
                        "prefill_device_ids": list(p_ids),
                        "decode_device_ids": list(d_ids),
                        "prefill_instance_rows": instance_evidence(p_inst),
                        "decode_instance_rows": instance_evidence(d_inst),
                        "prefill_step_ms": p_service,
                        "prefill_service_ms": p_service,
                        "prefill_chunks": max(int(item.get("prefill_steps", 1))
                                              for item in p_inst),
                        "decode_step_ms": d_step,
                        "prefill_batch": p_row.get("batch"),
                        "decode_batch": d_row.get("b_sla"),
                        "prefill_rps": p_cap, "decode_rps": d_cap,
                        "decode_effective_rps": d_eff,
                        "raw_capacity_rps": raw, "admitted_rps": admitted,
                        "utilization": (admitted / raw if raw > 0 else 0.0),
                        "utilization_target": (admitted / raw if raw > 0 else 0.0),
                        "rho_prefill": rho_p, "rho_decode": rho_d,
                        "kv_wire_rps": handoff["wire_requests_per_s"],
                        "kv_nic_gbps": effective_nic,
                        "p_to_d_path_evidence": path_evidence,
                        "d_hbm_jitter": jitter,
                        "transfer_ms": handoff["transfer_ms"],
                        "transfer_queue_ms": queue_kv,
                        "handoff_ms": handoff.get("handoff_ms", 0.0),
                        "queue_prefill_ms": queue_p, "queue_decode_ms": queue_d,
                        "remap_ms": remap_ms,
                        "remap_bytes": composition["remap_total_bytes"],
                        "unique_logical_bytes": composition["unique_logical_bytes"],
                        "destination_replica_bytes": composition["destination_replica_bytes"],
                        "handoff_header_bytes": composition["handoff_header_bytes"],
                        "wire_bytes": transfer_wire,
                        "remap_bridge": remap_bridge,
                        "bridge_lifetime_plan": lifetime.to_dict(),
                        "bridge_prefill_staging_bytes":
                            lifetime.phase_peak_bytes("prefill"),
                        "bridge_decode_staging_bytes":
                            lifetime.phase_peak_bytes("decode"),
                        "shared_cut_demands": composed,
                        "prefill_instance_share": composition["alpha"],
                        "decode_instance_share": composition["beta"],
                        "handoff_cut_demands": composition["handoff_demands"],
                        "remap_cut_demands": composition["remap_demands"],
                        "shared_cut_bytes_per_output_token": {
                            key: value / float(output)
                            for key, value in composed.items()},
                        "shared_cut_rps": (cut_rps if math.isfinite(cut_rps)
                                           else None),
                        "shared_cut_limiting_resource": limiting_cut,
                        "shared_cut_utilization": (
                            admitted / cut_rps
                            if math.isfinite(cut_rps) and cut_rps > 0 else 0.0),
                        "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
                        "goodput_rps": admitted,
                        "goodput_output_tps": admitted * output,
                        "goodput_total_tps": admitted * (prompt + output),
                        "goodput_tps": admitted * output,
                        "goodput_tps_per_npu": admitted * output / total,
                        "saturated_resources": list(
                            handoff.get("saturated_resources", ())),
                        "kv_cache_identity_verified": identity_ok,
                        "route_qualified": bool(
                            path_ok and all(item.get("route_qualified", True)
                                            for item in p_inst + d_inst)),
                        "feasible": feasible, "slo_feasible": feasible,
                        "qualified": bool(feasible and identity_ok
                                          and not allow_unqualified
                                          and path_ok
                                          and all(item.get("route_qualified", True)
                                                  for item in p_inst + d_inst)),
                        "slo_qualification_mode": "analytic_md1_mean",
                        "tail_quantile_certified": False,
                        "reason": None if feasible else last_reason,
                    })
    feasible_rows = [row for row in rows if row["feasible"]]
    for row in feasible_rows:
        row["pareto_optimal"] = not any(
            other is not row
            and other["goodput_output_tps"] >= row["goodput_output_tps"] - 1e-12
            and other["used_npus"] <= row["used_npus"]
            and (other["goodput_output_tps"] > row["goodput_output_tps"] + 1e-12
                 or other["used_npus"] < row["used_npus"])
            for other in feasible_rows)
    for row in rows:
        row.setdefault("pareto_optimal", False)
    qualified_rows = [row for row in feasible_rows if row["qualified"]]
    best = max(qualified_rows or feasible_rows,
               key=lambda row: (row["goodput_output_tps"], -row["used_npus"],
                                -row["n_prefill"], -row["n_decode"],
                                row["decode_parallel"]["pcp_lane_policy"] == "park_extra_lanes"),
               default=None)
    top_qualified = bool(best is not None and best.get("qualified"))
    reason = (None if best is not None else "all_pool_candidates_rejected")
    if bridge_failures:
        base_report["candidate_pair_rejections"] = bridge_failures
        if best is None and len(bridge_failures) == len(p_candidates) * len(d_candidates):
            reason = bridge_failures[0]["reason"]
    if best is not None and not identity_ok:
        reason = "kv_cache_identity_unverified"
    base_report.update({
        "rows": rows, "feasible_rows": feasible_rows,
        "pareto_frontier": [row for row in rows if row["pareto_optimal"]],
        "best_configuration": best,
        "status": "feasible" if best is not None else "no_feasible_configuration",
        "qualified": top_qualified,
        "qualification_reason": (None if top_qualified else
                                 reason or "uncertified_candidate"),
        "reason": reason,
    })
    return base_report

__all__ = ["DEFAULT_RHO", "Flow", "completed_requests_per_s", "kv_transfer_contention",
           "logical_transfer_bytes", "pool_config_facts", "progressive_max_min",
           "queue_wait_ms", "solve_xpyd_pools"]
