"""Decode cost model, causal two-lane schedule and per-step breakdown.

[Role]
  Price one decode step from declared model geometry, the fabric tree and the
  MoE straggler model.  The step record exposes the useful-work ledger (global
  tokens, assignments, per-owner FLOPs) and the explicit PP stage durations.

[Invariants]
  * Global tokens entering the MoE domain are ``DP * B_per_dp_rank``; per-rank
    assignments are ``T_global*top_k/(EP*EDP)`` and every one is executed on a
    ``1/ETP`` slice of the expert matrix, so summing over ``EP*EDP*ETP`` ranks
    conserves ``T_global*top_k*6*H*I`` FLOPs identically for every admitted
    ``E/EP >= ETP`` or ``E/EP < ETP`` geometry.
  * The straggler ratio compares EP-owner assignments with the EP-owner mean,
    never an EP-owner numerator against an ``ETP*EDP`` denominator.
  * Dispatch -> Expert -> Combine is causal *within* a microbatch; attention of
    layer ``l+1`` waits for layer ``l``'s Combine.  A non-serial mode is priced
    by the explicit two-resource dependency schedule, never a scalar subtraction.
  * PP latency comes from the exact ``(microbatch, stage)`` trace; one
    microbatch pays only the point-to-point hops and no extra fill.
  * ``eplb_rebalance_ms`` is charged exactly once and staging bytes are not
    re-charged as per-step FFN weight reads.
  * The step also emits one executed communication ledger: TP/ETP/EDP/PCP/DCP
    groups, separate MoE dispatch/combine or one MC2 call, and every PP hop.
  * L1/UB zero forces the uncached fallback and a strictly larger step.

[Timeouts & Locks] Pure bounded arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Fresh immutable records; caller Cases untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.nr_s7_validate import (
    _validate_nested, _validate_record_domains, strict_number, typed_inputs, typed_sequence,
)
from mindie_sim.catalog.cache import state_bytes_per_request_per_rank
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import stage_weight_terms
from mindie_sim.fabric_topology import HierarchicalFabricTree
from mindie_sim.moe_skew_eplb import eplb_effect, max_rank_tokens
from mindie_sim.parallel_context import dcp_merge_payloads, model_families
from mindie_sim.parallel_comm_ledger import (CommLedgerBuilder,
                                             merge_ledgers, reduce_route_evidence)
from mindie_sim.parallel_stage_ops import (stage_groups, stage_layer_slices, axis_layer_indices)
from mindie_sim.parallel_cuts import (aggregate_switch_bound,
                                      collective_cut_demands, merge_demands,
                                      switch_bound_from_demands,
                                      unique_cut_demands)
from mindie_sim.parallel_schedule import (layer_partition, pp_bubble_ms,
                                          pp_microbatch_schedule_ms,
                                          pp_pipeline_ms, stage_durations,
                                          two_stream_schedule,
                                          validate_overlap_mode)
from mindie_sim.prediction.capacity import Memory
from mindie_sim.parallel_step_records import (AXIS_COLLECTIVES,
                                             SCALING_MODES, StepBreakdown,
                                             _int, _interconnect_bw_gbps,
                                             _positive, check_route_qualified,
                                             execution_context, iter_axis_groups,
                                             validate_endpoint_map)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes

ZERO = 0
ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
HALF = 0.5
THOUSAND_MS = 1000.0
DEFAULT_CONTEXT_TOKENS = 4096



@typed_inputs
@dataclass(frozen=True)
class DecodeCostModel:
    """Roofline step model with a causal two-lane communication schedule."""

    case: CaseSpec
    non_expert_bytes: float
    expert_bytes: float
    non_expert_params: float
    expert_params: float
    hbm_bw: float
    cube_peak: float
    vector_peak: float
    launch_ms: float
    layers: int
    moe_layers: int
    top_k: int
    num_experts: int
    kv_per_token: float
    state_per_request: float
    hidden: int
    element_bytes: float
    attn_heads_width: float
    context_tokens: int = 4096
    tp: int = 1
    pp: int = 1
    dp: int = 1
    pcp: int = 1
    dcp: int = 1
    ep: int = 1
    kv_heads: int = 1
    head_dim: int = 128
    families: tuple[str, ...] = ()
    etp: int = 1
    expert_dp: int = 1
    moe_mask: tuple[bool, ...] = ()
    query_heads: int = 1
    l1_bytes: float = 0.0
    ub_bytes: float = 0.0
    tile_tokens: int = 0
    uncached_fallback: bool = False
    mlapo_prefetch: bool = True
    #: First physical group per axis as local-rank offsets of the instance:
    #: ``(name, offsets)``.  Timing must price a real communicator group, not
    #: every device of a multi-replica instance.
    axis_endpoint_offsets: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        _validate_record_domains(self)

    @classmethod
    def from_case(cls, case: CaseSpec | Mapping[str, Any], *,
                  context_tokens: int = DEFAULT_CONTEXT_TOKENS) -> "DecodeCostModel":
        _int("context_tokens", context_tokens)
        parsed = parse_case(case)
        _validate_nested(parsed, "case")
        model, p = parsed.model, parsed.runtime.parallel
        quant = model.quant or QuantSpec("bf16")
        non_bytes = expert_bytes = non_params = expert_params = 0.0
        heads_width = 0.0
        for terms in stage_weight_terms(model, p, include_draft=False,
                                        include_engram_tables=not parsed.runtime.engram_cpu_offload,
                                        layer_partition=parsed.runtime.pp_layer_partition or None):
            for term in terms:
                value = float(term_bytes(term, quant, p, parsed.machine.ranks_per_node))
                shard = 1.0 if term.shard == "replicated" else (
                    float(p.ep * p.etp) if "expert" in term.shard else float(p.tp))
                if term.category == "routed_expert":
                    expert_bytes += value
                    expert_params += float(term.logical_parameters) / shard
                else:
                    non_bytes += value
                    non_params += float(term.logical_parameters) / shard
        for plan in model.layer_plans:
            width = (plan.head_dim if plan.attention == "gqa" else
                     plan.qk_nope_head_dim + plan.qk_rope_head_dim + plan.v_head_dim)
            heads_width += float(plan.num_heads) * float(width)
        die, mem, ctx = parsed.machine.dies[0], Memory(parsed), max(1, int(context_tokens))
        plans = tuple(model.layer_plans)
        buffers = dict(die.buffers)
        l1_bytes = float(buffers.get("l1", 0.0) or 0.0)
        ub_bytes = float(buffers.get("ub", 0.0) or 0.0)
        per_token_activation = float(model.hidden_size) * float(
            dtype_bytes(model.activation_dtype)) * 4.0
        tile_tokens = (max(1, int(l1_bytes // per_token_activation))
                       if l1_bytes > 0.0 and per_token_activation > 0.0 else 0)
        uncached = bool(l1_bytes <= 0.0 or ub_bytes <= 0.0 or tile_tokens <= 0)
        prefetch = bool(ub_bytes >= per_token_activation) and not uncached
        query_heads = max(1, max(
            (int(plan.num_heads or 0) for plan in plans), default=1) // max(1, p.tp))
        kv = max(ZERO_FLOAT, float(sum(mem.cache_bytes(ctx, padded=False))) / float(ctx))
        return cls(
            parsed, non_bytes, expert_bytes, non_params, expert_params,
            float(dict(die.rates)["hbm"]),
            float(die.peak(parsed.model.compute_dtype or "bf16")),
            float(dict(die.rates)["vector"]), float(die.launch_us) / 1000.0,
            max(1, int(model.layers)),
            (sum(1 for plan in plans if plan.ffn == "moe") if plans
             else (int(model.layers) if model.ffn == "moe" else 0)),
            int(model.top_k or 0), int(model.num_experts or 0), kv,
            float(state_bytes_per_request_per_rank(model, p)),
            int(model.hidden_size), float(dtype_bytes(model.activation_dtype)),
            heads_width / max(1, p.tp), ctx, int(p.tp), int(p.pp), int(p.dp),
            int(p.pcp), int(p.cp), int(p.ep),
            max((int(plan.num_kv_heads or 0) for plan in plans), default=1),
            max((int(plan.head_dim or 0) for plan in plans), default=128),
            model_families(model), int(p.etp),
            int(p.lane_plane // max(1, p.ep * int(p.etp))),
            tuple(str(plan.ffn) == "moe" for plan in plans),
            query_heads,
            l1_bytes, ub_bytes, tile_tokens, uncached, prefetch,
            axis_endpoint_offsets=(
                ("tp", tuple(range(int(p.tp)))),
                ("etp", tuple(range(int(p.etp)))),
                ("ep", tuple(index * int(p.etp) for index in range(int(p.ep)))),
                ("dcp", tuple(range(int(p.dcp)))),
                ("pcp", tuple(lane * max(1, int(p.dp) * int(p.tp))
                              for lane in range(int(p.pcp)))),))

    def _attn_flops(self, batch: int, context: int) -> float:
        return float(batch) * 4.0 * float(context) * self.attn_heads_width

    def _active_fraction(self, batch: float) -> float:
        if self.num_experts <= 0:
            return 0.0
        if batch >= 4096:
            return 1.0
        return 1.0 - ((self.num_experts - self.top_k) / self.num_experts) ** batch

    def step(self, batch_tokens: int, *, context: int | None = None,
             fabric: Any = None, routing_skew: float = ONE_FLOAT,
             eplb_redundant_experts: int = ZERO, enable_eplb: bool = False,
             mc2: bool | None = None,
             scaling_mode: str = "fixed_per_dp_batch",
             global_batch: int | None = None,
             per_replica_batch: int | None = None,
             active_replicas: int | None = None,
             host_overlap: float = HALF,
             eplb_placement: Any = None,
             extra_expert_bytes_per_rank: float = ZERO_FLOAT,
             microbatches: int = ONE_FLOAT,
             overlap_mode: str = "serial",
             eplb_rebalance_ms: float = ZERO_FLOAT,
             endpoints: Sequence[int] | None = None,
             dtype: str | None = None,
             topology: str | None = None,
             pp_p2p_ms: float = ZERO_FLOAT,
             axis_groups: Mapping[str, Any] | None = None,
             ) -> StepBreakdown:
        _int("batch_tokens", batch_tokens)
        for name, value in (("context", context), ("global_batch", global_batch),
                            ("per_replica_batch", per_replica_batch), ("active_replicas", active_replicas)):
            if value is not None:
                _int(name, value)
        _int("eplb_redundant_experts", eplb_redundant_experts, minimum=0)
        strict_number("routing_skew", routing_skew, minimum=1.0)
        overlap = _positive("host_overlap", host_overlap, strict=False)
        if overlap > 1.0:
            raise ModelingError("parallel_explorer_host_overlap_invalid")
        _positive("pp_p2p_ms", pp_p2p_ms, strict=False)
        if endpoints is not None:
            for rank in endpoints:
                _int("endpoint", rank, minimum=0)
        if fabric is not None and not callable(getattr(fabric, "collective", None)):
            raise ModelingError("parallel_explorer_fabric_invalid")
        if eplb_placement is not None:
            typed_sequence("rank_tokens", getattr(eplb_placement, "rank_tokens", None), float)
        if scaling_mode not in SCALING_MODES:
            raise ModelingError(f"parallel_explorer_scaling_mode_invalid:{scaling_mode}")
        per_rep = _int("per_replica_batch",
                       per_replica_batch if per_replica_batch is not None else batch_tokens)
        replicas = _int("active_replicas", active_replicas if active_replicas is not None
                        else self.dp * self.case.runtime.parallel.active_pcp_lanes)
        if scaling_mode == "fixed_global_batch":
            total = _int("global_batch", global_batch if global_batch is not None
                         else per_rep * replicas)
            if total % replicas != 0:
                raise ModelingError(f"parallel_explorer_global_batch_not_divisible:{total}%{replicas}")
            per_rep = total // replicas
        else:
            total = per_rep * replicas
        if context is not None and (
                isinstance(context, bool) or not isinstance(context, int)):
            raise ModelingError(f"parallel_explorer_context_invalid:{context}")
        ctx = max(1, int(context or self.context_tokens))
        p = self.case.runtime.parallel
        micro = _int("microbatches", microbatches)
        if micro > per_rep:
            raise ModelingError("parallel_explorer_microbatches_exceed_batch")
        sizes = tuple(per_rep // micro + (1 if index < per_rep % micro else 0)
                      for index in range(micro))
        if sum(sizes) != per_rep:
            raise ModelingError("parallel_explorer_microbatch_partition")
        tokens_in_ep = float(self.dp) * float(per_rep)
        assignments = tokens_in_ep * max(0, self.top_k)
        owner_domains = max(1, self.ep) * max(1, self.expert_dp)
        assignments_per_owner = assignments / owner_domains
        moe = self.num_experts > 0 and self.ep > 1
        extra_expert = _positive("extra_expert_bytes_per_rank",
                                 extra_expert_bytes_per_rank, strict=False)
        local_experts = max(1, int(self.num_experts) // max(1, self.ep))
        per_expert_params = self.expert_params / local_experts if local_experts else 0.0
        tree = fabric or HierarchicalFabricTree.from_hardware(self.case.machine)
        topology_dtype = dtype
        if axis_groups is None:
            from mindie_sim.parallel_placement import build_placement
            plan = build_placement(self.case, p).to_dict()
        else:
            plan = dict(axis_groups)
        iter_axis_groups(plan)
        endpoint_ids = validate_endpoint_map(endpoints, int(p.world_size),
                                             tree=tree)

        def physical(group: Sequence[int]) -> tuple[int, ...]:
            result = []
            for value in group:
                rank = int(value)
                if rank < 0 or rank >= len(endpoint_ids):
                    raise ModelingError(
                        f"parallel_explorer_group_rank_invalid:{rank}")
                result.append(endpoint_ids[rank])
            return tuple(result)

        def service(axis: str, kind: str, payload: float, groups=None
                    ) -> tuple[float, bool, str]:
            results = []
            active_groups = tuple(plan.get(axis) or ()) if groups is None else groups
            for group in active_groups:
                ids = physical(group)
                if len(ids) < 2:
                    continue
                results.append(tree.collective(
                    kind, payload, len(ids), endpoints=ids,
                    dtype=dtype, topology=topology))
            if not results:
                return 0.0, True, "single_rank"
            worst = max(float(result.total_us) for result in results)
            # Groups of one synchronous axis share the same physical switch:
            # their concurrent demand is aggregated, never priced as if each
            # group owned 100% of the cut alone.
            switch_us = aggregate_switch_bound(
                tree, [tuple(physical(group)) for group in active_groups],
                kind, payload)
            worst = max(worst, switch_us)
            qualified = all(bool(result.route_qualified) for result in results)
            evidence = next((result.evidence for result in results
                             if not result.route_qualified),
                            next((result.evidence for result in results
                                  if result.evidence not in {"trivial", "declared"}),
                                 "declared"))
            return worst, qualified, evidence

        partition, stage_layers, layer_masks = stage_layer_slices(self.case)
        mask = self.moe_mask or (self.moe_layers > 0,) * max(1, self.layers)
        moe_layers = max(1, self.moe_layers)
        kind = "mc2" if (mc2 if mc2 is not None
                         else self.case.runtime.moe_recipe == "mc2") else "all_to_all_v"
        overlap_mode = validate_overlap_mode(overlap_mode)
        stage_size = max(1, int(p.pcp)) * int(p.dp) * int(p.tp)
        hop_union: dict[tuple[int, int], tuple[int, ...]] = {}
        hop_ok = True
        hop_evidence = "declared"
        for stage in range(max(0, self.pp - 1)):
            left = tuple(range(stage * stage_size, (stage + 1) * stage_size))
            right = tuple(range((stage + 1) * stage_size, (stage + 2) * stage_size))
            union = physical(left) + physical(right)
            hop_union[(stage, stage + 1)] = union
            members = getattr(tree, "member_ranks", None)
            if members and not set(union) <= set(members):
                hop_ok = False
                hop_evidence = "disconnected_participants"

        def hop_service(payload: float, stage: int
                        ) -> tuple[float, bool, str, tuple[tuple[str, float], ...]]:
            union = hop_union.get((stage, stage + 1))
            if union is None:
                return 0.0, True, "single_stage", ()
            if hasattr(tree, "peer_transfer"):
                route = tree.peer_transfer(payload, union,
                                           dtype=topology_dtype, topology=topology)
                resources = ()
                if hasattr(tree, "peer_resources"):
                    try:
                        resources = tuple(tree.peer_resources(
                            payload, union, dtype=topology_dtype,
                            topology=topology))
                    except ModelingError:
                        resources = ()
                return (float(route.total_us), bool(route.route_qualified),
                        str(route.evidence), resources)
            # Compatibility for an injected tree without the peer API: price
            # the single declared cut, never unrelated hardware bandwidth.
            bandwidth = None
            if hasattr(tree, "cut_bandwidth_bytes_per_ms"):
                try:
                    bandwidth = tree.cut_bandwidth_bytes_per_ms(union)
                except ModelingError:
                    bandwidth = None
            if bandwidth is None or bandwidth <= 0:
                return 0.0, False, "no_declared_peer_route", ()
            return (payload / float(bandwidth) * 1000.0, True, "declared_cut",
                    ())

        def micro_cost(size: int) -> dict[str, Any]:
            tokens = float(self.dp) * float(size)
            active = self._active_fraction(tokens)
            weight_ms = ((self.expert_bytes * active + extra_expert)
                         / self.hbm_bw * 1000.0 if self.expert_bytes else 0.0)
            expert_flops = 2.0 * (tokens * max(0, self.top_k) / owner_domains) \
                * per_expert_params
            expert_ms = max(expert_flops / self.cube_peak * 1000.0, weight_ms)
            ratio = 1.0
            if moe:
                ep_mean = tokens * max(0, self.top_k) / max(1, self.ep)
                peak = max_rank_tokens(int(round(tokens)), self.top_k, self.ep,
                                       routing_skew, num_experts=self.num_experts)
                if enable_eplb and eplb_redundant_experts > 0:
                    if eplb_placement is not None and per_rep > 0:
                        peak = (max(float(value) for value in eplb_placement.rank_tokens)
                                * float(size) / float(per_rep))
                    else:
                        peak = eplb_effect(
                            int(round(tokens)), self.top_k, self.ep, routing_skew,
                            num_experts=self.num_experts,
                            redundant_experts=eplb_redundant_experts).peak_tokens
                ratio = max(1.0, float(peak) / max(ep_mean, 1e-12))
            straggler_ms = expert_ms * (ratio - 1.0)
            tp_payload = float(size * self.hidden * self.element_bytes)
            ledger = CommLedgerBuilder(tree, "decode")
            axis_layer_ms = [0.0] * self.layers
            dcp_layer_ms = [0.0] * self.layers
            switch_layer_ms = [0.0] * self.layers
            ep_layer_ms = [0.0] * self.layers
            flags = []
            dispatch_total = combine_total = mc2_total = 0.0
            payload = float(tokens * self.top_k * self.hidden * self.element_bytes)
            for stage, indices in enumerate(stage_layers):
                stage_ledger = CommLedgerBuilder(tree, "decode")
                services = {}
                for axis, collective_kind in (("tp", "all_reduce"), ("etp", "all_reduce"),
                                              ("edp", "all_reduce"), ("pcp", "all_gather"),
                                              ("dcp", "all_reduce")):
                    owned = axis_layer_indices(indices, layer_masks, axis)
                    groups = stage_groups(plan, stage, axis)
                    value = tp_payload
                    if axis == "dcp":
                        if self.dcp <= 1:
                            continue
                        value = dcp_merge_payloads(self.tp, self.dcp, size,
                            max(1, self.query_heads), self.head_dim, self.element_bytes)["total_bytes"]
                    if not owned:
                        continue
                    us, ok, evidence = service(axis + "_groups", collective_kind, value, groups)
                    flags.append((ok, evidence))
                    services[axis] = us / 1000.0
                    stage_ledger.collective(axis, collective_kind, value,
                        tuple(physical(g) for g in groups), count=len(owned), dedupe=True)
                stage_moe = axis_layer_indices(indices, layer_masks, "ep")
                if moe and stage_moe:
                    groups = stage_groups(plan, stage, "ep")
                    us, ok, evidence = service("ep_groups", kind, payload, groups)
                    flags.append((ok, evidence))
                    calls = 1 if kind == "mc2" else 2
                    for role in (("ep_mc2_dispatch_combine",) if kind == "mc2" else
                                 ("ep_dispatch", "ep_combine")):
                        stage_ledger.collective(role, kind, payload,
                            tuple(physical(g) for g in groups), count=len(stage_moe))
                    for i in stage_moe:
                        ep_layer_ms[i] = calls * us / 1000.0
                    if kind == "mc2":
                        mc2_total += us / 1000.0 * len(stage_moe)
                    else:
                        dispatch_total += us / 1000.0 * len(stage_moe)
                        combine_total += us / 1000.0 * len(stage_moe)
                emitted = stage_ledger.build()
                ledger.merge(emitted)
                stage_switch_ms = switch_bound_from_demands(tree, emitted.cut_demands()) / 1000.0
                for i in indices:
                    expert, attention, dense = layer_masks[i]
                    axis_layer_ms[i] = max(2 * services.get("tp", 0) if attention or dense else 0,
                                           2 * services.get("etp", 0) if expert else 0,
                                           services.get("edp", 0) if expert else 0,
                                           services.get("pcp", 0) if attention else 0)
                    dcp_layer_ms[i] = services.get("dcp", 0) if attention else 0
                    switch_layer_ms[i] = stage_switch_ms / len(indices)
            axis_ms, dcp_ms = sum(axis_layer_ms), sum(dcp_layer_ms)
            switch_ms = sum(switch_layer_ms)
            dispatch_ms = dispatch_total / moe_layers
            combine_ms = combine_total / moe_layers
            mc2_ms = mc2_total / moe_layers
            expert_layer = expert_ms / moe_layers if self.moe_layers else 0.0
            straggler_layer = straggler_ms / moe_layers if self.moe_layers else 0.0
            moe_comm_layer = mc2_ms if kind == "mc2" else dispatch_ms + combine_ms
            kv_read = (
                max(ZERO_FLOAT, float(sum(Memory(self.case).cache_bytes(ctx, padded=False))))
                if ctx != self.context_tokens and isinstance(self.case, CaseSpec)
                and self.case.model.layer_plans
                else self.kv_per_token * ctx
            )
            attn_ms = max(self._attn_flops(size, ctx) / self.cube_peak,
                          size * (kv_read + self.state_per_request)
                          / self.hbm_bw) * THOUSAND_MS
            stream = two_stream_schedule(
                self.moe_layers, expert_layer + straggler_layer, moe_comm_layer,
                overlap_mode, attn_per_layer_ms=attn_ms / moe_layers)
            exposed = float(stream["exposed_ms"])
            vector_ops = float(size * self.hidden * self.layers * 8.0)
            stage_compute = max(2.0 * size * self.non_expert_params / self.cube_peak,
                                self.non_expert_bytes / self.hbm_bw,
                                vector_ops / self.vector_peak) * 1000.0
            tile_wave = 0.0
            uncached = 0.0
            if self.uncached_fallback or self.tile_tokens <= 0:
                uncached = (float(size) * float(self.hidden) * self.element_bytes
                            * max(1, self.layers) * 8.0 / self.hbm_bw * 1000.0)
            else:
                waves = int(math.ceil(float(size) / float(max(1, self.tile_tokens))))
                tile_wave = max(0, waves - 1) * self.launch_ms * 0.5
            prefetch_penalty = 0.0
            if not self.mlapo_prefetch:
                prefetch_penalty = (float(size) * float(self.hidden) * self.element_bytes
                                    * max(1, self.layers) / self.hbm_bw
                                    * 1000.0 * 0.25)
            per_layer = []
            for index in range(self.layers):
                moe_here = bool(mask[index]) if index < len(mask) else False
                cost = (attn_ms + stage_compute) / max(1, self.layers)
                cost += axis_layer_ms[index] + dcp_layer_ms[index] + switch_layer_ms[index]
                if moe_here:
                    cost += expert_layer + straggler_layer + ep_layer_ms[index]
                per_layer.append(cost)
            stages = list(stage_durations(partition, per_layer))
            stages[0] += tile_wave + uncached + prefetch_penalty
            stages = tuple(stages)
            routed_ok, routed_evidence = reduce_route_evidence(*flags)
            return {"size": size, "tokens": tokens, "active": active,
                    "weight_ms": weight_ms,
                    "expert_ms": expert_ms, "straggler_ms": straggler_ms,
                    "ratio": ratio, "attn_ms": attn_ms, "exposed": exposed,
                    "stage_compute": stage_compute, "tp_ms": axis_ms,
                    "dcp_ms": dcp_ms, "switch_ms": switch_ms,
                    "raw_comm": max(exposed, math.fsum(ep_layer_ms)) + axis_ms + dcp_ms + switch_ms,
                    "tile_wave": tile_wave, "uncached": uncached,
                    "prefetch": prefetch_penalty, "stages": stages,
                    "ok": routed_ok, "evidence": routed_evidence,
                    "dispatch_ms": dispatch_ms, "combine_ms": combine_ms,
                    "mc2_ms": mc2_ms,
                    "ledger": ledger.build(
                        tokens=tokens,
                        requests=float(per_rep * replicas))}

        costs = [micro_cost(size) for size in sizes]
        full = costs[0] if micro == 1 else micro_cost(per_rep)
        pp_stages = full["stages"]
        max_stage_ms, pp_fill = pp_bubble_ms(pp_stages, micro)
        hops, hop_resources = [], []
        hop_ledger = CommLedgerBuilder(tree, "decode")
        for index, size in enumerate(sizes):
            row, resource_row = [], []
            for stage in range(max(0, self.pp - 1)):
                payload = (float(size) * float(self.hidden) * self.element_bytes
                           / float(max(1, self.tp)))
                us, ok_hop, ev_hop, resources = hop_service(payload, stage)
                union = hop_union.get((stage, stage + 1))
                if union is not None:
                    # Every concurrent DP replica issues its own activation hop.
                    hop_ledger.peer("pp_hop", payload, union, count=self.dp)
                if not ok_hop:
                    hop_ok = False
                    hop_evidence = ev_hop
                row.append(us / 1000.0)
                resource_row.append(
                    tuple((key, service / 1000.0) for key, service in resources))
            row.extend([0.0] * (max(0, self.pp) - len(row)))
            resource_row.extend([()] * (max(0, self.pp) - len(resource_row)))
            hops.append(tuple(row))
            hop_resources.append(tuple(resource_row))
        pipeline = pp_microbatch_schedule_ms(
            tuple(cost["stages"] for cost in costs), tuple(hops),
            tuple(hop_resources))
        ledger = merge_ledgers("decode", [cost["ledger"] for cost in costs])
        ledger = ledger.merged(hop_ledger.build(
            tokens=float(per_rep * self.dp), requests=float(per_rep * replicas)))
        step_demands = ledger.cut_demands()
        p2p_total = float(pipeline["p2p_ms"]) + max(0.0, float(pp_p2p_ms))
        attn_sum = sum(cost["attn_ms"] for cost in costs)
        expert_sum = sum(cost["expert_ms"] for cost in costs)
        straggler_sum = sum(cost["straggler_ms"] for cost in costs)
        exposed_sum = sum(cost["exposed"] for cost in costs)
        host_work = self.launch_ms * self.layers * 4.0 * micro
        device_work = attn_sum + expert_sum + straggler_sum
        covered = min(host_work, device_work) * _positive(
            "host_overlap", host_overlap, strict=False)
        if covered > host_work:
            raise ModelingError("parallel_explorer_host_overlap_exceeds_work")
        host_exposed = host_work - covered
        rebalance = _positive("eplb_rebalance_ms", eplb_rebalance_ms, strict=False)
        # One execution latency: the exact compute-and-link resource schedule,
        # plus once-charged host and rebalance terms that do not overlap it.
        makespan = float(pipeline["makespan_ms"])
        non_moe = (makespan - attn_sum - expert_sum - straggler_sum - exposed_sum)
        raw_comm = (sum(cost["raw_comm"] for cost in costs) + p2p_total)
        terms = {"attention": attn_sum, "expert_compute": expert_sum,
                 "expert_straggler": straggler_sum,
                 "fabric_communication": exposed_sum,
                 "non_moe": non_moe, "host": host_exposed,
                 "eplb_rebalance": rebalance}
        external_p2p = max(0.0, float(pp_p2p_ms))
        total_ms = makespan + host_exposed + rebalance + external_p2p
        host_ratio = host_exposed / total_ms if total_ms > 0 else 0.0
        useful = per_rep * replicas
        tps = useful / (total_ms / 1000.0) if total_ms > 0 else 0.0
        routed_ok = hop_ok and all(cost["ok"] for cost in costs)
        routed_evidence = (hop_evidence if not hop_ok else next(
            (cost["evidence"] for cost in costs
             if cost["evidence"] != "declared"), "declared"))
        expert_flops_total = 2.0 * assignments_per_owner * per_expert_params
        return StepBreakdown(
            per_rep, total_ms, attn_sum, expert_sum, straggler_sum, exposed_sum,
            non_moe, full["weight_ms"], full["active"],
            max((cost["ratio"] for cost in costs), default=1.0),
            max(terms, key=terms.get), self.moe_layers, self.layers, raw_comm,
            host_exposed, host_ratio, pp_fill, full["tp_ms"], full["dcp_ms"],
            replicas, total, per_rep, useful, assignments,
            assignments_per_owner, scaling_mode,
            kind == "mc2", tuple((name, value) for name, value in terms.items()),
            tps, tps / max(1, p.world_size), expert_flops_total,
            expert_flops_total, tokens_in_ep, self.tile_tokens,
            sum(cost["tile_wave"] for cost in costs),
            sum(cost["uncached"] for cost in costs),
            bool(self.mlapo_prefetch), micro, overlap_mode, rebalance,
            pp_stage_ms=pp_stages, pp_max_stage_ms=max_stage_ms,
            two_stream_exposed_ms=exposed_sum,
            pp_pipeline_bubble_ms=float(pipeline["bubble_ms"]),
            pp_pipeline_makespan_ms=float(pipeline["makespan_ms"]),
            pp_serial_stage_ms=float(pipeline["serial_ms"]),
            pp_p2p_ms=p2p_total,
            route_qualified=routed_ok, route_evidence=routed_evidence,
            comm_ledger=ledger,
            shared_cut_demands=tuple(sorted(step_demands.items())))

    def step_ms(self, batch_tokens: int, **kwargs: Any) -> float:
        return self.step(batch_tokens, **kwargs).step_ms

@typed_inputs
def linear_state_step_overhead(config: Any, layers: Sequence[Any], *,
                               batch: int, context: int,
                               hbm_bytes_per_s: float,
                               compute_flops_per_s: float | None = None,
                               kernel_launch_ms: float = 0.0,
                               accepted_tokens: int | None = None,
                               accept_rate: float = 0.0,
                               layerwise: bool = False,
                               overlap_fraction: float = 0.0) -> dict[str, Any]:
    """Per-step recurrent-state verify/commit plus align pre-copy overhead.

    This is a non-invasive companion to :meth:`DecodeCostModel.step`: the
    declared KDA/GDN layers keep their attention/decode pricing, and this adds
    the state-rollback kernel and cross-block copy that the generic step does
    not model.  ``config`` is a ``linear_state_physics.RecoverSsmConfig``.
    """
    from mindie_sim.linear_state_replay import (precopy_cost,
                                                verify_commit_cost)
    overlap = strict_number("overlap_fraction", overlap_fraction)
    if overlap > 1.0:
        raise ModelingError("linear_state_overlap_fraction_invalid")
    verify = verify_commit_cost(
        config, layers, batch=batch, sequence_tokens=context,
        hbm_bytes_per_s=hbm_bytes_per_s,
        compute_flops_per_s=compute_flops_per_s,
        kernel_launch_ms=kernel_launch_ms,
        accepted_tokens=accepted_tokens, accept_rate=accept_rate)
    if config.mamba_cache_mode == "align":
        precopy = precopy_cost(
            config, layers, sequence_start=context,
            sequence_end=context + config.spec_query_len, batch=batch,
            hbm_bytes_per_s=hbm_bytes_per_s,
            kernel_launch_ms=kernel_launch_ms, layerwise=layerwise,
            overlap_fraction=overlap_fraction)
        precopy_ms = (precopy["layerwise_overlapped_ms"] if layerwise
                      else precopy["bulk_ms"])
    else:
        precopy = None
        precopy_ms = 0.0
    return {"verify": verify, "precopy": precopy,
            "precopy_ms": precopy_ms,
            "step_overhead_ms": verify["total_ms"] + precopy_ms}

