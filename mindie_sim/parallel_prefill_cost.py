"""Family-complete prefill service with a physical PP resource schedule.

[Role]
  Price one prefill pass per layer from that layer's actual attention family,
  compression ratio, window, sparse indexer and PCP query ownership; include
  ``pcp_index_k_gather`` when sparse indexing is active; and return the latency
  of one exact compute-and-link resource schedule rather than a serial sum.

[Invariants]
  * ``mla_csa`` with ``compress_ratio=16`` uses ratio 16, never a hardcoded 4.
  * Heterogeneous layer stacks compose their own pair counts, FLOPs, KV reads
    and indexer dependencies; the model is never collapsed to one family or one
    maximum dimension.
  * Every ``pcp_index_k_gather`` is issued, priced and route-qualified per
    sparse layer at that layer's actual payload; sequential collectives on one
    PCP communicator are summed and all their bytes are recorded on the cut.
  * Multi-chunk prompts execute and sum a real ledger per chunk: a fixed
    full-context gather runs once per executed chunk, never a fractional ratio.
  * Prefill MoE emits dispatch and combine separately at the actual MoE-layer count.
  * The emitted DCP all-reduce is priced and qualified at its actual payload.
  * PP hops are priced on their actual adjacent endpoints: startup, switch
    capacity, oversubscription and route qualification are all retained.
    ``step_ms`` is the schedule makespan plus once-charged host work.
  * The step emits its executed communication ledger: base axis rows, every
    per-layer PCP KV/index gather and every PP hop, with resolved cut bytes.

[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Fresh dict result; caller Case untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from mindie_sim.catalog.attention_common import active_dcp_degree
from mindie_sim.fabric_topology import HierarchicalFabricTree
from mindie_sim.parallel_comm_ledger import (CommLedgerBuilder, empty_ledger,
                                             merge_ledgers, reduce_route_evidence)
from mindie_sim.parallel_context import (family_of_layer, pcp_attention_pairs,
                                         pcp_index_k_gather_bytes,
                                         pcp_kv_gather_bytes)
from mindie_sim.parallel_stage_ops import (stage_groups, stage_layer_slices, axis_layer_indices)
from mindie_sim.parallel_cuts import (merge_demands,
                                      sequential_cut_demands,
                                      switch_bound_from_demands,
                                      unique_cut_demands)
from mindie_sim.parallel_decode_cost import (DecodeCostModel, _int, _positive,
                                             validate_endpoint_map)
from mindie_sim.parallel_placement import build_placement
from mindie_sim.parallel_schedule import (layer_partition,
                                          pp_microbatch_schedule_ms,
                                          stage_durations)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError

ONE = 1
HALF = 0.5


def _layer_view(plan: Any, fallback: Mapping[str, Any]) -> dict[str, Any]:
    attention = str(getattr(plan, "attention", "gqa") or "gqa").lower()
    width = (int(getattr(plan, "num_heads", 0) or 0)
             * int(getattr(plan, "head_dim", 0) or 0)
             if attention == "gqa" else
             int(getattr(plan, "num_heads", 0) or 0)
             * (int(getattr(plan, "qk_nope_head_dim", 0) or 0)
                + int(getattr(plan, "qk_rope_head_dim", 0) or 0)
                + int(getattr(plan, "v_head_dim", 0) or 0)))
    return {
        "family": family_of_layer(plan),
        "index_topk": max(0, int(getattr(plan, "index_topk", 0) or 0)),
        "compress_ratio": max(0, int(getattr(plan, "compress_ratio", 0) or 0)),
        "window": max(0, int(getattr(plan, "window", 0) or 0)),
        "index_heads": max(0, int(getattr(plan, "index_heads", 0) or 0)),
        "index_dim": max(0, int(getattr(plan, "index_dim", 0) or 0)),
        "head_dim": max(1, int(getattr(plan, "head_dim", 0) or 1)),
        "num_heads": max(0, int(getattr(plan, "num_heads", 0) or 0)),
        "width": max(0, int(width)),
        "moe": str(getattr(plan, "ffn", "dense")).lower() == "moe",
        "fallback": fallback,
    }


def _model_layer_view(model: Any) -> dict[str, Any]:
    attention = str(getattr(model, "attention", "gqa") or "gqa").lower()
    width = (int(getattr(model, "num_heads", 0) or 0)
             * int(getattr(model, "head_dim", 0) or 0) if attention == "gqa"
             else int(getattr(model, "num_heads", 0) or 0)
             * (int(getattr(model, "qk_nope_head_dim", 0) or 0)
                + int(getattr(model, "qk_rope_head_dim", 0) or 0)
                + int(getattr(model, "v_head_dim", 0) or 0)))
    return {"family": attention,
            "index_topk": max(0, int(getattr(model, "index_topk", 0) or 0)),
            "compress_ratio": max(0, int(getattr(model, "compress_ratio", 0) or 0)),
            "window": max(0, int(getattr(model, "sliding_window", 0) or 0)),
            "index_heads": max(0, int(getattr(model, "index_heads", 0) or 0)),
            "index_dim": max(0, int(getattr(model, "index_dim", 0) or 0)),
            "head_dim": max(1, int(getattr(model, "head_dim", 0) or 1)),
            "num_heads": max(0, int(getattr(model, "num_heads", 0) or 0)),
            "width": max(0, int(width)),
            "moe": str(getattr(model, "ffn", "dense")).lower() == "moe",
            "fallback": {}}


def prefill_step_ms(case: CaseSpec | Mapping[str, Any], *, isl: int,
                    prefix_hit_rate: float, batch: int, fabric: Any = None,
                    host_overlap: float = HALF, microbatches: int = ONE,
                    mc2: bool | None = None,
                    endpoints: Sequence[int] | None = None,
                    dtype: str | None = None,
                    topology: str | None = None) -> dict[str, Any]:
    """Per-layer prefill service driven by one physical PP schedule."""
    parsed = parse_case(case)
    model, p = parsed.model, parsed.runtime.parallel
    context = _int("prefill_isl", isl)
    # Strict topology/endpoint validation precedes every zero-work shortcut: a
    # full prefix hit still names concrete physical communicator members.
    tree = fabric or HierarchicalFabricTree.from_hardware(parsed.machine)
    endpoint_ids = validate_endpoint_map(endpoints, int(p.world_size), tree=tree)
    plan = build_placement(parsed, p).to_dict()
    q_total = max(0, int(math.ceil(context * (1.0 - float(prefix_hit_rate)))))
    batch = _int("prefill_batch", batch)
    micro = _int("prefill_microbatches", microbatches)
    if micro > batch:
        raise ModelingError("parallel_explorer_microbatches_exceed_batch")
    lanes = max(1, int(p.pcp))
    if q_total == 0:
        residual_ms = 0.05 + 0.01 * p.pp
        return {"q_tokens": 0, "chunk_tokens": 0, "steps": 1,
                "step_ms": residual_ms,
                "mu_requests_per_s": batch / (residual_ms / 1000.0),
                "compute_ms": 0.0, "collective_ms": 0.0,
                "transition_ms": residual_ms, "moe_comm_ms": 0.0,
                "pp_fill_ms": 0.0, "pp_p2p_ms": 0.0,
                "route_qualified": True, "route_evidence": "single_rank",
                "comm_ledger": empty_ledger("prefill"),
                "shared_cut_demands": {}}
    model_cost = DecodeCostModel.from_case(parsed, context_tokens=context)
    capacity = max(1, int(parsed.runtime.max_num_batched_tokens))
    # The actual executed chunks: the last one is the exact remainder, never a
    # fractional scaling of a full chunk's fixed operations.
    chunk_sizes = tuple(min(capacity, q_total - offset)
                        for offset in range(0, q_total, capacity))
    chunk = chunk_sizes[0]
    steps = len(chunk_sizes)
    sizes = tuple(batch // micro + (1 if index < batch % micro else 0)
                  for index in range(micro))
    if sum(sizes) != batch:
        raise ModelingError("parallel_explorer_microbatch_partition")
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    if plans:
        layers = tuple(_layer_view(plan, {}) for plan in plans)
    else:
        layers = tuple(_model_layer_view(model)
                       for _ in range(max(1, int(getattr(model, "layers", 1)))))
    n_layers = max(1, len(layers))

    def physical(group: Sequence[int]) -> tuple[int, ...]:
        result = []
        for value in group:
            rank = int(value)
            if rank < 0 or rank >= len(endpoint_ids):
                raise ModelingError(f"parallel_explorer_group_rank_invalid:{rank}")
            result.append(endpoint_ids[rank])
        return tuple(result)

    def service(axis: str, kind: str, payload: float, groups=None
                ) -> tuple[float, bool, str]:
        results = []
        for group in (plan.get(axis) or ()) if groups is None else groups:
            ids = physical(group)
            if len(ids) < 2:
                continue
            results.append(tree.collective(kind, payload, len(ids),
                                           endpoints=ids, dtype=dtype,
                                           topology=topology))
        if not results:
            return 0.0, True, "single_rank"
        worst = max(float(result.total_us) for result in results)
        qualified, evidence = reduce_route_evidence(*(
            (result.route_qualified, result.evidence) for result in results))
        return worst, qualified, evidence

    stage_size = lanes * int(p.dp) * int(p.tp)
    hop_union: dict[tuple[int, int], tuple[int, ...]] = {}
    hop_ok, hop_evidence = True, "declared"
    for stage in range(max(0, int(p.pp) - 1)):
        left = tuple(range(stage * stage_size, (stage + 1) * stage_size))
        right = tuple(range((stage + 1) * stage_size, (stage + 2) * stage_size))
        union = physical(left) + physical(right)
        hop_union[(stage, stage + 1)] = union
        members = getattr(tree, "member_ranks", None)
        if members and not set(union) <= set(members):
            hop_ok, hop_evidence = False, "disconnected_participants"

    def hop_ms(payload: float, stage: int
               ) -> tuple[float, bool, str, tuple[tuple[str, float], ...]]:
        union = hop_union.get((stage, stage + 1))
        if union is None:
            return 0.0, True, "single_stage", ()
        if hasattr(tree, "peer_transfer"):
            route = tree.peer_transfer(payload, union, dtype=dtype,
                                       topology=topology)
            resources = ()
            if hasattr(tree, "peer_resources"):
                try:
                    resources = tuple(tree.peer_resources(
                        payload, union, dtype=dtype, topology=topology))
                except ModelingError:
                    resources = ()
            return (float(route.total_us) / 1000.0, bool(route.route_qualified),
                    str(route.evidence), resources)
        return 0.0, False, "no_declared_peer_route", ()

    owner_domains = max(1, int(p.ep)) * max(1, model_cost.expert_dp)
    local_experts = max(1, int(model_cost.num_experts) // max(1, int(p.ep)))
    per_expert_params = (model_cost.expert_params / local_experts
                         if local_experts else 0.0)
    partition, stage_layers, layer_masks = stage_layer_slices(parsed)
    layer_stages = tuple(stage for stage, indices in enumerate(stage_layers) for _ in indices)
    moe_layers = sum(1 for layer in layers if layer["moe"])
    layers_with_moe = max(1, moe_layers)
    pair_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def pairs_for(index: int, layer: Mapping[str, Any],
                  chunk_tokens: int) -> dict[str, Any]:
        key = (index, int(chunk_tokens))
        cached = pair_cache.get(key)
        if cached is None:
            cached = pcp_attention_pairs(
                isl=context, q_tokens=int(chunk_tokens), pcp=lanes,
                family=layer["family"], index_topk=layer["index_topk"],
                sliding_window=layer["window"],
                compress_ratio=layer["compress_ratio"],
                load_balance=str(dict(parsed.runtime.metadata).get(
                    "pcp_load_balance", "DualChunkSwap")))
            pair_cache[key] = cached
        return cached

    def micro_cost(size: int, chunk_tokens: int) -> dict[str, Any]:
        q_per_lane = float(chunk_tokens) / float(lanes)
        tokens_global = float(int(p.dp) * size * chunk_tokens)
        tokens_per_lane = tokens_global / lanes
        expert_flops_total = (2.0 * (tokens_per_lane
                                     * max(0, model_cost.top_k)
                                     / owner_domains) * per_expert_params)
        non_expert_flops_total = (2.0 * float(size * chunk_tokens) / lanes
                                  * model_cost.non_expert_params)
        compute_ms = 0.0
        critical = 0
        layer_flops: list[float] = []
        layer_ms: list[float] = []
        for index, layer in enumerate(layers):
            pairs_profile = pairs_for(index, layer, chunk_tokens)
            critical_pairs = int(pairs_profile["critical_pairs"])
            pairs = float(size * max(1, critical_pairs))
            critical = max(critical, critical_pairs)
            width = layer["width"] / max(1, int(p.tp))
            indexer = 0.0
            if layer["index_heads"] > 0 and layer["index_dim"] > 0:
                indexer = (2.0 * float(size) * q_per_lane * float(context)
                           * float(layer["index_heads"]) * float(layer["index_dim"]))
            flops = 4.0 * pairs * width + indexer
            per_layer_share = 1.0 / float(n_layers)
            flops += (non_expert_flops_total * per_layer_share)
            bytes_term = 0.0
            if layer["family"] in {"dsa", "swa", "csa", "csa2", "hca"} \
                    or lanes > 1 or layer["window"] > 0:
                bytes_term = (pairs * float(layer["head_dim"])
                              * model_cost.element_bytes * 2.0)
            bytes_term += (model_cost.non_expert_bytes * per_layer_share)
            if layer["moe"]:
                flops += expert_flops_total / float(layers_with_moe)
                bytes_term += (model_cost.expert_bytes
                               / float(layers_with_moe))
            layer_flops.append(flops)
            value_ms = max(flops / model_cost.cube_peak,
                           bytes_term / model_cost.hbm_bw) * 1000.0
            layer_ms.append(value_ms)
            compute_ms += value_ms
        expert_flops = expert_flops_total
        tp_payload = float(size * chunk_tokens * model_cost.hidden
                           * model_cost.element_bytes)
        ledger = CommLedgerBuilder(tree, "prefill")
        axis_layer_ms = [0.0] * n_layers
        pcp_layer_ms = [0.0] * n_layers
        dcp_layer_ms = [0.0] * n_layers
        switch_layer_ms = [0.0] * n_layers
        ep_layer_ms = [0.0] * n_layers
        flags = []
        index_ms = 0.0
        moe_payload = tokens_global * max(1, model_cost.top_k) * model_cost.hidden * model_cost.element_bytes
        use_mc2 = mc2 if mc2 is not None else parsed.runtime.moe_recipe == "mc2"
        for stage, indices in enumerate(stage_layers):
            stage_ledger = CommLedgerBuilder(tree, "prefill")
            services = {}
            for axis, kind in (("tp", "all_reduce"), ("etp", "all_reduce"),
                               ("edp", "all_reduce"), ("pcp", "all_gather"),
                               ("dcp", "all_reduce")):
                owned = axis_layer_indices(indices, layer_masks, axis)
                if not owned:
                    continue
                groups = stage_groups(plan, stage, axis)
                value = tp_payload
                if axis == "dcp":
                    if active_dcp_degree(p, "prefill") <= ONE:
                        continue
                    from mindie_sim.parallel_context import dcp_merge_payloads
                    value = dcp_merge_payloads(int(p.tp), int(p.dcp), size,
                        max(1, model_cost.query_heads), model_cost.head_dim,
                        model_cost.element_bytes)["total_bytes"]
                us, ok, evidence = service(axis + "_groups", kind, value, groups)
                flags.append((ok, evidence))
                services[axis] = us / 1000.0
                stage_ledger.collective(axis, kind, value, tuple(physical(g) for g in groups),
                                        count=len(owned), dedupe=True)
            groups = stage_groups(plan, stage, "pcp")
            for i in indices:
                expert, attention, dense = layer_masks[i]
                axis_layer_ms[i] = max(2 * services.get("tp", 0) if attention or dense else 0,
                                       2 * services.get("etp", 0) if expert else 0,
                                       services.get("edp", 0) if expert else 0)
                dcp_layer_ms[i] = services.get("dcp", 0) if attention else 0
                if p.pcp <= 1 or not attention:
                    continue
                layer = layers[i]
                gather = pcp_kv_gather_bytes(int(math.ceil(chunk_tokens / lanes)), lanes,
                    max(1, model_cost.kv_heads), layer["head_dim"], model_cost.element_bytes)
                payloads = [("pcp_sequential", gather["total_bytes"] * float(size))]
                if layer["index_topk"] > 0 or layer["index_dim"] > 0:
                    gathered = pcp_index_k_gather_bytes(context, lanes,
                        max(1, layer["index_dim"]), float(model_cost.element_bytes),
                        top_k=layer["index_topk"])
                    payloads.append(("pcp_index", gathered["received_bytes"] * float(size)))
                for role, value in payloads:
                    us, ok, evidence = service("pcp_groups", "all_gather", value, groups)
                    flags.append((ok, evidence))
                    pcp_layer_ms[i] += us / 1000.0
                    if role == "pcp_index":
                        index_ms += us / 1000.0
                    stage_ledger.collective("pcp_sequential", "all_gather", value,
                                             tuple(physical(g) for g in groups))
            owned = axis_layer_indices(indices, layer_masks, "ep")
            if model_cost.num_experts > 0 and int(p.ep) > 1 and owned:
                kind = "mc2" if use_mc2 else "all_to_all_v"
                roles = ("ep_mc2_dispatch_combine",) if use_mc2 else ("ep_dispatch", "ep_combine")
                groups = stage_groups(plan, stage, "ep")
                us, ok, evidence = service("ep_groups", kind, moe_payload, groups)
                flags.append((ok, evidence))
                for i in owned:
                    ep_layer_ms[i] = len(roles) * us / 1000.0
                for role in roles:
                    stage_ledger.collective(role, kind, moe_payload,
                        tuple(physical(g) for g in groups), count=len(owned))
            emitted = stage_ledger.build()
            ledger.merge(emitted)
            stage_switch_ms = switch_bound_from_demands(tree, emitted.cut_demands()) / 1000.0
            for i in indices:
                switch_layer_ms[i] = stage_switch_ms / len(indices)
        axis_ms, pcp_ms, dcp_ms = sum(axis_layer_ms), sum(pcp_layer_ms), sum(dcp_layer_ms)
        switch_ms, moe_comm_ms = sum(switch_layer_ms), sum(ep_layer_ms)
        switch_demands = ledger.build().cut_demands()
        tile_wave = 0.0
        uncached = 0.0
        if model_cost.uncached_fallback or model_cost.tile_tokens <= 0:
            uncached = (tokens_global * model_cost.hidden
                        * model_cost.element_bytes
                        * max(1, model_cost.layers) * 8.0
                        / model_cost.hbm_bw * 1000.0)
        else:
            waves = int(math.ceil(float(chunk_tokens)
                                  / float(max(1, model_cost.tile_tokens))))
            tile_wave = max(0, waves - int(size)) * model_cost.launch_ms * 0.5
        prefetch_penalty = 0.0
        if not model_cost.mlapo_prefetch:
            prefetch_penalty = (tokens_global * model_cost.hidden
                                * model_cost.element_bytes
                                * max(1, model_cost.layers)
                                / model_cost.hbm_bw * 1000.0 * 0.25)
        per_layer = [layer_ms[i] + max(axis_layer_ms[i], pcp_layer_ms[i], dcp_layer_ms[i])
                     + switch_layer_ms[i] + ep_layer_ms[i] for i in range(n_layers)]
        stages = list(stage_durations(partition, per_layer))
        stages[0] += tile_wave + uncached + prefetch_penalty
        routed_ok, routed_evidence = reduce_route_evidence(*flags)
        return {"compute_ms": compute_ms, "axis_ms": axis_ms,
                "pcp_ms": pcp_ms, "index_ms": index_ms, "dcp_ms": dcp_ms,
                "switch_demands": switch_demands,
                "moe_comm_ms": moe_comm_ms, "tile_wave": tile_wave,
                "uncached": uncached, "prefetch": prefetch_penalty,
                "stages": tuple(stages), "expert_flops": expert_flops,
                "ok": routed_ok, "evidence": routed_evidence,
                "tokens_global": tokens_global, "critical": critical,
                "ledger": ledger.build(tokens=tokens_global)}

    overlap = _positive("host_overlap", host_overlap, strict=False)
    ledgers = []
    compute_total = p2p_total = collectives = host_exposed = 0.0
    moe_total = tile_total = uncached_total = 0.0
    expert_total = 0.0
    critical_pairs = 0
    tokens_total = 0.0
    step_total = 0.0
    makespan_total = bubble_total = 0.0
    stage_columns: list[tuple[float, ...]] = []
    routed_ok = hop_ok
    routed_evidence = hop_evidence if not hop_ok else "declared"
    for chunk_tokens in chunk_sizes:
        costs = [micro_cost(size, chunk_tokens) for size in sizes]
        q_per_lane = float(chunk_tokens) / float(lanes)
        hop_ledger = CommLedgerBuilder(tree, "prefill")
        for size in sizes:
            for stage in range(max(0, int(p.pp) - 1)):
                union = hop_union.get((stage, stage + 1))
                if union is None:
                    continue
                payload = (float(size) * q_per_lane * float(model_cost.hidden)
                           * model_cost.element_bytes
                           / float(max(1, int(p.tp))))
                # Every concurrent DP replica issues its own activation hop.
                hop_ledger.peer("pp_hop", payload, union, count=int(p.dp))
        full = costs[0] if micro == 1 else micro_cost(batch, chunk_tokens)
        pp_stages = full["stages"]
        stage_columns.append(tuple(pp_stages))
        chunk_hop_ok, chunk_hop_evidence = True, "declared"
        hops, hop_resources = [], []
        for size in sizes:
            row, resource_row = [], []
            for stage in range(max(0, int(p.pp) - 1)):
                payload = (float(size) * q_per_lane
                           * float(model_cost.hidden)
                           * model_cost.element_bytes
                           / float(max(1, int(p.tp))))
                hop, ok_hop, ev_hop, resources = hop_ms(payload, stage)
                chunk_hop_ok, chunk_hop_evidence = reduce_route_evidence(
                    (chunk_hop_ok, chunk_hop_evidence), (ok_hop, ev_hop))
                row.append(hop)
                resource_row.append(
                    tuple((key, service / 1000.0)
                          for key, service in resources))
            row.extend([0.0] * (max(0, int(p.pp)) - len(row)))
            resource_row.extend([()] * (max(0, int(p.pp)) - len(resource_row)))
            hops.append(tuple(row))
            hop_resources.append(tuple(resource_row))
        pipeline = pp_microbatch_schedule_ms(
            tuple(cost["stages"] for cost in costs), tuple(hops),
            tuple(hop_resources))
        p2p_chunk = float(pipeline["p2p_ms"])
        compute_chunk = sum(cost["compute_ms"] for cost in costs)
        host_work = model_cost.launch_ms * model_cost.layers * 4.0 * micro
        covered = min(host_work, compute_chunk) * overlap
        step_chunk = (float(pipeline["makespan_ms"])
                      + max(0.0, host_work - covered))
        chunk_ledger = merge_ledgers(
            "prefill", [cost["ledger"] for cost in costs])
        chunk_ledger = chunk_ledger.merged(hop_ledger.build(
            tokens=float(sum(cost["tokens_global"] for cost in costs))))
        ledgers.append(chunk_ledger)
        p2p_total += p2p_chunk
        compute_total += compute_chunk
        collectives += sum(
            cost["axis_ms"] + cost["pcp_ms"] + cost["dcp_ms"]
            + cost["moe_comm_ms"] for cost in costs) + p2p_chunk
        host_exposed += max(0.0, host_work - covered)
        moe_total += sum(cost["moe_comm_ms"] for cost in costs)
        tile_total += sum(cost["tile_wave"] for cost in costs)
        uncached_total += sum(cost["uncached"] for cost in costs)
        expert_total += sum(cost["expert_flops"] for cost in costs)
        critical_pairs = max([critical_pairs]
                             + [cost["critical"] for cost in costs])
        tokens_total += sum(cost["tokens_global"] for cost in costs)
        step_total += step_chunk
        makespan_total += float(pipeline["makespan_ms"])
        bubble_total += float(pipeline["bubble_ms"])
        routed_ok, routed_evidence = reduce_route_evidence(
            (routed_ok, routed_evidence), (chunk_hop_ok, chunk_hop_evidence),
            *((cost["ok"], cost["evidence"]) for cost in costs))
    pp_stages = tuple(max(column) for column in zip(*stage_columns))
    max_stage = max(pp_stages, default=0.0)
    families = sorted({layer["family"] for layer in layers})
    ledger = merge_ledgers("prefill", ledgers)
    total_switch_demands = ledger.cut_demands()
    return {"q_tokens": q_total, "chunk_tokens": chunk, "steps": steps,
            "step_ms": step_total,
            "mu_requests_per_s": batch / (step_total / 1000.0),
            "compute_ms": compute_total, "collective_ms": collectives,
            "transition_ms": 0.0, "host_ms": host_exposed,
            "moe_comm_ms": moe_total,
            "pp_fill_ms": 0.0, "pp_p2p_ms": p2p_total,
            "pp_pipeline_bubble_ms": bubble_total,
            "pp_pipeline_makespan_ms": makespan_total,
            "microbatches": micro,
            "tile_wave_ms": tile_total,
            "uncached_fallback_ms": uncached_total,
            "pp_stage_ms": pp_stages, "pp_max_stage_ms": max_stage,
            "expert_flops": expert_total,
            "tokens_global": tokens_total,
            "chunk_steps": steps,
            "critical_pairs": critical_pairs,
            "attention_family": "+".join(families),
            "shared_cut_demands": total_switch_demands,
            "comm_ledger": ledger,
            "route_qualified": routed_ok,
            "route_evidence": routed_evidence}

__all__ = ["prefill_step_ms"]
