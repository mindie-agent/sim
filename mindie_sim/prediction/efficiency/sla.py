"""SLA sizing under hardware counterfactuals.

[Role] Evaluate graph latency and capacity to size concurrency and throughput
  across hardware rate and capacity scenarios.
[Invariants] Every feasible candidate passes latency and lifecycle capacity
  checks on the same evaluated geometry.
[Timeouts & Locks] Finite concurrency probes honor the active evaluation
  deadline. The solver acquires no additional locks.
[Side-Effect Policy] Cases are read-only and the solver owns its probe caches.
"""
from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence
from functools import partial
from dataclasses import replace
from typing import Any

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import GlobalRank

from ..capacity import Memory
from ..evaluation import check_active_deadline
from ..operator_analysis import target_graphs
from ..structure import parse_case
from .analysis import analyze_network_efficiency, scale_case_hardware
from .attribution import _lower_and_run_graph, _pkg_attr
from .capacity_window import (
    _prefill_step_token_budget, _resolve_speculative_tokens,
    _vram_max_sequences_per_dp, _workspace_tokens_for_window,
    apply_sla_baseline_gains, capacity_window, select_jointly_feasible_candidate,
)
from .peaks import DEFAULT_OUTPUT_BUDGET, _BOUNDS, _SCALE_KEYS, _check_pos_float, _validate_geometry


DEFAULT_SLA_MS = 30.0
DEFAULT_SLA_CONTEXT = 126000
DEFAULT_MAX_SEQUENCES = 256
#: Counterfactual-set selectors for :func:`solve_sla_sizing`.
SLA_SENSITIVITY_BASELINE = "baseline"
SLA_SENSITIVITY_FULL = "full"
#: A per-sequence marginal step below this is flat (or negative): no secant.
SECANT_MIN_SLOPE_MS = 1e-9
_MIN_ROW_LEADING_DIMS = 2
_US_PER_MS = 1000.0
_DUPLEX_HALF_DIVISOR = 2.0
_MIN_COLL_GROUP_SIZE = 2
_MIN_COLL_PAIR_GROUP = (0, 1)
_BAND_FLOOR_COLLECTIVES = ("allgather", "reduce_scatter", "allreduce")
_BAND_RING_ALGOS = frozenset({"ring", "pairwise"})

DEFAULT_SLA_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"id": "baseline", "label": "1x HBM, 1x Compute", "hbm_bw_scale": 1.0, "cube_flops_scale": 1.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
    {"id": "2x_hbm", "label": "2x HBM, 1x Compute", "hbm_bw_scale": 2.0, "cube_flops_scale": 1.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
    {"id": "2x_compute", "label": "1x HBM, 2x Cube + 2x Vector", "hbm_bw_scale": 1.0, "cube_flops_scale": 2.0, "vector_ops_scale": 2.0, "link_bw_scale": 1.0},
    {"id": "2x_cube_only", "label": "1x HBM, 2x Cube, 1x Vector", "hbm_bw_scale": 1.0, "cube_flops_scale": 2.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
    {"id": "2x_hbm_2x_compute", "label": "2x HBM, 2x Cube + 2x Vector", "hbm_bw_scale": 2.0, "cube_flops_scale": 2.0, "vector_ops_scale": 2.0, "link_bw_scale": 1.0},
)


def solve_sla_sizing(
    case: CaseSpec | Mapping[str, Any], *, sla_ms: float = DEFAULT_SLA_MS,
    context: int = DEFAULT_SLA_CONTEXT, tokens_per_seq: int | None = None,
    emitted_tokens_per_seq: float | None = None,
    max_sequences_per_dp: int = DEFAULT_MAX_SEQUENCES, phase: str = "decode",
    scope: str = "model", output_budget: int = DEFAULT_OUTPUT_BUDGET,
    scenarios: Sequence[Mapping[str, Any]] | None = None, b1_analysis: Mapping[str, Any] | None = None,
    evaluation_context: Any | None = None,
    deadline_monotonic: float | None = None,
    sensitivity: str = SLA_SENSITIVITY_FULL,
    jobs: int | str = 1,
    fast_span_probes: bool = True,
    base_b_hint: int = 1,
) -> dict[str, Any]:
    """Solve SLA-constrained maximum concurrency B*_SLA and throughput across hardware scenarios."""
    from ..efficiency.unmodeled_gate import assert_timing_supported
    check_active_deadline()
    _check_pos_float("sla_ms", sla_ms)
    if isinstance(max_sequences_per_dp, bool) or not isinstance(max_sequences_per_dp, int) or max_sequences_per_dp < 1:
        raise ValueError(f"max_sequences_per_dp_must_be_positive_int:{max_sequences_per_dp}")
    if isinstance(output_budget, bool) or not isinstance(output_budget, int) or output_budget < 0:
        raise ValueError(f"output_budget_must_be_nonnegative_int:{output_budget}")

    # ``sensitivity`` only selects the counterfactual *set*; an explicit
    # ``scenarios`` list stays authoritative.  One scenario per call keeps the
    # caller's flat worker pool the only pool (no nested pools).
    if scenarios is not None:
        scenario_source = list(scenarios)
    elif str(sensitivity).strip().lower() == SLA_SENSITIVITY_BASELINE:
        scenario_source = [DEFAULT_SLA_SCENARIOS[0]]
    else:
        scenario_source = list(DEFAULT_SLA_SCENARIOS)
    scenario_list = [dict(scenario) for scenario in scenario_source]
    for scenario in scenario_list:
        if "target_mbu" in scenario:
            raise ValueError("target_efficiency_scaling_retired:use_optimize_ladder")
        for key in _SCALE_KEYS:
            _check_pos_float(f"scenario_scale_{key}", scenario.get(key, 1.0))

    assert_timing_supported(case, entry="solve_sla_sizing")
    parsed = parse_case(case)
    q_tokens, emit_tokens = _resolve_speculative_tokens(
        parsed, output_budget=output_budget, tokens_per_seq=tokens_per_seq,
        emitted_tokens_per_seq=(None if phase == "prefill" else emitted_tokens_per_seq),
        phase=phase)
    # A prefill step processes its query; the following decode step commits its
    # own accepted tokens.  Keep both quantities: the lifecycle capacity must
    # use the decode commit, never the prompt-processing volume.
    decode_q_tokens, decode_emit_tokens = _resolve_speculative_tokens(
        parsed, output_budget=output_budget, tokens_per_seq=None,
        emitted_tokens_per_seq=(emitted_tokens_per_seq if phase == "prefill" else None),
        phase="decode")
    _validate_geometry(phase, q_tokens, context, 1, scope)
    batch_budget = (
        _prefill_step_token_budget(parsed)
        if phase == "prefill"
        else max(1, int(parsed.runtime.max_num_batched_tokens))
    )
    chunkable = phase == "prefill" and bool(parsed.runtime.enable_chunked_prefill)
    unchunkable = (not chunkable and q_tokens > batch_budget)
    rt_max_b = (
        int(parsed.runtime.max_num_seqs)
        if chunkable
        else min(int(parsed.runtime.max_num_seqs), batch_budget // q_tokens)
    )
    b_max = min(int(max_sequences_per_dp), rt_max_b)
    dp = int(parsed.runtime.parallel.dp)
    fleet_dp = dp * max(1, int(getattr(parsed.runtime.parallel, "external_dp", 1) or 1))


    run = _SLASizingRun(
        parsed, sla_ms=sla_ms, context=context, phase=phase, scope=scope,
        output_budget=output_budget, q_tokens=q_tokens, emit_tokens=emit_tokens,
        decode_q_tokens=decode_q_tokens, decode_emit_tokens=decode_emit_tokens,
        b_max=b_max, dp=fleet_dp, unchunkable=unchunkable, base_b_hint=base_b_hint,
        fast_span_probes=fast_span_probes, evaluation_context=evaluation_context)

    results = [run.solve_one(scen) for scen in scenario_list]
    base_r = next((row for row in results if row["scenario_id"] == "baseline"), None)
    if base_r is None:
        base_r = next((row for row in results if tuple(row["scales"].values()) == run.baseline_scales), None)
    if base_r is None:
        base_r = run.solve_one(DEFAULT_SLA_SCENARIOS[0])
    apply_sla_baseline_gains(results, base_r)
    async_mem_summary = _sla_async_memory(
        parsed, base_r, phase, context, q_tokens, emit_tokens,
        decode_q_tokens, decode_emit_tokens, output_budget)
    prefill = phase == "prefill"
    from ..frontend.body_spec import describe_included_components
    return {"schema": "mindie-sim.hardware-sla/1", "sla_ms": sla_ms, "context": context, "phase": phase, "scope": scope, "timing_boundary": "target_body_only" if scope == "body" else "full_model_step", "included_components": describe_included_components(parsed, scope=scope), "tokens_per_seq": q_tokens, "emitted_tokens_per_seq": emit_tokens, "prefill_tokens_per_seq": q_tokens if prefill else None, "decode_emitted_tokens_per_seq": decode_emit_tokens, "dp": dp, "max_sequences_per_dp": b_max,
            "solver": {"monotonicity_assumed": False, "b_sla_is_proven_max": False,
                       "objective": "maximize_throughput_under_sla",
                       "certified_dual_bound_search": True,
                       "bounded_non_monotonic_dispatch_probe": True,
                       "status_interpretation": "feasible_candidate_found_or_explicit_sla_capacity_domain_infeasibility"},
            "async_scheduling_memory": async_mem_summary, "scenarios": results}


def _sla_async_memory(parsed, base_r, phase, context, q_tokens, emit_tokens,
                      decode_q_tokens, decode_emit_tokens, output_budget):
    ws_b1 = _workspace_tokens_for_window(parsed, phase=phase, q_tokens=q_tokens, sequences=1)
    mem_async_b1 = Memory(replace(parsed, runtime=replace(parsed.runtime, async_scheduling=True)), workspace_tokens=ws_b1)
    prefill = phase == "prefill"
    lifecycle_query = decode_q_tokens if prefill else q_tokens
    lifecycle_emit = decode_emit_tokens if prefill else emit_tokens
    seq_sync_len = max(1, (context + q_tokens) if prefill else (context + max(q_tokens, output_budget)))
    seq_lifecycle_len = max(1, (context + q_tokens + max(lifecycle_query, output_budget)) if prefill else seq_sync_len)
    seq_async_len = max(1, seq_lifecycle_len + mem_async_b1.async_in_flight_lookahead_steps * lifecycle_query)
    if prefill:
        life_ctx = context + q_tokens
        life_sync = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=lifecycle_query,
                                                output_budget=output_budget, emitted_tokens_per_seq=lifecycle_emit,
                                                async_scheduling=False, phase="decode")
        life_async = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=lifecycle_query,
                                                 output_budget=output_budget, emitted_tokens_per_seq=lifecycle_emit,
                                                 async_scheduling=True, phase="decode")
    else:
        life_ctx = context
        life_sync, life_async = base_r["b_vram_max_sync"], base_r["b_vram_max_async"]
    capacity_window_b1 = capacity_window(
        parsed, phase=phase, context=context, tokens_per_seq=q_tokens, sequences=1,
        output_budget=output_budget, emitted_tokens_per_seq=emit_tokens)
    lifecycle_capacity_window_b1 = capacity_window(
        parsed, phase="decode", context=life_ctx, tokens_per_seq=lifecycle_query, sequences=1,
        output_budget=output_budget, emitted_tokens_per_seq=lifecycle_emit)

    return {
        "configured_async_scheduling": bool(parsed.runtime.async_scheduling),
        "capacity_window": "prefill_prompt_only" if prefill else "decode_step",
        "lifecycle_window": "prefix_query_plus_output_budget" if prefill else "decode_step",
        "prefill_tokens_per_seq": q_tokens if prefill else None,
        "decode_emitted_tokens_per_seq": lifecycle_emit,
        "b_vram_max_sync": base_r["b_vram_max_sync"],
        "b_vram_max_async": base_r["b_vram_max_async"],
        "b_vram_max_lifecycle_sync": int(life_sync),
        "b_vram_max_lifecycle_async": int(life_async),
        "b_vram_max_joint_sync": int(min(base_r["b_vram_max_sync"], life_sync)),
        "b_vram_max_joint_async": int(min(base_r["b_vram_max_async"], life_async)),
        "delta_b_vram_from_disabling_async": base_r["delta_b_vram_from_disabling_async"],
        "delayed_release_steps_async": mem_async_b1.async_delayed_release_steps,
        "in_flight_lookahead_tokens_async": mem_async_b1.async_in_flight_lookahead_steps * lifecycle_query,
        "kv_bytes_per_seq_sync_per_stage": [float(value) for value in mem_async_b1.cache_bytes(seq_sync_len, padded=True)],
        "kv_bytes_per_seq_lifecycle_per_stage": [float(value) for value in mem_async_b1.cache_bytes(seq_lifecycle_len, padded=True)],
        "kv_bytes_per_seq_async_per_stage": [float(value) for value in mem_async_b1.cache_bytes(seq_async_len, padded=True)],
        "async_workspace_overhead_bytes_at_b1_per_stage": [float(value) for value in mem_async_b1.async_workspace],
        "capacity_window_b1": capacity_window_b1,
        "lifecycle_capacity_window_b1": lifecycle_capacity_window_b1,
        "joint_capacity_b1": {
            "supply_bytes": min(capacity_window_b1["supply_bytes"], lifecycle_capacity_window_b1["supply_bytes"]),
            "active_kv_bytes": capacity_window_b1["active_kv_bytes"],
            "retiring_kv_bytes": lifecycle_capacity_window_b1["retiring_kv_bytes"],
            "workspace_bytes": max(capacity_window_b1["workspace_bytes"], lifecycle_capacity_window_b1["workspace_bytes"]),
            "peak_bytes": max(capacity_window_b1["peak_bytes"], lifecycle_capacity_window_b1["peak_bytes"]),
            "headroom_bytes": min(capacity_window_b1["headroom_bytes"], lifecycle_capacity_window_b1["headroom_bytes"]),
            "fits": bool(capacity_window_b1["fits"] and lifecycle_capacity_window_b1["fits"]),
        },
        "hbm_bound_at_baseline": bool(base_r["b_vram_max"] < base_r["b_sla"]),
        "disabling_async_unlocks_concurrency": bool(base_r["b_vram_max_sync"] > base_r["b_vram_max_async"] and base_r["b_vram_max_async"] < base_r["b_sla"]),
    }


def _cohort_die_ranks(case: CaseSpec, group: tuple[int, ...]) -> tuple[int, ...]:
    placement = case.machine.validate_placement(case.runtime.parallel)
    return tuple(int(rk) for rk in placement.physical_ranks_for_cohort(group))


def _cohort_hw_signature(case: CaseSpec, group: tuple[int, ...]) -> tuple[Any, ...]:
    """Canonical die, placement-partition, and routed link-supply signature for one cohort."""
    from ..collectives.topology import Topology
    placement = case.machine.validate_placement(case.runtime.parallel)
    ranks = placement.physical_ranks_for_cohort(group)
    local_idx = {int(rk): idx for idx, rk in enumerate(ranks)}
    die_sigs = tuple(placement.die_for_rank(rk).resource_signature for rk in ranks)
    topo = Topology(case.machine, case.runtime.parallel)
    node_ids: dict[Any, int] = {}
    pkg_ids: dict[Any, int] = {}
    placement_part = tuple(
        (
            node_ids.setdefault(topo._node_of(rk), len(node_ids)),
            pkg_ids.setdefault(topo._package_key(rk), len(pkg_ids)),
        )
        for rk in ranks
    )
    link_alias: dict[str, int] = {}
    hop_cache: dict[tuple[str, int, int], tuple[Any, ...]] = {}

    def _shared_external(lk: Any) -> int:
        if (lk.semantics == "shared"
                or lk.switch_capacity_bytes_per_ms is not None
                or bool(lk.root_complex_id)):
            return sum(1 for member in lk.members if int(member) not in local_idx)
        return 0

    def _hop_sig(name: str, src: int, dst: int) -> tuple[Any, ...]:
        key = (name, src, dst)
        cached = hop_cache.get(key)
        if cached is not None:
            return cached
        lk = topo.links[name]
        sig = (
            link_alias.setdefault(name, len(link_alias)),
            str(lk.channel), str(lk.protocol), str(lk.semantics),
            str(lk.scope), repr(topo.link_rail(name)),
            float(lk.capacity_at(src)), float(lk.capacity_at(dst)),
            float(lk.capacity_bytes_per_ms), lk.peak_capacity_bytes_per_ms,
            float(lk.bw_utilization), float(lk.propagation_us),
            lk.switch_capacity_bytes_per_ms, lk.transaction_bytes,
            int(lk.alignment_bytes), int(lk.protocol_overhead_bytes),
            lk.protocol_bands, _shared_external(lk),
        )
        hop_cache[key] = sig
        return sig

    distinct_rails = topo.distinct_rails
    rail_selectors: tuple[int | str | None, ...] = (
        (None, *distinct_rails) if len(distinct_rails) > 1 else (None,)
    )
    route_sigs: list[tuple[int, int, str, tuple[Any, ...]]] = []
    for i, src in enumerate(ranks):
        for j, dst in enumerate(ranks):
            if i == j:
                continue
            for rail_id in rail_selectors:
                try:
                    hops = tuple(
                        _hop_sig(name, h_src, h_dst)
                        for name, h_src, h_dst in topo.route(src, dst, rail_id=rail_id)
                    )
                except ValueError:
                    hops = (("missing",),)
                route_sigs.append((i, j, repr(rail_id), hops))

    host_sigs = tuple(sorted(
        (
            str(lk.channel), str(lk.protocol), str(lk.semantics), str(lk.scope),
            float(lk.capacity_bytes_per_ms), float(lk.propagation_us),
            tuple(sorted(
                (local_idx[int(m)], float(lk.capacity_at(int(m))))
                for m in lk.members if int(m) in local_idx
            )),
            _shared_external(lk),
        )
        for lk in case.machine.links
        if any(int(member) in local_idx for member in lk.members)
        and (
            len(lk.members) == 1
            or str(lk.scope or "") in {"host", "h2d", "d2h", "cpu", "pcie"}
            or any(tag in str(lk.channel).lower() for tag in ("host", "h2d", "d2h", "pcie", "dma"))
        )
    ))
    return (die_sigs, placement_part, tuple(route_sigs), host_sigs)


def _external_worker_transit_ranks(
    case: CaseSpec, worker_ranks: tuple[GlobalRank, ...],
) -> tuple[GlobalRank, ...]:
    """Collect any intermediate transit ranks required to route between ``worker_ranks``."""
    if len(worker_ranks) <= 1 or not case.machine.links:
        return ()
    from ..collectives.topology import Topology
    topo = Topology(case.machine, case.runtime.parallel)
    distinct_rails = topo.distinct_rails
    rail_selectors: tuple[int | str | None, ...] = (
        (None, *distinct_rails) if len(distinct_rails) > 1 else (None,)
    )
    seen: set[int] = {int(rk) for rk in worker_ranks}
    transit: list[GlobalRank] = []
    pairs = tuple(
        (int(src), int(dst))
        for src in worker_ranks
        for dst in worker_ranks
        if src != dst
    )
    for src, dst in pairs:
        for rail_id in rail_selectors:
            try:
                hops = topo.route(src, dst, rail_id=rail_id)
            except ValueError:
                continue
            for _link_name, hop_src, hop_dst in hops:
                for endpoint in (int(hop_src), int(hop_dst)):
                    if endpoint not in seen:
                        seen.add(endpoint)
                        transit.append(GlobalRank(endpoint))
    return tuple(transit)


def _external_worker_case(case: CaseSpec, ext_idx: int) -> CaseSpec:
    """Project one external DP worker slice ``ext_idx`` onto a single-worker ``Case``."""
    p = case.runtime.parallel
    ext_dp = max(1, int(getattr(p, "external_dp", 1) or 1))
    if ext_dp <= 1:
        return case
    dp_inner = max(1, int(p.dp))
    worker_replicas = tuple(range(ext_idx * dp_inner, (ext_idx + 1) * dp_inner))
    placement = case.machine.validate_placement(p)
    worker_ranks = placement.physical_ranks_for_cohort(worker_replicas)
    worker_rank_set = frozenset(worker_ranks)
    transit_ranks = _external_worker_transit_ranks(case, worker_ranks)
    worker_dies = tuple(placement.die_for_rank(rk) for rk in worker_ranks) + tuple(
        placement.die_for_rank(rk) for rk in transit_ranks
    )
    worker_links = case.machine.induced_links_for(worker_rank_set | frozenset(transit_ranks))
    worker_rank_table = None
    if p.rank_table is not None:
        worker_rank_table = tuple(
            group for group in p.rank_table
            if all(int(r) in worker_rank_set for r in group)
        ) or None
    worker_parallel = replace(p, external_dp=1, rank_table=worker_rank_table)
    worker_machine = replace(case.machine, dies=worker_dies, links=worker_links)
    return replace(
        case,
        machine=worker_machine,
        runtime=replace(case.runtime, parallel=worker_parallel),
    )


def _distinct_eval_cohorts(case: CaseSpec) -> tuple[tuple[int, ...], ...]:
    """Return one representative cohort per distinct die+link hardware class across all external_dp*dp replicas."""
    from ..execution.dispatch import cohorts
    p = case.runtime.parallel
    dp_inner = max(1, int(p.dp))
    ext_dp = max(1, int(getattr(p, "external_dp", 1) or 1))
    if ext_dp > 1:
        raw_groups: list[tuple[int, ...]] = []
        for ext_idx in range(ext_dp):
            worker = _external_worker_case(case, ext_idx)
            for group in cohorts(worker.runtime.parallel, worker.machine):
                raw_groups.append(tuple(ext_idx * dp_inner + int(r) for r in group))
        groups = tuple(raw_groups)
    else:
        groups = tuple(tuple(int(r) for r in group) for group in cohorts(p, case.machine))
    if not groups:
        return ()
    if len(groups) == 1:
        return groups[:1]
    seen: set[tuple[Any, ...]] = set()
    distinct: list[tuple[int, ...]] = []
    for group in groups:
        sig = _cohort_hw_signature(case, group)
        if sig not in seen:
            seen.add(sig)
            distinct.append(group)
    return tuple(distinct)


def _prefill_chunk_schedule(parsed: CaseSpec, *, phase: str, context: int, q_tokens: int) -> tuple[tuple[int, int], ...]:
    """Split a chunked prefill prompt exceeding the step token budget into ordered chunks."""
    budget = _prefill_step_token_budget(parsed)
    if phase != "prefill" or not parsed.runtime.enable_chunked_prefill or q_tokens <= budget:
        return ((int(context), int(q_tokens)),)
    chunks: list[tuple[int, int]] = []
    cur_ctx = int(context)
    remaining = int(q_tokens)
    while remaining > 0:
        width = min(remaining, budget)
        chunks.append((cur_ctx, width))
        cur_ctx += width
        remaining -= width
    return tuple(chunks)


def _prefill_batch_chunk_schedule(
    parsed: CaseSpec, *, phase: str, context: int, q_tokens: int, sequences: int,
) -> tuple[tuple[int, int, int], ...]:
    """Split ``sequences`` chunked prefill prompts into ordered ``(ctx, tok, seqs)`` steps within budget."""
    b = max(1, int(sequences))
    budget = _prefill_step_token_budget(parsed)
    if phase != "prefill" or not parsed.runtime.enable_chunked_prefill or (b * q_tokens) <= budget:
        return ((int(context), int(q_tokens), b),)
    if b == 1:
        return tuple((ctx, tok, 1) for ctx, tok in _prefill_chunk_schedule(
            parsed, phase=phase, context=context, q_tokens=q_tokens))
    chunks: list[tuple[int, int, int]] = []
    rem_seqs = b
    while rem_seqs > 0:
        wave_seqs = min(rem_seqs, budget)
        per_seq = max(1, budget // wave_seqs)
        cur_ctx = int(context)
        remaining = int(q_tokens)
        while remaining > 0:
            width = min(remaining, per_seq)
            chunks.append((cur_ctx, width, wave_seqs))
            cur_ctx += width
            remaining -= width
        rem_seqs -= wave_seqs
    return tuple(chunks)


class _SLASizingRun:
    """Own graph and memory reuse, plus the search domain shared by the scenarios."""

    def __init__(
        self, parsed, *, sla_ms, context, phase, scope, output_budget,
        q_tokens, emit_tokens, decode_q_tokens, decode_emit_tokens,
        b_max, dp, unchunkable, base_b_hint, fast_span_probes, evaluation_context,
    ):
        self.parsed = parsed
        self.sla_ms = sla_ms
        self.context = context
        self.phase = phase
        self.scope = scope
        self.output_budget = output_budget
        self.q_tokens = q_tokens
        self.emit_tokens = emit_tokens
        self.decode_q_tokens = decode_q_tokens
        self.decode_emit_tokens = decode_emit_tokens
        self.b_max = b_max
        self.dp = dp
        self.unchunkable = unchunkable
        self.evaluation_context = evaluation_context
        self.eval_cohorts = _distinct_eval_cohorts(parsed)
        self.prefill_chunks = _prefill_chunk_schedule(
            parsed, phase=phase, context=context, q_tokens=q_tokens)
        self._chunks_by_b: dict[int, tuple[tuple[int, int, int], ...]] = {}
        self.scaled_cases: dict[tuple[float, ...], CaseSpec] = {}
        self.eval_contexts: dict[tuple[float, ...], Any] = {}
        self.worker_base_cases: dict[int, CaseSpec] = {}
        self.worker_scaled_cases: dict[tuple[tuple[float, ...], int], CaseSpec] = {}
        self.worker_eval_contexts: dict[tuple[tuple[float, ...], int], Any] = {}
        self.floor_caps_cache: dict[tuple[Any, ...], Any] = {}
        self.graphs_by_b: dict[Any, Any] = {}
        self.vram_by_cap: dict[float, tuple[int, int, int]] = {}
        self.graph_cache: dict[tuple[tuple[float, ...], int, bool], tuple[Any, CaseSpec, dict[str, Any]]] = {}
        self.fast_spans = bool(fast_span_probes)
        self.analysis_cache: dict[tuple[tuple[float, ...], int], dict[str, Any]] = {}
        self.baseline_scales = tuple(1.0 for _ in _SCALE_KEYS)
        self.base_b_hint = max(1, int(base_b_hint))
        self.domain_cap = b_max
        self.domain_reason = ("prompt_exceeds_batch_budget_without_chunking"
                         if unchunkable else None)
        self.memory_cache: dict[tuple[int, bool | None, float], Memory] = {}

    def get_scaled(self, s_key: tuple[float, ...]) -> CaseSpec:
        if s_key not in self.scaled_cases:
            self.scaled_cases[s_key] = scale_case_hardware(self.parsed, **dict(zip(_SCALE_KEYS, s_key)))
        return self.scaled_cases[s_key]

    def get_eval_context(self, s_key: tuple[float, ...]) -> Any:
        if s_key == self.baseline_scales and self.evaluation_context is not None:
            return self.evaluation_context
        if s_key not in self.eval_contexts:
            from ..evaluation import EvaluationContext
            self.eval_contexts[s_key] = EvaluationContext.create(self.get_scaled(s_key))
        return self.eval_contexts[s_key]

    def get_worker_base_case(self, ext_idx: int) -> CaseSpec:
        if ext_idx not in self.worker_base_cases:
            self.worker_base_cases[ext_idx] = _external_worker_case(self.parsed, ext_idx)
        return self.worker_base_cases[ext_idx]

    def get_worker_scaled_case(self, s_key: tuple[float, ...], ext_idx: int) -> CaseSpec:
        key = (s_key, ext_idx)
        if key not in self.worker_scaled_cases:
            self.worker_scaled_cases[key] = _external_worker_case(self.get_scaled(s_key), ext_idx)
        return self.worker_scaled_cases[key]

    def get_worker_eval_context(self, s_key: tuple[float, ...], ext_idx: int) -> Any:
        key = (s_key, ext_idx)
        if key not in self.worker_eval_contexts:
            from ..evaluation import EvaluationContext
            self.worker_eval_contexts[key] = EvaluationContext.create(self.get_worker_scaled_case(s_key, ext_idx))
        return self.worker_eval_contexts[key]

    def _chunks_for(self, b: int) -> tuple[tuple[int, int, int], ...]:
        b_int = max(1, int(b))
        if b_int not in self._chunks_by_b:
            self._chunks_by_b[b_int] = _prefill_batch_chunk_schedule(
                self.parsed, phase=self.phase, context=self.context,
                q_tokens=self.q_tokens, sequences=b_int,
            )
        return self._chunks_by_b[b_int]

    def _run_one_geometry(
        self, s_key: tuple[float, ...], b: int, *,
        chunk_ctx: int, chunk_tok: int, multi_chunk: bool, fast: bool,
    ) -> tuple[Any, CaseSpec, dict[str, Any]]:
        parsed, phase, scope = self.parsed, self.phase, self.scope
        graphs_by_b = self.graphs_by_b
        p = parsed.runtime.parallel
        dp_inner = max(1, int(p.dp))
        ext_dp = max(1, int(getattr(p, "external_dp", 1) or 1))
        context_reuse = self.get_eval_context(s_key)
        scaled_case = self.get_scaled(s_key)
        if len(self.eval_cohorts) <= 1 and ext_dp <= 1:
            g_key: Any = (b, chunk_ctx, chunk_tok) if multi_chunk else b
            if g_key not in graphs_by_b:
                graphs_by_b[g_key] = _pkg_attr("target_graphs", target_graphs)(
                    parsed, phase=phase, tokens=chunk_tok, context=chunk_ctx,
                    sequences=b, scope=scope)
            g, ec, gr = _pkg_attr("_lower_and_run_graph", _lower_and_run_graph)(
                scaled_case, phase=phase, tokens=chunk_tok, context=chunk_ctx,
                sequences=b, scope=scope, run_graph=True,
                lowered_graphs=graphs_by_b[g_key], evaluation_context=context_reuse,
                fast_span_only=bool(fast))
            assert gr is not None
            return g, ec, gr
        slowest: tuple[Any, CaseSpec, dict[str, Any]] | None = None
        for group in self.eval_cohorts:
            g_key = (b, chunk_ctx, chunk_tok, group) if multi_chunk else (b, group)
            ext_idx = (group[0] // dp_inner) if ext_dp > 1 else 0
            local_group = tuple(r - ext_idx * dp_inner for r in group) if ext_dp > 1 else group
            base_case = self.get_worker_base_case(ext_idx) if ext_dp > 1 else parsed
            run_case = self.get_worker_scaled_case(s_key, ext_idx) if ext_dp > 1 else scaled_case
            run_ctx = self.get_worker_eval_context(s_key, ext_idx) if ext_dp > 1 else context_reuse
            if g_key not in graphs_by_b:
                graphs_by_b[g_key] = _pkg_attr("target_graphs", target_graphs)(
                    base_case, phase=phase, tokens=chunk_tok, context=chunk_ctx,
                    sequences=b, scope=scope, cohort=local_group)
            g, ec, gr = _pkg_attr("_lower_and_run_graph", _lower_and_run_graph)(
                run_case, phase=phase, tokens=chunk_tok, context=chunk_ctx,
                sequences=b, scope=scope, run_graph=True,
                lowered_graphs=graphs_by_b[g_key], evaluation_context=run_ctx,
                fast_span_only=bool(fast))
            assert gr is not None
            if ext_dp > 1:
                gr = {**gr, "_eval_ext_idx": ext_idx}
            if slowest is None or float(gr["duration_ms"]) > float(slowest[2]["duration_ms"]):
                slowest = (g, ec, gr)
        assert slowest is not None
        return slowest

    def get_bundle(self, s_key: tuple[float, ...], b: int, *,
                   fast: bool = False) -> tuple[Any, CaseSpec, dict[str, Any]]:
        graph_cache = self.graph_cache
        check_active_deadline()
        if (s_key, b, fast) not in graph_cache:
            chunks = self._chunks_for(b)
            if len(chunks) == 1:
                chunk_ctx, chunk_tok, chunk_seqs = chunks[0]
                graph_cache[s_key, b, fast] = self._run_one_geometry(
                    s_key, chunk_seqs, chunk_ctx=chunk_ctx, chunk_tok=chunk_tok,
                    multi_chunk=False, fast=fast)
            else:
                total_ms = 0.0
                chunk_bundles: list[tuple[Any, CaseSpec, dict[str, Any]]] = []
                for chunk_ctx, chunk_tok, chunk_seqs in chunks:
                    bundle = self._run_one_geometry(
                        s_key, chunk_seqs, chunk_ctx=chunk_ctx, chunk_tok=chunk_tok,
                        multi_chunk=True, fast=fast)
                    total_ms += float(bundle[2]["duration_ms"])
                    chunk_bundles.append(bundle)
                last_g, last_ec, last_gr = chunk_bundles[-1]
                agg_gr = {
                    **last_gr,
                    "duration_ms": total_ms,
                    "chunk_durations_ms": [float(item[2]["duration_ms"]) for item in chunk_bundles],
                    "_chunk_bundles": chunk_bundles,
                }
                graph_cache[s_key, b, fast] = (last_g, last_ec, agg_gr)
        return graph_cache[s_key, b, fast]

    def get_analysis(self, s_key: tuple[float, ...], b: int) -> dict[str, Any]:
        phase, q_tokens, context, scope = self.phase, self.q_tokens, self.context, self.scope
        analysis_cache = self.analysis_cache
        check_active_deadline()
        if (s_key, b) not in analysis_cache:
            ext_dp = max(1, int(getattr(self.parsed.runtime.parallel, "external_dp", 1) or 1))
            chunks = self._chunks_for(b)
            if len(chunks) == 1:
                bundle = self.get_bundle(s_key, b, fast=False)
                ext_idx = bundle[2].get("_eval_ext_idx") if ext_dp > 1 else None
                eff_scaled = self.get_worker_scaled_case(s_key, int(ext_idx)) if ext_idx is not None else self.get_scaled(s_key)
                eff_ctx = self.get_worker_eval_context(s_key, int(ext_idx)) if ext_idx is not None else self.get_eval_context(s_key)
                analysis_cache[s_key, b] = _pkg_attr("analyze_network_efficiency", analyze_network_efficiency)(
                    eff_scaled, phase=phase, tokens=q_tokens, context=context,
                    sequences=b, scope=scope, run_graph=True,
                    _precomputed_graph=bundle,
                    evaluation_context=eff_ctx)
            else:
                agg_bundle = self.get_bundle(s_key, b, fast=False)
                chunk_bundles = agg_bundle[2]["_chunk_bundles"]
                chunk_reports = []
                for (chunk_ctx, chunk_tok, chunk_seqs), chunk_bundle in zip(chunks, chunk_bundles):
                    c_ext = chunk_bundle[2].get("_eval_ext_idx") if ext_dp > 1 else None
                    c_scaled = self.get_worker_scaled_case(s_key, int(c_ext)) if c_ext is not None else self.get_scaled(s_key)
                    c_ctx = self.get_worker_eval_context(s_key, int(c_ext)) if c_ext is not None else self.get_eval_context(s_key)
                    chunk_reports.append(
                        _pkg_attr("analyze_network_efficiency", analyze_network_efficiency)(
                            c_scaled, phase=phase, tokens=chunk_tok,
                            context=chunk_ctx, sequences=chunk_seqs, scope=scope, run_graph=True,
                            _precomputed_graph=chunk_bundle,
                            evaluation_context=c_ctx)
                    )
                total_graph = sum(float(rep["graph_duration_ms"]) for rep in chunk_reports)
                total_op_sum = sum(float(rep["operator_sum_ms"]) for rep in chunk_reports)
                safe_g = max(total_graph, 1e-12)
                safe_op = max(total_op_sum, 1e-12)
                tot_hbm = sum(float(rep.get("total_hbm_ideal_ms", 0.0) or 0.0) for rep in chunk_reports)
                tot_cube = sum(float(rep.get("total_cube_ideal_ms", 0.0) or 0.0) for rep in chunk_reports)
                tot_vec = sum(float(rep.get("total_vec_ideal_ms", 0.0) or 0.0) for rep in chunk_reports)
                tot_link = sum(float(rep.get("total_link_ideal_ms", 0.0) or 0.0) for rep in chunk_reports)
                tot_roof = sum(float(rep.get("total_roofline_ideal_ms", 0.0) or 0.0) for rep in chunk_reports)
                bound_totals = {
                    bound: sum(
                        float(rep["breakdown_by_bound"][bound]["total_ms"])
                        for rep in chunk_reports
                    )
                    for bound in _BOUNDS
                }
                analysis_cache[s_key, b] = {
                    **chunk_reports[-1],
                    "graph_duration_ms": total_graph,
                    "operator_sum_ms": total_op_sum,
                    "network_mbu": tot_hbm / safe_g,
                    "network_cube_mfu": tot_cube / safe_g,
                    "network_vec_mfu": tot_vec / safe_g,
                    "network_link_util": tot_link / safe_g,
                    "network_roofline_efficiency": tot_roof / safe_g,
                    "breakdown_by_bound": {
                        bound: {
                            **chunk_reports[-1]["breakdown_by_bound"][bound],
                            "total_ms": bound_totals[bound],
                            "share_of_operator_sum": bound_totals[bound] / safe_op,
                        }
                        for bound in _BOUNDS
                    },
                }
        return analysis_cache[s_key, b]

    def eval_at(self, s_key: tuple[float, ...], b: int, *, full: bool = False) -> dict[str, Any]:
        if self.unchunkable:
            return {"step_ms": 0.0, "step_basis": "geometry_rejected",
                    "step_is_lower_bound": False, "operator_sum_ms": 0.0,
                    "network_mbu": None,
                    "network_cube_mfu": None, "network_vec_mfu": None,
                    "network_link_util": None,
                    "network_roofline_efficiency": None,
                    "bottleneck_shares": {}}
        if not full:
            return {"step_ms": float(
                self.get_bundle(s_key, b, fast=self.fast_spans)[2]["duration_ms"])}
        ans = self.get_analysis(s_key, b)
        shares = {k: ans["breakdown_by_bound"][k]["share_of_operator_sum"] for k in _BOUNDS}
        return {"step_ms": float(ans["graph_duration_ms"]), "step_basis": "graph_evaluation", "step_is_lower_bound": False, "operator_sum_ms": float(ans["operator_sum_ms"]), "network_mbu": float(ans["network_mbu"]), "network_cube_mfu": float(ans["network_cube_mfu"]), "network_vec_mfu": float(ans["network_vec_mfu"]), "network_link_util": float(ans["network_link_util"]), "network_roofline_efficiency": float(ans["network_roofline_efficiency"]), "bottleneck_shares": shares}

    def try_step_ok(
        self, s_key: tuple[float, ...], b: int, *, window_for,
        seen: set[int] | None = None, checks: dict[int, dict[str, Any]] | None = None,
        capacity_max: int | None = None,
    ) -> bool:
        sla_ms = self.sla_ms
        if seen is not None:
            seen.add(int(b))
        if b > self.domain_cap:
            return False
        try:
            step = float(self.eval_at(s_key, b, full=False)["step_ms"])
            capacity_pass = True
            if checks is not None:
                if window_for is None:
                    raise ModelingError("sla_capacity_window_provider_not_bound")
                current_window, lifecycle_window = window_for(int(b))
                capacity_pass = bool(
                    (capacity_max is None or b <= capacity_max)
                    and current_window["fits"] and lifecycle_window["fits"])
                checks[int(b)] = {
                    "b": int(b), "step_ms": step,
                    "sla": {"limit_ms": float(sla_ms), "pass": step <= sla_ms},
                    "capacity": {"b_vram_max": int(capacity_max or 0),
                                 "current_fits": bool(current_window["fits"]),
                                 "lifecycle_fits": bool(lifecycle_window["fits"]),
                                 "pass": capacity_pass},
                    "feasible": bool(step <= sla_ms and capacity_pass)}
            return bool(step <= sla_ms)
        except UnmodeledCapabilityError:
            # An unmodeled core is not an infeasible candidate.
            raise
        except ModelingError:
            raise

    def memory_for(self, tokens: int, async_scheduling: bool | None, scale: float) -> Memory:
        parsed, memory_cache = self.parsed, self.memory_cache
        key = (int(tokens), async_scheduling, float(scale))
        if key not in memory_cache:
            base_case = (scale_case_hardware(parsed, hbm_capacity_scale=scale)
                         if scale != 1.0 else parsed)
            if async_scheduling is not None:
                base_case = replace(base_case, runtime=replace(
                    base_case.runtime, async_scheduling=bool(async_scheduling)))
            memory_cache[key] = Memory(base_case, workspace_tokens=int(tokens))
        return memory_cache[key]

    def _scenario_capacity(self, hbm_capacity_scale):
        parsed, phase, context = self.parsed, self.phase, self.context
        q_tokens, emit_tokens = self.q_tokens, self.emit_tokens
        decode_q_tokens, decode_emit_tokens = self.decode_q_tokens, self.decode_emit_tokens
        output_budget, unchunkable = self.output_budget, self.unchunkable
        vram_by_cap, memory_for = self.vram_by_cap, self.memory_for
        if hbm_capacity_scale not in vram_by_cap and unchunkable:
            vram_by_cap[hbm_capacity_scale] = (0, 0, 0)
        if hbm_capacity_scale not in vram_by_cap:
            ws_b1 = _workspace_tokens_for_window(parsed, phase=phase, q_tokens=q_tokens, sequences=1)
            b_sync = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=hbm_capacity_scale, emitted_tokens_per_seq=emit_tokens, async_scheduling=False, phase=phase, memory=memory_for(ws_b1, False, hbm_capacity_scale))
            b_async = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=hbm_capacity_scale, emitted_tokens_per_seq=emit_tokens, async_scheduling=True, phase=phase, memory=memory_for(ws_b1, True, hbm_capacity_scale))
            b_cfg = b_async if parsed.runtime.async_scheduling else b_sync
            if parsed.runtime.memory:
                b_cfg = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=hbm_capacity_scale, emitted_tokens_per_seq=emit_tokens, phase=phase, memory=memory_for(ws_b1, None, hbm_capacity_scale))
            vram_by_cap[hbm_capacity_scale] = (b_cfg, b_sync, b_async)
        (b_vram, b_vram_sync, b_vram_async) = vram_by_cap[hbm_capacity_scale]
        if phase == "prefill":
            life_ctx = context + q_tokens
            verify_tokens = decode_q_tokens
            b_life_sync = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=verify_tokens,
                                                      output_budget=output_budget, hbm_capacity_scale=hbm_capacity_scale,
                                                      emitted_tokens_per_seq=decode_emit_tokens, async_scheduling=False, phase="decode",
                                                      memory=memory_for(verify_tokens, False, hbm_capacity_scale))
            b_life_async = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=verify_tokens,
                                                       output_budget=output_budget, hbm_capacity_scale=hbm_capacity_scale,
                                                       emitted_tokens_per_seq=decode_emit_tokens, async_scheduling=True, phase="decode",
                                                       memory=memory_for(verify_tokens, True, hbm_capacity_scale))
            b_lifecycle = b_life_async if parsed.runtime.async_scheduling else b_life_sync
        else:
            life_ctx = context
            verify_tokens = q_tokens
            b_life_sync, b_life_async, b_lifecycle = b_vram_sync, b_vram_async, b_vram
        window_cache: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}

        def rejected_window():
            empty = {
                "supply_bytes": 0.0, "active_kv_bytes": 0.0,
                "retiring_kv_bytes": 0.0, "workspace_bytes": 0.0,
                "peak_bytes": 0.0, "headroom_bytes": 0.0, "fits": False,
            }
            return empty

        def candidate_windows(b: int) -> tuple[dict[str, Any], dict[str, Any]]:
            if b in window_cache:
                return window_cache[b]
            if unchunkable:
                window_cache[b] = (rejected_window(), rejected_window())
                return window_cache[b]
            current_window = capacity_window(
                parsed, phase=phase, context=context, tokens_per_seq=q_tokens,
                sequences=b, output_budget=output_budget,
                emitted_tokens_per_seq=emit_tokens, hbm_capacity_scale=hbm_capacity_scale)
            if phase == "decode":
                lifecycle_window = current_window
            else:
                lifecycle_window = capacity_window(
                    parsed, phase="decode", context=life_ctx, tokens_per_seq=verify_tokens,
                    sequences=b, output_budget=output_budget,
                    emitted_tokens_per_seq=decode_emit_tokens,
                    hbm_capacity_scale=hbm_capacity_scale)
            window_cache[b] = (current_window, lifecycle_window)
            return current_window, lifecycle_window
        return (
            (b_vram, b_vram_sync, b_vram_async),
            (b_lifecycle, b_life_sync, b_life_async),
            life_ctx, verify_tokens, candidate_windows,
        )

    def solve_one(self, scen: Mapping[str, Any]) -> dict[str, Any]:
        check_active_deadline()
        s_key = tuple(float(scen.get(k, 1.0)) for k in _SCALE_KEYS)
        hbm_capacity_scale = s_key[-1]
        current_caps, lifecycle_caps, life_ctx, verify_tokens, candidate_windows = (
            self._scenario_capacity(hbm_capacity_scale))
        b_joint = min(current_caps[0], lifecycle_caps[0])
        unchunkable, sla_ms = self.unchunkable, self.sla_ms
        parsed = self.parsed
        b1_ms = (0.0 if unchunkable else
                 float(self.get_bundle(s_key, 1, fast=self.fast_spans)[2]["duration_ms"]))
        current_one, lifecycle_one = candidate_windows(1)
        b1_capacity = bool(b_joint >= 1 and current_one["fits"]
                           and lifecycle_one["fits"] and not unchunkable)
        candidate_checks: dict[int, dict[str, Any]] = {
            1: {"b": 1, "step_ms": b1_ms,
                "step_basis": ("geometry_rejected" if unchunkable
                               else "graph_evaluation"),
                "sla": {"limit_ms": float(sla_ms),
                        "pass": bool(b1_ms <= sla_ms and not unchunkable)},
                "capacity": {"b_vram_max": int(b_joint),
                             "current_fits": bool(current_one["fits"]),
                             "lifecycle_fits": bool(lifecycle_one["fits"]),
                             "pass": b1_capacity},
                "feasible": bool(b1_ms <= sla_ms and b1_capacity)}}
        evaluated_b: set[int] = set(candidate_checks)
        eff_max = min(self.b_max, self.domain_cap)
        bucket_keys = sorted({int(key) for key in (getattr(parsed.runtime, "graph_capture_keys", ()) or ())}
                             | {int(key) for key in (getattr(parsed.runtime, "piecewise_capture_keys", ()) or ())})

        (
            b_sla, eval_sla, bounded_probes, secant_estimate,
            certified_optimal, hi_cut,
        ) = self._search_concurrency(
            s_key, b1_ms, eff_max, bucket_keys, candidate_checks, evaluated_b,
            b_joint, candidate_windows)
        if scen.get("id") == "baseline" and b_sla > 0:
            self.base_b_hint = b_sla
        return self._scenario_result(
            scen, s_key, current_caps, lifecycle_caps, b_joint,
            life_ctx, verify_tokens, candidate_windows, b1_ms, bucket_keys,
            candidate_checks, evaluated_b, b_sla, eval_sla, bounded_probes,
            secant_estimate, certified_optimal=certified_optimal, hi_cut=hi_cut)

    @staticmethod
    def _collective_band_floor_info(eff_case: CaseSpec, eval_ctx: Any, links: Sequence[Any]):
        """Return ``(band_floor_ms_by_family, recipe_book, topology)`` sound across all ``b' >= b``."""
        bindings = getattr(eval_ctx, "bindings", eval_ctx) if eval_ctx is not None else None
        topo = getattr(bindings, "topology", None) if bindings is not None else None
        if topo is None or not getattr(topo, "_has_primitive_table", False):
            return {}, None, None
        kernels = getattr(eff_case.runtime, "kernels", None) or {}
        pos_links = [
            lk for lk in links
            if float(getattr(lk, "capacity_bytes_per_ms", 0.0) or 0.0) > 0.0
            and len(getattr(lk, "members", ()) or ()) >= _MIN_COLL_GROUP_SIZE
        ]
        if not pos_links:
            return {}, None, None
        all_bands = []
        for lk in pos_links:
            bands = tuple(topo.link_protocol_bands(lk.name))
            if not bands:
                return {}, None, None
            all_bands.extend(bands)
        if not all_bands:
            return {}, None, None
        p18_col = str(getattr(topo, "_p18_measured_collective", "") or "").strip().lower()
        p18_cands_us: list[float] = []
        for attr in ("p18_startup_us", "p18_eager_startup_us"):
            val = float(getattr(topo, attr, 0.0) or 0.0)
            if val > 0.0:
                p18_cands_us.append(val)
        p19_cands_us: list[float] = []
        uncoalesced = bool(getattr(topo, "_p19_uncoalesced_duplex", False))
        host_sub_us = float(getattr(topo, "_p15_host_submit_us", 0.0) or 0.0)
        p19_blk = float(getattr(topo, "_p19_duplex_blocked_us", 0.0) or 0.0)
        if p19_blk > 0.0:
            p19_cands_us.append(p19_blk / _DUPLEX_HALF_DIVISOR if uncoalesced else p19_blk)
        p19_egr = float(getattr(topo, "_p19_duplex_eager_us", 0.0) or 0.0)
        if p19_egr > 0.0:
            egr_floor = (
                p19_egr / _DUPLEX_HALF_DIVISOR
                if uncoalesced
                else max(0.0, p19_egr - host_sub_us)
            )
            if egr_floor > 0.0:
                p19_cands_us.append(egr_floor)
        oneshot_us = float(topo._oneshot_startup_ms()) * _US_PER_MS
        floors_ms: dict[str, float] = {}
        for fam in _BAND_FLOOR_COLLECTIVES:
            min_ds_us = min(
                float(topo.collective_device_step_us(
                    fam, _MIN_COLL_PAIR_GROUP,
                    alpha_us=float(getattr(band, "startup_us", 0.0) or 0.0),
                    stack="graph", phase_count=1, total_rounds=1,
                    startup_scope=str(getattr(band, "startup_scope", "step") or "step"),
                ))
                for band in all_bands
            )
            if min_ds_us <= 0.0:
                continue
            cands = [min_ds_us, *p19_cands_us]
            if p18_cands_us and (not p18_col or p18_col == fam):
                cands.extend(p18_cands_us)
            if fam == "allreduce" and oneshot_us > 0.0:
                cands.append(oneshot_us)
            floor_us = min(cands)
            if floor_us > 0.0:
                floors_ms[fam] = floor_us / _US_PER_MS
        if not floors_ms:
            return {}, None, None
        from ..program import RecipeBook
        return floors_ms, RecipeBook.from_mapping(kernels), topo

    @staticmethod
    def _hardware_floor_caps(eff_case: CaseSpec, eval_ctx: Any = None, *, dies: Sequence[Any] | None = None):
        dies = tuple(dies) if dies is not None else (getattr(getattr(eff_case, "machine", None), "dies", ()) or ())
        pt = getattr(eval_ctx, "primitive_table", None) if eval_ctx is not None else None
        datapath_fn = overlay_fn = None
        if pt is not None:
            from ..execution.context import datapath_for_devices as datapath_fn
            from mindie_sim.primitives.loader import resolve_sim_rate_overlay as overlay_fn
        max_hbm_ms = 0.0
        max_vec_ms = 0.0
        max_l2_bytes = 0.0
        cube_peaks_ms: dict[str, float] = {}
        custom_inv = bool((getattr(eff_case.runtime, "kernels", None) or {}).get("invocations"))
        invs_per_die: list[dict[str, float]] = []
        links = getattr(getattr(eff_case, "machine", None), "links", ()) or ()
        tchannels = getattr(getattr(eff_case, "machine", None), "transfer_channels", ()) or ()
        for die in dies:
            rates = dict(getattr(die, "rates", ()) or ())
            peaks = dict(die.compute_peaks)
            hbm_s = float(rates.get("hbm", 0.0) or 0.0)
            vec_s = float(rates.get("vector", 0.0) or 0.0)
            dp = getattr(die, "datapath", None)
            hbm_r = float(getattr(dp, "hbm_read_bytes_per_s", 0.0) or 0.0) if dp is not None else 0.0
            hbm_w = float(getattr(dp, "hbm_write_bytes_per_s", 0.0) or 0.0) if dp is not None else 0.0
            if pt is not None and datapath_fn is not None and overlay_fn is not None:
                _, dp_eff = datapath_fn((die,), links=links, transfer_channels=tchannels)
                ov = overlay_fn(pt, dp_eff, sku=die.sku)
                ov_rates = dict(ov.rates)
                if "hbm" in ov_rates:
                    hbm_s = min(float(ov_rates["hbm"]), hbm_s) if hbm_s > 0.0 else float(ov_rates["hbm"])
                if "vector" in ov_rates:
                    vec_s = min(float(ov_rates["vector"]), vec_s) if vec_s > 0.0 else float(ov_rates["vector"])
                if ov.hbm_read_bytes_per_s is not None:
                    raw_h = float(rates.get("hbm", 0.0) or 0.0)
                    hbm_r = min(float(ov.hbm_read_bytes_per_s), raw_h) if raw_h > 0.0 else float(ov.hbm_read_bytes_per_s)
                if ov.hbm_write_bytes_per_s is not None:
                    raw_h = float(rates.get("hbm", 0.0) or 0.0)
                    hbm_w = min(float(ov.hbm_write_bytes_per_s), raw_h) if raw_h > 0.0 else float(ov.hbm_write_bytes_per_s)
                peaks.update(dict(ov.compute_peaks))
            for fmt, val in peaks.items():
                f_ms = float(val or 0.0) / 1000.0
                if f_ms > cube_peaks_ms.get(str(fmt).lower(), 0.0):
                    cube_peaks_ms[str(fmt).lower()] = f_ms
            max_vec_ms = max(max_vec_ms, vec_s / 1000.0)
            max_hbm_ms = max(max_hbm_ms, max(hbm_s, hbm_r, hbm_w) / 1000.0)
            max_l2_bytes = max(max_l2_bytes, float(dict(getattr(die, "buffers", ()) or ()).get("l2", 0.0) or 0.0))
            if not custom_inv:
                invs_per_die.append({
                    str(k): float(getattr(v, "minimum_ms", 0.0) or 0.0)
                    for k, v in dict(getattr(die, "kernel_invocations", ()) or ()).items()
                })
        min_inv_by_class: dict[str, float] = {}
        if not custom_inv and invs_per_die and len(invs_per_die) == len(dies):
            shared_classes = set.intersection(*(set(m) for m in invs_per_die))
            for cls in shared_classes:
                min_inv_by_class[cls] = min(m[cls] for m in invs_per_die)
        per_rank_egress: dict[int, float] = {}
        min_link_lat_ms = math.inf
        has_pos_cap = any(float(getattr(item, "capacity_bytes_per_ms", 0.0) or 0.0) > 0.0 for item in links)
        for lk in links:
            cap = float(getattr(lk, "capacity_bytes_per_ms", 0.0) or 0.0)
            lat_us = float(getattr(lk, "latency_us", 0.0) or 0.0)
            if cap > 0.0 or not has_pos_cap:
                min_link_lat_ms = min(min_link_lat_ms, max(0.0, lat_us) / 1000.0)
            members = tuple(getattr(lk, "members", ()) or ())
            sem = str(getattr(lk, "semantics", "") or "").lower()
            mult = max(1, len(members) - 1) if (sem == "per_pair" or not sem) and len(members) > 1 else 1
            for r in (members or (0,)):
                per_rank_egress[int(r)] = per_rank_egress.get(int(r), 0.0) + cap * float(mult)
        max_link_ms = max(per_rank_egress.values(), default=0.0)
        hop_lat_ms = 0.0 if not math.isfinite(min_link_lat_ms) else min_link_lat_ms
        coll_floor_info = _SLASizingRun._collective_band_floor_info(eff_case, eval_ctx, links)
        return (
            max_hbm_ms, max_vec_ms, cube_peaks_ms, max(cube_peaks_ms.values(), default=0.0),
            max_link_ms, hop_lat_ms, max_l2_bytes, min_inv_by_class, links, coll_floor_info,
        )

    @staticmethod
    def _op_floor_contribution(
        op: Any,
        *,
        b: int,
        inv_b: float,
        q_tokens: int,
        graph_sp_padded_rows: int,
        phase: str,
        context: int,
        runtime_shared_prefix: bool,
        caps: tuple[Any, ...],
        attn_fams: frozenset[str],
        non_kernel_fams: frozenset[str],
        stage_wb: dict[int, float],
        stage_gg: dict[int, float],
        stage_kv: dict[int, float],
        stage_cube: dict[int, float],
        stage_vec: dict[int, float],
        stage_link: dict[int, float],
    ) -> tuple[float, float, bool]:
        """Accumulate one op's stage work floors and return ``(a_op, c_op, used_gg)``."""
        (
            max_hbm_ms, max_vec_ms, cube_peaks_ms, max_cube_ms,
            max_link_ms, hop_lat_ms, _max_l2_bytes, min_inv_by_class, _links, coll_floor_info,
        ) = caps
        coll_band_floor_ms, floor_recipes, floor_topo = coll_floor_info
        min_cube_inv = min_inv_by_class.get("cube", 0.0)
        min_vec_inv = min_inv_by_class.get("vector", 0.0)
        st = int(getattr(op, "stage", 0) or 0)
        sem = getattr(op, "semantic_shape", ()) or ()
        loc = getattr(op, "local_shape", ()) or ()
        sem_scale = (
            min(1.0, float(sem[0]) / float(loc[0]))
            if (sem and loc and float(loc[0]) > 0.0 and float(sem[0]) > 0.0)
            else 1.0
        )
        md = dict(getattr(op, "metadata", ()) or ())
        fam = str(getattr(op, "family", "") or "")
        active_rows = int(md.get("graph_active_query_rows") or (int(b) * max(1, int(q_tokens))))
        tokens_global = int(md.get("tokens_global") or 0)
        phys_rows = max(
            int(md.get("graph_physical_query_rows") or 0),
            tokens_global,
            int(graph_sp_padded_rows),
        )
        if active_rows > 0 and phys_rows > active_rows:
            # Unpadded multi-dim ops (attention, logits) keep leading dim <= active_rows
            # and tokens_global <= active_rows; captured and SP-padded ops carry padded rows.
            op_unpadded = (
                int(md.get("graph_physical_query_rows") or 0) <= active_rows
                and tokens_global <= active_rows
                if fam == "grouped_gemm"
                else (
                    len(sem) >= _MIN_ROW_LEADING_DIMS
                    and float(sem[0]) <= float(active_rows)
                    and (not loc or float(loc[0]) <= float(active_rows))
                    and tokens_global <= active_rows
                )
            )
            if not op_unpadded:
                sem_scale = min(sem_scale, float(active_rows) / float(phys_rows))
        if (active_rows > 0 and fam != "grouped_gemm"
                and len(sem) >= _MIN_ROW_LEADING_DIMS and float(sem[0]) > float(active_rows)):
            sem_scale = min(sem_scale, float(active_rows) / float(sem[0]))
        op_hbm = float(getattr(op, "hbm_bytes", 0.0) or 0.0)
        wb = float(md.get("packed_weight_bytes") or md.get("weight_bytes") or 0.0)
        if op_hbm > 0.0 and wb <= 0.0 and fam == "native_fusion":
            phases = md.get("fusion_phases") or ()
            if isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)):
                wb = sum(
                    float(
                        ph.get("packed_weight_bytes") or ph.get("weight_bytes")
                        or dict(ph.get("metadata") or ()).get("packed_weight_bytes")
                        or dict(ph.get("metadata") or ()).get("weight_bytes")
                        or 0.0
                    )
                    for ph in phases if isinstance(ph, Mapping)
                )
        a_op, c_op, used_gg = 0.0, 0.0, False
        if op_hbm > 0.0 and wb > 0.0:
            w_eff = min(wb, op_hbm)
            if fam == "grouped_gemm" or md.get("active_experts") is not None:
                w_eff *= sem_scale
            stage_wb[st] = stage_wb.get(st, 0.0) + w_eff
            if max_hbm_ms > 0.0:
                a_op = w_eff / max_hbm_ms
        elif op_hbm > 0.0 and wb <= 0.0 and fam == "grouped_gemm":
            gg_eff = op_hbm * sem_scale
            stage_gg[st] = stage_gg.get(st, 0.0) + gg_eff
            if max_hbm_ms > 0.0:
                a_op = gg_eff / max_hbm_ms
                used_gg = True
        elif (op_hbm > 0.0 and wb <= 0.0 and phase == "decode"
              and context > 0 and fam in attn_fams):
            kv_b = min(float(md.get("kv_read_bytes") or op_hbm), op_hbm) * sem_scale
            op_shared_prefix = (
                runtime_shared_prefix
                or str(md.get("prefix_cache_mode", "") or "").strip().lower() == "shared"
            )
            if op_shared_prefix:
                stage_wb[st] = stage_wb.get(st, 0.0) + kv_b
                if max_hbm_ms > 0.0:
                    a_op = kv_b / max_hbm_ms
            else:
                stage_kv[st] = stage_kv.get(st, 0.0) + kv_b
                if max_hbm_ms > 0.0:
                    c_op = (kv_b / max_hbm_ms) * inv_b
        mflops = getattr(op, "matrix_flops", ()) or ()
        c_ms = 0.0
        if mflops and max_cube_ms > 0.0:
            c_ms = sum(
                float(fl or 0.0) * sem_scale / cube_peaks_ms.get(str(fmt).lower(), max_cube_ms)
                for fmt, fl in mflops
                if float(fl or 0.0) > 0.0
            )
        else:
            fl = float(getattr(op, "flops", 0.0) or 0.0)
            if fl > 0.0 and max_cube_ms > 0.0:
                fmt = str(md.get("compute_format") or getattr(op, "dtype", "") or "").lower()
                c_ms = fl * sem_scale / cube_peaks_ms.get(fmt, max_cube_ms)
        if c_ms > 0.0:
            stage_cube[st] = stage_cube.get(st, 0.0) + c_ms
            if c_ms > a_op + c_op * float(b):
                a_op, c_op = 0.0, c_ms * inv_b
        vops = float(getattr(op, "vector_ops", 0.0) or 0.0)
        if vops > 0.0 and max_vec_ms > 0.0:
            stage_vec[st] = stage_vec.get(st, 0.0) + (vops * sem_scale / max_vec_ms)
        lbytes = float(getattr(op, "link_bytes", 0.0) or 0.0)
        if lbytes > 0.0 and max_link_ms > 0.0:
            # Full-duplex compulsory send volume across all incident peer links of a die.
            compulsory_link_ms = (0.5 * lbytes * sem_scale) / max_link_ms
            stage_link[st] = stage_link.get(st, 0.0) + compulsory_link_ms
            coll_fixed_ms = hop_lat_ms
            band_ms = coll_band_floor_ms.get(fam, 0.0)
            if (band_ms > coll_fixed_ms and floor_recipes is not None and floor_topo is not None
                    and not md.get("collective_startup_mode") and not md.get("megamoe_geometry")):
                from ..collectives import plan_options as collective_options
                rec = floor_recipes.select(op)
                exp_algo = getattr(rec, "algorithm", None)
                is_band_rec = (
                    exp_algo in _BAND_RING_ALGOS
                    or (exp_algo is None and bool(getattr(rec, "mesh_capable", False))
                        and not floor_topo.declares_direct_mesh(op, 0))
                )
                if is_band_rec and not collective_options(rec, op):
                    coll_fixed_ms = band_ms
            if coll_fixed_ms + compulsory_link_ms > a_op + c_op * float(b):
                a_op, c_op = coll_fixed_ms, compulsory_link_ms * inv_b
        if (lbytes <= 0.0 and fam not in non_kernel_fams
                and getattr(op, "coverage", "") != "pool_release"
                and getattr(op, "source_boundary", "") != "host"
                and md.get("service_boundary", "device_kernel") == "device_kernel"):
            pe = md.get("program_execution")
            exp_cls = md.get("invocation_class") or (pe.get("execution_class") if isinstance(pe, Mapping) else None)
            if exp_cls and (c_ms > 0.0 or vops > 0.0 or op_hbm > 0.0 or getattr(op, "source_boundary", "") == "kernel"):
                inv_min = min_inv_by_class.get(str(exp_cls), 0.0)
            elif not exp_cls:
                inv_min = max(min_cube_inv if c_ms > 0.0 else 0.0, min_vec_inv if vops > 0.0 else 0.0)
            else:
                inv_min = 0.0
            if inv_min > a_op + c_op * float(b):
                a_op, c_op = inv_min, 0.0
        return a_op, c_op, used_gg

    def _work_floor_rays(
        self, s_key: tuple[float, ...], b: int, step_ms: float,
    ) -> tuple[tuple[float, float], ...]:
        """Compute monotone affine lower-bound rays ``(fixed_ms, linear_per_seq_ms)`` for ``b' >= b``."""
        if self.unchunkable or len(self._chunks_for(b)) > 1 or step_ms <= 0.0 or b < 1:
            return ()
        cached = self.graph_cache.get((s_key, b, self.fast_spans)) or self.graph_cache.get((s_key, b, False))
        if cached is None:
            return ()
        graphs, eff_case, _gr = cached
        ops_0 = next(iter(graphs.values()), ()) if isinstance(graphs, Mapping) else ()
        all_dies = getattr(getattr(eff_case, "machine", None), "dies", ()) or ()
        if not ops_0 or not all_dies:
            return ()
        if len(self.eval_cohorts) > 1 and isinstance(graphs, Mapping):
            cohort_group = tuple(sorted(int(r) for r in graphs.keys()))
            placement = eff_case.machine.validate_placement(eff_case.runtime.parallel)
            die_ranks = placement.physical_ranks_for_cohort(cohort_group)
            dies = tuple(placement.die_for_rank(rk) for rk in die_ranks) or all_dies
            caps_key: tuple[Any, ...] = (s_key, tuple(int(rk) for rk in die_ranks))
        elif len(all_dies) > eff_case.runtime.parallel.world_size:
            placement = eff_case.machine.validate_placement(eff_case.runtime.parallel)
            die_ranks = placement.active_ranks
            dies = tuple(placement.die_for_rank(rk) for rk in die_ranks) or all_dies
            caps_key = (s_key, tuple(int(rk) for rk in die_ranks))
        else:
            dies = all_dies
            caps_key = (s_key, None)
        caps = self.floor_caps_cache.get(caps_key)
        if caps is None:
            caps = self._hardware_floor_caps(eff_case, self.get_eval_context(s_key), dies=dies)
            self.floor_caps_cache[caps_key] = caps
        max_hbm_ms = caps[0]
        max_l2_bytes = caps[6]
        stage_wb: dict[int, float] = {}
        stage_gg: dict[int, float] = {}
        stage_kv: dict[int, float] = {}
        stage_cube: dict[int, float] = {}
        stage_vec: dict[int, float] = {}
        stage_link: dict[int, float] = {}
        dp_val: dict[str, float] = {}
        dp_ray: dict[str, tuple[float, float]] = {}
        dag_used_gg = False
        attn_fams = frozenset({"attention", "mla_attention", "indexed_attention", "paged_mla_attention"})
        non_kernel_fams = frozenset({"collective", "barrier", "token_commit", "prefetch", "pool_release"})
        runtime_shared_prefix = (
            str(getattr(eff_case.runtime, "prefix_cache_mode", "") or "").strip().lower() == "shared"
        )
        inv_b = 1.0 / float(b)
        graph_sp_padded_rows = max(
            (int(dict(getattr(op, "metadata", ()) or ()).get("tokens_global") or 0)
             for op in ops_0),
            default=0,
        )
        for op in ops_0:
            a_op, c_op, used_gg = self._op_floor_contribution(
                op, b=b, inv_b=inv_b, q_tokens=self.q_tokens,
                graph_sp_padded_rows=graph_sp_padded_rows,
                phase=self.phase, context=self.context,
                runtime_shared_prefix=runtime_shared_prefix, caps=caps,
                attn_fams=attn_fams, non_kernel_fams=non_kernel_fams,
                stage_wb=stage_wb, stage_gg=stage_gg, stage_kv=stage_kv,
                stage_cube=stage_cube, stage_vec=stage_vec, stage_link=stage_link,
            )
            if used_gg:
                dag_used_gg = True
            best_v, best_r = 0.0, (0.0, 0.0)
            for dep in getattr(op, "deps", ()) or ():
                dv = dp_val.get(dep, 0.0)
                if dv > best_v:
                    best_v, best_r = dv, dp_ray[dep]
            cur_r = (best_r[0] + a_op, best_r[1] + c_op)
            dp_ray[op.op_id] = cur_r
            dp_val[op.op_id] = cur_r[0] + cur_r[1] * float(b)
        rays: list[tuple[float, float]] = []
        if max_hbm_ms > 0.0:
            for st in set(stage_wb) | set(stage_gg) | set(stage_kv):
                gg_b = stage_gg.get(st, 0.0)
                fixed_b = stage_wb.get(st, 0.0) + max(0.0, gg_b - max_l2_bytes) if gg_b > 0.0 else stage_wb.get(st, 0.0)
                fixed = fixed_b / max_hbm_ms
                lin = (stage_kv.get(st, 0.0) / max_hbm_ms) * inv_b
                if fixed > 0.0 or lin > 0.0:
                    rays.append((fixed, lin))
        for stage_map in (stage_cube, stage_vec, stage_link):
            top = max(stage_map.values(), default=0.0)
            if top > 0.0:
                rays.append((0.0, top * inv_b))
        if dp_val:
            top_id = max(dp_val, key=dp_val.__getitem__)
            dag_a, dag_c = dp_ray[top_id]
            if dag_used_gg and max_hbm_ms > 0.0:
                dag_a = max(0.0, dag_a - max_l2_bytes / max_hbm_ms)
            if dag_a > 0.0 or dag_c > 0.0:
                rays.append((dag_a, dag_c))
        if not rays:
            return ()
        at_b = max(a + c * float(b) for a, c in rays)
        if not math.isfinite(at_b) or at_b <= 0.0 or at_b > step_ms * (1.0 + 1e-6):
            return ()
        return tuple(rays)

    @staticmethod
    def _eval_rays(rays: tuple[tuple[float, float], ...], b_prime: int) -> float:
        return max((a + c * float(b_prime) for a, c in rays), default=0.0)

    def _work_floor_ms(
        self, s_key: tuple[float, ...], b: int, step_ms: float,
    ) -> float:
        """Compute monotone physical lower bound on ``step_ms(b')`` for all ``b' >= b``."""
        rays = self._work_floor_rays(s_key, b, step_ms)
        return min(self._eval_rays(rays, b), step_ms) if rays else 0.0

    def _ray_sla_cutoff(self, rays: tuple[tuple[float, float], ...], b_eval: int, hi_cut: int) -> int:
        """Return tightest integer ``b' <= hi_cut`` not certified SLA-infeasible by ``rays``."""
        sla_ms = float(self.sla_ms)
        cut = int(hi_cut)
        domain_floor = max(0, int(b_eval) - 1)
        for fixed, lin in rays:
            if fixed > sla_ms:
                return min(cut, domain_floor)
            if lin > 0.0:
                max_b = max(domain_floor, int(math.floor((sla_ms - fixed) / lin + 1e-12)))
                if max_b < cut:
                    cut = max_b
        return cut

    def _certified_upper_cutoff(
        self, eval_b, floors: dict[int, float], rays_map: dict[int, tuple[tuple[float, float], ...]],
        eff_max: int, b_joint: int,
    ) -> int:
        """Theorem 1: find largest ``b <= eff_max`` not certified SLA-infeasible by ``work_floor``."""
        sla_ms = float(self.sla_ms)
        hi_cut = self._ray_sla_cutoff(rays_map.get(1, ()), 1, int(eff_max))
        if floors[1] > sla_ms:
            return 0
        prev_p = 1
        probe = 2
        anchor_hi = min(hi_cut, int(b_joint)) if b_joint >= 2 else 0
        if anchor_hi >= 2:
            _, fl_a = eval_b(anchor_hi)
            hi_cut = self._ray_sla_cutoff(rays_map.get(anchor_hi, ()), anchor_hi, hi_cut)
            if fl_a <= sla_ms:
                if anchor_hi >= hi_cut:
                    return hi_cut
                prev_p = anchor_hi
                probe = anchor_hi * 2
            else:
                lo_c, hi_c = 1, anchor_hi
                while hi_c - lo_c > 1:
                    mid_c = (lo_c + hi_c) // 2
                    _, fl_m = eval_b(mid_c)
                    hi_cut = self._ray_sla_cutoff(rays_map.get(mid_c, ()), mid_c, hi_cut)
                    if fl_m > sla_ms:
                        hi_c = mid_c
                    else:
                        lo_c = mid_c
                return min(hi_cut, lo_c)
        while probe <= hi_cut:
            _, fl_p = eval_b(probe)
            hi_cut = self._ray_sla_cutoff(rays_map.get(probe, ()), probe, hi_cut)
            if fl_p > sla_ms:
                lo_c, hi_c = prev_p, probe
                while hi_c - lo_c > 1:
                    mid_c = (lo_c + hi_c) // 2
                    _, fl_m = eval_b(mid_c)
                    hi_cut = self._ray_sla_cutoff(rays_map.get(mid_c, ()), mid_c, hi_cut)
                    if fl_m > sla_ms:
                        hi_c = mid_c
                    else:
                        lo_c = mid_c
                hi_cut = min(hi_cut, lo_c)
                break
            prev_p = probe
            probe *= 2
        if hi_cut >= 1 and hi_cut not in floors:
            probe_hi = hi_cut
            _, fl_hi = eval_b(probe_hi)
            hi_cut = self._ray_sla_cutoff(rays_map.get(probe_hi, ()), probe_hi, hi_cut)
            if fl_hi > sla_ms:
                hi_cut = min(hi_cut, max(0, probe_hi - 1))
        return hi_cut

    def _search_concurrency(
        self, s_key, b1_ms, eff_max, bucket_keys, candidate_checks, evaluated_b,
        b_joint, candidate_windows,
    ):
        """Dual-certificate branch-and-bound SLA sizing maximizing throughput ``B / step_ms``."""
        if self.unchunkable or eff_max < 1:
            return 0, self.eval_at(s_key, 1, full=True), [], None, True, 0

        sla_ms = float(self.sla_ms)
        bounded_probes: list[int] = []
        probed_seen: set[int] = set()
        floors: dict[int, float] = {}
        rays_map: dict[int, tuple[tuple[float, float], ...]] = {}
        best_sla: tuple[float, int] | None = None
        best_joint: tuple[float, int] | None = None

        def _tps(b_val: int, step_val: float) -> float:
            return (float(b_val) / step_val) if step_val > 0.0 else math.inf

        def _record_best(b_val: int) -> None:
            nonlocal best_sla, best_joint
            rec = candidate_checks.get(b_val)
            if rec is None:
                return
            score = (_tps(b_val, float(rec["step_ms"])), b_val)
            if bool(rec["sla"]["pass"]) and (best_sla is None or score > best_sla):
                best_sla = score
            if bool(rec["feasible"]) and (best_joint is None or score > best_joint):
                best_joint = score

        def _eval_b(b_val: int) -> tuple[float, float]:
            check_active_deadline()
            b_int = int(b_val)
            if b_int not in candidate_checks:
                self.try_step_ok(
                    s_key, b_int, window_for=candidate_windows,
                    seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint,
                )
            if b_int > 1 and b_int not in probed_seen:
                probed_seen.add(b_int)
                bounded_probes.append(b_int)
            step_val = float(candidate_checks[b_int]["step_ms"])
            if b_int not in floors:
                rays = self._work_floor_rays(s_key, b_int, step_val)
                rays_map[b_int] = rays
                floors[b_int] = min(self._eval_rays(rays, b_int), step_val) if rays else 0.0
            _record_best(b_int)
            return step_val, floors[b_int]

        rays_1 = self._work_floor_rays(s_key, 1, float(b1_ms))
        rays_map[1] = rays_1
        floors[1] = min(self._eval_rays(rays_1, 1), float(b1_ms)) if rays_1 else 0.0
        _record_best(1)
        hi_cut = self._certified_upper_cutoff(
            _eval_b, floors, rays_map, eff_max, b_joint,
        )
        q_tokens = max(1, int(self.q_tokens))
        split_anchors = sorted({
            int(raw_b)
            for bk in bucket_keys
            for raw_b in (bk, bk // q_tokens)
            if 2 <= raw_b < hi_cut
        } | ({int(self.base_b_hint)} if 2 <= self.base_b_hint < hi_cut else set()))

        def _interval_ub(lo_i: int, hi_i: int) -> float:
            b_top = hi_i - 1
            fl_top = floors.get(lo_i, 0.0)
            for b_e, rays_e in rays_map.items():
                if b_e <= lo_i and rays_e:
                    val_e = self._eval_rays(rays_e, b_top)
                    if val_e > fl_top:
                        fl_top = val_e
            return math.inf if fl_top <= 0.0 else float(b_top) / fl_top

        def _is_dominated(lo_i: int, hi_i: int) -> bool:
            if hi_i - lo_i <= 1:
                return True
            fl_lo = floors.get(lo_i, 0.0)
            if fl_lo > sla_ms:
                return True
            if fl_lo <= 0.0:
                return False
            ub_all = _interval_ub(lo_i, hi_i)
            sla_dom = best_sla is not None and (
                ub_all < best_sla[0]
                or (ub_all <= best_sla[0] and (hi_i - 1) <= best_sla[1])
            )
            if not sla_dom:
                return False
            if b_joint < 1 or lo_i >= b_joint:
                return True
            joint_hi = min(hi_i, b_joint + 1)
            if joint_hi - lo_i <= 1:
                return True
            ub_joint = _interval_ub(lo_i, joint_hi)
            return best_joint is not None and (
                ub_joint < best_joint[0]
                or (ub_joint <= best_joint[0] and (joint_hi - 1) <= best_joint[1])
            )

        heap: list[tuple[float, int, int, int]] = []
        if hi_cut >= 2:
            if hi_cut not in candidate_checks:
                _eval_b(hi_cut)
            pts = sorted(b_i for b_i in candidate_checks if 1 <= b_i <= hi_cut)
            for idx in range(len(pts) - 1):
                lo_i, hi_i = pts[idx], pts[idx + 1]
                if not _is_dominated(lo_i, hi_i):
                    heapq.heappush(heap, (-_interval_ub(lo_i, hi_i), -(hi_i - lo_i), lo_i, hi_i))

        while heap:
            _neg_ub, _neg_span, lo_i, hi_i = heapq.heappop(heap)
            hi_eff = min(hi_i, hi_cut + 1)
            if lo_i >= hi_cut or _is_dominated(lo_i, hi_eff):
                continue
            mid = (lo_i + hi_eff) // 2
            inner_anchors = [a for a in split_anchors if lo_i < a < hi_eff]
            if inner_anchors:
                mid = min(inner_anchors, key=lambda a: (abs(a - mid), a))
            _, fl_m = _eval_b(mid)
            hi_cut = self._ray_sla_cutoff(rays_map.get(mid, ()), mid, hi_cut)
            if fl_m > sla_ms:
                hi_cut = min(hi_cut, mid - 1)
                if not _is_dominated(lo_i, mid):
                    heapq.heappush(heap, (-_interval_ub(lo_i, mid), -(mid - lo_i), lo_i, mid))
                continue
            if not _is_dominated(lo_i, mid):
                heapq.heappush(heap, (-_interval_ub(lo_i, mid), -(mid - lo_i), lo_i, mid))
            hi_next = min(hi_eff, hi_cut + 1)
            if mid < hi_next and not _is_dominated(mid, hi_next):
                heapq.heappush(heap, (-_interval_ub(mid, hi_next), -(hi_next - mid), mid, hi_next))

        b_sla = int(best_sla[1]) if best_sla is not None else 0
        eval_sla = self.eval_at(s_key, b_sla if b_sla > 0 else 1, full=True)
        return b_sla, eval_sla, bounded_probes, None, True, hi_cut

    def _candidate_record(
        self, b, evaluation, *, scen, s_key, b_joint, b_vram, b_lifecycle,
        life_ctx, verify_tokens, candidate_windows,
    ):
        phase, context, scope = self.phase, self.context, self.scope
        q_tokens, emit_tokens = self.q_tokens, self.emit_tokens
        output_budget, decode_emit_tokens = self.output_budget, self.decode_emit_tokens
        sla_ms, unchunkable = self.sla_ms, self.unchunkable
        sla_pass = float(evaluation["step_ms"]) <= sla_ms
        current_window, lifecycle_window = candidate_windows(int(b))
        capacity_pass = bool(
            b <= b_joint and current_window["fits"]
            and lifecycle_window["fits"] and not unchunkable)
        return {
            "b": int(b), "step_ms": float(evaluation["step_ms"]),
            "step_basis": evaluation.get("step_basis", "graph_evaluation"),
            "input_boundary": {"phase": phase, "tokens": q_tokens, "context": context,
                               "sequences": int(b), "scope": scope,
                               "output_budget": output_budget,
                               "emitted_tokens_per_seq": emit_tokens,
                               "prefill_tokens_per_seq": q_tokens if phase == "prefill" else None,
                               "decode_context_tokens": life_ctx,
                               "decode_query_tokens": verify_tokens,
                               "decode_emitted_tokens_per_seq": (
                                   decode_emit_tokens if phase == "prefill" else emit_tokens)},
            "hardware": {"scenario_id": scen.get("id", "custom"),
                         "scales": dict(zip(_SCALE_KEYS, s_key))},
            "plan": {"kind": "target_graphs", "selection": "case_lowering"},
            "sla": {"limit_ms": float(sla_ms), "pass": bool(sla_pass)},
            "capacity": {"b_vram_max": int(b_joint), "b_vram_current": int(b_vram),
                         "b_vram_lifecycle": int(b_lifecycle), "pass": bool(capacity_pass),
                         "supply_bytes": current_window["supply_bytes"],
                         "active_kv_bytes": current_window["active_kv_bytes"],
                         "retiring_kv_bytes": current_window["retiring_kv_bytes"],
                         "workspace_bytes": current_window["workspace_bytes"],
                         "peak_bytes": current_window["peak_bytes"],
                         "headroom_bytes": current_window["headroom_bytes"],
                         "lifecycle_headroom_bytes": lifecycle_window["headroom_bytes"]},
            "service_lifecycle": {
                "phase": "decode", "context_tokens": life_ctx,
                "query_tokens_per_sequence": verify_tokens,
                "emitted_tokens_per_sequence": (
                    decode_emit_tokens if phase == "prefill" else emit_tokens),
                "capacity": {
                    "supply_bytes": lifecycle_window["supply_bytes"],
                    "active_kv_bytes": lifecycle_window["active_kv_bytes"],
                    "retiring_kv_bytes": lifecycle_window["retiring_kv_bytes"],
                    "workspace_bytes": lifecycle_window["workspace_bytes"],
                    "peak_bytes": lifecycle_window["peak_bytes"],
                    "headroom_bytes": lifecycle_window["headroom_bytes"],
                    "fits": lifecycle_window["fits"],
                },
            },
            "feasible": bool(sla_pass and capacity_pass),
            "source": "full_graph_reevaluation",
            "network_mbu": evaluation.get("network_mbu"), "network_cube_mfu": evaluation.get("network_cube_mfu"),
            "network_vec_mfu": evaluation.get("network_vec_mfu"), "network_link_util": evaluation.get("network_link_util"),
            "network_roofline_efficiency": evaluation.get("network_roofline_efficiency"),
            "bottleneck_shares": dict(evaluation.get("bottleneck_shares") or {}),
        }

    def _scenario_result(
        self, scen, s_key, current_caps, lifecycle_caps, b_joint,
        life_ctx, verify_tokens, candidate_windows, b1_ms, bucket_keys,
        candidate_checks, evaluated_b, b_sla, eval_sla, bounded_probes,
        secant_estimate, *, certified_optimal: bool = False, hi_cut: int | None = None,
    ):
        b_vram, b_vram_sync, b_vram_async = current_caps
        b_lifecycle, b_life_sync, b_life_async = lifecycle_caps
        b_max, domain_cap, domain_reason = self.b_max, self.domain_cap, self.domain_reason
        sla_ms, dp, phase = self.sla_ms, self.dp, self.phase
        q_tokens, emit_tokens = self.q_tokens, self.emit_tokens
        decode_emit_tokens = self.decode_emit_tokens
        eval_at = self.eval_at
        candidate_record = partial(
            self._candidate_record, scen=scen, s_key=s_key, b_joint=b_joint,
            b_vram=b_vram, b_lifecycle=b_lifecycle, life_ctx=life_ctx,
            verify_tokens=verify_tokens, candidate_windows=candidate_windows)
        # Probe the capacity ceiling and the latency best as actual candidates even
        # when neither was reached by the binary path; all checks share this set.
        for probe in sorted({b_vram, b_sla}):
            if 0 < probe <= min(b_max, domain_cap) and probe not in candidate_checks:
                step = float(eval_at(s_key, probe, full=False)["step_ms"])
                candidate_checks[probe] = {
                    "b": int(probe), "step_ms": step,
                    "sla": {"limit_ms": float(sla_ms), "pass": step <= sla_ms},
                    "capacity": {"b_vram_max": int(b_joint), "b_vram_current": int(b_vram),
                                 "b_vram_lifecycle": int(b_lifecycle), "pass": probe <= b_joint},
                    "feasible": bool(step <= sla_ms and probe <= b_joint)}
                evaluated_b.add(probe)
        latency_candidate = candidate_record(b_sla, eval_sla) if b_sla > 0 else None
        feasible_check = select_jointly_feasible_candidate(list(candidate_checks.values()))
        b_feas = int(feasible_check["b"]) if feasible_check is not None else 0
        feasible_ev = eval_at(s_key, b_feas, full=True) if b_feas > 0 else None
        feasible_candidate = (candidate_record(b_feas, feasible_ev)
                              if feasible_ev is not None else None)
        if feasible_candidate is not None and not feasible_candidate["feasible"]:
            feasible_candidate = None
            b_feas = 0
        capacity_feasible = b_joint >= 1
        latency_sla_feasible = bool(latency_candidate is not None and latency_candidate["sla"]["pass"])
        jointly_feasible = feasible_candidate is not None
        exhaustive = set(range(1, min(b_max, domain_cap) + 1)) <= set(candidate_checks)
        domain_certified = bool(exhaustive or certified_optimal)
        if domain_cap < 1:
            feasibility_status = "domain_infeasible"
        elif not capacity_feasible:
            feasibility_status = "capacity_infeasible"
        elif jointly_feasible:
            feasibility_status = "feasible" if b_sla <= b_feas else "feasible_capacity_limited"
        elif latency_candidate is not None:
            feasibility_status = ("no_jointly_feasible_candidate_in_evaluated_domain" if domain_certified
                                  else "no_feasible_candidate_in_bounded_search")
        else:
            feasibility_status = ("sla_infeasible_within_evaluated_domain" if domain_certified
                                  else "no_feasible_candidate_in_bounded_search")
        tps_sla = (b_sla * dp * emit_tokens * 1000.0) / eval_sla["step_ms"] if b_sla > 0 else 0.0
        tps_feas = (b_feas * dp * emit_tokens * 1000.0) / feasible_ev["step_ms"] if jointly_feasible and feasible_ev is not None else 0.0
        step_basis = eval_sla.get("step_basis", "graph_evaluation") if b_sla > 0 else "graph_evaluation_at_b1_reference"
        output_network = feasible_ev if (jointly_feasible and feasible_ev is not None) else eval_sla
        output_network_source = "feasible_candidate" if jointly_feasible else "latency_candidate_not_feasible"
        selected_b = b_feas if b_feas > 0 else (b_sla if b_sla > 0 else 1)
        selected_current_window, selected_lifecycle_window = candidate_windows(selected_b)
        return {
            "scenario_id": scen.get("id", "custom"), "label": scen.get("label", scen.get("id", "custom")), "scales": dict(zip(_SCALE_KEYS, s_key)),
            "b_sla": b_sla, "b_vram_max": b_vram, "b_vram_max_joint": b_joint,
            "b_vram_max_lifecycle_sync": b_life_sync, "b_vram_max_lifecycle_async": b_life_async,
            "b_vram_max_sync": b_vram_sync, "b_vram_max_async": b_vram_async,
            "prefill_tokens_per_seq": q_tokens if phase == "prefill" else None,
            "decode_emitted_tokens_per_seq": decode_emit_tokens if phase == "prefill" else emit_tokens,
            "emitted_tokens_per_seq": emit_tokens,
            "capacity_window": selected_current_window,
            "lifecycle_capacity_window": selected_lifecycle_window,
            "joint_capacity": {
                "sequences": selected_b,
                "current_supply_bytes": selected_current_window["supply_bytes"],
                "current_active_kv_bytes": selected_current_window["active_kv_bytes"],
                "current_retiring_kv_bytes": selected_current_window["retiring_kv_bytes"],
                "current_workspace_bytes": selected_current_window["workspace_bytes"],
                "current_peak_bytes": selected_current_window["peak_bytes"],
                "current_headroom_bytes": selected_current_window["headroom_bytes"],
                "lifecycle_supply_bytes": selected_lifecycle_window["supply_bytes"],
                "lifecycle_active_kv_bytes": selected_lifecycle_window["active_kv_bytes"],
                "lifecycle_retiring_kv_bytes": selected_lifecycle_window["retiring_kv_bytes"],
                "lifecycle_workspace_bytes": selected_lifecycle_window["workspace_bytes"],
                "lifecycle_peak_bytes": selected_lifecycle_window["peak_bytes"],
                "lifecycle_headroom_bytes": selected_lifecycle_window["headroom_bytes"],
                "fits": bool(selected_current_window["fits"] and selected_lifecycle_window["fits"]),
            },
            "delta_b_vram_from_disabling_async": max(0, b_vram_sync - b_vram_async), "hbm_bound": bool(b_joint < b_sla),
            "b_feasible": b_feas, "domain_max_b": domain_cap, "domain_limit_reason": domain_reason,
            "sla_feasible": jointly_feasible, "latency_sla_feasible": latency_sla_feasible,
            "capacity_feasible": capacity_feasible, "jointly_feasible": jointly_feasible, "feasible": jointly_feasible,
            "feasibility_status": feasibility_status,
            "b_sla_lower_bound": b_sla, "b_sla_is_proven_max": False,
            "latency_candidate": latency_candidate, "feasible_candidate": feasible_candidate,
            "search": {"method": "dual_certificate_branch_and_bound",
                       "objective": "maximize_throughput_under_sla",
                       "secant_warmup_probe": secant_estimate,
                       "monotonicity_assumed": False,
                       "certified_optimal": bool(certified_optimal),
                       "certified_upper_cutoff_b": int(hi_cut if hi_cut is not None else min(b_max, domain_cap)),
                       "non_monotonic_dispatch_detected": bool(
                           b_sla > 1 and b1_ms > sla_ms),
                       "non_monotonic_dispatch_check": "dual_certificate_branch_and_bound",
                       "bounded_dispatch_probes": bounded_probes, "bounded_search_exhaustive": bool(domain_certified),
                       "evaluated_candidates": sorted(evaluated_b), "b_sla_is_proven_max": False,
                       "candidate_step_source": "full_graph_reevaluation" if jointly_feasible else "none",
                       "lower_bound_can_prove_feasibility": False},
            "total_concurrency": b_sla * dp, "total_feasible_concurrency": b_feas * dp,
            "step_basis": step_basis, "step_is_lower_bound": False,
            "step_ms_at_b1": b1_ms,
            "tps_at_b1": ((dp * emit_tokens * 1000.0) / b1_ms
                          if b1_ms > 0.0 else 0.0),
            "step_ms_at_sla": eval_sla["step_ms"], "tps_at_sla": tps_sla,
            "target_step_tps_at_sla": (b_sla * dp * q_tokens * 1000.0) / eval_sla["step_ms"] if b_sla > 0 else 0.0,
            "step_ms_feasible": feasible_candidate["step_ms"] if jointly_feasible and feasible_candidate is not None else None,
            "tps_feasible": tps_feas,
            "feasible_network_mbu": feasible_candidate["network_mbu"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_cube_mfu": feasible_candidate["network_cube_mfu"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_vec_mfu": feasible_candidate["network_vec_mfu"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_link_util": feasible_candidate["network_link_util"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_roofline_efficiency": feasible_candidate["network_roofline_efficiency"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_bottleneck_shares": feasible_candidate["bottleneck_shares"] if jointly_feasible and feasible_candidate is not None else None,
            "network_source": output_network_source,
            "network_mbu": output_network["network_mbu"], "network_mbu_at_sla": eval_sla["network_mbu"],
            "network_cube_mfu": output_network["network_cube_mfu"], "network_cube_mfu_at_sla": eval_sla["network_cube_mfu"],
            "network_vec_mfu": output_network["network_vec_mfu"], "network_vec_mfu_at_sla": eval_sla["network_vec_mfu"],
            "network_link_util": output_network["network_link_util"], "network_roofline_efficiency": output_network["network_roofline_efficiency"],
            "bottleneck_shares": output_network["bottleneck_shares"],
        }


solve_hardware_sla_throughput = solve_sla_sizing
