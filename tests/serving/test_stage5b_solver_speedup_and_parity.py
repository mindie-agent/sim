"""Stage 5B solver speedup, C11 epoch continuation, DP-cohort dedup, and parity locks.

[Role] Verify zero-drift equivalence for Stage 5B accelerations:
  (1) C11 ``_engine_accel`` (``mode="c"`` and ``mode="auto"``) persistent hybrid
      epoch continuation matches pure Python to ``<= 1e-12`` (0 ns drift) on
      coupled multi-unit Qwen3.8 and Kimi-K3.1 graphs with positive native steps;
  (2) Symmetric multi-DP EP cohort deduplication (``MINDIE_SIM_DP_DEDUP=1`` vs
      ``0``) matches full multi-replica expansion to ``<= 1e-9`` across
      symmetric GDN/MoE/KDA graphs and fails closed on cross-DP MegaMoE peer
      RX links;
  (3) Provable decode per-step KV/recurrent state HBM read lower bounds tighten
      ``_provable_decode_tps_upper_bound`` while remaining strictly sound
      (``ub_tps >= actual_tps``);
  (4) Parallel static admission, parallel multi-probe ``pd_decode`` waves, and
      unified bottleneck/waterfall evaluation match ``granted_workers=1``
      bit-for-bit under ``granted_workers=4`` with zero zombie workers.
[Invariants] Uses ``monkeypatch`` exclusively (zero host env or filesystem
  mutation) and leaves zero active child processes on exit.
[Timeouts & Locks] Bounded small-layer graphs and small-device sweeps; no I/O.
[Side-Effect Policy] Read-only over repository files; process-local env toggles
  only via ``monkeypatch``.
"""
from __future__ import annotations

import math
import multiprocessing

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import _lower_and_run_graph
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.runtime.dp_cohort_dedup import (
    _has_megamoe_cross_dp_group,
    can_dedup_collective_to_canonical,
    evaluate_dp_cohort_dedup,
    is_symmetric_dp_cohort,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan
from mindie_sim.solve import solve_deployment
from mindie_sim.sweep.pd import (
    _provable_decode_per_seq_read_by_stage,
    _provable_decode_tps_upper_bound,
    _solve_b_vram_max,
    solve_optimal_deployment,
)

pytestmark = pytest.mark.tier1

_PARITY_TOL_MS = 1e-12
_DEDUP_TOL_MS = 1e-9
_SOUNDNESS_REL_TOL = 1e-9
_TEST_PROMPT_SHORT = 1024
_TEST_PROMPT_LONG = 131072
_TEST_OUTPUT_SHORT = 128
_TEST_OUTPUT_LONG = 1024
_TEST_TPOT_SLA_MS = 30.0
_TEST_DECODE_TOKENS = 4
_TEST_SEQS = 4
_TEST_LAYERS_SMALL = 2
_TEST_LAYERS_HYBRID = 4
_TEST_B_LIMIT = 16
_TEST_EMIT_EST = 1.0
_TEST_DEVICES_SMALL = 8
_TEST_WORKERS_ONE = 1
_TEST_WORKERS_FOUR = 4
_TEST_MAX_SEQS_SWEEP = 8
_TEST_SWEEP_TPOT_MS = 50.0


def _run_graph_with_backend(
    monkeypatch: pytest.MonkeyPatch,
    case: dict,
    *,
    backend: str,
    tokens: int = _TEST_DECODE_TOKENS,
    context: int = _TEST_PROMPT_SHORT,
    sequences: int = _TEST_SEQS,
) -> dict:
    monkeypatch.setenv("MINDIE_SIM_ACCEL", backend)
    parsed = parse_case(case)
    graphs, eff_case, _ = _lower_and_run_graph(
        parsed, phase="decode", tokens=tokens, context=context,
        sequences=sequences, scope="model", run_graph=False,
    )
    ctx = EvaluationContext.create(eff_case)
    return graph_costs(
        eff_case, graphs, isolated=False, evaluation=ctx,
        fast_span_only=True, timeline_replicas=None,
    )


def test_c11_hybrid_epoch_continuation_zero_ns_parity_qwen38_and_kimi31(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C11 auto/c modes match pure Python to <= 1e-12 ms on Qwen3.8 and Kimi-K3.1."""
    cases = (
        (
            "qwen3.8-flash-next",
            dict(
                hardware="Ascend950DT_9582",
                feature_track="shipping",
                role="pd_decode",
                quant="auto_va",
                layers=_TEST_LAYERS_HYBRID,
                dp=2, tp=2, ep=4, pp=1, dcp=1,
                prompt_tokens=_TEST_PROMPT_SHORT,
                output_tokens=_TEST_OUTPUT_SHORT,
            ),
        ),
        (
            "kimi-k3.1",
            dict(
                hardware="Ascend950DT_9582",
                feature_track="shipping",
                role="pd_decode",
                quant="auto_va",
                layers=_TEST_LAYERS_HYBRID,
                dp=2, tp=2, ep=4, pp=1, dcp=2,
                prompt_tokens=_TEST_PROMPT_SHORT,
                output_tokens=_TEST_OUTPUT_SHORT,
            ),
        ),
    )
    for model, kwargs in cases:
        raw = compose_case(model, **kwargs)
        rep_py = _run_graph_with_backend(monkeypatch, raw, backend="python")
        rep_auto = _run_graph_with_backend(monkeypatch, raw, backend="auto")
        rep_c = _run_graph_with_backend(monkeypatch, raw, backend="c")

        assert math.isclose(
            rep_auto["duration_ms"], rep_py["duration_ms"], rel_tol=0.0, abs_tol=_PARITY_TOL_MS,
        )
        assert math.isclose(
            rep_c["duration_ms"], rep_py["duration_ms"], rel_tol=0.0, abs_tol=_PARITY_TOL_MS,
        )
        c_status = rep_c.get("accel_status") or {}
        auto_status = rep_auto.get("accel_status") or {}
        assert int(c_status.get("native_steps", 0) or 0) > 0
        assert int(auto_status.get("native_steps", 0) or 0) > 0


def test_dp_cohort_dedup_parity_and_megamoe_peer_collective_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric DP dedup matches full replay to <= 1e-9 ms and synthesizes MegaMoE peers."""
    symmetric_cases = (
        (
            "qwen3.8-flash-next",
            "shipping",
            dict(layers=_TEST_LAYERS_HYBRID, dp=4, tp=1, ep=4, pp=1, dcp=1, exclude_optimizations=("cann_megamoe_fusion",)),
            True,
            True,
        ),
        (
            "qwen3.8-flash-next",
            "target",
            dict(layers=_TEST_LAYERS_HYBRID, dp=4, tp=2, ep=8, pp=1, dcp=1),
            True,
            False,
        ),
        (
            "kimi-k3.1",
            "shipping",
            dict(layers=_TEST_LAYERS_SMALL, dp=4, tp=1, ep=4, pp=1, dcp=1),
            True,
            True,
        ),
    )
    for model, track, par_kw, expect_dedup, expect_r0_only in symmetric_cases:
        raw = compose_case(
            model,
            hardware="Ascend950DT_9582",
            feature_track=track,
            role="pd_decode",
            quant="auto_va",
            prompt_tokens=_TEST_PROMPT_SHORT,
            output_tokens=_TEST_OUTPUT_SHORT,
            **par_kw,
        )
        parsed = parse_case(raw)
        graphs = target_graphs(
            parsed, phase="decode", tokens=_TEST_DECODE_TOKENS,
            context=_TEST_PROMPT_SHORT, sequences=_TEST_SEQS, scope="model",
        )
        if not expect_r0_only:
            assert _has_megamoe_cross_dp_group(graphs[0]) is True

        book = RecipeBook.from_mapping(parsed.runtime.kernels)
        ctx = EvaluationContext.create(parsed, book, None)
        ex = Execution(parsed, ctx.cost, recipes=book, bindings=ctx.bindings, evaluation=ctx)
        try:
            works = {
                rep: IterationWork(
                    graphs[rep], IterationPlan(()), 1, _TEST_DECODE_TOKENS,
                    _TEST_DECODE_TOKENS, None, "graph", "declared_graph",
                )
                for rep in sorted(graphs)
            }
            monkeypatch.setenv("MINDIE_SIM_DP_DEDUP", "1")
            elig = evaluate_dp_cohort_dedup(ex, works)
            assert is_symmetric_dp_cohort(ex, works) is expect_dedup
            assert elig.symmetric is expect_dedup
            assert can_dedup_collective_to_canonical(
                ex.topology, ex.engine, requires_peer_collectives=elig.requires_peer_collectives,
            ) is expect_r0_only
        finally:
            accel = getattr(ex.engine, "_accel", None)
            if accel is not None:
                accel.close()

        monkeypatch.setenv("MINDIE_SIM_DP_DEDUP", "0")
        full = graph_costs(parsed, graphs, isolated=False, compact=True)
        monkeypatch.setenv("MINDIE_SIM_DP_DEDUP", "1")
        dedup = graph_costs(parsed, graphs, isolated=False, compact=True)
        assert abs(full["duration_ms"] - dedup["duration_ms"]) <= _DEDUP_TOL_MS
        assert full["operator_occurrences"] == dedup["operator_occurrences"]


def test_provable_kv_and_state_upper_bound_tightens_soundly() -> None:
    """Mandatory per-step KV/recurrent state reads tighten ub_tps while remaining >= actual."""
    for model, par in (
        ("qwen3.8-flash-next", dict(dp=4, tp=1, ep=4, pp=2, dcp=1)),
        ("kimi-k3.1", dict(dp=2, tp=8, ep=16, pp=2, dcp=8)),
    ):
        raw = compose_case(
            model,
            hardware="Ascend950DT_9582",
            feature_track="shipping",
            role="pd_decode",
            quant="auto_va",
            prompt_tokens=_TEST_PROMPT_LONG,
            output_tokens=_TEST_OUTPUT_LONG,
            **par,
        )
        parsed = parse_case(raw)
        mem, _, b_vram = _solve_b_vram_max(
            parsed, base_ws=1, decode_q=1, seq_len=_TEST_PROMPT_LONG + _TEST_OUTPUT_LONG,
        )
        assert b_vram >= 1
        per_seq_reads = _provable_decode_per_seq_read_by_stage(
            parsed, mem, prompt_tokens=_TEST_PROMPT_LONG, pp_stages=int(parsed.runtime.parallel.pp),
        )
        assert sum(per_seq_reads) > 0.0
        ub_zero_ctx = _provable_decode_tps_upper_bound(
            parsed, mem, b_limit=min(b_vram, _TEST_B_LIMIT), q_tok=1,
            emit_est=_TEST_EMIT_EST, tpot_sla_ms=_TEST_TPOT_SLA_MS, prompt_tokens=0,
        )
        ub_long_ctx = _provable_decode_tps_upper_bound(
            parsed, mem, b_limit=min(b_vram, _TEST_B_LIMIT), q_tok=1,
            emit_est=_TEST_EMIT_EST, tpot_sla_ms=_TEST_TPOT_SLA_MS, prompt_tokens=_TEST_PROMPT_LONG,
        )
        assert 0.0 < ub_long_ctx < ub_zero_ctx

    # Soundness check against exact Engine replay on a fast small-layer case.
    raw_s = compose_case(
        "qwen3.8-flash-next",
        hardware="Ascend950DT_9582",
        feature_track="shipping",
        role="pd_decode",
        quant="auto_va",
        layers=_TEST_LAYERS_HYBRID,
        dp=2, tp=2, ep=4, pp=1, dcp=1,
        prompt_tokens=_TEST_PROMPT_SHORT,
        output_tokens=_TEST_OUTPUT_SHORT,
    )
    parsed_s = parse_case(raw_s)
    mem_s = Memory(parsed_s, workspace_tokens=_TEST_SEQS)
    ub_s = _provable_decode_tps_upper_bound(
        parsed_s, mem_s, b_limit=_TEST_SEQS, q_tok=1,
        emit_est=_TEST_EMIT_EST, tpot_sla_ms=_TEST_TPOT_SLA_MS, prompt_tokens=_TEST_PROMPT_SHORT,
    )
    graphs_s, eff_s, _ = _lower_and_run_graph(
        parsed_s, phase="decode", tokens=1, context=_TEST_PROMPT_SHORT,
        sequences=_TEST_SEQS, scope="model", run_graph=False,
    )
    rep_s = graph_costs(eff_s, graphs_s, isolated=False, fast_span_only=True, timeline_replicas=None)
    per_dp_dies = int(parsed_s.runtime.parallel.tp) * int(parsed_s.runtime.parallel.pp)
    actual_tps = ((float(_TEST_SEQS) * _TEST_EMIT_EST) / float(per_dp_dies)) * (1000.0 / float(rep_s["duration_ms"]))
    assert ub_s >= actual_tps * (1.0 - _SOUNDNESS_REL_TOL)


def test_parallel_multiprobe_pd_decode_solver_bit_for_bit_parity() -> None:
    """Parallel static admission + multi-probe pd_decode waves match workers=1 bit-for-bit."""
    opt_kw = dict(
        num_devices=_TEST_DEVICES_SMALL,
        prompt_tokens=_TEST_PROMPT_SHORT,
        output_tokens=_TEST_OUTPUT_SHORT,
        tpot_sla_ms=_TEST_SWEEP_TPOT_MS,
        deployment_mode="pd_decode",
        mtp_proposal_tokens=0,
        max_sequences_per_dp=_TEST_MAX_SEQS_SWEEP,
        layers=_TEST_LAYERS_SMALL,
        allow_unconstrained_ttft=True,
    )
    opt_1 = solve_optimal_deployment(
        "qwen3-235b-a22b", "a5", granted_workers=_TEST_WORKERS_ONE, **opt_kw,
    )
    opt_4 = solve_optimal_deployment(
        "qwen3-235b-a22b", "a5", granted_workers=_TEST_WORKERS_FOUR, **opt_kw,
    )
    assert opt_1["feasible"] is True
    assert opt_4["best_configuration"] == opt_1["best_configuration"]
    assert opt_4["candidates"] == opt_1["candidates"]
    assert opt_4["search_coverage"] == opt_1["search_coverage"]
    assert multiprocessing.active_children() == []

    sol_kw = dict(
        num_devices=_TEST_DEVICES_SMALL,
        prompt_tokens=_TEST_PROMPT_SHORT,
        output_tokens=_TEST_OUTPUT_SHORT,
        tpot_slo_ms=_TEST_SWEEP_TPOT_MS,
        deployment_mode="pd_decode",
        mtp_tokens=0,
        max_sequences_per_dp=_TEST_MAX_SEQS_SWEEP,
        layers=_TEST_LAYERS_SMALL,
        mode="modeling",
    )
    sol_1 = solve_deployment("qwen3-235b-a22b", "a5", granted_workers=_TEST_WORKERS_ONE, **sol_kw)
    sol_4 = solve_deployment("qwen3-235b-a22b", "a5", granted_workers=_TEST_WORKERS_FOUR, **sol_kw)
    assert sol_1["feasible"] is True
    assert sol_4 == sol_1
    assert multiprocessing.active_children() == []


def test_fleet_pd_cap_stern_brocot_pareto_and_dual_cert_batch_search() -> None:
    """Exact Fraction SB/Pareto match brute force, fleet cap stays flat, dual-cert handles cliffs."""
    from mindie_sim.sweep import pd as pd_mod

    st = {
        "infeasible": None, "rps_p": 3e-12, "p_kv_rps_per_inst": 1.0,
        "rps_d": 4e-12, "d_kv_rps_per_inst": 1.0, "fleet_kv_rps_cap": None,
        "decode_tps_per_die": 1.0, "decode_budget": 1.0, "raw_rps_d": 4e-12,
        "decode_step": 1.0, "effective_tpot": 1.0, "first_itl": 1.0,
    }
    best_density = pd_mod._solve_optimal_pd_ratio_closed_form(
        st, total=3, p_dies=1, d_dies=1, output=2, mode="direct_blocking_d2d", objective="density",
    )
    assert best_density is not None and best_density["pd_ratio"] == "1P:1D"
    best_total = pd_mod._solve_optimal_pd_ratio_closed_form(
        st, total=3, p_dies=1, d_dies=1, output=2, mode="direct_blocking_d2d", objective="total_tps",
    )
    assert best_total is not None and best_total["pd_ratio"] == "2P:1D"

    # Full-budget exact Fraction Pareto vs brute force on asymmetric die costs (p_dies=2, d_dies=3, total=16).
    st_asym = {**st, "rps_p": 1.3, "rps_d": 2.9, "raw_rps_d": 2.9}
    brute_rows = [
        pd_mod._build_pd_ratio_row(
            st_asym, n_p=np_i, n_d=nd_i, total=16, p_dies=2, d_dies=3, output=16, mode="direct_blocking_d2d",
        )
        for np_i in range(1, 9) for nd_i in range(1, 6)
        if np_i * 2 + nd_i * 3 <= 16
    ]
    bf_dens = max(brute_rows, key=lambda r: (r["e2e_tps_per_die"], r["sustainable_tps"], -r["used_dies"]))
    bf_tot = max(brute_rows, key=lambda r: (r["sustainable_tps"], r["e2e_tps_per_die"], -r["used_dies"]))
    cf_dens = pd_mod._solve_optimal_pd_ratio_closed_form(
        st_asym, total=16, p_dies=2, d_dies=3, output=16, mode="direct_blocking_d2d", objective="density",
    )
    cf_tot = pd_mod._solve_optimal_pd_ratio_closed_form(
        st_asym, total=16, p_dies=2, d_dies=3, output=16, mode="direct_blocking_d2d", objective="total_tps",
    )
    assert cf_dens is not None and cf_dens["e2e_tps_per_die"] == pytest.approx(bf_dens["e2e_tps_per_die"], rel=1e-12)
    assert cf_tot is not None and cf_tot["sustainable_tps"] == pytest.approx(bf_tot["sustainable_tps"], rel=1e-12)

    # Shared fleet KV cap must NOT multiply by n_p or n_d.
    st_cap = {
        **st, "rps_p": 10.0, "p_kv_rps_per_inst": 100.0,
        "rps_d": 10.0, "d_kv_rps_per_inst": 100.0, "fleet_kv_rps_cap": 15.0,
    }
    row_cap = pd_mod._build_pd_ratio_row(
        st_cap, n_p=4, n_d=4, total=8, p_dies=1, d_dies=1, output=2, mode="direct_blocking_d2d",
    )
    assert row_cap["kv_transfer_rps"] == pytest.approx(15.0)
    assert row_cap["sustainable_rps"] == pytest.approx(15.0)

    # Dual-certificate branch-and-bound batch search: non-monotone, super-linear, and ACLGraph padding cliff.
    def _run_certified_probes(limit: int, sla_ms: float, probe_fn, *, floors_fn=None, capture_bs=()):
        memo: dict[int, float] = {}
        rays_map: dict[int, tuple[tuple[float, float], ...]] = {}
        floors: dict[int, float] = {}
        while True:
            nxt = pd_mod._certified_sla_next_probes(limit, sla_ms, memo, rays_map, floors, capture_bs)
            if not nxt:
                break
            for cand_b in nxt:
                memo[cand_b] = probe_fn(cand_b)
                if floors_fn is not None:
                    floors[cand_b] = floors_fn(cand_b)
        return memo

    non_mono = {1: 4.0, 2: 6.0, 3: 4.0, 4: 6.0}
    memo_nm = _run_certified_probes(4, 5.0, lambda b: non_mono[b])
    assert pd_mod._select_best_feasible_batch(memo_nm, 4, 5.0)[:2] == (3, 4.0)

    super_lin = {1: 1.0, 2: 1.5, 3: 4.0, 4: 4.9}
    memo_sl = _run_certified_probes(4, 5.0, lambda b: super_lin[b])
    assert pd_mod._select_best_feasible_batch(memo_sl, 4, 5.0)[:2] == (2, 1.5)

    # Strictly padded b=10 (padded to 16) probes capture boundary 8 and picks unpadded 8 when 8/2.0 > 10/3.0;
    # unpadded exact capture b=8 terminates with certified floor pruning.
    cap_bs = (1, 2, 3, 4, 5, 6, 7, 8, 16)
    memo_pad = _run_certified_probes(
        10, 5.0, lambda b: 2.0 if b <= 8 else 3.0,
        floors_fn=lambda b: 2.0 if b <= 8 else 3.0, capture_bs=cap_bs,
    )
    assert {8, 10}.issubset(memo_pad)
    assert pd_mod._select_best_feasible_batch(memo_pad, 10, 5.0)[:2] == (8, 2.0)
    memo_exact = _run_certified_probes(
        8, 50.0, lambda _b: 0.6, floors_fn=lambda _b: 0.6, capture_bs=cap_bs,
    )
    assert set(memo_exact) == {1, 2, 4, 8}
    assert pd_mod._select_best_feasible_batch(memo_exact, 8, 50.0)[:2] == (8, 0.6)

    # Exact Fraction tie-break (3*0.3 float rounding must not flip 3P:3D -> 1P:1D).
    st_tie = dict(
        infeasible=None, rps_p=0.3, rps_d=0.3, p_kv_rps_per_inst=100.0, d_kv_rps_per_inst=100.0,
        decode_budget=1.0, decode_step=1.0, effective_tpot=1.0, first_itl=1.0,
    )
    row_tie = pd_mod._solve_optimal_pd_ratio_closed_form(
        st_tie, total=6, p_dies=1, d_dies=1, output=2, mode="direct_blocking_d2d",
    )
    assert row_tie is not None and row_tie["pd_ratio"] == "3P:3D"

    # Non-monotone interior dip (b=1 & b=8..9 exceed 1.091 ms, b=7 fits and maximizes throughput).
    from dataclasses import replace
    from mindie_sim.api import compose_case as compose_spec
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.disaggregated.handoff import compute_pd_handoff_plan
    from mindie_sim.prediction.pd_connector.handoff import PDTransfer
    from mindie_sim.schema.hardware_links import Link

    c_dip = compose_spec(
        "deepseek-v3.2", "a5",
        options=dict(tp=1, ep=1, layers=1, mtp_proposal_tokens=0, prompt_tokens=128, output_tokens=8),
    )
    dip_res = pd_mod._solve_sla_archetype_aware(c_dip, sla_ms=1.091, context=128, output_budget=8, max_sequences_per_dp=9)
    assert dip_res["scenarios"][0]["b_feasible"] == 7

    # Direct RoCE placement/endpoint_ids/framing + pooled fleet switch cap + decode HBM global rank index + layer clocks.
    m_roce = compose_spec("qwen3-32b", "a5", options=dict(tp=4, layers=1, ranks_per_node=2)).machine
    roce_link = Link(
        "roce", 128.0, (0, 1, 2, 3), channel="d2d_inter_node", scope="cross_node", semantics="per_port",
        endpoint_ids=((0, "nic"), (1, "nic")), transaction_bytes=64, alignment_bytes=64,
        protocol_overhead_bytes=64, switch_capacity_bytes_per_ms=256.0,
    )
    c_roce = replace(
        compose_spec("qwen3-32b", "a5", options=dict(tp=2, layers=1, ranks_per_node=2)),
        machine=replace(m_roce, links=tuple(x for x in m_roce.links if x.channel != "d2d_inter_node") + (roce_link,)),
    )
    h_roce = dict(
        transfers=(PDTransfer(0, 2, 1), PDTransfer(1, 3, 1)),
        source_rank_bytes={0: 1, 1: 1}, destination_rank_bytes={2: 1, 3: 1}, total_transfer_bytes=2,
    )
    for n_dies in (4, 16):
        info = pd_mod._resolve_pd_link_bandwidth(c_roce, c_roce, total_dies=n_dies, kv_transfer_bw_gbps=None)
        assert info["source"] == "sku_link:d2d_inter_node"
        assert pd_mod._compute_d2d_wire_ms(h_roce, info) == pytest.approx(2.0, rel=1e-9)
    sw_pool = pd_mod.sweep_pd_disaggregation_ratios(
        c_roce, c_roce, total_dies=16, prompt_tokens=128, output_tokens=8, max_sequences_per_dp=1,
        ttft_sla_ms=1000.0, tpot_sla_ms=1000.0, pd_transfer_mode="pooled_dram_ssd", pool_dram_ratio=1.0,
    )
    assert sw_pool["ratios"][0]["sustainable_rps"] <= (256000.0 / sw_pool["kv_bytes_per_request"]) + 1e-9

    p_hbm = compose_spec("qwen3-32b", "a5", options=dict(tp=2, ep=1, layers=1, mtp_proposal_tokens=0))
    d_hbm = replace(p_hbm, machine=replace(p_hbm.machine, dies=tuple(
        replace(d, hbm_write_bytes_per_s=bw) for d, bw in zip(p_hbm.machine.dies, (1024000.0, 102400.0))
    )))
    dst_bytes = compute_pd_handoff_plan(p_hbm, d_hbm, tokens=1)["destination_rank_bytes"]
    assert pd_mod._pool_kv_hbm_ms(d_hbm, Memory(d_hbm), 1, direction="write", per_die_bytes=dst_bytes) == pytest.approx(20.0)

    c_clk = compose_spec("qwen3-32b", "a5", options=dict(tp=1, ep=1, layers=2, mtp_proposal_tokens=0, prompt_tokens=128, output_tokens=8))
    clocks: dict[int, float] = {}
    _, _, _, p_step = pd_mod._measure_prefill_stage(
        c_clk, prompt=128, compute_tokens=128, hit_tokens=0, cache=None, cache_id=None, layer_finish_ms_out=clocks,
    )
    assert set(clocks) == {0, 1} and 0.0 < clocks[0] < clocks[1] <= p_step + 1e-6


def test_integer_pd_handoff_draft_kv_asymmetric_tp_and_cohort_reciprocity() -> None:
    """Integer PDTransfer rejects floats, conserves asymmetric-TP draft KV, & rejects cyclic cohorts."""
    from types import SimpleNamespace
    from mindie_sim import predict_pd
    from mindie_sim.api import compose_case as compose_spec
    from mindie_sim.prediction.disaggregated.handoff import (
        compute_pd_handoff_plan,
        resolve_and_validate_pd_connector,
    )
    from mindie_sim.prediction.pd_connector.handoff import PDTransfer
    from mindie_sim.prediction.runtime.dp_cohort_dedup import is_symmetric_native_program
    from mindie_sim.schema.common import ModelingError

    # PDTransfer strictly enforces integer payload_bytes (rejects float/bool).
    with pytest.raises(ModelingError, match="pd_transfer_payload_requires_positive_int_bytes"):
        PDTransfer(source_rank=0, destination_rank=1, payload_bytes=12.5)  # type: ignore[arg-type]

    # MLA DCP interleave (token 0 -> rank 2), PCP+DCP unique bytes, and prefill draft_max_model_len clamp.
    def _spec(model: str, tp: int, **kw):
        opts = dict(tp=tp, ep=1, layers=1, mtp_proposal_tokens=0, prompt_tokens=128, output_tokens=64)
        return compose_spec(model, "a5", options=opts | kw)

    h_dcp = compute_pd_handoff_plan(_spec("deepseek-v3.2", 2), _spec("deepseek-v3.2", 2, dcp=2), tokens=1)
    assert [(t.destination_rank, t.payload_bytes) for t in h_dcp["transfers"] if t.component == "mla_kv"] == [(2, 1152)]
    h_pcp = compute_pd_handoff_plan(_spec("qwen3-32b", 8, pcp=2, dcp=2), _spec("qwen3-32b", 8), tokens=256)
    assert h_pcp["unique_kv_bytes"] == 8 * 256 * 128 * 2 * 2
    h_cap = compute_pd_handoff_plan(
        _spec("kimi-k3.1", 2, ep=2, layers=4, mtp_proposal_tokens=3, draft_max_model_len=64, role="pd_prefill", ttft_slo_ms=5000.0),
        _spec("kimi-k3.1", 4, ep=4, layers=4, mtp_proposal_tokens=3, draft_max_model_len=128, role="pd_decode"),
        tokens=128,
    )
    t_cap = next(t for t in h_cap["transfers"] if t.component == "draft.mla_kv")
    assert t_cap.source_token_span == (0, 64) and t_cap.payload_bytes == 64 * 1152

    # Asymmetric K/V plane dtypes (INT8 K = 4096, BF16 V = 8192) & overlapping pools [0,1]->[1,2] real pair service.
    raw_ov = compose_case("qwen3-32b", "a5", ep=1, tp=2, layers=1, prompt_tokens=4, output_tokens=2, mtp_proposal_tokens=0)
    def _ov_pool(ranks: list[int]) -> dict:
        m = dict(raw_ov["machine"], devices=[{"rank": i} for i in ranks], links=[dict(lk, members=ranks) for lk in raw_ov["machine"]["links"]])
        return dict(machine=m, runtime=raw_ov["runtime"])
    raw_ov["pools"] = {"prefill": _ov_pool([0, 1]), "decode": _ov_pool([1, 2])}
    raw_ov["connector"] = {"link": dict(name="handoff", capacity_bytes_per_ms=64.0, semantics="per_pair"), "chunk_bytes": 1024}
    raw_ov["model"]["metadata"]["cache_representations"] = {"gqa_kv_key": {"dtype": "int8"}, "gqa_kv_value": {"dtype": "bf16"}}
    ov_exec = predict_pd(raw_ov, published={"prefill": {"gqa_k": {"output": "gqa_k"}, "gqa_v": {"output": "gqa_v"}}})["execution"]
    ov_t = ov_exec["pd_handoffs"][0]["transfers"]
    assert [sum(x["payload_bytes"] for x in ov_t if x["component"] == k) for k in ("gqa_k", "gqa_v")] == [4096, 8192]
    assert [ov_exec["serviced"].get("pd:link:handoff:pair" + p, 0.0) for p in ("0>2", "1>3")] == [6144.0, 6144.0]

    # Asymmetric-TP draft KV conservation in BOTH directions (TP_P < TP_D and TP_P > TP_D).
    for tp_p, tp_d in ((2, 4), (4, 2)):
        p_raw = compose_case(
            "kimi-k3.1", "950dt-96g-pod", role="pd_prefill", dp=1, tp=tp_p, ep=tp_p, pp=1,
            layers=4, prompt_tokens=1024, output_tokens=64, mtp_proposal_tokens=3, ttft_slo_ms=5000.0,
        )
        d_raw = compose_case(
            "kimi-k3.1", "950dt-96g-pod", role="pd_decode", dp=1, tp=tp_d, ep=tp_d, pp=1,
            layers=4, prompt_tokens=1024, output_tokens=64, mtp_proposal_tokens=3,
        )
        p_parsed, d_parsed = parse_case(p_raw), parse_case(d_raw)
        ok_conn, conn_kind, _ = resolve_and_validate_pd_connector(
            d_parsed.model, p_parsed.runtime.parallel, d_parsed.runtime.parallel,
        )
        assert ok_conn is True and conn_kind
        plan = compute_pd_handoff_plan(p_parsed, d_parsed, tokens=1024)
        assert all(isinstance(t.payload_bytes, int) and t.payload_bytes > 0 for t in plan["transfers"])
        assert any(t.component.startswith("draft") for t in plan["transfers"])
        assert int(plan["total_transfer_bytes"]) == sum(int(t.payload_bytes) for t in plan["transfers"])

    # Directed-cycle (0->1->2->0) has non-reciprocal routes and MUST be rejected by cohort dedup.
    cyc_matrix = {0: {1: 4}, 1: {2: 4}, 2: {0: 4}}
    cyc_inner = SimpleNamespace(
        peer_count=3,
        route_matrix=cyc_matrix,
        start_with_handles=lambda *_a, **_kw: None,
        frames_for=lambda rk: cyc_matrix[rk],
        payload_basis_for=lambda _rk: "exact",
        local_frames_for=lambda _rk: 0,
    )
    assert is_symmetric_native_program(SimpleNamespace(program=cyc_inner), 3) is False


def test_linear_attention_candidate_generator_filters_non_divisible_tp():
    """Hybrid linear-attention models (e.g. kimi-k3.1, 96 KDA heads) omit tp not dividing linear_state_heads."""
    from mindie_sim.catalog import load_model
    from mindie_sim.sweep.pd import _build_colocated_candidates, _build_pd_stage_candidates

    kimi_spec = load_model("kimi-k3.1")
    for role in ("pd_prefill", "pd_decode"):
        cands = _build_pd_stage_candidates(
            role=role, max_dies=64, is_moe=True, layers=4,
            prompt_tokens=4096, rec_topo=None, pinned={}, model_spec=kimi_spec,
        )
        assert cands and all(96 % int(c["tp"]) == 0 for c in cands)
        assert not any(int(c["tp"]) == 64 for c in cands)
    coloc = _build_colocated_candidates(
        num_devices=64, is_moe=True, layers=4, rec_topo=None, pinned={},
        model_spec=kimi_spec, prompt_tokens=4096, role="pd_decode",
    )
    assert coloc and all(96 % int(c["tp"]) == 0 for c in coloc)
    assert not any(int(c["tp"]) == 64 for c in coloc)
    pinned_64 = _build_pd_stage_candidates(
        role="pd_decode", max_dies=64, is_moe=True, layers=4,
        prompt_tokens=4096, rec_topo=None, pinned={"tp": 64}, model_spec=kimi_spec,
    )
    assert any(int(c["tp"]) == 64 for c in pinned_64)


def test_prefill_prefix_cache_threads_query_and_context_to_export_and_ablation(monkeypatch: pytest.MonkeyPatch):
    """Prefill winner with 2048 prompt and 1024 cached prefix passes (1024, 1024) to export and ablation."""
    from mindie_sim import solve as solve_mod

    raw = compose_case(
        "qwen3-32b", "a3", dp=1, tp=2, ep=1, layers=2,
        prompt_tokens=2048, output_tokens=64, prefix_tokens=1024,
        role="pd_prefill", ttft_slo_ms=5000.0,
    )
    solved = {
        "feasible": True,
        "winner_mode": "pd_prefill",
        "deployment_mode": "pd_prefill",
        "best_configuration": {"used_dies": 2, "b_feasible": 1, "mtp_proposal_tokens": 0},
        "prefill_case": raw,
        "recommended_case": raw,
    }
    _rec_case, req, _meta = solve_mod._bundle_inputs_for_solution(
        solved, model="qwen3-32b", hardware="a3", runtime_stack="vllm-ascend",
        base_kw={}, feature_track="shipping", prompt_tokens=2048, output_tokens=64, scope="model",
    )
    assert req.phase == "prefill"
    assert req.tokens == 1024 and req.context == 1024

    captured_analyze: list[tuple[str, int, int]] = []
    captured_wf: list[tuple[str, int, int]] = []

    def _spy_analyze(_case, *, phase, tokens, context, **_kw):
        captured_analyze.append((phase, int(tokens), int(context)))
        return {
            "geometry": {"dp": 1, "tp": 2, "ep": 1},
            "included_components": {},
            "graph_duration_ms": 1.0,
            "network_cube_mfu": 0.5,
            "network_vec_mfu": 0.2,
            "network_mbu": 0.4,
            "network_link_util": 0.1,
            "network_cv_overlap": 0.3,
            "network_roofline_efficiency": 0.5,
            "breakdown_by_family": [],
        }

    def _spy_compare(*_a, phase, tokens, context, **_kw):
        captured_wf.append((phase, int(tokens), int(context)))
        return {"waterfall": {}}

    with monkeypatch.context() as mp:
        mp.setattr(solve_mod, "_analyze_case_bounded", _spy_analyze)
        mp.setattr(solve_mod, "compare_feature_tracks", _spy_compare)
        solve_mod._evaluate_winner_analysis(
            solved=solved, before_solved=solved, after_solved=solved,
            model="qwen3-32b", hardware="a3", runtime_stack="vllm-ascend",
            prompt=2048, output=64, ttft_slo_ms=5000.0, tpot_slo_ms=None,
            layers=2, quant=None, kv_cache_dtype=None, indexer_kv_dtype=None,
            topology_preset=None, ranks_per_node=None, mtp_accept=None,
            ablate_prs=True, xlsx_path=None, trace_path=None,
            compare_track="shipping", target_track="target", mode="simulation",
        )
    assert captured_analyze == [("prefill", 1024, 1024)]
    assert captured_wf == [("prefill", 1024, 1024)]


def test_pooled_disaggregated_prefill_persists_prefix_hit_for_export_and_ablation(monkeypatch: pytest.MonkeyPatch):
    """Pooled disaggregated winner persists (8192, 8192) on the solution record for export and ablation."""
    from mindie_sim import solve as solve_mod
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.sweep import pd as pd_mod

    # Role case omits prefix_tokens (pooled hit ratio is evaluated dynamically per PD mode).
    p_raw = compose_case(
        "qwen3-32b", "a3", dp=1, tp=2, ep=1, layers=2,
        prompt_tokens=16384, output_tokens=1, role="pd_prefill", ttft_slo_ms=5000.0,
    )
    d_raw = compose_case(
        "qwen3-32b", "a3", dp=1, tp=2, ep=1, layers=2,
        prompt_tokens=16384, output_tokens=1, role="pd_decode", tpot_slo_ms=50.0,
    )
    p_parsed = parse_case(p_raw)
    d_parsed = parse_case(d_raw)
    stage = {
        "p_step": 100.0, "effective_ttft_ms": 100.0,
        "compute_tokens": 8192, "hit_tokens": 8192,
        "b_feasible": 1, "b_vram": 4, "configured_max_num_seqs": 4,
        "decode_step": 0.0, "handoff_tpot_ms": 0.0, "effective_tpot": 0.0,
        "avg_accepted_tokens": 1.0, "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
        "unused_sla_slack_ms": 0.0, "operating_regime": "SLA_LATENCY_BOUND",
        "limiting_constraint": "", "free_hbm_headroom_gib": 10.0,
        "kv_ms": 5.0, "infeasible": None,
    }
    best_ratio = {
        "pd_ratio": "1P:1D", "n_prefill": 1, "n_decode": 1,
        "prefill_dies": 2, "decode_dies": 2, "used_dies": 4,
        "sustainable_rps": 10.0, "prefill_rps": 10.0, "kv_transfer_rps": 20.0,
        "sustainable_tps": 10.0, "e2e_tps_per_die": 2.5,
    }
    winner_row = pd_mod._build_pd_candidate_row_from_stage(
        stage, p_parsed=p_parsed, d_parsed=d_parsed, best_ratio=best_ratio,
        pd_mode="pooled_dram_ssd", mtp_n=0, p_world=2, d_world=2,
    )
    assert winner_row["effective_prefill_tokens"] == 8192
    assert winner_row["prefix_hit_tokens"] == 8192

    solved = pd_mod._finalize_optimal_deployment(
        model_id="qwen3-32b", sku="a3", n_dev=4, prompt=16384, output=1,
        ttft=5000.0, tpot=50.0, dep_norm="pd_disaggregated",
        all_candidates=[winner_row], coloc_entries=[],
        pd_entries=[(winner_row, p_raw, d_raw)],
        compose_for=lambda *_a, **_kw: d_raw,
    )
    assert solved["pd_pool_summary"]["effective_prefill_tokens"] == 8192
    assert solved["pd_pool_summary"]["prefix_hit_tokens"] == 8192

    _rec_case, req, _meta = solve_mod._bundle_inputs_for_solution(
        solved, model="qwen3-32b", hardware="a3", runtime_stack="vllm-ascend",
        base_kw={}, feature_track="shipping", prompt_tokens=16384, output_tokens=1, scope="model",
    )
    assert req.phase == "prefill"
    assert (req.tokens, req.context) == (8192, 8192)

    captured_analyze: list[tuple[str, int, int]] = []
    captured_wf: list[tuple[str, int, int]] = []

    def _spy_analyze(_case, *, phase, tokens, context, **_kw):
        captured_analyze.append((phase, int(tokens), int(context)))
        return {
            "geometry": {"dp": 1, "tp": 2, "ep": 1},
            "included_components": {},
            "graph_duration_ms": 1.0,
            "network_cube_mfu": 0.5, "network_vec_mfu": 0.2,
            "network_mbu": 0.4, "network_link_util": 0.1,
            "network_cv_overlap": 0.3, "network_roofline_efficiency": 0.5,
            "breakdown_by_family": [],
        }

    def _spy_compare(*_a, phase, tokens, context, **_kw):
        captured_wf.append((phase, int(tokens), int(context)))
        return {"waterfall": {}}

    with monkeypatch.context() as mp:
        mp.setattr(solve_mod, "_analyze_case_bounded", _spy_analyze)
        mp.setattr(solve_mod, "compare_feature_tracks", _spy_compare)
        solve_mod._evaluate_winner_analysis(
            solved=solved, before_solved=solved, after_solved=solved,
            model="qwen3-32b", hardware="a3", runtime_stack="vllm-ascend",
            prompt=16384, output=1, ttft_slo_ms=5000.0, tpot_slo_ms=50.0,
            layers=2, quant=None, kv_cache_dtype=None, indexer_kv_dtype=None,
            topology_preset=None, ranks_per_node=None, mtp_accept=None,
            ablate_prs=True, xlsx_path=None, trace_path=None,
            compare_track="shipping", target_track="target", mode="simulation",
        )
    assert captured_analyze == [("prefill", 8192, 8192)]
    assert captured_wf == [("prefill", 8192, 8192)]
