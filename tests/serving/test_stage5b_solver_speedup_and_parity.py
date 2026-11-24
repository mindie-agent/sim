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
                feature_track="merged",
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
                feature_track="merged",
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
            "merged",
            dict(layers=_TEST_LAYERS_HYBRID, dp=4, tp=1, ep=4, pp=1, dcp=1),
            True,
            True,
        ),
        (
            "qwen3.8-flash-next",
            "expected_ideal",
            dict(layers=_TEST_LAYERS_HYBRID, dp=4, tp=2, ep=8, pp=1, dcp=1),
            True,
            False,
        ),
        (
            "kimi-k3.1",
            "merged",
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
            assert is_symmetric_dp_cohort(ex, works) is expect_dedup
            assert can_dedup_collective_to_canonical(ex.topology, ex.engine) is expect_r0_only
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
            feature_track="merged",
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
        feature_track="merged",
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
