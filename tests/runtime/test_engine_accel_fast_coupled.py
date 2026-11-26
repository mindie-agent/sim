"""C11 coupled-arbiter parity, native callback hooks, MSVC flags, and DP-cohort dedup.

[Role] Verify bit-exact / <= 1e-12 parity between C11 ``fast_single_tier_coupled_c``
  and pure-Python ``fast_single_tier_coupled_py``, native boundary callbacks +
  interned ``memory_scope`` frontier preservation, Windows MSVC compiler flags,
  and symmetric homogeneous DP-cohort deduplication across dense, MoE, MLA,
  KDA, GDN, DCP, PCP, and MC2 topologies (plus fail-closed asymmetry fallback).
[Invariants] Tier 0: pure in-memory + local shared-library checks, zero network,
  zero subprocess spawns, zero host environment leakage (all env toggles via
  ``monkeypatch``).
[Timeouts & Locks] Deterministic bounded graphs (< 10s total); no sleeps or I/O.
[Side-Effect Policy] Read-only over repository files; monkeypatched env only.
"""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from mindie_sim.api import compose_case
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.runtime import accel_build
from mindie_sim.prediction.runtime.arbiter_fast_coupled import (
    fast_single_tier_coupled_c,
    fast_single_tier_coupled_py,
)
from mindie_sim.prediction.runtime.dp_cohort_dedup import is_symmetric_dp_cohort
from mindie_sim.prediction.runtime.engine import Engine, Resource

pytestmark = pytest.mark.tier0


def _close(engine: Engine) -> None:
    accel = getattr(engine, "_accel", None)
    if accel is not None and accel.arena is not None:
        accel.arena.close()


def _max_rate_diff(left: dict[str, dict[int, float]], right: dict[str, dict[int, float]]) -> float:
    assert set(left) == set(right)
    worst = 0.0
    for rname in left:
        assert set(left[rname]) == set(right[rname])
        for seq in left[rname]:
            worst = max(worst, abs(left[rname][seq] - right[rname][seq]))
    return worst


@pytest.mark.parametrize(
    "scenario",
    [
        "single_unit_coupled",
        "multi_unit_skewed_overlap",
        "capped_and_widened_cohort",
        "mixed_coupled_and_uncoupled",
        "shared_multi_resource_bottleneck",
    ],
)
def test_coupled_native_matches_python_across_unit_regimes(scenario: str) -> None:
    """C11 fast_single_tier_coupled matches Python reference to <= 1e-12."""
    if scenario == "single_unit_coupled":
        resources = [Resource("hbm", 2400.0, units=1), Resource("mte2", 1200.0, units=1)]
        specs = [
            ({"hbm": 120.0, "mte2": 90.0}, {"hbm": 800.0}, True, 1, (), None),
            ({"hbm": 200.0, "mte2": 60.0}, {}, True, 1, (), None),
            ({"hbm": 50.0}, {"hbm": 300.0}, True, 1, (), None),
        ]
    elif scenario == "multi_unit_skewed_overlap":
        resources = [
            Resource("hbm", 3200.0, units=8),
            Resource("mte2", 1600.0, units=8),
            Resource("l2", 6400.0, units=4),
        ]
        specs = [
            (
                {"hbm": 100.0 + i * 3.0, "mte2": 80.0 + (i % 7), "l2": 60.0 + (i % 5)},
                {"hbm": 450.0, "mte2": 260.0} if i % 3 == 0 else {},
                True,
                1,
                (),
                {"hbm": (i % 8, (i + 1) % 8), "mte2": (i % 8,), "l2": (i % 4,)},
            )
            for i in range(24)
        ]
    elif scenario == "capped_and_widened_cohort":
        resources = [Resource("hbm", 1600.0, units=4), Resource("vec", 800.0, units=4)]
        specs = [
            ({"hbm": 240.0, "vec": 120.0}, {"hbm": 300.0}, True, 2, ("hbm",), {"hbm": (0, 1), "vec": (0, 1)}),
            ({"hbm": 180.0, "vec": 180.0}, {"vec": 150.0}, True, 2, ("hbm", "vec"), {"hbm": (1, 2), "vec": (0, 1)}),
            ({"hbm": 300.0}, {}, True, 1, (), {"hbm": (2, 3)}),
            ({"hbm": 90.0, "vec": 45.0}, {}, True, 1, (), None),
        ]
    elif scenario == "mixed_coupled_and_uncoupled":
        resources = [Resource("hbm", 2000.0, units=4), Resource("mte2", 1000.0, units=4)]
        specs = [
            ({"hbm": 150.0, "mte2": 100.0}, {}, True, 1, (), {"hbm": (0, 1), "mte2": (0,)}),
            ({"hbm": 220.0, "mte2": 110.0}, {"hbm": 400.0}, False, 1, (), {"hbm": (1, 2), "mte2": (0, 1)}),
            ({"hbm": 100.0}, {}, False, 1, (), None),
            ({"hbm": 80.0, "mte2": 160.0}, {}, True, 1, (), {"hbm": (0, 3), "mte2": (1,)}),
        ]
    else:
        resources = [
            Resource("hbm", 3000.0, units=6),
            Resource("mte2", 2100.0, units=6),
            Resource("mte3", 1800.0, units=3),
            Resource("l2", 5400.0, units=3),
        ]
        specs = [
            (
                {"hbm": 70.0 + i, "mte2": 95.0 + (i % 5) * 11.0, "mte3": 40.0 + (i % 3) * 7.0, "l2": 55.0},
                {"hbm": 350.0} if i % 2 == 0 else {"mte2": 220.0},
                True,
                1,
                (),
                {"hbm": (i % 6, (i + 2) % 6), "mte2": (i % 6,), "mte3": (i % 3,), "l2": (i % 3,)},
            )
            for i in range(18)
        ]

    eng = Engine(resources, backend="python")
    try:
        for idx, (work, caps, coupled, cohort, widened, placement) in enumerate(specs):
            eng.submit(
                f"op_{idx}",
                work=work,
                caps=caps,
                coupled=coupled,
                cohort_size=cohort,
                cohort_widened=widened,
                placement=placement,
            )
        active = set(eng._tasks)
        ref = fast_single_tier_coupled_py(eng.resources, eng._tasks, active)
        nat = fast_single_tier_coupled_c(eng.resources, eng._tasks, active)
        assert ref is not None and nat is not None
        assert _max_rate_diff(ref, nat) <= 1e-12
    finally:
        _close(eng)


def test_coupled_native_declines_fifo_and_multi_priority_and_preserves_underflow() -> None:
    """FIFO / multi-priority return None; subnormal scale raises ZeroDivisionError."""
    eng_mp = Engine([Resource("hbm", 1000.0, units=2)], backend="python")
    try:
        eng_mp.submit("p0", work={"hbm": 10.0}, priority=0, coupled=True)
        eng_mp.submit("p1", work={"hbm": 10.0}, priority=1, coupled=True)
        assert fast_single_tier_coupled_c(eng_mp.resources, eng_mp._tasks, set(eng_mp._tasks)) is None
    finally:
        _close(eng_mp)

    eng_fifo = Engine([Resource("q", 1000.0, policy="fifo", units=2)], backend="python")
    try:
        eng_fifo.submit("f0", work={"q": 10.0}, coupled=True)
        assert fast_single_tier_coupled_c(eng_fifo.resources, eng_fifo._tasks, set(eng_fifo._tasks)) is None
    finally:
        _close(eng_fifo)

    eng_uf = Engine([Resource("r0", 1e300, units=1), Resource("r1", 1e300, units=1)], backend="python")
    try:
        eng_uf.submit("tiny", work={"r0": 1e-300, "r1": 1e-300}, coupled=True)
        with pytest.raises(ZeroDivisionError, match="coupled_scale_underflow"):
            fast_single_tier_coupled_py(eng_uf.resources, eng_uf._tasks, set(eng_uf._tasks))
        with pytest.raises(ZeroDivisionError, match="coupled_scale_underflow"):
            fast_single_tier_coupled_c(eng_uf.resources, eng_uf._tasks, set(eng_uf._tasks))
    finally:
        _close(eng_uf)


def test_native_callback_hook_and_memory_scope_preserve_native_frontier() -> None:
    """memory_scope interned pools, unresolved promises, and native_hook=True keep c_native live."""
    results = {}
    for backend in ("python", "auto"):
        eng = Engine(
            [Resource("cube", 1000.0, units=2), Resource("hbm", 800.0, units=2)],
            backend=backend,
            accel_eager_attach=(backend == "auto"),
        )
        try:
            gate = eng.future("ext_gate")
            done_events = []
            fired: list[int] = []
            with eng.memory_scope(None, replica=0, stage=0, pools=("hbm", "l2")):
                prev = eng.submit("seed", work={"cube": 20.0})
                for idx in range(24):
                    deps = (prev,) if idx > 0 else (prev, gate)
                    ev = eng.submit(
                        f"stage_{idx}",
                        work={"cube": 30.0 + idx, "hbm": 25.0 + idx},
                        after=deps,
                        coupled=True,
                        placement={"cube": (idx % 2,), "hbm": (idx % 2,)},
                    )
                    if idx in (5, 14):
                        eng.on_complete(
                            ev,
                            lambda _e, mark=idx: fired.append(mark),
                            native_hook=True,
                        )
                    done_events.append(ev)
                    prev = ev
            eng.submit(
                "unlock",
                work={"hbm": 15.0},
                on_complete=lambda _e, p=gate: eng.resolve(p),
                native_hook=True,
            )
            eng.advance(until=(done_events[-1],), first=False)
            status = eng.accel_status()
            if backend == "auto" and status.get("toolchain_ready"):
                assert status["native_steps"] > 0
                assert status["disabled_reason"] is None
            results[backend] = (
                eng.now,
                dict(eng.serviced),
                dict(eng.busy_ms),
                tuple(fired),
                tuple(ev.time for ev in done_events),
            )
        finally:
            _close(eng)

    assert results["auto"][3] == (5, 14)
    assert math.isclose(results["auto"][0], results["python"][0], rel_tol=0.0, abs_tol=1e-12)
    assert results["auto"][4] == pytest.approx(results["python"][4], abs=1e-12)
    for rname, val in results["python"][1].items():
        assert results["auto"][1][rname] == pytest.approx(val, abs=1e-12)


def test_windows_msvc_compiler_discovery_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """MSVC cl.exe discovery and C11 shared-library flags format cleanly."""
    assert accel_build._is_msvc("cl")
    assert accel_build._is_msvc("C:/BuildTools/VC/Tools/MSVC/bin/cl.exe")
    assert not accel_build._is_msvc("/usr/bin/clang")
    flags = accel_build.compiler_flags("C:/tools/cl.exe")
    assert "/std:c11" in flags
    assert "/O2" in flags
    assert "/fp:precise" in flags
    assert "/LD" in flags

    monkeypatch.setattr(accel_build, "os", SimpleNamespace(name="nt", environ={}))
    monkeypatch.setattr(
        accel_build.shutil,
        "which",
        lambda name: "C:/MSVC/bin/cl.exe" if name == "cl" else None,
    )
    assert accel_build.discover_compiler() == ("C:/MSVC/bin/cl.exe", None)


def _run_dp_parity(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    hardware: str,
    options: dict[str, object],
    *,
    phase: str = "decode",
    tokens: int = 8,
    context: int = 1024,
    sequences: int = 4,
    compact: bool = True,
) -> tuple[dict, dict]:
    case = compose_case(model, hardware=hardware, options={"output_tokens": 64, **options})
    ops = target_graphs(case, phase=phase, tokens=tokens, context=context, sequences=sequences)
    monkeypatch.setenv("MINDIE_SIM_DP_DEDUP", "0")
    full = graph_costs(
        case, ops, external_inputs={"analysis_input": 0}, isolated=False, compact=compact,
    )
    monkeypatch.setenv("MINDIE_SIM_DP_DEDUP", "1")
    dedup = graph_costs(
        case, ops, external_inputs={"analysis_input": 0}, isolated=False, compact=compact,
    )
    return full, dedup


def _assert_graph_parity(full: dict, dedup: dict) -> None:
    assert abs(full["duration_ms"] - dedup["duration_ms"]) <= 1e-9
    assert full["operator_occurrences"] == dedup["operator_occurrences"]
    full_res = {row["resource"]: row["work"] for row in full["resources"]}
    dedup_res = {row["resource"]: row["work"] for row in dedup["resources"]}
    assert set(full_res) == set(dedup_res)
    for rname, expected in full_res.items():
        tol = max(1e-6, abs(expected) * 1e-9)
        assert abs(dedup_res[rname] - expected) <= tol, f"resource mismatch on {rname}"


def test_dp_cohort_dedup_parity_dense_and_pcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Symmetric DP dedup preserves duration and conserved work on dense TP+DP and PCP+DP."""
    full_tp, dedup_tp = _run_dp_parity(
        monkeypatch,
        "qwen3-32b",
        "a3",
        {"layers": 2, "tp": 2, "dp": 2, "concurrency": 8, "prompt_tokens": 1024},
        phase="decode",
        tokens=4,
        context=1024,
        sequences=4,
    )
    _assert_graph_parity(full_tp, dedup_tp)

    full_pcp, dedup_pcp = _run_dp_parity(
        monkeypatch,
        "qwen3-32b",
        "a3",
        {"layers": 2, "dp": 2, "pcp": 2, "concurrency": 4, "prompt_tokens": 1024},
        phase="prefill",
        tokens=512,
        context=0,
        sequences=2,
    )
    _assert_graph_parity(full_pcp, dedup_pcp)


def test_dp_cohort_dedup_parity_moe_mla_dcp_gdn_kda_mc2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Symmetric DP dedup matches full expansion across MoE, MLA+DCP, GDN, and KDA+MC2."""
    cases = [
        (
            "deepseek-v3.2",
            "a3",
            {"layers": 4, "tp": 2, "dp": 2, "ep": 4, "dcp": 2, "concurrency": 8, "prompt_tokens": 1024},
            4,
            4,
        ),
        (
            "qwen3-next-80b-a3b",
            "a3",
            {"layers": 2, "dp": 2, "ep": 2, "concurrency": 8, "prompt_tokens": 1024},
            4,
            4,
        ),
        (
            "kimi-k3.1",
            "Ascend950DT_9582",
            {
                "layers": 2, "tp": 1, "dp": 4, "ep": 4,
                "concurrency": 16, "prompt_tokens": 1024, "topology_preset": "a5_pod",
            },
            4,
            4,
        ),
        (
            "kimi-k3.1",
            "Ascend950DT_9582",
            {
                "layers": 2, "tp": 2, "dp": 2, "ep": 4,
                "concurrency": 16, "prompt_tokens": 1024, "topology_preset": "a5_pod",
            },
            4,
            4,
        ),
    ]
    for model, hw, opts, tok, seqs in cases:
        full, dedup = _run_dp_parity(
            monkeypatch, model, hw, opts, tokens=tok, context=1024, sequences=seqs,
        )
        _assert_graph_parity(full, dedup)


def test_dp_cohort_dedup_declines_asymmetric_replicas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any rank-asymmetric workload skew fails closed and falls back to full multi-replica."""
    case = compose_case(
        "qwen3-32b",
        hardware="a3",
        options={"layers": 2, "tp": 1, "dp": 2, "concurrency": 8, "prompt_tokens": 1024, "output_tokens": 64},
    )
    ops = target_graphs(case, phase="decode", tokens=4, context=1024, sequences=4)
    base_ops = ops[0] if isinstance(ops, dict) else tuple(ops)
    skewed_ops = {0: tuple(base_ops), 1: tuple(base_ops)}
    r1_ops = list(skewed_ops[1])
    target_idx = next(i for i, op in enumerate(r1_ops) if op.flops > 0)
    r1_ops[target_idx] = replace(r1_ops[target_idx], flops=r1_ops[target_idx].flops * 2.0)
    skewed_ops[1] = tuple(r1_ops)

    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.schema.operators import IterationWork
    from mindie_sim.schema.workload import IterationPlan

    book = RecipeBook.from_mapping(case.runtime.kernels)
    ctx = EvaluationContext.create(case, book, None)
    ex = Execution(case, ctx.cost, recipes=book, bindings=ctx.bindings, evaluation=ctx)
    try:
        works = {
            r: IterationWork(skewed_ops[r], IterationPlan(()), 1, 4, 4, None, "graph", "declared_graph")
            for r in (0, 1)
        }
        assert not is_symmetric_dp_cohort(ex, works)
    finally:
        _close(ex.engine)

    monkeypatch.setenv("MINDIE_SIM_DP_DEDUP", "0")
    ref = graph_costs(case, skewed_ops, external_inputs={"analysis_input": 0}, isolated=False, compact=True)
    monkeypatch.setenv("MINDIE_SIM_DP_DEDUP", "1")
    got = graph_costs(case, skewed_ops, external_inputs={"analysis_input": 0}, isolated=False, compact=True)
    _assert_graph_parity(ref, got)


@pytest.mark.parametrize("backend", ["python", "c"])
def test_uncoupled_fair_resource_underflows_alongside_coupled_task(backend: str) -> None:
    """Uncoupled fair link must raise service_rate_underflow even while a coupled task runs."""
    from mindie_sim.prediction.runtime.engine_types import ExecutionError

    eng = Engine([
        Resource("cube", 10.0),
        Resource("hbm", 10.0),
        Resource("link", 1e-322, policy="fair"),
    ], backend=backend)
    try:
        eng.submit("coupled", {"cube": 100.0, "hbm": 100.0}, coupled=True)
        for idx in range(100):
            eng.submit(f"u{idx}", {"link": 1.0}, coupled=False)
        with pytest.raises(ExecutionError, match="service_rate_underflow"):
            eng.advance(max_steps=16)
    finally:
        _close(eng)


@pytest.mark.parametrize("backend", ["python", "c"])
def test_start_observer_resolves_across_python_and_native_backends(backend: str) -> None:
    """Native epoch commits must resolve task.start_observer identically to Python mode."""
    eng = Engine([Resource("r", 10.0)], backend=backend)
    try:
        prep = eng.prepare_stage("work", {"r": 20.0})
        done, started = eng._submit_host_launch(prep, "work", "obs_started", after=())
        eng.advance(until=(done,))
        assert started.done and started.time == pytest.approx(0.0)
        assert done.done and done.time == pytest.approx(2.0)
    finally:
        _close(eng)


def test_ring_buffer_producer_extra_after_roundtrips_through_native_abi() -> None:
    """Native ABI ring validator must accept DEP_KIND_EXTRA edges from producer_extra_after."""
    from _support.prepared_terms_parity import PreparedTermsRunner
    from mindie_sim.prediction.runtime.ring_buffer import DEP_KIND_EXTRA, RingBuffer, lower_generation

    ring = RingBuffer(occurrence=1, die=0, owner=0, pool="ub", allocation="r", slot_bytes=1024, depth=2)
    g0 = lower_generation(
        ring, 0, producer="prod", consumers=("cons",),
        producer_work={"mte2": 1.0}, consumer_works={"cons": {"vec": 1.0}},
    )
    lower_generation(
        ring, 1, producer="prod", consumers=("cons",),
        producer_work={"mte2": 1.0}, consumer_works={"cons": {"vec": 1.0}},
        producer_extra_after=(g0.publish,),
    )
    _, deps = ring.to_c_records()
    assert any(d.kind == DEP_KIND_EXTRA for d in deps)
    with PreparedTermsRunner() as runner:
        assert runner.roundtrip_ring(ring) == ring.to_c_records()


def test_dp_cohort_dedup_fast_span_only_matches_detailed_peer_die_overlap_decision() -> None:
    """fast_span_only must not alter peer-die MC2 overlap decisions vs detailed execution."""
    from mindie_sim.prediction.runtime.dp_cohort_dedup import _has_unsynced_peer_die_overlap
    from mindie_sim.schema.operators import IterationWork, OpSpec
    from mindie_sim.schema.workload import IterationPlan

    op_main = OpSpec(
        "s0.l0.mc2", "grouped_gemm", "decode", 0, 0, (), (8, 64), (8, 64), (8, 64), (8, 64),
        dtype="bf16", stream="main", metadata=(("mc2_phase", "dispatch"), ("group_axis", "ep")),
    )
    op_aux = OpSpec(
        "s0.l0.shared", "gemm", "decode", 0, 0, (), (8, 64), (8, 64), (8, 64), (8, 64),
        dtype="bf16", stream="aux",
    )
    w0 = IterationWork((op_main, op_aux), IterationPlan(()), 1, 8, 8, None, "graph", "declared_graph")
    base_case = SimpleNamespace(
        runtime=SimpleNamespace(parallel=SimpleNamespace(dp=2, tp=1, ep=2, pp=1)),
    )
    topo = SimpleNamespace(namespace="dp_test", links={}, ranks=lambda _r, _s: (0, 1))
    eng = SimpleNamespace(resources={})
    detailed_ex = SimpleNamespace(_fast_span_only=False, case=base_case, topology=topo, engine=eng)
    fast_ex = SimpleNamespace(_fast_span_only=True, case=base_case, topology=topo, engine=eng)
    detailed_res = _has_unsynced_peer_die_overlap(detailed_ex, w0, 0)
    fast_res = _has_unsynced_peer_die_overlap(fast_ex, w0, 0)
    assert fast_res is detailed_res is True


def test_c11_sparse_epoch_dynamic_4096_res_and_multiunit_parity() -> None:
    """C11 dynamic resource table (>2048), multi-unit overlaps, & repeated epochs match Python."""
    from mindie_sim.prediction.runtime.engine import Engine, Resource

    count = 4096
    res_list = [Resource(f"r{i}", 1.0 + float(i % 7) * 0.25) for i in range(count)]
    res_list.extend([
        Resource("mu_fair", 16.0, units=8, policy="fair"),
        Resource("mu_wide", 8.0, units=4, policy="fair"),
    ])
    eng_py = Engine(res_list, backend="python")
    eng_c = Engine(res_list, backend="c")
    try:
        # Epoch 1: >2048 resources with coupled + uncoupled sparse active-coeff flows.
        ev_py = [
            eng_py.submit(f"t{i}", {f"r{i}": 1.0, f"r{(i + 1) % count}": 0.5}, coupled=(i % 3 == 0))
            for i in range(64)
        ]
        ev_c = [
            eng_c.submit(f"t{i}", {f"r{i}": 1.0, f"r{(i + 1) % count}": 0.5}, coupled=(i % 3 == 0))
            for i in range(64)
        ]
        # Overlapping & disjoint multi-unit placements (proportional widths on coupled multi-res).
        placements = [
            ({"mu_fair": 3.0, "mu_wide": 1.5}, {"mu_fair": (0, 1, 2, 3), "mu_wide": (0, 1)}, True),
            ({"mu_fair": 2.5, "mu_wide": 1.25}, {"mu_fair": (2, 3, 4, 5), "mu_wide": (1, 2)}, True),
            ({"mu_fair": 2.0}, {"mu_fair": (6, 7)}, True),
            ({"mu_wide": 2.0}, {"mu_wide": (2, 3)}, False),
        ]
        for idx, (work, place, cpl) in enumerate(placements):
            ev_py.append(eng_py.submit(f"mu{idx}", work, placement=place, coupled=cpl))
            ev_c.append(eng_c.submit(f"mu{idx}", work, placement=place, coupled=cpl))
        eng_py.advance()
        eng_c.advance()
        # Epoch 2: post-drain reuse verifies O(1) epoch bump clears stale res_chead/touched_res.
        e2_py = eng_py.submit("ep2_a", {"r4095": 1.5, "mu_fair": 2.0}, placement={"mu_fair": tuple(range(8))}, coupled=True)
        e2_c = eng_c.submit("ep2_a", {"r4095": 1.5, "mu_fair": 2.0}, placement={"mu_fair": tuple(range(8))}, coupled=True)
        e3_py = eng_py.submit("ep2_b", {"mu_fair": 1.0, "mu_wide": 0.5}, placement={"mu_fair": (0, 1), "mu_wide": (0,)}, coupled=True)
        e3_c = eng_c.submit("ep2_b", {"mu_fair": 1.0, "mu_wide": 0.5}, placement={"mu_fair": (0, 1), "mu_wide": (0,)}, coupled=True)
        eng_py.advance()
        eng_c.advance()
        ev_py.extend((e2_py, e3_py))
        ev_c.extend((e2_c, e3_c))

        st = eng_c.accel_status()
        assert st["fallback_reason"] is None and int(st["native_steps"]) > 0
        for a, b in zip(ev_py, ev_c, strict=True):
            assert b.time == pytest.approx(a.time, rel=1e-12, abs=1e-12)
        assert eng_c.now == pytest.approx(eng_py.now, rel=1e-12, abs=1e-12)
    finally:
        _close(eng_py)
        _close(eng_c)


def test_prepared_cache_clear_resets_floor_peaks_and_uses_monotonic_static_id() -> None:
    """Clearing prepared caches clears _seen_floor_peaks and monotonic static_id prevents id() reuse."""
    from mindie_sim.prediction.program.stages import Stage
    from mindie_sim.prediction.runtime.engine import Allocation, Engine, Resource

    eng = Engine(
        (Resource("mte2", 1024.0, units=4),),
        pools={"l0b": 262144.0},
        pool_units={"l0b": 4},
        backend="python",
    )
    try:
        a1 = Allocation("a1", "l0b", 32768.0, (0,))
        s1 = Stage(
            "k1",
            work={"mte2": 128.0},
            acquire=(a1,),
            free=(a1,),
            placement={"mte2": (0,)},
        )
        p1 = eng._prepare_static_stage(s1)
        eng._submit_prepared(p1, "t1")
        eng.advance()
        assert eng.peak_unit_allocated[("l0b", 0)] == 32768.0
        assert p1.static_id in eng._seen_floor_peaks

        eng._clear_prepared_caches()
        assert not eng._seen_floor_peaks

        a2 = Allocation("a2", "l0b", 65536.0, (0,))
        s2 = Stage(
            "k2",
            work={"mte2": 128.0},
            acquire=(a2,),
            free=(a2,),
            placement={"mte2": (0,)},
        )
        p2 = eng._prepare_static_stage(s2)
        assert p2.static_id > p1.static_id
        eng._submit_prepared(p2, "t2")
        eng.advance()
        assert eng.peak_unit_allocated[("l0b", 0)] == 65536.0
    finally:
        eng.close()


def test_engine_close_releases_accel_and_preserves_status() -> None:
    """Engine.close() deterministically frees the native arena while keeping final accel_status()."""
    from mindie_sim.prediction.runtime.engine import Engine, Resource

    eng = Engine((Resource("r0", 1.0),), backend="python")
    eng.submit("t0", {"r0": 2.0})
    eng.advance()
    before = eng.accel_status()
    eng.close()
    assert eng._accel is None
    assert eng.accel_status()["accel_backend"] == before["accel_backend"]
    eng.close()


def test_replica_block_reciprocity_accepts_tp_unbalanced_tokens_and_rejects_cycles() -> None:
    """Replica-block reciprocity accepts seq%tp!=0 DP-symmetric matrices and rejects directed cycles."""
    from mindie_sim.prediction.runtime.dp_cohort_dedup import is_symmetric_native_program

    # 2 DP replicas x 2 TP ranks: pos 0 sends 2048B to all peers, pos 1 sends 0B.
    rows_sym = (
        (0.0, 2048.0, 2048.0, 2048.0),
        (0.0, 0.0, 0.0, 0.0),
        (2048.0, 2048.0, 0.0, 2048.0),
        (0.0, 0.0, 0.0, 0.0),
    )
    inner_sym = SimpleNamespace(
        peer_count=4,
        route_matrix=None,
        route_matrix_bytes=rows_sym,
        start_with_handles=lambda *_a, **_kw: None,
        payload_basis_for=lambda _rk: "expected",
        local_frames_for=lambda _rk: 0,
    )
    assert is_symmetric_native_program(SimpleNamespace(program=inner_sym), 4, local_rank_count=2) is True

    # Directed cross-replica cycle (R0 -> R1 -> R2 -> R0, local_k=2) MUST be rejected.
    rows_cyc = [[0.0] * 6 for _ in range(6)]
    for rep in range(3):
        nxt = (rep + 1) % 3
        rows_cyc[rep * 2][nxt * 2] = 2048.0
    inner_cyc = SimpleNamespace(
        peer_count=6,
        route_matrix=None,
        route_matrix_bytes=tuple(tuple(r) for r in rows_cyc),
        start_with_handles=lambda *_a, **_kw: None,
        payload_basis_for=lambda _rk: "expected",
        local_frames_for=lambda _rk: 0,
    )
    assert is_symmetric_native_program(SimpleNamespace(program=inner_cyc), 6, local_rank_count=2) is False


def test_coupled_unplaced_multi_unit_bus_runs_native_and_hands_back_on_partial_peer() -> None:
    """Unplaced multi-unit bus stays C11-eligible with 0-ULP parity until a partial peer touches it."""
    from mindie_sim.prediction.runtime.engine import Engine, Resource

    res = (
        Resource("matrix", 32.0, units=32),
        Resource("mte1", 32.0, units=32),
        Resource("memory", 320.0, units=32),
    )
    # 1. Pure unplaced shared bus + proportional placed compute -> C11 native_steps > 0, exact parity.
    out_clean = {}
    for be in ("python", "c"):
        eng = Engine(res, backend=be)
        try:
            evs = [
                eng.submit(
                    f"g{i}",
                    {"matrix": 4.0, "mte1": 4.0, "memory": 40.0},
                    coupled=True,
                    placement={"matrix": (i,), "mte1": (i,)},
                )
                for i in range(16)
            ]
            eng.advance()
            st = eng.accel_status()
            out_clean[be] = ([float(e.time) for e in evs], int(st["native_steps"]), int(st["python_steps"]))
        finally:
            eng.close()
    assert out_clean["c"][1] > 0 and out_clean["c"][2] == 0
    assert out_clean["c"][0] == pytest.approx(out_clean["python"][0], rel=0.0, abs=1e-12)

    # 2. A peer partially placing on ``memory`` triggers handback so unplaced-span redistribution stays exact.
    out_mix = {}
    for be in ("python", "c"):
        eng = Engine(res, backend=be)
        try:
            e_c = eng.submit(
                "coupled_unplaced_bus",
                {"matrix": 1.0, "memory": 320.0},
                coupled=True,
                placement={"matrix": (0,)},
            )
            e_p = eng.submit(
                "partial_bus_peer",
                {"memory": 20.0},
                placement={"memory": (0,)},
            )
            eng.advance()
            out_mix[be] = (float(e_c.time), float(e_p.time))
        finally:
            eng.close()
    assert out_mix["c"] == pytest.approx(out_mix["python"], rel=0.0, abs=1e-12)
