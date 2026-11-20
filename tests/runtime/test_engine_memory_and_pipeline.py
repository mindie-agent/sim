"""Engine cohort arbitration, 3-ledger memory physics, ring buffer, tile solver and wave recorder.

[Role] Verify Stage-2 Workstream 2A runtime contracts: explicit cohort widening
  and mid-flight symmetry splitting, 3-ledger memory binding with dynamic L2
  working-set knees and 2-row directional HBM buses (Python == C11 within 8 ULP),
  vector sub-pipes, 3-event ring buffers, tile liveness solving, 5-point wave
  recording, and program package re-export parity.
[Invariants] Cohort merging matches unmerged execution within 1e-9 ms on both
  Python and C11; ByteLedger conserves hit + miss == aligned; solo HBM R/W has
  zero turnaround penalty while concurrent R+W satisfies max(Tr,Tw)+alpha*min(Tr,Tw);
  WaveRecorder enforces busy + fill + steady_bubble + drain + handoff == span_ns
  with zero schedule perturbation on Python and C11.
[Timeouts & Locks] Tier-0 bounded tests (<5s total, compiles/exercises C11
  shared library in isolated cache); no network, threads or sleeps.
[Side-Effect Policy] Pure in-memory Engine/C11 contexts; zero host mutation.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from mindie_sim.prediction.coupled import coupled_rates
from mindie_sim.prediction.engine import Engine, Resource
import mindie_sim.prediction.program as prog_pkg
from mindie_sim.prediction.program import (
    AddressRun,
    AddressRunPlan,
    AdmissionMemoryView,
    BurstTurnRule,
    HbmBusSpec,
    KernelProgram,
    MemoryAccessPattern,
    ParallelProgram,
    RecipeContext,
    ReuseKey,
    Stage,
    VectorInstructionGroup,
    VisibilityContract,
    coalesce_execution_program,
    merge_programs,
)
from mindie_sim.prediction.runtime.accel_terms import MsimAccess, PreparedTermsRunner
from mindie_sim.prediction.runtime.arbiter_fast_coupled import fast_single_tier_coupled
from mindie_sim.prediction.runtime.prepared_terms import (
    POLICY_FAIR,
    POLICY_FIFO,
    PreparedCapacityRow,
    PreparedDelivery,
    PreparedProgress,
    PreparedTerm,
    allocate_prepared_epoch,
    integrate_prepared_epoch,
)
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer, lower_generation
from mindie_sim.prediction.runtime.tile_solver import (
    LiveBufferSpec,
    SubtileWaveSpec,
    TileCandidate,
    TileInfeasibleError,
    TilePipelineSpec,
    TileShape,
    solve as solve_tiles,
)
from mindie_sim.prediction.runtime.wave_recorder import (
    RecordedPipelineSpec,
    WaveRecorder,
)
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0


def _ulp_distance(a: float, b: float) -> int:
    if a == b:
        return 0
    return int(round(abs(a - b) / max(math.ulp(a), math.ulp(b))))


def _make_context(*, alpha_rw: float = 0.25) -> RecipeContext:
    res_names = {
        "hbm": "hbm@d0", "mte2_aic": "mte2_aic@d0", "mte2_aiv": "mte2_aiv@d0",
        "mte3": "mte3@d0", "l2": "l2@d0", "vec_issue": "vec_issue@d0",
        "vec_alu": "vec_alu@d0", "vec_sfu": "vec_sfu@d0", "vec_reduce": "vec_reduce@d0",
    }
    rates = {
        "hbm@d0": 1_600_000_000.0, "mte2_aic@d0": 800_000_000.0,
        "mte2_aiv@d0": 800_000_000.0, "mte3@d0": 800_000_000.0,
        "l2@d0": 3_200_000_000.0, "vec_issue@d0": 1_000_000.0,
        "vec_alu@d0": 1_000_000.0, "vec_sfu@d0": 250_000.0, "vec_reduce@d0": 500_000.0,
    }
    return RecipeContext(
        op=SimpleNamespace(family="elementwise", recipe="direct", metadata={}),
        work={}, resource_names=res_names, rates=rates, core_count=20,
        physical_mechanisms=SimpleNamespace(hbm_alpha_rw=alpha_rw),
        compute_arch=SimpleNamespace(clock_hz=1_000_000_000.0),
    )


@pytest.mark.parametrize("backend", ["python", "c"])
@pytest.mark.parametrize("hbm_work,cube_work", [(1.0, 10.0), (10.0, 1.0)])
def test_widened_cohort_asymmetric_split_matches_unmerged_across_backends(
    backend: str, hbm_work: float, cube_work: float
):
    """19 widened owners + 1 solo owner must match 20 unmerged owners within 1e-9 ms."""
    def run(merged: bool) -> tuple[float, dict[str, float], int]:
        eng = Engine([Resource("hbm", 1.0, units=1), Resource("cube", 1.0, units=20)], backend=backend)
        start = eng.submit("start")
        progs = tuple(
            KernelProgram((Stage(
                "run",
                {"hbm": hbm_work * (29 if core == 19 else 28),
                 "cube": cube_work * (29 if core == 19 else 28)},
                caps={"hbm": 1.0, "cube": 1.0}, placement={"cube": (core,)}, coupled=True,
            ),))
            for core in range(20)
        )
        program = coalesce_execution_program(ParallelProgram(progs)) if merged else ParallelProgram(progs)
        done = program.start(eng, "op", start)
        eng.advance(until=(done,))
        return float(done.time), dict(eng.serviced), eng.accel_status()["native_steps"]

    t_m, s_m, steps_m = run(True)
    t_u, s_u, steps_u = run(False)
    assert abs(t_m - t_u) <= 1e-9
    for key, val in s_u.items():
        assert s_m[key] == pytest.approx(val, rel=1e-12, abs=1e-9)
    if backend == "c":
        assert steps_m > 0 and steps_u > 0


@pytest.mark.parametrize("backend", ["python", "c"])
def test_shared_multiunit_cohort_distinguished_from_widened_disjoint(backend: str):
    """Two members sharing units (0, 1) must not be misclassified as widened disjoint."""
    def run(merged: bool) -> tuple[float, dict[str, float]]:
        eng = Engine([Resource("cube", 2.0, units=2), Resource("hbm", 1.0, units=1)], backend=backend)
        start = eng.submit("start")
        k0 = KernelProgram((Stage("s", {"cube": 2.0, "hbm": 1.0}, placement={"cube": (0, 1)}, coupled=True),))
        k1 = KernelProgram((Stage("s", {"cube": 2.0, "hbm": 1.0}, placement={"cube": (0, 1)}, coupled=True),))
        peer = eng.submit("peer", {"cube": 2.0, "hbm": 1.0}, after=(start,), placement={"cube": (0, 1)}, coupled=True)
        if merged:
            cohort = merge_programs([k0, k1])
            assert cohort is not None and cohort.stages[0].cohort_size == 2
            assert cohort.stages[0].cohort_widened == ()
            done = eng.submit("done", after=(cohort.start(eng, "merged", start), peer))
        else:
            done = eng.submit("done", after=(k0.start(eng, "m0", start), k1.start(eng, "m1", start), peer))
        eng.advance(until=(done,))
        if backend == "c":
            assert eng.accel_status()["native_steps"] > 0
        return float(done.time), dict(eng.serviced)

    t_m, s_m = run(True)
    t_u, s_u = run(False)
    assert abs(t_m - t_u) <= 1e-9
    assert s_m == pytest.approx(s_u, rel=1e-12)


@pytest.mark.parametrize("backend", ["python", "c"])
def test_midflight_asymmetric_interference_splits_widened_cohort_members(backend: str):
    """An external task arriving mid-flight on unit 0 splits a 4-unit cohort accurately."""
    def run(merged: bool) -> tuple[float, dict[str, float]]:
        eng = Engine([Resource("cube", 4.0, units=4), Resource("hbm", 4.0, units=1)], backend=backend)
        start = eng.submit("start")
        members = [
            KernelProgram((Stage("k", {"cube": 2.0, "hbm": 1.0}, placement={"cube": (i,)}, coupled=True),))
            for i in range(4)
        ]
        if merged:
            merged_k = merge_programs(members)
            assert merged_k is not None and merged_k.stages[0].cohort_widened == ("cube",)
            main_ev = merged_k.start(eng, "coh", start)
        else:
            main_ev = eng.submit("join", after=tuple(m.start(eng, f"m{i}", start) for i, m in enumerate(members)))
        ext = eng.submit("ext", {"cube": 1.0}, release=0.5, placement={"cube": (0,)})
        done = eng.submit("done", after=(main_ev, ext))
        eng.advance(until=(done,))
        return float(done.time), dict(eng.serviced)

    t_m, s_m = run(True)
    t_u, s_u = run(False)
    assert abs(t_m - t_u) <= 1e-9
    assert s_m == pytest.approx(s_u, rel=1e-12)


def test_coupled_and_overlapping_uncoupled_python_c_parity_within_8_ulp():
    """Coupled multi-flow and overlapping uncoupled + coupled peers stay within 8 ULP."""
    res1 = [Resource("a", 26.289056632197735), Resource("b", 36.495015663833634)]
    works1 = [
        {"a": 7.05924503470419, "b": 8.494354162581535},
        {"a": 6.694301119164283, "b": 5.5353786183562095},
        {"a": 8.483307144965345, "b": 7.87160639063931},
    ]
    out1 = {}
    for be in ("python", "c"):
        eng = Engine(res1, backend=be)
        evs = [eng.submit(f"t{i}", w, coupled=True) for i, w in enumerate(works1)]
        eng.advance(until=tuple(evs), first=False)
        out1[be] = [float(e.time) for e in evs]
    assert all(_ulp_distance(a, b) <= 8 for a, b in zip(out1["python"], out1["c"]))

    res2 = [Resource("cube", 245.76, units=20), Resource("vector", 1.92, units=40)]
    out2 = {}
    for be in ("python", "c"):
        eng = Engine(res2, backend=be)
        u0 = eng.submit("u0", {"cube": 1.0154159211083948}, placement={"cube": (10, 15, 17)})
        u1 = eng.submit(
            "u1", {"cube": 3.6808735333841436},
            placement={"cube": (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19)},
        )
        c0 = eng.submit(
            "c0", {"cube": 500.0, "vector": 100.0}, coupled=True,
            placement={"cube": (3, 4), "vector": (6, 7, 8, 9)},
        )
        eng.advance(until=(u0, u1, c0), first=False)
        out2[be] = [float(u0.time), float(u1.time), float(c0.time)]
    assert all(_ulp_distance(a, b) <= 8 for a, b in zip(out2["python"], out2["c"]))


def test_fast_single_tier_coupled_matches_coupled_rates_on_widened_cohort():
    """Both Python coupled arbiters must agree identically on widened cohorts."""
    eng = Engine([Resource("cube", 20.0, units=20), Resource("hbm", 1.0, units=1)], backend="python")
    eng.submit(
        "coh", {"cube": 190.0, "hbm": 19.0}, placement={"cube": tuple(range(19))},
        coupled=True, cohort_size=19, cohort_widened=("cube",),
    )
    eng.submit("solo", {"cube": 10.0, "hbm": 1.0}, placement={"cube": (19,)}, coupled=True)
    eng._settle()
    r_fast = fast_single_tier_coupled(eng.resources, eng._tasks, eng._active)
    r_ref = coupled_rates(eng.resources, {k: eng._tasks[k] for k in eng._active})
    assert r_fast is not None
    assert r_fast["hbm"][1] == pytest.approx(19.0 / 20.0, rel=1e-12)
    assert r_fast["hbm"][2] == pytest.approx(1.0 / 20.0, rel=1e-12)
    for rname in ("cube", "hbm"):
        for seq in (1, 2):
            assert r_fast[rname][seq] == pytest.approx(r_ref[rname][seq], rel=1e-12)


def test_three_ledger_bind_access_sector_union_turnaround_and_empty_runs():
    """Discrete 656 B rows amplify to 1,376,256 B sectors; aligned 512 B has zero amp."""
    ctx = _make_context(alpha_rw=0.25)
    discrete = MemoryAccessPattern(
        tensor_id="kv_rows", version=1, die=0, owner=0, direction="read", pipe="mte2_aiv",
        logical_bytes=656 * 2048,
        runs=AddressRunPlan((AddressRun(base_byte=0, rows=2048, row_bytes=656, stride_bytes=672),)),
        reuse_key=ReuseKey(die=0, backing_allocation="kv", tensor_version=1),
        visibility=VisibilityContract(scope="owner"),
    )
    turn_rule = BurstTurnRule(min_row_bytes=1, max_row_bytes=1024, min_stride_bytes=1, max_stride_bytes=1024, penalty_bytes=164.0)
    bound_disc = ctx.bind_access(discrete, burst_rules=(turn_rule,))
    eta = 656.0 / (656.0 + 164.0)  # 0.8
    assert bound_disc.ledger.logical_tensor_bytes == 656 * 2048
    assert bound_disc.ledger.aligned_transfer_bytes == 672 * 2048
    assert bound_disc.ledger.dram_sector_bytes == 1_376_256
    assert bound_disc.ledger.dram_bus_transaction_bytes == pytest.approx(1_376_256 / eta)
    assert bound_disc.ledger.l2_hit_bytes + bound_disc.ledger.dram_miss_copy_bytes == bound_disc.ledger.aligned_transfer_bytes

    aligned = MemoryAccessPattern(
        tensor_id="dense_w", version=1, die=0, owner=0, direction="read", pipe="mte2_aic",
        logical_bytes=512 * 1024,
        runs=AddressRunPlan((AddressRun(base_byte=0, rows=1024, row_bytes=512, stride_bytes=512),)),
        reuse_key=ReuseKey(die=0, backing_allocation="w", tensor_version=1),
    )
    bound_al = ctx.bind_access(aligned)
    assert bound_al.ledger.dram_sector_bytes == 512 * 1024
    assert bound_al.ledger.dram_bus_transaction_bytes == pytest.approx(float(512 * 1024))

    empty = MemoryAccessPattern(
        tensor_id="empty", version=1, die=0, owner=0, direction="write", pipe="mte3",
        logical_bytes=0,
        runs=AddressRunPlan((AddressRun(base_byte=0, rows=0, row_bytes=64, stride_bytes=64),)),
    )
    bound_empty = ctx.bind_access(empty)
    assert bound_empty.constraints == () and bound_empty.deliveries == ()
    assert bound_empty.ledger.aligned_transfer_bytes == 0

    with PreparedTermsRunner() as runner:
        valid_acc = MsimAccess(
            logical_bytes=bound_disc.ledger.logical_tensor_bytes,
            aligned_bytes=bound_disc.ledger.aligned_transfer_bytes,
            hit_bytes=bound_disc.ledger.l2_hit_bytes,
            miss_bytes=bound_disc.ledger.dram_miss_copy_bytes,
            sector_bytes=bound_disc.ledger.dram_sector_bytes,
            dram_service_bytes=bound_disc.ledger.dram_bus_transaction_bytes,
            copy_pipe_row=0, l2_row=1, die=0, direction=0, flags=0,
        )
        assert runner.validate_access(valid_acc) == 0
        bad_acc = MsimAccess(
            logical_bytes=1024, aligned_bytes=1024, hit_bytes=512, miss_bytes=256,
            sector_bytes=256, dram_service_bytes=256.0, copy_pipe_row=0, l2_row=1,
            die=0, direction=0, flags=0,
        )
        assert runner.validate_access(bad_acc) != 0


def test_dynamic_l2_knee_n_streams_eviction_and_non_resurrection():
    """Shared L2 knee scales as floor(C_avail / (g*N)), evicts immediately, never resurrects."""
    mib = 1024 * 1024
    with pytest.raises(ModelingError, match="shared_l2_buffer_requires_one_die_owner"):
        AdmissionMemoryView({0: 128 * mib}, l2_buffer_units_by_die={0: 2})

    view = AdmissionMemoryView({0: 128 * mib}, l2_buffer_units_by_die={0: 1})
    assert view.effective_l2_bytes(0) == 128 * mib

    key_a = ReuseKey(die=0, backing_allocation="weights_a", tensor_version=1, reuse_stream="s0")
    pat_96m = MemoryAccessPattern(
        tensor_id="w96", version=1, die=0, owner=0, direction="read", pipe="mte2_aic",
        logical_bytes=96 * mib,
        runs=AddressRunPlan((AddressRun(base_byte=0, rows=1, row_bytes=96 * mib, stride_bytes=96 * mib),)),
        reuse_key=key_a, visibility=VisibilityContract(scope="die"),
    )
    view.enter_stream(0, "s0")
    assert view.admit_access(pat_96m, 96 * mib) == (0, 96 * mib)
    assert view.is_resident(key_a)
    assert view.admit_access(pat_96m, 96 * mib) == (96 * mib, 0)

    view.enter_stream(0, "s1")
    assert view.effective_l2_bytes(0) == 64 * mib
    assert not view.is_resident(key_a)

    view.enter_stream(0, "s2")
    assert view.effective_l2_bytes(0) == (128 * mib // (128 * 3)) * 128
    view.enter_stream(0, "s3")
    assert view.effective_l2_bytes(0) == 32 * mib

    view.exit_stream(0, "s3")
    view.exit_stream(0, "s2")
    view.exit_stream(0, "s1")
    assert view.effective_l2_bytes(0) == 128 * mib
    assert not view.is_resident(key_a)  # evicted entry never resurrects
    assert view.admit_access(pat_96m, 96 * mib) == (0, 96 * mib)
    assert view.admit_access(pat_96m, 96 * mib) == (96 * mib, 0)

    view.grant_reservation(0, "persist", 80 * mib)
    assert view.effective_l2_bytes(0) == 48 * mib
    assert not view.is_resident(key_a)
    view.release_reservation(0, "persist")
    assert not view.is_resident(key_a)


@pytest.mark.parametrize("alpha_rw", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("backend", ["python", "c"])
def test_hbm_two_row_directional_bus_laws_python_and_c(alpha_rw: float, backend: str):
    """Solo R/W incurs zero penalty; concurrent R+W yields max(Tr,Tw)+alpha*min(Tr,Tw)."""
    bus = HbmBusSpec(die=0, read_bytes_per_ms=1000.0, write_bytes_per_ms=1000.0, alpha_rw=alpha_rw)
    t_r, t_w, t_v = 4.0, 2.0, 1.5

    def run_case(mode: str) -> float:
        eng = Engine([Resource("vec", 1.0)], backend=backend)
        eng.add_hbm_bus(bus)
        r_work = {t.row_id: t.ms_per_progress for t in bus.terms_for("read", t_r * 1000.0) if t.ms_per_progress > 0.0}
        w_work = {t.row_id: t.ms_per_progress for t in bus.terms_for("write", t_w * 1000.0) if t.ms_per_progress > 0.0}
        if mode == "solo_r":
            ev = eng.submit("r", r_work, coupled=True)
        elif mode == "solo_w":
            ev = eng.submit("w", w_work, coupled=True)
        elif mode == "serial":
            er = eng.submit("r", r_work, coupled=True)
            ev = eng.submit("w", w_work, after=(eng.submit("v", {"vec": t_v}, after=(er,)),), coupled=True)
        else:
            ev = eng.submit("done", after=(eng.submit("r", r_work, coupled=True), eng.submit("w", w_work, coupled=True)))
        eng.advance(until=(ev,))
        if backend == "c":
            assert eng.accel_status()["native_steps"] > 0
        return float(ev.time)

    assert run_case("solo_r") == pytest.approx(t_r, rel=1e-12)
    assert run_case("solo_w") == pytest.approx(t_w, rel=1e-12)
    assert run_case("serial") == pytest.approx(t_r + t_v + t_w, rel=1e-12)
    assert run_case("concurrent") == pytest.approx(max(t_r, t_w) + alpha_rw * min(t_r, t_w), rel=1e-12)


def test_prepared_terms_c11_matches_python_within_8_ulp_and_conserves_work():
    """Direct C11 prepared-row solver matches Python reference across mixed R/W, prio and FIFO."""
    bus = HbmBusSpec(die=0, read_bytes_per_ms=1_600_000.0, write_bytes_per_ms=1_200_000.0, alpha_rw=0.35)
    rows = [
        PreparedCapacityRow(row_id=0, policy=POLICY_FAIR, capacity=1.0),
        PreparedCapacityRow(row_id=1, policy=POLICY_FAIR, capacity=1.0),
        PreparedCapacityRow(row_id=2, policy=POLICY_FIFO, capacity=1.0),
    ]
    d_r, d_w = 3_200_000.0, 2_400_000.0
    (tr_r, tr_w), (tw_r, tw_w) = bus.terms_for("read", d_r), bus.terms_for("write", d_w)

    with PreparedTermsRunner() as runner:
        c_r, c_w = runner.hbm_terms(bus, "read", d_r)
        assert _ulp_distance(c_r, tr_r.ms_per_progress) <= 1 and _ulp_distance(c_w, tr_w.ms_per_progress) <= 1
        progs_py = [
            PreparedProgress(sequence=1, task=1, priority=0, placement_id=0, member_map=0, z_left=1.0, weight=1.0),
            PreparedProgress(sequence=2, task=2, priority=0, placement_id=0, member_map=0, z_left=1.0, weight=1.0),
            PreparedProgress(sequence=3, task=3, priority=1, placement_id=0, member_map=0, z_left=1.0, weight=2.0),
            PreparedProgress(sequence=4, task=4, priority=0, placement_id=0, member_map=0, z_left=1.0, weight=1.0),
        ]
        progs_c = [
            PreparedProgress(p.sequence, p.task, p.priority, p.placement_id, p.member_map, p.z_left, p.weight, p.max_dz_ms)
            for p in progs_py
        ]
        terms = [
            PreparedTerm(0, 0, tr_r.ms_per_progress), PreparedTerm(1, 0, tr_w.ms_per_progress),
            PreparedTerm(0, 1, tw_r.ms_per_progress), PreparedTerm(1, 1, tw_w.ms_per_progress),
            PreparedTerm(2, 1, 0.5), PreparedTerm(0, 2, 1.0), PreparedTerm(2, 3, 0.5),
        ]
        deliveries = [
            PreparedDelivery(0, 0, d_r), PreparedDelivery(1, 1, d_w),
            PreparedDelivery(2, 2, 100.0), PreparedDelivery(2, 3, 50.0),
        ]
        runner.config_rows(rows)
        runner.prepare_terms(progs_c, terms, deliveries, ncounters=3)
        ctrs_py = [0.0, 0.0, 0.0]

        for _ in range(10):
            if all(p.z_left <= 0.0 for p in progs_py):
                break
            dz_py = allocate_prepared_epoch(rows, progs_py, terms)
            dz_c = runner.allocate_epoch()
            assert all(_ulp_distance(a, b) <= 8 for a, b in zip(dz_py, dz_c))
            dt = min(p.z_left / r for p, r in zip(progs_py, dz_py) if p.z_left > 0.0 and r > 0.0)
            s_py = integrate_prepared_epoch(progs_py, deliveries, dz_py, dt, ctrs_py)
            s_c = runner.integrate_epoch(dz_c, dt)
            assert all(_ulp_distance(a, b) <= 8 for a, b in zip(s_py, s_c))

        z_c, ctrs_c = runner.read_state()
        assert all(z == 0.0 for z in z_c)
        assert ctrs_c[0] == pytest.approx(d_r, rel=1e-12)
        assert ctrs_c[1] == pytest.approx(d_w, rel=1e-12)
        assert ctrs_c[2] == pytest.approx(150.0, rel=1e-12)
        assert all(_ulp_distance(a, b) <= 8 for a, b in zip(ctrs_py, ctrs_c))


def test_vector_subpipes_target_eff_and_raw_latency_invariance():
    """Vector sub-pipes lower onto vec_issue + sub-pipes; target_eff leaves RAW cycles invariant."""
    ctx = _make_context()
    groups = (
        VectorInstructionGroup("exp", "vec_sfu", count=10, issue_slots=1, initiation_cycles=1, result_latency_cycles=20),
        VectorInstructionGroup("sum", "vec_reduce", count=4, issue_slots=1, initiation_cycles=1, result_latency_cycles=10, dependencies=("exp",)),
        VectorInstructionGroup("mul", "vec_alu", count=8, issue_slots=1, initiation_cycles=1, dependencies=("sum",)),
    )
    full = ctx.bind_vector(groups, target_eff=1.0)
    half = ctx.bind_vector(groups, target_eff=0.5)
    assert full.raw_latency_ms == pytest.approx(half.raw_latency_ms, rel=1e-12)
    assert half.span_ms > full.span_ms
    assert all("vector" not in stage.work for stage in full.stages)
    assert full.vector_projection == pytest.approx(22.0)


def test_ring_buffer_three_event_distance_d_credit_and_multi_consumer_last_use():
    """Generation k waits on last_use(k - d) across ALL consumers, not the earliest consumer."""
    ring = RingBuffer(occurrence=1, die=0, owner=0, pool="l1", allocation="kv_ring", slot_bytes=4096, depth=2)
    gens = [
        lower_generation(
            ring, k, producer="load", consumers=("fast_c", "slow_c"),
            producer_work={"mte2": 1.0},
            consumer_works={"fast_c": {"cube": 1.0}, "slow_c": {"vec": 3.0}},
        )
        for k in range(4)
    ]
    assert gens[2].credit == gens[0].last_use
    assert gens[3].credit == gens[1].last_use
    slot_recs, dep_recs = ring.to_c_records()
    assert len(slot_recs) == 4 and len(dep_recs) > 0

    stages = ring.to_stages()
    eng = Engine([Resource("mte2", 1.0), Resource("cube", 1.0), Resource("vec", 1.0)], pools={"l1": 8192.0}, pool_units={"l1": 1}, backend="python")
    done = KernelProgram(stages).start(eng, "ring_op", eng.submit("ready"))
    eng.advance(until=(done,))
    assert done.time >= 6.0 - 1e-12  # gen 2 waits for gen 0 slow consumer (t=4), then runs load+slow (4+4=8 or 1+3+3=7)


def test_tile_solver_liveness_simt_carveout_fallback_ladder_and_core_waves():
    """TileSolver enforces SIMT UB carveout, alias folding, fallback ladder and core-control waves."""
    cand_1pass = TileCandidate(
        algorithm="resident_1pass", tile_shape=TileShape(m=10, n=64, k=64, head_dim=4096),
        depth=1, owner_partition=(8, 16), work_items=10, tiles_per_owner=3,
        buffers=(
            LiveBufferSpec("q", "ub", 48 * 1024, live_start=0, live_end=2),
            LiveBufferSpec("scores", "ub", 160 * 1024, live_start=1, live_end=2),
        ),
        logical_flops=2 * 10 * 64 * 64, instruction_block=(16, 16, 16), uses_simt=True,
    )
    cand_online = TileCandidate(
        algorithm="online_reduce", tile_shape=TileShape(m=10, n=64, k=64, head_dim=4096),
        depth=1, owner_partition=(8, 16), work_items=10, tiles_per_owner=3,
        buffers=(
            LiveBufferSpec("q", "ub", 48 * 1024, live_start=0, live_end=2),
            LiveBufferSpec("tmp_a", "ub", 64 * 1024, live_start=0, live_end=1, alias_group="scratch"),
            LiveBufferSpec("tmp_b", "ub", 64 * 1024, live_start=1, live_end=2, alias_group="scratch"),
        ),
        logical_flops=2 * 10 * 64 * 64, instruction_block=(16, 16, 16), uses_simt=True,
        native_mode="narrow_m",
    )
    spec = TilePipelineSpec(
        family="attention", candidates=(cand_1pass, cand_online),
        simt_ub_carveout_bytes=32 * 1024,
        subtile_waves=(SubtileWaveSpec("narrow_m", tile_m=32, tile_n=64, tile_k=64, independently_issuable_m=16),),
    )
    hw = SimpleNamespace(
        buffers=(("ub", 192 * 1024),), compute_units=4,
        stream_owners=SimpleNamespace(aic_limit=4, aiv_limit=8),
    )
    sol = solve_tiles(spec, None, hw)
    assert sol.algorithm == "online_reduce"
    assert sol.waves == 3  # ceil(10 / 4)
    assert sol.issued_flops > cand_online.logical_flops  # legal instruction-block padding, no fudge factor
    assert len(sol.rejections) == 1 and sol.rejections[0].algorithm == "resident_1pass"

    hw_tiny = SimpleNamespace(buffers=(("ub", 32 * 1024),), compute_units=4)
    with pytest.raises(TileInfeasibleError) as exc_info:
        solve_tiles(spec, None, hw_tiny)
    assert len(exc_info.value.rejections) == 2


@pytest.mark.parametrize("backend", ["python", "c"])
def test_wave_recorder_five_point_conservation_and_zero_perturbation(backend: str):
    """WaveRecorder preserves busy+fill+steady_bubble+drain+handoff == span_ns with zero perturbation."""
    spec = RecordedPipelineSpec("1", "s0", "sig", "fused", "rec", "abi", members=("m0",))

    def run(with_recorder: bool):
        eng = Engine([Resource("mte2", 1.0), Resource("cube", 1.0), Resource("vec", 1.0), Resource("mte3", 1.0)], backend=backend)
        rec = WaveRecorder(declared_pipes=("mte2", "cube", "vec", "mte3")) if with_recorder else None
        if rec is not None:
            eng.attach_wave_recorder(rec)
        s0 = eng.submit("load0", {"mte2": 1.0}, owner="tile0")
        c0 = eng.submit("cube0", {"cube": 2.0}, after=(s0,), owner="tile0")
        v0 = eng.submit("vec0", {"vec": 1.5}, after=(s0,), owner="tile0")
        s1 = eng.submit("load1", {"mte2": 1.0}, after=(s0,), owner="tile1")
        c1 = eng.submit("cube1", {"cube": 2.0}, after=(c0, s1), owner="tile1")
        done = eng.submit("done", after=(c1, v0))
        eng.advance(until=(done,))
        if backend == "c":
            assert eng.accel_status()["native_steps"] > 0
        desc = rec.finish(spec, "occ0") if rec is not None else None
        return float(done.time), dict(eng.serviced), desc

    t_base, s_base, _ = run(False)
    t_obs, s_obs, desc = run(True)
    assert t_obs == pytest.approx(t_base, rel=1e-12, abs=1e-12)
    assert s_obs == pytest.approx(s_base, rel=1e-12, abs=1e-12)
    assert desc is not None
    span_ns = desc.timeline.end_ns - desc.timeline.start_ns
    assert span_ns == int(round(t_base * 1_000_000.0))
    for row in desc.timeline.pipe_rows:
        assert row.busy_ns + row.fill_ns + row.steady_bubble_ns + row.drain_ns + row.handoff_ns == row.span_ns
    mte3_row = next(r for r in desc.timeline.pipe_rows if r.pipe_name == "mte3")
    assert mte3_row.busy_ns == 0 and mte3_row.inactive_in_window and mte3_row.fill_ns == span_ns
    assert desc.timeline.cube_vector_overlap_ns > 0


def test_wave_recorder_periodic_skip_conservation():
    """Periodic skip records translate cleanly into WaveRecorder without rounding drift."""
    spec = RecordedPipelineSpec("1", "s1", "sig", "fused", "rec", "abi")
    rec = WaveRecorder(declared_pipes=("cube", "mte2"))
    rec.on_service(1, "mte2", 0, 0.0, 1.0, 1.0)
    rec.on_period(1.0, 2.0, 5, "tpl", "dig", service_pattern=(("cube", 0.0, 1.5), ("mte2", 0.0, 1.0)))
    rec.on_service(2, "cube", 0, 11.0, 12.0, 1.0)
    desc = rec.finish(spec, "occ1")
    assert desc.timeline.end_ns - desc.timeline.start_ns == 12_000_000
    for row in desc.timeline.pipe_rows:
        assert row.busy_ns + row.fill_ns + row.steady_bubble_ns + row.drain_ns + row.handoff_ns == row.span_ns


def test_program_package_split_reexports_and_stage_builtin_guards():
    """Destructive program/ package preserves all public re-exports and strict Stage guards."""
    assert hasattr(prog_pkg, "__path__")
    for name in (
        "Allocation", "GENERIC_MATRIX_PROGRAM", "HostStaging", "KernelProgram",
        "NativeProgram", "PIPELINE_COHORT_WINDOW", "PIPELINE_FOLD_WINDOW",
        "PIPELINE_UNROLL_LIMIT", "ParallelProgram", "PipelinePhase",
        "PipelineProgram", "PipelineTemplate", "PipelineTemplateProgram",
        "Program", "Recipe", "RecipeBook", "RecipeContext", "RecurrentScan",
        "ResourceEnvelope", "Stage", "TransferRecipe", "coalesce_execution_program",
        "coalesce_programs", "fold_interior_pipeline",
        "indexed_pipeline_consumer_available", "interior_fold_plan",
        "is_pure_static_program", "merge_programs", "program_signature",
        "scope_allocations", "split_cohort_kernel", "stage_from_fragment",
    ):
        assert hasattr(prog_pkg, name), f"missing re-export: {name}"
    with pytest.raises(ModelingError, match="stage_cohort_widened_invalid"):
        Stage("bad", cohort_widened=(123,))  # type: ignore[arg-type]
